"""
SC-HRA: State-Conditioned Hybrid Reward Architecture for 1v1 Catan
 
Three independent DQN critics (one per reward channel) combined via a
learned meta-weighting network. The meta-network is trained with a
retrospective bandit objective at the end of each episode.
 
Requirements:
    pip install catanatron[gym] torch tensorboard
"""
 
import os
import random
import math
import time
import multiprocessing as mp
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.logger import configure as configure_sb3_logger
from stable_baselines3.common.vec_env import SubprocVecEnv

import gymnasium
from gymnasium import Wrapper
 
from catanatron import Color, Player
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.state_functions import player_key, player_num_resource_cards
import catanatron.gym
 
# ================================================================
# CONFIG
# ================================================================
 
DEVICE = torch.device("cpu")
FEATURE_DIM = 25
N_CHANNELS = 3  # resource, position, vp


def _resolve_device(name: str) -> torch.device:
    """Map a CLI device string to a torch.device.

    "cpu"  → CPU
    "cuda" → NVIDIA GPU (errors if unavailable)
    "mps"  → Apple Silicon GPU (errors if unavailable)
    "gpu" / "auto" → cuda if available, else mps, else cpu

    Sets PYTORCH_ENABLE_MPS_FALLBACK=1 when MPS is selected so torch falls
    back to CPU for ops not yet implemented on the MPS backend.
    """
    name = name.lower()
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if name == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("--device mps requested but MPS is not available")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    if name in ("gpu", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            return torch.device("mps")
        print("[device] No GPU available, falling back to CPU.")
        return torch.device("cpu")
    raise ValueError(f"Unknown device: {name!r}")
 
RESOURCE_TYPES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
 
MODEL_DIR = os.path.join(os.path.dirname(__file__), "schra_model")
 
 
# ================================================================
# STATE HELPERS (shared with baseline)
# ================================================================
 
def get_player_hand(state, color):
    key = player_key(state, color)
    return {r: state.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCE_TYPES}


def get_hand_size(state, color):
    return player_num_resource_cards(state, color)


def get_victory_points(state, color):
    key = player_key(state, color)
    return state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
 
 
def get_knights_played(state, color):
    key = player_key(state, color)
    return state.player_state.get(f"{key}_PLAYED_KNIGHT", 0)
 
 
def get_longest_road_length(state, color):
    key = player_key(state, color)
    return state.player_state.get(f"{key}_LONGEST_ROAD_LENGTH", 0)
 
 
# ================================================================
# REWARD CHANNEL FUNCTIONS (shared with baseline)
# ================================================================
 
def resource_flow_score(state, color):
    hand = get_player_hand(state, color)
    hand_size = sum(hand.values())
    distinct_types = sum(1 for v in hand.values() if v > 0)
    diversity_bonus = distinct_types * 0.2
    quantity_bonus = hand_size * 0.1
    hoard_penalty = 0.0
    if hand_size > 9:
        hoard_penalty = 0.2 + 0.1 * (hand_size - 9)
    return diversity_bonus + quantity_bonus - hoard_penalty
 
 
def network_position_score(state, color):
    """
    Network Position Channel (score function, reward = delta between turns):
    - +0.15 per production pip covered by settlements (doubled for cities)
      (zeroed if robber is on that tile)
    - +0.8 per unique resource type accessible from settlements/cities
    - +0.2 for having >= 1 generic port
    - Per specific-resource port connected: +0.05 per effective pip of that
      resource owned (so a port with no matching production is worthless;
      one fed by 5+ pips is worth ~0.25)
    - +0.03 per total road (longest-road bonus removed — already counted
      via the VP-delta channel; was double-counting)
    - +0.06 * (total pips of adjacent tiles) per open settlement spot
      reachable via road network
    """
    board = state.board
    score = 0.0
    pip_counts = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 0, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
    tile_coord = {tile.id: coord for coord, tile in board.map.land_tiles.items()}
    robber_coordinate = board.robber_coordinate
    resource_types_accessible = set()
    pips_by_resource = {}
    buildings = board.buildings

    for node_id, (bldg_color, bldg_type) in buildings.items():
        if bldg_color != color:
            continue
        multiplier = 2 if bldg_type == "CITY" else 1
        for tile in board.map.adjacent_tiles.get(node_id, []):
            if tile.resource is None:
                continue
            resource_types_accessible.add(tile.resource)
            if tile_coord.get(tile.id) == robber_coordinate:
                continue
            pips = pip_counts.get(tile.number, 0) if tile.number else 0
            score += pips * 0.15 * multiplier
            pips_by_resource[tile.resource] = (
                pips_by_resource.get(tile.resource, 0) + pips * multiplier
            )

    score += len(resource_types_accessible) * 0.8

    has_generic_port = False
    for resource, node_ids in board.map.port_nodes.items():
        if not any(nid in buildings and buildings[nid][0] == color for nid in node_ids):
            continue
        if resource is None:
            has_generic_port = True
        else:
            matching_pips = pips_by_resource.get(resource, 0)
            score += 0.05 * matching_pips
    if has_generic_port:
        score += 0.2

    roads = [(edge, c) for edge, c in board.roads.items() if c == color]
    num_roads = len(roads) // 2
    score += num_roads * 0.03

    subgraphs = board.find_connected_components(color)
    buildable = board.buildable_node_ids(color) if subgraphs else []
    for node_id in buildable:
        pip_sum = 0
        for tile in board.map.adjacent_tiles.get(node_id, []):
            if tile.number:
                pip_sum += pip_counts.get(tile.number, 0)
        score += 0.06 * pip_sum

    return score
 
 
def vp_proximity_reward(state, color, prev_vps, prev_knights, opp_color):
    current_vps = get_victory_points(state, color)
    vp_delta = current_vps - prev_vps
    current_knights = get_knights_played(state, color)
    new_knights = current_knights - prev_knights
    knight_reward = 0.0
    if new_knights > 0:
        opp_knights = get_knights_played(state, opp_color)
        knights_needed = max(3, opp_knights + 1)
        total_played = current_knights + opp_knights
        knights_remaining_in_deck = 14 - total_played
        achievable = knights_needed <= current_knights + knights_remaining_in_deck
        if achievable:
            knights_remaining = knights_needed - current_knights
            feasibility = min(1.0, 3.0 / knights_needed)
            knight_reward = new_knights * 0.3 * feasibility / (1 + max(0, knights_remaining))
    return float(vp_delta) + knight_reward
 
 
def terminal_reward(game, color):
    winning_color = game.winning_color()
    if winning_color is None:
        return 0.0
    state = game.state
    my_vps = get_victory_points(state, color)
    opp_vps = max(
        (get_victory_points(state, c) for c in state.colors if c != color),
        default=0,
    )
    #vp_margin = 0.5 * (my_vps - opp_vps)
    return (15.0 if winning_color == color else -15.0)
 
 
# ================================================================
# FEATURE COMPUTATION (shared with baseline)
# ================================================================
 
def compute_features(state, color, opp_color):
    hand = get_player_hand(state, color)
    opp_hand = get_player_hand(state, opp_color)
    hand_size = sum(hand.values())
    my_vps = get_victory_points(state, color)
    opp_vps = get_victory_points(state, opp_color)
    my_knights = get_knights_played(state, color)
    opp_knights = get_knights_played(state, opp_color)
    my_road = get_longest_road_length(state, color)
    opp_road = get_longest_road_length(state, opp_color)
    pos_score = network_position_score(state, color)
    opp_pos_score = network_position_score(state, opp_color)
 
    return np.array([
        hand_size / 10.0,
        sum(1 for v in hand.values() if v > 0) / 5.0,
        hand.get("WOOD", 0) / 5.0,
        hand.get("BRICK", 0) / 5.0,
        hand.get("SHEEP", 0) / 5.0,
        hand.get("WHEAT", 0) / 5.0,
        hand.get("ORE", 0) / 5.0,
        sum(opp_hand.values()) / 10.0,
        my_vps / 15.0,
        opp_vps / 15.0,
        (my_vps - opp_vps) / 15.0,
        my_vps / 15.0,
        max(0, 15 - my_vps) / 15.0,
        my_knights / 5.0,
        opp_knights / 5.0,
        float(my_knights >= 3 and my_knights > opp_knights),
        float(opp_knights >= 3 and opp_knights > my_knights),
        my_road / 10.0,
        opp_road / 10.0,
        float(my_road >= 5 and my_road > opp_road),
        float(opp_road >= 5 and opp_road > my_road),
        pos_score / 10.0,
        opp_pos_score / 10.0,
        (pos_score - opp_pos_score) / 10.0,
        resource_flow_score(state, color) / 2.0,
    ], dtype=np.float32)
 
 
# ================================================================
# NEURAL NETWORKS
# ================================================================
 
class QNetwork(nn.Module):
    """DQN critic for a single reward channel.
    Input: state features (FEATURE_DIM).  Output: Q(s,a) for all actions.

    ``hidden`` may be an int (single hidden size, two layers) or a sequence
    of ints (one Linear+ReLU per element). Default (512, 256) gives a wider
    first layer than the second to absorb the ~290-action output head.
    """

    def __init__(self, state_dim, n_actions, hidden=(512, 256)):
        super().__init__()
        if isinstance(hidden, int):
            hidden = (hidden, hidden)
        layers = []
        prev = state_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
 
 
class MetaWeightNetwork(nn.Module):
    """Outputs softmax weights over N reward channels conditioned on state."""
 
    def __init__(self, state_dim, n_channels=N_CHANNELS, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_channels),
        )
 
    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)
 
 
# ================================================================
# REPLAY BUFFER
# ================================================================
 
# Transition is kept around as a documentation aid (field names + dtypes)
# and for back-compat in load_checkpoint when reading older deque-of-tuple
# replay buffers. The vectorized ReplayBuffer below does not use it on the
# hot path.
Transition = namedtuple("Transition", [
    "state",            # np.array (FEATURE_DIM,)
    "action",           # int
    "rewards",          # np.array (N_CHANNELS,) — [r_resource, r_position, r_vp]
    "next_state",       # np.array (FEATURE_DIM,)
    "done",             # bool
    "action_mask",      # np.array (n_actions,) bool
    "next_action_mask", # np.array (n_actions,) bool
])


class ReplayBuffer:
    """Vectorized circular replay buffer.

    Stores each Transition field as a preallocated numpy array of shape
    (capacity, ...). push() writes at a rolling index; sample() does a
    single np.random.randint to pick row indices and fancy-indexes all
    fields in one shot, eliminating the per-sample Python iteration that
    dominated the old deque-of-namedtuples implementation.
    """

    def __init__(self, capacity, state_dim, n_actions, n_channels):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.n_channels = int(n_channels)

        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros((self.capacity, self.n_channels), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.action_masks = np.zeros((self.capacity, self.n_actions), dtype=bool)
        self.next_action_masks = np.zeros((self.capacity, self.n_actions), dtype=bool)

        self.idx = 0   # next write position
        self.size = 0  # number of valid entries (<= capacity)

    def push(self, state, action, rewards, next_state, done,
             action_mask, next_action_mask):
        i = self.idx
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = rewards
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.action_masks[i] = action_mask
        self.next_action_masks[i] = next_action_mask
        self.idx = (self.idx + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size):
        # Sampling WITH replacement: O(batch_size). For buffer_size=200k and
        # batch_size=256 the chance of a collision is ~16%, which is well
        # within the noise of DQN updates — most DQN reference impls do
        # uniform-with-replacement for exactly this reason.
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]).to(DEVICE),
            torch.from_numpy(self.actions[idx]).to(DEVICE),
            torch.from_numpy(self.rewards[idx]).to(DEVICE),
            torch.from_numpy(self.next_states[idx]).to(DEVICE),
            torch.from_numpy(self.dones[idx]).to(DEVICE),
            torch.from_numpy(self.action_masks[idx]).to(DEVICE),
            torch.from_numpy(self.next_action_masks[idx]).to(DEVICE),
        )

    def __len__(self):
        return self.size

    # --- Checkpoint helpers -----------------------------------------------

    def to_state_dict(self):
        """Snapshot the buffer as a dict of numpy arrays for torch.save().
        Trims to actual size so we don't dump zeros for unfilled rows."""
        n = self.size
        return {
            "capacity": self.capacity,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "n_channels": self.n_channels,
            "size": n,
            "idx": self.idx,
            "states": self.states[:n].copy(),
            "actions": self.actions[:n].copy(),
            "rewards": self.rewards[:n].copy(),
            "next_states": self.next_states[:n].copy(),
            "dones": self.dones[:n].copy(),
            "action_masks": self.action_masks[:n].copy(),
            "next_action_masks": self.next_action_masks[:n].copy(),
        }

    def load_state_dict(self, sd):
        """Inverse of to_state_dict(). Tolerates a smaller saved capacity
        by copying into the head of the new buffer."""
        n = int(sd["size"])
        n = min(n, self.capacity)
        self.states[:n] = sd["states"][:n]
        self.actions[:n] = sd["actions"][:n]
        self.rewards[:n] = sd["rewards"][:n]
        self.next_states[:n] = sd["next_states"][:n]
        self.dones[:n] = sd["dones"][:n]
        self.action_masks[:n] = sd["action_masks"][:n]
        self.next_action_masks[:n] = sd["next_action_masks"][:n]
        self.size = n
        # New writes go at the next slot after the loaded data; if the
        # buffer is full again we overwrite from the start, which is
        # equivalent to the original circular behavior.
        self.idx = n % self.capacity

    def load_from_tuples(self, tuples):
        """Legacy load path: list of (state, action, rewards, next_state,
        done, action_mask, next_action_mask) tuples (or Transitions).
        Used when reading a checkpoint written before the buffer was
        vectorized."""
        for t in tuples:
            self.push(*t)
 
 
# ================================================================
# Q-VALUE NORMALIZER
# ================================================================
 
class RunningNormalizer:
    """Welford's online algorithm for running mean/std.
    Used to normalize Q-values before combining in the meta-network
    so all channels contribute at comparable scale."""
 
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
 
    def update(self, values):
        """Update with a batch of values (numpy array or scalar)."""
        values = np.atleast_1d(np.asarray(values, dtype=np.float64))
        for v in values:
            self.count += 1
            delta = v - self.mean
            self.mean += delta / self.count
            delta2 = v - self.mean
            self.var += (delta * delta2 - self.var) / self.count
 
    def normalize(self, x):
        std = max(math.sqrt(self.var), 1e-8)
        return (x - self.mean) / std
 
 
# ================================================================
# SC-HRA ENVIRONMENT WRAPPER
# ================================================================
 
class SCHRAWrapper(Wrapper):
    """
    Wraps Catanatron-v0 to:
    1. Replace observations with 25-dim hand-crafted features.
    2. Compute per-channel rewards at every step and expose in info.
    3. Compute terminal reward at episode end and fold it into the VP
       channel so the VP critic actually sees win/loss signal.
    4. Track episode stats for logging.

    Reward channels:

    - r_resource: PPO-style instantaneous (NOT a potential-difference on
      hand size). Rewards earning, rewards spending-when-building, only
      penalizes hoarding past the discard threshold. Previous version
      used delta(resource_flow_score) which made building NEGATIVE in
      this channel, biasing the agent toward passing.

    - r_position: delta of network_position_score, scaled 0.3. Unchanged.

    - r_vp: vp_proximity_reward + r_terminal on the final step. Folding
      the ±15 terminal reward into the VP channel gives Q_vp a magnitude
      that the meta-net's softmax-weighted-sum can actually fit against
      its r_terminal regression target (previously the meta-net had to
      reach ±15 from a convex combo of channel Q-values all bounded near
      ~10, which forced softmax saturation onto whichever channel was
      most negative — usually the broken resource channel).
    """

    # Tuned to match 185_ppo.CatanRewardWrapper exactly.
    EARN_REWARD = 0.05
    SPEND_REWARD = 0.05
    HOARD_PENALTY = 0.05

    def __init__(self, env):
        super().__init__(env)
        self.p0_color = Color.BLUE
        self.opp_color = Color.RED
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(FEATURE_DIM,), dtype=np.float32
        )
        self._prev_hand_size = 0
        self._prev_building_count = 0
        self._prev_road_count = 0
        self._prev_position_score = 0.0
        self._prev_vps = 0
        self._prev_knights = 0

        # episode accumulators
        self._ep_r = np.zeros(N_CHANNELS)
        self._ep_r_terminal = 0.0
        self._ep_steps = 0

    def _features(self, state):
        return compute_features(state, self.p0_color, self.opp_color)

    def _safe_position_score(self, state):
        try:
            return network_position_score(state, self.p0_color)
        except Exception:
            return 0.0

    def _building_count(self, state):
        return sum(
            1 for _, (c, _) in state.board.buildings.items() if c == self.p0_color
        )

    def _road_count(self, state):
        return sum(1 for _, c in state.board.roads.items() if c == self.p0_color) // 2

    def get_valid_actions(self):
        return self.env.unwrapped.get_valid_actions()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        state = self.env.unwrapped.game.state
        self._prev_hand_size = get_hand_size(state, self.p0_color)
        self._prev_building_count = self._building_count(state)
        self._prev_road_count = self._road_count(state)
        self._prev_position_score = self._safe_position_score(state)
        self._prev_vps = get_victory_points(state, self.p0_color)
        self._prev_knights = get_knights_played(state, self.p0_color)
        self._ep_r = np.zeros(N_CHANNELS)
        self._ep_r_terminal = 0.0
        self._ep_steps = 0
        return self._features(state), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        game = self.env.unwrapped.game
        state = game.state
        done = terminated or truncated

        # Channel 0: resource throughput (instantaneous, PPO-style).
        #   + EARN_REWARD per card drawn (positive hand delta from any source)
        #   + SPEND_REWARD per card spent in the same step that we placed
        #     a new building or road (so spending only counts when it
        #     translates into a build, not when it's a discard / robber /
        #     monopoly steal)
        #   - HOARD_PENALTY per card held above the discard threshold (9)
        cur_hand = get_hand_size(state, self.p0_color)
        cur_buildings = self._building_count(state)
        cur_roads = self._road_count(state)
        delta_hand = cur_hand - self._prev_hand_size
        built_something = (
            cur_buildings > self._prev_building_count
            or cur_roads > self._prev_road_count
        )
        r_resource = 0.0
        if delta_hand > 0:
            r_resource += self.EARN_REWARD * delta_hand
        elif delta_hand < 0 and built_something:
            r_resource += self.SPEND_REWARD * (-delta_hand)
        if cur_hand > 9:
            r_resource -= self.HOARD_PENALTY * (cur_hand - 9)
        self._prev_hand_size = cur_hand
        self._prev_building_count = cur_buildings
        self._prev_road_count = cur_roads

        # Channel 1: network position delta (scaled 0.3)
        cur_position = self._safe_position_score(state)
        r_position = 0.3 * (cur_position - self._prev_position_score)
        self._prev_position_score = cur_position

        # Channel 2: VP proximity. Terminal win/loss reward is folded in
        # below on the done step so Q_vp can actually capture the outcome
        # signal that the meta-net regresses against.
        r_vp = vp_proximity_reward(
            state, self.p0_color,
            self._prev_vps, self._prev_knights,
            self.opp_color,
        )
        self._prev_vps = get_victory_points(state, self.p0_color)
        self._prev_knights = get_knights_played(state, self.p0_color)

        r_terminal = 0.0
        if done:
            r_terminal = terminal_reward(game, self.p0_color)
            r_vp += r_terminal

        rewards = np.array([r_resource, r_position, r_vp], dtype=np.float32)

        # expose everything in info. r_terminal is also kept as its own
        # entry (UNCHANGED) so train_meta_for_env can still use it as the
        # retrospective bandit regression target.
        info["rewards"] = rewards
        info["r_terminal"] = r_terminal
        if done:
            info["win"] = int(game.winning_color() == self.p0_color)

        self._ep_r += rewards
        self._ep_r_terminal += r_terminal
        self._ep_steps += 1

        if done:
            info["ep_r_resource"] = float(self._ep_r[0])
            info["ep_r_position"] = float(self._ep_r[1])
            info["ep_r_vp"] = float(self._ep_r[2])
            info["ep_r_terminal"] = float(self._ep_r_terminal)
            info["ep_steps"] = self._ep_steps

        return self._features(state), 0.0, terminated, truncated, info
 
 
# ================================================================
# N-STEP RETURN PROCESSOR
# ================================================================

class NStepProcessor:
    """Per-env rolling queue that converts 1-step env transitions into
    n-step Bellman targets before they enter the replay buffer.

    Given the sequence of transitions (s_t, a_t, r_t, s_{t+1}, done_t)
    for one env, it emits replay-buffer entries of the form:

        (s_t, a_t, R_n(t), s_{t+n}, term_within_window,
         mask_t, mask_{t+n})

    where R_n(t) = Σ_{k=0..n-1} γ^k · r_{t+k}, truncated whenever the
    episode ends within the window (in which case term_within_window is
    True and s_{t+n}/mask_{t+n} are clamped to the post-terminal values,
    which are zeroed out anyway by (1 - done) in the Bellman target).

    For n=1 the behavior is identical to the original 1-step setup, so
    this class is the single code path regardless of self.n_step.
    """

    def __init__(self, n, gamma, n_channels):
        self.n = int(n)
        self.gamma = float(gamma)
        self.n_channels = int(n_channels)
        # Per-env FIFO queue. Lazy-init keyed by env_idx so we don't need
        # to know n_envs at construction time. List (not deque) because n
        # is tiny (1–5) and list pop(0) is O(n) but n=3 ⇒ negligible.
        self.queues: dict = {}
        # γ^k for k=0..n-1, precomputed for the steady-state path.
        self.gamma_powers = np.array(
            [gamma ** k for k in range(self.n)], dtype=np.float32
        )

    def push(self, env_idx, state, action, rewards, next_state, done,
             action_mask, next_action_mask):
        """Add a 1-step transition for env_idx. Returns 0..N replay-buffer
        emissions ready to be passed positionally to ReplayBuffer.push().

        Steady state (queue full, no done): 1 emission.
        On done: flushes the queue → up to n emissions (truncated returns).
        Otherwise: 0 emissions (queue still warming up)."""
        q = self.queues.setdefault(env_idx, [])
        q.append((state, action, np.asarray(rewards, dtype=np.float32), action_mask))

        emissions = []

        if done:
            # Flush every queued transition with its truncated n-step
            # return, ending at the terminal step. done=True so the
            # Bellman bootstrap is zeroed out anyway.
            L = len(q)
            for j in range(L):
                s, a, _, am = q[j]
                n_step_r = np.zeros(self.n_channels, dtype=np.float32)
                for k in range(j, L):
                    n_step_r += (self.gamma ** (k - j)) * q[k][2]
                emissions.append(
                    (s, a, n_step_r, next_state, True, am, next_action_mask)
                )
            q.clear()
        elif len(q) >= self.n:
            # Standard n-step emit for the oldest entry, then pop it.
            n_step_r = np.zeros(self.n_channels, dtype=np.float32)
            for k in range(self.n):
                n_step_r += self.gamma_powers[k] * q[k][2]
            s, a, _, am = q[0]
            emissions.append(
                (s, a, n_step_r, next_state, False, am, next_action_mask)
            )
            q.pop(0)

        return emissions

    def reset_env(self, env_idx):
        """Drop any pending entries for an env (e.g., on manual reset)."""
        if env_idx in self.queues:
            self.queues[env_idx].clear()


# ================================================================
# SC-HRA AGENT
# ================================================================

class SCHRAAgent:
    """
    State-Conditioned Hybrid Reward Architecture agent.
 
    Components:
        - 3 independent Double-DQN critics (one per reward channel)
        - 3 target networks (for stable TD targets)
        - 1 meta-weighting network (outputs channel weights conditioned on state)
        - 3 Q-value normalizers (running mean/std per channel)
 
    Training:
        - Critics: standard Double-DQN with experience replay
        - Meta-network: retrospective bandit at episode end
    """
 
    def __init__(
        self,
        n_actions,
        state_dim=FEATURE_DIM,
        hidden_critic=(512, 256),
        hidden_meta=128,
        lr_critic=1e-4,
        lr_meta=3e-4,
        gamma=0.999,
        buffer_size=200_000,
        batch_size=256,
        target_update_freq=2000,
        train_freq=4,
        warmup_steps=10_000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=200_000,
        epsilon_mid=None,
        epsilon_decay_steps_total=None,
        epsilon_schedule=None,
        n_step=3,
        polyak_tau=0.005,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        # NOTE: target_update_freq is retained for hparams-checkpoint back
        # compat but is no longer used — target nets are updated every grad
        # step via Polyak averaging with polyak_tau below.
        self.target_update_freq = target_update_freq
        self.polyak_tau = float(polyak_tau)
        self.train_freq = train_freq
        self.warmup_steps = warmup_steps
        # n-step return horizon and the corresponding Bellman discount factor
        # γ^n. n_step=1 reproduces the original 1-step TD behavior; higher
        # n gives lower-bias targets at the cost of higher variance.
        self.n_step = int(n_step)
        self.gamma_n = gamma ** self.n_step
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        # Optional second-phase decay (two-phase kink schedule). If
        # epsilon_mid and epsilon_decay_steps_total are both set, the
        # schedule is piecewise linear:
        #   phase 1: epsilon_start  → epsilon_mid  over [0, epsilon_decay_steps]
        #   phase 2: epsilon_mid    → epsilon_end  over [epsilon_decay_steps, epsilon_decay_steps_total]
        # Otherwise falls back to single-phase epsilon_start → epsilon_end
        # over epsilon_decay_steps.
        self.epsilon_mid = epsilon_mid
        self.epsilon_decay_steps_total = epsilon_decay_steps_total
        # Optional N-waypoint schedule. If set (list of (step, value)
        # tuples sorted by step), takes precedence over the two-phase and
        # single-phase paths above. Linear interpolation between waypoints;
        # outside the range, clamps to the first/last value. Use this to
        # express schedules with more than one kink (e.g., fast initial
        # decay → moderate decay → long slow tail).
        self.epsilon_schedule = epsilon_schedule
 
        # Critics (one per channel)
        self.critics = [
            QNetwork(state_dim, n_actions, hidden_critic).to(DEVICE)
            for _ in range(N_CHANNELS)
        ]
        self.target_critics = [
            QNetwork(state_dim, n_actions, hidden_critic).to(DEVICE)
            for _ in range(N_CHANNELS)
        ]
        for i in range(N_CHANNELS):
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())
            self.target_critics[i].eval()
 
        self.critic_optimizers = [
            optim.Adam(self.critics[i].parameters(), lr=lr_critic)
            for i in range(N_CHANNELS)
        ]
 
        # Meta-weighting network
        self.meta_net = MetaWeightNetwork(state_dim, N_CHANNELS, hidden_meta).to(DEVICE)
        self.meta_optimizer = optim.Adam(self.meta_net.parameters(), lr=lr_meta)
 
        # Q-value normalizers (one per channel)
        self.q_normalizers = [RunningNormalizer() for _ in range(N_CHANNELS)]
 
        # Replay buffer (preallocated numpy arrays — see ReplayBuffer above)
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            state_dim=state_dim,
            n_actions=n_actions,
            n_channels=N_CHANNELS,
        )
 
        # Episode buffer for meta-network training (single-env path)
        # Each entry: (state_tensor, q_values_tensor[3])
        self.episode_buffer = []

        # Per-env episode buffers for vectorized training (keyed by env idx).
        # Disjoint from self.episode_buffer; populated by select_actions_batch
        # and consumed by train_meta_for_env.
        self.episode_buffers: dict = {}

        # Per-env n-step return processor. All store_transition() calls go
        # through this; for n_step=1 it's a pass-through.
        self.nstep_processor = NStepProcessor(
            n=self.n_step, gamma=gamma, n_channels=N_CHANNELS,
        )

        # Running ω stats across every (env, step) sample seen by
        # select_actions_batch since the last pop_omega_stats() call.
        # Reset by pop_omega_stats() at each TB dump, so the logged
        # mean/std/min/max are aggregated over the full dump window
        # (~32k env-steps × n_envs ≈ hundreds of thousands of samples).
        self._omega_sum = np.zeros(N_CHANNELS, dtype=np.float64)
        self._omega_sumsq = np.zeros(N_CHANNELS, dtype=np.float64)
        self._omega_min = np.full(N_CHANNELS, np.inf, dtype=np.float64)
        self._omega_max = np.full(N_CHANNELS, -np.inf, dtype=np.float64)
        self._omega_count = 0

        # Step counter
        self.total_steps = 0
 
    def _epsilon(self):
        """Epsilon schedule.

        Priority:
          1. N-waypoint piecewise linear if epsilon_schedule is set.
          2. Two-phase kink if epsilon_mid + epsilon_decay_steps_total set.
          3. Single linear epsilon_start → epsilon_end over epsilon_decay_steps.
        """
        if self.epsilon_schedule:
            return self._interp_epsilon_schedule()

        if self.epsilon_mid is None or self.epsilon_decay_steps_total is None:
            progress = min(1.0, self.total_steps / max(1, self.epsilon_decay_steps))
            return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

        # Phase 1: start → mid over [0, epsilon_decay_steps]
        if self.total_steps <= self.epsilon_decay_steps:
            t = self.total_steps / max(1, self.epsilon_decay_steps)
            return self.epsilon_start + (self.epsilon_mid - self.epsilon_start) * t

        # Phase 2: mid → end over [epsilon_decay_steps, epsilon_decay_steps_total]
        phase2_span = max(1, self.epsilon_decay_steps_total - self.epsilon_decay_steps)
        t = min(1.0, (self.total_steps - self.epsilon_decay_steps) / phase2_span)
        return self.epsilon_mid + (self.epsilon_end - self.epsilon_mid) * t

    def _interp_epsilon_schedule(self):
        """Linearly interpolate self.epsilon_schedule at self.total_steps.

        Schedule is a list of (step, value) tuples sorted ascending by step.
        Below the first step: clamps to first value. Above the last step:
        clamps to last value. Between two consecutive waypoints, linearly
        interpolates.
        """
        sched = self.epsilon_schedule
        step = self.total_steps
        if step <= sched[0][0]:
            return sched[0][1]
        if step >= sched[-1][0]:
            return sched[-1][1]
        for i in range(len(sched) - 1):
            s0, v0 = sched[i]
            s1, v1 = sched[i + 1]
            if step <= s1:
                t = (step - s0) / max(1, s1 - s0)
                return v0 + (v1 - v0) * t
        return sched[-1][1]
 
    def select_action(self, state, valid_actions):
        """Epsilon-greedy over Q_final = Σ ω_i · Q_i, masked to legal actions.
        Also stores Q-values in episode buffer for meta-network training."""
 
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
 
        with torch.no_grad():
            # Get Q-values from each critic
            q_values = []
            for i in range(N_CHANNELS):
                q_i = self.critics[i](state_t).squeeze(0)  # (n_actions,)
                q_values.append(q_i)
 
            # Get meta-weights
            omega = self.meta_net(state_t).squeeze(0)  # (N_CHANNELS,)
 
            # Normalize Q-values before combining
            q_normalized = []
            for i in range(N_CHANNELS):
                q_np = q_values[i].cpu().numpy()
                # Update normalizer with Q-values of valid actions only
                valid_q = q_np[valid_actions]
                if len(valid_q) > 0:
                    self.q_normalizers[i].update(valid_q)
                q_norm = torch.FloatTensor(
                    [self.q_normalizers[i].normalize(v) for v in q_np]
                ).to(DEVICE)
                q_normalized.append(q_norm)
 
            # Composite Q-value: Q_final = Σ ω_i · Q_i_normalized
            q_final = torch.zeros(self.n_actions, device=DEVICE)
            for i in range(N_CHANNELS):
                q_final += omega[i] * q_normalized[i]
 
            # Mask illegal actions
            mask = torch.full((self.n_actions,), float("-inf"), device=DEVICE)
            mask[valid_actions] = 0.0
            q_final = q_final + mask
 
        # Store in episode buffer (unnormalized Q of chosen action, detached)
        # We store after action selection so we know which action was taken
 
        # Epsilon-greedy
        if random.random() < self._epsilon():
            action = random.choice(valid_actions)
        else:
            action = q_final.argmax().item()
 
        # Store the NORMALIZED per-channel Q for the chosen action in the
        # episode buffer. Action selection uses q_normalized (above) to
        # form Q_final = Σ ω_i · Q̃_i, so the meta-net regression target
        # must be computed against the same q̃ representation; storing
        # raw Q_i (as we used to) makes train_meta fit ω in a space that
        # doesn't match action selection, silently biasing ω to absorb
        # the per-channel scale difference.
        q_at_action = torch.FloatTensor([
            q_normalized[i][action].item() for i in range(N_CHANNELS)
        ])
        self.episode_buffer.append((
            torch.FloatTensor(state),  # (FEATURE_DIM,)
            q_at_action,               # (N_CHANNELS,) — normalized, detached
        ))
 
        return action
 
    def store_transition(self, state, action, rewards, next_state, done,
                         action_mask, next_action_mask, env_idx=0):
        """Add a 1-step env transition. Internally routed through the
        per-env n-step processor, which only emits to the replay buffer
        once n consecutive transitions have accumulated (or on episode
        end, when it flushes truncated returns).

        env_idx identifies which env queue to use in the n-step processor;
        defaults to 0 for the legacy single-env path. The vectorized
        league trainer passes the actual env index per env per iter.

        total_steps is incremented per ENV STEP, not per buffer push, so
        warmup / epsilon / train_freq cadences are unaffected by n_step."""
        emissions = self.nstep_processor.push(
            env_idx,
            state, action, rewards, next_state, done,
            action_mask, next_action_mask,
        )
        for em in emissions:
            self.replay_buffer.push(*em)
        self.total_steps += 1

    def train_critics(self):
        """One gradient step on each critic using a batch from replay buffer.

        Uses:
          - Double-DQN target (online net picks action, target net evals).
          - n-step Bellman bootstrap: target = R_n + γ^n · (1-done) · max_Q'(s_{t+n}).
            R_n and done are produced by NStepProcessor; γ^n is precomputed
            as self.gamma_n.
          - Huber (smooth_l1) loss to limit gradient blow-up from the ±15
            terminal-VP reward channel.
          - Polyak (soft) target update every grad step instead of a hard
            copy every target_update_freq steps. Smoother targets → no
            loss spikes at copy points.
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
        if self.total_steps < self.warmup_steps:
            return {}
        if self.total_steps % self.train_freq != 0:
            return {}

        states, actions, rewards, next_states, dones, _, next_masks = \
            self.replay_buffer.sample(self.batch_size)

        losses = {}
        tau = self.polyak_tau
        for i in range(N_CHANNELS):
            # Double DQN target
            with torch.no_grad():
                # Online network selects best action
                next_q_online = self.critics[i](next_states)   # (batch, n_actions)
                next_q_online[~next_masks] = float("-inf")
                best_actions = next_q_online.argmax(dim=1, keepdim=True)  # (batch, 1)

                # Target network evaluates that action
                next_q_target = self.target_critics[i](next_states)  # (batch, n_actions)
                max_next_q = next_q_target.gather(1, best_actions).squeeze(1)  # (batch,)

                target = rewards[:, i] + self.gamma_n * (1.0 - dones) * max_next_q

            # Current Q-value for taken action
            current_q = self.critics[i](states).gather(
                1, actions.unsqueeze(1)
            ).squeeze(1)  # (batch,)

            loss = F.smooth_l1_loss(current_q, target)

            self.critic_optimizers[i].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.critics[i].parameters(), 10.0)
            self.critic_optimizers[i].step()

            losses[f"critic_{i}"] = loss.item()

            # Polyak (soft) target update: θ' ← (1-τ)·θ' + τ·θ
            with torch.no_grad():
                for p_t, p in zip(
                    self.target_critics[i].parameters(),
                    self.critics[i].parameters(),
                ):
                    p_t.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

        return losses
 
    def train_meta(self, r_terminal):
        """Retrospective bandit update for the meta-weighting network.
 
        At episode end, for each visited state t:
            G_t = γ^(T-1-t) · r_terminal
            Q_stored = [Q_0(s_t,a_t), Q_1(s_t,a_t), Q_2(s_t,a_t)]  (constants)
            ω = meta_net(s_t)
            Q_pred = Σ ω_i · Q_stored_i
            loss = (G_t - Q_pred)²
 
        Gradients flow through ω only. Q_stored are detached constants.
        """
        if len(self.episode_buffer) == 0:
            return 0.0
        if self.total_steps < self.warmup_steps:
            self.episode_buffer.clear()
            return 0.0
 
        T = len(self.episode_buffer)
        states = torch.stack([s for s, _ in self.episode_buffer]).to(DEVICE)    # (T, FEATURE_DIM)
        q_stored = torch.stack([q for _, q in self.episode_buffer]).to(DEVICE)  # (T, N_CHANNELS)
 
        # G_t = γ^(T-1-t) · r_terminal
        exponents = torch.arange(T - 1, -1, -1, dtype=torch.float32, device=DEVICE)
        G = (self.gamma ** exponents) * r_terminal  # (T,)
 
        # Meta-network forward
        omega = self.meta_net(states)          # (T, N_CHANNELS)
        q_pred = (omega * q_stored).sum(dim=1) # (T,)
 
        loss = F.mse_loss(q_pred, G)
 
        self.meta_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.meta_net.parameters(), 5.0)
        self.meta_optimizer.step()
 
        self.episode_buffer.clear()
        return loss.item()
 
    def get_omega(self, state):
        """Get current meta-weights for a state (for logging/visualization).

        Single-state point sample. Prefer pop_omega_stats() for logging
        during vectorized training; it aggregates ω across all states
        the agent actually saw between dumps.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return self.meta_net(state_t).squeeze(0).cpu().numpy()

    def pop_omega_stats(self):
        """Return aggregated ω statistics across every (env, step) sample
        observed by select_actions_batch since the last call, and reset
        the accumulators.

        Returns a dict {"mean", "std", "min", "max", "count"} where each
        value (except count) is a length-N_CHANNELS float32 array, or
        None if no samples have been accumulated.
        """
        n = self._omega_count
        if n == 0:
            return None
        mean = self._omega_sum / n
        # Var ≥ 0 in exact arithmetic; clamp for float roundoff.
        var = np.maximum(0.0, self._omega_sumsq / n - mean ** 2)
        std = np.sqrt(var)
        stats = {
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "min": self._omega_min.astype(np.float32),
            "max": self._omega_max.astype(np.float32),
            "count": n,
        }
        self._omega_sum.fill(0.0)
        self._omega_sumsq.fill(0.0)
        self._omega_min.fill(np.inf)
        self._omega_max.fill(-np.inf)
        self._omega_count = 0
        return stats

    def select_actions_batch(self, states, valid_actions_list, masks_array):
        """Vectorized action selection for a batch of envs.

        Args:
            states: array-like (n_envs, FEATURE_DIM)
            valid_actions_list: list of n_envs lists of valid action indices
            masks_array: bool array (n_envs, n_actions)

        Returns:
            np.ndarray (n_envs,) int64 — chosen action per env

        Side effects:
            - Updates each per-channel running normalizer with valid Q-values.
            - Appends (state, q_at_action) to self.episode_buffers[env_idx]
              for the meta-network's retrospective update.
        """
        states_np = np.asarray(states, dtype=np.float32)
        n_envs = states_np.shape[0]
        states_t = torch.from_numpy(states_np).to(DEVICE)

        with torch.no_grad():
            # Stack per-channel Q-values: (N_CHANNELS, n_envs, n_actions)
            q_stack = torch.stack(
                [self.critics[c](states_t) for c in range(N_CHANNELS)], dim=0
            )
            q_stack_np = q_stack.cpu().numpy()

            # Update normalizers with valid Q-values from each env (per channel).
            for c in range(N_CHANNELS):
                pieces = []
                for env_idx in range(n_envs):
                    va = valid_actions_list[env_idx]
                    if va:
                        pieces.append(q_stack_np[c, env_idx, va])
                if pieces:
                    self.q_normalizers[c].update(np.concatenate(pieces))

            # Normalize per channel using the just-updated stats.
            means = torch.tensor(
                [n.mean for n in self.q_normalizers], dtype=torch.float32, device=DEVICE
            ).view(N_CHANNELS, 1, 1)
            stds = torch.tensor(
                [max(math.sqrt(n.var), 1e-8) for n in self.q_normalizers],
                dtype=torch.float32, device=DEVICE,
            ).view(N_CHANNELS, 1, 1)
            q_norm = (q_stack - means) / stds  # (N_CHANNELS, n_envs, n_actions)

            # Meta-weights per env: (n_envs, N_CHANNELS).
            omega_raw = self.meta_net(states_t)

            # Accumulate ω stats over (env, step) samples so TB logs the
            # distributional mean/std/min/max over the dump window rather
            # than a single-state snapshot. ~free (one cpu copy per iter).
            omega_np = omega_raw.detach().cpu().numpy()  # (n_envs, N_CHANNELS)
            self._omega_sum += omega_np.sum(axis=0)
            self._omega_sumsq += np.square(omega_np).sum(axis=0)
            self._omega_min = np.minimum(self._omega_min, omega_np.min(axis=0))
            self._omega_max = np.maximum(self._omega_max, omega_np.max(axis=0))
            self._omega_count += omega_np.shape[0]

            # Reshape for broadcasted weighted sum: (N_CHANNELS, n_envs, 1)
            omega = omega_raw.T.unsqueeze(-1)
            q_final = (omega * q_norm).sum(dim=0)  # (n_envs, n_actions)

            # Mask invalid actions.
            mask_t = torch.from_numpy(masks_array).to(DEVICE)
            q_final = q_final.masked_fill(~mask_t, float("-inf"))

            best_actions = q_final.argmax(dim=1).cpu().numpy()  # (n_envs,)

        # Epsilon-greedy per env.
        eps = self._epsilon()
        actions = np.empty(n_envs, dtype=np.int64)
        for env_idx in range(n_envs):
            if random.random() < eps and valid_actions_list[env_idx]:
                actions[env_idx] = random.choice(valid_actions_list[env_idx])
            else:
                actions[env_idx] = best_actions[env_idx]

        # Store the NORMALIZED per-channel Q (NOT raw q_stack_np) for each
        # env's chosen action. Action selection above mixes Q values via
        # Σ ω_i · q_norm_i, so train_meta_for_env must regress ω against
        # the same q̃ representation — otherwise ω is learned in raw-Q
        # space and applied in normalized-Q space, silently distorting
        # the channel weights by each channel's std.
        q_norm_np = q_norm.cpu().numpy()  # (N_CHANNELS, n_envs, n_actions)
        env_range = np.arange(n_envs)
        q_at_actions = q_norm_np[:, env_range, actions]  # (N_CHANNELS, n_envs)
        for env_idx in range(n_envs):
            buf = self.episode_buffers.setdefault(env_idx, [])
            buf.append((
                torch.from_numpy(states_np[env_idx].copy()),
                torch.from_numpy(q_at_actions[:, env_idx].copy()),
            ))

        return actions

    def train_meta_for_env(self, env_idx, r_terminal):
        """Per-env variant of train_meta that consumes self.episode_buffers[env_idx]."""
        buf = self.episode_buffers.get(env_idx, [])
        if not buf:
            return None
        if self.total_steps < self.warmup_steps:
            buf.clear()
            return None

        T = len(buf)
        states = torch.stack([s for s, _ in buf]).to(DEVICE)
        q_stored = torch.stack([q for _, q in buf]).to(DEVICE)

        exponents = torch.arange(T - 1, -1, -1, dtype=torch.float32, device=DEVICE)
        G = (self.gamma ** exponents) * r_terminal

        omega = self.meta_net(states)
        q_pred = (omega * q_stored).sum(dim=1)
        loss = F.mse_loss(q_pred, G)

        self.meta_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.meta_net.parameters(), 5.0)
        self.meta_optimizer.step()

        buf.clear()
        return loss.item()
 
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        for i in range(N_CHANNELS):
            torch.save(self.critics[i].state_dict(), os.path.join(path, f"critic_{i}.pt"))
            torch.save(self.target_critics[i].state_dict(), os.path.join(path, f"target_critic_{i}.pt"))
        torch.save(self.meta_net.state_dict(), os.path.join(path, "meta_net.pt"))
        print(f"SC-HRA model saved to {path}")
 
    def load(self, path):
        for i in range(N_CHANNELS):
            self.critics[i].load_state_dict(
                torch.load(os.path.join(path, f"critic_{i}.pt"), map_location=DEVICE)
            )
            self.target_critics[i].load_state_dict(
                torch.load(os.path.join(path, f"target_critic_{i}.pt"), map_location=DEVICE)
            )
        self.meta_net.load_state_dict(
            torch.load(os.path.join(path, "meta_net.pt"), map_location=DEVICE)
        )
        print(f"SC-HRA model loaded from {path}")

    def save_checkpoint(self, path, extra=None):
        """Save a full training checkpoint so training can resume with no dropoff.

        Includes: all network weights, all optimizer states, replay buffer,
        Q-value normalizers, total_steps, episode_buffer, RNG states, and
        whatever extra trainer state is passed in (episode_count, best_win_rate,
        rolling stat deques, etc.).
        """
        os.makedirs(path, exist_ok=True)
        # Weights via existing save() so eval can still use just the .pt files
        self.save(path)

        ckpt = {
            # Optimizer states
            "critic_optimizers": [opt.state_dict() for opt in self.critic_optimizers],
            "meta_optimizer": self.meta_optimizer.state_dict(),
            # Replay buffer: dict of numpy arrays. No module-bound types
            # (Transition namedtuple etc.) appear in the pickle, so it
            # survives the schra-loaded-via-importlib name issue that
            # broke older saves on Modal.
            "replay_buffer": self.replay_buffer.to_state_dict(),
            "q_normalizers": [
                {"mean": n.mean, "var": n.var, "count": n.count}
                for n in self.q_normalizers
            ],
            # Counters
            "total_steps": self.total_steps,
            # Mid-episode buffer (may be non-empty if checkpointed mid-episode)
            "episode_buffer": self.episode_buffer,
            # Hyperparameters (for sanity-check on resume)
            "hparams": {
                "n_actions": self.n_actions,
                "gamma": self.gamma,
                "batch_size": self.batch_size,
                "target_update_freq": self.target_update_freq,
                "train_freq": self.train_freq,
                "warmup_steps": self.warmup_steps,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay_steps": self.epsilon_decay_steps,
            },
            # RNG states
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
            "extra": extra or {},
        }
        torch.save(ckpt, os.path.join(path, "checkpoint.pt"))
        print(f"SC-HRA full checkpoint saved to {path}/checkpoint.pt")

    def load_checkpoint(self, path):
        """Restore a full training checkpoint. Returns the `extra` dict."""
        # Weights
        self.load(path)

        ckpt_path = os.path.join(path, "checkpoint.pt")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

        # Sanity check hyperparameters
        hp = ckpt.get("hparams", {})
        for key in ("n_actions", "gamma", "warmup_steps",
                    "epsilon_start", "epsilon_end", "epsilon_decay_steps"):
            if key in hp and getattr(self, key) != hp[key]:
                print(
                    f"[resume] WARNING: hparam '{key}' changed "
                    f"({hp[key]} -> {getattr(self, key)})"
                )

        # Optimizer states
        for i, sd in enumerate(ckpt["critic_optimizers"]):
            self.critic_optimizers[i].load_state_dict(sd)
        self.meta_optimizer.load_state_dict(ckpt["meta_optimizer"])

        # Replay buffer. Two on-disk shapes:
        #   - dict (new): snapshot of numpy arrays from to_state_dict().
        #   - list (legacy): list of plain tuples (or Transitions) from
        #     pre-vectorized saves; replayed via push().
        loaded_buf = ckpt["replay_buffer"]
        if isinstance(loaded_buf, dict):
            self.replay_buffer.load_state_dict(loaded_buf)
        else:
            self.replay_buffer.load_from_tuples(loaded_buf)
        for i, n in enumerate(ckpt["q_normalizers"]):
            self.q_normalizers[i].mean = n["mean"]
            self.q_normalizers[i].var = n["var"]
            self.q_normalizers[i].count = n["count"]

        # Counters + mid-episode buffer
        self.total_steps = ckpt["total_steps"]
        self.episode_buffer = ckpt.get("episode_buffer", [])

        # RNG states
        rng = ckpt.get("rng", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])

        extra = ckpt.get("extra", {})
        print(
            f"SC-HRA full checkpoint loaded from {ckpt_path} "
            f"(total_steps={self.total_steps}, buffer={len(self.replay_buffer)})"
        )
        return extra
 
 
# ================================================================
# ENVIRONMENT FACTORY
# ================================================================

def make_env(enemy=None):
    if enemy is None:
        enemy = WeightedRandomPlayer(Color.RED)
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "enemies": [enemy],
            "vps_to_win": 15,
            # Match the SCHRAWrapper HOARD_PENALTY threshold (>9). The env's
            # default discard_limit is 7, which would force the agent to
            # discard down to 7 on every 7-roll — eating the very excess
            # cards we're trying to penalize the agent for holding, AND
            # making the penalty schedule env-mediated rather than
            # consequence-mediated. With discard_limit=9 the env only
            # forces a discard once the agent is already deep in the
            # penalized zone, so the hoard penalty correctly represents
            # the *risk* of having to discard.
            "discard_limit": 9,
        },
    )
    env = SCHRAWrapper(env)
    return env


# ================================================================
# LEAGUE TRAINING
# ================================================================

# Mirrors 185_ppo.py: easy/medium/hard tiers, weights per stage, win-rate
# thresholds for advancing. The stage value is now an mp.Manager.Value proxy
# so it can be shared across SubprocVecEnv worker processes.
OPPONENT_TIERS = [
    ("easy",   lambda: WeightedRandomPlayer(Color.RED)),
    ("medium", lambda: AlphaBetaPlayer(Color.RED, depth=1)),
    ("hard",   lambda: ValueFunctionPlayer(Color.RED)),
]

STAGE_WEIGHTS = [
    [0.70, 0.20, 0.10],
    [0.30, 0.50, 0.20],
    [0.20, 0.40, 0.40],
]

STAGE_UP_THRESHOLDS = [0.70, 0.50]
MIN_TIER_EPISODES = 50


class LeagueWrapper(Wrapper):
    """
    Swaps the opponent inside CatanatronEnv at each episode reset based on
    the current league stage stored in an mp.Manager.Value proxy (shared
    across SubprocVecEnv workers). Injects ``tier_idx`` into info on episode
    end so the trainer can track per-tier win rates.

    Wrapper stack: CatanatronEnv → LeagueWrapper → SCHRAWrapper
    """

    def __init__(self, env, stage_val):
        super().__init__(env)
        self._stage_val = stage_val
        self._tier_idx = 0

    def reset(self, **kwargs):
        stage = min(self._stage_val.value, len(STAGE_WEIGHTS) - 1)
        weights = STAGE_WEIGHTS[stage]
        self._tier_idx = int(np.random.choice(len(weights), p=weights))
        new_opp = OPPONENT_TIERS[self._tier_idx][1]()

        from catanatron.gym.envs.catanatron_env import CatanatronEnv as _CatanatronEnv
        unwrapped: _CatanatronEnv = self.env.unwrapped  # type: ignore[assignment]
        unwrapped.enemies[0] = new_opp
        unwrapped.players[1] = new_opp

        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            info["tier_idx"] = self._tier_idx
        return obs, reward, terminated, truncated, info


def make_league_env(stage_val):
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "enemies": [WeightedRandomPlayer(Color.RED)],
            "vps_to_win": 15,
            "discard_limit": 9,  # see comment in make_env
        },
    )
    env = LeagueWrapper(env, stage_val)
    env = SCHRAWrapper(env)
    return env


class _LeagueEnvFn:
    """Picklable env factory capturing a Manager.Value proxy so SubprocVecEnv
    workers (spawned on macOS / Windows) can share the league stage."""

    def __init__(self, stage_val):
        self._stage_val = stage_val

    def __call__(self):
        # SubprocVecEnv forkserver workers start fresh and do NOT inherit the
        # parent's top-level imports. The gymnasium env id "catanatron/..." is
        # registered as an *import side-effect* of catanatron.gym, so we have
        # to re-trigger it here or gymnasium.make() raises NamespaceNotFound.
        import catanatron.gym  # noqa: F401
        return make_league_env(self._stage_val)
 
 
# ================================================================
# TRAINING LOOP
# ================================================================
 
def train(
    total_timesteps=1_000_000,
    save_path=MODEL_DIR,
    log_dir="./schra_logs",
    eval_freq=10_000,
    eval_episodes=20,
    enemy=None,
    # Agent hyperparameters
    lr_critic=1e-4,
    lr_meta=3e-4,
    gamma=0.999,
    buffer_size=200_000,
    batch_size=256,
    target_update_freq=2000,
    train_freq=4,
    warmup_steps=10_000,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_steps=200_000,
    # Resume / checkpointing
    resume_path=None,
    checkpoint_freq=1_000_000,
    device="cpu",
):
    global DEVICE
    DEVICE = _resolve_device(device)
    print(f"[device] Using {DEVICE}")
    env = make_env(enemy)
    eval_env = make_env(enemy)
    n_actions = env.action_space.n
 
    agent = SCHRAAgent(
        n_actions=n_actions,
        lr_critic=lr_critic,
        lr_meta=lr_meta,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        train_freq=train_freq,
        warmup_steps=warmup_steps,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
    )
 
    writer = SummaryWriter(log_dir)
 
    # Rolling stats
    recent_wins = deque(maxlen=100)
    recent_ep_rewards = deque(maxlen=100)
    recent_ep_lengths = deque(maxlen=100)
 
    obs, info = env.reset()
    episode_reward = 0.0
    episode_channel_rewards = np.zeros(N_CHANNELS)
    episode_steps = 0
    episode_count = 0
    best_win_rate = 0.0
    start_step = 0

    # Resume from checkpoint if requested
    if resume_path is not None:
        extra = agent.load_checkpoint(resume_path)
        start_step = agent.total_steps
        episode_count = extra.get("episode_count", 0)
        best_win_rate = extra.get("best_win_rate", 0.0)
        for x in extra.get("recent_wins", []):
            recent_wins.append(x)
        for x in extra.get("recent_ep_rewards", []):
            recent_ep_rewards.append(x)
        for x in extra.get("recent_ep_lengths", []):
            recent_ep_lengths.append(x)
        # Mid-episode env state is not persisted; start a fresh episode.
        agent.episode_buffer.clear()
        obs, info = env.reset()
        episode_channel_rewards = np.zeros(N_CHANNELS)
        episode_steps = 0
        print(
            f"[resume] Resuming from step {start_step}/{total_timesteps} "
            f"(episode {episode_count}, best_wr={best_win_rate:.1%})"
        )

    print("=" * 60)
    print("SC-HRA Training for 1v1 Catan")
    print(f"Total timesteps: {total_timesteps} (starting from step {start_step})")
    print(f"Action space: {n_actions}")
    print(f"Feature dim: {FEATURE_DIM}")
    print(f"Logging to: {log_dir}")
    print("=" * 60)

    if start_step >= total_timesteps:
        print(
            f"[resume] total_steps ({start_step}) already >= total_timesteps "
            f"({total_timesteps}); nothing to do."
        )
        writer.close()
        env.close()
        eval_env.close()
        return agent

    def _trainer_extra():
        return {
            "episode_count": episode_count,
            "best_win_rate": best_win_rate,
            "recent_wins": list(recent_wins),
            "recent_ep_rewards": list(recent_ep_rewards),
            "recent_ep_lengths": list(recent_ep_lengths),
        }

    for step in range(start_step, total_timesteps):
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            valid_actions = [0]
 
        # Build action mask
        action_mask = np.zeros(n_actions, dtype=bool)
        action_mask[valid_actions] = True
 
        # Select action
        action = agent.select_action(obs, valid_actions)
 
        # Step
        next_obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards = info["rewards"]  # (N_CHANNELS,)
 
        # Next action mask
        next_valid = env.get_valid_actions()
        if not next_valid:
            next_valid = [0]
        next_action_mask = np.zeros(n_actions, dtype=bool)
        next_action_mask[next_valid] = True
 
        # Store transition
        agent.store_transition(
            obs, action, rewards, next_obs, done,
            action_mask, next_action_mask,
        )
 
        # Track episode stats
        episode_channel_rewards += rewards
        episode_steps += 1
 
        # Train critics
        critic_losses = agent.train_critics()
 
        if done:
            # Train meta-network
            r_terminal = info["r_terminal"]
            meta_loss = agent.train_meta(r_terminal)
 
            # Track stats
            won = info.get("win", 0)
            recent_wins.append(won)
            ep_total = float(episode_channel_rewards.sum()) + r_terminal
            recent_ep_rewards.append(ep_total)
            recent_ep_lengths.append(episode_steps)
            episode_count += 1
 
            # Log to TensorBoard
            if episode_count % 10 == 0 and len(recent_wins) >= 10:
                win_rate = np.mean(recent_wins)
                writer.add_scalar("game/win_rate", win_rate, step)
                writer.add_scalar("game/ep_length", np.mean(recent_ep_lengths), step)
                writer.add_scalar("game/ep_reward", np.mean(recent_ep_rewards), step)
                writer.add_scalar("channel/resource", float(episode_channel_rewards[0]), step)
                writer.add_scalar("channel/position", float(episode_channel_rewards[1]), step)
                writer.add_scalar("channel/vp", float(episode_channel_rewards[2]), step)
                writer.add_scalar("channel/terminal", r_terminal, step)
                writer.add_scalar("train/epsilon", agent._epsilon(), step)
                writer.add_scalar("train/meta_loss", meta_loss, step)
 
                # Log meta-weights at current state (early/mid/late proxy)
                omega = agent.get_omega(obs)
                writer.add_scalar("omega/resource", omega[0], step)
                writer.add_scalar("omega/position", omega[1], step)
                writer.add_scalar("omega/vp", omega[2], step)
 
                if critic_losses:
                    for name, val in critic_losses.items():
                        writer.add_scalar(f"train/{name}_loss", val, step)
 
            # Print progress
            if episode_count % 50 == 0:
                wr = np.mean(recent_wins) if recent_wins else 0
                avg_r = np.mean(recent_ep_rewards) if recent_ep_rewards else 0
                avg_l = np.mean(recent_ep_lengths) if recent_ep_lengths else 0
                eps = agent._epsilon()
                print(
                    f"Step {step:>8d} | Ep {episode_count:>5d} | "
                    f"WR {wr:.1%} | R {avg_r:>7.1f} | L {avg_l:>5.0f} | "
                    f"ε {eps:.3f}"
                )
 
            # Periodic evaluation
            if episode_count % (eval_freq // 500 + 1) == 0 and len(recent_wins) >= 20:
                win_rate = np.mean(recent_wins)
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    agent.save_checkpoint(
                        os.path.join(save_path, "best"), extra=_trainer_extra()
                    )
 
            # Reset
            obs, info = env.reset()
            episode_channel_rewards = np.zeros(N_CHANNELS)
            episode_steps = 0
        else:
            obs = next_obs

        # Periodic full-state checkpoint (weights + optimizers + buffer + RNG)
        if checkpoint_freq > 0 and (step + 1) % checkpoint_freq == 0:
            agent.save_checkpoint(
                os.path.join(save_path, "checkpoint"), extra=_trainer_extra()
            )

    # Save final model + final full checkpoint
    agent.save(os.path.join(save_path, "final"))
    agent.save_checkpoint(
        os.path.join(save_path, "checkpoint"), extra=_trainer_extra()
    )
    writer.close()
    env.close()
    eval_env.close()
    print(f"Training complete. Final model at {save_path}/final")
    return agent


def league_train(
    total_timesteps=5_000_000,
    save_path=MODEL_DIR,
    log_dir="./schra_logs",
    eval_freq=20_000,
    eval_episodes=20,
    # Agent hyperparameters. lr is annealed linearly across the full run
    # (3e-4 → 1e-4), matching 185_ppo's AnnealingCallback. Epsilon is NOT
    # tied to total_timesteps — PPO's policy goes greedy well before
    # total_timesteps (entropy is a soft regularizer, not a randomness gate),
    # so we match that effective profile by decaying ε over a small fraction
    # of the run (epsilon_decay_frac below) and letting the agent exploit
    # the rest of the way.
    lr_critic_start=3e-4,
    lr_critic_end=1e-4,
    lr_meta_start=3e-4,
    lr_meta_end=1e-4,
    gamma=0.999,
    buffer_size=200_000,
    batch_size=256,
    target_update_freq=2000,
    train_freq=4,
    warmup_steps=10_000,
    # Four-phase piecewise-linear ε schedule expressed as (step, value)
    # waypoints. Default mirrors 185_ppo's "0.5 → 0.05 over 5M" fast initial
    # decay (matches the new PPO entropy schedule), then continues sc-hra
    # down to 0.01 by 20M, with a long slow tail to 0.005 by 100M. Pass a
    # custom list to override; pass None to use the default built from
    # total_timesteps below.
    epsilon_schedule=None,
    # Resume / checkpointing
    resume_path=None,
    checkpoint_freq=1_000_000,
    start_stage=0,
    device="cpu",
    n_envs=4,
):
    """Opponent-sampling / league training for SC-HRA. Mirrors 185_ppo.league_train:
    starts vs WeightedRandom and shifts the opponent distribution toward
    AlphaBeta/ValueFunction as win-rate thresholds are met. Eval uses a fixed
    WeightedRandom opponent so the curve stays comparable across stages.

    Vectorized: runs ``n_envs`` parallel game instances via SubprocVecEnv,
    each feeding a single replay buffer + DQN learner. Action selection
    is a single batched forward pass across all envs; gradient steps still
    run on the main process."""

    global DEVICE
    DEVICE = _resolve_device(device)
    print(f"[device] Using {DEVICE}")
    print(f"[parallel] n_envs={n_envs}")

    # Manager owns a shared Value proxy that the SubprocVecEnv workers read
    # (and the main process updates) for league-stage advancement. The proxy
    # is picklable across spawn, which fork-incompatible mp.Value isn't on
    # all platforms.
    with mp.Manager() as manager:
        stage_val = manager.Value("i", start_stage)
        env_fns = [_LeagueEnvFn(stage_val) for _ in range(n_envs)]
        vec_env = SubprocVecEnv(env_fns)
        eval_env = make_env(WeightedRandomPlayer(Color.RED))
        n_actions = vec_env.action_space.n

        # Default ε schedule (4-phase piecewise linear). Mirrors the new
        # 185_ppo entropy schedule (fast 0.5 → 0.05 over the first 5M)
        # then keeps decaying past where PPO clamps: 0.05 → 0.01 over
        # 5M–20M, plus a slow tail 0.01 → 0.005 from 20M out to
        # total_timesteps. Pass `epsilon_schedule` to override the default.
        if epsilon_schedule is None:
            epsilon_schedule = [
                (0, 0.5),
                (5_000_000, 0.05),
                (20_000_000, 0.01),
                (max(20_000_001, total_timesteps), 0.005),
            ]

        agent = SCHRAAgent(
            n_actions=n_actions,
            lr_critic=lr_critic_start,
            lr_meta=lr_meta_start,
            gamma=gamma,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            train_freq=train_freq,
            warmup_steps=warmup_steps,
            epsilon_schedule=epsilon_schedule,
        )

        # SB3 logger: prints the same key/value table format 185_ppo gets from
        # MaskablePPO(verbose=1) and also writes TensorBoard scalars.
        os.makedirs(log_dir, exist_ok=True)
        existing = [d for d in os.listdir(log_dir) if d.startswith("SCHRA_")]
        next_id = 1 + max(
            (int(d.split("_", 1)[1]) for d in existing if d.split("_", 1)[1].isdigit()),
            default=0,
        )
        run_dir = os.path.join(log_dir, f"SCHRA_{next_id}")
        sb3_logger = configure_sb3_logger(run_dir, ["stdout", "tensorboard"])
        print(f"[tensorboard] writing events to {run_dir}")

        recent_wins = deque(maxlen=100)
        tier_wins = [deque(maxlen=200) for _ in range(len(OPPONENT_TIERS))]
        recent_ep_rewards = deque(maxlen=100)
        recent_ep_lengths = deque(maxlen=100)
        recent_r_resource = deque(maxlen=100)
        recent_r_position = deque(maxlen=100)
        recent_r_vp = deque(maxlen=100)
        recent_r_terminal = deque(maxlen=100)

        episode_count = 0
        best_win_rate = 0.0
        start_step = 0

        if resume_path is not None:
            extra = agent.load_checkpoint(resume_path)
            start_step = agent.total_steps
            episode_count = extra.get("episode_count", 0)
            best_win_rate = extra.get("best_win_rate", 0.0)
            for x in extra.get("recent_wins", []):
                recent_wins.append(x)
            for x in extra.get("recent_ep_rewards", []):
                recent_ep_rewards.append(x)
            for x in extra.get("recent_ep_lengths", []):
                recent_ep_lengths.append(x)
            for tier_i, ws in enumerate(extra.get("tier_wins", [[] for _ in OPPONENT_TIERS])):
                for x in ws:
                    tier_wins[tier_i].append(x)
            stage_val.value = extra.get("league_stage", start_stage)
            agent.episode_buffer.clear()
            agent.episode_buffers.clear()
            print(
                f"[resume] Resuming from step {start_step}/{total_timesteps} "
                f"(episode {episode_count}, stage {stage_val.value}, best_wr={best_win_rate:.1%})"
            )

        def _apply_anneal(step):
            progress = min(1.0, step / max(1, total_timesteps))
            lr_c = lr_critic_start + (lr_critic_end - lr_critic_start) * progress
            lr_m = lr_meta_start + (lr_meta_end - lr_meta_start) * progress
            for opt in agent.critic_optimizers:
                for pg in opt.param_groups:
                    pg["lr"] = lr_c
            for pg in agent.meta_optimizer.param_groups:
                pg["lr"] = lr_m
            return lr_c, lr_m

        def _trainer_extra():
            return {
                "episode_count": episode_count,
                "best_win_rate": best_win_rate,
                "recent_wins": list(recent_wins),
                "recent_ep_rewards": list(recent_ep_rewards),
                "recent_ep_lengths": list(recent_ep_lengths),
                "tier_wins": [list(w) for w in tier_wins],
                "league_stage": stage_val.value,
            }

        if start_step >= total_timesteps:
            print(
                f"[resume] total_steps ({start_step}) already >= total_timesteps "
                f"({total_timesteps}); nothing to do."
            )
            vec_env.close()
            eval_env.close()
            return agent

        # Match SB3 MaskablePPO league_train cadence: dump table every n_steps*n_envs
        # = 8192*4 = 32768 env steps so output volume is similar to 185_ppo.
        DUMP_INTERVAL = 32768
        last_dump_step = start_step
        last_checkpoint_step = start_step
        iterations = 0
        last_critic_losses: dict = {}
        last_meta_loss = None
        lr_c, lr_m = lr_critic_start, lr_meta_start
        start_time = time.time()

        try:
            from tqdm.rich import tqdm as _tqdm
        except ImportError:
            from tqdm import tqdm as _tqdm
        pbar = _tqdm(total=total_timesteps, initial=start_step)

        # SB3 SubprocVecEnv.reset() returns just the (n_envs, FEATURE_DIM) obs.
        obs = vec_env.reset()
        episode_channel_rewards = [np.zeros(N_CHANNELS) for _ in range(n_envs)]
        episode_steps = [0] * n_envs

        step = start_step
        while step < total_timesteps:
            valid_actions_list = vec_env.env_method("get_valid_actions")
            valid_actions_list = [
                va if va else [0] for va in valid_actions_list
            ]
            masks = np.zeros((n_envs, n_actions), dtype=bool)
            for i, va in enumerate(valid_actions_list):
                masks[i, va] = True

            actions = agent.select_actions_batch(obs, valid_actions_list, masks)

            new_obs, _, dones, infos = vec_env.step(actions)

            next_valid_list = vec_env.env_method("get_valid_actions")
            next_valid_list = [
                va if va else [0] for va in next_valid_list
            ]
            next_masks = np.zeros((n_envs, n_actions), dtype=bool)
            for i, va in enumerate(next_valid_list):
                next_masks[i, va] = True

            for i in range(n_envs):
                rewards_i = np.asarray(infos[i]["rewards"], dtype=np.float32)
                done_i = bool(dones[i])

                # On done, SubprocVecEnv has already auto-reset env i, so
                # new_obs[i] is the new episode's first obs. The TD target
                # zeros next-Q via (1-done), so the stale next_state / mask
                # don't affect learning. env_idx=i routes through the
                # correct per-env queue in the n-step processor; the
                # processor also flushes its queue on done_i=True so
                # transitions from different episodes never get joined.
                agent.store_transition(
                    obs[i], int(actions[i]), rewards_i,
                    new_obs[i], done_i,
                    masks[i], next_masks[i],
                    env_idx=i,
                )
                episode_channel_rewards[i] += rewards_i
                episode_steps[i] += 1

                if done_i:
                    r_terminal = float(infos[i].get("r_terminal", 0.0))
                    meta_loss = agent.train_meta_for_env(i, r_terminal)
                    if meta_loss is not None:
                        last_meta_loss = meta_loss

                    won = int(infos[i].get("win", 0))
                    tier_idx = int(infos[i].get("tier_idx", 0))
                    recent_wins.append(won)
                    tier_wins[tier_idx].append(won)
                    ep_total = float(episode_channel_rewards[i].sum()) + r_terminal
                    recent_ep_rewards.append(ep_total)
                    recent_ep_lengths.append(episode_steps[i])
                    recent_r_resource.append(float(episode_channel_rewards[i][0]))
                    recent_r_position.append(float(episode_channel_rewards[i][1]))
                    recent_r_vp.append(float(episode_channel_rewards[i][2]))
                    recent_r_terminal.append(float(r_terminal))
                    episode_count += 1

                    # Stage advancement: check primary tier for current stage.
                    # Multiple envs can finish in the same iter; the first to
                    # cross the threshold bumps the stage and the others see
                    # the new value via the shared Manager.Value.
                    stage = stage_val.value
                    if stage < len(STAGE_UP_THRESHOLDS):
                        wins = tier_wins[stage]
                        if len(wins) >= MIN_TIER_EPISODES:
                            rate = float(np.mean(list(wins)[-100:]))
                            threshold = STAGE_UP_THRESHOLDS[stage]
                            if rate >= threshold:
                                stage_val.value = stage + 1
                                tier_name = OPPONENT_TIERS[stage][0]
                                print(
                                    f"\n[League] Stage {stage} → {stage + 1}  "
                                    f"({tier_name} win rate: {rate:.1%} ≥ {threshold:.0%})"
                                )

                    episode_channel_rewards[i] = np.zeros(N_CHANNELS)
                    episode_steps[i] = 0

            obs = new_obs

            lr_c, lr_m = _apply_anneal(step)

            critic_losses = agent.train_critics()
            if critic_losses:
                last_critic_losses = critic_losses

            step += n_envs
            pbar.update(n_envs)

            if step - last_dump_step >= DUMP_INTERVAL:
                iterations += 1
                elapsed = max(1e-9, time.time() - start_time)

                if recent_ep_rewards:
                    sb3_logger.record("rollout/ep_rew_mean", float(np.mean(recent_ep_rewards)))
                    sb3_logger.record("rollout/ep_len_mean", float(np.mean(recent_ep_lengths)))
                if recent_wins:
                    sb3_logger.record("rollout/win_rate", float(np.mean(recent_wins)))
                if recent_r_resource:
                    sb3_logger.record("channel/resource", float(np.mean(recent_r_resource)))
                    sb3_logger.record("channel/position", float(np.mean(recent_r_position)))
                    sb3_logger.record("channel/vp", float(np.mean(recent_r_vp)))
                    sb3_logger.record("channel/terminal", float(np.mean(recent_r_terminal)))

                sb3_logger.record("league/stage", float(stage_val.value))
                for i, (name, _) in enumerate(OPPONENT_TIERS):
                    if tier_wins[i]:
                        sb3_logger.record(
                            f"league/win_rate_{name}",
                            float(np.mean(list(tier_wins[i])[-100:])),
                        )

                sb3_logger.record("train/epsilon", float(agent._epsilon()))
                sb3_logger.record("train/lr_critic", float(lr_c))
                sb3_logger.record("train/lr_meta", float(lr_m))
                if last_meta_loss is not None:
                    sb3_logger.record("train/meta_loss", float(last_meta_loss))
                for name, val in last_critic_losses.items():
                    sb3_logger.record(f"train/{name}_loss", float(val))

                # ω stats aggregated across every state seen since the last
                # dump (mean/std/min/max per channel). Falls back to the
                # single-state get_omega(obs[0]) snapshot if the agent
                # hasn't produced any action-selection batches yet (e.g.
                # warmup state on the very first dump).
                omega_stats = agent.pop_omega_stats()
                _omega_channels = ("resource", "position", "vp")
                if omega_stats is not None:
                    for c, name in enumerate(_omega_channels):
                        sb3_logger.record(f"omega/{name}", float(omega_stats["mean"][c]))
                        sb3_logger.record(f"omega/{name}_std", float(omega_stats["std"][c]))
                        sb3_logger.record(f"omega/{name}_min", float(omega_stats["min"][c]))
                        sb3_logger.record(f"omega/{name}_max", float(omega_stats["max"][c]))
                else:
                    omega = agent.get_omega(obs[0])
                    for c, name in enumerate(_omega_channels):
                        sb3_logger.record(f"omega/{name}", float(omega[c]))

                sb3_logger.record("time/fps", int((step - start_step) / elapsed))
                sb3_logger.record("time/iterations", iterations)
                sb3_logger.record("time/time_elapsed", int(elapsed))
                sb3_logger.record("time/total_timesteps", step)

                sb3_logger.dump(step=step)
                last_dump_step = step

            # Periodic full-state checkpoint + best-update. Step increments by
            # n_envs each iter so we can't use modulo; track when we last saved.
            if checkpoint_freq > 0 and step - last_checkpoint_step >= checkpoint_freq:
                agent.save_checkpoint(
                    os.path.join(save_path, "checkpoint_league"), extra=_trainer_extra()
                )
                if len(recent_wins) >= 20:
                    win_rate = float(np.mean(recent_wins))
                    if win_rate > best_win_rate:
                        best_win_rate = win_rate
                        agent.save_checkpoint(
                            os.path.join(save_path, "best_league"), extra=_trainer_extra()
                        )
                last_checkpoint_step = step

        pbar.refresh()
        pbar.close()
        agent.save(os.path.join(save_path, "league_final"))
        agent.save_checkpoint(
            os.path.join(save_path, "checkpoint_league"), extra=_trainer_extra()
        )
        sb3_logger.close()
        vec_env.close()
        eval_env.close()
        print(f"Model saved to {save_path}/league_final")
        print(f"TensorBoard logs at {log_dir} — run: tensorboard --logdir {log_dir}")
        return agent


# ================================================================
# EVALUATION
# ================================================================
 
def evaluate(agent_or_path, n_games=100, enemy=None, verbose=True):
    """Evaluate an SC-HRA agent against an opponent."""
    env = make_env(enemy)
    n_actions = env.action_space.n
 
    if isinstance(agent_or_path, str):
        agent = SCHRAAgent(n_actions)
        agent.load(agent_or_path)
    else:
        agent = agent_or_path
 
    # Set epsilon to 0 for deterministic evaluation
    saved_epsilon_end = agent.epsilon_end
    agent.epsilon_end = 0.0
    agent.epsilon_start = 0.0
 
    wins = 0
    total_vp_margin = 0
    omega_early = []
    omega_late = []
 
    for game_idx in range(n_games):
        obs, info = env.reset()
        done = False
        game_steps = 0
 
        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                valid_actions = [0]
            action = agent.select_action(obs, valid_actions)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            game_steps += 1
 
        # Record omega at different phases
        agent.episode_buffer.clear()
 
        state = env.env.unwrapped.game.state
        my_vps = get_victory_points(state, Color.BLUE)
        opp_vps = max(
            (get_victory_points(state, c) for c in state.colors if c != Color.BLUE),
            default=0,
        )
 
        if env.env.unwrapped.game.winning_color() == Color.BLUE:
            wins += 1
        total_vp_margin += my_vps - opp_vps
 
        if verbose and (game_idx + 1) % 10 == 0:
            print(
                f"Game {game_idx+1}/{n_games} | "
                f"WR {wins/(game_idx+1):.1%} | "
                f"VP margin {total_vp_margin/(game_idx+1):+.1f}"
            )
 
    # Restore epsilon
    agent.epsilon_end = saved_epsilon_end
 
    if verbose:
        print("\n" + "=" * 60)
        print(f"RESULTS ({n_games} games)")
        print(f"Win rate: {wins/n_games:.1%}")
        print(f"Avg VP margin: {total_vp_margin/n_games:+.1f}")
        print("=" * 60)
 
    env.close()
    return wins / n_games
 
 
# ================================================================
# CATANATRON PLAYER CLASS
# ================================================================
 
class SCHRAPlayer(Player):
    """Catanatron Player that uses a trained SC-HRA agent."""
 
    def __init__(self, color, model_path=None):
        super().__init__(color)
        self._env = gymnasium.make("catanatron/Catanatron-v0")
        n_actions = self._env.action_space.n
        self.agent = SCHRAAgent(n_actions)
        if model_path:
            self.agent.load(model_path)
        self.opp_color = None
        # No exploration at play time
        self.agent.epsilon_start = 0.0
        self.agent.epsilon_end = 0.0
 
    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
 
        state = game.state
        if self.opp_color is None:
            self.opp_color = next(c for c in state.colors if c != self.color)
 
        obs = compute_features(state, self.color, self.opp_color)
 
        self._env.reset()
        self._env.unwrapped.game = game
        valid_actions = self._env.unwrapped.get_valid_actions()
        if not valid_actions:
            return playable_actions[0]
 
        action = self.agent.select_action(obs, valid_actions)
        self.agent.episode_buffer.clear()  # don't accumulate across games
        return self._env.unwrapped.actions[action]
 
 
# ================================================================
# MAIN
# ================================================================
 
if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser(description="SC-HRA for 1v1 Catan")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--model-path", type=str, default=os.path.join(MODEL_DIR, "final"))
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--log-dir", type=str, default="./schra_logs")
    parser.add_argument("--save-path", type=str, default=MODEL_DIR)
    parser.add_argument(
        "--enemy", choices=["random", "value", "alphabeta"], default="random",
        help="Opponent type for training/evaluation",
    )
    parser.add_argument("--lr-critic", type=float, default=1e-4)
    parser.add_argument("--lr-meta", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--epsilon-decay", type=int, default=200_000)
    parser.add_argument("--warmup", type=int, default=10_000)
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint directory (containing checkpoint.pt) to "
             "resume training from with no dropoff.",
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=1_000_000,
        help="Save a full training checkpoint every N env steps (0 to disable).",
    )
    parser.add_argument(
        "--league", action="store_true",
        help="Train with opponent sampling / league training (mirrors 185_ppo).",
    )
    parser.add_argument(
        "--league-continue", type=str, default=None,
        help="Path to a checkpoint directory to resume league training from.",
    )
    parser.add_argument(
        "--start-stage", type=int, default=0,
        help="League stage to start/resume from (0=easy, 1=medium, 2=hard).",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Compute device: cpu, gpu (auto-pick cuda/mps), mps (Mac GPU), cuda, or auto.",
    )
    parser.add_argument(
        "--n-envs", type=int, default=15,
        help="Number of parallel envs for league training via SubprocVecEnv "
             "(only used with --league / --league-continue). Default 15 matches "
             "Modal cpu=16 reservation; bump up on local machines with more "
             "P-cores (e.g. 16 on a 20-core M-series Mac).",
    )

    args = parser.parse_args()

    enemy_map = {
        "random": WeightedRandomPlayer(Color.RED),
        "value": ValueFunctionPlayer(Color.RED),
        "alphabeta": AlphaBetaPlayer(Color.RED, depth=1),
    }
    enemy = enemy_map[args.enemy]

    if args.league or args.league_continue:
        league_train(
            total_timesteps=args.timesteps,
            save_path=args.save_path,
            log_dir=args.log_dir,
            warmup_steps=args.warmup,
            resume_path=args.league_continue,
            checkpoint_freq=args.checkpoint_freq,
            start_stage=args.start_stage,
            device=args.device,
            n_envs=args.n_envs,
        )
    elif args.mode == "train":
        train(
            total_timesteps=args.timesteps,
            save_path=args.save_path,
            log_dir=args.log_dir,
            enemy=enemy,
            lr_critic=args.lr_critic,
            lr_meta=args.lr_meta,
            gamma=args.gamma,
            epsilon_decay_steps=args.epsilon_decay,
            warmup_steps=args.warmup,
            resume_path=args.resume,
            checkpoint_freq=args.checkpoint_freq,
            device=args.device,
        )
    elif args.mode == "eval":
        agent = SCHRAAgent(n_actions=290)
        agent.load(args.model_path)
        evaluate(agent, n_games=args.eval_games, enemy=enemy)