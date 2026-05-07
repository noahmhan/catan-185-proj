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
from collections import deque, namedtuple
 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
 
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
 
RESOURCE_TYPES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
 
MODEL_DIR = os.path.join(os.path.dirname(__file__), "schra_model")
 
 
# ================================================================
# STATE HELPERS (shared with baseline)
# ================================================================
 
def get_player_hand(state, color):
    key = player_key(state, color)
    return {r: state.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCE_TYPES}
 
 
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
    board = state.board
    score = 0.0
    pip_counts = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 0, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
    tile_coord = {tile.id: coord for coord, tile in board.map.land_tiles.items()}
    robber_coordinate = board.robber_coordinate
    resource_types_accessible = set()
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
            score += pips * 0.1 * multiplier
 
    score += len(resource_types_accessible) * 0.6
 
    has_generic_port = False
    for resource, node_ids in board.map.port_nodes.items():
        if any(nid in buildings and buildings[nid][0] == color for nid in node_ids):
            if resource is None:
                has_generic_port = True
            else:
                score += 0.4
    if has_generic_port:
        score += 0.8
 
    roads = [(edge, c) for edge, c in board.roads.items() if c == color]
    num_roads = len(roads) // 2
    longest_road = get_longest_road_length(state, color)
    score += num_roads * 0.1 + longest_road * 0.1
 
    subgraphs = board.find_connected_components(color)
    buildable = board.buildable_node_ids(color) if subgraphs else []
    for node_id in buildable:
        pip_sum = 0
        for tile in board.map.adjacent_tiles.get(node_id, []):
            if tile.number:
                pip_sum += pip_counts.get(tile.number, 0)
        score += 0.04 * pip_sum
 
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
    vp_margin = 0.5 * (my_vps - opp_vps)
    return (15.0 if winning_color == color else -15.0) + vp_margin
 
 
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
    Input: state features (FEATURE_DIM).  Output: Q(s,a) for all actions."""
 
    def __init__(self, state_dim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
 
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
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
 
    def push(self, *args):
        self.buffer.append(Transition(*args))
 
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor(np.array([t.state for t in batch])).to(DEVICE)
        actions = torch.LongTensor([t.action for t in batch]).to(DEVICE)
        rewards = torch.FloatTensor(np.array([t.rewards for t in batch])).to(DEVICE)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(DEVICE)
        dones = torch.FloatTensor([float(t.done) for t in batch]).to(DEVICE)
        action_masks = torch.BoolTensor(np.array([t.action_mask for t in batch])).to(DEVICE)
        next_action_masks = torch.BoolTensor(np.array([t.next_action_mask for t in batch])).to(DEVICE)
        return states, actions, rewards, next_states, dones, action_masks, next_action_masks
 
    def __len__(self):
        return len(self.buffer)
 
 
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
    3. Compute terminal reward at episode end.
    4. Track episode stats for logging.
    """
 
    def __init__(self, env):
        super().__init__(env)
        self.p0_color = Color.BLUE
        self.opp_color = Color.RED
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(FEATURE_DIM,), dtype=np.float32
        )
        self._prev_resource_score = 0.0
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
 
    def get_valid_actions(self):
        return self.env.unwrapped.get_valid_actions()
 
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        state = self.env.unwrapped.game.state
        self._prev_resource_score = resource_flow_score(state, self.p0_color)
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
 
        # Channel 0: resource flow delta
        cur_resource = resource_flow_score(state, self.p0_color)
        r_resource = cur_resource - self._prev_resource_score
        self._prev_resource_score = cur_resource
 
        # Channel 1: network position delta (scaled 0.3)
        cur_position = self._safe_position_score(state)
        r_position = 0.3 * (cur_position - self._prev_position_score)
        self._prev_position_score = cur_position
 
        # Channel 2: VP proximity
        r_vp = vp_proximity_reward(
            state, self.p0_color,
            self._prev_vps, self._prev_knights,
            self.opp_color,
        )
        self._prev_vps = get_victory_points(state, self.p0_color)
        self._prev_knights = get_knights_played(state, self.p0_color)
 
        rewards = np.array([r_resource, r_position, r_vp], dtype=np.float32)
 
        r_terminal = 0.0
        if done:
            r_terminal = terminal_reward(game, self.p0_color)
 
        # expose everything in info
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
        hidden_critic=256,
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
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_freq = train_freq
        self.warmup_steps = warmup_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
 
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
 
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
 
        # Episode buffer for meta-network training
        # Each entry: (state_tensor, q_values_tensor[3])
        self.episode_buffer = []
 
        # Step counter
        self.total_steps = 0
 
    def _epsilon(self):
        """Linear epsilon decay."""
        progress = min(1.0, self.total_steps / self.epsilon_decay_steps)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress
 
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
 
        # Store Q_i(s, a) for the chosen action in episode buffer
        q_at_action = torch.FloatTensor([
            q_values[i][action].item() for i in range(N_CHANNELS)
        ])
        self.episode_buffer.append((
            torch.FloatTensor(state),  # (FEATURE_DIM,)
            q_at_action,               # (N_CHANNELS,) — detached from graph
        ))
 
        return action
 
    def store_transition(self, state, action, rewards, next_state, done,
                         action_mask, next_action_mask):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(
            state, action, rewards, next_state, done,
            action_mask, next_action_mask,
        )
        self.total_steps += 1
 
    def train_critics(self):
        """One gradient step on each critic using a batch from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        if self.total_steps < self.warmup_steps:
            return {}
        if self.total_steps % self.train_freq != 0:
            return {}
 
        states, actions, rewards, next_states, dones, _, next_masks = \
            self.replay_buffer.sample(self.batch_size)
 
        losses = {}
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
 
                target = rewards[:, i] + self.gamma * (1.0 - dones) * max_next_q
 
            # Current Q-value for taken action
            current_q = self.critics[i](states).gather(
                1, actions.unsqueeze(1)
            ).squeeze(1)  # (batch,)
 
            loss = F.mse_loss(current_q, target)
 
            self.critic_optimizers[i].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.critics[i].parameters(), 10.0)
            self.critic_optimizers[i].step()
 
            losses[f"critic_{i}"] = loss.item()
 
        # Update target networks
        if self.total_steps % self.target_update_freq == 0:
            for i in range(N_CHANNELS):
                self.target_critics[i].load_state_dict(
                    self.critics[i].state_dict()
                )
 
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
        """Get current meta-weights for a state (for logging/visualization)."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return self.meta_net(state_t).squeeze(0).cpu().numpy()
 
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
            # Replay buffer + normalizers
            "replay_buffer": self.replay_buffer.buffer,  # deque of Transitions
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

        # Replay buffer + normalizers
        self.replay_buffer.buffer = ckpt["replay_buffer"]
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
        },
    )
    env = SCHRAWrapper(env)
    return env
 
 
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
    checkpoint_freq=50_000,
):
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
        "--checkpoint-freq", type=int, default=50_000,
        help="Save a full training checkpoint every N env steps (0 to disable).",
    )
 
    args = parser.parse_args()
 
    enemy_map = {
        "random": WeightedRandomPlayer(Color.RED),
        "value": ValueFunctionPlayer(Color.RED),
        "alphabeta": AlphaBetaPlayer(Color.RED, depth=1),
    }
    enemy = enemy_map[args.enemy]
 
    if args.mode == "train":
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
        )
    elif args.mode == "eval":
        agent = SCHRAAgent(n_actions=290)
        agent.load(args.model_path)
        evaluate(agent, n_games=args.eval_games, enemy=enemy)