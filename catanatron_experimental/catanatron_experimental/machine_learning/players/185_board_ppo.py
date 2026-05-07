"""
Full-game 1v1 Catan PPO trainer using the board-state observation
encoding from initial_placement_ppo.py.

Mirrors 185_ppo.py for everything except the observation and the
resource reward channel:
    - Composite reward (cumulative resource throughput + 0.3 * network
      position delta + VP proximity + terminal). The resource channel
      pays out on cards earned AND on cards spent for builds, instead
      of the old delta-of-stock formulation that locally punished
      spending.
    - Training modes: from-scratch vs WeightedRandom / AlphaBeta /
      epsilon-greedy ValueFunction; league training with opponent
      sampling and auto-promotion
    - Same league callbacks, entropy/lr schedules, VecNormalize on
      reward, MaskablePPO + action masking

Difference: the agent observes the board directly
    - per-node port one-hot + pip-by-resource + settlements/cities (blue/red)
    - per-edge roads (blue/red)
    - phase one-hot for the 4 initial-build decisions (zeros after)

OBS_DIM ≈ 958 (54 nodes × 15 + 72 edges × 2 + 4), so the policy
network is wider than the hand-crafted version.
"""

import os
import importlib.util
import multiprocessing as mp

import numpy as np
import gymnasium
from gymnasium import Wrapper, ObservationWrapper, spaces

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from catanatron import Color, Player
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.models.map import build_map, NUM_NODES, NUM_EDGES
from catanatron.models.board import get_edges
from catanatron.models.enums import RESOURCES, ActionType, ActionPrompt
from catanatron.gym.envs.action_space import get_action_array
from catanatron.cli import register_cli_player
import catanatron.gym  # noqa: F401  registers the env

import torch
torch.distributions.Distribution.set_default_validate_args(False)


# ================================================================
# Pull stable bits from 185_ppo.py via importlib (digit-leading name).
# Reward calc functions, league infra, helpers, and logging callbacks
# are reused as-is. The CatanRewardWrapper is copied inline below so
# the user can edit reward shaping in this file without touching
# 185_ppo.py.
# ================================================================

_PPO_185_PATH = os.path.join(os.path.dirname(__file__), "185_ppo.py")
_spec = importlib.util.spec_from_file_location("ppo_185", _PPO_185_PATH)
_ppo_185 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ppo_185)

resource_flow_score = _ppo_185.resource_flow_score   # kept for back-compat; unused below
vp_proximity_reward = _ppo_185.vp_proximity_reward
terminal_reward = _ppo_185.terminal_reward
get_victory_points = _ppo_185.get_victory_points
get_knights_played = _ppo_185.get_knights_played
get_longest_road_length = _ppo_185.get_longest_road_length
get_hand_size = _ppo_185.get_hand_size

LeagueWrapper = _ppo_185.LeagueWrapper
LeagueAdaptCallback = _ppo_185.LeagueAdaptCallback
RewardLoggingCallback = _ppo_185.RewardLoggingCallback
EntropyAnnealCallback = _ppo_185.EntropyAnnealCallback
EpsilonGreedyPlayer = _ppo_185.EpsilonGreedyPlayer
linear_schedule = _ppo_185.linear_schedule


# ================================================================
# TUNED network_position_score
# Reweighted from the 185_ppo.py version:
#   - pip score per pip:       0.1   → 0.15
#   - resource diversity:      0.6   → 0.8 per type
#   - generic 3:1 port:        0.8   → 0.2
#   - specific port:  0.1 + 0.06*pips → 0.05 * matching_pips (no flat baseline)
#   - per-road bonus:          0.1   → 0.03
#   - longest-road bonus:      0.1*L → 0   (already counted via the VP-delta
#                                            channel; was double-counting)
# Goal: make settlements/cities worth more relative to roads and ports
# (so the bot stops rushing 3:1 and over-building roads), without changing
# the function used by 185_ppo.py / initial_placement_ppo.py.
# ================================================================

_PIP_COUNTS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 0, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}


def network_position_score(state, color):
    board = state.board
    score = 0.0

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

            pips = _PIP_COUNTS.get(tile.number, 0) if tile.number else 0
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
    # longest-road bonus removed: longest-road VP already flows through vp_proximity_reward.

    subgraphs = board.find_connected_components(color)
    buildable = board.buildable_node_ids(color) if subgraphs else []
    for node_id in buildable:
        pip_sum = 0
        for tile in board.map.adjacent_tiles.get(node_id, []):
            if tile.number:
                pip_sum += _PIP_COUNTS.get(tile.number, 0)
        score += 0.06 * pip_sum

    return score


# ================================================================
# CONSTANTS
# ================================================================

MODEL_DIR = os.path.join(os.path.dirname(__file__), "ppo_board_model")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_ppo_board", "best_model")
MAP_TYPE = "BASE"
AGENT_COLOR = Color.BLUE
OPPONENT_COLOR = Color.RED
PLAYER_COLORS = (AGENT_COLOR, OPPONENT_COLOR)


# ================================================================
# OBSERVATION ENCODING (from initial_placement_ppo.py)
# ================================================================

RESOURCE_TO_IDX = {r: i for i, r in enumerate(RESOURCES)}
PORT_RESOURCE_DIM = len(RESOURCES) + 1   # +1 for generic 3:1
PIP_RESOURCE_DIM = len(RESOURCES)
PIP_SCALE = 5.0 / 36.0   # one max-pip tile (6/8) contributes 1.0 per resource

_REF_MAP = build_map(MAP_TYPE)
_NODE_IDS = sorted(_REF_MAP.land_nodes)
_EDGES = sorted(tuple(sorted(e)) for e in get_edges(_REF_MAP.land_nodes))
_NODE_IDX = {nid: i for i, nid in enumerate(_NODE_IDS)}
_EDGE_IDX = {e: i for i, e in enumerate(_EDGES)}

# Per-node: port one-hot (6) + pip-by-resource (5) + blue/red settle/city (4)
NODE_FEATS_DIM = PORT_RESOURCE_DIM + PIP_RESOURCE_DIM + 4
EDGE_FEATS_DIM = 2
PHASE_DIM = 4

OBS_DIM = (
    NUM_NODES * NODE_FEATS_DIM
    + NUM_EDGES * EDGE_FEATS_DIM
    + PHASE_DIM
)


def encode_board_state(board, state) -> np.ndarray:
    """Flat board encoding shared with initial_placement_ppo.

    Layout (in order):
        per-node block: 54 × 15
            [0:6]   port one-hot (5 resource ports + 1 generic)
            [6:11]  pip-by-resource (sum of adjacent-tile pip probs)
            [11:13] blue/red settlement
            [13:15] blue/red city
        per-edge block: 72 × 2 (blue/red road)
        phase one-hot: 4   (zeros after initial build phase)
    """
    out = np.zeros((OBS_DIM,), dtype=np.float32)
    cursor = 0

    node_block = out[cursor : cursor + NUM_NODES * NODE_FEATS_DIM].reshape(
        NUM_NODES, NODE_FEATS_DIM
    )
    cursor += NUM_NODES * NODE_FEATS_DIM

    port_end = PORT_RESOURCE_DIM
    pip_end = port_end + PIP_RESOURCE_DIM

    # Ports (port resources reshuffle each episode).
    for resource, node_ids in board.map.port_nodes.items():
        slot = len(RESOURCES) if resource is None else RESOURCE_TO_IDX[resource]
        for nid in node_ids:
            i = _NODE_IDX.get(nid)
            if i is not None:
                node_block[i, slot] = 1.0

    # Pip-prob totals per resource from precomputed node_production.
    for nid, production in board.map.node_production.items():
        i = _NODE_IDX.get(nid)
        if i is None:
            continue
        for resource, pip in production.items():
            node_block[i, port_end + RESOURCE_TO_IDX[resource]] = float(pip) / PIP_SCALE

    # Buildings.
    for node_id, (color, bldg_type) in board.buildings.items():
        i = _NODE_IDX.get(node_id)
        if i is None:
            continue
        if bldg_type == "SETTLEMENT":
            off = 0
        elif bldg_type == "CITY":
            off = 2
        else:
            continue
        if color == AGENT_COLOR:
            node_block[i, pip_end + off] = 1.0
        else:
            node_block[i, pip_end + off + 1] = 1.0

    # Roads.
    edge_block = out[cursor : cursor + NUM_EDGES * EDGE_FEATS_DIM].reshape(
        NUM_EDGES, EDGE_FEATS_DIM
    )
    cursor += NUM_EDGES * EDGE_FEATS_DIM
    for edge, color in board.roads.items():
        canon = tuple(sorted(edge))
        j = _EDGE_IDX.get(canon)
        if j is None:
            continue
        if color == AGENT_COLOR:
            edge_block[j, 0] = 1.0
        else:
            edge_block[j, 1] = 1.0

    # Phase one-hot.
    phase = placement_phase_index(state, board)
    if 0 <= phase < PHASE_DIM:
        out[cursor + phase] = 1.0
    cursor += PHASE_DIM

    return out


def placement_phase_index(state, board) -> int:
    """0=1st settle, 1=1st road, 2=2nd settle, 3=2nd road. -1 after."""
    if not state.is_initial_build_phase:
        return -1
    blue_settlements = sum(
        1 for _, (c, t) in board.buildings.items()
        if c == AGENT_COLOR and t == "SETTLEMENT"
    )
    blue_roads = sum(1 for e, c in board.roads.items() if c == AGENT_COLOR) // 2
    placing_road = state.current_prompt == ActionPrompt.BUILD_INITIAL_ROAD

    if blue_settlements == 0 and not placing_road:
        return 0
    if blue_settlements == 1 and placing_road and blue_roads == 0:
        return 1
    if blue_settlements == 1 and not placing_road:
        return 2
    if blue_settlements == 2 and placing_road and blue_roads == 1:
        return 3
    return -1


class BoardStateObsWrapper(ObservationWrapper):
    """Replaces the default flat Catanatron observation with the board-state
    encoding above (per-node ports/pips/buildings + per-edge roads + phase)."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

    def observation(self, obs):
        game = self.env.unwrapped.game
        return encode_board_state(game.state.board, game.state)


# ================================================================
# REWARD WRAPPER (composite reward)
#
# Resource channel reworked vs. 185_ppo.py:
#   The old design rewarded the delta of a hand "stock score" (diversity +
#   quantity − hoard). That channel locally PUNISHED building, since a 4-card
#   settlement spend dropped quantity_bonus by 0.4 even though the build is
#   exactly what we want.
#
#   New design rewards resource *throughput* cumulatively:
#     + EARN_REWARD per card gained (positive hand delta from any source —
#       dice, trades, year-of-plenty, monopoly we play)
#     + SPEND_REWARD per card spent on a build (negative hand delta in the
#       same step that a new building or road appeared for us)
#     − HOARD_PENALTY per card over 9 (held-step rate, prevents sitting on
#       a 7-vulnerable hand)
#   Hand drops with no concurrent build (discards on 7, robbed, monopolized
#   against us) give 0 — neither rewarded nor punished.
#
#   Diversity is no longer rewarded directly: in Catan, diversity matters
#   only insofar as it enables building, and the SPEND_REWARD already pays
#   that out.
# ================================================================

class CatanRewardWrapper(Wrapper):
    """cumulative resource throughput + 0.3 * network position delta + VP proximity + terminal.

    On episode end, injects per-channel totals + win flag into info for
    RewardLoggingCallback.
    """

    EARN_REWARD = 0.05
    SPEND_REWARD = 0.05
    HOARD_PENALTY = 0.05

    def __init__(self, env):
        super().__init__(env)
        self.p0_color = AGENT_COLOR
        self.opp_color = OPPONENT_COLOR
        self._prev_hand_size = 0
        self._prev_building_count = 0
        self._prev_road_count = 0
        self._prev_position_score = 0.0
        self._prev_vps = 0
        self._prev_knights = 0
        self._prev_opp_vps = 0
        self._ep_r_resource = 0.0
        self._ep_r_position = 0.0
        self._ep_r_vp = 0.0
        self._ep_r_terminal = 0.0

    def _building_count(self, state):
        return sum(
            1 for _, (c, _) in state.board.buildings.items() if c == self.p0_color
        )

    def _road_count(self, state):
        return sum(1 for _, c in state.board.roads.items() if c == self.p0_color) // 2

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        state = self.env.unwrapped.game.state
        # CatanatronEnv doesn't forward friendly_robber to Game(), so flip it
        # on after construction. The robber rule is read fresh from state at
        # each action-generation step, so this persists for the whole episode.
        state.friendly_robber = True
        self._prev_hand_size = get_hand_size(state, self.p0_color)
        self._prev_building_count = self._building_count(state)
        self._prev_road_count = self._road_count(state)
        self._prev_position_score = network_position_score(state, self.p0_color)
        self._prev_vps = get_victory_points(state, self.p0_color)
        self._prev_knights = get_knights_played(state, self.p0_color)
        self._prev_opp_vps = get_victory_points(state, self.opp_color)
        self._ep_r_resource = 0.0
        self._ep_r_position = 0.0
        self._ep_r_vp = 0.0
        self._ep_r_terminal = 0.0
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        game = self.env.unwrapped.game
        state = game.state
        done = terminated or truncated

        # ---- Resource (cumulative throughput) channel ----
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

        # ---- Position channel (delta of network_position_score) ----
        current_position = network_position_score(state, self.p0_color)
        r_position = 0.3 * (current_position - self._prev_position_score)
        self._prev_position_score = current_position

        # ---- VP proximity ----
        r_vp = vp_proximity_reward(
            state, self.p0_color,
            self._prev_vps, self._prev_knights,
            self.opp_color,
        )
        self._prev_vps = get_victory_points(state, self.p0_color)
        self._prev_knights = get_knights_played(state, self.p0_color)
        self._prev_opp_vps = get_victory_points(state, self.opp_color)

        # ---- Terminal ----
        r_terminal = terminal_reward(game, self.p0_color) if done else 0.0

        self._ep_r_resource += r_resource
        self._ep_r_position += r_position
        self._ep_r_vp += r_vp
        self._ep_r_terminal += r_terminal

        if done:
            info["ep_r_resource"] = self._ep_r_resource
            info["ep_r_position"] = self._ep_r_position
            info["ep_r_vp"] = self._ep_r_vp
            info["ep_r_terminal"] = self._ep_r_terminal
            info["win"] = int(game.winning_color() == self.p0_color)

        reward = r_resource + r_position + r_vp + r_terminal
        return obs, reward, terminated, truncated, info


# ================================================================
# ACTION MASK
# ================================================================

def mask_fn(env) -> np.ndarray:
    valid = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=bool)
    if valid:
        mask[valid] = True
    else:
        mask[0] = True   # never all-False (NaN softmax guard)
    return mask


# ================================================================
# ENV FACTORIES
# Wrapper stack: CatanatronEnv → CatanRewardWrapper [→ LeagueWrapper]
#                → BoardStateObsWrapper → ActionMasker
# ================================================================

def make_env():
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "enemies": [WeightedRandomPlayer(OPPONENT_COLOR)],
            "map_type": MAP_TYPE,
            "vps_to_win": 15,
            "discard_limit": 9,
        },
    )
    env = CatanRewardWrapper(env)
    env = BoardStateObsWrapper(env)
    env = ActionMasker(env, mask_fn)
    return env


def make_env_hard():
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "enemies": [AlphaBetaPlayer(OPPONENT_COLOR, depth=1)],
            "map_type": MAP_TYPE,
            "vps_to_win": 15,
            "discard_limit": 9,
        },
    )
    env = CatanRewardWrapper(env)
    env = BoardStateObsWrapper(env)
    env = ActionMasker(env, mask_fn)
    return env


def make_env_medium(epsilon=0.4):
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "enemies": [EpsilonGreedyPlayer(ValueFunctionPlayer(OPPONENT_COLOR), epsilon=epsilon)],
            "map_type": MAP_TYPE,
            "vps_to_win": 15,
            "discard_limit": 9,
        },
    )
    env = CatanRewardWrapper(env)
    env = BoardStateObsWrapper(env)
    env = ActionMasker(env, mask_fn)
    return env


def make_league_env(stage_val):
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "enemies": [WeightedRandomPlayer(OPPONENT_COLOR)],
            "map_type": MAP_TYPE,
            "vps_to_win": 15,
            "discard_limit": 9,
        },
    )
    env = CatanRewardWrapper(env)
    env = LeagueWrapper(env, stage_val)
    env = BoardStateObsWrapper(env)
    env = ActionMasker(env, mask_fn)
    return env


class _MediumEnvFn:
    """Picklable env factory for SubprocVecEnv (Windows spawn-safe)."""

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self):
        return make_env_medium(self.epsilon)


class _LeagueEnvFn:
    def __init__(self, stage_val):
        self._stage_val = stage_val

    def __call__(self):
        return make_league_env(self._stage_val)


# ================================================================
# TRAINING
# ================================================================

# Wider net than 185_ppo to handle the ~958-dim board observation.
_DEFAULT_NET_ARCH = dict(pi=[512, 512, 256], vf=[512, 512, 256])


def train(
    total_timesteps=1_000_000,
    save_path=MODEL_DIR,
    log_dir="./ppo_board_logs",
    eval_freq=10_000,
    n_eval_episodes=20,
    n_envs=4,
    env_fn=make_env,
    device="auto",
):
    env = SubprocVecEnv([env_fn] * n_envs)
    eval_env = env_fn()

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=512,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=device,
        policy_kwargs={"net_arch": _DEFAULT_NET_ARCH},
    )

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}/best",
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, RewardLoggingCallback()],
        progress_bar=True,
    )
    model.save(f"{save_path}/final_model")
    print(f"Model saved to {save_path}/final_model")
    print(f"TensorBoard logs at {log_dir} — run: tensorboard --logdir {log_dir}")
    env.close()
    eval_env.close()
    return model


def train_vs_medium(
    epsilon=0.4,
    total_timesteps=5_000_000,
    save_path=MODEL_DIR,
    log_dir="./ppo_board_logs",
    eval_freq=20_000,
    n_eval_episodes=20,
    n_envs=4,
    load_path=None,
    device="auto",
):
    """Train (or fine-tune) against an epsilon-greedy ValueFunctionPlayer."""
    vec_normalize_path = os.path.join(save_path, "vecnormalize_medium.pkl")

    raw_env = SubprocVecEnv([_MediumEnvFn(epsilon) for _ in range(n_envs)])
    if load_path and os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, raw_env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(raw_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    eval_env = VecNormalize(
        DummyVecEnv([_MediumEnvFn(epsilon)]),
        norm_obs=False, norm_reward=False, training=False,
    )

    if load_path:
        model = MaskablePPO.load(load_path, env=env, device=device)
    else:
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            n_steps=8192,
            batch_size=512,
            n_epochs=10,
            gamma=0.999,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.08,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device=device,
            policy_kwargs={"net_arch": _DEFAULT_NET_ARCH},
        )

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}/best_medium",
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, RewardLoggingCallback()],
        reset_num_timesteps=(load_path is None),
        progress_bar=True,
    )
    model.save(f"{save_path}/medium_model")
    env.save(vec_normalize_path)
    print(f"Model saved to {save_path}/medium_model")
    print(f"VecNormalize stats saved to {vec_normalize_path}")
    print(f"TensorBoard logs at {log_dir} — run: tensorboard --logdir {log_dir}")
    env.close()
    eval_env.close()
    return model


def continue_training(
    load_path,
    save_path=MODEL_DIR,
    log_dir="./ppo_board_logs",
    total_timesteps=500_000,
    eval_freq=10_000,
    n_eval_episodes=20,
    n_envs=4,
    device="auto",
):
    """Fine-tune an existing checkpoint against AlphaBetaPlayer(depth=1)."""
    env = SubprocVecEnv([make_env_hard] * n_envs)
    eval_env = make_env_hard()

    model = MaskablePPO.load(load_path, env=env, device=device)

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}/best_hard",
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, RewardLoggingCallback()],
        reset_num_timesteps=False,
        progress_bar=True,
    )
    model.save(f"{save_path}/final_model_hard")
    print(f"Model saved to {save_path}/final_model_hard")
    env.close()
    eval_env.close()
    return model


def league_train(
    total_timesteps=2_000_000,
    save_path=MODEL_DIR,
    log_dir="./ppo_board_logs",
    eval_freq=20_000,
    n_eval_episodes=20,
    n_envs=4,
    load_path=None,
    start_stage=0,
    device="auto",
):
    """Train (or fine-tune) with opponent sampling across the easy/medium/hard
    tiers in 185_ppo.OPPONENT_TIERS. Eval uses a fixed WeightedRandom opponent
    so the mean-reward curve stays comparable across stages."""
    with mp.Manager() as manager:
        stage_val = manager.Value("i", start_stage)
        env_fns = [_LeagueEnvFn(stage_val) for _ in range(n_envs)]
        vec_normalize_path = os.path.join(save_path, "vecnormalize.pkl")

        if load_path and os.path.exists(vec_normalize_path):
            env = VecNormalize.load(vec_normalize_path, SubprocVecEnv(env_fns))
            env.training = True
            env.norm_reward = True
        else:
            env = VecNormalize(SubprocVecEnv(env_fns), norm_obs=False, norm_reward=True, clip_reward=10.0)
        eval_env = VecNormalize(DummyVecEnv([make_env]), norm_obs=False, norm_reward=False, training=False)

        if load_path:
            model = MaskablePPO.load(load_path, env=env, device=device)
            learn_timesteps = max(1, total_timesteps - model.num_timesteps)
        else:
            model = MaskablePPO(
                MaskableActorCriticPolicy,
                env,
                verbose=1,
                tensorboard_log=log_dir,
                learning_rate=linear_schedule(3e-4, 1e-5),
                n_steps=8192,
                batch_size=1024,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.08,
                vf_coef=0.5,
                max_grad_norm=0.5,
                device=device,
                policy_kwargs={"net_arch": _DEFAULT_NET_ARCH},
            )

        eval_callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=f"{save_path}/best_league",
            log_path=log_dir,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
        )
        entropy_callback = (
            EntropyAnnealCallback(0.01, 0.005) if load_path
            else EntropyAnnealCallback(0.08, 0.01)
        )
        model.learn(
            total_timesteps=learn_timesteps if load_path else total_timesteps,
            callback=[eval_callback, RewardLoggingCallback(), LeagueAdaptCallback(stage_val), entropy_callback],
            reset_num_timesteps=(load_path is None),
            progress_bar=True,
        )
        model.save(f"{save_path}/league_model")
        env.save(vec_normalize_path)
        print(f"Model saved to {save_path}/league_model")
        print(f"VecNormalize stats saved to {vec_normalize_path}")
        print(f"TensorBoard logs at {log_dir} — run: tensorboard --logdir {log_dir}")
        env.close()
        eval_env.close()


# ================================================================
# PLAYER CLASS
# ================================================================

_ACTIONS_ARRAY = get_action_array(PLAYER_COLORS, MAP_TYPE)
_ACTION_SPACE_SIZE = len(_ACTIONS_ARRAY)


class PPOBoardPlayer(Player):
    """PPO bot trained on the board-state observation. CLI code: PPOBOARD."""

    def __init__(self, color, model_path=MODEL_PATH):
        super().__init__(color, is_bot=True)
        self.model = MaskablePPO.load(model_path)

    def __reduce__(self):
        # Replay server may unpickle saved games without this class importable.
        return (Player, (self.color, True))

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        state = game.state
        opp_color = next(c for c in state.colors if c != self.color)
        obs = encode_board_state(state.board, state)

        # MOVE_ROBBER actions embed victim colors; remap self/opp to BLUE/RED
        # so indices match the training action space.
        mask = np.zeros(_ACTION_SPACE_SIZE, dtype=bool)
        idx_to_action = {}
        for action in playable_actions:
            value = action.value
            if action.action_type == ActionType.MOVE_ROBBER and value is not None:
                coords, victim = value
                if victim == self.color:
                    victim = AGENT_COLOR
                elif victim == opp_color:
                    victim = OPPONENT_COLOR
                value = (coords, victim)
            try:
                gym_idx = _ACTIONS_ARRAY.index((action.action_type, value))
                mask[gym_idx] = True
                idx_to_action[gym_idx] = action
            except ValueError:
                pass

        action_idx, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        return idx_to_action.get(int(action_idx), playable_actions[0])


register_cli_player("PPOBOARD", PPOBoardPlayer)


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--continue-training",
        metavar="MODEL_PATH",
        default=None,
        help="Path to a trained model to fine-tune against AlphaBetaPlayer",
    )
    parser.add_argument(
        "--hard",
        action="store_true",
        help="Train from scratch against AlphaBetaPlayer instead of WeightedRandomPlayer",
    )
    parser.add_argument(
        "--medium",
        action="store_true",
        help="Train from scratch against epsilon-greedy ValueFunctionPlayer",
    )
    parser.add_argument(
        "--medium-continue",
        metavar="MODEL_PATH",
        default=None,
        help="Fine-tune existing checkpoint vs epsilon-greedy ValueFunctionPlayer",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.4,
        help="Epsilon for the epsilon-greedy medium opponent (default: 0.4)",
    )
    parser.add_argument(
        "--league",
        action="store_true",
        help="Train from scratch with opponent sampling / league training",
    )
    parser.add_argument(
        "--league-continue",
        metavar="MODEL_PATH",
        default=None,
        help="Fine-tune existing checkpoint with league training",
    )
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument(
        "--start-stage",
        type=int,
        default=0,
        help="League stage to resume from (0=easy, 1=medium, 2=hard)",
    )
    parser.add_argument("--save-path", default=MODEL_DIR)
    parser.add_argument("--log-dir", default="./ppo_board_logs")
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: 'auto', 'cuda', 'cpu'",
    )
    args = parser.parse_args()

    if args.medium_continue:
        train_vs_medium(
            epsilon=args.epsilon,
            load_path=args.medium_continue,
            save_path=args.save_path,
            log_dir=args.log_dir,
            total_timesteps=args.timesteps or 5_000_000,
            device=args.device,
        )
    elif args.medium:
        train_vs_medium(
            epsilon=args.epsilon,
            save_path=args.save_path,
            log_dir=args.log_dir,
            total_timesteps=args.timesteps or 5_000_000,
            device=args.device,
        )
    elif args.league_continue:
        league_train(
            load_path=args.league_continue,
            save_path=args.save_path,
            log_dir=args.log_dir,
            total_timesteps=args.timesteps or 5_000_000,
            start_stage=args.start_stage,
            device=args.device,
        )
    elif args.league:
        league_train(
            save_path=args.save_path,
            log_dir=args.log_dir,
            total_timesteps=args.timesteps or 5_000_000,
            device=args.device,
        )
    elif args.continue_training:
        continue_training(
            load_path=args.continue_training,
            save_path=args.save_path,
            log_dir=args.log_dir,
            total_timesteps=args.timesteps or 500_000,
            device=args.device,
        )
    elif args.hard:
        train(
            total_timesteps=args.timesteps or 500_000,
            save_path=args.save_path,
            log_dir=args.log_dir,
            env_fn=make_env_hard,
            device=args.device,
        )
    else:
        train(
            total_timesteps=args.timesteps or 500_000,
            save_path=args.save_path,
            log_dir=args.log_dir,
            device=args.device,
        )
