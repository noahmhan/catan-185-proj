"""
Initial-Placement-Only PPO trainer.

Trains an MLP (via MaskablePPO) whose ONLY job is to make BLUE's 2 initial
settlement/road pairs. The opponent (RED) plays the other two picks via a
fixed opponent bot. Draft seating is randomized per episode: with prob. 0.5
BLUE is seated first (picks 1+4), otherwise BLUE is seated second (picks
2+3). The base CatanatronEnv always puts p0=BLUE at index 0 in
self.players; we override that list at reset() time to shuffle seating.

Episode terminates as soon as is_initial_build_phase flips to False — the
rest of the game is never played. Reward is the per-step delta of the
`network_position_score` function imported from `185_ppo.py`, plus a
terminal bonus equal to (agent network_position_score − opponent
network_position_score) so the agent optimizes for a positional lead over
the opponent at the end of placement.

Designed to run detached on a single A10 GPU:

    nohup python -u initial_placement_ppo.py \\
        --total-timesteps 3000000 \\
        --n-envs 8 \\
        --save-path ./initial_placement_model \\
        --log-dir ./initial_placement_logs \\
        > init_place_train.log 2>&1 &

"""

import os
import importlib.util
import random

import numpy as np
import gymnasium
from gymnasium import Wrapper, ObservationWrapper
from gymnasium import spaces

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from catanatron import Color, Player
from catanatron.models.map import build_map, NUM_NODES, NUM_EDGES
from catanatron.models.board import get_edges
from catanatron.models.enums import RESOURCES, ActionType, ActionPrompt
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.gym.envs.action_space import get_action_array, from_action_space
import catanatron.gym  # noqa: F401 — registers the gym env

import torch
torch.distributions.Distribution.set_default_validate_args(False)


# ================================================================
# IMPORT network_position_score FROM 185_ppo.py
# Module name starts with a digit and cannot be imported normally.
# ================================================================

_PPO_185_PATH = os.path.join(os.path.dirname(__file__), "185_ppo.py")
_spec = importlib.util.spec_from_file_location("ppo_185", _PPO_185_PATH)
_ppo_185 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ppo_185)
network_position_score = _ppo_185.network_position_score


# ================================================================
# CONSTANTS / MAP LAYOUT
# ================================================================

MODEL_DIR = os.path.join(os.path.dirname(__file__), "initial_placement_model")
MAP_TYPE = "BASE"
AGENT_COLOR = Color.BLUE
OPPONENT_COLOR = Color.RED
PLAYER_COLORS = (AGENT_COLOR, OPPONENT_COLOR)

# Resource one-hot ordering used across observation encoding.
RESOURCE_TO_IDX = {r: i for i, r in enumerate(RESOURCES)}   # 5 resources
PORT_RESOURCE_DIM = len(RESOURCES) + 1                       # +1 for GENERIC (3:1)
PIP_RESOURCE_DIM = len(RESOURCES)                            # pip-prob totals per resource

# Single-tile pip probability is at most 5/36 (numbers 6, 8). Scale so one
# max-pip tile contributes 1.0 to the per-resource sum; a node touching three
# 6/8 tiles of the same resource would reach 3.0.
PIP_SCALE = 5.0 / 36.0

# Build static references so we can encode observations with a fixed ordering.
# Topology (which node_id is adjacent to which tile, which edges connect which
# nodes, and where port *locations* sit on the coast) is constant across games
# — only the resources/numbers at each position are shuffled per episode.
_REF_MAP = build_map(MAP_TYPE)
_NODE_IDS = sorted(_REF_MAP.land_nodes)                                 # 54 ids
_EDGES = sorted(tuple(sorted(e)) for e in get_edges(_REF_MAP.land_nodes))  # 72 edges
_NODE_IDX = {nid: i for i, nid in enumerate(_NODE_IDS)}
_EDGE_IDX = {e: i for i, e in enumerate(_EDGES)}

# Per-node features:
#   port one-hot (6)                — which port (if any) sits at this node
#   pip-by-resource (5)             — sum of adjacent-tile pip probs, per resource
#   blue_settle, red_settle         — current settlements
#   blue_city, red_city             — unused during initial placement, kept for
#                                     compatibility if the encoding is reused
# Total: 15
NODE_FEATS_DIM = PORT_RESOURCE_DIM + PIP_RESOURCE_DIM + 4

# Per-edge: blue_road + red_road = 2
EDGE_FEATS_DIM = 2

# Phase one-hot: 0=1st settle, 1=1st road, 2=2nd settle, 3=2nd road
PHASE_DIM = 4

OBS_DIM = (
    NUM_NODES * NODE_FEATS_DIM
    + NUM_EDGES * EDGE_FEATS_DIM
    + PHASE_DIM
)


# ================================================================
# OBSERVATION WRAPPER
# ================================================================

class InitialPlacementObsWrapper(ObservationWrapper):
    """
    Replaces the default CatanatronEnv observation with a flat board-state
    vector suitable for an MLP. Encoded per-node:
        - port one-hot (dynamic — port resources reshuffle each episode)
        - pip-prob totals per resource from adjacent tiles
        - current settlements (blue/red); city slots kept unused
    Plus a per-edge road block (blue/red) and a phase one-hot (which of the
    4 Blue decisions is next).

    Deliberately OMITS hand, VPs, dev cards, bank, and the robber (which
    never moves during initial placement), since this model only makes
    the 4 initial placements.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0, high=3.0, shape=(OBS_DIM,), dtype=np.float32
        )

    def observation(self, obs):
        game = self.env.unwrapped.game
        state = game.state
        board = state.board
        return encode_board_state(board, state)


def encode_board_state(board, state) -> np.ndarray:
    """Build the flat observation vector described in OBS_DIM."""
    out = np.zeros((OBS_DIM,), dtype=np.float32)
    cursor = 0

    # ---- Per-node block ----
    node_block = out[cursor : cursor + NUM_NODES * NODE_FEATS_DIM].reshape(
        NUM_NODES, NODE_FEATS_DIM
    )
    cursor += NUM_NODES * NODE_FEATS_DIM

    port_end = PORT_RESOURCE_DIM                # 6
    pip_end = port_end + PIP_RESOURCE_DIM       # 11
    # layout: [0:6] ports, [6:11] pip-by-resource, [11:15] blue/red settle/city

    # Ports: dynamic per-episode from the live map (port resources reshuffle).
    for resource, node_ids in board.map.port_nodes.items():
        slot = len(RESOURCES) if resource is None else RESOURCE_TO_IDX[resource]
        for nid in node_ids:
            i = _NODE_IDX.get(nid)
            if i is not None:
                node_block[i, slot] = 1.0

    # Pip-prob totals per resource, from catanatron's precomputed node_production.
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

    # ---- Per-edge block ----
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

    # ---- Phase one-hot ----
    phase = placement_phase_index(state, board)
    if 0 <= phase < PHASE_DIM:
        out[cursor + phase] = 1.0
    cursor += PHASE_DIM

    return out


def placement_phase_index(state, board) -> int:
    """
    Return which Blue initial decision is next:
        0 = first settlement, 1 = first road,
        2 = second settlement, 3 = second road.
    After the initial phase is over, returns -1.
    """
    if not state.is_initial_build_phase:
        return -1

    blue_settlements = sum(
        1 for _, (c, t) in board.buildings.items()
        if c == AGENT_COLOR and t == "SETTLEMENT"
    )
    blue_roads = sum(1 for e, c in board.roads.items() if c == AGENT_COLOR) // 2

    # current_prompt indicates whether we're about to place a settlement or road
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


# ================================================================
# REWARD WRAPPER (initial-placement-only)
# ================================================================

class InitialPlacementRewardWrapper(Wrapper):
    """
    Runs ONLY the initial placement phase. Terminates the episode as soon
    as state.is_initial_build_phase becomes False (right after Blue's 2nd
    road is placed).

    Reward per step = delta in network_position_score for BLUE.
    At terminal, a bonus equal to (agent network_position_score − opponent
    network_position_score) is added so the agent is directly optimizing
    for a positional lead over the opponent at the end of placement.

    Episode-level metrics (total position score, per-channel deltas) are
    injected into info for TensorBoard logging.
    """

    def __init__(self, env, terminal_bonus_weight=1.0):
        super().__init__(env)
        self.terminal_bonus_weight = terminal_bonus_weight
        self._prev_score = 0.0
        self._ep_delta = 0.0
        self._ep_terminal = 0.0
        self._ep_final_score = 0.0
        self._ep_opponent_final_score = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        state = self.env.unwrapped.game.state
        self._prev_score = self._safe_score(state, AGENT_COLOR)
        self._ep_delta = 0.0
        self._ep_terminal = 0.0
        self._ep_final_score = 0.0
        self._ep_opponent_final_score = 0.0
        return obs, info

    def _safe_score(self, state, color):
        try:
            return float(network_position_score(state, color))
        except Exception:
            return 0.0

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        game = self.env.unwrapped.game
        state = game.state

        # Terminate when initial build phase completes.
        phase_done = not state.is_initial_build_phase
        terminated = terminated or phase_done
        done = terminated or truncated

        current_score = self._safe_score(state, AGENT_COLOR)
        r_delta = current_score - self._prev_score
        self._prev_score = current_score
        self._ep_delta += r_delta

        r_terminal = 0.0
        if done:
            opponent_score = self._safe_score(state, OPPONENT_COLOR)
            r_terminal = self.terminal_bonus_weight * (current_score - opponent_score)
            self._ep_terminal = r_terminal
            self._ep_final_score = current_score
            self._ep_opponent_final_score = opponent_score
            info["ep_delta_score"] = self._ep_delta
            info["ep_terminal_score"] = self._ep_terminal
            info["ep_final_position_score"] = self._ep_final_score
            info["ep_opponent_final_position_score"] = self._ep_opponent_final_score
            info["ep_final_position_score_diff"] = current_score - opponent_score

        reward = r_delta + r_terminal
        return obs, reward, terminated, truncated, info


# ================================================================
# RANDOM DRAFT ORDER WRAPPER
# ================================================================

class RandomPlacementOrderWrapper(Wrapper):
    """
    Randomize draft seating per episode.

    The base CatanatronEnv fixes self.players = [p0=BLUE, *enemies], which
    in a 2-player snake draft means BLUE always gets picks 1 and 4. This
    wrapper reorders self.players before each reset() so that with
    probability agent_first_prob BLUE is seated first (picks 1+4),
    otherwise BLUE is seated last (picks 2+3).

    Only player-list ordering changes; the agent's color stays BLUE, so
    AGENT_COLOR / OPPONENT_COLOR, action-space indexing (which depends on
    player_colors), and encode_board_state all remain correct.
    """

    def __init__(self, env, agent_first_prob=0.5):
        super().__init__(env)
        self.agent_first_prob = agent_first_prob

    def reset(self, **kwargs):
        base = self.env.unwrapped
        p0 = getattr(base, "p0")
        enemies = getattr(base, "enemies")
        if random.random() < self.agent_first_prob:
            setattr(base, "players", [p0, *enemies])
        else:
            setattr(base, "players", [*enemies, p0])
        return self.env.reset(**kwargs)


# ================================================================
# ACTION MASK
# ================================================================

def mask_fn(env) -> np.ndarray:
    """
    Allow only the Blue initial-phase settle/road actions that the underlying
    env reports as playable. The env's get_valid_actions already returns
    action-space indices for the current prompt, so this is just a passthrough
    into a boolean mask.
    """
    valid = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=bool)
    if valid:
        mask[valid] = True
    else:
        mask[0] = True  # never all-False (NaN softmax guard)
    return mask


# ================================================================
# ENV FACTORY
# ================================================================

def make_env():
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "enemies": [ValueFunctionPlayer(OPPONENT_COLOR)],
            "map_type": MAP_TYPE,
            "vps_to_win": 15,   # irrelevant; episode ends before play phase
            "discard_limit": 9,
        },
    )
    env = RandomPlacementOrderWrapper(env, agent_first_prob=0.5)
    env = InitialPlacementRewardWrapper(env)
    env = InitialPlacementObsWrapper(env)
    env = ActionMasker(env, mask_fn)
    return env


def _env_fn():
    """Picklable env factory for SubprocVecEnv (Windows spawn-safe)."""
    return make_env()


# ================================================================
# CALLBACK
# ================================================================

class PositionLoggingCallback(BaseCallback):
    """Logs rolling averages of per-episode network_position_score metrics."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._final_scores = []
        self._opponent_final_scores = []
        self._final_score_diffs = []
        self._delta_sums = []
        self._terminal_bonuses = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "ep_final_position_score" not in info:
                continue
            self._final_scores.append(info["ep_final_position_score"])
            self._opponent_final_scores.append(info.get("ep_opponent_final_position_score", 0.0))
            self._final_score_diffs.append(info.get("ep_final_position_score_diff", 0.0))
            self._delta_sums.append(info["ep_delta_score"])
            self._terminal_bonuses.append(info["ep_terminal_score"])

        if len(self._final_scores) >= 10:
            self.logger.record("position/final_score", float(np.mean(self._final_scores[-200:])))
            self.logger.record("position/opponent_final_score", float(np.mean(self._opponent_final_scores[-200:])))
            self.logger.record("position/final_score_diff", float(np.mean(self._final_score_diffs[-200:])))
            self.logger.record("position/ep_delta_sum", float(np.mean(self._delta_sums[-200:])))
            self.logger.record("position/ep_terminal_bonus", float(np.mean(self._terminal_bonuses[-200:])))
        return True


# ================================================================
# TRAINING
# ================================================================

def train(
    total_timesteps=3_000_000,
    save_path=MODEL_DIR,
    log_dir="./initial_placement_logs",
    eval_freq=20_000,
    n_eval_episodes=20,
    n_envs=8,
    device="auto",
    load_path=None,
):
    """
    Train MaskablePPO on the initial-placement-only env. Each episode is
    exactly 4 Blue decisions, so rollouts cover many episodes even at small
    n_steps.

    Model size rationale:
        obs dim ≈ {OBS} (nodes + edges + phase)
        action dim = 290 (BASE map action space, heavily masked down to ~50
                           valid settlement nodes + road edges at placement time)
        → MLP [512, 512, 256] is comfortably sized; A10 GPU has no trouble.
    """.format(OBS=OBS_DIM)

    os.makedirs(save_path, exist_ok=True)

    env = SubprocVecEnv([_env_fn for _ in range(n_envs)])
    eval_env = DummyVecEnv([_env_fn])

    if load_path:
        model = MaskablePPO.load(load_path, env=env, device=device)
    else:
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            n_steps=512,        # 4 steps/episode × ~128 episodes per env per rollout
            batch_size=512,
            n_epochs=10,
            gamma=0.995,        # episodes are short but we still want the terminal bonus to matter
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.03,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device=device,
            policy_kwargs={
                "net_arch": dict(pi=[512, 512, 256], vf=[512, 512, 256]),
            },
        )

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, "best"),
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, PositionLoggingCallback()],
        reset_num_timesteps=(load_path is None),
        progress_bar=True,
    )
    model.save(os.path.join(save_path, "final_model"))
    print(f"Model saved to {save_path}/final_model")
    print(f"TensorBoard logs at {log_dir} — run: tensorboard --logdir {log_dir}")
    env.close()
    eval_env.close()
    return model


# ================================================================
# PLAYER CLASS (optional: hand off to a full-game bot after placements)
# ================================================================

_ACTIONS_ARRAY = get_action_array(PLAYER_COLORS, MAP_TYPE)
_ACTION_SPACE_SIZE = len(_ACTIONS_ARRAY)


class InitialPlacementPlayer(Player):
    """
    Plays ONLY the initial placements using the trained MLP. After the
    initial phase, delegates to a fallback player (default: WeightedRandom)
    so the game can still complete if this player is used standalone.
    """

    def __init__(self, color, model_path=None, fallback=None):
        super().__init__(color, is_bot=True)
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, "final_model")
        self.model = MaskablePPO.load(model_path)
        self.fallback = fallback or WeightedRandomPlayer(color)

    def __reduce__(self):
        # Flask replay server unpickles saved games but never imports this
        # class. Reduce to a plain Player so game records load cleanly.
        return (Player, (self.color, True))

    def decide(self, game, playable_actions):
        state = game.state
        if not state.is_initial_build_phase:
            return self.fallback.decide(game, playable_actions)
        if len(playable_actions) == 1:
            return playable_actions[0]

        obs = encode_board_state(state.board, state)

        mask = np.zeros(_ACTION_SPACE_SIZE, dtype=bool)
        idx_to_action = {}
        for action in playable_actions:
            try:
                gym_idx = _ACTIONS_ARRAY.index((action.action_type, action.value))
                mask[gym_idx] = True
                idx_to_action[gym_idx] = action
            except ValueError:
                pass

        action_idx, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        return idx_to_action.get(int(action_idx), playable_actions[0])


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=3_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--save-path", default=MODEL_DIR)
    parser.add_argument("--log-dir", default="./initial_placement_logs")
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: 'auto', 'cuda', 'cpu'. A10 → 'cuda'.",
    )
    parser.add_argument(
        "--load-path",
        default=None,
        help="Optional checkpoint to continue training from",
    )
    args = parser.parse_args()

    train(
        total_timesteps=args.total_timesteps,
        save_path=args.save_path,
        log_dir=args.log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        n_envs=args.n_envs,
        device=args.device,
        load_path=args.load_path,
    )
