import json
import os
import random
import multiprocessing as mp

import numpy as np
import gymnasium
from gymnasium import Wrapper
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
from catanatron.state_functions import player_key, player_num_resource_cards
from catanatron.models.enums import ActionType
from catanatron.gym.envs.action_space import get_action_array
from catanatron.cli import register_cli_player
import catanatron.gym

import torch
torch.distributions.Distribution.set_default_validate_args(False)


MODEL_DIR = os.path.join(os.path.dirname(__file__), "ppo_catan_model")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_ppo", "best_model")

RESOURCE_TYPES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]


# ================================================================
# STATE HELPER FUNCTIONS
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
# REWARD CHANNEL FUNCTIONS
# ================================================================

def resource_flow_score(state, color):
    """
    Resource Flow Channel (potential score, delta used as reward):
    - +0.2 per distinct resource type in hand
    - +0.1 per total resource card in hand
    - -0.2 + -0.1 * (hand_size - 9) if hand_size > 9

    Returns a score; the wrapper computes the step-to-step delta so the
    reward fires only when hand composition actually changes.
    """
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
    - +0.1 per production pip covered by settlements (doubled for cities)
      (zeroed if robber is on that tile)
    - +0.6 per unique resource type accessible from settlements/cities
    - +0.8 for having >= 1 generic port
    - Per specific-resource port connected: +0.1 baseline + 0.06 per
      effective pip of that resource owned (so a port with no matching
      production is nearly worthless, while one fed by 5+ pips matches or
      exceeds the old flat +0.4)
    - +0.1 per total road + 0.1 per road in longest connected path
    - +0.06 * (total pips of adjacent tiles) per open settlement spot
      reachable via road network
    """
    board = state.board
    score = 0.0

    # Pip counts for dice values (2-12, index by number)
    pip_counts = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 0, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

    # LandTile has no coordinate attribute; build id->coordinate lookup from map
    tile_coord = {tile.id: coord for coord, tile in board.map.land_tiles.items()}
    robber_coordinate = board.robber_coordinate

    resource_types_accessible = set()
    # Effective pips per resource (post-robber, city-doubled). Used by the
    # port block so a specific-resource port is valued in proportion to how
    # much of that resource you actually produce.
    pips_by_resource = {}

    # board.buildings is Dict[NodeId, Tuple[Color, FastBuildingType]]
    buildings = board.buildings
    for node_id, (bldg_color, bldg_type) in buildings.items():
        if bldg_color != color:
            continue

        multiplier = 2 if bldg_type == "CITY" else 1

        # board.map.adjacent_tiles is Dict[NodeId, List[LandTile]]
        for tile in board.map.adjacent_tiles.get(node_id, []):
            if tile.resource is None:  # desert
                continue

            resource_types_accessible.add(tile.resource)

            # Zero pips if robber is on this tile
            if tile_coord.get(tile.id) == robber_coordinate:
                continue

            pips = pip_counts.get(tile.number, 0) if tile.number else 0
            score += pips * 0.1 * multiplier
            pips_by_resource[tile.resource] = (
                pips_by_resource.get(tile.resource, 0) + pips * multiplier
            )

    # Resource diversity
    score += len(resource_types_accessible) * 0.6

    # Port access — board.map.port_nodes is Dict[FastResource|None, Set[NodeId]]
    has_generic_port = False
    for resource, node_ids in board.map.port_nodes.items():
        if not any(nid in buildings and buildings[nid][0] == color for nid in node_ids):
            continue
        if resource is None:  # generic 3:1 port
            has_generic_port = True
        else:
            matching_pips = pips_by_resource.get(resource, 0)
            score += 0.1 + 0.06 * matching_pips

    if has_generic_port:
        score += 0.8

    # Road network
    roads = [(edge, c) for edge, c in board.roads.items() if c == color]
    num_roads = len(roads) // 2  # roads stored bidirectionally
    longest_road = get_longest_road_length(state, color)
    score += num_roads * 0.1 + longest_road * 0.1

    # Open settlement spots weighted by pip quality
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
    """
    VP Proximity Channel (event-based):
    - +1 per VP gained, -1 per VP lost
    - +0.05 * (my_vps - opp_vps) to reward pulling ahead / penalize falling behind
    - Scaled knight reward based on army achievability
    """
    current_vps = get_victory_points(state, color)
    vp_delta = current_vps - prev_vps

    opp_vps = get_victory_points(state, opp_color)
    relative_vps = (current_vps - opp_vps)
    relative_reward = 0.05 * relative_vps

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

    return float(vp_delta) + relative_reward + knight_reward


FEATURE_DIM = 25


def compute_features(state, color, opp_color):
    """25-dim hand-crafted feature vector using the same information
    the heuristic bots use to make decisions."""
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
        # Resource state (8 features)
        hand_size / 10.0,
        sum(1 for v in hand.values() if v > 0) / 5.0,
        hand.get("WOOD", 0) / 5.0,
        hand.get("BRICK", 0) / 5.0,
        hand.get("SHEEP", 0) / 5.0,
        hand.get("WHEAT", 0) / 5.0,
        hand.get("ORE", 0) / 5.0,
        sum(opp_hand.values()) / 10.0,

        # VP state (5 features)
        my_vps / 15.0,
        opp_vps / 15.0,
        (my_vps - opp_vps) / 15.0,
        my_vps / 15.0,                # progress toward win
        max(0, 15 - my_vps) / 15.0,  # distance to win

        # Army race (4 features)
        my_knights / 5.0,
        opp_knights / 5.0,
        float(my_knights >= 3 and my_knights > opp_knights),
        float(opp_knights >= 3 and opp_knights > my_knights),

        # Road race (4 features)
        my_road / 10.0,
        opp_road / 10.0,
        float(my_road >= 5 and my_road > opp_road),
        float(opp_road >= 5 and opp_road > my_road),

        # Board position (3 features)
        pos_score / 10.0,
        opp_pos_score / 10.0,
        (pos_score - opp_pos_score) / 10.0,

        # Resource flow score (1 feature)
        resource_flow_score(state, color) / 2.0,
    ], dtype=np.float32)


def terminal_reward(game, color):
    """
    Terminal Reward (fired once on game end):
    - Win: +15 + 0.5 * VP margin
    - Loss: -15 + 0.5 * VP margin
    Base scaled to ±15 to match vps_to_win=15 and stay dominant over shaping.
    """
    winning_color = game.winning_color()
    if winning_color is None:
        return 0.0

    return (15.0 if winning_color == color else -15.0)


# ================================================================
# REWARD WRAPPER
# ================================================================

class CatanRewardWrapper(Wrapper):
    """
    Wraps Catanatron-v0 to provide the composite reward (resource flow +
    network position delta + VP proximity + terminal) used by the PPO baseline.
    Replaces the reward_function config param, which can't track state across steps.

    On episode end, injects per-channel episode totals and win flag into info:
        info["ep_r_resource"], info["ep_r_position"], info["ep_r_vp"],
        info["ep_r_terminal"], info["win"]
    These are picked up by RewardLoggingCallback for TensorBoard.
    """

    def __init__(self, env):
        super().__init__(env)
        self.p0_color = Color.BLUE
        self.opp_color = Color.RED
        self._prev_resource_score = 0.0
        self._prev_position_score = 0.0
        self._prev_vps = 0
        self._prev_knights = 0
        self._ep_r_resource = 0.0
        self._ep_r_position = 0.0
        self._ep_r_vp = 0.0
        self._ep_r_terminal = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        state = self.env.unwrapped.game.state
        self._prev_resource_score = resource_flow_score(state, self.p0_color)
        self._prev_position_score = network_position_score(state, self.p0_color)
        self._prev_vps = get_victory_points(state, self.p0_color)
        self._prev_knights = get_knights_played(state, self.p0_color)
        self._ep_r_resource = 0.0
        self._ep_r_position = 0.0
        self._ep_r_vp = 0.0
        self._ep_r_terminal = 0.0
        return obs, info

    def _safe_position_score(self, state):
        try:
            return network_position_score(state, self.p0_color)
        except Exception:
            return 0.0

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        game = self.env.unwrapped.game
        state = game.state
        done = terminated or truncated

        # Resource flow: delta-based so it only fires when hand composition changes.
        # Typical delta: ±0.1–0.5 per step.
        current_resource = resource_flow_score(state, self.p0_color)
        r_resource = current_resource - self._prev_resource_score
        self._prev_resource_score = current_resource

        # Network position: delta-based; scaled by 0.3 so build events (+3–8 raw)
        # land at ~+1–2, comparable to a VP gain.
        current_position = network_position_score(state, self.p0_color)
        r_position = 0.3 * (current_position - self._prev_position_score)
        self._prev_position_score = current_position

        r_vp = vp_proximity_reward(
            state, self.p0_color,
            self._prev_vps, self._prev_knights,
            self.opp_color,
        )
        self._prev_vps = get_victory_points(state, self.p0_color)
        self._prev_knights = get_knights_played(state, self.p0_color)

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
# FEATURE OBSERVATION WRAPPER
# ================================================================

class CatanFeatureWrapper(gymnasium.ObservationWrapper):
    """
    Replaces the flat Catanatron observation (300+ raw indices) with a
    25-dim hand-crafted feature vector computed from the same scoring
    functions used in rewards. The network no longer needs to rediscover
    board structure from raw tile indices.
    """

    def __init__(self, env):
        super().__init__(env)
        self.p0_color = Color.BLUE
        self.opp_color = Color.RED
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(FEATURE_DIM,), dtype=np.float32
        )

    def observation(self, obs):
        state = self.env.unwrapped.game.state
        return compute_features(state, self.p0_color, self.opp_color)


# ================================================================
# EPSILON-GREEDY OPPONENT WRAPPER
# ================================================================

class EpsilonGreedyPlayer(Player):
    """
    Wraps any Player: with probability epsilon picks a uniformly random
    action, otherwise delegates to the wrapped player's decide().
    Makes a strong deterministic opponent beatable so the agent can
    receive positive terminal rewards and learn from wins.
    """

    def __init__(self, player, epsilon):
        super().__init__(player.color, is_bot=True)
        self._player = player
        self.epsilon = epsilon

    def decide(self, game, playable_actions):
        if random.random() < self.epsilon:
            return random.choice(playable_actions)
        return self._player.decide(game, playable_actions)

    def reset_state(self):
        self._player.reset_state()


# ================================================================
# ACTION MASK FUNCTION
# ================================================================

def mask_fn(env) -> np.ndarray:
    valid_actions = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=bool)
    if valid_actions:
        mask[valid_actions] = True
    else:
        mask[0] = True  # safety: never return all-False (causes NaN softmax)
    return mask


# ================================================================
# LOGGING CALLBACK
# ================================================================

class AnnealingCallback(BaseCallback):
    """
    Linearly anneals learning rate and ent_coef over training.

    Progress is measured from the start of THIS run (step_offset is captured
    in _on_training_start), so --league-continue resumes from the saved values
    rather than resetting to the initial ones.

    On training end, saves {lr, ent_coef} to state_file so the next
    --league-continue call can read them as starting values.
    """

    def __init__(self, start_lr, end_lr, start_ent, end_ent, total_new_steps, state_file, verbose=0):
        super().__init__(verbose)
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.start_ent = start_ent
        self.end_ent = end_ent
        self.total_new_steps = total_new_steps
        self.state_file = state_file
        self._step_offset = 0

    def _on_training_start(self):
        # num_timesteps == loaded checkpoint steps (0 for fresh runs)
        self._step_offset = self.num_timesteps
        self._apply(0.0)

    def _apply(self, t: float):
        lr = self.start_lr + (self.end_lr - self.start_lr) * t
        ent = self.start_ent + (self.end_ent - self.start_ent) * t
        for param_group in self.model.policy.optimizer.param_groups:
            param_group["lr"] = lr
        self.model.ent_coef = ent
        self.logger.record("train/lr", lr)
        self.logger.record("train/ent_coef", ent)

    def _on_step(self) -> bool:
        steps_into_run = self.num_timesteps - self._step_offset
        t = min(1.0, steps_into_run / max(1, self.total_new_steps))
        self._apply(t)
        return True

    def _on_training_end(self):
        steps_into_run = self.num_timesteps - self._step_offset
        t = min(1.0, steps_into_run / max(1, self.total_new_steps))
        lr = self.start_lr + (self.end_lr - self.start_lr) * t
        ent = self.start_ent + (self.end_ent - self.start_ent) * t
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump({"lr": lr, "ent_coef": ent}, f)
        print(f"[AnnealingCallback] Saved schedule state: lr={lr:.2e}, ent_coef={ent:.4f}")


class RewardLoggingCallback(BaseCallback):
    """
    Reads per-channel episode stats injected by CatanRewardWrapper and logs
    them to TensorBoard under the 'reward/' prefix, plus a rolling win rate.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._wins = []
        self._ep_r_resource = []
        self._ep_r_position = []
        self._ep_r_vp = []
        self._ep_r_terminal = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "win" not in info:
                continue
            self._wins.append(info["win"])
            self._ep_r_resource.append(info["ep_r_resource"])
            self._ep_r_position.append(info["ep_r_position"])
            self._ep_r_vp.append(info["ep_r_vp"])
            self._ep_r_terminal.append(info["ep_r_terminal"])

        if len(self._wins) >= 10:
            self.logger.record("reward/win_rate", np.mean(self._wins[-100:]))
            self.logger.record("reward/ep_resource", np.mean(self._ep_r_resource[-100:]))
            self.logger.record("reward/ep_position", np.mean(self._ep_r_position[-100:]))
            self.logger.record("reward/ep_vp", np.mean(self._ep_r_vp[-100:]))
            self.logger.record("reward/ep_terminal", np.mean(self._ep_r_terminal[-100:]))

        return True


# ================================================================
# TRAINING
# ================================================================

def make_env():
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "enemies": [WeightedRandomPlayer(Color.RED)],
            "vps_to_win": 15,
            "discard_limit": 9,
        },
    )
    env = CatanRewardWrapper(env)
    env = CatanFeatureWrapper(env)
    env = ActionMasker(env, mask_fn)
    return env


def make_env_hard():
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "enemies": [AlphaBetaPlayer(Color.RED, depth=1)],
            "vps_to_win": 15,
            "discard_limit": 9,
        },
    )
    env = CatanRewardWrapper(env)
    env = CatanFeatureWrapper(env)
    env = ActionMasker(env, mask_fn)
    return env


def make_env_medium(epsilon=0.4):
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "enemies": [EpsilonGreedyPlayer(ValueFunctionPlayer(Color.RED), epsilon=epsilon)],
            "vps_to_win": 15,
            "discard_limit": 9,
        },
    )
    env = CatanRewardWrapper(env)
    env = CatanFeatureWrapper(env)
    env = ActionMasker(env, mask_fn)
    return env


class _MediumEnvFn:
    """Picklable env factory for SubprocVecEnv (lambdas are not reliably picklable on Windows)."""

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self):
        return make_env_medium(self.epsilon)


def train_vs_medium(
    epsilon=0.4,
    total_timesteps=5_000_000,
    save_path=MODEL_PATH,
    log_dir="./ppo_catan_logs",
    eval_freq=20_000,
    n_eval_episodes=20,
    n_envs=4,
    load_path=None,
):
    """
    Train (or fine-tune from load_path) against an epsilon-greedy
    ValueFunctionPlayer. epsilon=0.4 means 40% random moves, making
    the opponent beatable so the agent receives positive terminal rewards
    and can learn from wins. Tune epsilon down as the agent improves.
    """
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
        model = MaskablePPO.load(load_path, env=env)
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
            device="cpu",
            policy_kwargs={"net_arch": dict(pi=[256, 256, 256], vf=[256, 256, 256])},
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


def train(
    total_timesteps=1_000_000,
    save_path=MODEL_PATH,
    log_dir="./ppo_catan_logs",
    eval_freq=10_000,
    n_eval_episodes=20,
    n_envs=4,
    env_fn=make_env,
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
        device="cpu",
        policy_kwargs={"net_arch": dict(pi=[256, 256], vf=[256, 256])},
    )

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}/best",
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )
    reward_callback = RewardLoggingCallback()

    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, reward_callback], progress_bar=True)
    model.save(f"{save_path}/final_model")
    print(f"Model saved to {save_path}/final_model")
    print(f"TensorBoard logs at {log_dir} — run: tensorboard --logdir {log_dir}")
    env.close()
    eval_env.close()
    return model


def continue_training(
    load_path=f"{MODEL_PATH}/final_model",
    save_path=MODEL_PATH,
    log_dir="./ppo_catan_logs",
    total_timesteps=500_000,
    eval_freq=10_000,
    n_eval_episodes=20,
    n_envs=4,
):
    env = SubprocVecEnv([make_env_hard] * n_envs)
    eval_env = make_env_hard()

    model = MaskablePPO.load(load_path, env=env)

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}/best_hard",
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )
    reward_callback = RewardLoggingCallback()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, reward_callback],
        reset_num_timesteps=False,
        progress_bar=True,
    )
    model.save(f"{save_path}/final_model_hard")
    print(f"Model saved to {save_path}/final_model_hard")
    env.close()
    eval_env.close()
    return model


# ================================================================
# LEAGUE / OPPONENT-SAMPLING TRAINING
# ================================================================

# Opponent tiers ordered Easy → Hard based on the catanatron leaderboard:
#   WeightedRandom < MCTS(n=100) < GreedyPlayouts(n=25) < AlphaBeta
# We use MCTSPlayer(n=25) as Medium — fast enough to use as a live training
# opponent (GreedyPlayoutsPlayer runs ~185 s per initial-placement decision).
# ValueFunctionPlayer is a fast greedy one-step lookahead that sits between
# WeightedRandom and AlphaBeta on the leaderboard.
OPPONENT_TIERS = [
    ("easy",   lambda: WeightedRandomPlayer(Color.RED)),
    ("medium",   lambda: AlphaBetaPlayer(Color.RED, depth=1)),
    ("hard", lambda: ValueFunctionPlayer(Color.RED)),
]

# Sampling weights [easy, medium, hard] per stage
STAGE_WEIGHTS = [
    [0.80, 0.20, 0.00],   # Stage 0: build basics vs WeightedRandom
    [0.30, 0.50, 0.20],   # Stage 1: focus on MCTS, keep basics
    [0.20, 0.40, 0.40],   # Stage 2: sharpen vs AlphaBeta
]

# Win-rate threshold (over a rolling window) to advance from each stage.
# Index matches the stage number; we check win rate against the *primary*
# tier for that stage (tier 0 for stage 0, tier 1 for stage 1).
STAGE_UP_THRESHOLDS = [0.80, 0.50]
MIN_TIER_EPISODES = 50   # minimum same-tier episodes before checking


class LeagueWrapper(Wrapper):
    """
    Swaps the opponent inside CatanatronEnv at each episode reset based on
    the current league stage stored in a shared multiprocessing Value.
    Injects ``tier_idx`` into the info dict at episode end so
    LeagueAdaptCallback can track per-tier win rates.

    Expected wrapper stack (inner → outer):
        CatanatronEnv → CatanRewardWrapper → LeagueWrapper → ActionMasker
    """

    def __init__(self, env, stage_val):
        super().__init__(env)
        self._stage_val = stage_val
        self._tier_idx = 0

    def reset(self, **kwargs):
        stage = min(self._stage_val.value, len(STAGE_WEIGHTS) - 1)
        weights = STAGE_WEIGHTS[stage]
        self._tier_idx = int(np.random.choice(len(weights), p=weights))
        new_opp = OPPONENT_TIERS[self._tier_idx][1]()   # call factory

        # CatanatronEnv.reset() re-creates the Game from self.players, so
        # updating both list slots is enough — no env recreation needed.
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


class LeagueAdaptCallback(BaseCallback):
    """
    Reads tier_idx / win from episode-end info injected by LeagueWrapper /
    CatanRewardWrapper and advances the shared stage when thresholds are met:

      Stage 0 → 1 : easy   win rate ≥ STAGE_UP_THRESHOLDS[0] over last 100 ep
      Stage 1 → 2 : medium win rate ≥ STAGE_UP_THRESHOLDS[1] over last 100 ep

    Logs ``league/stage`` and ``league/win_rate_{tier}`` to TensorBoard.
    """

    def __init__(self, stage_val, window=100, verbose=0):
        super().__init__(verbose)
        self._stage_val = stage_val
        self._window = window
        self._tier_wins = [[] for _ in range(len(OPPONENT_TIERS))]

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "tier_idx" not in info or "win" not in info:
                continue
            self._tier_wins[info["tier_idx"]].append(info["win"])

        stage = self._stage_val.value
        if stage < len(STAGE_UP_THRESHOLDS):
            wins = self._tier_wins[stage]   # check tier matching current stage
            if len(wins) >= MIN_TIER_EPISODES:
                rate = float(np.mean(wins[-self._window:]))
                threshold = STAGE_UP_THRESHOLDS[stage]
                if rate >= threshold:
                    self._stage_val.value = stage + 1
                    tier_name = OPPONENT_TIERS[stage][0]
                    print(
                        f"\n[League] Stage {stage} → {stage + 1}  "
                        f"({tier_name} win rate: {rate:.1%} ≥ {threshold:.0%})"
                    )

        self.logger.record("league/stage", float(self._stage_val.value))
        for i, (name, _) in enumerate(OPPONENT_TIERS):
            if len(self._tier_wins[i]) >= 10:
                self.logger.record(
                    f"league/win_rate_{name}",
                    float(np.mean(self._tier_wins[i][-self._window:])),
                )
        return True


class _LeagueEnvFn:
    """
    Picklable env factory that captures a Manager-proxy stage value so
    SubprocVecEnv worker processes (spawned on Windows) can share it.
    """

    def __init__(self, stage_val):
        self._stage_val = stage_val

    def __call__(self):
        return make_league_env(self._stage_val)


def make_league_env(stage_val):
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={"enemies": [WeightedRandomPlayer(Color.RED)], "vps_to_win": 15, "discard_limit": 9},
    )
    env = CatanRewardWrapper(env)
    env = LeagueWrapper(env, stage_val)
    env = CatanFeatureWrapper(env)
    env = ActionMasker(env, mask_fn)
    return env


def league_train(
    total_timesteps=2_000_000,
    save_path=MODEL_PATH,
    log_dir="./ppo_catan_logs",
    eval_freq=20_000,
    n_eval_episodes=20,
    n_envs=4,
    load_path=None,
    start_stage=0,
):
    """
    Train (or fine-tune) with opponent sampling / league training.

    Starts all envs against WeightedRandom and automatically shifts the
    opponent distribution toward harder bots as win-rate thresholds are met.
    The eval callback uses a fixed WeightedRandom opponent so the mean-reward
    curve stays comparable across stages.

    Args:
        load_path: if provided, load an existing checkpoint and continue
                   (reset_num_timesteps=False so TensorBoard x-axis is
                   continuous and schedules, if any, behave correctly).
    """
    # Manager creates a server process that owns the value; the returned
    # proxy is picklable so SubprocVecEnv workers can share it via spawn.
    with mp.Manager() as manager:
        stage_val = manager.Value("i", start_stage)
        env_fns: list = [_LeagueEnvFn(stage_val) for _ in range(n_envs)]
        vec_normalize_path = os.path.join(save_path, "vecnormalize.pkl")
        if load_path and os.path.exists(vec_normalize_path):
            env = VecNormalize.load(vec_normalize_path, SubprocVecEnv(env_fns))
            env.training = True
            env.norm_reward = True
        else:
            env = VecNormalize(SubprocVecEnv(env_fns), norm_obs=False, norm_reward=True, clip_reward=10.0)
        eval_env = VecNormalize(DummyVecEnv([make_env]), norm_obs=False, norm_reward=False, training=False)

        # Load saved schedule values so --league-continue resumes from where
        # the previous run ended instead of jumping back to initial values.
        start_lr, start_ent = 3e-4, 0.08
        if load_path and os.path.exists(LEAGUE_SCHEDULE_STATE):
            with open(LEAGUE_SCHEDULE_STATE) as f:
                state = json.load(f)
            start_lr = state["lr"]
            start_ent = state["ent_coef"]
            print(f"[League] Resuming schedules: lr={start_lr:.2e}, ent_coef={start_ent:.4f}")

        if load_path:
            model = MaskablePPO.load(load_path, env=env)
            learn_timesteps = max(1, total_timesteps - model.num_timesteps)
        else:
            model = MaskablePPO(
                MaskableActorCriticPolicy,
                env,
                verbose=1,
                tensorboard_log=log_dir,
                learning_rate=start_lr,
                n_steps=8192,
                batch_size=1024,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=start_ent,
                vf_coef=0.5,
                max_grad_norm=0.5,
                device="cpu",
                policy_kwargs={"net_arch": dict(pi=[512, 256], vf=[512, 256])},
            )

        annealing_cb = AnnealingCallback(
            start_lr=start_lr,
            end_lr=1e-5,
            start_ent=start_ent,
            end_ent=0.01,
            total_new_steps=total_timesteps,
            state_file=LEAGUE_SCHEDULE_STATE,
        )

        eval_callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=f"{save_path}/best_league2",
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
            total_timesteps=total_timesteps,
            callback=[eval_callback, RewardLoggingCallback(), LeagueAdaptCallback(stage_val), annealing_cb],
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

# Action array the model was trained with (BLUE=agent, RED=opponent).
# Cached at module level so PPOPlayer instances don't rebuild it each call.
_TRAINING_COLORS = (Color.BLUE, Color.RED)
_ACTIONS_ARRAY = get_action_array(_TRAINING_COLORS, "BASE")
_ACTION_SPACE_SIZE = len(_ACTIONS_ARRAY)


class PPOPlayer(Player):
    """PPO-trained bot. Use code PPO in the CLI, e.g. --players=PPO,H."""

    def __init__(self, color, model_path=MODEL_PATH):
        super().__init__(color, is_bot=True)
        self.model = MaskablePPO.load(model_path)

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        state = game.state
        opp_color = next(c for c in state.colors if c != self.color)
        obs = compute_features(state, self.color, opp_color)

        # Map each playable action to its index in the training action array.
        # MOVE_ROBBER actions embed victim colors; remap self.color→BLUE and
        # opp_color→RED so indices match what the model was trained against.
        mask = np.zeros(_ACTION_SPACE_SIZE, dtype=bool)
        idx_to_action = {}
        for action in playable_actions:
            value = action.value
            if action.action_type == ActionType.MOVE_ROBBER and value is not None:
                coords, victim = value
                if victim == self.color:
                    victim = Color.BLUE
                elif victim == opp_color:
                    victim = Color.RED
                value = (coords, victim)
            try:
                gym_idx = _ACTIONS_ARRAY.index((action.action_type, value))
                mask[gym_idx] = True
                idx_to_action[gym_idx] = action
            except ValueError:
                pass  # action not in training space; will never be selected

        action_idx, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        return idx_to_action.get(int(action_idx), playable_actions[0])


register_cli_player("PPO", PPOPlayer)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--continue-training",
        metavar="MODEL_PATH",
        default=None,
        help="Path to a trained model to fine-tune against AlphaBetaPlayer (e.g. ppo_catan_model/final_model)",
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
        help="Fine-tune an existing checkpoint against epsilon-greedy ValueFunctionPlayer",
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
        help="Fine-tune an existing checkpoint with league training (e.g. ppo_catan_model/league_model)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total_timesteps for any training mode",
    )
    parser.add_argument(
        "--start-stage",
        type=int,
        default=0,
        help="League stage to resume from when using --league-continue (0=easy, 1=medium, 2=hard)",
    )
    parser.add_argument(
        "--save-path",
        default=MODEL_DIR,
        help=f"Directory to save the model (default: {MODEL_DIR})",
    )
    parser.add_argument(
        "--log-dir",
        default="./ppo_catan_logs",
        help="TensorBoard log directory (default: ./ppo_catan_logs)",
    )
    args = parser.parse_args()

    if args.medium_continue:
        train_vs_medium(
            epsilon=args.epsilon,
            load_path=args.medium_continue,
            save_path=args.save_path,
            log_dir=args.log_dir,
            total_timesteps=args.timesteps or 5_000_000,
        )
    elif args.medium:
        train_vs_medium(
            epsilon=args.epsilon,
            save_path=args.save_path,
            log_dir=args.log_dir,
            total_timesteps=args.timesteps or 15_000_000,
        )
    elif args.league_continue:
        league_train(
            load_path=args.league_continue,
            save_path=args.save_path,
            log_dir=args.log_dir,
            total_timesteps=args.timesteps or 5_000_000,
            start_stage=args.start_stage,
        )
    elif args.league:
        league_train(
            save_path=args.save_path,
            log_dir=args.log_dir,
            total_timesteps=args.timesteps or 5_000_000,
        )
    elif args.continue_training:
        continue_training(
            load_path=args.continue_training,
            save_path=args.save_path,
            log_dir=args.log_dir,
            total_timesteps=args.timesteps or 500_000,
        )
    elif args.hard:
        train(
            total_timesteps=args.timesteps or 500_000,
            save_path=args.save_path,
            log_dir=args.log_dir,
            env_fn=make_env_hard,
        )
    else:
        train(
            total_timesteps=args.timesteps or 500_000,
            save_path=args.save_path,
            log_dir=args.log_dir,
        )
