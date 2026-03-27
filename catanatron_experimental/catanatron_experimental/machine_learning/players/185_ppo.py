import os

import numpy as np
import gymnasium
from gymnasium import Wrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from catanatron import Color, Player
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.state_functions import player_key, player_num_resource_cards
import catanatron.gym


MODEL_DIR = os.path.join(os.path.dirname(__file__), "ppo_catan_model")
MODEL_PATH = os.path.join(MODEL_DIR, "final_model")

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
    - +0.1 * pips per settlement tile (doubled for cities), zeroed if robber present
    - +0.6 per unique resource type accessible
    - +0.8 for generic port access
    - +0.4 per specific port access
    - +0.1 per road + +0.1 per road in longest connected path
    - +0.04 * pip_sum per open buildable node reachable via road network
    """
    pip_counts = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 0, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

    board = state.board
    buildings = board.buildings
    robber_coordinate = board.robber_coordinate
    score = 0.0
    resource_types_accessible = set()

    for node_id, (bldg_color, bldg_type) in buildings.items():
        if bldg_color != color:
            continue

        multiplier = 2 if bldg_type == "CITY" else 1

        try:
            tiles = board.map.adjacent_tiles.get(node_id, [])
        except AttributeError:
            tiles = []

        for tile in tiles:
            if tile.resource is None:
                continue
            resource_types_accessible.add(tile.resource)
            if tile.coordinate == robber_coordinate:
                continue
            pips = pip_counts.get(tile.number, 0) if tile.number else 0
            score += pips * 0.1 * multiplier

    score += len(resource_types_accessible) * 0.6

    has_generic_port = False
    try:
        for resource, node_ids in board.map.port_nodes.items():
            if any(nid in buildings and buildings[nid][0] == color for nid in node_ids):
                if resource is None:
                    has_generic_port = True
                else:
                    score += 0.4
    except (AttributeError, TypeError):
        pass

    if has_generic_port:
        score += 0.8

    roads = [(edge, c) for edge, c in board.roads.items() if c == color]
    num_roads = len(roads) // 2
    longest_road = get_longest_road_length(state, color)
    score += num_roads * 0.1 + longest_road * 0.1

    try:
        buildable = board.buildable_node_ids(color)
        for node_id in buildable:
            pip_sum = 0
            try:
                tiles = board.map.adjacent_tiles.get(node_id, [])
                for tile in tiles:
                    if tile.number:
                        pip_sum += pip_counts.get(tile.number, 0)
            except (AttributeError, TypeError):
                pip_sum = 3
            score += 0.04 * pip_sum
    except (AttributeError, TypeError):
        pass

    return score


def vp_proximity_reward(state, color, prev_vps, prev_knights, opp_color):
    """
    VP Proximity Channel (event-based):
    - +1 per VP gained, -1 per VP lost
    - Scaled knight reward based on army achievability
    """
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
    """
    Terminal Reward (fired once on game end):
    - Win: +15 + 0.5 * VP margin
    - Loss: -15 + 0.5 * VP margin
    Base scaled to ±15 to match vps_to_win=15 and stay dominant over shaping.
    """
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
        self._prev_position_score = self._safe_position_score(state)
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
        current_position = self._safe_position_score(state)
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
# ACTION MASK FUNCTION
# ================================================================

def mask_fn(env) -> np.ndarray:
    valid_actions = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[valid_actions] = 1
    return np.array([bool(i) for i in mask])


# ================================================================
# LOGGING CALLBACK
# ================================================================

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
        },
    )
    env = CatanRewardWrapper(env)
    env = ActionMasker(env, mask_fn)
    return env


def make_env_hard():
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "enemies": [AlphaBetaPlayer(Color.RED)],
            "vps_to_win": 15,
        },
    )
    env = CatanRewardWrapper(env)
    env = ActionMasker(env, mask_fn)
    return env


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
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
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

    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, reward_callback])
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
    )
    model.save(f"{save_path}/final_model_hard")
    print(f"Model saved to {save_path}/final_model_hard")
    env.close()
    eval_env.close()
    return model


# ================================================================
# PLAYER CLASS
# ================================================================

class PPOPlayer(Player):
    """Catanatron Player that uses a trained MaskablePPO model to decide actions."""

    def __init__(self, color, model_path=MODEL_PATH):
        super().__init__(color)
        self.model = MaskablePPO.load(model_path)
        self._env = gymnasium.make("catanatron/Catanatron-v0")

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        obs, _ = self._env.reset()
        self._env.unwrapped.game = game
        self._env.unwrapped.p0 = self.color
        obs = self._env.unwrapped._get_obs()

        valid_actions = self._env.unwrapped.get_valid_actions()
        action_mask = np.zeros(self._env.action_space.n, dtype=bool)
        action_mask[valid_actions] = True

        action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)
        return self._env.unwrapped.actions[action]


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

    if args.continue_training:
        continue_training(
            load_path=args.continue_training,
            save_path=args.save_path,
            log_dir=args.log_dir,
            total_timesteps=500_000,
        )
    elif args.hard:
        train(
            total_timesteps=500_000,
            save_path=args.save_path,
            log_dir=args.log_dir,
            env_fn=make_env_hard,
        )
    else:
        train(
            total_timesteps=500_000,
            save_path=args.save_path,
            log_dir=args.log_dir,
        )
