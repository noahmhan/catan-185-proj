"""
PPO Baseline for 1v1 Catan using Catanatron + MaskablePPO (SB3-Contrib)

This is the flat monolithic baseline for comparison against SC-HRA.
Reward = sum of all 3 channel rewards + terminal game-outcome reward.

Requirements:
    pip install catanatron[gym]
    pip install sb3-contrib
    pip install tensorboard
"""

# Use this command to start localhost to view logs: tensorboard --logdir ./ppo_catan_logs


import numpy as np
import gymnasium
from gymnasium import Wrapper
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback

import catanatron.gym
from catanatron import Color
from catanatron.models.enums import ActionType
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.state_functions import (
    player_key,
    player_num_resource_cards,
)
import torch
torch.distributions.Distribution.set_default_validate_args(False)

# ================================================================
# REWARD SCALING
# ================================================================
# The resource channel fires every sub-action (~1-2 per step) and
# dominates the composite reward over 400+ step episodes.
# Scale it down so terminal and VP signals aren't drowned out.

RESOURCE_SCALE = 0.1       # scale resource channel down 10x
POSITION_SCALE = 1.0       # position deltas are already sparse
VP_SCALE = 2.0             # amplify VP events so they stand out
TERMINAL_SCALE = 1.0       # terminal reward already scaled via base amount


# ================================================================
# REWARD CHANNEL FUNCTIONS
# ================================================================

RESOURCE_TYPES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]


def get_player_hand(state, color):
    """Get resource card counts for a player."""
    key = player_key(state, color)
    hand = {}
    for r in RESOURCE_TYPES:
        hand[r] = state.player_state[f"{key}_{r}_IN_HAND"]
    return hand


def get_hand_size(state, color):
    """Total resource cards in hand."""
    return player_num_resource_cards(state, color)


def get_victory_points(state, color):
    """Get actual victory points for a player."""
    key = player_key(state, color)
    return state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]


def get_knights_played(state, color):
    """Number of knights played by a player."""
    key = player_key(state, color)
    return state.player_state.get(f"{key}_PLAYED_KNIGHT", 0)


def get_longest_road_length(state, color):
    """Length of longest road for a player."""
    key = player_key(state, color)
    return state.player_state.get(f"{key}_LONGEST_ROAD_LENGTH", 0)


def resource_flow_reward(state, color):
    """
    Resource Flow Channel (snapshot-based):
    - +0.2 per distinct resource type in hand
    - +0.1 per total resource card in hand
    - -0.2 + -0.1 * (hand_size - 9) if hand_size > 9
    """
    hand = get_player_hand(state, color)
    hand_size = sum(hand.values())

    # Diversity bonus
    distinct_types = sum(1 for v in hand.values() if v > 0)
    diversity_bonus = distinct_types * 0.2

    # Quantity bonus
    quantity_bonus = hand_size * 0.1

    # Hoarding penalty (1v1 rule: lose half at 10+)
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
    - +0.4 per specific resource port connected
    - +0.1 per total road + 0.1 per road in longest connected path
    - +0.04 * (total pips of adjacent tiles) per open settlement spot
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

    # Resource diversity
    score += len(resource_types_accessible) * 0.6

    # Port access — board.map.port_nodes is Dict[FastResource|None, Set[NodeId]]
    has_generic_port = False
    for resource, node_ids in board.map.port_nodes.items():
        if any(nid in buildings and buildings[nid][0] == color for nid in node_ids):
            if resource is None:  # generic 3:1 port
                has_generic_port = True
            else:
                score += 0.4

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
        score += 0.04 * pip_sum

    return score


def vp_proximity_reward(state, color, prev_vps, prev_knights, opp_color):
    """
    VP Proximity Channel (event-based):
    - +1 per VP gained, -1 per VP lost
    - Scaled knight reward: +0.3 / (1 + knights_remaining) per knight played,
      but 0 if army is unachievable
    """
    current_vps = get_victory_points(state, color)
    vp_delta = current_vps - prev_vps

    # Knight reward
    current_knights = get_knights_played(state, color)
    new_knights = current_knights - prev_knights
    knight_reward = 0.0

    if new_knights > 0:
        opp_knights = get_knights_played(state, opp_color)
        knights_needed = max(3, opp_knights + 1)

        # Check achievability: 14 total knights in deck
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
    - Win:  +100 + 2 * (your_VPs - opponent_VPs)
    - Loss: -100 + 2 * (your_VPs - opponent_VPs)

    Scaled up to ±100 so winning/losing dominates the cumulative
    per-step channel rewards (~50-80 total after scaling).
    """
    winning_color = game.winning_color()
    if winning_color is None:
        return 0.0

    state = game.state
    my_vps = get_victory_points(state, color)

    # Find opponent VPs
    opp_vps = 0
    for p in state.colors:
        if p != color:
            opp_vps = max(opp_vps, get_victory_points(state, p))

    vp_margin = 2.0 * (my_vps - opp_vps)

    if winning_color == color:
        return 10.0 + vp_margin
    else:
        return -10.0 + vp_margin


# ================================================================
# CUSTOM REWARD WRAPPER WITH PER-CHANNEL LOGGING
# ================================================================

class CatanRewardWrapper(Wrapper):
    """
    Wraps the Catanatron Gymnasium environment to provide our custom
    composite reward (sum of all 3 channels + terminal) for the PPO baseline.
    Tracks per-channel and per-episode statistics for logging.
    """

    def __init__(self, env):
        super().__init__(env)
        self.p0_color = Color.BLUE
        self.opp_color = Color.RED
        self._prev_position_score = 0.0
        self._prev_vps = 0
        self._prev_knights = 0
        self._turn_count = 0

        # Per-episode accumulators for logging
        self._ep_resource = 0.0
        self._ep_position = 0.0
        self._ep_vp = 0.0
        self._ep_terminal = 0.0
        self._ep_won = False

        # Rolling window for win rate tracking
        self._recent_wins = []
        self._total_games = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        state = self.env.unwrapped.game.state

        self._prev_position_score = network_position_score(state, self.p0_color)
        self._prev_vps = get_victory_points(state, self.p0_color)
        self._prev_knights = get_knights_played(state, self.p0_color)
        self._turn_count = 0

        # Reset episode accumulators
        self._ep_resource = 0.0
        self._ep_position = 0.0
        self._ep_vp = 0.0
        self._ep_terminal = 0.0
        self._ep_won = False

        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        game = self.env.unwrapped.game
        state = game.state
        done = terminated or truncated

        # ---- Channel 1: Resource Flow (snapshot, scaled down) ----
        r_resource_raw = resource_flow_reward(state, self.p0_color)
        r_resource = r_resource_raw * RESOURCE_SCALE

        # ---- Channel 2: Network Position (delta) ----
        current_position = network_position_score(state, self.p0_color)
        r_position_raw = current_position - self._prev_position_score
        r_position = r_position_raw * POSITION_SCALE
        self._prev_position_score = current_position

        # ---- Channel 3: VP Proximity (event, scaled up) ----
        r_vp_raw = vp_proximity_reward(
            state, self.p0_color,
            self._prev_vps, self._prev_knights,
            self.opp_color
        )
        r_vp = r_vp_raw * VP_SCALE
        self._prev_vps = get_victory_points(state, self.p0_color)
        self._prev_knights = get_knights_played(state, self.p0_color)

        # ---- Terminal Reward ----
        r_terminal = 0.0
        if done:
            r_terminal = terminal_reward(game, self.p0_color) * TERMINAL_SCALE
            self._ep_won = (game.winning_color() == self.p0_color)

        # ---- Composite Reward ----
        reward = r_resource + r_position + r_vp + r_terminal

        # ---- Accumulate for logging ----
        self._ep_resource += r_resource
        self._ep_position += r_position
        self._ep_vp += r_vp
        self._ep_terminal += r_terminal
        self._turn_count += 1

        # ---- Attach episode stats to info on done ----
        if done:
            self._total_games += 1
            self._recent_wins.append(1.0 if self._ep_won else 0.0)
            if len(self._recent_wins) > 100:
                self._recent_wins.pop(0)

            info["episode_channel_stats"] = {
                "channel/resource_flow": self._ep_resource,
                "channel/network_position": self._ep_position,
                "channel/vp_proximity": self._ep_vp,
                "channel/terminal": self._ep_terminal,
                "channel/composite": self._ep_resource + self._ep_position + self._ep_vp + self._ep_terminal,
                "game/turns": self._turn_count,
                "game/won": 1.0 if self._ep_won else 0.0,
                "game/win_rate_100": np.mean(self._recent_wins),
                "game/total_games": self._total_games,
            }

        return obs, reward, terminated, truncated, info


# ================================================================
# CHANNEL LOGGING CALLBACK
# ================================================================

class ChannelLoggingCallback(BaseCallback):
    """
    Custom callback to log per-channel rewards and win rate to TensorBoard.
    Reads episode stats from info dict populated by CatanRewardWrapper.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        logged = False
        for info in infos:
            if "episode_channel_stats" in info:
                stats = info["episode_channel_stats"]
                for key, value in stats.items():
                    self.logger.record(key, value)
                logged = True
        if logged:
            self.logger.dump(self.num_timesteps)
        return True


# ================================================================
# ACTION MASK FUNCTION
# ================================================================

def mask_fn(env) -> np.ndarray:
    """Create boolean action mask from valid actions.

    When the game is over, get_valid_actions() returns [] which would
    produce an all-zero mask. MaskablePPO stores that mask in the rollout
    buffer and later calls evaluate_actions on it, causing softmax over
    all-(-inf) logits — a Simplex constraint violation. Fall back to
    allowing action 0 as a dummy so the distribution stays valid.
    """
    valid_actions = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=bool)
    if valid_actions:
        mask[valid_actions] = True
    else:
        mask[0] = True  # dummy: episode is done, action won't be executed
    return mask


# ================================================================
# TRAINING PIPELINE
# ================================================================

def make_env():
    """Create and wrap the 1v1 Catan environment."""
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "map_type": "BASE",
            "vps_to_win": 15,          # 15 VP to win (1v1 variant)
            "enemies": [
                AlphaBetaPlayer(Color.RED),
            ],
        },
    )
    env = CatanRewardWrapper(env)
    env = ActionMasker(env, mask_fn)
    return env


def train(
    total_timesteps: int = 1_000_000,
    log_dir: str = "./ppo_catan_logs",
    model_save_path: str = "./ppo_catan_model",
    eval_freq: int = 10_000,
    n_eval_episodes: int = 20,
):
    """Train a MaskablePPO agent on 1v1 Catan."""
    env = make_env()
    eval_env = make_env()

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={
            "net_arch": dict(pi=[256, 256], vf=[256, 256]),
        },
    )

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=f"{model_save_path}/best",
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    channel_callback = ChannelLoggingCallback()

    print("=" * 60)
    print("Starting PPO Baseline Training for 1v1 Catan")
    print(f"Total timesteps: {total_timesteps}")
    print(f"VP to win: 15")
    print(f"Reward scaling: resource={RESOURCE_SCALE}, position={POSITION_SCALE}, "
          f"vp={VP_SCALE}, terminal={TERMINAL_SCALE}")
    print(f"Logging to: {log_dir}")
    print("=" * 60)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, channel_callback],
        progress_bar=True,
    )

    model.save(f"{model_save_path}/final_model")
    print(f"Model saved to {model_save_path}/final_model")

    env.close()
    eval_env.close()

    return model


def evaluate(model_path: str, n_games: int = 100):
    """Evaluate a trained model against the WeightedRandomPlayer."""
    model = MaskablePPO.load(model_path)
    env = make_env()

    wins = 0
    total_vp_margin = 0

    for game_idx in range(n_games):
        obs, info = env.reset()
        done = False
        while not done:
            valid_actions = env.unwrapped.get_valid_actions()
            mask = np.zeros(env.action_space.n, dtype=bool)
            mask[valid_actions] = True

            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        game = env.unwrapped.game
        state = game.state
        p0_color = Color.BLUE
        my_vps = get_victory_points(state, p0_color)
        opp_vps = 0
        for c in state.colors:
            if c != p0_color:
                opp_vps = max(opp_vps, get_victory_points(state, c))

        if game.winning_color() == p0_color:
            wins += 1
        total_vp_margin += my_vps - opp_vps

        if (game_idx + 1) % 10 == 0:
            print(f"Game {game_idx + 1}/{n_games} | "
                  f"Win rate: {wins/(game_idx+1):.1%} | "
                  f"Avg VP margin: {total_vp_margin/(game_idx+1):.1f}")

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS ({n_games} games)")
    print(f"Win rate: {wins/n_games:.1%}")
    print(f"Average VP margin: {total_vp_margin/n_games:.1f}")
    print("=" * 60)

    env.close()


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PPO Baseline for 1v1 Catan")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--model-path", type=str, default="./ppo_catan_model/final_model")
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--log-dir", type=str, default="./ppo_catan_logs")

    args = parser.parse_args()

    if args.mode == "train":
        train(
            total_timesteps=args.timesteps,
            log_dir=args.log_dir,
        )
    elif args.mode == "eval":
        evaluate(
            model_path=args.model_path,
            n_games=args.eval_games,
        )
