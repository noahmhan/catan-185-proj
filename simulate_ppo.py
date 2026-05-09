"""
Simulate trained PPO bots vs AlphaBetaPlayer or ValueFunctionPlayer.
Games are saved to the database so you can open them in the GUI afterwards.

Usage:
    python simulate_ppo.py                              # league model vs alphabeta
    python simulate_ppo.py --model medium               # medium model
    python simulate_ppo.py --model best_league          # best league checkpoint
    python simulate_ppo.py --model best_medium          # best medium checkpoint
    python simulate_ppo.py --opponent value             # vs ValueFunctionPlayer
    python simulate_ppo.py --games 3                    # play 3 games (default 1)
    python simulate_ppo.py --depth 1                    # AlphaBeta depth (default 2)
    python simulate_ppo.py --no-gui                     # skip opening browser at end
    python simulate_ppo.py --alphabeta-init             # AlphaBeta n=2 plays initial placements
"""

import argparse
import os
import sys

import numpy as np

# Make sure catanatron packages are importable
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "catanatron"))
sys.path.insert(0, os.path.join(ROOT, "catanatron_experimental"))

from sb3_contrib.ppo_mask import MaskablePPO

from catanatron import Color, Game, Player
from catanatron.features import create_sample_vector, get_feature_ordering
from catanatron.gym.envs.action_space import (
    from_action_space,
    get_action_array,
    to_action_space,
)
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.web.models import GameState, database_session
from catanatron.web.utils import ensure_link, open_link
import importlib as _il
_ppo_mod = _il.import_module("catanatron_experimental.machine_learning.players.185_ppo")
compute_features = _ppo_mod.compute_features
from catanatron_experimental.machine_learning.players.initial_placement_ppo import (
    encode_board_state as compute_init_placement_features,
    OBS_DIM as INIT_PLACEMENT_OBS_DIM,
)

# ---------------------------------------------------------------------------
# Paths to trained artefacts
# ---------------------------------------------------------------------------

BEST_MODEL = os.path.join(ROOT, "best_ppo", "league_model.zip")
BEST_NORM = os.path.join(ROOT, "best_ppo", "vecnormalize.pkl")

LEAGUE_MODEL  = os.path.join(ROOT, "ppo_league", "league_model.zip")
BEST_LEAGUE   = os.path.join(ROOT, "ppo_league", "best_league", "best_model.zip")
LEAGUE_NORM   = os.path.join(ROOT, "ppo_league", "vecnormalize.pkl")

MEDIUM_MODEL  = os.path.join(ROOT, "ppo_medium_eps", "medium_model.zip")
BEST_MEDIUM   = os.path.join(ROOT, "ppo_medium_eps", "best_medium", "best_model.zip")
MEDIUM_NORM   = os.path.join(ROOT, "ppo_medium_eps", "vecnormalize_medium.pkl")

# Initial-placement-only model (trained by initial_placement_ppo.py).
INIT_PLACE_DIR   = os.path.join(
    ROOT, "catanatron_experimental", "catanatron_experimental",
    "machine_learning", "players", "initial_placement_model",
)
INIT_PLACE_FINAL = os.path.join(INIT_PLACE_DIR, "final_model.zip")
INIT_PLACE_BEST  = os.path.join(INIT_PLACE_DIR, "best", "best_model.zip")

MODEL_REGISTRY = {
    "league":      (LEAGUE_MODEL, LEAGUE_NORM),
    "best_league": (BEST_LEAGUE,  LEAGUE_NORM),
    "medium":      (MEDIUM_MODEL, MEDIUM_NORM),
    "best_medium": (BEST_MEDIUM,  MEDIUM_NORM),
    "best":        (BEST_MODEL, BEST_NORM),
    "none":        (None, None),
}

# ---------------------------------------------------------------------------
# PPOPlayer  — wraps a MaskablePPO checkpoint as a catanatron Player
# ---------------------------------------------------------------------------

PLAYER_COLORS    = (Color.BLUE, Color.RED)  # must match training setup
MAP_TYPE         = "BASE"
FEATURES         = get_feature_ordering(num_players=2, map_type=MAP_TYPE)
ACTION_ARRAY     = get_action_array(PLAYER_COLORS, MAP_TYPE)
ACTION_SPACE_SIZE = len(ACTION_ARRAY)


class PPOPlayer(Player):
    """Wraps a trained MaskablePPO model as a catanatron Player."""

    def __init__(self, color: Color, model_path: str, name: str = "PPO"):
        super().__init__(color, is_bot=True)
        self.name = name
        self.model = MaskablePPO.load(model_path)
        # The shipped checkpoints were trained with two different observation
        # encoders: the 25-dim hand-crafted vector from 185_ppo.compute_features
        # (medium / best_medium) and the 614-dim raw catanatron feature vector
        # (league / best_league / best). Pick the encoder that matches what the
        # model expects, otherwise the first .predict() call crashes with a
        # shape error.
        obs_space = self.model.observation_space
        if obs_space is None or obs_space.shape is None:
            raise ValueError(f"Model {model_path} has no Box observation_space")
        expected_dim = int(obs_space.shape[0])
        if expected_dim == len(FEATURES):  # 614 — raw catanatron features
            self._obs_kind = "raw"
        elif expected_dim == 25:           # hand-crafted compute_features
            self._obs_kind = "compute"
        else:
            raise ValueError(
                f"Unsupported observation dim {expected_dim} for {model_path}. "
                f"Expected 25 (compute_features) or {len(FEATURES)} (raw catanatron)."
            )
        print(f"[PPOPlayer] Loaded model from {model_path} "
              f"(obs dim={expected_dim}, encoder={self._obs_kind})")

    def _get_obs(self, game: Game) -> np.ndarray:
        if self._obs_kind == "raw":
            return np.asarray(create_sample_vector(game, self.color, FEATURES), dtype=np.float32)
        opp_color = Color.RED if self.color == Color.BLUE else Color.BLUE
        return compute_features(game.state, self.color, opp_color)

    def _get_action_mask(self, game: Game) -> np.ndarray:
        valid_ints = {
            to_action_space(a, PLAYER_COLORS, MAP_TYPE)
            for a in game.playable_actions
        }
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        for i in valid_ints:
            mask[i] = True
        if not mask.any():
            mask[0] = True  # safety fallback
        return mask

    def __reduce__(self):
        # The Flask server unpickles game states to serve them as JSON, but it
        # has never imported PPOPlayer. Reconstruct as a plain Player so the
        # pickle is portable and the server can load the game without errors.
        return (Player, (self.color, True))

    def decide(self, game: Game, playable_actions):
        obs  = self._get_obs(game)
        mask = self._get_action_mask(game)

        action_int, _ = self.model.predict(
            obs[np.newaxis, :],
            action_masks=mask[np.newaxis, :],
            deterministic=True,
        )
        action_int = int(action_int[0])

        try:
            catan_action = from_action_space(action_int, self.color, PLAYER_COLORS, MAP_TYPE)
            if catan_action in playable_actions:
                return catan_action
        except Exception:
            pass

        # Fallback: re-predict with only valid actions unmasked
        fallback_mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        for i in sorted(to_action_space(a, PLAYER_COLORS, MAP_TYPE) for a in playable_actions):
            fallback_mask[i] = True
        action_int, _ = self.model.predict(
            obs[np.newaxis, :],
            action_masks=fallback_mask[np.newaxis, :],
            deterministic=True,
        )
        return from_action_space(int(action_int[0]), self.color, PLAYER_COLORS, MAP_TYPE)


class InitPlacementPPOPlayer(Player):
    """Plays the 4 initial placements with a dedicated model that consumes the
    board-state observation from initial_placement_ppo.encode_board_state, then
    delegates every other decision to a fallback player."""

    def __init__(self, color: Color, model_path: str, fallback: Player, name: str = "InitPlace"):
        super().__init__(color, is_bot=True)
        self.name = name
        self.model = MaskablePPO.load(model_path)
        self.fallback = fallback

    def __reduce__(self):
        return (Player, (self.color, True))

    def decide(self, game: Game, playable_actions):
        state = game.state
        if not state.is_initial_build_phase:
            return self.fallback.decide(game, playable_actions)
        if len(playable_actions) == 1:
            return playable_actions[0]

        obs = compute_init_placement_features(state.board, state)

        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        idx_to_action = {}
        for action in playable_actions:
            try:
                idx = to_action_space(action, PLAYER_COLORS, MAP_TYPE)
            except Exception:
                continue
            mask[idx] = True
            idx_to_action[idx] = action
        if not mask.any():
            mask[0] = True

        action_idx, _ = self.model.predict(
            obs[np.newaxis, :],
            action_masks=mask[np.newaxis, :],
            deterministic=True,
        )
        return idx_to_action.get(int(action_idx[0]), playable_actions[0])


class AlphaBetaInitPlacementPlayer(Player):
    """Plays the initial build phase with an AlphaBetaPlayer, then delegates
    every other decision to a fallback player."""

    def __init__(self, color: Color, depth: int, fallback: Player, name: str = "ABInitPlace"):
        super().__init__(color, is_bot=True)
        self.name = name
        self.depth = depth
        self.alphabeta = AlphaBetaPlayer(color, depth=depth)
        self.fallback = fallback

    def __reduce__(self):
        return (Player, (self.color, True))

    def decide(self, game: Game, playable_actions):
        if game.state.is_initial_build_phase:
            return self.alphabeta.decide(game, playable_actions)
        return self.fallback.decide(game, playable_actions)


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def make_opponent(opponent: str, depth: int) -> Player:
    if opponent == "alphabeta":
        return AlphaBetaPlayer(Color.RED, depth=depth)
    elif opponent == "value":
        return ValueFunctionPlayer(Color.RED)
    else:
        raise ValueError(f"Unknown opponent: {opponent}")


def play_game(ppo_player: Player, opponent: Player, game_num: int) -> tuple:
    """Play one game, saving every state to the DB in one session for replay. Returns (winner, url)."""
    players = [ppo_player, opponent]

    blue_label = getattr(ppo_player, "name", type(ppo_player).__name__)
    opp_label = type(opponent).__name__
    print(f"\n=== Game {game_num} ===")
    print(f"  BLUE: {blue_label}")
    print(f"  RED:  {opp_label}")

    game = Game(players, vps_to_win=15, discard_limit=7)

    # Collect all states during play, then commit once — avoids opening a new
    # DB engine on every tick (the main performance killer).
    states = []
    while game.winning_color() is None:
        game.play_tick()
        states.append(GameState.from_game(game))

    with database_session() as session:
        for gs in states:
            session.add(gs)
        session.commit()

    winner = game.winning_color()
    url = ensure_link(game, get_replay_link=True)
    print(f"  Winner: {winner}")
    print(f"  URL:    {url}")

    return winner, url, game


def main():
    parser = argparse.ArgumentParser(description="Simulate PPO bot vs AlphaBeta or ValueFunction")
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default="best",
        help="Which trained model to load (default: best)",
    )
    parser.add_argument(
        "--opponent",
        choices=["alphabeta", "value"],
        default="alphabeta",
        help="Opponent type (default: alphabeta)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1,
        help="Number of games to play (default: 1)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="AlphaBeta search depth, ignored for value opponent (default: 2)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Save to DB but do not open browser at the end",
    )
    parser.add_argument(
        "--init-placement",
        action="store_true",
        help="Use the initial-placement PPO model for the 4 opening decisions, "
             "then hand off to --model for the rest of the game.",
    )
    parser.add_argument(
        "--init-placement-path",
        default=None,
        help="Path to initial-placement model .zip. Defaults to the eval-best "
             "checkpoint if present, else final_model.zip.",
    )
    parser.add_argument(
        "--alphabeta-init",
        action="store_true",
        help="Use AlphaBetaPlayer for the initial placements, then hand off to "
             "--model for the rest of the game. Mutually exclusive with --init-placement.",
    )
    parser.add_argument(
        "--alphabeta-init-depth",
        type=int,
        default=2,
        help="Depth for the AlphaBeta initial-placement player (default: 2).",
    )
    args = parser.parse_args()

    if args.init_placement and args.alphabeta_init:
        print("ERROR: --init-placement and --alphabeta-init are mutually exclusive")
        sys.exit(1)

    model_path, _ = MODEL_REGISTRY[args.model]

    if args.model == "none":
        if not (args.init_placement or args.alphabeta_init):
            print("ERROR: --model none only makes sense with --init-placement or --alphabeta-init")
            sys.exit(1)
        main_player = WeightedRandomPlayer(Color.BLUE)
        main_label = "WeightedRandom"
    else:
        if not os.path.exists(model_path):
            print(f"ERROR: model file not found: {model_path}")
            sys.exit(1)
        main_player = PPOPlayer(
            color=Color.BLUE,
            model_path=model_path,
            name=f"PPO-{args.model}",
        )
        main_label = f"PPO-{args.model}"

    if args.init_placement:
        ip_path = args.init_placement_path
        if ip_path is None:
            ip_path = INIT_PLACE_BEST if os.path.exists(INIT_PLACE_BEST) else INIT_PLACE_FINAL
        if not os.path.exists(ip_path):
            print(f"ERROR: initial-placement model not found: {ip_path}")
            sys.exit(1)
        ppo_player = InitPlacementPPOPlayer(
            color=Color.BLUE,
            model_path=ip_path,
            fallback=main_player,
            name=f"InitPlace+{main_label}",
        )
        print(f"[InitPlacementPPOPlayer] Loaded init-placement model from {ip_path} "
              f"(obs dim={INIT_PLACEMENT_OBS_DIM})")
    elif args.alphabeta_init:
        ppo_player = AlphaBetaInitPlacementPlayer(
            color=Color.BLUE,
            depth=args.alphabeta_init_depth,
            fallback=main_player,
            name=f"AB{args.alphabeta_init_depth}Init+{main_label}",
        )
        print(f"[AlphaBetaInitPlacementPlayer] AlphaBeta(depth={args.alphabeta_init_depth}) "
              f"will play initial placements, then hand off to {main_label}")
    else:
        ppo_player = main_player

    results = []
    urls = []
    last_game = None
    for i in range(1, args.games + 1):
        opponent = make_opponent(args.opponent, args.depth)
        winner, url, last_game = play_game(ppo_player, opponent, i)
        results.append(winner)
        urls.append(url)

    wins = sum(1 for w in results if w == Color.BLUE)
    print(f"\n=== Summary ({args.games} game(s)) ===")
    print(f"  PPO wins:      {wins}")
    print(f"  Opponent wins: {args.games - wins}")
    print(f"\nURLs:")
    for i, url in enumerate(urls, 1):
        print(f"  Game {i}: {url}")

    if not args.no_gui and last_game is not None:
        open_link(last_game)
        print(f"\nOpened last game in browser: {urls[-1]}")


if __name__ == "__main__":
    main()
