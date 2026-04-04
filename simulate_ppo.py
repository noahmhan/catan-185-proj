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
from catanatron.features import create_sample, get_feature_ordering
from catanatron.gym.envs.action_space import (
    from_action_space,
    get_action_array,
    to_action_space,
)
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.web.utils import ensure_link

# ---------------------------------------------------------------------------
# Paths to trained artefacts
# ---------------------------------------------------------------------------

LEAGUE_MODEL  = os.path.join(ROOT, "ppo_league", "league_model.zip")
BEST_LEAGUE   = os.path.join(ROOT, "ppo_league", "best_league", "best_model.zip")
LEAGUE_NORM   = os.path.join(ROOT, "ppo_league", "vecnormalize.pkl")

MEDIUM_MODEL  = os.path.join(ROOT, "ppo_medium_eps", "medium_model.zip")
BEST_MEDIUM   = os.path.join(ROOT, "ppo_medium_eps", "best_medium", "best_model.zip")
MEDIUM_NORM   = os.path.join(ROOT, "ppo_medium_eps", "vecnormalize_medium.pkl")

MODEL_REGISTRY = {
    "league":      (LEAGUE_MODEL, LEAGUE_NORM),
    "best_league": (BEST_LEAGUE,  LEAGUE_NORM),
    "medium":      (MEDIUM_MODEL, MEDIUM_NORM),
    "best_medium": (BEST_MEDIUM,  MEDIUM_NORM),
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
        print(f"[PPOPlayer] Loaded model from {model_path}")

    def _get_obs(self, game: Game) -> np.ndarray:
        sample = create_sample(game, self.color)
        return np.array([sample[f] for f in FEATURES], dtype=np.float32)

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


def play_game(ppo_player: PPOPlayer, opponent: Player, game_num: int) -> tuple:
    """Play one game to completion, then save the final state once. Returns (winner, url)."""
    players = [ppo_player, opponent]

    opp_label = type(opponent).__name__
    print(f"\n=== Game {game_num} ===")
    print(f"  BLUE: {ppo_player.name}")
    print(f"  RED:  {opp_label}")

    game = Game(players, vps_to_win=15)
    game.play()  # run to completion with no DB calls

    winner = game.winning_color()
    url = ensure_link(game)  # single DB write after the game
    print(f"  Winner: {winner}")
    print(f"  URL:    {url}")

    return winner, url


def main():
    parser = argparse.ArgumentParser(description="Simulate PPO bot vs AlphaBeta or ValueFunction")
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default="league",
        help="Which trained model to load (default: league)",
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
    args = parser.parse_args()

    model_path, _ = MODEL_REGISTRY[args.model]
    if not os.path.exists(model_path):
        print(f"ERROR: model file not found: {model_path}")
        sys.exit(1)

    ppo_player = PPOPlayer(
        color=Color.BLUE,
        model_path=model_path,
        name=f"PPO-{args.model}",
    )

    results = []
    urls = []
    for i in range(1, args.games + 1):
        opponent = make_opponent(args.opponent, args.depth)
        winner, url = play_game(ppo_player, opponent, i)
        results.append(winner)
        urls.append(url)

    wins = sum(1 for w in results if w == Color.BLUE)
    print(f"\n=== Summary ({args.games} game(s)) ===")
    print(f"  PPO wins:      {wins}")
    print(f"  Opponent wins: {args.games - wins}")
    print(f"\nURLs:")
    for i, url in enumerate(urls, 1):
        print(f"  Game {i}: {url}")

    if not args.no_gui:
        import webbrowser
        webbrowser.open(urls[-1])
        print(f"\nOpened last game in browser: {urls[-1]}")


if __name__ == "__main__":
    main()
