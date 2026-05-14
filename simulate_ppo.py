"""
Simulate trained PPO bots vs AlphaBetaPlayer or ValueFunctionPlayer.
Games are saved to the database so you can open them in the GUI afterwards.

Usage:
    python simulate_ppo.py                              # league model vs alphabeta
    python simulate_ppo.py --model medium               # medium model
    python simulate_ppo.py --model best_league          # best league checkpoint
    python simulate_ppo.py --model best_medium          # best medium checkpoint
    python simulate_ppo.py --model schra                # SC-HRA latest checkpoint
    python simulate_ppo.py --model best_schra           # SC-HRA best-eval checkpoint
    python simulate_ppo.py --model schra_nsteps1        # Modal-trained SC-HRA (nsteps1 best_league_step_29000145)
    python simulate_ppo.py --schra-path PATH            # any SC-HRA checkpoint directory
    python simulate_ppo.py --opponent value             # vs ValueFunctionPlayer
    python simulate_ppo.py --games 3                    # play 3 games (default 1)
    python simulate_ppo.py --depth 1                    # AlphaBeta depth (default 2)
    python simulate_ppo.py --no-gui                     # skip opening browser at end
    python simulate_ppo.py --alphabeta-init             # AlphaBeta n=2 plays initial placements
"""

import argparse
import csv
import datetime
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
import importlib.util as _ilu
_ppo_mod = _il.import_module("catanatron_experimental.machine_learning.players.185_ppo")
compute_features = _ppo_mod.compute_features
from catanatron_experimental.machine_learning.players.initial_placement_ppo import (
    encode_board_state as compute_init_placement_features,
    OBS_DIM as INIT_PLACEMENT_OBS_DIM,
)


def _import_schra():
    """Import sc-hra.py despite the hyphen in its filename."""
    path = os.path.join(
        ROOT, "catanatron_experimental", "catanatron_experimental",
        "machine_learning", "players", "sc-hra.py",
    )
    spec = _ilu.spec_from_file_location("schra", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load sc-hra module from {path}")
    schra = _ilu.module_from_spec(spec)
    spec.loader.exec_module(schra)
    return schra

# ---------------------------------------------------------------------------
# Paths to trained artefacts
# ---------------------------------------------------------------------------
BEST_MODEL = os.path.join(ROOT, "continued_league", "league_model.zip")
BEST_NORM = os.path.join(ROOT, "continued_league", "vecnormalize.pkl")

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

# SC-HRA (sc-hra.py) checkpoint directories — each is a folder containing
# critic_*.pt, target_critic_*.pt, meta_net.pt and optionally checkpoint.pt.
SCHRA_MODEL_DIR  = os.path.join(
    ROOT, "catanatron_experimental", "catanatron_experimental",
    "machine_learning", "players", "schra_model",
)
SCHRA_BEST       = os.path.join(SCHRA_MODEL_DIR, "best")
SCHRA_CHECKPOINT = os.path.join(SCHRA_MODEL_DIR, "checkpoint")

# Modal-trained SC-HRA checkpoints downloaded to the user's local machine.
SCHRA_NSTEPS1 = os.path.join(
    os.path.expanduser("~"),
    "modal_downloads", "schra-data", "models", "nsteps1",
    "best_league_step_29000145",
)

MODEL_REGISTRY = {
    "league":         (LEAGUE_MODEL, LEAGUE_NORM),
    "best_league":    (BEST_LEAGUE,  LEAGUE_NORM),
    "medium":         (MEDIUM_MODEL, MEDIUM_NORM),
    "best_medium":    (BEST_MEDIUM,  MEDIUM_NORM),
    "best":           (BEST_MODEL, BEST_NORM),
    "schra":          (SCHRA_CHECKPOINT, None),
    "best_schra":     (SCHRA_BEST, None),
    "schra_nsteps1":  (SCHRA_NSTEPS1, None),
    "none":           (None, None),
}

SCHRA_MODELS = {"schra", "best_schra", "schra_nsteps1"}

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


class SCHRAModelPlayer(Player):
    """Wraps a trained SC-HRA agent (3 DQN critics + meta-weighting net) as a
    catanatron Player. SC-HRA uses 25-dim hand-crafted features and chooses
    actions by combining per-channel Q-values via state-conditioned weights."""

    # Names for SC-HRA's 3 reward channels, matching sc-hra.py SCHRAWrapper.
    CHANNEL_NAMES = ("resource", "position", "vp")

    def __init__(self, color: Color, model_path: str, name: str = "SCHRA"):
        super().__init__(color, is_bot=True)
        self.name = name
        self._schra = _import_schra()
        # SC-HRA's critic outputs Q-values indexed against the gym env's
        # action enumeration; we need a live env to map indices back to
        # catanatron actions in decide().
        import gymnasium
        self._env = gymnasium.make("catanatron/Catanatron-v0")
        n_actions = self._env.action_space.n
        self.agent = self._schra.SCHRAAgent(n_actions)
        self.agent.load(model_path)
        # Restore Q-value normalizer stats from the full checkpoint if it
        # exists, so inference uses the training-time normalization rather
        # than rebuilding stats from a fresh (mean=0, var=1) baseline.
        ckpt_pt = os.path.join(model_path, "checkpoint.pt")
        if os.path.exists(ckpt_pt):
            import torch
            ckpt = torch.load(ckpt_pt, map_location="cpu", weights_only=False)
            for i, n in enumerate(ckpt.get("q_normalizers", [])):
                self.agent.q_normalizers[i].mean = n["mean"]
                self.agent.q_normalizers[i].var = n["var"]
                self.agent.q_normalizers[i].count = n["count"]
        # No exploration at play time.
        self.agent.epsilon_start = 0.0
        self.agent.epsilon_end = 0.0
        self.opp_color = None
        # Per-decision omega + chosen-action Q trace for the current game.
        # Reset between games via reset_omega_log(); flushed to disk via
        # dump_omega_log().
        self.omega_log = []
        self.omega_log_dir = os.path.join(ROOT, "schra_omega_logs")
        print(f"[SCHRAModelPlayer] Loaded SC-HRA model from {model_path} "
              f"(n_actions={n_actions})")

    def reset_omega_log(self):
        self.omega_log = []

    def dump_omega_log(self, tag: str = "") -> str | None:
        if not self.omega_log:
            return None
        os.makedirs(self.omega_log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"omega_{ts}{('_' + tag) if tag else ''}.csv"
        path = os.path.join(self.omega_log_dir, fname)
        cols = self.omega_log[0].keys()
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(self.omega_log)
        # Summary: mean omega per channel across all decisions in the game.
        omegas = np.array([
            [row[f"omega_{c}"] for c in self.CHANNEL_NAMES]
            for row in self.omega_log
        ])
        means = omegas.mean(axis=0)
        first = omegas[0]
        last  = omegas[-1]
        print(f"  Omega log:  {path}  ({len(self.omega_log)} decisions)")
        print(f"    mean   ω: " + ", ".join(
            f"{c}={m:.3f}" for c, m in zip(self.CHANNEL_NAMES, means)))
        print(f"    first  ω: " + ", ".join(
            f"{c}={v:.3f}" for c, v in zip(self.CHANNEL_NAMES, first)))
        print(f"    last   ω: " + ", ".join(
            f"{c}={v:.3f}" for c, v in zip(self.CHANNEL_NAMES, last)))
        return path

    def __reduce__(self):
        # Same trick as PPOPlayer — the Flask server unpickles game states
        # but doesn't know about SCHRAModelPlayer.
        return (Player, (self.color, True))

    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        state = game.state
        if self.opp_color is None:
            self.opp_color = next(c for c in state.colors if c != self.color)

        obs = self._schra.compute_features(state, self.color, self.opp_color)

        self._env.reset()
        self._env.unwrapped.game = game
        valid_actions = self._env.unwrapped.get_valid_actions()
        if not valid_actions:
            return playable_actions[0]

        action_idx = self.agent.select_action(obs, valid_actions)

        # Capture this turn's meta-weights ω and the per-channel normalized
        # Q for the chosen action. select_action already pushed (state,
        # q_at_action) to episode_buffer — we grab it before clearing.
        import torch
        with torch.no_grad():
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(self._schra.DEVICE)
            omega = self.agent.meta_net(state_t).squeeze(0).cpu().numpy()
        q_at_action = None
        if self.agent.episode_buffer:
            q_at_action = self.agent.episode_buffer[-1][1].cpu().numpy()
        # Now safe to clear the meta-net training buffer.
        self.agent.episode_buffer.clear()

        my_vps = self._schra.get_victory_points(state, self.color)
        opp_vps = self._schra.get_victory_points(state, self.opp_color)
        row: dict[str, float] = {
            "turn": int(state.num_turns),
            "decision_idx": len(self.omega_log),
            "my_vps": int(my_vps),
            "opp_vps": int(opp_vps),
            "action_idx": int(action_idx),
        }
        for i, ch in enumerate(self.CHANNEL_NAMES):
            row[f"omega_{ch}"] = float(omega[i])
            row[f"q_{ch}"] = float(q_at_action[i]) if q_at_action is not None else float("nan")
        self.omega_log.append(row)

        catan_action = from_action_space(action_idx, self.color, PLAYER_COLORS, MAP_TYPE)
        if catan_action in playable_actions:
            return catan_action
        # Fallback: map the gym index back to a playable_action by index match.
        for a in playable_actions:
            if to_action_space(a, PLAYER_COLORS, MAP_TYPE) == action_idx:
                return a
        return playable_actions[0]


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
    parser.add_argument(
        "--schra-path",
        default=None,
        help="Override the SC-HRA model directory (must contain critic_*.pt, "
             "target_critic_*.pt, meta_net.pt; checkpoint.pt optional). When "
             "set, --model is forced to a SC-HRA loader regardless of its value.",
    )
    args = parser.parse_args()

    if args.init_placement and args.alphabeta_init:
        print("ERROR: --init-placement and --alphabeta-init are mutually exclusive")
        sys.exit(1)

    model_path, _ = MODEL_REGISTRY[args.model]

    if args.schra_path is not None:
        schra_path = os.path.abspath(os.path.expanduser(args.schra_path))
        if not os.path.isdir(schra_path):
            print(f"ERROR: SC-HRA model directory not found: {schra_path}")
            sys.exit(1)
        schra_label = os.path.basename(schra_path.rstrip(os.sep)) or "schra"
        main_player = SCHRAModelPlayer(
            color=Color.BLUE,
            model_path=schra_path,
            name=f"SCHRA-{schra_label}",
        )
        main_label = f"SCHRA-{schra_label}"
    elif args.model == "none":
        if not (args.init_placement or args.alphabeta_init):
            print("ERROR: --model none only makes sense with --init-placement or --alphabeta-init")
            sys.exit(1)
        main_player = WeightedRandomPlayer(Color.BLUE)
        main_label = "WeightedRandom"
    elif args.model in SCHRA_MODELS:
        if not os.path.isdir(model_path):
            print(f"ERROR: SC-HRA model directory not found: {model_path}")
            sys.exit(1)
        main_player = SCHRAModelPlayer(
            color=Color.BLUE,
            model_path=model_path,
            name=f"SCHRA-{args.model}",
        )
        main_label = f"SCHRA-{args.model}"
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
        if isinstance(main_player, SCHRAModelPlayer):
            main_player.reset_omega_log()
        opponent = make_opponent(args.opponent, args.depth)
        winner, url, last_game = play_game(ppo_player, opponent, i)
        results.append(winner)
        urls.append(url)
        if isinstance(main_player, SCHRAModelPlayer):
            winner_str = "BLUE" if winner == Color.BLUE else ("RED" if winner == Color.RED else "draw")
            main_player.dump_omega_log(tag=f"game{i}_{winner_str}")

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
