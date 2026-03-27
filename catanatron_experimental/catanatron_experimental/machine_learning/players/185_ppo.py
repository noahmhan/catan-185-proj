import os

import numpy as np
import gymnasium
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from catanatron import Color, Player
from catanatron.players.weighted_random import WeightedRandomPlayer
import catanatron.gym


MODEL_PATH = os.path.join(os.path.dirname(__file__), "ppo_catan_model")


def mask_fn(env) -> np.ndarray:
    valid_actions = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[valid_actions] = 1
    return np.array([bool(i) for i in mask])

def accumulated_reward_fn(action, game, p0_color):
    ...

def train(total_timesteps=100_000, save_path=MODEL_PATH):
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "enemies": [
                WeightedRandomPlayer(Color.RED),
            ],
            "reward_function": accumulated_reward_fn,
        },
    )
    env = ActionMasker(env, mask_fn)
    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"Model saved to {save_path}")
    return model


class PPOPlayer(Player):
    """Catanatron Player that uses a trained MaskablePPO model to decide actions."""

    def __init__(self, color, model_path=MODEL_PATH):
        super().__init__(color)
        self.model = MaskablePPO.load(model_path)
        # Build a temporary env just to get the action space / observation info
        self._env = gymnasium.make("catanatron/Catanatron-v0")

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        obs, _ = self._env.reset()
        # Overwrite the env's internal game state with the live game
        self._env.unwrapped.game = game
        self._env.unwrapped.p0 = self.color
        obs = self._env.unwrapped._get_obs()

        valid_actions = self._env.unwrapped.get_valid_actions()
        action_mask = np.zeros(self._env.action_space.n, dtype=bool)
        action_mask[valid_actions] = True

        action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)
        # Map gym action index back to a catanatron Action object
        return self._env.unwrapped.actions[action]


if __name__ == "__main__":
    train(total_timesteps=100_000)
