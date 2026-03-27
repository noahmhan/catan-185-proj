# CS185 PPO Catan Bot

Trains a Proximal Policy Optimization (PPO) agent to play Settlers of Catan using the catanatron gymnasium environment and action masking via `sb3-contrib`.

## Prerequisites

- Python 3.11+

## Setup

### 1. Clone the repo

```bash
git clone <repo-url>
cd catan-185-proj
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

```bash
# macOS/Linux
source .venv/bin/activate

# Windows (Git Bash)
source .venv/Scripts/activate

# Windows (cmd/PowerShell)
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -e ".[gym,rl]"
```

This installs:
- `catanatron` + the gymnasium environment (`gym` extra)
- `stable-baselines3`, `sb3-contrib`, `torch`, and `tensorboard` (`rl` extra)

> **GPU note:** The above installs a CPU-only version of PyTorch. For GPU/CUDA support (Windows), install PyTorch with CUDA first, then run the above:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
> pip install -e ".[gym,rl]"
> ```

## Training

The training script is at:
```
catanatron_experimental/catanatron_experimental/machine_learning/players/185_ppo.py
```

Run it directly:

```bash
python catanatron_experimental/catanatron_experimental/machine_learning/players/185_ppo.py
```

This trains for 500,000 timesteps against a `WeightedRandomPlayer` opponent and saves the model to:
```
catanatron_experimental/catanatron_experimental/machine_learning/players/ppo_catan_model/final_model.zip
catanatron_experimental/catanatron_experimental/machine_learning/players/ppo_catan_model/best/
```

To save to a different directory (e.g. to avoid overwriting a previous run):

```bash
python catanatron_experimental/catanatron_experimental/machine_learning/players/185_ppo.py \
    --save-path ppo_catan_model_run2 \
    --log-dir ./ppo_catan_logs/run2
```

### Train from scratch against AlphaBetaPlayer

To train a fresh model directly against the stronger `AlphaBetaPlayer` opponent:

```bash
python catanatron_experimental/catanatron_experimental/machine_learning/players/185_ppo.py \
    --hard \
    --save-path ppo_catan_model_hard \
    --log-dir ./ppo_catan_logs/hard
```

### Curriculum training against AlphaBetaPlayer

After an initial run, you can fine-tune the saved model against the stronger `AlphaBetaPlayer` (depth-2 minimax) opponent for another 500,000 timesteps:

```bash
python catanatron_experimental/catanatron_experimental/machine_learning/players/185_ppo.py \
    --continue-training ppo_catan_model/final_model
```

The fine-tuned model is saved to:
```
catanatron_experimental/catanatron_experimental/machine_learning/players/ppo_catan_model/final_model_hard.zip
catanatron_experimental/catanatron_experimental/machine_learning/players/ppo_catan_model/best_hard/
```

TensorBoard logs continue from the same run (timestep count is not reset), so the full training curve appears in one chart.

> **Note:** `AlphaBetaPlayer` runs a minimax search each turn and is significantly slower per step than `WeightedRandomPlayer`. Expect the fine-tuning stage to take longer in wall-clock time.

### Monitoring with TensorBoard

In a separate terminal (with the venv activated):

```bash
tensorboard --logdir ./ppo_catan_logs
```

Then open `http://localhost:6006` in your browser.

### Customizing training

Edit the `train()` call at the bottom of `185_ppo.py`:

```python
train(
    total_timesteps=500_000,
    save_path="./my_model",
    log_dir="./my_logs",
    eval_freq=5_000,
    n_eval_episodes=10,
)
```

### Reward function

The composite reward is computed by `CatanRewardWrapper` and consists of four channels:

- **Resource flow** — delta reward for changes in hand diversity and size (fires only when hand composition changes)
- **Network position** — delta reward for board position improvements, scaled ×0.3 (fires on builds)
- **VP proximity** — event reward for VP gains and knight progress (+1 per VP)
- **Terminal** — ±15 on win/loss, scaled by VP margin

Each channel's episode total is logged separately to TensorBoard under `reward/ep_resource`, `reward/ep_position`, `reward/ep_vp`, `reward/ep_terminal`, and `reward/win_rate`.

## Using the trained bot

```python
from catanatron import Color
from catanatron_experimental.machine_learning.players.185_ppo import PPOPlayer

player = PPOPlayer(Color.BLUE)
# Use player in any catanatron game or simulation
```

## File overview

| File | Purpose |
|------|---------|
| `185_ppo.py` | Training script + `PPOPlayer` class |
| `pyproject.toml` | Project dependencies (add `rl` extra) |
