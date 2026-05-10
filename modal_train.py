"""
Modal entry point for SC-HRA league training.

One-time setup:
    pip install modal
    modal token new                          # browser-based auth, no key in code

Kick off training (returns once the container exits — long jobs run detached):
    modal run --detach modal_train.py \\
        --timesteps 30000000 \\
        --save-name run1

Watch progress (after the job is running) with the live TensorBoard URL:
    modal serve modal_train.py
    # → opens a public URL backed by /data/logs

Pull results back to your laptop:
    modal volume get schra-data /models/run1 ./run1_model
    modal volume get schra-data /logs/run1   ./run1_logs

Resume a partially trained run:
    modal run --detach modal_train.py \\
        --timesteps 30000000 \\
        --save-name run1 \\
        --resume-from run1
"""

import modal

APP_NAME = "schra-league"
VOLUME_NAME = "schra-data"
REMOTE_PROJECT = "/root/catan"

# Files inside the local project that have no business inside the image.
# (Patterns are gitignore-style, evaluated relative to the local source root.)
_IGNORE = [
    ".git",
    ".venv",
    "venv",
    "db-data",
    "node_modules",
    "**/__pycache__",
    "**/*.pyc",
    "**/*.egg-info",
    ".pytest_cache",
    ".benchmarks",
    "ui/build",
    "ui/node_modules",
    "ppo_catan_model*",
    "ppo_catan_logs",
    "schra_model",
    "schra_logs",
    "continued_league",
    "*.log",
    ".DS_Store",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch",
        "stable-baselines3>=2.0.0",
        "sb3-contrib>=2.0.0",
        "tensorboard",
        "tqdm",
        "rich",
        "gymnasium<=0.29.1",
        "numpy",
        "pandas",
        "networkx",
        "click",
        "fastparquet",
        "pygame",
    )
    .add_local_dir(".", remote_path=REMOTE_PROJECT, copy=True, ignore=_IGNORE)
    .run_commands(
        f"cd {REMOTE_PROJECT} && pip install -e .",
        f"cd {REMOTE_PROJECT} && pip install -e ./catanatron_experimental",
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App(APP_NAME, image=image)


def _import_schra():
    """Import sc-hra.py despite the hyphen in its filename."""
    import importlib.util

    path = (
        f"{REMOTE_PROJECT}/catanatron_experimental/catanatron_experimental"
        f"/machine_learning/players/sc-hra.py"
    )
    spec = importlib.util.spec_from_file_location("schra", path)
    schra = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(schra)
    return schra


@app.function(
    gpu="H100",
    volumes={"/data": volume},
    timeout=24 * 60 * 60,  # 24h cap; bump if a run will exceed this
)
def run_league(
    timesteps: int = 5_000_000,
    save_name: str = "default",
    start_stage: int = 0,
    device: str = "gpu",
    resume_from: str = "",
):
    """Launches sc-hra's league_train inside a GPU container.

    save_name: namespace under the persistent volume:
        models go to /data/models/{save_name}
        logs   go to /data/logs/{save_name}

    resume_from: name of a previous save_name to resume from. Reads
        /data/models/{resume_from}/checkpoint_league/checkpoint.pt.
    """
    import os

    schra = _import_schra()

    save_path = f"/data/models/{save_name}"
    log_dir = f"/data/logs/{save_name}"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    resume_path = None
    if resume_from:
        resume_path = f"/data/models/{resume_from}/checkpoint_league"
        if not os.path.isdir(resume_path):
            raise FileNotFoundError(
                f"resume_from='{resume_from}' but {resume_path} doesn't exist on volume"
            )

    schra.league_train(
        total_timesteps=timesteps,
        save_path=save_path,
        log_dir=log_dir,
        start_stage=start_stage,
        device=device,
        resume_path=resume_path,
    )

    # Make new files visible to other containers / `modal volume get`.
    volume.commit()


@app.function(
    volumes={"/data": volume},
    timeout=4 * 60 * 60,  # idle timeout; refresh if you need longer sessions
    max_containers=1,
)
@modal.web_server(port=6006, startup_timeout=60)
def tensorboard():
    """Serve TensorBoard at a public Modal URL backed by /data/logs.

    Run via `modal serve modal_train.py`. Modal prints the URL.
    """
    import subprocess

    subprocess.Popen(
        [
            "tensorboard",
            "--logdir",
            "/data/logs",
            "--bind_all",
            "--port",
            "6006",
        ]
    )


@app.local_entrypoint()
def main(
    timesteps: int = 5_000_000,
    save_name: str = "default",
    start_stage: int = 0,
    device: str = "gpu",
    resume_from: str = "",
):
    """`modal run modal_train.py --timesteps 30000000 --save-name run1` etc."""
    run_league.remote(
        timesteps=timesteps,
        save_name=save_name,
        start_stage=start_stage,
        device=device,
        resume_from=resume_from,
    )
