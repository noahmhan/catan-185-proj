"""Pull TensorBoard logs from one or more Modal volumes.

Workaround for `modal volume get` crashing on Windows when recursing into a
directory (errno 13 on the destination dir). Iterates files via the SDK and
writes them locally one-by-one.

Each "target" is a `volume/run` pair, e.g. `schra-data/run1`. Local files go
to `{dest}/{volume}/{run}/...` so multiple volumes can share one TB root.

Usage:
    python modal_pull_logs.py
        # defaults: board-ppo-data/boardrun1 + schra-data/run1,run2,run3

    python modal_pull_logs.py schra-data/nsteps5 board-ppo-data/boardrun1
        # explicit list of volume/run pairs

    python modal_pull_logs.py --dest C:/tmp/tb schra-data/run1
"""

import argparse
import os

import modal
from modal.volume import FileEntryType

DEFAULT_TARGETS = [
    "board-ppo-data/boardrun1",
    "schra-data/run1",
    "schra-data/run2",
    "schra-data/run3",
]
DEFAULT_DEST = os.path.join(os.path.expanduser("~"), "modal_tb_logs")


def pull_run(vol: modal.Volume, run: str, local_root: str) -> None:
    remote_root = f"logs/{run}"
    os.makedirs(local_root, exist_ok=True)

    entries = list(vol.iterdir(remote_root, recursive=True))
    files = [e for e in entries if e.type == FileEntryType.FILE]
    total_bytes = sum(e.size for e in files)
    print(f"  {len(files)} files, {total_bytes / 1e6:.1f} MB")

    done_bytes = 0
    for i, entry in enumerate(files, 1):
        rel = os.path.relpath(entry.path, remote_root)
        local_path = os.path.join(local_root, rel)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if os.path.exists(local_path) and os.path.getsize(local_path) == entry.size:
            done_bytes += entry.size
            print(f"    [{i}/{len(files)}] skip {rel} (already downloaded)")
            continue

        with open(local_path, "wb") as f:
            for chunk in vol.read_file(entry.path):
                f.write(chunk)
        done_bytes += entry.size
        pct = done_bytes / total_bytes if total_bytes else 1.0
        print(
            f"    [{i}/{len(files)}] {rel} "
            f"({entry.size / 1e6:.2f} MB, {pct:.0%} total)"
        )


def parse_target(target: str) -> tuple[str, str]:
    if "/" not in target:
        raise ValueError(f"target must be 'volume/run', got {target!r}")
    volume, run = target.split("/", 1)
    return volume, run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "targets",
        nargs="*",
        default=DEFAULT_TARGETS,
        help="volume/run pairs, e.g. schra-data/run1",
    )
    ap.add_argument("--dest", default=DEFAULT_DEST)
    args = ap.parse_args()

    print(f"Destination: {args.dest}")
    vol_cache: dict[str, modal.Volume] = {}
    for target in args.targets:
        volume, run = parse_target(target)
        if volume not in vol_cache:
            vol_cache[volume] = modal.Volume.from_name(volume)
        local_root = os.path.join(args.dest, volume, run)
        print(f"[{volume}/{run}] -> {local_root}")
        pull_run(vol_cache[volume], run, local_root)
    print("Done.")


if __name__ == "__main__":
    main()
