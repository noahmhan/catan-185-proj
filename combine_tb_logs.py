"""Combine two TensorBoard event files, dropping overlapping steps from the later run.

Keeps every event from the earlier run, then appends only events from the later run
whose step is strictly greater than the earlier run's max step. Wall times and step
numbers are preserved exactly as they appear in the source files.

Usage:
    python combine_tb_logs.py \
        --earlier C:/Users/noahm/schra_tb_logs/nsteps5/SCHRA_8 \
        --later   C:/Users/noahm/schra_tb_logs/nsteps5/SCHRA_15 \
        --out     C:/Users/noahm/schra_tb_logs/nsteps5/SCHRA_8_15_combined
"""

import argparse
import glob
import os
import struct

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.compat.proto.event_pb2 import Event


def event_files(run_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not files:
        raise FileNotFoundError(f"No event files in {run_dir}")
    return files


def iter_events(run_dir: str):
    for path in event_files(run_dir):
        for raw in EventFileLoader(path).Load():
            yield raw


def max_step(run_dir: str) -> int:
    return max((e.step for e in iter_events(run_dir) if e.HasField("summary")), default=-1)


_MASK_DELTA = 0xA282EAD8


def _u32(x: int) -> int:
    return x & 0xFFFFFFFF


def _crc32c(data: bytes) -> int:
    # CRC-32C (Castagnoli) computed via the table-free reference implementation.
    # Matches tf.io.tf_record / tensorboard's expected checksum.
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 & -(crc & 1))
            crc &= 0xFFFFFFFF
    return crc ^ 0xFFFFFFFF


def _masked_crc(data: bytes) -> int:
    c = _crc32c(data)
    return _u32(((c >> 15) | (c << 17)) + _MASK_DELTA)


def _write_record(fp, payload: bytes) -> None:
    length = struct.pack("<Q", len(payload))
    fp.write(length)
    fp.write(struct.pack("<I", _masked_crc(length)))
    fp.write(payload)
    fp.write(struct.pack("<I", _masked_crc(payload)))


def combine(earlier: str, later: str, out_dir: str) -> tuple[int, int, int]:
    cutoff = max_step(earlier)
    os.makedirs(out_dir, exist_ok=True)
    # Mimic the standard tfevents filename so EventAccumulator / TensorBoard pick it up.
    template = os.path.basename(event_files(earlier)[0])
    out_path = os.path.join(out_dir, template)

    kept_earlier = kept_later = dropped = 0
    with open(out_path, "wb") as fp:
        for event in iter_events(earlier):
            _write_record(fp, event.SerializeToString())
            kept_earlier += 1
        for event in iter_events(later):
            if event.HasField("summary") and event.step <= cutoff:
                dropped += 1
                continue
            _write_record(fp, event.SerializeToString())
            kept_later += 1

    return kept_earlier, kept_later, dropped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--earlier", required=True, help="Run directory whose events come first.")
    p.add_argument("--later", required=True, help="Run directory continued from the earlier one.")
    p.add_argument("--out", required=True, help="Output directory for the combined event file.")
    args = p.parse_args()

    cutoff = max_step(args.earlier)
    print(f"earlier max step: {cutoff}")
    kept_e, kept_l, dropped = combine(args.earlier, args.later, args.out)
    print(f"kept {kept_e} events from earlier")
    print(f"kept {kept_l} events from later, dropped {dropped} overlapping events")
    print(f"wrote {kept_e + kept_l} events to {args.out}")


if __name__ == "__main__":
    main()
