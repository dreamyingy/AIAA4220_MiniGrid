"""Reproduce the official-seed assessment of the frozen inference controller."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import sys

import episodic_memory as memory


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads((args.output / "official_selection.json").read_text())
    if (lock["method"] != "baseline" or lock["controller"] != "count"
            or lock["train_seeds"] != [1, 2, 3] or lock["episodes_per_seed"] != 100
            or lock["confirmation_8x8_mean"] < 0.90
            or lock["official_first_episode_seeds"] != {"6": 20000, "8": 30000}
            or lock["settings"] != asdict(memory.SETTINGS["count"])
            or sha256(Path(memory.__file__)) != lock["controller_script_sha256"]):
        raise ValueError("The frozen model, evidence or protocol differs from the selection lock.")
    for seed in lock["train_seeds"]:
        model = memory.ROOT / "results" / "baseline" / f"seed_{seed}" / "final.zip"
        if sha256(model) != lock["checkpoint_sha256"][str(seed)]:
            raise ValueError(f"Checkpoint SHA-256 mismatch for seed {seed}.")
    for split in ("screening", "confirmation"):
        summary = args.output / split / "summary.csv"
        if sha256(summary) != lock[f"{split}_summary_sha256"]:
            raise ValueError(f"Evidence summary SHA-256 mismatch for {split}.")
    memory.SPLITS["official"] = (20000, 30000)
    sys.argv[1:] = [
        "--output", str(args.output), "--split", "official", "--methods", "baseline", "--controllers", "count",
    ]
    memory.main()


if __name__ == "__main__":
    main()
