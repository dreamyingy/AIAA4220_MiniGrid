"""Run and record a fixed-budget DoorKey shape-randomization experiment."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (
    "rand_rect_under_8", "rand_rect_axis_8", "rand_rect_axis_9", "rand_rect_near_8",
)
REFERENCES = ("baseline", "rand_size", "prog_size")
SEEDS = (1, 2, 3)
EPISODES = 100
GRIDS = {"6x6": "MiniGrid-DoorKey-6x6-v0", "8x8": "MiniGrid-DoorKey-8x8-v0"}
SPLITS = {
    "screening": (41000, 51000),
    "confirmation": (42000, 52000),
    "official": (20000, 30000),
}
SELECTION_RULE = "Highest screening 8x8 mean over three seeds, then 6x6 mean, then method name."
THREAD_ENV = {
    **os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "",
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def command(arguments, log):
    with log.open("a", encoding="utf-8") as file:
        print(f"{datetime.now(timezone.utc).isoformat()} {arguments}", file=file, flush=True)
        subprocess.run(
            [sys.executable, "-u", *map(str, arguments)], cwd=ROOT, env=THREAD_ENV,
            stdout=file, stderr=subprocess.STDOUT, check=True,
        )


def evaluate(model, config, output, split, grid, start, log):
    phase = "final" if split == "official" else "dev"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        with TemporaryDirectory(dir=output.parent) as directory:
            staged = Path(directory) / output.name
            command(
                ["evaluate.py", "--model", model, "--env-id", GRIDS[grid],
                 "--episodes", EPISODES, "--episode-seed-start", start, "--phase", phase,
                 "--output", staged, "--device", "cpu", "--record-actions"],
                log,
            )
            staged.rename(output)
    rows = read_csv(output)
    if (sorted(int(row["episode_seed"]) for row in rows) != list(range(start, start + EPISODES))
            or any(row["phase"] != phase
                   or row["method"] != config["method"]
                   or int(row["train_seed"]) != config["seed"]
                   or int(row["model_step"]) != config["actual_steps"]
                   or row["env_id"] != GRIDS[grid]
                   or row["success"] not in ("0", "1")
                   or len(row["actions"].split()) != int(row["steps"])
                   for row in rows)):
        raise ValueError(f"Incomplete or inconsistent evaluation: {output}")
    return sum(int(row["success"]) for row in rows) / EPISODES


def evaluate_model(output, reference_dir, method, seed, split):
    candidate = method in CANDIDATES
    directory = output / ("runs" if candidate else "references") / method / f"seed_{seed}"
    log = output / "logs" / f"{method}_seed_{seed}_{split}.console.log"
    model_directory = directory if candidate else reference_dir / method / f"seed_{seed}"
    model = model_directory / "final.zip"
    if candidate and split == "screening" and not model.is_file():
        command(
            ["train.py", "--config", ROOT / "configs" / f"{method}.json", "--seed", seed,
             "--output-dir", model_directory, "--device", "cpu", "--record-actions"],
            log,
        )
    config = json.loads((model_directory / "config.json").read_text())
    if config["method"] != method or config["seed"] != seed or config["actual_steps"] != 1001472:
        raise ValueError(f"Unexpected finished model metadata: {model_directory}")
    rates = {"method": method, "train_seed": seed}
    prefix = "final" if split == "official" else split
    for grid, start in zip(GRIDS, SPLITS[split]):
        rates[f"{grid}_success_rate"] = evaluate(
            model, config, directory / f"{prefix}_{grid}.csv", split, grid, start, log,
        )
    return rates


def run_evaluations(pool, output, reference_dir, stages, errors):
    work = {
        pool.submit(evaluate_model, output, reference_dir, method, seed, split): (split, method, seed)
        for split, methods in stages.items() for method in methods for seed in SEEDS
    }
    results = {split: [] for split in stages}
    for future in as_completed(work):
        split, method, seed = work[future]
        try:
            rates = future.result()
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
            error = {"split": split, "method": method, "train_seed": seed, "error": str(exc)}
            errors.append(error)
            write_json(output / "errors.json", errors)
            print(f"Failed {split} {method} seed={seed}: {exc}", flush=True)
        else:
            results[split].append(rates)
            print(f"Completed {split} {rates}", flush=True)
    for split, rows in results.items():
        rows.sort(key=lambda row: (row["method"], row["train_seed"]))
        write_csv(output / f"{split}_by_seed.csv", rows)
    return results


def summarize(rows):
    results = []
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        result = {"method": method, "n_train_seeds": len(method_rows)}
        for grid in GRIDS:
            values = [row[f"{grid}_success_rate"] for row in method_rows]
            result[f"{grid}_mean"] = statistics.mean(values)
            result[f"{grid}_sample_sd"] = statistics.stdev(values) if len(values) == 3 else None
        results.append(result)
    return results


def index_cases(output):
    """Index every evaluated episode and count all completed training episodes."""
    paths = {
        "periodic": "periodic_episodes.csv", "screening": "screening_*.csv",
        "confirmation": "confirmation_*.csv", "official": "final_*.csv",
    }
    counts = Counter()
    training_counts = []
    with gzip.open(output / "evaluation_cases.csv.gz", "wt", newline="", encoding="utf-8") as file:
        writer = None
        for group in ("runs", "references"):
            for directory in sorted((output / group).glob("*/seed_*")):
                for split, pattern in paths.items():
                    for path in sorted(directory.glob(pattern)):
                        for row in read_csv(path):
                            if writer is None:
                                writer = csv.DictWriter(file, fieldnames=["split", *row])
                                writer.writeheader()
                            writer.writerow({"split": split, **row})
                            counts[split, row["method"], row["env_id"], row["outcome"]] += 1
                monitor = directory / "monitor.csv"
                if monitor.is_file():
                    with monitor.open(newline="", encoding="utf-8") as stream:
                        next(stream)
                        episodes = list(csv.DictReader(stream))
                    config = json.loads((directory / "config.json").read_text())
                    wins = sum(int(row["success"]) for row in episodes)
                    training_counts.append({
                        "method": config["method"], "train_seed": config["seed"],
                        "completed_episodes": len(episodes), "successes": wins,
                        "failures": len(episodes) - wins,
                        "recorded_steps": sum(int(row["l"]) for row in episodes),
                        "actual_steps": config.get("actual_steps"),
                    })
    write_csv(output / "evaluation_case_counts.csv", [
        {"split": split, "method": method, "env_id": env_id, "outcome": outcome, "episodes": count}
        for (split, method, env_id, outcome), count in sorted(counts.items())
    ])
    write_csv(output / "training_case_counts.csv", training_counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    output = args.output.resolve()
    reference_dir = args.reference_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    config_paths = [ROOT / "configs" / f"{method}.json" for method in CANDIDATES]
    configs = [json.loads(path.read_text()) for path in config_paths]
    for config in configs:
        if any(width == height == 8 or width * height >= 64 for width, height in config["train_shapes"]):
            raise ValueError(f"Training shapes must exclude 8x8 and have area below 64: {config['method']}")
    source_paths = config_paths + [
        ROOT / path for path in ("common.py", "train.py", "evaluate.py", "analyze.py", "experiments/size_transfer.py")
    ]
    write_json(output / "protocol.json", {
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths
        },
        "training_seeds": SEEDS, "episode_count_per_seed_and_grid": EPISODES,
        "episode_seed_starts": {split: dict(zip(GRIDS, starts)) for split, starts in SPLITS.items()},
        "selection_rule": SELECTION_RULE, "target_8x8_success_rate": 0.9,
        "candidates": configs, "references": REFERENCES,
    })
    errors = []
    write_json(output / "errors.json", errors)
    print(f"Starting {len(CANDIDATES) * len(SEEDS)} training runs with up to {args.jobs} workers", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            screening = run_evaluations(
                pool, output, reference_dir, {"screening": (*CANDIDATES, *REFERENCES)}, errors,
            )["screening"]
            summary = summarize(screening)
            write_csv(output / "screening_summary.csv", summary)
            eligible = [
                row for row in summary if row["method"] in CANDIDATES and row["n_train_seeds"] == len(SEEDS)
            ]
            ranked = sorted(eligible, key=lambda row: (-row["8x8_mean"], -row["6x6_mean"], row["method"]))
            if not ranked:
                raise RuntimeError("No new candidate completed screening on all three seeds.")
            selection = {"selected": ranked[0]["method"], "screening_ranking": ranked,
                         "locked_at_utc": datetime.now(timezone.utc).isoformat(), "rule": SELECTION_RULE}
            selection_path = output / "selection.json"
            if selection_path.exists():
                locked = json.loads(selection_path.read_text())
                if locked["selected"] != selection["selected"]:
                    raise ValueError("Screening ranking conflicts with the existing selection lock.")
            else:
                write_json(selection_path, selection)
            selected = selection["selected"]
            print(f"Locked selected method before confirmation and official evaluation: {selected}", flush=True)
            assessments = run_evaluations(
                pool, output, reference_dir,
                {"confirmation": (selected, *REFERENCES), "official": (selected,)}, errors,
            )
            for split, rows in assessments.items():
                write_csv(output / f"{split}_summary.csv", summarize(rows))
        if len(assessments["official"]) != len(SEEDS):
            raise RuntimeError("Official evaluation is missing at least one seed.")
        command(
            ["analyze.py", "--input", reference_dir / "baseline", "--compare",
             output / "runs" / selected, "--output", output / "official_comparison"],
            output / "logs" / "analyze.console.log",
        )
    finally:
        index_cases(output)
    print("SIZE TRANSFER EXPERIMENT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
