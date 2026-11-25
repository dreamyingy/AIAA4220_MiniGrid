"""Screen and verify fixed-budget DoorKey runs, with explicit holdout stages."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
from uuid import uuid4
from zipfile import is_zipfile

from size_transfer import EPISODES, GRIDS, ROOT, SEEDS, THREAD_ENV, read_csv


SPLITS = {
    "screen": ("screening", "dev", (43000, 53000)),
    "confirm": ("confirmation", "dev", (44000, 54000)),
    "official": ("final", "final", (20000, 30000)),
}
TARGET = 0.9
ACTUAL_STEPS = 1_001_472
METHOD_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")
FIXED_CONFIG = {
    "env_id": "MiniGrid-DoorKey-6x6-v0",
    "seeds": [1, 2, 3], "n_envs": 1, "total_steps": 1_000_000,
    "features_dim": 128, "torch_threads": 1,
    "eval_freq": 50_000, "eval_episodes": 100, "periodic_seed_start": 10000,
    "final_6x6_seed_start": 20000, "final_8x8_seed_start": 30000,
}


def now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_once(path, data):
    """Create immutable metadata, or verify the previously created version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if load_json(path) != data:
            raise ValueError(f"Existing metadata disagrees with this request: {path}")
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")


def replace_csv(path, rows):
    if not rows:
        return
    with NamedTemporaryFile(mode="w", dir=path.parent, newline="", encoding="utf-8",
                            delete=False) as stream:
        staged = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    staged.replace(path)


def parse_config(config_name):
    path = Path(config_name)
    if path.suffix != ".json":
        path = ROOT / "configs" / f"{config_name}.json"
    config = load_json(path)
    method = config["method"]
    if not METHOD_NAME.fullmatch(method):
        raise ValueError(f"Invalid method name: {method!r}")
    for field, value in FIXED_CONFIG.items():
        if config.get(field) != value:
            raise ValueError(f"{method}: expected {field}={value!r}")
    if config["ppo"]["n_steps"] != 2048:
        raise ValueError(f"{method}: expected PPO n_steps=2048")
    sizes = list(config.get("train_sizes") or [])
    shapes = list(config.get("train_shapes") or [])
    for _, schedule_sizes in config.get("size_schedule") or []:
        for size in schedule_sizes:
            if isinstance(size, (tuple, list)):
                shapes.append(size)
            else:
                sizes.append(size)
    for field in ("shape_schedule", "train_shape_schedule"):
        for _, schedule_shapes in config.get(field) or []:
            shapes = [*shapes, *schedule_shapes]
    if any(int(size) == 8 for size in sizes) or any(tuple(map(int, shape)) == (8, 8) for shape in shapes):
        raise ValueError(f"{method}: training includes the held-out 8x8 shape")
    return config


def model_dir(root, method, seed):
    return root / "runs" / method / f"seed_{seed}"


def saved_config(root, method):
    if not METHOD_NAME.fullmatch(method):
        raise ValueError(f"Invalid method name: {method!r}")
    path = root / "configs" / f"{method}.json"
    return load_json(path)


def require_match(path, config, seed):
    saved = load_json(path / "config.json")
    for field, value in {**config, "seed": seed, "device": "cpu",
                         "record_actions": True, "actual_steps": ACTUAL_STEPS}.items():
        if saved.get(field) != value:
            raise ValueError(f"{path}: expected {field}={value!r}; found {saved.get(field)!r}")
    return saved


def command(arguments, log, batch_lock):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        print(f"{now()} {arguments}", file=stream, flush=True)
        subprocess.run(
            [sys.executable, "-u", *map(str, arguments)], cwd=ROOT, env=THREAD_ENV,
            stdout=stream, stderr=subprocess.STDOUT, check=True,
            pass_fds=(batch_lock,),
        )


def train(root, config, seed, batch_lock):
    method = config["method"]
    run = model_dir(root, method, seed)
    model = run / "final.zip"
    config_path = run / "config.json"
    if is_zipfile(model) and config_path.is_file():
        try:
            saved = load_json(config_path)
        except json.JSONDecodeError:
            saved = {}
        if saved.get("actual_steps") == ACTUAL_STEPS:
            require_match(run, config, seed)
            return
    if run.exists():
        attempt = root / "attempts" / method / f"seed_{seed}_{uuid4().hex}"
        attempt.parent.mkdir(parents=True, exist_ok=True)
        run.rename(attempt)
        print(f"Preserved incomplete run {attempt}", flush=True)
    command(
        ["train.py", "--config", root / "configs" / f"{method}.json", "--seed", seed,
         "--output-dir", run, "--device", "cpu", "--record-actions"],
        root / "logs" / f"{method}_seed_{seed}_train.log", batch_lock,
    )
    require_match(run, config, seed)
    if not model.is_file():
        raise FileNotFoundError(model)


def rate(path, method, seed, grid, start, phase):
    rows = read_csv(path)
    if sorted(int(row["episode_seed"]) for row in rows) != list(range(start, start + EPISODES)):
        raise ValueError(f"Unexpected eval seed set: {path}")
    for row in rows:
        if (row["method"] != method or int(row["train_seed"]) != seed
                or int(row["model_step"]) != ACTUAL_STEPS or row["env_id"] != GRIDS[grid]
                or row["phase"] != phase or row["success"] not in ("0", "1")
                or len(row["actions"].split()) != int(row["steps"])):
            raise ValueError(f"Unexpected episode metadata: {path}")
    return statistics.mean(int(row["success"]) for row in rows)


def evaluate(root, config, seed, stage, batch_lock):
    method = config["method"]
    run = model_dir(root, method, seed)
    if stage == "screen":
        train(root, config, seed, batch_lock)
    require_match(run, config, seed)
    prefix, phase, starts = SPLITS[stage]
    for grid, start in zip(GRIDS, starts):
        output = run / f"{prefix}_{grid}.csv"
        if not output.exists():
            with TemporaryDirectory(dir=run) as temporary:
                staged = Path(temporary) / output.name
                command(
                    ["evaluate.py", "--model", run / "final.zip", "--env-id", GRIDS[grid],
                     "--episodes", EPISODES, "--episode-seed-start", start, "--phase", phase,
                     "--output", staged, "--device", "cpu", "--record-actions"],
                    root / "logs" / f"{method}_seed_{seed}_{stage}.log", batch_lock,
                )
                rate(staged, method, seed, grid, start, phase)
                staged.rename(output)
        rate(output, method, seed, grid, start, phase)


def method_rates(root, method, stage):
    prefix, phase, starts = SPLITS[stage]
    scores = []
    for seed in SEEDS:
        run = model_dir(root, method, seed)
        paths = [run / f"{prefix}_{grid}.csv" for grid in GRIDS]
        if all(path.is_file() for path in paths):
            scores.append({
                "method": method, "train_seed": seed,
                **{f"{grid}_success_rate": rate(path, method, seed, grid, start, phase)
                   for grid, start, path in zip(GRIDS, starts, paths)},
            })
    return scores


def summary(scores):
    result = {"method": scores[0]["method"], "n_train_seeds": len(scores)}
    for grid in GRIDS:
        values = [row[f"{grid}_success_rate"] for row in scores]
        result[f"{grid}_mean"] = statistics.mean(values)
        result[f"{grid}_sample_sd"] = statistics.stdev(values) if len(values) == len(SEEDS) else None
    return result


def write_summaries(root):
    methods = [path.stem for path in sorted((root / "configs").glob("*.json"))]
    for stage, (prefix, _, _) in SPLITS.items():
        by_method = [method_rates(root, method, stage) for method in methods]
        by_method = [scores for scores in by_method if scores]
        summaries = sorted(map(summary, by_method), key=lambda row: row["8x8_mean"], reverse=True)
        label = "official" if stage == "official" else prefix
        replace_csv(root / f"{label}_summary.csv", summaries)
        replace_csv(root / f"{label}_by_seed.csv", [score for scores in by_method for score in scores])


def require_target(root, method, stages, allow_below_target):
    evidence = {}
    for stage in stages:
        scores = method_rates(root, method, stage)
        if len(scores) != len(SEEDS):
            raise ValueError(f"{method}: {stage} requires all three complete training seeds")
        result = summary(scores)
        if not allow_below_target and result["8x8_mean"] < TARGET:
            raise ValueError(f"{method}: {stage} 8x8 mean {result['8x8_mean']:.1%} is below {TARGET:.0%}")
        evidence[stage] = result
    return evidence


def run_batch(root, configs, seeds, stage, jobs, batch_lock):
    errors = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        work = {
            pool.submit(evaluate, root, config, seed, stage, batch_lock): (config["method"], seed)
            for config in configs for seed in seeds
        }
        for future in as_completed(work):
            method, seed = work[future]
            try:
                future.result()
            except Exception as error:
                row = {"time": now(), "stage": stage, "method": method, "seed": seed, "error": str(error)}
                with (root / "errors.jsonl").open("a", encoding="utf-8") as stream:
                    print(json.dumps(row), file=stream)
                errors.append(row)
                print(f"Failed {stage} {method} seed {seed}: {error}", flush=True)
            else:
                print(f"Complete {stage} {method} seed {seed}", flush=True)
    return errors


def archive_training(root, runs):
    counts = []
    with NamedTemporaryFile(dir=root, delete=False) as temporary:
        staged = Path(temporary.name)
    fields = [
        "method", "train_seed", "attempt", "episode_index", "r", "l", "t",
        "layout_seed", "layout_rng_state", "train_width", "train_height", "success", "actions",
    ]
    with gzip.open(staged, "wt", encoding="utf-8", newline="") as archive:
        writer = csv.DictWriter(archive, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for run in runs:
            monitor = run / "monitor.csv"
            if not monitor.is_file():
                continue
            try:
                config = load_json(run / "config.json")
            except (OSError, json.JSONDecodeError):
                config = {"method": run.parent.name, "seed": int(run.name.split("_")[1])}
            identifier = {"method": config["method"], "train_seed": config["seed"],
                          "attempt": str(run.relative_to(root))}
            outcomes = Counter()
            steps = 0
            with monitor.open(encoding="utf-8", newline="") as stream:
                next(stream, None)
                for index, row in enumerate(csv.DictReader(stream), 1):
                    if None in row.values():
                        continue  # A terminated writer can leave a partial final line.
                    writer.writerow({**identifier, "episode_index": index, **row})
                    outcomes[row.get("success", "")] += 1
                    steps += int(row["l"])
            counts.append({
                **identifier, "completed_episodes": outcomes.total(),
                "successes": outcomes["1"], "failures": outcomes["0"],
                "success_unknown": outcomes[""], "recorded_steps": steps,
                "actual_steps": config.get("actual_steps"),
            })
    staged.replace(root / "training_cases.csv.gz")
    replace_csv(root / "training_case_counts.csv", counts)


def archive_evaluations(root, runs):
    fields = [
        "split", "attempt", "method", "train_seed", "model_step", "phase", "env_id",
        "episode_seed", "success", "return", "steps", "terminated", "truncated",
        "key_step", "door_step", "goal_step", "elapsed_seconds", "outcome", "actions",
    ]
    patterns = {"periodic": "periodic_episodes.csv"}
    patterns.update({
        "official" if stage == "official" else prefix: f"{prefix}_*.csv"
        for stage, (prefix, _, _) in SPLITS.items()
    })
    counts = Counter()
    with NamedTemporaryFile(dir=root, delete=False) as temporary:
        staged = Path(temporary.name)
    with gzip.open(staged, "wt", encoding="utf-8", newline="") as archive:
        writer = csv.DictWriter(archive, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for run in runs:
            for split, pattern in patterns.items():
                for path in sorted(run.glob(pattern)):
                    for row in read_csv(path):
                        if None in row.values():
                            continue
                        writer.writerow({"split": split, "attempt": str(run.relative_to(root)), **row})
                        counts[split, row["method"], row["env_id"], row["outcome"]] += 1
    staged.replace(root / "evaluation_cases.csv.gz")
    replace_csv(root / "evaluation_case_counts.csv", [
        {"split": split, "method": method, "env_id": env_id, "outcome": outcome, "episodes": count}
        for (split, method, env_id, outcome), count in sorted(counts.items())
    ])


def archive_cases(root):
    paths = sorted((root / "runs").glob("*/seed_*")) + sorted((root / "attempts").glob("*/seed_*"))
    archive_training(root, paths)
    archive_evaluations(root, paths)
    write_summaries(root)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*SPLITS, "lock", "archive"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", default=[])
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--allow-below-target", action="store_true",
                        help="Permit an explicitly labeled diagnostic confirmation or lock.")
    args = parser.parse_args()
    if args.jobs <= 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Jobs must be positive and seeds distinct.")
    if args.stage in ("screen", "confirm", "lock") and not args.configs:
        parser.error(f"{args.stage} requires --configs.")
    if args.stage == "lock" and len(args.configs) != 1:
        parser.error("Lock exactly one candidate.")
    if args.stage in ("archive", "official") and args.configs:
        parser.error(f"{args.stage} reads configs from the batch.")
    return args


def main():
    args = parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".batch.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit(f"Batch already active: {root}") from error
        json_once(root / "protocol.json", {
            "splits": {stage: dict(zip(GRIDS, data[2])) for stage, data in SPLITS.items()},
            "training_seeds": list(SEEDS), "training_steps": 1_000_000,
            "actual_steps": ACTUAL_STEPS, "episodes_per_grid_seed": EPISODES,
            "target_8x8_success_rate": TARGET,
        })
        if args.stage == "archive":
            archive_cases(root)
            return

        if args.stage == "screen":
            configs = [parse_config(name) for name in args.configs]
            if len({config["method"] for config in configs}) != len(configs):
                raise ValueError("Method names must be unique within a batch.")
            for config in configs:
                json_once(root / "configs" / f"{config['method']}.json", config)
                files = (
                    "train.py", "evaluate.py", "common.py", "analyze.py", "exploration.py",
                    "progress_potential.py", "experiments/target90.py", "experiments/size_transfer.py",
                )
                evidence = {
                    "source_sha256_at_screen": {
                        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                        for name in files if (ROOT / name).is_file()
                    },
                }
                manifest = root / "manifests" / f"{config['method']}.json"
                if not manifest.exists():
                    json_once(manifest, evidence)
        elif args.stage in ("confirm", "lock"):
            configs = [saved_config(root, name) for name in args.configs]
        else:
            selection = load_json(root / "selection.json")
            configs = [saved_config(root, selection["method"])]
            require_target(root, selection["method"], ("screen", "confirm"), selection["diagnostic"])
            if set(args.seeds) != set(SEEDS):
                raise ValueError("Official evaluation requires all three seeds.")

        if args.stage == "lock":
            method = configs[0]["method"]
            evidence = require_target(root, method, ("screen", "confirm"), args.allow_below_target)
            path = root / "selection.json"
            if path.exists():
                prior = load_json(path)
                if prior["method"] != method or prior["dev_evidence"] != evidence:
                    raise ValueError("Batch already has a different selection lock.")
            else:
                json_once(path, {"method": method, "locked_at_utc": now(), "dev_evidence": evidence,
                                 "diagnostic": any(row["8x8_mean"] < TARGET for row in evidence.values())})
            print(f"Locked {method}", flush=True)
            return
        if args.stage == "confirm":
            for config in configs:
                require_target(root, config["method"], ("screen",), args.allow_below_target)

        try:
            errors = run_batch(root, configs, args.seeds, args.stage, args.jobs, lock.fileno())
            if errors:
                raise RuntimeError(f"{len(errors)} jobs failed; see {root / 'errors.jsonl'}")
            if args.stage == "official":
                analysis = root / "official_comparison"
                if not (analysis / "comparison_table.csv").exists():
                    with TemporaryDirectory(dir=root) as directory:
                        staged = Path(directory) / analysis.name
                        command(
                            ["analyze.py", "--input", args.reference_dir.resolve() / "baseline",
                             "--compare", root / "runs" / configs[0]["method"], "--output", staged],
                            root / "logs" / "analyze.log", lock.fileno(),
                        )
                        staged.rename(analysis)
        finally:
            archive_cases(root)
        if args.stage == "official":
            method = configs[0]["method"]
            final = summary(method_rates(root, method, "official"))
            passed = final["8x8_mean"] >= TARGET
            json_once(root / "official_verdict.json", {
                "method": method, "target_8x8_success_rate": TARGET,
                "official_8x8_success_rate": final["8x8_mean"], "target_met": passed,
            })
            print(f"Official 8x8 mean {final['8x8_mean']:.1%}; target met: {passed}", flush=True)
        print(f"TARGET90 {args.stage.upper()} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
