"""Summarize the three training seeds and plot periodic and final results."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ENVS = {"6x6": "MiniGrid-DoorKey-6x6-v0", "8x8": "MiniGrid-DoorKey-8x8-v0"}
METRICS = ("success_rate", "mean_return", "median_success_steps")


def read_episodes(path, config, seed, env_id, phase, model_steps, seed_start):
    """Reject incomplete or mixed evaluations before computing statistics."""
    frame = pd.read_csv(path)
    count = config["eval_episodes"]
    expected_seeds = set(range(seed_start, seed_start + count))
    if not (
        len(frame) == len(model_steps) * count
        and set(frame["model_step"]) == set(model_steps)
        and frame["train_seed"].eq(seed).all()
        and frame["method"].eq(config["method"]).all()
        and frame["env_id"].eq(env_id).all()
        and frame["phase"].eq(phase).all()
        and frame["success"].isin([0, 1]).all()
        and not frame.duplicated(["model_step", "episode_seed"]).any()
        and all(set(group["episode_seed"]) == expected_seeds
                for _, group in frame.groupby("model_step"))
    ):
        raise ValueError(f"Incomplete or inconsistent evaluation: {path}")
    if not np.isfinite(frame[["return", "steps", "elapsed_seconds"]]).all().all():
        raise ValueError(f"Non-finite episode results: {path}")
    return frame


def episode_metrics(frame):
    successful_steps = frame.loc[frame["success"].eq(1), "steps"]
    return {
        "success_rate": float(frame["success"].mean()),
        "mean_return": float(frame["return"].mean()),
        "median_success_steps": float(successful_steps.median()),
    }


def mean_sd(values):
    """Only report a three-seed aggregate when all three values exist."""
    values = pd.Series(values, dtype=float)
    valid_n = int(values.notna().sum())
    if len(values) != 3 or valid_n != 3:
        return np.nan, np.nan, valid_n
    return float(values.mean()), float(values.std(ddof=1)), valid_n


def aggregate(frame, metrics):
    result = {}
    for metric in metrics:
        mean, sd, valid_n = mean_sd(frame[metric])
        result.update({f"{metric}_mean": mean, f"{metric}_std": sd,
                       f"{metric}_valid_n": valid_n})
    return result


def collect_results(input_dir):
    with (input_dir / "seed_1" / "config.json").open() as file:
        config = json.load(file)
    if config["seeds"] != [1, 2, 3] or config["total_steps"] % config["eval_freq"]:
        raise ValueError("Expected seeds [1, 2, 3] and a complete periodic evaluation grid.")
    budget = config["total_steps"]
    grid = list(range(0, budget + 1, config["eval_freq"]))
    rollout_size = config["n_envs"] * config["ppo"]["n_steps"]
    actual_steps = ((budget + rollout_size - 1) // rollout_size) * rollout_size
    fixed_keys = ("method", "env_id", "algo", "seeds", "total_steps", "n_envs",
                  "ppo", "features_dim", "eval_freq", "eval_episodes",
                  "periodic_seed_start", "final_6x6_seed_start", "final_8x8_seed_start")
    periodic_rows, seed_rows, final_rows = [], [], []
    for seed in config["seeds"]:
        directory = input_dir / f"seed_{seed}"
        with (directory / "config.json").open() as file:
            run = json.load(file)
        if any(run[key] != config[key] for key in fixed_keys) or run["actual_steps"] != actual_steps:
            raise ValueError(f"Training configurations or actual budgets differ: {directory}")
        periodic = read_episodes(directory / "periodic_episodes.csv", run, seed,
                                 ENVS["6x6"], "periodic", grid, run["periodic_seed_start"])
        curve = []
        for step, episodes in periodic.groupby("model_step", sort=True):
            row = {"method": run["method"], "train_seed": seed, "model_step": int(step),
                   **episode_metrics(episodes)}
            periodic_rows.append(row)
            curve.append(row["success_rate"])
        reached = [step for step, success in zip(grid, curve) if success >= 0.8]
        budget_label = "1M" if budget == 1_000_000 else str(budget)
        seed_row = {
            "method": run["method"], "train_seed": seed,
            "nominal_steps": budget, "actual_steps": actual_steps,
            "normalized_auc": float(np.trapezoid(curve, grid) / budget),
            "t80": reached[0] if reached else np.nan,
            "t80_status": "reached" if reached else f"not reached (>{budget_label})",
            "learn_seconds": run["timing"]["learn_seconds"],
            "periodic_eval_seconds": run["timing"]["periodic_eval_seconds"],
            "final_eval_seconds": 0.0,
        }
        for size, env_id in ENVS.items():
            final = read_episodes(directory / f"final_{size}.csv", run, seed, env_id,
                                  "final", [actual_steps], run[f"final_{size}_seed_start"])
            metrics = episode_metrics(final)
            final_rows.append({"method": run["method"], "train_seed": seed,
                               "env_id": env_id, **metrics})
            seed_row.update({f"final_{size}_{name}": value for name, value in metrics.items()})
            seed_row["final_eval_seconds"] += float(final["elapsed_seconds"].sum())
        seed_rows.append(seed_row)
    return config, pd.DataFrame(periodic_rows), pd.DataFrame(seed_rows), pd.DataFrame(final_rows)


def write_summaries(config, periodic, seeds, final, output):
    periodic_summary = pd.DataFrame([
        {"method": config["method"], "model_step": step, **aggregate(group, METRICS)}
        for step, group in periodic.groupby("model_step", sort=True)
    ])
    final_summary = pd.DataFrame([
        {"method": config["method"], "env_id": env_id,
         "episodes_per_seed": config["eval_episodes"], **aggregate(group, METRICS)}
        for env_id, group in final.groupby("env_id", sort=True)
    ])
    training_summary = pd.DataFrame([{
        "method": config["method"], "nominal_steps": config["total_steps"],
        "actual_steps": int(seeds["actual_steps"].iloc[0]), "n_seeds": 3,
        "t80_reached_seeds": int(seeds["t80"].notna().sum()),
        **aggregate(seeds, ("normalized_auc", "t80", "learn_seconds",
                            "periodic_eval_seconds", "final_eval_seconds")),
    }])
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("periodic_by_seed", periodic), ("periodic_summary", periodic_summary),
                        ("seed_metrics", seeds), ("final_summary", final_summary),
                        ("training_summary", training_summary)):
        frame.to_csv(output / f"{name}.csv", index=False, na_rep="NaN")
    return periodic_summary, final_summary


def save_figure(fig, directory, name):
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_results(config, periodic, periodic_summary, final, final_summary, output):
    directory = output / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for seed, group in periodic.groupby("train_seed"):
        ax.plot(group["model_step"], group["success_rate"], lw=1.5, alpha=0.9,
                label=f"Seed {seed}")
    x = periodic_summary["model_step"].to_numpy()
    mean = periodic_summary["success_rate_mean"].to_numpy()
    sd = periodic_summary["success_rate_std"].to_numpy()
    ax.plot(x, mean, color="black", ls="--", lw=2, label="Three-seed mean", zorder=3)
    ax.fill_between(x, mean - sd, mean + sd, color="gray", alpha=0.18,
                    label="± sample SD (ddof=1)", zorder=0)
    ax.axhline(0.8, color="tab:purple", ls=":", lw=1.4, label="80% success")
    ax.set(xlabel="Training environment steps", ylabel="Success rate (SR)",
           xticks=[0, 200_000, 400_000, 600_000, 800_000, 1_000_000],
           xticklabels=["0", "200k", "400k", "600k", "800k", "1M"],
           xlim=(0, config["total_steps"]),
           ylim=(min(0, float((mean - sd).min())) - 0.03,
                 max(1, float((mean + sd).max())) + 0.03),
           yticks=np.linspace(0, 1, 6),
           title="6×6 Learning Curve")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, directory, "learning_curve")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    lower = final_summary["success_rate_mean"] - final_summary["success_rate_std"]
    upper = final_summary["success_rate_mean"] + final_summary["success_rate_std"]
    limits = (min(0, float(lower.min())), max(1, float(upper.max())) + 0.18)
    for ax, (size, env_id) in zip(axes, ENVS.items()):
        group = final.loc[final["env_id"].eq(env_id)].sort_values("train_seed")
        summary = final_summary.loc[final_summary["env_id"].eq(env_id)].iloc[0]
        mean, sd = summary["success_rate_mean"], summary["success_rate_std"]
        bars = ax.bar(group["train_seed"], group["success_rate"], width=0.6,
                      color=["tab:blue", "tab:orange", "tab:green"], zorder=2)
        ax.bar_label(bars, labels=[f"{rate:.0%}" for rate in group["success_rate"]],
                     padding=4, fontsize=10)
        ax.axhline(mean, color="black", ls="--", lw=1.4, label="Three-seed mean", zorder=3)
        ax.axhspan(mean - sd, mean + sd, color="gray", alpha=0.18,
                   label="± sample SD (ddof=1)", zorder=0)
        ax.text(0.5, 0.96, f"Mean ± sample SD: {mean:.2%} ± {sd * 100:.2f} pp",
                transform=ax.transAxes, ha="center", va="top", fontsize=10)
        title = "6×6" if size == "6x6" else "8×8 (Zero-shot Transfer)"
        ax.set(title=title, xlabel="Training seed", xticks=group["train_seed"],
               xticklabels=[f"Seed {seed}" for seed in group["train_seed"]],
               xlim=(0.4, 3.6), ylim=limits, yticks=np.linspace(0, 1, 6))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Success rate (SR)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Final 6×6 and 8×8 comparison with uncertainty", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    save_figure(fig, directory, "final_success")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory containing seed_1, seed_2, seed_3")
    parser.add_argument("--output", type=Path, required=True, help="Directory for summary CSVs and figures")
    args = parser.parse_args()
    config, periodic, seeds, final = collect_results(args.input)
    periodic_summary, final_summary = write_summaries(config, periodic, seeds, final, args.output)
    plot_results(config, periodic, periodic_summary, final, final_summary, args.output)
    print(f"Wrote three-seed summaries and figures to {args.output}")
    print(f"T80 reached: {seeds['t80'].notna().sum()}/3 seeds")


if __name__ == "__main__":
    main()
