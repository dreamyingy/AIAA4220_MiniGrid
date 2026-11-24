"""Summarize three-seed runs and generate single-method or controlled comparison plots."""

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
PERIODIC_METRICS = ("success_rate", "mean_return", "median_success_steps")
FINAL_METRICS = (
    "success_rate",
    "mean_return",
    "median_success_steps",
    "fail_no_key_rate",
    "fail_key_no_door_rate",
    "fail_door_no_goal_rate",
    "post_key_fail_rate",
)
METHOD_LABELS = {
    "baseline": "Baseline (6×6, k=1)",
    "frame_stack": "Frame Stack (k=4)",
    "one_hot": "One-Hot CNN (20-ch)",
    "rand_size": "Size Rand ({5,6,7})",
    "rand_layout": "Layout Rand ({5,6,7}+Goal)",
    "rand_size_fs4": "Size Rand + Frame Stack",
}
SHORT_TICK_LABELS = {
    "baseline": "Baseline\n(6×6, k=1)",
    "frame_stack": "Frame Stack\n(6×6, k=4)",
    "one_hot": "One-Hot CNN\n(6×6, 20-ch)",
    "rand_size": "Size Rand\n({5,6,7}, k=1)",
    "rand_layout": "Layout Rand\n({5,6,7}+Goal)",
    "rand_size_fs4": "Size Rand + FS\n({5,6,7}, k=4)",
}


def display_name(method):
    return METHOD_LABELS.get(method, method.replace("_", " ").title())


def tick_name(method):
    return SHORT_TICK_LABELS.get(method, display_name(method))


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


def episode_metrics(frame, include_failure_stages=False):
    successful_steps = frame.loc[frame["success"].eq(1), "steps"]
    metrics = {
        "success_rate": float(frame["success"].mean()),
        "mean_return": float(frame["return"].mean()),
        "median_success_steps": float(successful_steps.median()),
    }
    if include_failure_stages:
        failed = frame["success"].eq(0)
        has_key = frame["key_step"].notna()
        has_door = frame["door_step"].notna()
        metrics.update({
            "fail_no_key_rate": float((failed & ~has_key).mean()),
            "fail_key_no_door_rate": float((failed & has_key & ~has_door).mean()),
            "fail_door_no_goal_rate": float((failed & has_door).mean()),
            "post_key_fail_rate": float((failed & has_key).mean()),
        })
    return metrics


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
    with (input_dir / "seed_1" / "config.json").open(encoding="utf-8") as file:
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
    frame_stack = int(config.get("frame_stack", 1))
    representation = str(config.get("representation", "raw"))
    train_sizes = config.get("train_sizes", None)
    randomize_goal = bool(config.get("randomize_goal", False))
    periodic_rows, seed_rows, final_rows = [], [], []
    for seed in config["seeds"]:
        directory = input_dir / f"seed_{seed}"
        with (directory / "config.json").open(encoding="utf-8") as file:
            run = json.load(file)
        if (any(run[key] != config[key] for key in fixed_keys)
                or int(run.get("frame_stack", 1)) != frame_stack
                or str(run.get("representation", "raw")) != representation
                or run.get("train_sizes", None) != train_sizes
                or bool(run.get("randomize_goal", False)) != randomize_goal
                or run["actual_steps"] != actual_steps):
            raise ValueError(f"Training configurations or actual budgets differ: {directory}")
        periodic = read_episodes(directory / "periodic_episodes.csv", run, seed,
                                 ENVS["6x6"], "periodic", grid, run["periodic_seed_start"])
        curve = []
        for step, episodes in periodic.groupby("model_step", sort=True):
            row = {"method": run["method"], "train_seed": seed, "model_step": int(step),
                   **episode_metrics(episodes, include_failure_stages=False)}
            periodic_rows.append(row)
            curve.append(row["success_rate"])
        reached = [step for step, success in zip(grid, curve) if success >= 0.8]
        budget_label = "1M" if budget == 1_000_000 else str(budget)
        seed_row = {
            "method": run["method"],
            "frame_stack": frame_stack,
            "train_seed": seed,
            "nominal_steps": budget,
            "actual_steps": actual_steps,
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
            metrics = episode_metrics(final, include_failure_stages=True)
            final_rows.append({"method": run["method"], "train_seed": seed,
                               "env_id": env_id, **metrics})
            seed_row.update({f"final_{size}_{name}": value for name, value in metrics.items()})
            seed_row["final_eval_seconds"] += float(final["elapsed_seconds"].sum())
        seed_rows.append(seed_row)
    return config, pd.DataFrame(periodic_rows), pd.DataFrame(seed_rows), pd.DataFrame(final_rows)


def summarize_method(config, periodic, seeds, final):
    periodic_summary = pd.DataFrame([
        {"method": config["method"], "model_step": step, **aggregate(group, PERIODIC_METRICS)}
        for step, group in periodic.groupby("model_step", sort=True)
    ])
    final_summary = pd.DataFrame([
        {"method": config["method"], "env_id": env_id,
         "episodes_per_seed": config["eval_episodes"], **aggregate(group, FINAL_METRICS)}
        for env_id, group in final.groupby("env_id", sort=True)
    ])
    training_summary = pd.DataFrame([{
        "method": config["method"],
        "frame_stack": int(config.get("frame_stack", 1)),
        "nominal_steps": config["total_steps"],
        "actual_steps": int(seeds["actual_steps"].iloc[0]),
        "n_seeds": 3,
        "t80_reached_seeds": int(seeds["t80"].notna().sum()),
        **aggregate(seeds, ("normalized_auc", "t80", "learn_seconds",
                            "periodic_eval_seconds", "final_eval_seconds")),
    }])
    return periodic_summary, final_summary, training_summary


def write_single_summaries(config, periodic, seeds, final, output):
    periodic_summary, final_summary, training_summary = summarize_method(config, periodic, seeds, final)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("periodic_by_seed", periodic), ("periodic_summary", periodic_summary),
                        ("seed_metrics", seeds), ("final_summary", final_summary),
                        ("training_summary", training_summary)):
        frame.to_csv(output / f"{name}.csv", index=False, na_rep="NaN")
    return periodic_summary, final_summary


def verify_matched_protocol(base_cfg, mod_cfg):
    """Ensure baseline and modification share identical task, budget, seeds, and eval protocol."""
    matched_keys = (
        "env_id", "algo", "seeds", "total_steps", "n_envs", "ppo", "features_dim",
        "eval_freq", "eval_episodes", "periodic_seed_start",
        "final_6x6_seed_start", "final_8x8_seed_start",
    )
    mismatches = {
        key: {"baseline": base_cfg.get(key), "modified": mod_cfg.get(key)}
        for key in matched_keys
        if base_cfg.get(key) != mod_cfg.get(key)
    }
    if mismatches:
        raise ValueError(f"Unfair comparison — protocol mismatch between methods: {mismatches}")
    if base_cfg["method"] == mod_cfg["method"]:
        raise ValueError(f"Baseline and comparison directories share the same method name: {base_cfg['method']}")
    isolated_change = {}
    for field, default in (
        ("frame_stack", 1),
        ("representation", "raw"),
        ("train_sizes", None),
        ("randomize_goal", False),
    ):
        b_val = base_cfg.get(field, default)
        m_val = mod_cfg.get(field, default)
        if b_val != m_val:
            isolated_change[field] = {"baseline": b_val, "modified": m_val}
    return {
        "fair_protocol_verified": True,
        "baseline_method": base_cfg["method"],
        "modified_method": mod_cfg["method"],
        "isolated_change": isolated_change,
        "matched_protocol": {key: base_cfg[key] for key in matched_keys},
    }


def build_comparison_table(base_seeds, mod_seeds):
    base_indexed = base_seeds.sort_values("train_seed").set_index("train_seed")
    mod_indexed = mod_seeds.sort_values("train_seed").set_index("train_seed")
    metric_specs = (
        ("final_6x6_success_rate", "6x6 Episode Success Rate"),
        ("final_8x8_success_rate", "8x8 Zero-Shot Success Rate"),
        ("final_6x6_mean_return", "6x6 Mean Original Return"),
        ("final_8x8_mean_return", "8x8 Mean Original Return"),
        ("final_6x6_median_success_steps", "6x6 Median Steps (Successful Episodes)"),
        ("final_8x8_median_success_steps", "8x8 Median Steps (Successful Episodes)"),
        ("normalized_auc", "6x6 Training Normalized AUC"),
        ("t80", "Steps to First 80% Success (T80)"),
        ("final_6x6_fail_no_key_rate", "6x6 Fail Rate: Key Not Picked"),
        ("final_6x6_post_key_fail_rate", "6x6 Fail Rate: Post-Key Cycle/Timeout"),
        ("final_8x8_fail_no_key_rate", "8x8 Fail Rate: Key Not Picked"),
        ("final_8x8_fail_key_no_door_rate", "8x8 Fail Rate: Key Picked, Door Locked"),
        ("final_8x8_fail_door_no_goal_rate", "8x8 Fail Rate: Door Opened, No Goal"),
        ("final_8x8_post_key_fail_rate", "8x8 Fail Rate: Post-Key Cycle/Timeout"),
        ("learn_seconds", "Training Wall-Clock Seconds (incl. periodic eval)"),
        ("periodic_eval_seconds", "Periodic Evaluation Seconds"),
        ("final_eval_seconds", "Final Evaluation Seconds"),
    )
    rows = []
    for col, label in metric_specs:
        b_vals = base_indexed[col]
        m_vals = mod_indexed[col]
        b_mean, b_sd, b_n = mean_sd(b_vals)
        m_mean, m_sd, m_n = mean_sd(m_vals)
        diffs = m_vals - b_vals
        d_mean, d_sd, d_n = mean_sd(diffs)
        rows.append({
            "metric": col,
            "description": label,
            "baseline_mean": b_mean,
            "baseline_std": b_sd,
            "baseline_valid_n": b_n,
            "modified_mean": m_mean,
            "modified_std": m_sd,
            "modified_valid_n": m_n,
            "delta_mean": d_mean,
            "paired_seed_delta_std": d_sd,
            "paired_valid_n": d_n,
            "baseline_seed_1": float(b_vals.loc[1]),
            "baseline_seed_2": float(b_vals.loc[2]),
            "baseline_seed_3": float(b_vals.loc[3]),
            "modified_seed_1": float(m_vals.loc[1]),
            "modified_seed_2": float(m_vals.loc[2]),
            "modified_seed_3": float(m_vals.loc[3]),
        })
    return pd.DataFrame(rows)


def save_figure(fig, directory, name):
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def step_ticks(budget):
    if budget == 1_000_000:
        return [0, 200_000, 400_000, 600_000, 800_000, 1_000_000], ["0", "200k", "400k", "600k", "800k", "1M"]
    ticks = np.linspace(0, budget, 6, dtype=int).tolist()
    labels = [f"{t // 1000}k" if t >= 1000 and t % 1000 == 0 else str(t) for t in ticks]
    return ticks, labels


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
    band_low = np.clip(mean - sd, 0.0, 1.0)
    band_high = np.clip(mean + sd, 0.0, 1.0)
    ax.plot(x, mean, color="black", ls="--", lw=2, label="Three-seed mean", zorder=3)
    ax.fill_between(x, band_low, band_high, color="gray", alpha=0.18,
                    label="± sample SD (ddof=1, clipped [0, 100%])", zorder=0)
    ax.axhline(0.8, color="tab:purple", ls=":", lw=1.4, label="80% success")
    xticks, xticklabels = step_ticks(config["total_steps"])
    ax.set(xlabel="Training environment steps", ylabel="Success rate (SR)",
           xticks=xticks, xticklabels=xticklabels,
           xlim=(0, config["total_steps"]),
           ylim=(-0.02, 1.05),
           yticks=np.linspace(0, 1, 6),
           title=f"6×6 Learning Curve — {display_name(config['method'])}")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, directory, "learning_curve")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, (size, env_id) in zip(axes, ENVS.items()):
        group = final.loc[final["env_id"].eq(env_id)].sort_values("train_seed")
        summary = final_summary.loc[final_summary["env_id"].eq(env_id)].iloc[0]
        mean, sd = summary["success_rate_mean"], summary["success_rate_std"]
        bars = ax.bar(group["train_seed"], group["success_rate"], width=0.6,
                      color=["tab:blue", "tab:orange", "tab:green"], zorder=2)
        ax.bar_label(bars, labels=[f"{rate:.0%}" for rate in group["success_rate"]],
                     padding=4, fontsize=10)
        ax.axhline(mean, color="black", ls="--", lw=1.4, label="Three-seed mean", zorder=3)
        ax.axhspan(max(0.0, float(mean - sd)), min(1.0, float(mean + sd)),
                   color="gray", alpha=0.18,
                   label="± sample SD (ddof=1, clipped [0, 100%])", zorder=0)
        ax.text(0.5, 0.96, f"Mean ± sample SD: {mean:.2%} ± {sd * 100:.2f} pp",
                transform=ax.transAxes, ha="center", va="top", fontsize=10)
        title = "6×6" if size == "6x6" else "8×8 (Zero-shot Transfer)"
        ax.set(title=title, xlabel="Training seed", xticks=group["train_seed"],
               xticklabels=[f"Seed {seed}" for seed in group["train_seed"]],
               xlim=(0.4, 3.6), ylim=(0.0, 1.16), yticks=np.linspace(0, 1, 6))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Success rate (SR)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"Final 6×6 and 8×8 Evaluation — {display_name(config['method'])}", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    save_figure(fig, directory, "final_success")


def plot_comparison(base_cfg, mod_cfg, base_periodic, mod_periodic,
                    base_ps, mod_ps, base_final, mod_final, base_fs, mod_fs, output):
    directory = output / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    b_name = display_name(base_cfg["method"])
    m_name = display_name(mod_cfg["method"])
    colors = {base_cfg["method"]: "#1f77b4", mod_cfg["method"]: "#d95f02"}

    # 1. Overlaid three-seed 6x6 learning curves
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    for cfg, periodic, summary in (
        (base_cfg, base_periodic, base_ps),
        (mod_cfg, mod_periodic, mod_ps),
    ):
        method = cfg["method"]
        color = colors[method]
        label = display_name(method)
        for idx, (_, group) in enumerate(periodic.groupby("train_seed")):
            ax.plot(
                group["model_step"], group["success_rate"],
                color=color, lw=1.0, alpha=0.28,
                label=f"{label} individual seeds" if idx == 0 else None,
            )
        x = summary["model_step"].to_numpy()
        mean = summary["success_rate_mean"].to_numpy()
        sd = summary["success_rate_std"].to_numpy()
        ax.plot(x, mean, color=color, lw=2.4, label=f"{label} mean (n=3 seeds)", zorder=3)
        ax.fill_between(
            x, np.clip(mean - sd, 0.0, 1.0), np.clip(mean + sd, 0.0, 1.0),
            color=color, alpha=0.16, zorder=1,
        )
    ax.axhline(0.8, color="tab:purple", ls=":", lw=1.4, label="80% T80 threshold")
    xticks, xticklabels = step_ticks(base_cfg["total_steps"])
    ax.set(
        xlabel="Training environment steps (MiniGrid-DoorKey-6x6-v0)",
        ylabel="Periodic evaluation Success rate (100 episodes / seed)",
        xticks=xticks, xticklabels=xticklabels,
        xlim=(0, base_cfg["total_steps"]),
        ylim=(-0.02, 1.05),
        yticks=np.linspace(0, 1, 6),
        title=f"6×6 Learning Curve Comparison: {b_name} vs. {m_name} (± sample SD, ddof=1)",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.22)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.8)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    save_figure(fig, directory, "learning_curve_comparison")

    # 2. Final 6x6 and 8x8 zero-shot comparison with per-seed dots + mean ± SD
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.9), sharey=True)
    seed_offsets = {1: -0.09, 2: 0.0, 3: 0.09}
    seed_markers = {1: "o", 2: "s", 3: "^"}
    for ax, (size, env_id) in zip(axes, ENVS.items()):
        b_group = base_final.loc[base_final["env_id"].eq(env_id)].sort_values("train_seed")
        m_group = mod_final.loc[mod_final["env_id"].eq(env_id)].sort_values("train_seed")
        b_sum = base_fs.loc[base_fs["env_id"].eq(env_id)].iloc[0]
        m_sum = mod_fs.loc[mod_fs["env_id"].eq(env_id)].iloc[0]

        means = [b_sum["success_rate_mean"], m_sum["success_rate_mean"]]
        sds = [b_sum["success_rate_std"], m_sum["success_rate_std"]]
        low_err = [min(m, s) for m, s in zip(means, sds)]
        high_err = [min(1.0 - m, s) for m, s in zip(means, sds)]

        bars = ax.bar(
            [0, 1], means, width=0.52,
            color=[colors[base_cfg["method"]], colors[mod_cfg["method"]]],
            alpha=0.82, zorder=2,
        )
        ax.errorbar(
            [0, 1], means, yerr=[low_err, high_err],
            fmt="none", ecolor="black", elinewidth=1.8, capsize=6, zorder=4,
            label="± sample SD (n=3 seeds, ddof=1)",
        )
        for seed in (1, 2, 3):
            b_val = float(b_group.loc[b_group["train_seed"].eq(seed), "success_rate"].iloc[0])
            m_val = float(m_group.loc[m_group["train_seed"].eq(seed), "success_rate"].iloc[0])
            dx = seed_offsets[seed]
            ax.plot([0 + dx, 1 + dx], [b_val, m_val], color="#475569", ls="--", lw=0.9, alpha=0.6, zorder=3)
            ax.scatter(
                [0 + dx, 1 + dx], [b_val, m_val],
                marker=seed_markers[seed], s=46, color="white", edgecolors="#0f172a",
                linewidths=1.3, zorder=5, label=f"Seed {seed}" if size == "6x6" else None,
            )
        for idx, (bar, mean, sd) in enumerate(zip(bars, means, sds)):
            top = min(1.0, mean + sd)
            ax.text(
                idx, max(top, mean) + 0.04,
                f"{mean:.1%} ± {sd * 100:.1f} pp",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold",
            )
        delta = (means[1] - means[0]) * 100
        title = "6×6 Training Task (seeds 20000–20099)" if size == "6x6" else "8×8 Zero-Shot Transfer (seeds 30000–30099)"
        ax.text(0.5, 0.97, f"Δ ({m_name} − {b_name}): {delta:+.2f} pp",
                transform=ax.transAxes, ha="center", va="top", fontsize=9.2, color="#334155")
        ax.set(
            title=title, xticks=[0, 1],
            xticklabels=[tick_name(base_cfg["method"]), tick_name(mod_cfg["method"])],
            xlim=(-0.55, 1.55), ylim=(0.0, 1.25), yticks=np.linspace(0, 1, 6),
        )
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Final Episode Success rate (100 episodes / seed)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.8)
    fig.suptitle(f"Controlled Final Evaluation ({base_cfg['total_steps']:,} nominal steps / seed)", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    save_figure(fig, directory, "final_comparison")

    # 3. Episode outcome & failure-stage breakdown on 6x6 and 8x8
    stages = (
        ("success_rate", "Success", "#16a34a"),
        ("fail_no_key_rate", "Fail: Key Not Picked", "#dc2626"),
        ("fail_key_no_door_rate", "Fail: Key Picked, Door Locked", "#f59e0b"),
        ("fail_door_no_goal_rate", "Fail: Door Opened, No Goal", "#7c3aed"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    for ax, (size, env_id) in zip(axes, ENVS.items()):
        b_sum = base_fs.loc[base_fs["env_id"].eq(env_id)].iloc[0]
        m_sum = mod_fs.loc[mod_fs["env_id"].eq(env_id)].iloc[0]
        x = np.arange(len(stages))
        width = 0.35
        for offset, summary, label, hatch in (
            (-width / 2, b_sum, b_name, ""),
            (width / 2, m_sum, m_name, "//"),
        ):
            vals = [summary[f"{col}_mean"] for col, _, _ in stages]
            sds = [summary[f"{col}_std"] for col, _, _ in stages]
            low = [min(v, s) for v, s in zip(vals, sds)]
            high = [min(1.0 - v, s) for v, s in zip(vals, sds)]
            ax.bar(
                x + offset, vals, width=width,
                color=[c for _, _, c in stages],
                alpha=0.85 if not hatch else 0.62,
                edgecolor="#0f172a", linewidth=0.8, hatch=hatch,
                label=label if size == "6x6" else None, zorder=2,
            )
            ax.errorbar(
                x + offset, vals, yerr=[low, high],
                fmt="none", ecolor="black", elinewidth=1.2, capsize=3, zorder=3,
            )
        ax.set(
            title="6×6 Outcome Breakdown" if size == "6x6" else "8×8 Zero-Shot Outcome Breakdown",
            xticks=x,
            xticklabels=["Success", "No Key", "Key Only\n(Door Locked)", "Door Opened\n(No Goal)"],
            ylim=(0.0, 1.12), yticks=np.linspace(0, 1, 6),
        )
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.22)
    from matplotlib.patches import Patch

    axes[0].set_ylabel("Episode fraction (mean ± sample SD across 3 seeds)")
    legend_handles = [
        Patch(facecolor="#94a3b8", edgecolor="#0f172a", label=f"{b_name} (solid)"),
        Patch(facecolor="#cbd5e1", edgecolor="#0f172a", hatch="//", label=f"{m_name} (hatched)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle("Stage-Level Success and Failure Decomposition", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    save_figure(fig, directory, "failure_breakdown_comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Directory containing seed_1, seed_2, seed_3 (baseline in --compare mode)")
    parser.add_argument("--compare", type=Path, default=None,
                        help="Optional modified-method directory containing seed_1, seed_2, seed_3")
    parser.add_argument("--output", type=Path, required=True,
                        help="Directory for summary CSVs and figures")
    args = parser.parse_args()

    if args.compare is None:
        config, periodic, seeds, final = collect_results(args.input)
        periodic_summary, final_summary = write_single_summaries(
            config, periodic, seeds, final, args.output
        )
        plot_results(config, periodic, periodic_summary, final, final_summary, args.output)
        print(f"Wrote three-seed summaries and figures for '{config['method']}' to {args.output}")
        print(f"T80 reached: {seeds['t80'].notna().sum()}/3 seeds")
        return

    base_cfg, base_periodic, base_seeds, base_final = collect_results(args.input)
    mod_cfg, mod_periodic, mod_seeds, mod_final = collect_results(args.compare)
    protocol = verify_matched_protocol(base_cfg, mod_cfg)

    base_ps, base_fs, base_ts = summarize_method(base_cfg, base_periodic, base_seeds, base_final)
    mod_ps, mod_fs, mod_ts = summarize_method(mod_cfg, mod_periodic, mod_seeds, mod_final)
    comparison_table = build_comparison_table(base_seeds, mod_seeds)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "protocol_check.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    pd.concat([base_periodic, mod_periodic], ignore_index=True).to_csv(
        args.output / "periodic_by_seed.csv", index=False, na_rep="NaN"
    )
    pd.concat([base_ps, mod_ps], ignore_index=True).to_csv(
        args.output / "periodic_summary.csv", index=False, na_rep="NaN"
    )
    pd.concat([base_seeds, mod_seeds], ignore_index=True).to_csv(
        args.output / "seed_metrics.csv", index=False, na_rep="NaN"
    )
    pd.concat([base_final, mod_final], ignore_index=True).to_csv(
        args.output / "final_by_seed.csv", index=False, na_rep="NaN"
    )
    pd.concat([base_fs, mod_fs], ignore_index=True).to_csv(
        args.output / "final_summary.csv", index=False, na_rep="NaN"
    )
    pd.concat([base_ts, mod_ts], ignore_index=True).to_csv(
        args.output / "training_summary.csv", index=False, na_rep="NaN"
    )
    comparison_table.to_csv(args.output / "comparison_table.csv", index=False, na_rep="NaN")

    plot_comparison(
        base_cfg, mod_cfg, base_periodic, mod_periodic,
        base_ps, mod_ps, base_final, mod_final, base_fs, mod_fs, args.output,
    )
    print(f"Verified fair protocol and wrote comparison tables + figures to {args.output}")
    print(f"Baseline T80 reached: {base_seeds['t80'].notna().sum()}/3 seeds | "
          f"Modified T80 reached: {mod_seeds['t80'].notna().sum()}/3 seeds")


if __name__ == "__main__":
    main()
