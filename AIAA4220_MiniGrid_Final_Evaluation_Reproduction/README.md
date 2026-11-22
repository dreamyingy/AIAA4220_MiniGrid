# MiniGrid final evaluation reproduction

| Field | Record |
|---|---|
| Responsible team member | Guorun WANG |
| Evaluation date | 22 November 2026 (Asia/Shanghai) |
| Evaluation environment | macOS ARM64, CPU, Python 3.11.15, repository-pinned packages |

The archived runs used Linux and Python 3.11.18.

Repository snapshot: [e70d6e93c7b6519e8af758527aa3bc6213e6bc4a](https://github.com/dreamyingy/AIAA4220_MiniGrid/tree/e70d6e93c7b6519e8af758527aa3bc6213e6bc4a)

The repository's `evaluate.py` evaluated all six archived `final.zip` PPO models using deterministic actions. Each model ran 100 episodes on 6×6 (episode seeds 20000–20099) and 100 on 8×8 (30000–30099). The repository's `analyze.py` used the new final CSVs and the archived run configs and periodic-evaluation CSVs. Training curves, AUC, and T80 were recalculated from those periodic records. The models came from the archived training runs.

## Results

All 1,200 new episode records matched the original records field for field, excluding elapsed time. The rerun reproduced success, return, episode length, termination and truncation flags, and key, door, and goal milestones. All seven analysis tables matched the archived results to numerical precision after excluding local final-evaluation timings.

| Evaluation | Baseline | Four frames | Mean paired change |
|---|---:|---:|---:|
| 6×6 final success | 91.67% ± 6.81 pp | 97.00% ± 1.00 pp | +5.33 pp |
| 8×8 zero-shot success | 49.33% ± 22.28 pp | 58.33% ± 16.26 pp | +9.00 pp |

Values are mean success ± sample standard deviation across three training seeds. pp denotes percentage points.

## Evidence

- [Final result summary](comparison/final_summary.csv) and [comparison table](comparison/comparison_table.csv)
- [New episode records](final_episodes/) covering all 1,200 episodes in 12 CSVs
- [Verification record](verification.json) with per-seed success counts, match checks, and archived model SHA-256 hashes
- [Analysis outputs](comparison/) with seven analysis tables, the protocol check, and figures
- [Execution logs](logs/) with the evaluation and analysis commands and their output
- [Installed dependencies](environment.txt)
