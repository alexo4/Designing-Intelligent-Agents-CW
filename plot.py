"""Plot learning curves (mean ± std across seeds) from the eval logs written by train.py.

Each run directory holds ``<algo>_eval.csv``. Runs are grouped by directory name with the
``_s<seed>`` suffix removed, so ``runs/ppo_s0`` and ``runs/ppo_s1`` are averaged as "ppo",
and ``runs/ppo_lr1e-4_s0`` forms a separate "ppo_lr1e-4" curve.

Examples::

    python plot.py runs/* --out results/learning_curves.png
    python plot.py runs/ppo_s* runs/ppo_lr1e-4_s* --out results/ppo_lr.png
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_runs(run_dirs):
    """{group: [run, ...]} where each run is a dict of column -> np.array."""
    runs = defaultdict(list)
    for d in map(Path, run_dirs):
        for path in sorted(d.glob("*_eval.csv")):
            algo = re.sub(r"_s\d+$", "", d.name)
            with open(path) as f:
                rows = list(csv.DictReader(f))
            if rows:
                runs[algo].append({k: np.array([float(r[k]) for r in rows]) for k in rows[0]})
    return runs


def stack(runs, column):
    """Stack a column across runs, truncated to the shortest run."""
    n = min(len(r[column]) for r in runs)
    return np.stack([r[column][:n] for r in runs])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--out", type=Path, default=Path("results/learning_curves.png"))
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_runs(args.run_dirs)
    if not runs:
        raise SystemExit("no *_eval.csv files found")

    opponents = [("win_vs_random", "vs random"), ("win_vs_heuristic", "vs scripted heuristic")]
    has_variant = all("win_vs_heuristic_variant" in r for group in runs.values() for r in group)
    if has_variant:
        opponents.append(("win_vs_heuristic_variant", "vs held-out variant"))
    fig, axes = plt.subplots(1, len(opponents), figsize=(5.5 * len(opponents), 4), sharey=True)

    header = ["Run", "Seeds"] + [f"Final win {title}" for _, title in opponents] + ["Best win vs scripted heuristic"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for algo, algo_runs in sorted(runs.items()):
        steps = stack(algo_runs, "env_steps").mean(0)
        for ax, (column, title) in zip(axes, opponents):
            wins = stack(algo_runs, column)
            mean, std = wins.mean(0), wins.std(0)
            label = f"{algo} ({len(algo_runs)} seed{'s' * (len(algo_runs) > 1)})"
            colour = f"C{sorted(runs).index(algo)}"
            ax.plot(steps, mean, color=colour, label=label)
            ax.fill_between(steps, mean - std, mean + std, color=colour, alpha=0.2)
            ax.set_title(f"Win rate {title}")

        cells = [algo, str(len(algo_runs))]
        for column, _ in opponents:
            final = stack(algo_runs, column)[:, -1]
            cells.append(f"{final.mean():.1%} ± {final.std():.1%}")
        best = stack(algo_runs, "win_vs_heuristic").max(1)
        cells.append(f"{best.mean():.1%} ± {best.std():.1%}")
        print("| " + " | ".join(cells) + " |")

    for ax in axes:
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("environment steps")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("win rate")
    axes[0].legend(loc="lower right")
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
