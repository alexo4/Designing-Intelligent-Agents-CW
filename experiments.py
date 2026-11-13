"""Run controlled experiments over several seeds and report win rates with 95% CIs.

Each experiment is the v1 baseline plus ONE change (see EXPERIMENTS). Every run is trained
with the same budget and evaluated with the same protocol:

    final evaluation   greedy actions, FINAL_GAMES games per opponent, fixed game seeds,
                       sides alternated; opponents: random, heuristic, held-out variant
    learning curves    the in-training evaluation in <run>/<algo>_eval.csv (same for all runs)

Results are appended to results/experiments.csv (one row per run and checkpoint, including
the run's full config), and `report` turns them into results/summary.md and a plot.

Examples::

    python experiments.py list
    python experiments.py run v1 advnorm --seeds 0 1 2 --jobs 5
    python experiments.py eval-checkpoint v1_legacy_dqn20k dqn:runs/dqn_20k_s0/dqn_best.pt --seed 0
    python experiments.py report
"""

import argparse
import csv
import fcntl
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
RUNS = ROOT / "runs" / "exp"
RESULTS = ROOT / "results"
RESULTS_CSV = RESULTS / "experiments.csv"

FINAL_GAMES = 500
SMOKE = False
FINAL_SEED = 100_000

# Shared training budget and evaluation settings (don't change between runs).
# Both budgets are 819,200 environment steps (400 PPO updates x 2048 steps), with an
# evaluation roughly every 41k steps (20 PPO updates; ~500 DQN episodes of ~83 steps).
BASE_ARGS = {
    "ppo": ["--updates", "400", "--eval-every", "20", "--eval-games", "100", "--threads", "1"],
    # Replay buffer cut from 5M to 200k transitions (~1 GB) so 3 runs fit in 8 GB of RAM.
    "dqn": ["--env-steps", "819200", "--eval-every", "500", "--eval-games", "100", "--threads", "1",
            "--hp", "MEMORY_SIZE=200000", "--hp", "EPS_DECAY=1000000"],
}

# name -> (description, extra train.py arguments). Each is v1 + exactly one change.
EXPERIMENTS = {
    "v1": ("baseline: current rewards and trainer", []),
    "timeout_numbers": ("time-out win/loss on numbers = +1 / -1",
                        ["--reward", "win_on_numbers=1.0", "--reward", "lose_on_numbers=-1.0"]),
    "team_terminal": ("win/lose reward also to players eliminated earlier",
                      ["--reward", "terminal_to_eliminated=True"]),
    "pickup0": ("pickup reward = 0", ["--reward", "pickup=0.0"]),
    "miss0": ("miss penalty = 0", ["--reward", "miss=0.0"]),
    "sparse": ("only win/lose (hit, eliminated, pickup, miss = 0)",
               ["--reward", "hit=0.0", "--reward", "eliminated=0.0",
                "--reward", "pickup=0.0", "--reward", "miss=0.0"]),
    "idle": ("idle penalty -0.01", ["--reward", "idle=-0.01"]),
    "advnorm": ("PPO advantage normalisation", ["--hp", "NORM_ADV=True"]),
    "mix": ("opponent mix: 40% self, 30% past, 20% heuristic, 10% random",
            ["--hp", "P_SELF=0.4", "--hp", "P_PAST=0.3", "--hp", "P_HEURISTIC=0.2", "--hp", "P_RANDOM=0.1"]),
    "asym": ("ablation (later): win/lose = +2 / -1.5", ["--reward", "lose=-1.5"]),
    # v2: the changes that beat v1 over 5 seeds (mix, idle), combined.
    "v2": ("v2 = mix + idle penalty (400 updates)",
           ["--hp", "P_SELF=0.4", "--hp", "P_PAST=0.3", "--hp", "P_HEURISTIC=0.2", "--hp", "P_RANDOM=0.1",
            "--reward", "idle=-0.01"]),
    "v2_long": ("v2 trained for 1500 updates",
                ["--hp", "P_SELF=0.4", "--hp", "P_PAST=0.3", "--hp", "P_HEURISTIC=0.2", "--hp", "P_RANDOM=0.1",
                 "--reward", "idle=-0.01", "--updates", "1500"]),
    # DQN with the same 0.82M-step budget and v1 rewards (buffer 200k, see BASE_ARGS["dqn"]).
    "dqn_self": ("DQN self-play, 0.82M steps, buffer 200k", []),
    "dqn_mix": ("DQN + opponent mix (as PPO mix), 0.82M steps, buffer 200k",
                ["--hp", "P_SELF=0.4", "--hp", "P_PAST=0.3", "--hp", "P_HEURISTIC=0.2", "--hp", "P_RANDOM=0.1"]),
}
EXPERIMENTS["dqn_mix_buf50k"] = (
    "dqn_mix with a 50k replay buffer (tests whether stale replay data is why mixing doesn't help DQN)",
    EXPERIMENTS["dqn_mix"][1] + ["--hp", "MEMORY_SIZE=50000"])   # later --hp wins over BASE_ARGS
ALGO = {"dqn_self": "dqn", "dqn_mix": "dqn", "dqn_mix_buf50k": "dqn"}   # every other experiment is PPO


def algo_of(name):
    return ALGO.get(name, "ppo")

FIELDS = ["experiment", "description", "algo", "seed", "checkpoint", "iterations", "env_steps",
          "win_random", "win_heuristic", "win_heuristic_variant", "draw_heuristic_variant",
          "pickups_per_game", "throws_per_game", "idle_frac", "games", "config", "timestamp"]

# two-sided 95% t critical values by degrees of freedom
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def use_smoke_settings():
    """Tiny budget in separate folders, to check the pipeline end to end."""
    global RUNS, RESULTS, RESULTS_CSV, FINAL_GAMES, SMOKE
    SMOKE = True
    RUNS = ROOT / "runs" / "smoke"
    RESULTS = ROOT / "runs" / "smoke" / "results"
    RESULTS_CSV = RESULTS / "experiments.csv"
    FINAL_GAMES = 20
    BASE_ARGS["ppo"] = ["--updates", "2", "--eval-every", "1", "--eval-games", "10", "--threads", "1",
                        "--hp", "TIME=256"]
    BASE_ARGS["dqn"] = ["--env-steps", "300", "--eval-every", "2", "--eval-games", "10", "--threads", "1",
                        "--hp", "MEMORY_SIZE=5000", "--hp", "BATCH_SIZE=32"]


def run_dir(name, seed):
    return RUNS / f"{name}_s{seed}"


def train_command(name, seed):
    _, extra = EXPERIMENTS[name]
    algo = algo_of(name)
    return [sys.executable, "-u", str(ROOT / "train.py"), algo, *BASE_ARGS[algo], *extra,
            "--seed", str(seed), "--out-dir", str(run_dir(name, seed))]


# ----------------------------------------------------------------- evaluation

def evaluate_policy(policy):
    from dodgeball import DodgeballEnv
    from dodgeball.policies import head_to_head
    from train import eval_opponents

    env = DodgeballEnv()
    results = {}
    for i, (name, opponent) in enumerate(eval_opponents(env.grid_size, seed=FINAL_SEED).items()):
        results[name] = head_to_head(env, policy, opponent, FINAL_GAMES, seed=FINAL_SEED + 10_000 * i)
    return results


def result_row(experiment, description, algo, seed, checkpoint, results, config, iterations="", env_steps=""):
    return {
        "experiment": experiment, "description": description, "algo": algo, "seed": seed,
        "checkpoint": checkpoint, "iterations": iterations, "env_steps": env_steps,
        "win_random": results["random"]["win"],
        "win_heuristic": results["heuristic"]["win"],
        "win_heuristic_variant": results["heuristic_variant"]["win"],
        "draw_heuristic_variant": results["heuristic_variant"]["draw"],
        "pickups_per_game": results["random"]["pickups_per_game"],
        "throws_per_game": results["random"]["throws_per_game"],
        "idle_frac": round(results["random"]["idle_frac"], 4),
        "games": FINAL_GAMES, "config": json.dumps(config, sort_keys=True),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def append_rows(rows):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS_CSV, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)   # several runs may finish at once
        new = f.tell() == 0
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerows(rows)
        fcntl.flock(f, fcntl.LOCK_UN)


def read_rows():
    if not RESULTS_CSV.exists():
        return []
    with open(RESULTS_CSV) as f:
        return list(csv.DictReader(f))


def eval_run(name, seed):
    """Final evaluation of one finished run: its final and best checkpoints."""
    import torch
    from dodgeball import DodgeballEnv
    from dodgeball.policies import make_policy

    torch.set_num_threads(1)
    d = run_dir(name, seed)
    algo = algo_of(name)
    config = json.loads((d / "config.json").read_text())
    with open(d / f"{algo}_eval.csv") as f:
        last = list(csv.DictReader(f))[-1]
    env = DodgeballEnv()
    rows = []
    for checkpoint, path in (("final", d / f"{algo}.pt"), ("best", d / f"{algo}_best.pt")):
        results = evaluate_policy(make_policy(f"{algo}:{path}", env))
        rows.append(result_row(name, EXPERIMENTS[name][0], algo, seed, checkpoint, results, config,
                               last["iteration"], last["env_steps"]))
    append_rows(rows)
    print(f"{name} s{seed}: final win vs random {rows[0]['win_random']:.1%}, "
          f"vs held-out {rows[0]['win_heuristic_variant']:.1%}")


def eval_checkpoint(label, spec, seed, description):
    import torch
    from dodgeball import DodgeballEnv
    from dodgeball.policies import make_policy

    torch.set_num_threads(1)
    env = DodgeballEnv()
    results = evaluate_policy(make_policy(spec, env))
    algo, _, path = spec.partition(":")
    config_path = Path(path).parent / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    append_rows([result_row(label, description, algo, seed, Path(path).name, results, config)])
    print(f"{label}: win vs random {results['random']['win']:.1%}, "
          f"vs held-out {results['heuristic_variant']['win']:.1%}")


# --------------------------------------------------------------------- running

def run(names, seeds, jobs):
    done = {(r["experiment"], int(r["seed"])) for r in read_rows()}
    queue = [(n, s) for n in names for s in seeds if (n, s) not in done]
    if not queue:
        print("nothing to run: all requested runs are already in", RESULTS_CSV)
        return
    print(f"{len(queue)} runs, {jobs} at a time")
    RUNS.mkdir(parents=True, exist_ok=True)
    running = []
    while queue or running:
        while queue and len(running) < jobs:
            name, seed = queue.pop(0)
            d = run_dir(name, seed)
            log = open(f"{d}.log", "w")
            train = subprocess.list2cmdline(train_command(name, seed))
            evaluate = subprocess.list2cmdline([sys.executable, str(ROOT / "experiments.py"),
                                                *(["--smoke"] if SMOKE else []), "eval-run", name, str(seed)])
            proc = subprocess.Popen(f"{train} && {evaluate}", shell=True, stdout=log, stderr=subprocess.STDOUT,
                                    cwd=ROOT)
            running.append((proc, name, seed, log, time.time()))
            print(f"started {name} s{seed}")
        time.sleep(10)
        for item in list(running):
            proc, name, seed, log, start = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                status = "done" if proc.returncode == 0 else f"FAILED ({proc.returncode}), see {run_dir(name, seed)}.log"
                print(f"{status}: {name} s{seed} after {(time.time() - start) / 60:.0f} min")


# --------------------------------------------------------------------- report

def mean_ci(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return values.mean(), float("nan")
    half = T95.get(n - 1, 1.96) * values.std(ddof=1) / math.sqrt(n)
    return values.mean(), half


def fmt(values):
    mean, half = mean_ci(values)
    return f"{mean:.1%}" if math.isnan(half) else f"{mean:.1%} ± {half:.1%}"


def report(names, out):
    rows = read_rows()
    if not rows:
        raise SystemExit("no results yet")
    order = list(EXPERIMENTS) + sorted({r["experiment"] for r in rows} - set(EXPERIMENTS))
    names = names or [n for n in order if any(r["experiment"] == n for r in rows)]

    lines = [f"Final evaluation: greedy actions, {FINAL_GAMES} games per opponent per run, sides "
             f"alternated. Mean over seeds ± 95% CI (t-distribution).", ""]
    for checkpoint in ("final", "best"):
        lines += [f"### {checkpoint.capitalize()} checkpoint", "",
                  "| Experiment | Change | Seeds | Win vs random | Win vs held-out heuristic | "
                  "Win vs heuristic | Pickups/game | Throws/game | Idle |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name in names:
            # legacy rows store a file name (e.g. dqn_best.pt) instead of "final"/"best"
            group = [r for r in rows if r["experiment"] == name and
                     ("best" if "best" in r["checkpoint"] else "final") == checkpoint]
            if not group:
                continue
            col = lambda k: [float(r[k]) for r in group]
            lines.append(
                f"| {name} | {group[0]['description']} | {len(group)} | {fmt(col('win_random'))} "
                f"| {fmt(col('win_heuristic_variant'))} | {fmt(col('win_heuristic'))} "
                f"| {np.mean(col('pickups_per_game')):.1f} | {np.mean(col('throws_per_game')):.1f} "
                f"| {np.mean(col('idle_frac')):.0%} |")
        lines.append("")
    text = "\n".join(lines)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.md").write_text(text)
    print(text)
    plot_curves([n for n in names if n in EXPERIMENTS], out)


def plot_curves(names, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("win_vs_random", "Win rate vs random"), ("win_vs_heuristic_variant", "Win rate vs held-out heuristic")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for i, name in enumerate(names):
        curves = []
        for d in sorted(RUNS.glob(f"{name}_s*")):
            path = d / f"{algo_of(name)}_eval.csv"
            if d.is_dir() and path.exists():
                with open(path) as f:
                    curves.append(list(csv.DictReader(f)))
        if not curves:
            continue
        n = min(len(c) for c in curves)
        steps = np.mean([[float(r["env_steps"]) for r in c[:n]] for c in curves], axis=0)
        for ax, (column, title) in zip(axes, panels):
            data = np.array([[float(r[column]) for r in c[:n]] for c in curves])
            mean = data.mean(0)
            half = (T95.get(len(curves) - 1, 1.96) * data.std(0, ddof=1) / math.sqrt(len(curves))
                    if len(curves) > 1 else np.zeros_like(mean))
            ax.plot(steps, mean, color=f"C{i % 10}", label=f"{name} ({len(curves)} seeds)",
                    linestyle="-" if i < 10 else "--")
            ax.fill_between(steps, np.clip(mean - half, 0, 1), np.clip(mean + half, 0, 1), color=f"C{i % 10}", alpha=0.12)
            ax.set_title(title)
    for ax in axes:
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
        ax.set_xlabel("environment steps")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("win rate (greedy, mean ± 95% CI)")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="tiny budget in runs/smoke, to test the pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("run")
    p.add_argument("names", nargs="+", choices=list(EXPERIMENTS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--jobs", type=int, default=5)
    p = sub.add_parser("eval-run")
    p.add_argument("name", choices=list(EXPERIMENTS))
    p.add_argument("seed", type=int)
    p = sub.add_parser("eval-checkpoint", help="evaluate any checkpoint (e.g. older runs) with the same protocol")
    p.add_argument("label")
    p.add_argument("spec", help="dqn:<path> or ppo:<path>")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--description", default="")
    p = sub.add_parser("report")
    p.add_argument("names", nargs="*")
    p.add_argument("--out", type=Path, default=None, help="plot path (default: results/learning_curves.png)")
    args = parser.parse_args()
    if args.smoke:
        use_smoke_settings()

    if args.cmd == "list":
        for name, (desc, extra) in EXPERIMENTS.items():
            print(f"{name:16s} {desc}\n{'':16s} {' '.join(extra)}")
    elif args.cmd == "run":
        run(args.names, args.seeds, args.jobs)
    elif args.cmd == "eval-run":
        eval_run(args.name, args.seed)
    elif args.cmd == "eval-checkpoint":
        eval_checkpoint(args.label, args.spec, args.seed, args.description)
    else:
        report(args.names, args.out or RESULTS / "learning_curves.png")


if __name__ == "__main__":
    main()
