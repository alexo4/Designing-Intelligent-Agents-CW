"""Train a DQN or PPO agent, evaluating it against fixed opponents as it learns.

Every ``--eval-every`` iterations the current network is saved to ``<out-dir>/ckpts/`` and
plays ``--eval-games`` games (greedy actions, half as blue, half as red) against:

    random              uniformly random valid actions
    heuristic           the scripted heuristic (may also be a training opponent)
    heuristic_variant   held-out scripted opponent, never used in training

Results go to ``<out-dir>/<algo>_eval.csv`` (learning curves: plot.py / experiments.py).
``<algo>_best.pt`` is the checkpoint with the best win rate against the standard heuristic.
The held-out variant is deliberately NOT used to pick it, so it stays an independent test.

Examples::

    python train.py ppo --updates 300 --seed 0 --out-dir runs/ppo_s0
    python train.py dqn --episodes 3000 --seed 0 --out-dir runs/dqn_s0

    # override hyperparameters (any upper-case attribute of DQNTrainer / PPOTrainer)
    python train.py ppo --hp LR=1e-4 --hp EPOCHS=8 --seed 0 --out-dir runs/ppo_lr1e-4_s0

    # override rewards (any field of dodgeball.env.RewardConfig)
    python train.py ppo --reward pickup=0 --reward idle=-0.01 --seed 0 --out-dir runs/ppo_idle_s0
"""

import argparse
import ast
import csv
import dataclasses
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from dodgeball import DodgeballEnv
from dodgeball.agents import DQNTrainer, PPOTrainer
from dodgeball.env import RewardConfig
from dodgeball.policies import HeuristicPolicy, NetworkPolicy, RandomPolicy, head_to_head

OPPONENTS = ("random", "heuristic", "heuristic_variant")
EVAL_FIELDS = ["iteration", "env_steps", "seconds"] + [f"win_vs_{o}" for o in OPPONENTS] + [
    "draw_vs_heuristic", "pickups_per_game", "throws_per_game", "idle_frac"]


def eval_opponents(grid_size, seed=0):
    return {
        "random": RandomPolicy(seed),
        "heuristic": HeuristicPolicy(grid_size),
        "heuristic_variant": HeuristicPolicy.variant(grid_size),
    }


class Evaluator:
    """Called by the trainer after each iteration; evaluates every ``every`` iterations."""

    def __init__(self, trainer, algo, out_dir, every, games, seed):
        self.trainer = trainer
        self.algo = algo
        self.every = every
        self.games = games
        self.env = DodgeballEnv()   # separate env, so evaluation never disturbs training
        self.opponents = eval_opponents(self.env.grid_size, seed)
        self.ckpt_dir = out_dir / "ckpts"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = out_dir / f"{algo}_best.pt"
        self.best = -1.0
        self.start = time.time()
        self.file = open(out_dir / f"{algo}_eval.csv", "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=EVAL_FIELDS)
        self.writer.writeheader()

    def __call__(self, iteration):
        if iteration % self.every:
            return
        self.trainer.save(self.ckpt_dir / f"{iteration}.pt")
        policy = NetworkPolicy(self.trainer.policy_net, self.algo, greedy=True)
        res = {name: head_to_head(self.env, policy, opp, self.games, seed=10_000 * (i + 1))
               for i, (name, opp) in enumerate(self.opponents.items())}
        row = {
            "iteration": iteration,
            "env_steps": self.trainer.env_steps,
            "seconds": round(time.time() - self.start, 1),
            **{f"win_vs_{name}": r["win"] for name, r in res.items()},
            "draw_vs_heuristic": res["heuristic"]["draw"],
            # behaviour measured in the games against the random policy
            "pickups_per_game": res["random"]["pickups_per_game"],
            "throws_per_game": res["random"]["throws_per_game"],
            "idle_frac": round(res["random"]["idle_frac"], 4),
        }
        self.writer.writerow(row)
        self.file.flush()
        print(f"  eval @ {iteration}: win vs random {res['random']['win']:.0%}, "
              f"heuristic {res['heuristic']['win']:.0%}, held-out variant {res['heuristic_variant']['win']:.0%}")

        if res["heuristic"]["win"] > self.best:
            self.best = res["heuristic"]["win"]
            self.trainer.save(self.best_path)

    def close(self):
        self.file.close()


def parse_overrides(items, valid, what):
    values = {}
    for item in items:
        name, _, raw = item.partition("=")
        if name not in valid:
            raise SystemExit(f"unknown {what} '{name}'; choose from {sorted(valid)}")
        values[name] = ast.literal_eval(raw)
    return values


def apply_overrides(trainer_cls, overrides):
    """Return a subclass of ``trainer_cls`` with ``NAME=value`` hyperparameters replaced."""
    valid = {k for k in dir(trainer_cls) if k.isupper()}
    values = parse_overrides(overrides, valid, f"hyperparameter for {trainer_cls.__name__}")
    return type(trainer_cls.__name__, (trainer_cls,), values)


def make_rewards(overrides):
    valid = {f.name for f in dataclasses.fields(RewardConfig)}
    return dataclasses.replace(RewardConfig(), **parse_overrides(overrides, valid, "reward field"))


def hyperparameters(trainer_cls):
    return {k: getattr(trainer_cls, k) for k in dir(trainer_cls) if k.isupper()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("algo", choices=["dqn", "ppo"])
    parser.add_argument("--episodes", type=int, default=3000, help="DQN: number of training episodes")
    parser.add_argument("--updates", type=int, default=300, help="PPO: number of policy updates")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--eval-every", type=int, default=None,
                        help="iterations between evaluations (default: 100 episodes for DQN, 10 updates for PPO)")
    parser.add_argument("--eval-games", type=int, default=100, help="games per opponent at each evaluation")
    parser.add_argument("--hp", action="append", default=[], metavar="NAME=VALUE",
                        help="override a trainer hyperparameter, e.g. --hp LR=1e-4 (repeatable)")
    parser.add_argument("--reward", action="append", default=[], metavar="FIELD=VALUE",
                        help="override a RewardConfig field, e.g. --reward pickup=0 (repeatable)")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads (1 for parallel runs)")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    rewards = make_rewards(args.reward)
    env = DodgeballEnv(rewards=rewards)
    if args.seed is not None:
        env.reset(seed=args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trainer_cls = apply_overrides(DQNTrainer if args.algo == "dqn" else PPOTrainer, args.hp)
    trainer = trainer_cls(env)
    config = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "hyperparameters": hyperparameters(trainer_cls),
        "rewards": dataclasses.asdict(rewards),
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    every = args.eval_every or (100 if args.algo == "dqn" else 10)
    evaluator = Evaluator(trainer, args.algo, args.out_dir, every, args.eval_games, args.seed or 0)

    try:
        if args.algo == "dqn":
            episode_rewards = trainer.train(args.episodes, callback=evaluator)
        else:
            episode_rewards = trainer.train(args.updates, callback=evaluator)
    finally:
        evaluator.close()

    model_path = args.out_dir / f"{args.algo}.pt"
    log_path = args.out_dir / f"{args.algo}_rewards.csv"
    trainer.save(model_path)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "total_reward"])
        writer.writerows(enumerate(episode_rewards))

    print(f"saved {model_path}, {evaluator.best_path} (best vs heuristic: {evaluator.best:.0%}), "
          f"{log_path} and {args.algo}_eval.csv")


if __name__ == "__main__":
    main()
