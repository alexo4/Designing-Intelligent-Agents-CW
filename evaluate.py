"""Play two policies against each other and report win / loss / draw counts.

A policy is ``random``, ``heuristic``, ``dqn:<checkpoint>`` or ``ppo:<checkpoint>``.
The first policy controls team blue, the second controls team red. Use ``--swap`` to play
half the games with the sides swapped and report results for the first policy.

Examples::

    python evaluate.py heuristic random --games 500
    python evaluate.py ppo:checkpoints/ppo.pt heuristic --games 1000 --swap
    python evaluate.py ppo:checkpoints/ppo.pt random --games 1 --render
    python evaluate.py ppo:checkpoints/ppo.pt heuristic --games 1 --render --delay 1 --cell 60
"""

import argparse
import random
from collections import Counter

import numpy as np
import torch

from dodgeball import DodgeballEnv
from dodgeball.policies import head_to_head, make_policy, play_game


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("blue", help="policy for team blue")
    parser.add_argument("red", help="policy for team red")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--swap", action="store_true", help="alternate sides; report for the first policy")
    parser.add_argument("--render", action="store_true", help="show games in a Tk window")
    parser.add_argument("--delay", type=float, default=0.4, help="seconds per step when rendering")
    parser.add_argument("--cell", type=int, default=None,
                        help="pixels per grid square when rendering (default: fit the screen)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = DodgeballEnv()
    blue, red = make_policy(args.blue, env), make_policy(args.red, env)

    if args.swap:
        result = head_to_head(env, blue, red, args.games, seed=args.seed)
        print(f"{blue.name} vs {red.name}, {args.games} games (sides alternated)")
        for outcome in ("win", "loss", "draw"):
            print(f"  {outcome:5s} {result[outcome]:.1%}")
        return

    renderer = None
    if args.render:
        from dodgeball.render import CourtRenderer
        renderer = CourtRenderer(env.grid_size, cell=args.cell)

    results = Counter(play_game(env, blue, red, seed=args.seed + i, renderer=renderer, delay=args.delay)
                      for i in range(args.games))
    print(f"blue = {blue.name}, red = {red.name}, games = {args.games}")
    for outcome in ("blue", "red", "draw"):
        print(f"  {outcome:5s} {results[outcome]:5d}  ({results[outcome] / args.games:.1%})")
    if renderer is not None:
        print("close the window to exit")
        renderer.wait_until_closed()


if __name__ == "__main__":
    main()
