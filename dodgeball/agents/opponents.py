"""Opponent sampling for training: self-play, the scripted heuristic, or a past copy of the network.

Pure self-play only ever trains against the current network, so the strategy can drift and
forget how to beat other styles of play. Mixing in fixed opponents (the heuristic) and frozen
snapshots of earlier versions of the network keeps the training opponents varied.
"""

import copy
import random
from collections import deque

from dodgeball.entities import BLUE, RED
from dodgeball.policies import HeuristicPolicy, NetworkPolicy, RandomPolicy


class OpponentPool:

    def __init__(self, env, kind, p_self=1.0, p_heuristic=0.0, p_past=0.0, pool_size=5, p_random=0.0):
        probs = [p_self, p_heuristic, p_past, p_random]
        total = sum(probs)
        if total <= 0 or min(probs) < 0:
            raise ValueError("opponent probabilities must be non-negative and not all zero")
        self.env = env
        self.kind = kind
        self.weights = [p / total for p in probs]
        # Only the standard heuristic: the held-out variant must never be a training opponent.
        self.heuristic = HeuristicPolicy(env.grid_size)
        self.random = RandomPolicy(random.randrange(2**31)) if p_random > 0 else None
        self.snapshots = deque(maxlen=pool_size)
        self.learner_team = RED     # flipped before the first episode, so blue goes first
        self.opponent = None        # None means self-play

    def sample_episode(self):
        """Pick the learner's side and the opponent for the next episode.

        Returns ``(learner_team, opponent)``; ``opponent`` is None for self-play.
        """
        self.learner_team = BLUE if self.learner_team == RED else RED
        if self.weights[0] == 1.0:
            # Pure self-play: don't touch the random state, so seeded runs match older ones.
            self.opponent = None
            return self.learner_team, None
        choice = random.choices(("self", "heuristic", "past", "random"), weights=self.weights)[0]
        if choice == "heuristic":
            self.opponent = self.heuristic
        elif choice == "random":
            self.opponent = self.random
        elif choice == "past" and self.snapshots:
            self.opponent = random.choice(self.snapshots)
        else:
            self.opponent = None
        return self.learner_team, self.opponent

    @property
    def self_play(self):
        return self.opponent is None

    def is_learner(self, agent):
        """True if the network being trained controls ``agent`` in the current episode."""
        if self.opponent is None:
            return True
        return (agent >= self.env.players_per_team) == (self.learner_team == RED)

    def snapshot(self, net):
        """Add a frozen copy of ``net`` to the pool (the oldest is dropped when full)."""
        frozen = copy.deepcopy(net)
        frozen.eval()
        for p in frozen.parameters():
            p.requires_grad_(False)
        self.snapshots.append(NetworkPolicy(frozen, self.kind, name=f"past {self.kind}", greedy=False))
