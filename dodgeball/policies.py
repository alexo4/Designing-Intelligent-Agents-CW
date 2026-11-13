"""Policies that can play a team, and helpers to pit them against each other.

Every policy has ``act(obs, action_mask) -> int`` and only uses what a learned agent sees:
the egocentric observation and the action mask. So the scripted heuristic is a fair
benchmark: it has no access to hidden state.
"""

from collections import Counter
from math import floor, sqrt

import numpy as np
import torch

from dodgeball.env import FORWARD, BACK, RIGHT, LEFT, STAY, N_MOVE_ACTIONS, GRID_CHANNELS, hit_chance


class RandomPolicy:
    name = "random"

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, action_mask):
        return int(self.rng.choice(np.flatnonzero(action_mask)))


class HeuristicPolicy:
    """Hand-written strategy, used as a stronger benchmark than random play.

    Holding a ball:
        throw at the opponent with the best hit chance if it is at least ``min_chance``
        (or at anyone once the possession timer is nearly up), otherwise walk forward to
        get closer. At the half-way line it throws anyway, unless ``patient``: then it holds
        the ball (which also makes it harder to hit) until the shot is good enough or the
        possession timer forces a throw.
    Not holding a ball:
        walk to the nearest loose ball on our half; if there is none, back away from the
        half-way line when the opponents hold balls, to make their throws less accurate.
    """

    name = "heuristic"

    # The held-out evaluation opponent: waits for a 70% shot and only panic-throws on the
    # last step. Beats the standard heuristic ~60% of the time. Never used in training.
    VARIANT = {"min_chance": 0.7, "panic_timer": 1}

    @classmethod
    def variant(cls, grid_size=8):
        policy = cls(grid_size, **cls.VARIANT)
        policy.name = "heuristic_variant"
        return policy

    def __init__(self, grid_size=8, min_chance=0.6, panic_timer=2, patient=False):
        self.g = grid_size
        self.h = 2 * grid_size
        self.min_chance = min_chance
        self.panic_timer = panic_timer
        self.patient = patient

    def _parse(self, obs):
        grid = obs[:GRID_CHANNELS * self.h * self.g].reshape(GRID_CHANNELS, self.h, self.g)
        extra = obs[GRID_CHANNELS * self.h * self.g:]
        teammates, opponents, balls, me = grid
        my_y, my_x = np.argwhere(me > 0)[0]
        return teammates, opponents, balls, (my_y, my_x), extra

    def act(self, obs, action_mask):
        teammates, opponents, balls, (my_y, my_x), extra = self._parse(obs)
        if not np.any(action_mask[1:]):
            return STAY

        if np.any(action_mask[N_MOVE_ACTIONS:]):
            action = self._throw_or_advance(opponents, balls, my_x, my_y, extra, action_mask)
        else:
            action = self._fetch_or_retreat(teammates, balls, my_x, my_y, extra, action_mask)
        return action if action_mask[action] else STAY

    def _throw_or_advance(self, opponents, balls, my_x, my_y, extra, mask):
        # Opponents in the same order the env uses for throw actions: by column, then by
        # egocentric row (nearest the half-way line first).
        targets = sorted((x, y) for y, x in np.argwhere(opponents > 0))
        best_action, best_chance = None, -1.0
        for k, (x, y) in enumerate(targets):
            dist = floor(sqrt((x - my_x) ** 2 + (y - my_y) ** 2))
            blocking = balls[y, x] > 0
            chance = hit_chance(dist, 1, blocking, max_dist=self.h - 1)
            if chance > best_chance:
                best_action, best_chance = N_MOVE_ACTIONS + k, chance

        own_timer = extra[1] * 10  # possession timer, scaled back to steps
        if best_action is not None and (best_chance >= self.min_chance or own_timer <= self.panic_timer):
            return best_action
        if mask[FORWARD]:
            return FORWARD
        if self.patient:
            return STAY
        return best_action if best_action is not None else STAY

    def _fetch_or_retreat(self, teammates, balls, my_x, my_y, extra, mask):
        loose = [(y, x) for y, x in np.argwhere(balls[:self.g] > 0) if teammates[y, x] == 0]
        if loose:
            ty, tx = min(loose, key=lambda p: abs(p[0] - my_y) + abs(p[1] - my_x))
            options = []
            if ty > my_y:
                options.append(FORWARD)
            if ty < my_y:
                options.append(BACK)
            if tx > my_x:
                options.append(RIGHT)
            if tx < my_x:
                options.append(LEFT)
            for a in options:
                if mask[a]:
                    return a

        opponents_have_balls = extra[4] > 0
        if opponents_have_balls and mask[BACK]:
            return BACK
        return STAY


class NetworkPolicy:
    """Wraps a DQN or PPO network.

    ``greedy=True`` takes the highest-scoring valid action (Q-value for DQN, policy logit for
    PPO); this is what evaluation uses. ``greedy=False`` samples PPO's policy (DQN is always
    greedy), which is how past snapshots play as training opponents.
    """

    def __init__(self, net, kind, name=None, greedy=True):
        self.net = net
        self.kind = kind
        self.name = name or kind
        self.greedy = greedy

    def act(self, obs, action_mask):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        invalid = torch.as_tensor(action_mask) == 0
        with torch.no_grad():
            if self.kind == "dqn":
                scores = self.net(obs_t)
            elif self.greedy:
                scores, _ = self.net(obs_t)
            else:
                action, _, _, _ = self.net.select_action_and_value(obs_t, action_mask)
                return int(action.item())
            scores[invalid] = -float("inf")
            return int(torch.argmax(scores).item())


def load_checkpoint(model_cls, path, env):
    obs_size = env.observation_space(0).shape[0]
    model = model_cls(env.n_actions, obs_size)
    state = torch.load(path, map_location="cpu")
    first_layer = next(iter(model.state_dict()))
    if state[first_layer].shape[1] != obs_size:
        raise ValueError(f"{path} expects {state[first_layer].shape[1]} observation features but the "
                         f"environment produces {obs_size}; it was trained on an older environment version")
    model.load_state_dict(state)
    model.eval()
    return model


def make_policy(spec, env):
    """``random``, ``heuristic``, ``dqn:<checkpoint>`` or ``ppo:<checkpoint>``."""
    from dodgeball.agents import DQN, PPO

    if spec == "random":
        return RandomPolicy()
    if spec == "heuristic":
        return HeuristicPolicy(env.grid_size)
    if spec == "heuristic_variant":
        return HeuristicPolicy.variant(env.grid_size)
    kind, _, path = spec.partition(":")
    if kind in ("dqn", "ppo") and path:
        model_cls = DQN if kind == "dqn" else PPO
        return NetworkPolicy(load_checkpoint(model_cls, path, env), kind, name=spec)
    raise ValueError(f"unknown policy '{spec}' (expected random, heuristic, heuristic_variant, "
                     f"dqn:<path> or ppo:<path>)")


def play_game(env, blue, red, seed=None, renderer=None, delay=0.1, stats=None):
    """Play one game and return the winner: 'blue', 'red' or 'draw'.

    If ``stats`` is a ``{"blue": Counter(), "red": Counter()}`` dict, each team's counts of
    agent-steps, pickups, throws and idle steps (staying put when it could have moved or
    thrown) are added to it.
    """
    import time

    obs, infos = env.reset(seed=seed)
    half = env.players_per_team
    while env.agents:
        actions = {}
        for agent in env.agents:
            policy = blue if agent < half else red
            mask = infos[agent]["action_mask"]
            actions[agent] = policy.act(obs[agent], mask)
            if stats is not None:
                team = stats["blue" if agent < half else "red"]
                team["agent_steps"] += 1
                team["throws"] += int(actions[agent] >= N_MOVE_ACTIONS)
                team["idle"] += int(actions[agent] == STAY and mask[1:].any())
        obs, _, _, _, infos = env.step(actions)
        if stats is not None:
            for agent, info in infos.items():
                stats["blue" if agent < half else "red"]["pickups"] += int(info.get("picked_up", False))
        if renderer is not None:
            renderer.draw(env)
            time.sleep(delay)
    return env.winner


def head_to_head(env, policy, opponent, games, seed=0):
    """Play ``games`` games, with ``policy`` on each side for half of them.

    Returns win, loss and draw rates from ``policy``'s point of view, plus its behaviour:
    pickups and throws per game, and the fraction of its agent-steps spent idle.
    """
    results = Counter()
    behaviour = Counter()
    for i in range(games):
        as_blue = i % 2 == 0
        blue, red = (policy, opponent) if as_blue else (opponent, policy)
        stats = {"blue": Counter(), "red": Counter()}
        winner = play_game(env, blue, red, seed=seed + i, stats=stats)
        behaviour.update(stats["blue" if as_blue else "red"])
        if winner == "draw":
            results["draw"] += 1
        elif (winner == "blue") == as_blue:
            results["win"] += 1
        else:
            results["loss"] += 1
    out = {k: results[k] / games for k in ("win", "loss", "draw")}
    out["pickups_per_game"] = behaviour["pickups"] / games
    out["throws_per_game"] = behaviour["throws"] / games
    out["idle_frac"] = behaviour["idle"] / max(behaviour["agent_steps"], 1)
    return out
