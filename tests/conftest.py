import numpy as np
import pytest

from dodgeball.env import EMPTY, STAY, DodgeballEnv


@pytest.fixture
def env():
    e = DodgeballEnv()
    e.reset(seed=0)
    return e


def set_layout(env, positions, balls=None):
    """Put the environment into a custom state.

    ``positions`` maps agent id -> (x, y) in world coordinates; agents not listed are
    eliminated. ``balls`` maps (x, y) -> number of balls.
    """
    env.occupant[:] = EMPTY
    env.balls[:] = 0
    for p in env.players:
        if p.agent_id in positions:
            p.x, p.y = positions[p.agent_id]
            p.alive = True
            env.occupant[p.y, p.x] = p.agent_id
        else:
            p.alive = False
    for (x, y), n in (balls or {}).items():
        env.balls[y, x] = n
    env.agents = sorted(positions)


def actions_with(env, **chosen):
    """Every live agent stays, except ``a<id>=<action>`` overrides, e.g. actions_with(env, a0=1)."""
    actions = {a: STAY for a in env.agents}
    actions.update({int(k[1:]): v for k, v in chosen.items()})
    return actions


def set_hit_chance(monkeypatch, value):
    monkeypatch.setattr("dodgeball.env.hit_chance", lambda *args, **kwargs: value)


def random_actions(env, infos, rng):
    return {a: int(rng.choice(np.flatnonzero(infos[a]["action_mask"]))) for a in env.agents}
