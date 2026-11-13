import random

import numpy as np
import pytest
import torch

from dodgeball import DodgeballEnv
from dodgeball.agents import DQNTrainer, PPOTrainer
from dodgeball.agents.opponents import OpponentPool
from dodgeball.entities import BLUE, RED
from dodgeball.policies import HeuristicPolicy


def make_trainer(cls, seed=0, **hyperparameters):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = DodgeballEnv()
    env.reset(seed=seed)
    return type(cls.__name__, (cls,), hyperparameters)(env)


def test_learner_side_alternates_and_only_learners_are_learners():
    env = DodgeballEnv()
    pool = OpponentPool(env, "dqn", p_self=0, p_heuristic=1)
    n = env.players_per_team
    for expected_team in (BLUE, RED, BLUE, RED):
        team, opponent = pool.sample_episode()
        assert team == expected_team
        assert isinstance(opponent, HeuristicPolicy)
        learners = [a for a in env.possible_agents if pool.is_learner(a)]
        assert learners == (list(range(n)) if team == BLUE else list(range(n, 2 * n)))


def test_self_play_makes_everyone_a_learner():
    env = DodgeballEnv()
    pool = OpponentPool(env, "dqn")
    pool.sample_episode()
    assert pool.self_play
    assert all(pool.is_learner(a) for a in env.possible_agents)


def test_past_falls_back_to_self_play_when_pool_is_empty():
    env = DodgeballEnv()
    pool = OpponentPool(env, "dqn", p_self=0, p_past=1)
    pool.sample_episode()
    assert pool.self_play


def test_snapshots_are_frozen_and_pool_is_bounded():
    env = DodgeballEnv()
    trainer = make_trainer(DQNTrainer)
    pool = OpponentPool(env, "dqn", pool_size=2)
    for _ in range(3):
        pool.snapshot(trainer.policy_net)
    assert len(pool.snapshots) == 2

    frozen = pool.snapshots[-1].net
    before = [p.clone() for p in frozen.parameters()]
    with torch.no_grad():
        for p in trainer.policy_net.parameters():
            p.add_(1.0)
    assert all(torch.equal(a, b) for a, b in zip(before, frozen.parameters()))


def test_dqn_stores_only_learner_transitions_against_heuristic():
    trainer = make_trainer(DQNTrainer, P_SELF=0.0, P_HEURISTIC=1.0)
    stored_teams = []
    original_push = trainer.memory.push

    def push(*args):
        stored_teams.append(trainer.opponents.learner_team)
        original_push(*args)

    trainer.memory.push = push
    trainer.train(2, log_every=100)
    assert set(stored_teams) == {BLUE, RED}             # learner played blue, then red
    # One stored transition per learner action, and the heuristic never used select_action.
    assert trainer.steps_done == len(trainer.memory)


def test_dqn_default_is_pure_self_play():
    trainer = make_trainer(DQNTrainer)
    trainer.train(1, log_every=100)
    assert trainer.opponents.self_play
    assert trainer.steps_done == len(trainer.memory)


def test_ppo_buffer_only_marks_learner_agents_valid():
    trainer = make_trainer(PPOTrainer, TIME=256, P_SELF=0.0, P_HEURISTIC=1.0)
    n = trainer.env.players_per_team
    trainer.train(1)

    # At every step only one team (the learner) is stored, never both.
    valid = trainer.buffer.valid
    blue_rows, red_rows = valid[:, :n].any(1), valid[:, n:].any(1)
    assert not (blue_rows & red_rows).any()
    assert blue_rows.any()   # the first episode's learner is blue

    advantages, returns = trainer.compute_advantages(torch.zeros(trainer.n_agents))
    assert torch.isfinite(advantages).all() and torch.isfinite(returns).all()


@pytest.mark.parametrize("cls", [DQNTrainer, PPOTrainer])
def test_mixed_training_runs(cls):
    hp = dict(P_SELF=0.4, P_HEURISTIC=0.3, P_PAST=0.3, SNAPSHOT_EVERY=1)
    if cls is PPOTrainer:
        trainer = make_trainer(cls, TIME=128, **hp)
        trainer.train(3)
    else:
        trainer = make_trainer(cls, **hp)
        trainer.train(3, log_every=100)
    assert len(trainer.opponents.snapshots) == 3
