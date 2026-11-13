"""Reward options in RewardConfig, and how the trainers route rewards to eliminated players."""

import random
from collections import Counter

import numpy as np
import pytest
import torch

from conftest import actions_with, set_hit_chance, set_layout
from dodgeball.agents import DQNTrainer, PPOTrainer
from dodgeball.agents.dqn import ReplayMemory
from dodgeball.env import BACK, FORWARD, LEFT, N_MOVE_ACTIONS, RIGHT, STAY, DodgeballEnv, RewardConfig
from dodgeball.policies import HeuristicPolicy, NetworkPolicy, RandomPolicy, head_to_head, play_game

THROW_0 = N_MOVE_ACTIONS


def make_env(**reward_fields):
    env = DodgeballEnv(rewards=RewardConfig(**reward_fields))
    env.reset(seed=0)
    return env


# --------------------------------------------------------------------- pickup

def test_pickup_cannot_be_farmed_by_stepping_on_and_off():
    env = make_env()
    set_layout(env, {0: (3, 3), 6: (3, 12)}, balls={(4, 3): 1})
    _, rewards, _, _, infos = env.step(actions_with(env, a0=RIGHT))
    assert rewards[0] == pytest.approx(env.rewards.pickup) and infos[0]["picked_up"]

    # The ball moves with the player, so stepping back and forth never finds a loose ball.
    for action in (LEFT, RIGHT, LEFT, RIGHT, FORWARD, BACK):
        _, rewards, _, _, infos = env.step(actions_with(env, a0=action))
        assert rewards[0] == 0.0 and not infos[0]["picked_up"]
    assert env.balls.sum() == 1 and env.balls[env.players[0].y, env.players[0].x] == 1


def test_pickup_only_on_a_real_move_onto_balls():
    env = make_env()
    # Ball under a teammate: the square is occupied, so the move fails and there's no reward.
    set_layout(env, {0: (3, 3), 1: (4, 3), 6: (3, 12)}, balls={(4, 3): 1})
    _, rewards, _, _, infos = env.step(actions_with(env, a0=RIGHT))
    assert rewards[0] == 0.0 and not infos[0]["picked_up"]
    # Staying on a square with a ball isn't a pickup either.
    _, rewards, _, _, _ = env.step(actions_with(env))
    assert rewards[1] == 0.0


def test_pickup_rewards_over_random_games_match_balls_collected():
    """Every pickup reward corresponds to moving onto a square that had loose balls."""
    env = make_env()
    rng = np.random.default_rng(0)
    for seed in range(20):
        obs, infos = env.reset(seed=seed)
        while env.agents:
            before = env.balls.copy()
            actions = {a: int(rng.choice(np.flatnonzero(infos[a]["action_mask"]))) for a in env.agents}
            obs, rewards, _, _, infos = env.step(actions)
            for a, info in infos.items():
                if info["picked_up"]:
                    p = env.players[a]
                    assert before[p.y, p.x] > 0


# ------------------------------------------------------------- end of game

def test_timeout_on_numbers_uses_its_own_reward():
    env = make_env(win_on_numbers=1.0, lose_on_numbers=-1.0)
    set_layout(env, {0: (0, 0), 1: (1, 0), 6: (0, 15)})
    env.steps = env.max_steps - 1
    _, rewards, *_ = env.step(actions_with(env))
    assert env.winner == "blue"
    assert rewards[0] == pytest.approx(1.0) and rewards[6] == pytest.approx(-1.0)


def test_timeout_draw_stays_zero():
    env = make_env(win_on_numbers=1.0, lose_on_numbers=-1.0)
    set_layout(env, {0: (0, 0), 6: (0, 15)})
    env.steps = env.max_steps - 1
    _, rewards, *_ = env.step(actions_with(env))
    assert env.winner == "draw" and rewards[0] == 0.0 and rewards[6] == 0.0


def test_elimination_win_still_uses_full_reward_with_timeout_option(monkeypatch):
    set_hit_chance(monkeypatch, 1.0)
    env = make_env(win_on_numbers=1.0, lose_on_numbers=-1.0)
    set_layout(env, {0: (3, 7), 6: (3, 8)}, balls={(3, 7): 1})
    _, rewards, *_ = env.step(actions_with(env, a0=THROW_0))
    assert rewards[0] == pytest.approx(env.rewards.hit + env.rewards.win)


def test_default_gives_no_terminal_reward_to_eliminated_players(monkeypatch):
    set_hit_chance(monkeypatch, 1.0)
    env = make_env()
    set_layout(env, {0: (3, 7), 1: (0, 0), 6: (3, 8), 7: (5, 8)}, balls={(3, 7): 1, (3, 8): 1})
    # Blue 0 hits red 6, so 6 is eliminated before the game ends.
    env.step(actions_with(env, a0=THROW_0))
    assert not env.players[6].alive
    env.steps = env.max_steps - 1
    env.step(actions_with(env))
    assert env.late_rewards == {}


def test_terminal_reward_reaches_players_eliminated_earlier(monkeypatch):
    set_hit_chance(monkeypatch, 1.0)
    env = make_env(terminal_to_eliminated=True)
    set_layout(env, {0: (3, 7), 1: (0, 0), 6: (3, 8), 7: (5, 8)}, balls={(3, 7): 1})
    env.step(actions_with(env, a0=THROW_0))       # 6 is out, game continues (7 left)
    assert not env.players[6].alive and 6 not in env.agents and env.late_rewards == {}

    env.steps = env.max_steps - 1                  # 2 blue vs 1 red at time-out
    _, rewards, *_ = env.step(actions_with(env))
    assert env.winner == "blue"
    # (set_layout marks unused players as eliminated, so they get late rewards too)
    assert env.late_rewards[6] == pytest.approx(env.rewards.lose)
    assert all(not env.players[a].alive for a in env.late_rewards)
    assert rewards[7] == pytest.approx(env.rewards.lose)
    assert rewards[0] == pytest.approx(env.rewards.win)


def test_player_eliminated_on_the_last_step_gets_terminal_reward(monkeypatch):
    set_hit_chance(monkeypatch, 1.0)
    for flag, expected in ((False, -1.0), (True, -1.0 - 2.0)):
        env = make_env(terminal_to_eliminated=flag)
        set_layout(env, {0: (3, 7), 6: (3, 8)}, balls={(3, 7): 1})
        _, rewards, *_ = env.step(actions_with(env, a0=THROW_0))
        assert rewards[6] == pytest.approx(expected)


# --------------------------------------------------------------------- idle

def test_idle_penalty_only_when_another_action_was_possible():
    env = make_env(idle=-0.01)
    set_layout(env, {0: (3, 3), 6: (3, 12)})
    _, rewards, *_ = env.step(actions_with(env))
    assert rewards[0] == pytest.approx(-0.01)
    _, rewards, *_ = env.step(actions_with(env, a0=RIGHT, a6=LEFT))
    assert rewards[0] == 0.0 and rewards[6] == 0.0


def test_idle_penalty_off_by_default():
    env = make_env()
    set_layout(env, {0: (3, 3), 6: (3, 12)})
    _, rewards, *_ = env.step(actions_with(env))
    assert rewards[0] == 0.0


# ------------------------------------------------------------ trainers

def test_replay_memory_add_reward_tracks_ids_after_overflow():
    memory = ReplayMemory(3)
    ids = [memory.push(i, 0, None, 0.0, None) for i in range(5)]   # 0 and 1 dropped
    memory.add_reward(ids[3], 2.0)
    memory.add_reward(ids[0], 5.0)                                # gone: ignored
    assert [t.reward for t in memory.memory] == [0.0, 2.0, 0.0]
    assert [t.state for t in memory.memory] == [2, 3, 4]


def seeded(cls, env, **hp):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    return type(cls.__name__, (cls,), hp)(env)


def test_dqn_adds_late_rewards_to_the_last_transition():
    env = DodgeballEnv(rewards=RewardConfig(terminal_to_eliminated=True))
    env.reset(seed=0)
    trainer = seeded(DQNTrainer, env, MEMORY_SIZE=100_000)
    added = Counter()
    original = trainer.memory.add_reward

    def spy(tid, reward):
        added[tid] += 1
        assert trainer.memory.memory[tid - (trainer.memory.pushed - len(trainer.memory))].next_state is None
        original(tid, reward)

    trainer.memory.add_reward = spy
    trainer.train(5, log_every=100)
    assert sum(added.values()) > 0                       # some players were eliminated early
    assert all(v == 1 for v in added.values())           # at most once per transition


def test_ppo_adds_late_rewards_and_normalises_advantages():
    env = DodgeballEnv(rewards=RewardConfig(terminal_to_eliminated=True))
    env.reset(seed=0)
    trainer = seeded(PPOTrainer, env, TIME=512, NORM_ADV=True)
    trainer.train(1)
    buf = trainer.buffer
    # Every late reward lands on a step where that agent was done.
    assert torch.all(buf.dones[buf.valid & (buf.rewards.abs() >= 2.0)] == 1)


# ---------------------------------------------------- evaluation helpers

def test_greedy_ppo_policy_is_deterministic_and_valid():
    env = DodgeballEnv()
    obs, infos = env.reset(seed=0)
    from dodgeball.agents import PPO
    policy = NetworkPolicy(PPO(env.n_actions, env.observation_space(0).shape[0]), "ppo", greedy=True)
    for a in env.agents:
        mask = infos[a]["action_mask"]
        first = policy.act(obs[a], mask)
        assert mask[first] == 1
        assert all(policy.act(obs[a], mask) == first for _ in range(5))


def test_variant_is_a_named_stronger_heuristic():
    v = HeuristicPolicy.variant()
    assert v.name == "heuristic_variant" and v.min_chance == 0.7 and v.panic_timer == 1


def test_head_to_head_reports_behaviour():
    env = DodgeballEnv()
    res = head_to_head(env, HeuristicPolicy(), RandomPolicy(0), games=20)
    assert res["throws_per_game"] > 0 and res["pickups_per_game"] >= 0
    assert 0 <= res["idle_frac"] <= 1


def test_dqn_stops_after_the_episode_that_reaches_the_step_budget():
    env = DodgeballEnv()
    env.reset(seed=0)
    trainer = seeded(DQNTrainer, env, MEMORY_SIZE=10_000)
    rewards = trainer.train(10 ** 6, log_every=100, max_env_steps=50)
    assert trainer.env_steps >= 50
    assert trainer.env_steps - 50 < env.max_steps          # stopped within one episode
    assert len(rewards) < 10
