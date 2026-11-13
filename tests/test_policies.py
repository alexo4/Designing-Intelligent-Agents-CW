import numpy as np
import pytest

from dodgeball.agents import DQN, PPO
from dodgeball.env import BACK, FORWARD, N_MOVE_ACTIONS, RIGHT, STAY, DodgeballEnv
from dodgeball.policies import HeuristicPolicy, NetworkPolicy, RandomPolicy, head_to_head, play_game
from tests.conftest import set_layout


def heuristic_action(env, agent):
    return HeuristicPolicy(env.grid_size).act(env._observe(agent), env.action_mask(agent))


class ValidActionChecker:
    """Wraps a policy and fails if it ever picks an action the mask forbids."""

    def __init__(self, policy):
        self.policy = policy

    def act(self, obs, mask):
        action = self.policy.act(obs, mask)
        assert mask[action] == 1, f"{self.policy.name} chose invalid action {action}"
        return action


@pytest.mark.parametrize("make", [
    lambda env: HeuristicPolicy(env.grid_size),
    lambda env: NetworkPolicy(DQN(env.n_actions, env.observation_space(0).shape[0]), "dqn"),
    lambda env: NetworkPolicy(PPO(env.n_actions, env.observation_space(0).shape[0]), "ppo"),
])
def test_policies_only_choose_valid_actions(make):
    env = DodgeballEnv()
    policy = ValidActionChecker(make(env))
    opponent = ValidActionChecker(RandomPolicy(0))
    for seed in range(10):
        play_game(env, policy, opponent, seed=seed)
        play_game(env, opponent, policy, seed=seed)


@pytest.mark.parametrize("thrower, near_target, far_target", [
    (6, 1, 0),   # red throws at blue
    (0, 7, 6),   # blue throws at red
])
def test_heuristic_throws_at_the_closest_opponent(env, thrower, near_target, far_target):
    if thrower == 6:
        positions = {thrower: (3, 8), near_target: (3, 7), far_target: (0, 0)}
        balls = {(3, 8): 1}
    else:
        positions = {thrower: (3, 7), near_target: (3, 8), far_target: (0, 15)}
        balls = {(3, 7): 1}
    set_layout(env, positions, balls)
    action = heuristic_action(env, thrower)
    assert action >= N_MOVE_ACTIONS
    assert env.opponents_in_target_order(thrower)[action - N_MOVE_ACTIONS].agent_id == near_target


def test_heuristic_advances_when_the_target_is_out_of_range(env):
    set_layout(env, {0: (3, 0), 6: (3, 15)}, balls={(3, 0): 1})
    assert heuristic_action(env, 0) == FORWARD


def test_heuristic_throws_anyway_when_possession_is_about_to_expire(env):
    set_layout(env, {0: (3, 0), 6: (3, 15)}, balls={(3, 0): 1})
    env.possession_timer[0] = 1
    assert heuristic_action(env, 0) >= N_MOVE_ACTIONS


def test_heuristic_fetches_loose_balls(env):
    set_layout(env, {0: (2, 0), 6: (2, 15)}, balls={(5, 0): 1, (2, 12): 1})
    assert heuristic_action(env, 0) == RIGHT
    assert heuristic_action(env, 6) == FORWARD   # egocentric: towards the half-way line


def test_heuristic_retreats_when_opponents_hold_the_balls(env):
    set_layout(env, {0: (2, 5), 6: (2, 10)}, balls={(2, 10): 1})
    assert heuristic_action(env, 0) == BACK


def test_heuristic_is_much_stronger_than_random():
    env = DodgeballEnv()
    result = head_to_head(env, HeuristicPolicy(env.grid_size), RandomPolicy(0), games=100)
    assert result["win"] > 0.85


def test_head_to_head_alternates_sides():
    env = DodgeballEnv()
    # Random vs random, from either side, should be close to 50/50.
    result = head_to_head(env, RandomPolicy(1), RandomPolicy(2), games=400)
    assert abs(result["win"] - result["loss"]) < 0.15
    assert np.isclose(result["win"] + result["loss"] + result["draw"], 1.0)


def test_patient_heuristic_holds_a_bad_shot_at_the_line(env):
    # At the half-way line (can't move forward), with the only target ~10 squares away (40% shot).
    set_layout(env, {0: (7, 7), 6: (0, 15)}, balls={(7, 7): 1})
    env.possession_timer[0] = env.possession_limit
    obs, mask = env._observe(0), env.action_mask(0)
    assert HeuristicPolicy(env.grid_size).act(obs, mask) >= N_MOVE_ACTIONS      # throws anyway
    assert HeuristicPolicy(env.grid_size, patient=True).act(obs, mask) == STAY  # waits
    env.possession_timer[0] = 1                                                 # ...until forced
    assert HeuristicPolicy(env.grid_size, patient=True).act(env._observe(0), mask) >= N_MOVE_ACTIONS
