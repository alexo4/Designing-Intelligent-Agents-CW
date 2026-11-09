import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from conftest import actions_with, random_actions, set_hit_chance, set_layout
from dodgeball.entities import BLUE
from dodgeball.env import BACK, FORWARD, LEFT, N_MOVE_ACTIONS, RIGHT, STAY, DodgeballEnv, RewardConfig, hit_chance

R = RewardConfig()
THROW_0 = N_MOVE_ACTIONS


# ---------------------------------------------------------------- reset / observations

def test_initial_layout(env):
    blue = [p for p in env.players if p.team == BLUE]
    red = [p for p in env.players if p.team != BLUE]
    assert [(p.x, p.y) for p in blue] == [(x, 0) for x in (0, 1, 2, 4, 5, 7)]
    assert [(p.x, p.y) for p in red] == [(x, 15) for x in (0, 1, 2, 4, 5, 7)]

    assert env.balls.sum() == 5
    assert sorted([env.balls[0].sum(), env.balls[15].sum()]) == [2, 3]
    for y, x in zip(*env.balls.nonzero()):
        assert env.occupant[y, x] != -1, "balls start in players' hands"


def test_observations_match_declared_space(env):
    obs, _ = env.reset(seed=1)
    for agent, o in obs.items():
        assert o.dtype == np.float32
        assert env.observation_space(agent).contains(o)


def test_mirrored_players_get_identical_observations():
    env = DodgeballEnv(balls=4)  # an even ball count makes the start position fully symmetric
    obs, infos = env.reset(seed=0)
    for i in range(env.players_per_team):
        np.testing.assert_array_equal(obs[i], obs[i + env.players_per_team])
        np.testing.assert_array_equal(infos[i]["action_mask"], infos[i + env.players_per_team]["action_mask"])


def test_same_seed_gives_same_game():
    def play(seed):
        env = DodgeballEnv()
        obs, infos = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        trace = []
        while env.agents:
            obs, rewards, *_ , infos = env.step(random_actions(env, infos, rng))
            trace.append(sorted(rewards.items()))
        return trace, env.winner

    assert play(3) == play(3)


# ---------------------------------------------------------------------------- movement

@pytest.mark.parametrize("agent, start, end, opponent", [
    (0, (3, 0), (3, 1), {6: (7, 8)}),     # blue forward = down the grid
    (6, (3, 15), (3, 14), {0: (7, 7)}),   # red forward = up the grid
])
def test_forward_moves_towards_half_way_line(env, agent, start, end, opponent):
    set_layout(env, {agent: start, **opponent})
    env.step(actions_with(env, **{f"a{agent}": FORWARD}))
    assert (env.players[agent].x, env.players[agent].y) == end


def test_cannot_cross_half_way_line(env):
    set_layout(env, {0: (3, 7), 6: (3, 8)})
    assert env.action_mask(0)[FORWARD] == 0
    assert env.action_mask(6)[FORWARD] == 0


def test_cannot_leave_court_or_enter_occupied_square(env):
    set_layout(env, {0: (0, 0), 1: (1, 0), 6: (5, 15)})
    mask = env.action_mask(0)
    assert mask[BACK] == 0 and mask[LEFT] == 0 and mask[RIGHT] == 0
    assert mask[FORWARD] == 1 and mask[STAY] == 1


def test_invalid_action_is_treated_as_stay(env):
    set_layout(env, {0: (0, 0), 6: (5, 15)})
    env.step(actions_with(env, a0=LEFT))
    assert (env.players[0].x, env.players[0].y) == (0, 0)


def test_picking_up_and_carrying_balls(env):
    set_layout(env, {0: (3, 3), 6: (5, 15)}, balls={(3, 4): 1})
    _, rewards, *_ = env.step(actions_with(env, a0=FORWARD))
    assert rewards[0] == pytest.approx(R.pickup)
    assert env.balls[4, 3] == 1

    env.step(actions_with(env, a0=RIGHT))
    assert env.balls[4, 3] == 0 and env.balls[4, 4] == 1


# ---------------------------------------------------------------------------- throwing

def test_throw_actions_need_a_ball_and_match_alive_opponents(env):
    set_layout(env, {0: (0, 7), 1: (1, 7), 6: (0, 8), 7: (1, 8), 8: (2, 8)}, balls={(0, 7): 1})
    assert list(env.action_mask(0)[THROW_0:]) == [1, 1, 1, 0, 0, 0]
    assert list(env.action_mask(1)[THROW_0:]) == [0] * 6


def test_red_never_targets_own_team(env):
    """Regression: red's target search used to include row 8, which is red's own half."""
    set_layout(env, {0: (0, 7), 1: (5, 2), 6: (0, 8), 7: (1, 8), 8: (2, 10)}, balls={(2, 10): 1})
    targets = env.opponents_in_target_order(8)
    assert [p.agent_id for p in targets] == [0, 1]


@pytest.mark.parametrize("thrower, thrower_pos, opponents, expected", [
    (0, (2, 2), {6: (2, 12), 7: (2, 8), 8: (0, 15)}, [8, 7, 6]),  # by column, then nearest the line
    (6, (2, 12), {0: (2, 1), 1: (2, 7), 2: (0, 0)}, [2, 1, 0]),   # same ordering from red's side
])
def test_target_order(env, thrower, thrower_pos, opponents, expected):
    set_layout(env, {thrower: thrower_pos, **opponents})
    assert [p.agent_id for p in env.opponents_in_target_order(thrower)] == expected


def test_hit_eliminates_target_and_moves_ball(env, monkeypatch):
    set_hit_chance(monkeypatch, 1.0)
    set_layout(env, {0: (3, 7), 1: (0, 0), 6: (3, 8), 7: (6, 15)}, balls={(3, 7): 1})
    _, rewards, terms, _, _ = env.step(actions_with(env, a0=THROW_0))  # target 0 = agent 6

    assert not env.players[6].alive and terms[6] and 6 not in env.agents
    assert rewards[0] == pytest.approx(R.hit)
    assert rewards[6] == pytest.approx(R.eliminated)
    assert env.balls[7, 3] == 0 and env.balls[8, 3] == 1


def test_miss_leaves_target_and_moves_ball(env, monkeypatch):
    set_hit_chance(monkeypatch, 0.0)
    set_layout(env, {0: (3, 7), 6: (3, 8)}, balls={(3, 7): 1})
    _, rewards, terms, _, _ = env.step(actions_with(env, a0=THROW_0))

    assert env.players[6].alive and not terms[6]
    assert rewards[0] == pytest.approx(R.miss)
    assert env.balls[8, 3] == 1


def test_combined_throws_and_simultaneous_hits(env, monkeypatch):
    set_hit_chance(monkeypatch, 1.0)
    set_layout(env, {0: (3, 7), 1: (4, 7), 2: (0, 0), 6: (3, 8), 7: (7, 15)},
               balls={(3, 7): 1, (4, 7): 1, (3, 8): 1})
    # Blue 0 and 1 both throw at red 6, and red 6 throws at blue 0, all in the same step.
    # Red's targets in order are [2 (column 0), 0 (column 3), 1 (column 4)].
    assert [p.agent_id for p in env.opponents_in_target_order(6)] == [2, 0, 1]
    _, rewards, terms, _, _ = env.step(actions_with(env, a0=THROW_0, a1=THROW_0, a6=THROW_0 + 1))

    assert terms[0] and terms[6], "simultaneous throws both resolve"
    assert env.balls[8, 3] == 2, "both blue balls land on red 6's square"
    assert rewards[1] == pytest.approx(R.hit)


def test_target_throwing_its_only_ball_cannot_block(env, monkeypatch):
    calls = []
    monkeypatch.setattr("dodgeball.env.hit_chance", lambda d, n, blocking, max_dist: calls.append(blocking) or 0.0)
    set_layout(env, {0: (3, 7), 6: (3, 8), 7: (5, 8)}, balls={(3, 7): 1, (3, 8): 1, (5, 8): 1})
    env.step(actions_with(env, a0=THROW_0, a6=THROW_0))
    assert calls == [False, False]


def test_hit_chance_table():
    max_dist = 15
    assert hit_chance(1, 1, False, max_dist) == pytest.approx(0.8)
    assert hit_chance(1, 1, True, max_dist) == pytest.approx(0.6)
    assert hit_chance(14, 1, False, max_dist) == 0
    singles = [hit_chance(d, 1, False, max_dist) for d in range(1, 16)]
    assert singles == sorted(singles, reverse=True), "chance never rises with distance"
    for d in range(16):
        for n in range(1, 7):
            for blocking in (False, True):
                assert 0.0 <= hit_chance(d, n, blocking, max_dist) <= 1.0


def test_throwing_resets_possession_timer(env, monkeypatch):
    set_hit_chance(monkeypatch, 0.0)
    set_layout(env, {0: (3, 7), 6: (3, 8)}, balls={(3, 7): 1})
    env.possession_timer[BLUE] = 3
    env.step(actions_with(env, a0=THROW_0))
    # The throw resets blue's timer; the ball then sits on red's half so red's timer ticks.
    assert env.possession_timer[BLUE] == env.possession_limit
    assert env.possession_timer[1 - BLUE] == env.possession_limit - 1


# -------------------------------------------------------------------------- possession

def test_possession_timeout_only_eliminates_majority_team(env):
    """Regression: red's timeout used to scan rows 1-15, eliminating blue ball holders too."""
    set_layout(env, {0: (3, 3), 1: (0, 0), 6: (3, 12), 7: (5, 12), 8: (7, 15)},
               balls={(3, 3): 1, (3, 12): 1, (5, 12): 1})
    env.possession_timer = [10, 1]
    _, rewards, terms, _, _ = env.step(actions_with(env))

    assert terms[6] and terms[7] and not terms[8]
    assert not terms[0], "blue ball holder is unaffected by red's timeout"
    assert rewards[6] == pytest.approx(R.eliminated)


def test_possession_timer_only_ticks_for_majority_and_not_on_tie():
    env = DodgeballEnv(balls=4)
    env.reset(seed=0)
    env.step({a: STAY for a in env.agents})
    assert env.possession_timer == [env.possession_limit] * 2

    set_layout(env, {0: (3, 3), 6: (3, 12)}, balls={(3, 3): 3, (3, 12): 1})
    env.step(actions_with(env))
    assert env.possession_timer == [env.possession_limit - 1, env.possession_limit]


# ---------------------------------------------------------------------------- game end

def test_eliminating_a_team_ends_the_game(env, monkeypatch):
    set_hit_chance(monkeypatch, 1.0)
    set_layout(env, {0: (3, 7), 1: (0, 0), 6: (3, 8)}, balls={(3, 7): 1})
    _, rewards, terms, truncs, infos = env.step(actions_with(env, a0=THROW_0))

    assert all(terms.values()) and not any(truncs.values())
    assert env.winner == "blue" and infos[0]["winner"] == "blue"
    assert env.agents == []
    assert rewards[0] == pytest.approx(R.hit + R.win)
    assert rewards[1] == pytest.approx(R.win)
    assert rewards[6] == pytest.approx(R.eliminated)


def test_time_limit_truncates_and_more_players_wins(env):
    """Regression: the losing team used to get True/False instead of a negative reward."""
    set_layout(env, {0: (0, 0), 1: (1, 0), 6: (0, 15)})
    env.steps = env.max_steps - 1
    _, rewards, terms, truncs, _ = env.step(actions_with(env))

    assert all(truncs.values()) and not any(terms.values())
    assert env.winner == "blue"
    assert rewards[0] == pytest.approx(R.win)
    assert rewards[6] == pytest.approx(R.lose) and R.lose < 0


def test_time_limit_draw(env):
    set_layout(env, {0: (0, 0), 6: (0, 15)})
    env.steps = env.max_steps - 1
    _, rewards, *_ = env.step(actions_with(env))
    assert env.winner == "draw"
    assert rewards[0] == pytest.approx(R.draw)


# --------------------------------------------------------------------- API / fairness

def test_pettingzoo_parallel_api():
    parallel_api_test(DodgeballEnv(), num_cycles=1000)


def test_random_play_is_fair():
    """Regression: red used to win ~54% of random games vs ~45% for blue."""
    env = DodgeballEnv()
    rng = np.random.default_rng(0)
    wins = {"blue": 0, "red": 0, "draw": 0}
    games = 1000
    for g in range(games):
        _, infos = env.reset(seed=g)
        while env.agents:
            _, _, _, _, infos = env.step(random_actions(env, infos, rng))
        wins[env.winner] += 1
    assert abs(wins["blue"] - wins["red"]) / games < 0.06, wins
