"""Multi-agent dodgeball environment (PettingZoo ParallelEnv).

Court
    A ``(2 * grid_size) x grid_size`` grid. Team blue (agents ``0 .. n-1``) owns the top half
    (rows ``0 .. grid_size-1``) and team red (agents ``n .. 2n-1``) owns the bottom half.
    Players can never leave their own half.

Egocentric frame
    Observations and movement actions are expressed from the acting team's point of view:
    the red team's view is flipped vertically so that every agent sees its own half at the
    top and "forward" always means towards the half-way line. A blue and a red player in
    mirrored situations therefore receive identical observations, so a single shared
    network can play either side.

Actions (``5 + n`` discrete actions)
    0       stay
    1       forward  (towards the half-way line)
    2       back     (away from the half-way line)
    3       right    (x + 1)
    4       left     (x - 1)
    5 + k   throw a held ball at opponent k, where opponents are ordered column by column
            (left to right) and, within a column, nearest the half-way line first.
    Invalid actions (see ``infos[agent]["action_mask"]``) are treated as "stay".

Step order
    1. Throw targets are fixed from the state the agents observed.
    2. Moves are applied in agent-id order. Moving onto a square with balls picks them up;
       a player carries every ball on their square.
    3. All throws are resolved simultaneously. Throws at the same target are combined.
       Thrown balls land on the target's square.
    4. Possession: the team with more balls on its half has its timer reduced by one. When
       it reaches zero, every player on that team holding a ball is eliminated. A team's
       timer resets whenever it throws.
    5. The game ends (termination) when a team has no players left, or is truncated after
       ``max_steps``. The team with more players left wins.
"""

from dataclasses import dataclass
from math import ceil, floor, sqrt

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from dodgeball.entities import BLUE, RED, TEAM_NAMES, Player

STAY, FORWARD, BACK, RIGHT, LEFT = range(5)
N_MOVE_ACTIONS = 5

EMPTY = -1  # value of ``occupant`` for squares with no player

GRID_CHANNELS = 4  # teammates, opponents, balls, self
N_EXTRA_FEATURES = 8


@dataclass(frozen=True)
class RewardConfig:
    hit: float = 1.0          # thrower, when their throw eliminates an opponent
    miss: float = -0.05       # thrower, when their throw misses
    eliminated: float = -1.0  # player who is hit or loses the ball by the possession rule
    pickup: float = 0.1       # moving onto a square that has balls
    win: float = 2.0          # every player still in the game when their team wins
    lose: float = -2.0        # every player still in the game when their team loses
    draw: float = 0.0
    # Result on time-out (more players left, but nobody eliminated). None = same as win/lose.
    win_on_numbers: float | None = None
    lose_on_numbers: float | None = None
    # Also give the win/lose reward to players eliminated earlier in the game. They are no
    # longer in ``env.agents``, so it is reported in ``env.late_rewards`` for the trainers to
    # add to that player's last transition.
    terminal_to_eliminated: bool = False
    idle: float = 0.0         # choosing to stay put when another action was available


def hit_chance(dist, n_throwers, target_blocking, max_dist):
    """Probability that a (combined) throw hits.

    ``dist`` is the shortest distance from any thrower to the target; ``target_blocking`` is
    True if the target is holding a ball that they are not throwing this step.
    """
    multi = n_throwers > 1
    if dist <= floor(max_dist * 0.3):
        chance = 0.8 - (0.2 if target_blocking else 0)
    elif dist <= floor(max_dist * 0.4):
        chance = 0.7 + (0.1 if multi else 0) - (0.2 if target_blocking else 0)
    elif dist <= ceil(max_dist * 0.5):
        chance = 0.6 * (1.1 ** n_throwers if multi else 1) - (0.3 if target_blocking else 0)
    elif dist <= floor(max_dist * 0.7):
        chance = 0.4 * (1.1 ** n_throwers if multi else 1) - (0.3 if target_blocking else 0)
    elif dist <= floor(max_dist * 0.8):
        chance = 0.3 * (1.1 ** n_throwers if multi else 1) - (0.2 if target_blocking else 0)
    elif dist <= floor(max_dist * 0.9):
        chance = 0 if target_blocking else 0.1 * (1.1 ** n_throwers if multi else 1)
    else:
        chance = 0
    return min(max(chance, 0.0), 1.0)


class DodgeballEnv(ParallelEnv):
    metadata = {"name": "dodgeball_v1", "render_modes": []}

    def __init__(self, grid_size=8, players_per_team=6, balls=5, max_steps=300,
                 possession_limit=10, rewards=RewardConfig()):
        super().__init__()
        if players_per_team > grid_size:
            raise ValueError("players_per_team cannot exceed grid_size (players start on one row)")

        self.grid_size = grid_size
        self.height = 2 * grid_size
        self.players_per_team = players_per_team
        self.n_balls = balls
        self.max_steps = max_steps
        self.possession_limit = possession_limit
        self.rewards = rewards
        self.render_mode = None

        self.possible_agents = list(range(2 * players_per_team))
        self.agents = []
        self.n_actions = N_MOVE_ACTIONS + players_per_team
        self.grid_shape = (GRID_CHANNELS, self.height, grid_size)
        obs_size = GRID_CHANNELS * self.height * grid_size + N_EXTRA_FEATURES

        self.observation_spaces = {
            a: spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32) for a in self.possible_agents
        }
        self.action_spaces = {a: spaces.Discrete(self.n_actions) for a in self.possible_agents}

        self._rng = np.random.default_rng()
        self.players = []
        self.occupant = None    # (height, grid_size) agent id on each square, or EMPTY
        self.balls = None       # (height, grid_size) number of balls on each square
        self.steps = 0
        self.possession_timer = [possession_limit, possession_limit]
        self.winner = None      # "blue", "red" or "draw" once the game is over
        self.late_rewards = {}  # agent -> terminal reward for players already eliminated

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # ------------------------------------------------------------------ setup

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        n, g = self.players_per_team, self.grid_size
        self.occupant = np.full((self.height, g), EMPTY, dtype=np.int64)
        self.balls = np.zeros((self.height, g), dtype=np.int64)
        self.steps = 0
        self.possession_timer = [self.possession_limit, self.possession_limit]
        self.winner = None
        self.late_rewards = {}

        # Spread each team evenly along its back row; red mirrors blue.
        columns = [floor(i * (g - 1) / (n - 1)) if n > 1 else g // 2 for i in range(n)]
        self.players = []
        for team, row in ((BLUE, 0), (RED, self.height - 1)):
            for i, x in enumerate(columns):
                agent = team * n + i
                self.players.append(Player(agent, team, x, row))
                self.occupant[row, x] = agent

        # Split the balls between the teams (a random team gets the odd one) and give them
        # to players spread along each back row.
        blue_balls = self.n_balls // 2
        if self.n_balls % 2 and self._rng.random() < 0.5:
            blue_balls += 1
        for team, row, count in ((BLUE, 0, blue_balls), (RED, self.height - 1, self.n_balls - blue_balls)):
            for idx in self._spread(count, n):
                self.balls[row, columns[idx]] += 1

        self.agents = list(self.possible_agents)
        observations = {a: self._observe(a) for a in self.agents}
        infos = {a: {"action_mask": self.action_mask(a)} for a in self.agents}
        return observations, infos

    @staticmethod
    def _spread(count, n):
        """Indices of ``count`` items spread evenly over ``n`` slots (both ends included)."""
        if count <= 0:
            return []
        if count == 1:
            return [n // 2]
        step = (n - 1) / (count - 1)
        return [min(ceil(i * step), n - 1) for i in range(count)]

    # ------------------------------------------------------- frame / helpers

    def _view_y(self, y, team):
        """Convert a world row to the given team's egocentric row (own half on top)."""
        return y if team == BLUE else self.height - 1 - y

    def _forward_dy(self, team):
        return 1 if team == BLUE else -1

    def _half(self, y):
        return BLUE if y < self.grid_size else RED

    def _alive(self, team):
        return [p for p in self.players if p.team == team and p.alive]

    def opponents_in_target_order(self, agent):
        """Alive opponents in the order used by throw actions (see module docstring)."""
        me = self.players[agent]
        opponents = self._alive(1 - me.team)
        return sorted(opponents, key=lambda p: (p.x, self._view_y(p.y, me.team)))

    def _move_target(self, player, action):
        dy = self._forward_dy(player.team)
        dx, dy = {FORWARD: (0, dy), BACK: (0, -dy), RIGHT: (1, 0), LEFT: (-1, 0)}[action]
        return player.x + dx, player.y + dy

    def _can_move_to(self, player, x, y):
        return (0 <= x < self.grid_size and 0 <= y < self.height
                and self._half(y) == player.team and self.occupant[y, x] == EMPTY)

    # ------------------------------------------------------------ observation

    def action_mask(self, agent):
        me = self.players[agent]
        mask = np.zeros(self.n_actions, dtype=np.int8)
        if not me.alive:
            mask[STAY] = 1
            return mask

        mask[STAY] = 1
        for action in (FORWARD, BACK, RIGHT, LEFT):
            mask[action] = self._can_move_to(me, *self._move_target(me, action))

        if self.balls[me.y, me.x] > 0:
            n_targets = len(self._alive(1 - me.team))
            mask[N_MOVE_ACTIONS:N_MOVE_ACTIONS + n_targets] = 1
        return mask

    def _observe(self, agent):
        me = self.players[agent]
        team = me.team
        grid = np.zeros(self.grid_shape, dtype=np.float32)

        for p in self.players:
            if p.alive:
                grid[0 if p.team == team else 1, p.y, p.x] = 1.0
        grid[2] = self.balls > 0
        if me.alive:
            grid[3, me.y, me.x] = 1.0
        if team == RED:
            grid = grid[:, ::-1, :]

        own_half = slice(0, self.grid_size) if team == BLUE else slice(self.grid_size, self.height)
        opp_half = slice(self.grid_size, self.height) if team == BLUE else slice(0, self.grid_size)
        n = self.players_per_team
        total_balls = max(self.n_balls, 1)
        extra = np.array([
            1.0 - self.steps / self.max_steps,
            self.possession_timer[team] / self.possession_limit,
            self.possession_timer[1 - team] / self.possession_limit,
            self.balls[own_half].sum() / total_balls,
            self.balls[opp_half].sum() / total_balls,
            float(me.alive and self.balls[me.y, me.x] > 0),
            len(self._alive(team)) / n,
            len(self._alive(1 - team)) / n,
        ], dtype=np.float32)

        return np.concatenate([grid.ravel(), extra])

    # ------------------------------------------------------------------- step

    def step(self, actions):
        acting = [a for a in self.agents if a in actions]
        rewards = {a: 0.0 for a in self.agents}
        eliminated = set()
        picked_up = set()
        self.late_rewards = {}

        # Replace invalid actions with STAY.
        masks = {a: self.action_mask(a) for a in acting}
        actions = {a: (int(actions[a]) if masks[a][int(actions[a])] else STAY) for a in acting}

        if self.rewards.idle:
            for a, act in actions.items():
                if act == STAY and masks[a][1:].any():
                    rewards[a] += self.rewards.idle

        # 1. Fix throw targets from the observed state.
        throws = {}
        for a, act in actions.items():
            if act >= N_MOVE_ACTIONS:
                throws[a] = self.opponents_in_target_order(a)[act - N_MOVE_ACTIONS].agent_id

        # 2. Moves, in agent-id order.
        for a in sorted(actions):
            if STAY < actions[a] < N_MOVE_ACTIONS:
                p = self.players[a]
                x, y = self._move_target(p, actions[a])
                if self._can_move_to(p, x, y):
                    if self.balls[y, x] > 0:
                        rewards[a] += self.rewards.pickup
                        picked_up.add(a)
                    self.balls[y, x] += self.balls[p.y, p.x]
                    self.balls[p.y, p.x] = 0
                    self.occupant[p.y, p.x] = EMPTY
                    self.occupant[y, x] = a
                    p.x, p.y = x, y

        # 3. Throws, resolved simultaneously.
        by_target = {}
        for thrower, target in throws.items():
            by_target.setdefault(target, []).append(thrower)

        blocking = {}
        for target in by_target:
            t = self.players[target]
            held_after_throwing = self.balls[t.y, t.x] - (1 if target in throws else 0)
            blocking[target] = held_after_throwing > 0

        for target, throwers in by_target.items():
            t = self.players[target]
            dist = min(floor(sqrt((self.players[s].x - t.x) ** 2 + (self.players[s].y - t.y) ** 2))
                       for s in throwers)
            chance = hit_chance(dist, len(throwers), blocking[target], max_dist=self.height - 1)
            hit = self._rng.random() < chance

            for s in throwers:
                thrower = self.players[s]
                self.balls[thrower.y, thrower.x] -= 1
                self.possession_timer[thrower.team] = self.possession_limit
                rewards[s] += self.rewards.hit if hit else self.rewards.miss
            self.balls[t.y, t.x] += len(throwers)
            if hit:
                eliminated.add(target)

        for a in eliminated:
            self._eliminate(a)

        # 4. Possession rule.
        blue_balls = self.balls[:self.grid_size].sum()
        red_balls = self.balls[self.grid_size:].sum()
        if blue_balls != red_balls:
            team = BLUE if blue_balls > red_balls else RED
            self.possession_timer[team] -= 1
            if self.possession_timer[team] <= 0:
                self.possession_timer[team] = self.possession_limit
                for p in self._alive(team):
                    if self.balls[p.y, p.x] > 0:
                        eliminated.add(p.agent_id)
                        self._eliminate(p.agent_id)

        for a in eliminated:
            rewards[a] += self.rewards.eliminated

        # 5. End of game.
        self.steps += 1
        blue_left, red_left = len(self._alive(BLUE)), len(self._alive(RED))
        game_over = blue_left == 0 or red_left == 0
        timed_out = not game_over and self.steps >= self.max_steps

        if game_over or timed_out:
            if blue_left == red_left:
                self.winner = "draw"
            else:
                self.winner = TEAM_NAMES[BLUE] if blue_left > red_left else TEAM_NAMES[RED]
            cfg = self.rewards
            win, lose = cfg.win, cfg.lose
            if timed_out:
                win = cfg.win if cfg.win_on_numbers is None else cfg.win_on_numbers
                lose = cfg.lose if cfg.lose_on_numbers is None else cfg.lose_on_numbers

            def terminal(p):
                if self.winner == "draw":
                    return cfg.draw
                return win if self.winner == TEAM_NAMES[p.team] else lose

            for p in self.players:
                if p.agent_id in self.agents:
                    if p.alive or cfg.terminal_to_eliminated:
                        rewards[p.agent_id] += terminal(p)
                elif cfg.terminal_to_eliminated:
                    self.late_rewards[p.agent_id] = terminal(p)

        terminations = {a: game_over or a in eliminated for a in self.agents}
        truncations = {a: timed_out and a not in eliminated for a in self.agents}
        observations = {a: self._observe(a) for a in self.agents}
        infos = {a: {"action_mask": self.action_mask(a), "picked_up": a in picked_up} for a in self.agents}
        if self.winner is not None:
            for a in self.agents:
                infos[a]["winner"] = self.winner

        self.agents = [a for a in self.agents if not (terminations[a] or truncations[a])]
        return observations, rewards, terminations, truncations, infos

    def _eliminate(self, agent):
        p = self.players[agent]
        if p.alive:
            p.alive = False
            self.occupant[p.y, p.x] = EMPTY
