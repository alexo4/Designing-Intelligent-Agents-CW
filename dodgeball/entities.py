"""Game objects and constants used by the dodgeball environment."""

from dataclasses import dataclass

BLUE, RED = 0, 1
TEAM_NAMES = ("blue", "red")


@dataclass
class Player:
    agent_id: int
    team: int       # BLUE or RED
    x: int          # column, 0 .. grid_size - 1
    y: int          # row in world coordinates, 0 .. 2 * grid_size - 1 (blue's half is the top)
    alive: bool = True
