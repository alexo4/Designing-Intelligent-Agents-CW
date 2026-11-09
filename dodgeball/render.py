"""Tkinter viewer for the dodgeball court.

Usage::

    renderer = CourtRenderer(env.grid_size)
    renderer.draw(env)   # call after every env.step()
"""

import tkinter as tk

from dodgeball.entities import TEAM_NAMES

MARGIN = 30     # pixels around the court
MIDLINE = 8     # height of the half-way line in pixels


class CourtRenderer:

    def __init__(self, grid_size, title="Dodgeball", cell=None):
        self.grid_size = grid_size
        self.root = tk.Tk()
        self.root.title(title)
        self.root.resizable(False, False)

        # Pixels per grid square: by default, make the court fill ~80% of the screen height.
        if cell is None:
            cell = (int(self.root.winfo_screenheight() * 0.8) - MIDLINE - 2 * MARGIN) // (2 * grid_size)
        self.cell = max(cell, 20)

        width = grid_size * self.cell + 2 * MARGIN
        height = 2 * grid_size * self.cell + MIDLINE + 2 * MARGIN
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="white")
        self.canvas.pack()
        self.root.geometry(f"{width}x{height}+50+30")
        self._draw_court()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.update()
        self.root.attributes("-topmost", False)

    def _cell_origin(self, x, y):
        """Top-left pixel of grid cell (x, y); the red half sits below the midline."""
        y_offset = MIDLINE if y >= self.grid_size else 0
        return MARGIN + x * self.cell, MARGIN + y_offset + y * self.cell

    def _draw_court(self):
        for y in range(self.grid_size * 2):
            for x in range(self.grid_size):
                x0, y0 = self._cell_origin(x, y)
                self.canvas.create_rectangle(x0, y0, x0 + self.cell, y0 + self.cell, outline="#cccccc")

        mid_y = MARGIN + self.grid_size * self.cell
        self.canvas.create_rectangle(MARGIN, mid_y, MARGIN + self.grid_size * self.cell, mid_y + MIDLINE,
                                     fill="purple", outline="purple")
        self.canvas.create_rectangle(MARGIN, MARGIN, MARGIN + self.grid_size * self.cell,
                                     MARGIN + 2 * self.grid_size * self.cell + MIDLINE,
                                     outline="purple", width=4)

    def draw(self, env):
        """Redraw all players and balls from the environment's current state."""
        self.canvas.delete("dynamic")

        for p in env.players:
            if p.alive:
                x0, y0 = self._cell_origin(p.x, p.y)
                self.canvas.create_oval(x0 + self.cell * 0.2, y0 + self.cell * 0.2, x0 + self.cell * 0.8, y0 + self.cell * 0.8,
                                        fill=TEAM_NAMES[p.team], tags="dynamic")
                self.canvas.create_text(x0 + self.cell / 2, y0 + self.cell / 2, text=str(p.agent_id),
                                        fill="white", font=("Helvetica", max(self.cell // 3, 10), "bold"), tags="dynamic")

        for y, x in zip(*env.balls.nonzero()):
            x0, y0 = self._cell_origin(x, y)
            r = self.cell * 0.12
            bx, by = x0 + self.cell * 0.18, y0 + self.cell * 0.18
            self.canvas.create_oval(bx - r, by - r, bx + r, by + r, fill="green", outline="darkgreen",
                                    tags="dynamic")
            if env.balls[y, x] > 1:
                self.canvas.create_text(bx + 2 * r, by, text=str(env.balls[y, x]),
                                        font=("Helvetica", max(self.cell // 4, 8), "bold"), tags="dynamic")

        status = f"step {env.steps}/{env.max_steps}"
        if env.winner is not None:
            status += f"   winner: {env.winner}"
        self.canvas.create_text(MARGIN, MARGIN // 2, anchor="w", text=status, font=("Helvetica", 14),
                                tags="dynamic")
        self.root.update()

    def wait_until_closed(self):
        """Keep the window open (showing the final state) until the user closes it."""
        self.root.mainloop()

    def close(self):
        self.root.destroy()
