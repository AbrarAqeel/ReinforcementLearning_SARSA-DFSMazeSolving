# maze_env.py
"""
Maze environment using DFS (recursive backtracking) generated mazes.

- Maze grid uses odd dimensions (rows, cols) where walls are on even indices,
  corridors on odd indices. (e.g., 21x21)
- Start and goal are cells in corridor coordinates (odd indices).
- Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
- Methods:
    - reset(seed=None) -> state (r,c)
    - step(action) -> next_state, reward, done, info
    - state_to_index / index_to_state
    - render_frame(cell_size=10) -> RGB numpy array
"""

from typing import List, Tuple, Optional
import numpy as np

Cell = Tuple[int, int]


class MazeEnv:
    def __init__(
            self,
            rows: int = 21,
            cols: int = 21,
            start: Optional[Cell] = None,
            goal: Optional[Cell] = None,
            seed: int = 42,
            step_reward: float = -1.0,
            goal_reward: float = 50.0,
            wall_penalty: float = -5.0,
            max_steps: int = 1000,
    ):
        assert rows % 2 == 1 and cols % 2 == 1, "Rows and cols should be odd numbers."
        self.rows = rows
        self.cols = cols
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.wall_penalty = wall_penalty
        self.max_steps = max_steps

        # grid: 0=free, 1=wall
        self.grid = np.ones((rows, cols), dtype=np.uint8)
        self._generate_maze()

        # default start/goal: top-left corridor cell and bottom-right corridor cell
        if start is None:
            self.start = (1, 1)
        else:
            self.start = start
        if goal is None:
            self.goal = (rows - 2, cols - 2)
        else:
            self.goal = goal

        # make sure start/goal are free
        self.grid[self.start] = 0
        self.grid[self.goal] = 0

        # action space
        self.action_space = 4
        self.reset()

    def _generate_maze(self):
        # initialize all walls
        self.grid[:, :] = 1
        # carve out cells at odd coordinates
        for r in range(1, self.rows, 2):
            for c in range(1, self.cols, 2):
                self.grid[r, c] = 0

        # recursive backtracking (DFS) using stack
        start_cell = (1, 1)
        stack = [start_cell]
        visited = set([start_cell])

        while stack:
            cell = stack[-1]
            r, c = cell
            # possible neighbours two steps away
            nbrs = []
            for dr, dc in [(-2, 0), (0, 2), (2, 0), (0, -2)]:
                nr, nc = r + dr, c + dc
                if 1 <= nr < self.rows - 1 and 1 <= nc < self.cols - 1:
                    if (nr, nc) not in visited:
                        nbrs.append((nr, nc))
            if nbrs:
                # choose random neighbor
                nxt = tuple(self.rng.choice(nbrs))
                nr, nc = nxt
                # remove wall between cell and nxt
                wall_r, wall_c = (r + nr) // 2, (c + nc) // 2
                self.grid[wall_r, wall_c] = 0
                visited.add(nxt)
                stack.append(nxt)
            else:
                stack.pop()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._generate_maze()
            self.grid[self.start] = 0
            self.grid[self.goal] = 0

        self.agent_pos = tuple(self.start)
        self.steps = 0
        return self.agent_pos

    def in_bounds(self, pos: Cell) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_wall(self, pos: Cell) -> bool:
        r, c = pos
        return self.grid[r, c] == 1

    def step(self, action: int):
        """
        action: 0=UP,1=RIGHT,2=DOWN,3=LEFT
        returns: next_state, reward, done, info
        """
        r, c = self.agent_pos
        if action == 0:
            nr, nc = r - 1, c
        elif action == 1:
            nr, nc = r, c + 1
        elif action == 2:
            nr, nc = r + 1, c
        elif action == 3:
            nr, nc = r, c - 1
        else:
            nr, nc = r, c

        if not self.in_bounds((nr, nc)):
            nr, nc = r, c  # stay
        reward = self.step_reward
        done = False

        if self.is_wall((nr, nc)):
            # bump into wall -> stay in place, penalty
            nr, nc = r, c
            reward += self.wall_penalty
        else:
            # move to free cell
            self.agent_pos = (nr, nc)
            if self.agent_pos == self.goal:
                reward += self.goal_reward
                done = True

        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        return self.agent_pos, reward, done, {}

    def state_to_index(self, state: Cell) -> int:
        r, c = state
        return r * self.cols + c

    def index_to_state(self, idx: int) -> Cell:
        r = idx // self.cols
        c = idx % self.cols
        return (r, c)

    def render_frame(self, cell_size: int = 10) -> "np.ndarray":
        """
        Return an RGB frame as uint8 numpy array.
        Colors:
         - wall: black
         - free: white
         - agent: blue
         - goal: green
        """
        h = self.rows * cell_size
        w = self.cols * cell_size
        frame = np.ones((h, w, 3), dtype=np.uint8) * 255

        # draw walls
        wall_color = (0, 0, 0)
        free_color = (255, 255, 255)
        for r in range(self.rows):
            for c in range(self.cols):
                y0 = r * cell_size
                x0 = c * cell_size
                if self.grid[r, c] == 1:
                    frame[y0:y0 + cell_size, x0:x0 + cell_size] = wall_color
                else:
                    frame[y0:y0 + cell_size, x0:x0 + cell_size] = free_color

        # draw goal
        gr, gc = self.goal
        y0 = gr * cell_size
        x0 = gc * cell_size
        frame[y0:y0 + cell_size, x0:x0 + cell_size] = (0, 200, 0)

        # draw agent
        ar, ac = self.agent_pos
        y0 = ar * cell_size
        x0 = ac * cell_size
        frame[y0:y0 + cell_size, x0:x0 + cell_size] = (0, 0, 200)

        # thin grid lines for clarity
        for rr in range(1, self.rows):
            y = rr * cell_size
            frame[y:y + 1, :] = (220, 220, 220)
        for cc in range(1, self.cols):
            x = cc * cell_size
            frame[:, x:x + 1] = (220, 220, 220)

        return frame
