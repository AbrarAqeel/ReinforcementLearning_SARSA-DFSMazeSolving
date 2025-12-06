# DFS Maze Solving with SARSA

## Overview
This project implements a **randomly generated maze** using **Depth-First Search (DFS)**  
and trains a **SARSA (on-policy)** reinforcement learning agent to navigate from a start  
cell to a goal cell.

The maze layout is generated using a **fixed seed** so it is fully reproducible, and  
students can easily generate new mazes by changing the seed. All configuration values  
are located at the top of `run.py`.

The outputs include:
- Reward curve
- Success rate plot
- SARSA Q-table
- A training-progress video showing the agent learning
- A final demo video showing the trained agent navigating the maze


---

## Features
- **Random DFS-generated maze**
- Environment with:
  - Walls, free paths, start & goal
  - Step penalty
  - Wall penalty
  - Goal reward
- **SARSA agent** with:
  - Q-table
  - Epsilon-greedy exploration
  - Linear epsilon decay
- **Two videos generated automatically:**
  1. `training_progress.mp4` — early learning (exploratory)
  2. `maze_demo.mp4` — final trained performance (greedy)
- All configuration in one place (top of `run.py`)
- Easy to modify maze seed, rewards, learning parameters

---

## Directory Structure
```

RLProject	/
│── maze_env.py          # DFS maze generation + rendering
│── sarsa_agent.py       # SARSA agent (Q-table)
│── video.py             # Simple mp4 writer (OpenCV)
│── run.py               # Training, evaluation, videos, plots
│── requirements.txt
└── outputs/
├── reward_curve.png
├── success_rate.png
├── sarsa_q_table.npy
├── training_progress.mp4
└── maze_demo.mp4

```

---

## Installation
Install dependencies:

```

pip install -r requirements.txt

```

---

## How to Run
Train the SARSA agent and generate all outputs:

```

python run.py

```

This will:
- Generate a random DFS maze (size 21×21)
- Train the SARSA agent for 1200 episodes
- Save:
  - reward curve
  - success rate plot
  - SARSA Q-table
  - metrics file
  - training-progress video
  - final demo video

All files will be saved inside the `outputs/` directory.

---

## Maze Configuration
The maze is generated using DFS. The following parameters are in `run.py`:

```

MAZE_ROWS = 21      # must be odd
MAZE_COLS = 21      # must be odd
MAZE_SEED = 42      # change this for a new maze layout
START = (1, 1)
GOAL = (MAZE_ROWS - 2, MAZE_COLS - 2)

```

Changing `MAZE_SEED` instantly produces a new maze.

Maze cells:
- Walls = 1 (black)
- Free paths = 0 (white)

The DFS algorithm carves perfect mazes with one unique path between cells.

---

## Reward Function
Defined in `run.py`:

```

STEP_REWARD = -1.0
GOAL_REWARD = 100.0
WALL_PENALTY = -5.0

```

Students can modify these values to satisfy the project requirement of customizing the environment.

---

## Videos

### 1. training_progress.mp4
Shows **early learning behavior**:
- Agent exploring randomly (epsilon-greedy)
- Hitting walls
- Slowly learning paths

This illustrates *how the agent learns*, matching the assignment requirement.

### 2. maze_demo.mp4
Shows **final performance** over several episodes:
- Agent uses its learned SARSA policy (greedy)
- Maze navigation looks much more stable
- Demonstrates the trained model’s behavior

---

## Important Notes
- Maze generation bug fix:  
  DFS neighbors returned NumPy arrays, so they were converted to tuples  
  to avoid the “unhashable type: numpy.ndarray” error.
- Maze size must be **odd** for proper DFS corridor carving.
- Success rate is calculated based on reaching the goal, not reward.
- Videos run at 10 FPS and last exactly 60 seconds for consistency.

---

## License
For academic use in the Reinforcement Learning course.
```