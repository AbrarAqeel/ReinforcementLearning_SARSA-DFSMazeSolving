# run.py
"""
Train a SARSA agent on a randomly generated DFS maze and produce:
- reward_curve.png
- success_rate.png
- q_table.npy
- training_progress.mp4
- maze_demo.mp4

All configuration values live at the top.
"""
from __future__ import annotations
import os
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from maze_env import MazeEnv
from sarsa_agent import SARSAgent
from video import VideoRecorder

# =========================
# CONFIGURATION (edit here)
# =========================

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Maze / env
MAZE_ROWS = 21  # must be odd
MAZE_COLS = 21  # must be odd
MAZE_SEED = 42  # change to re-generate a different maze
START = (1, 1)
GOAL = (MAZE_ROWS - 2, MAZE_COLS - 2)
MAX_STEPS_PER_EPISODE = 1000

# Rewards
STEP_REWARD = -1.0
GOAL_REWARD = 100.0
WALL_PENALTY = -5.0

# SARSA hyperparams
EPISODES = 1200
ALPHA = 0.1
GAMMA = 0.99

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 8000

# Video settings
VIDEO_DURATION_SEC = 60
VIDEO_FPS = 10
VIDEO_TOTAL_FRAMES = VIDEO_DURATION_SEC * VIDEO_FPS
TRAINING_DEMO_PATH = os.path.join(OUTPUT_DIR, "training_progress.mp4")
FINAL_DEMO_PATH = os.path.join(OUTPUT_DIR, "maze_demo.mp4")

# Training progress video settings
TRAIN_EPISODES_TO_SHOW = 30
FRAMES_PER_TRAIN_EP = 20

# Final demo settings
FINAL_EPISODES_IN_VIDEO = 4

PRINT_EVERY = 50


# =========================
# END CONFIG
# =========================


def plot_rewards(rewards: List[float], out_dir: str = OUTPUT_DIR):
    episodes = np.arange(1, len(rewards) + 1)
    rewards_np = np.array(rewards, dtype=float)

    plt.figure(figsize=(10, 4))
    plt.plot(episodes, rewards_np, label="Episode Reward")
    if len(rewards_np) >= 20:
        rolling = np.convolve(rewards_np, np.ones(20) / 20, mode="valid")
        plt.plot(np.arange(20, len(rewards_np) + 1), rolling, label="Rolling Mean (20)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("SARSA Maze Training Rewards")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(out_dir, "reward_curve.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[PLOT] Saved {out}")


def plot_success_rate(success_list: List[int], out_dir: str = OUTPUT_DIR):
    if len(success_list) >= 20:
        rate = np.convolve(success_list, np.ones(20) / 20, mode="valid")
    else:
        rate = np.array(success_list, dtype=float)

    plt.figure(figsize=(10, 4))
    plt.plot(rate, label="Success Rate (window=20)")
    plt.ylim(0, 1)
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.title("Success Rate (Reached Goal)")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(out_dir, "success_rate.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[PLOT] Saved {out}")


def train():
    env = MazeEnv(
        rows=MAZE_ROWS,
        cols=MAZE_COLS,
        start=START,
        goal=GOAL,
        seed=MAZE_SEED,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        wall_penalty=WALL_PENALTY,
        max_steps=MAX_STEPS_PER_EPISODE,
    )

    state_count = env.rows * env.cols
    action_count = env.action_space

    agent = SARSAgent(
        state_count=state_count,
        action_count=action_count,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon_start=EPS_START,
        epsilon_end=EPS_END,
        epsilon_decay_steps=EPS_DECAY_STEPS,
    )

    rewards: List[float] = []
    success_list: List[int] = []

    for ep in range(1, EPISODES + 1):
        state = env.reset(seed=None)
        s_idx = env.state_to_index(state)
        ep_reward = 0.0
        reached_goal = False

        # choose first action (epsilon-greedy)
        a = agent.select_action(s_idx)

        for step in range(MAX_STEPS_PER_EPISODE):
            ns, r, done, _ = env.step(a)
            ns_idx = env.state_to_index(ns)
            # select next action a' using policy (on-policy)
            na = agent.select_action(ns_idx)

            # update SARSA Q(s,a) with (s,a,r,s',a')
            agent.update(s_idx, a, r, ns_idx, na, done)

            s_idx = ns_idx
            a = na
            ep_reward += r

            if done:
                if env.agent_pos == env.goal:
                    reached_goal = True
                break

        rewards.append(ep_reward)
        success_list.append(1 if reached_goal else 0)

        if ep % PRINT_EVERY == 0 or ep == 1:
            avg = np.mean(rewards[-PRINT_EVERY:]) if len(rewards) >= PRINT_EVERY else np.mean(rewards)
            print(
                f"[TRAIN] Ep {ep}/{EPISODES} Reward={ep_reward:.1f} AvgLast={avg:.1f} Success%={np.mean(success_list) * 100:.2f}%")

    # Save Q-table
    qpath = os.path.join(OUTPUT_DIR, "sarsa_q_table.npy")
    agent.save(qpath)
    print(f"[INFO] Saved Q-table to {qpath}")

    # Plots
    plot_rewards(rewards)
    plot_success_rate(success_list)

    # Videos
    make_training_progress_video(env, agent)
    make_final_demo_video(env, agent)

    # metrics
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"episodes={EPISODES}\n")
        f.write(f"successes={int(np.sum(success_list))}\n")
        f.write(f"maze_seed={MAZE_SEED}\n")
    print(f"[INFO] Saved metrics to {metrics_path}")

    return agent, env, rewards


def make_training_progress_video(env: MazeEnv, agent: SARSAgent):
    rec = VideoRecorder(TRAINING_DEMO_PATH, fps=VIDEO_FPS)
    print(f"[VIDEO] Creating training-progress video ({TRAIN_EPISODES_TO_SHOW} episodes)...")

    for ep in range(1, TRAIN_EPISODES_TO_SHOW + 1):
        state = env.reset()
        done = False
        last_frame = env.render_frame()

        # during progress video, use agent.select_action (exploratory)
        for _ in range(FRAMES_PER_TRAIN_EP):
            if not done:
                s_idx = env.state_to_index(state)
                action = agent.select_action(s_idx)
                ns, r, done, _ = env.step(action)
                state = ns
                last_frame = env.render_frame()
            rec.write(last_frame)

    rec.close()
    print(f"[VIDEO] Training progress saved to {TRAINING_DEMO_PATH}")


def make_final_demo_video(env: MazeEnv, agent: SARSAgent):
    FRAMES_PER_EP = VIDEO_TOTAL_FRAMES // FINAL_EPISODES_IN_VIDEO
    rec = VideoRecorder(FINAL_DEMO_PATH, fps=VIDEO_FPS)
    print(f"[VIDEO] Creating final demo ({FINAL_EPISODES_IN_VIDEO} episodes)...")

    last_frame = env.render_frame()
    for ep in range(1, FINAL_EPISODES_IN_VIDEO + 1):
        state = env.reset()
        done = False
        for _ in range(FRAMES_PER_EP):
            if not done:
                s_idx = env.state_to_index(state)
                action = agent.best_action(s_idx)
                ns, r, done, _ = env.step(action)
                state = ns
                last_frame = env.render_frame()
            rec.write(last_frame)

    rec.close()
    print(f"[VIDEO] Final demo saved to {FINAL_DEMO_PATH}")


if __name__ == "__main__":
    train()
