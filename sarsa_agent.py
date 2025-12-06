# sarsa_agent.py
"""
Simple SARSA agent (on-policy) using Q-table.

- Q(s,a) updated with:
    Q <- Q + alpha * (r + gamma * Q(s',a') - Q)

- epsilon-greedy policy with linear decay
- methods: select_action(state_idx), update(...)
- save/load Q-table
"""

import numpy as np
from typing import Tuple


class SARSAgent:
    def __init__(
            self,
            state_count: int,
            action_count: int,
            alpha: float = 0.1,
            gamma: float = 0.99,
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.05,
            epsilon_decay_steps: int = 5000,
    ):
        self.state_count = state_count
        self.action_count = action_count
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = max(1, epsilon_decay_steps)
        self.total_steps = 0

        self.q = np.zeros((state_count, action_count), dtype=np.float32)

    def epsilon(self) -> float:
        frac = min(self.total_steps / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state_idx: int) -> int:
        if np.random.rand() < self.epsilon():
            return np.random.randint(self.action_count)
        return int(np.argmax(self.q[state_idx]))

    def best_action(self, state_idx: int) -> int:
        return int(np.argmax(self.q[state_idx]))

    def update(self, s_idx: int, a: int, r: float, ns_idx: int, na: int, done: bool):
        self.total_steps += 1
        current = self.q[s_idx, a]
        next_q = 0.0 if done else self.q[ns_idx, na]
        target = r + self.gamma * next_q
        self.q[s_idx, a] += self.alpha * (target - current)

    def save(self, path: str):
        np.save(path, self.q)

    def load(self, path: str):
        self.q = np.load(path)
