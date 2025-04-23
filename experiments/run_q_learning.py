# experiments/run_q_learning.py

import numpy as np
import matplotlib.pyplot as plt
from gridworld.env import GridWorld
from algorithms.q_learning import q_learning

def plot_learning_curves(deltas, rewards):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(deltas)
    axs[0].set_title("Q-learning: Max Δ per Episode (Convergence)")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Max TD Error")
    axs[0].grid()

    axs[1].plot(np.cumsum(rewards))
    axs[1].set_title("Q-learning: Cumulative Reward")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Cumulative Reward")
    axs[1].grid()

    plt.tight_layout()
    plt.show()


def main():
    env = GridWorld(
        shape=(4, 4),
        goal=(0, 1),
        fake_goal=(3, 0),
        start=(3, 3),
        bomb=None,
        stochastic_region={(0, 2), (0, 3), (1, 2), (1, 3)},
        slip_prob=0.1,
        gamma=0.9,
        noise=True
    )

    print("=== Initial Environment ===")
    env.plot_grid(title="Initial GridWorld")

    print("=== Q-learning Training ===")
    Q, policy, deltas, rewards = q_learning(env, alpha=0.1, epsilon=0.1, episodes=5000)

    V = np.max(Q, axis=-1)
    env.plot_grid(V=V, policy=policy, title="Q-learning: Optimal Policy and Values")

    plot_learning_curves(deltas, rewards)


if __name__ == "__main__":
    main()
