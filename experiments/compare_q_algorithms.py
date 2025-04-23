import numpy as np
import matplotlib.pyplot as plt

from gridworld.env import GridWorld
from algorithms.q_learning import q_learning
from algorithms.sarsa import sarsa
from algorithms.double_q_learning import double_q_learning


def plot_convergence(deltas_q, deltas_s, deltas_dq):
    plt.figure(figsize=(10, 6))
    plt.plot(deltas_q, label="Q-learning Δ")
    plt.plot(deltas_s, label="SARSA Δ")
    plt.plot(deltas_dq, label="Double Q-learning Δ")
    plt.xlabel("Episode")
    plt.ylabel("Max TD Error")
    plt.title("Convergence: TD Error Per Episode")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_and_plot():
    # Setup environment
    env = GridWorld()

    episodes = 5000
    alpha = 0.1
    epsilon = 0.1

    print("Training Q-learning...")
    Q_q, pi_q, deltas_q, rewards_q = q_learning(env, alpha=alpha, epsilon=epsilon, episodes=episodes)

    print("Training SARSA...")
    Q_s, pi_s, deltas_s, rewards_s = sarsa(env, alpha=alpha, epsilon=epsilon, episodes=episodes)

    print("Training Double Q-learning...")
    Q1, Q2, pi_dq, rewards_dq, deltas_dq = double_q_learning(env, alpha=alpha, epsilon=epsilon, episodes=episodes)

    # Convert rewards to cumulative
    cum_q = np.cumsum(rewards_q)
    cum_s = np.cumsum(rewards_s)
    cum_dq = np.cumsum(rewards_dq)

    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(cum_q, label="Q-learning")
    plt.plot(cum_s, label="SARSA")
    plt.plot(cum_dq, label="Double Q-learning")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.title("Comparison of RL Algorithms on GridWorld")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plot_convergence(deltas_q, deltas_s, deltas_dq)


if __name__ == "__main__":
    run_and_plot()
