import numpy as np
import matplotlib.pyplot as plt
from gridworld.env import GridWorld
from algorithms.q_learning import q_learning

def run_robustness_experiment(episodes=2000, alpha=0.1, epsilon=0.1):
    # Environment A: no wind or slip
    env_a = GridWorld(wind=None, slip_prob=0.0)

    # Environment B: wind and slippery surface
    wind = {
        (2, 1): (0, 1),
        (3, 2): (0, -1),
        (1, 0): (1, 0)
    }
    env_b = GridWorld(wind=wind, slip_prob=0.2)

    # Train Q-learning on env A
    Q, policy, _, _ = q_learning(env_a, alpha=alpha, epsilon=epsilon, episodes=episodes)

    def evaluate(env, policy, episodes=100, max_steps=100):
        rewards = []
        for _ in range(episodes):
            s = env.start
            total_reward = 0
            for _ in range(max_steps):
                if env.is_terminal(s):
                    break
                a = policy[s]
                s, r = env.step(s, a)
                total_reward += r
            rewards.append(total_reward)
        return rewards

    rewards_a = evaluate(env_a, policy)
    rewards_b = evaluate(env_b, policy)

    return rewards_a, rewards_b

def plot_results(rewards_a, rewards_b):
    plt.figure(figsize=(8, 5))
    window = 10
    smoothed_a = np.convolve(rewards_a, np.ones(window)/window, mode='valid')
    smoothed_b = np.convolve(rewards_b, np.ones(window)/window, mode='valid')

    plt.plot(smoothed_a, label="Env A (No Wind)")
    plt.plot(smoothed_b, label="Env B (Wind + Slippery Floor)")
    plt.title("Robustness of Policy: Evaluation Across Environments")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Episode Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    rewards_a, rewards_b = run_robustness_experiment()
    plot_results(rewards_a, rewards_b)
