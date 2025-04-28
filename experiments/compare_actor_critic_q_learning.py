import numpy as np
import matplotlib.pyplot as plt

from gridworld.env import GridWorld
from algorithms.q_learning import q_learning
from algorithms.actor_critic import actor_critic

def run_experiment():
    env = GridWorld()

    episodes = 5000
    max_steps = 100
    alpha = 0.1
    gamma = 0.9

    print("Training Q-Learning...")
    Q_q, pi_q, deltas_q, rewards_q = q_learning(env, alpha=alpha, epsilon=0.1, episodes=episodes, max_steps=max_steps)

    print("Training Actor-Critic...")
    V_ac, pi_ac, deltas_ac, rewards_ac = actor_critic(env, alpha_v=alpha, alpha_pi=alpha, gamma=gamma, episodes=episodes, max_steps=max_steps)

    # Cumulative rewards
    cum_rewards_q = np.cumsum(rewards_q)
    cum_rewards_ac = np.cumsum(rewards_ac)

    # Plot cumulative rewards
    plt.figure(figsize=(10,6))
    plt.plot(cum_rewards_q, label="Q-Learning")
    plt.plot(cum_rewards_ac, label="Actor-Critic")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward: Actor-Critic vs Q-Learning")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Plot convergence (TD errors)
    plt.figure(figsize=(10,6))
    plt.plot(deltas_q, label="Q-Learning TD Error")
    plt.plot(deltas_ac, label="Actor-Critic TD Error")
    plt.xlabel("Episodes")
    plt.ylabel("Max TD Error")
    plt.title("TD Error Convergence")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Visualize final policies
    env.plot_grid(policy=pi_q, title="Final Policy: Q-Learning")
    env.plot_grid(policy=pi_ac, title="Final Policy: Actor-Critic")

    # Print final value estimates
    print("\nFinal Value Function (Actor-Critic):")
    print(V_ac.round(2))
    print("\nQ-values (Q-Learning, max over actions):")
    print(np.max(Q_q, axis=-1).round(2))

if __name__ == "__main__":
    run_experiment()
