import numpy as np
import matplotlib.pyplot as plt
from gridworld.env import GridWorld
from algorithms.q_learning import q_learning

def evaluate_policy(env, policy, episodes=100, max_steps=100):
    rewards = []
    successes = 0
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
        if s == env.goal:
            successes += 1
    return np.mean(rewards), successes / episodes


def run_experiment():
    shape = (4, 4)
    goal = (3, 3)
    start = (0, 0)
    
    # Env A: no wind
    env_a = GridWorld(shape=shape, goal=goal, start=start, wind=None)

    # Env B: wind on left side pushes up
    wind = {(r, 0): (-1, 0) for r in range(1, 4)}  # vertical wind pushing up in first column
    wind = None 
    
    env_b = GridWorld(shape=shape, goal=goal, start=start, wind=wind)

    # Train Q-learning on Env A
    Q, policy, _, _ = q_learning(env_a, alpha=0.1, epsilon=0.1, episodes=5000)

    # Evaluate policy in both environments
    mean_reward_a, success_rate_a = evaluate_policy(env_a, policy)
    mean_reward_b, success_rate_b = evaluate_policy(env_b, policy)

    # Print results
    print("Trained on Env A (No Wind)")
    print(f"→ Avg Reward (Env A): {mean_reward_a:.2f}, Success Rate: {success_rate_a:.2%}")
    print(f"→ Avg Reward (Env B): {mean_reward_b:.2f}, Success Rate: {success_rate_b:.2%}")

    # Plot
    labels = ['Env A (No Wind)', 'Env B (Wind)']
    rewards = [mean_reward_a, mean_reward_b]
    success = [success_rate_a, success_rate_b]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
    ax[0].bar(labels, rewards, color=['skyblue', 'salmon'])
    ax[0].set_title('Average Reward per Episode')
    ax[0].set_ylabel('Reward')
    
    ax[1].bar(labels, success, color=['skyblue', 'salmon'])
    ax[1].set_title('Goal Reach Rate')
    ax[1].set_ylabel('Rate')
    
    plt.suptitle("Policy Robustness Under Environmental Shift (Wind)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment()
