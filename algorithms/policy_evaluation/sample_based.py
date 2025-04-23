import numpy as np

def sample_based_policy_evaluation(env, policy, alpha=0.1, episodes=5000, max_steps=100):
    """
    Sample-based (TD(0)) policy evaluation.

    Args:
        env:      GridWorld instance
        policy:   2D array of actions for each state
        alpha:    learning rate
        episodes: number of episodes to run
        max_steps: max steps per episode to prevent infinite loops

    Returns:
        V: 2D array of state-value estimates
    """
    V = np.zeros(env.shape)
    for _ in range(episodes):
        s = env.start
        for _ in range(max_steps):
            if env.is_terminal(s):
                break
            a = policy[s]
            ns, r = env.step(s, a)
            V[s] += alpha * (r + env.gamma * V[ns] - V[s])
            s = ns
    return V
