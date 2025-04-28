# algorithms/actor_critic.py
import numpy as np

def actor_critic(env, alpha_v=0.1, alpha_pi=0.1, gamma=0.9, episodes=5000, epsilon=0.1, max_steps=100):
    """
    Tabular Actor-Critic algorithm.

    Args:
        env: GridWorld
        alpha_v: learning rate for value
        alpha_pi: learning rate for policy
        gamma: discount factor
        episodes: number of episodes
        epsilon: exploration noise
        max_steps: max steps per episode

    Returns:
        V: value table
        policy: final policy
        deltas: list of TD errors per episode
        rewards: list of cumulative rewards
    """
    V = np.zeros(env.shape)
    preferences = np.zeros(env.shape + (env.nA,))
    deltas = []
    rewards = []

    def softmax(x):
        x = x - np.max(x)  # avoid overflow
        e_x = np.exp(x)
        return e_x / np.sum(e_x)

    for ep in range(episodes):
        s = env.start
        total_reward = 0
        max_delta = 0

        for _ in range(max_steps):
            if env.is_terminal(s):
                break

            probs = softmax(preferences[s])
            a = np.random.choice(env.nA, p=probs)

            ns, r = env.step(s, a)
            total_reward += r

            td_error = r + gamma * V[ns] - V[s]
            V[s] += alpha_v * td_error

            # Actor update (policy gradient)
            for b in range(env.nA):
                if b == a:
                    preferences[s + (b,)] += alpha_pi * td_error * (1 - probs[b])
                else:
                    preferences[s + (b,)] -= alpha_pi * td_error * probs[b]

            max_delta = max(max_delta, abs(td_error))
            s = ns

        deltas.append(max_delta)
        rewards.append(total_reward)

    # Derive greedy policy
    policy = np.argmax(preferences, axis=-1)
    return V, policy, deltas, rewards
