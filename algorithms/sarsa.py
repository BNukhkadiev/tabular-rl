import numpy as np

def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(Q.shape[-1])
    return np.argmax(Q[state])

def sarsa(env, alpha=0.1, epsilon=0.1, gamma=None, episodes=5000, max_steps=100):
    gamma = gamma if gamma is not None else env.gamma
    Q = np.zeros(env.shape + (env.nA,))
    deltas = []
    rewards = []

    for _ in range(episodes):
        s = env.start
        a = epsilon_greedy(Q, s, epsilon)
        max_delta = 0
        total_reward = 0

        for _ in range(max_steps):
            ns, r = env.step(s, a)
            na = epsilon_greedy(Q, ns, epsilon)
            td_target = r + gamma * Q[ns + (na,)] if not env.is_terminal(ns) else r
            td_error = td_target - Q[s + (a,)]
            Q[s + (a,)] += alpha * td_error
            max_delta = max(max_delta, abs(td_error))
            total_reward += r

            if env.is_terminal(ns):
                break

            s, a = ns, na

        deltas.append(max_delta)
        rewards.append(total_reward)

    policy = Q.argmax(axis=-1)
    return Q, policy, deltas, rewards
