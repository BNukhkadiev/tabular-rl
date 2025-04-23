import numpy as np

def double_q_learning(env, alpha=0.1, epsilon=0.1, episodes=5000, max_steps=100):
    Q_A = np.zeros(env.shape + (env.nA,))
    Q_B = np.zeros_like(Q_A)
    rewards = []
    deltas = []

    for _ in range(episodes):
        s = env.start
        total_reward = 0
        max_delta = 0

        for _ in range(max_steps):
            if env.is_terminal(s):
                break

            # ε-greedy action selection from combined Q
            if np.random.rand() < epsilon:
                a = np.random.randint(env.nA)
            else:
                q_sum = Q_A[s] + Q_B[s]
                a = np.random.choice(np.flatnonzero(q_sum == q_sum.max()))

            ns, r = env.step(s, a)
            total_reward += r

            if np.random.rand() < 0.5:
                # Update Q_A
                a_star = np.argmax(Q_A[ns])
                td_target = r + env.gamma * Q_B[ns + (a_star,)]
                td_error = td_target - Q_A[s + (a,)]
                Q_A[s + (a,)] += alpha * td_error
            else:
                # Update Q_B
                b_star = np.argmax(Q_B[ns])
                td_target = r + env.gamma * Q_A[ns + (b_star,)]
                td_error = td_target - Q_B[s + (a,)]
                Q_B[s + (a,)] += alpha * td_error

            max_delta = max(max_delta, abs(td_error))
            s = ns

        rewards.append(total_reward)
        deltas.append(max_delta)

    Q_sum = Q_A + Q_B
    policy = np.argmax(Q_sum, axis=-1)

    return Q_A, Q_B, policy, rewards, deltas
