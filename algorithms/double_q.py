# algorithms/double_q.py

import numpy as np

def double_q_learning(env, alpha=0.1, epsilon=0.1, episodes=5000):
    """
    Tabular Double Q-Learning.

    Args:
        env:      GridWorld instance
        alpha:    learning rate
        epsilon:  exploration rate for epsilon-greedy
        episodes: number of episodes

    Returns:
        Q1, Q2: 4D arrays of action-value estimates
        policy: 2D array (greedy wrt Q1+Q2)
    """
    Q1 = np.zeros(env.shape + (env.nA,))
    Q2 = np.zeros_like(Q1)
    for _ in range(episodes):
        s = env.start
        while not env.is_terminal(s):
            # ε-greedy on sum of Q1+Q2
            if np.random.rand() < epsilon:
                a = np.random.randint(env.nA)
            else:
                a = (Q1[s] + Q2[s]).argmax()

            ns, r = env.step(s, a)

            # Randomly pick which estimator to update
            if np.random.rand() < 0.5:
                best_next = Q1[ns].argmax()
                td_target = r + env.gamma * Q2[ns + (best_next,)]
                Q1[s + (a,)] += alpha * (td_target - Q1[s + (a,)])
            else:
                best_next = Q2[ns].argmax()
                td_target = r + env.gamma * Q1[ns + (best_next,)]
                Q2[s + (a,)] += alpha * (td_target - Q2[s + (a,)])

            s = ns

    Q = Q1 + Q2
    policy = Q.argmax(axis=-1)
    return Q1, Q2, policy
