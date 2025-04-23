# algorithms/policy_evaluation/mc_q_evaluation.py

import numpy as np
from collections import defaultdict

def first_visit_mc_q_evaluation(env, policy, episodes=5000, gamma=0.99):
    """
    First-visit Monte Carlo policy evaluation of Q^π.

    Args:
        env:     GridWorld environment
        policy:  Deterministic 2D array of actions
        episodes: Number of episodes to sample
        gamma:   Discount factor

    Returns:
        Q: Estimated action-value function Q(s, a)
    """
    Q = np.zeros(env.shape + (env.nA,))
    M = np.ones(env.shape + (env.nA,))  # count matrix, initialized to 1
    for ep in range(episodes):
        T = np.random.geometric(p=1 - gamma)
        s = env.start
        trajectory = []

        # generate trajectory of length 2T
        for _ in range(2 * T):
            if env.is_terminal(s):
                break
            a = policy[s]
            ns, r = env.step(s, a)
            trajectory.append((s, a, r))
            s = ns

        seen = set()
        G = 0.0
        for t in reversed(range(len(trajectory))):
            s_t, a_t, r_t = trajectory[t]
            G = r_t + gamma * G
            if (s_t, a_t) not in seen:
                seen.add((s_t, a_t))
                m = M[s_t + (a_t,)]
                Q[s_t + (a_t,)] = (1 / m) * G + ((m - 1) / m) * Q[s_t + (a_t,)]
                M[s_t + (a_t,)] += 1

    return Q
