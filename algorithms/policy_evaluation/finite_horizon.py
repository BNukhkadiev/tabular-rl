# algorithms/policy_evaluation/finite_horizon.py

import numpy as np

def finite_horizon_policy_evaluation(env, policy, T):
    """
    Policy evaluation for finite-horizon MDPs using backward induction.

    Args:
        env:    GridWorld environment (must have deterministic transitions)
        policy: 2D array of shape (rows, cols) with action indices
        T:      Time horizon

    Returns:
        V: (T+1, rows, cols) Value function at each timestep
        Q: (T, rows, cols, nA) Q-values at each timestep
    """
    V = np.zeros((T + 1,) + env.shape)
    Q = np.zeros((T,) + env.shape + (env.nA,))

    for t in reversed(range(T)):
        for s in env.all_states():
            if env.is_terminal(s):  # terminal states are 0 forever
                continue
            for a in range(env.nA):
                expected = 0
                for prob, ns in env.get_next_states(s, a):
                    r = env.rewards[ns]
                    expected += prob * (r + V[t + 1][ns])
                Q[t][s + (a,)] = expected
            # deterministic policy
            a = policy[s]
            V[t][s] = Q[t][s + (a,)]

    return V, Q
