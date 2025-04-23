# algorithms/policy_evaluation/iterative.py
import numpy as np

def iterative_policy_evaluation(env, policy, theta=1e-4):
    V = np.zeros(env.shape)
    deltas = []
    delta = 2 * theta

    while delta > theta:
        delta = 0
        for s in env.all_states():
            if env.is_terminal(s):
                continue
            v = V[s]
            a = policy[s]
            new_v = 0
            for prob, ns in env.get_next_states(s, a):
                r = env.rewards[ns]
                if ns == s and not env.is_terminal(s):
                    r -= 0.5  # extra penalty for staying in place
                new_v += prob * (r + env.gamma * V[ns])
            V[s] = new_v
            delta = max(delta, abs(v - new_v))
        deltas.append(delta)

    return V, deltas