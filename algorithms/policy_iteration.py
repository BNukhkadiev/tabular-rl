# algorithms/policy_iteration.py

import numpy as np

def greedy_policy_iteration(env, theta=1e-4):
    """
    Standard (greedy) Policy Iteration:
    - Policy Evaluation (iterative) until convergence < theta
    - Policy Improvement (greedy)
    Repeat until policy is stable.
    Returns:
        V: value function
        pi: greedy policy (2D array of action indices)
    """
    # Initialize a random policy (e.g., always action 0)
    pi = np.zeros(env.shape, dtype=int)
    V = np.zeros(env.shape)

    is_policy_stable = False
    while not is_policy_stable:
        # Policy Evaluation
        while True:
            delta = 0
            for s in env.all_states():
                if env.is_terminal(s):
                    V[s] = env.rewards[s]
                    continue
                v_old = V[s]
                a = pi[s]
                v_new = 0.0
                for prob, ns in env.get_next_states(s, a):
                    r = env.rewards[ns]
                    v_new += prob * (r + env.gamma * V[ns])
                V[s] = v_new
                delta = max(delta, abs(v_old - v_new))
            if delta < theta:
                break

        # Policy Improvement
        is_policy_stable = True
        for s in env.all_states():
            if env.is_terminal(s):
                continue
            old_action = pi[s]
            # compute action-values for all actions
            q = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, ns in env.get_next_states(s, a):
                    r = env.rewards[ns]
                    q[a] += prob * (r + env.gamma * V[ns])
            # greedy improvement
            pi[s] = np.argmax(q)
            if old_action != pi[s]:
                is_policy_stable = False

    return V, pi


def epsilon_soft_policy_iteration(env, epsilon=0.1, theta=1e-4):
    """
    Epsilon-soft Policy Iteration:
    - Policy Evaluation under current stochastic policy until convergence < theta
    - Policy Improvement to an epsilon-soft policy
    Repeat until policy probabilities converge.
    Returns:
        V: value function
        pi: stochastic policy (shape: env.shape + (env.nA,))
    """
    # Initialize uniform random epsilon-soft policy
    pi = np.ones(env.shape + (env.nA,)) / env.nA
    V = np.zeros(env.shape)

    while True:
        # Policy Evaluation for stochastic policy
        while True:
            delta = 0
            for s in env.all_states():
                if env.is_terminal(s):
                    V[s] = env.rewards[s]
                    continue
                v_old = V[s]
                v_new = 0.0
                for a in range(env.nA):
                    action_prob = pi[s + (a,)]
                    for prob, ns in env.get_next_states(s, a):
                        r = env.rewards[ns]
                        v_new += action_prob * prob * (r + env.gamma * V[ns])
                V[s] = v_new
                delta = max(delta, abs(v_old - v_new))
            if delta < theta:
                break

        # Policy Improvement to epsilon-soft
        policy_stable = True
        for s in env.all_states():
            if env.is_terminal(s):
                continue
            # compute greedy action
            q = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, ns in env.get_next_states(s, a):
                    r = env.rewards[ns]
                    q[a] += prob * (r + env.gamma * V[ns])
            best_a = np.argmax(q)
            # update to epsilon-soft distribution
            for a in range(env.nA):
                new_prob = epsilon / env.nA
                if a == best_a:
                    new_prob += 1.0 - epsilon
                if abs(pi[s + (a,)] - new_prob) > 1e-8:
                    policy_stable = False
                pi[s + (a,)] = new_prob

        if policy_stable:
            break

    return V, pi
