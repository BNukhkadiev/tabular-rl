import numpy as np
def value_iteration(env, theta=1e-4):
    V = np.zeros(env.shape)
    policy = np.zeros(env.shape, dtype=int)
    deltas = []
    c = 0
    while True:
        delta = 0
        c += 1
        for s in env.all_states():
            if env.is_terminal(s):
                V[s] = env.rewards[s]  # ← Add this line
                continue

            v = V[s]
            q = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, ns in env.get_next_states(s, a):
                    if isinstance(ns, tuple) and ns[0] == 'noop':
                        reward = -1.0  # or something worse than the step cost
                        q[a] += prob * (reward + env.gamma * V[ns[1]])
                    else:
                        reward = env.rewards[ns]
                        q[a] += prob * (reward + env.gamma * V[ns])


            V[s] = q.max()
            # Break ties randomly to avoid always picking ↑
            policy[s] = np.random.choice(np.flatnonzero(q == q.max()))
            delta = max(delta, abs(v - V[s]))
        
        
        deltas.append(delta)
        if delta < theta:
            break
    print(f"Took {c} iterations")
    print("Final Value Function:")
    print(V.round(2))

    print("\nOptimal Policy (0=↑, 1=↓, 2=←, 3=→):")
    print(policy)

    print("Reward Matrix:")
    print(env.rewards)


    return V, policy, deltas
