# algorithms/q_learning.py

import numpy as np

def q_learning(env,
               alpha=0.1,
               epsilon=0.1,
               episodes=5000,
               epsilon_schedule=None,
               max_steps=100):
    """
    Q-Learning with optional epsilon schedule.

    Args:
        env:              GridWorld
        alpha:            learning rate
        epsilon:          fixed epsilon if no schedule given
        episodes:         number of episodes
        epsilon_schedule: function ep -> epsilon_ep (overrides fixed epsilon)

    Returns:
        Q, policy, deltas, rewards
    """
    Q = np.zeros(env.shape + (env.nA,))
    deltas = []
    rewards = []

    for ep in range(episodes):
        s = env.start
        total_reward = 0
        max_delta = 0

        # pick epsilon for this episode
        eps = epsilon_schedule(ep) if epsilon_schedule is not None else epsilon

        for _ in range(max_steps):  # <-- bounded loop
            if env.is_terminal(s):
                break
            # ε-greedy
            if np.random.rand() < eps:
                a = np.random.randint(env.nA)
            else:
                a = np.argmax(Q[s])

            ns, r = env.step(s, a)
            total_reward += r

            td_target = r + env.gamma * np.max(Q[ns])
            td_error = td_target - Q[s + (a,)]
            Q[s + (a,)] += alpha * td_error
            max_delta = max(max_delta, abs(td_error))

            s = ns

        deltas.append(max_delta)
        rewards.append(total_reward)

    policy = np.argmax(Q, axis=-1)
    return Q, policy, deltas, rewards
