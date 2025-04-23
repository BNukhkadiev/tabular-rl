import numpy as np
from collections import defaultdict

def mc_policy_evaluation(env, policy, episodes=5000, max_steps=100):
    from collections import defaultdict
    returns_sum = defaultdict(float)
    returns_cnt = defaultdict(int)
    V = np.zeros(env.shape)
    
    for _ in range(episodes):
        episode = []
        s = env.start
        for _ in range(max_steps):
            if env.is_terminal(s):
                break
            a = policy[s]
            ns, r = env.step(s, a)
            episode.append((s, r))
            s = ns
        
        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            s_t, r_t1 = episode[t]
            G = env.gamma * G + r_t1
            if s_t not in visited:
                returns_sum[s_t] += G
                returns_cnt[s_t] += 1
                V[s_t] = returns_sum[s_t] / returns_cnt[s_t]
                visited.add(s_t)

    return V
