import time
import numpy as np
import matplotlib.pyplot as plt

from gridworld.env import GridWorld
from algorithms.value_iteration import value_iteration
from algorithms.policy_evaluation.iterative import iterative_policy_evaluation
from algorithms.policy_evaluation.monte_carlo import mc_policy_evaluation
from algorithms.policy_evaluation.sample_based import sample_based_policy_evaluation


def finite_horizon_evaluation(env, policy, H):
    """
    Finite‐horizon policy evaluation by H rounds of Bellman backups:
      V^0(s)=0;  V^t(s)=E[r + V^{t-1}(s')]
    Tracks the max‐difference Δ_t = max_s |V^t(s) − V^{t-1}(s)|.
    """
    V_prev = np.zeros(env.shape)
    deltas = []
    for t in range(1, H+1):
        V = np.zeros(env.shape)
        delta = 0.0
        for s in env.all_states():
            if env.is_terminal(s):
                V[s] = env.rewards[s]
            else:
                a = policy[s]
                v = 0.0
                for prob, ns in env.get_next_states(s, a):
                    v += prob * (env.rewards[ns] + V_prev[ns])
                V[s] = v
            delta = max(delta, abs(V[s] - V_prev[s]))
        deltas.append(delta)
        V_prev = V
    return deltas


def mc_convergence(env, policy, V_true, episodes=2000, max_steps=100):
    """First‐visit Monte Carlo; returns ||V_est - V_true||∞ per episode."""
    from collections import defaultdict
    returns_sum, returns_cnt = defaultdict(float), defaultdict(int)
    V = np.zeros(env.shape)
    errors = []

    for ep in range(episodes):
        # generate one episode
        episode, s = [], env.start
        for _ in range(max_steps):
            if env.is_terminal(s):
                break
            a = policy[s]
            ns, r = env.step(s, a)
            episode.append((s, r))
            s = ns

        G = 0
        visited = set()
        for (s_t, r_t1) in reversed(episode):
            G = env.gamma * G + r_t1
            if s_t not in visited:
                returns_sum[s_t] += G
                returns_cnt[s_t] += 1
                V[s_t] = returns_sum[s_t] / returns_cnt[s_t]
                visited.add(s_t)

        errors.append(np.max(np.abs(V - V_true)))
    return errors


def td0_convergence(env, policy, V_true, alpha=0.1, episodes=2000, max_steps=100):
    """TD(0) sample‐based; returns ||V_est - V_true||∞ per episode."""
    V = np.zeros(env.shape)
    errors = []

    for ep in range(episodes):
        s = env.start
        for _ in range(max_steps):
            if env.is_terminal(s):
                break
            a = policy[s]
            ns, r = env.step(s, a)
            V[s] += alpha * (r + env.gamma * V[ns] - V[s])
            s = ns
        errors.append(np.max(np.abs(V - V_true)))
    return errors


def main():
    # -- 1) Build the environment (discounted MDP) --
    env = GridWorld(
        shape=(4, 4),
        goal=(0, 1),
        fake_goal=(3, 0),
        start=(3, 3),
        stochastic_region={(0,2),(0,3),(1,2),(1,3)},
        slip_prob=0.1, gamma=0.9, noise=False
    )

    # -- 2) Get the optimal policy & value via Value Iteration --
    V_vi, pi_vi, deltas_vi = value_iteration(env, theta=1e-4)
    print(f"Value Iteration converged in {len(deltas_vi)} sweeps")

    # -- 3) Discounted Iterative Policy Eval (IPE) --
    V_ipe, deltas_ipe = iterative_policy_evaluation(env, pi_vi, theta=1e-4)
    print(f"IPE (discounted) converged in {len(deltas_ipe)} sweeps")

    # -- 4) Finite‐Horizon IPE (horizon H) --
    H = 20
    deltas_fh = finite_horizon_evaluation(env, pi_vi, H)
    print(f"Finite‐Horizon IPE (H={H}) ran {len(deltas_fh)} stages")

    # -- 5) Monte Carlo (MC) convergence error --
    errors_mc = mc_convergence(env, pi_vi, V_vi, episodes=2000)

    # -- 6) TD(0) convergence error --
    errors_td = td0_convergence(env, pi_vi, V_vi, alpha=0.1, episodes=2000)

    # -- 7) Plot DP convergence (finite vs discounted) --
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(deltas_fh)+1), deltas_fh,    label=f"Finite‐Horizon Δ (H={H})")
    plt.plot(range(len(deltas_vi)),      deltas_vi,    label="Value Iteration Δ")
    plt.plot(range(len(deltas_ipe)),     deltas_ipe,   label="IPE (γ=0.9) Δ")
    plt.yscale('log')
    plt.xlabel("Iteration / Stage")
    plt.ylabel("Max Δ")
    plt.title("Convergence of DP Methods")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -- 8) Plot sample‐based convergence (error vs episodes) --
    plt.figure(figsize=(8, 5))
    plt.plot(errors_mc, label="Monte Carlo Error")
    plt.plot(errors_td, label="TD(0) Error")
    plt.yscale('log')
    plt.xlabel("Episode")
    plt.ylabel("||V_est − V*||∞")
    plt.title("Convergence of Sample‐Based Methods")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -- 9) Summary of “how fast” --
    print(f"\nDP:  FI‐Horizon needed {len(deltas_fh)} stages, IPE needed {len(deltas_ipe)} sweeps, VI needed {len(deltas_vi)} sweeps")
    threshold = 0.1
    mc_eps = next((i for i,e in enumerate(errors_mc,1) if e < threshold), "≥2000")
    td_eps = next((i for i,e in enumerate(errors_td,1) if e < threshold), "≥2000")
    print(f"MC needed ~{mc_eps} episodes to drop below {threshold}")
    print(f"TD  needed ~{td_eps} episodes to drop below {threshold}\n")


if __name__ == "__main__":
    main()
