import numpy as np
import matplotlib.pyplot as plt
import time

from gridworld.env import GridWorld
from algorithms.value_iteration import value_iteration
from algorithms.policy_evaluation.iterative import iterative_policy_evaluation

# Add these if you place them in separate files
from algorithms.policy_evaluation.monte_carlo import mc_policy_evaluation
from algorithms.policy_evaluation.sample_based import sample_based_policy_evaluation


def main():
    env = GridWorld()

    print("=== Value Iteration ===")
    t0 = time.time()
    V_vi, pi_vi, deltas_vi = value_iteration(env)
    time_vi = time.time() - t0

    print("\n=== Iterative Policy Evaluation ===")
    t0 = time.time()
    V_ipe, deltas_ipe = iterative_policy_evaluation(env, pi_vi)
    time_ipe = time.time() - t0

    print("\n=== Monte Carlo Evaluation ===")
    t0 = time.time()
    V_mc = mc_policy_evaluation(env, pi_vi, episodes=10000)
    time_mc = time.time() - t0

    print("\n=== Sample-Based Evaluation ===")
    t0 = time.time()
    V_td = sample_based_policy_evaluation(env, pi_vi, episodes=10000)
    time_td = time.time() - t0

    # ======================
    # Comparison Metrics
    # ======================
    def max_diff(v1, v2):
        return np.max(np.abs(v1 - v2))

    print("\n=== Evaluation Error vs. V* ===")
    print(f"Iterative PE   : {max_diff(V_ipe, V_vi):.4f}")
    print(f"Monte Carlo PE : {max_diff(V_mc, V_vi):.4f}")
    print(f"Sample-Based PE: {max_diff(V_td, V_vi):.4f}")

    print("\n=== Execution Time (s) ===")
    print(f"Value Iteration: {time_vi:.4f}")
    print(f"Iterative PE   : {time_ipe:.4f}")
    print(f"Monte Carlo PE : {time_mc:.4f}")
    print(f"Sample-Based PE: {time_td:.4f}")

    # ======================
    # Visualization
    # ======================
    env.plot_grid(V=V_vi, policy=pi_vi, title="Value Iteration: V* and π*")
    env.plot_grid(V=V_ipe, policy=pi_vi, title="Iterative PE: V^π")
    env.plot_grid(V=V_mc, policy=pi_vi, title="Monte Carlo: V^π")
    env.plot_grid(V=V_td, policy=pi_vi, title="Sample-Based: V^π")

    # Error Bar Plot
    diffs = [
        max_diff(V_ipe, V_vi),
        max_diff(V_mc, V_vi),
        max_diff(V_td, V_vi),
    ]
    labels = ['Iterative PE', 'Monte Carlo', 'Sample-Based']
    plt.figure()
    plt.bar(labels, diffs)
    plt.title("Max Error vs. V*")
    plt.ylabel("||V_est - V*||∞")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
