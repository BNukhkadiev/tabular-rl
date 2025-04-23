# experiments/compare_policy_iteration.py

import time
import numpy as np
import matplotlib.pyplot as plt

from gridworld.env import GridWorld
from algorithms.policy_iteration import greedy_policy_iteration, epsilon_soft_policy_iteration

def main():
    # 1) Setup the environment
    env = GridWorld(
        shape=(4, 4),
        goal=(0, 1),
        fake_goal=(3, 0),
        start=(3, 3),
        stochastic_region={(0, 2), (0, 3), (1, 2), (1, 3)},
        slip_prob=0.1,
        gamma=0.9,
        noise=True
    )

    print("=== Initial GridWorld ===")
    env.plot_grid(title="Initial GridWorld")

    # 2) Run Greedy Policy Iteration
    print("\n--- Greedy Policy Iteration ---")
    t0 = time.time()
    V_greedy, pi_greedy = greedy_policy_iteration(env, theta=1e-4)
    t_greedy = time.time() - t0
    print(f"Greedy PI took {t_greedy:.4f}s")

    # 3) Run ε-soft Policy Iteration
    print("\n--- Epsilon-soft Policy Iteration ---")
    t0 = time.time()
    V_eps, pi_eps = epsilon_soft_policy_iteration(env, epsilon=0.1, theta=1e-4)
    t_eps = time.time() - t0
    print(f"ε-soft PI took {t_eps:.4f}s")

    # Derive a deterministic “best‐action” policy from the ε-soft distribution
    pi_eps_det = np.argmax(pi_eps, axis=-1)

    # 4) Compare value functions
    max_diff = np.max(np.abs(V_greedy - V_eps))
    print(f"\nMax |V_greedy − V_ε| = {max_diff:.4f}")

    # 5) Visualize results
    env.plot_grid(V=V_greedy, policy=pi_greedy, title="Greedy Policy Iteration\nV & π")
    env.plot_grid(V=V_eps,    policy=pi_eps_det, title="ε-soft Policy Iteration\nV & π")

    # 6) Bar chart of runtimes
    plt.figure(figsize=(6,4))
    labels = ["Greedy PI", "ε-soft PI"]
    times  = [t_greedy,   t_eps]
    plt.bar(labels, times, color=['C0','C1'])
    plt.ylabel("Time (s)")
    plt.title("Runtime Comparison")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # 7) (Optional) Heatmap of |V_greedy − V_ε|
    plt.figure(figsize=(4,4))
    diff_mat = np.abs(V_greedy - V_eps)
    im = plt.imshow(diff_mat, cmap='Reds', origin='upper')
    for (i,j),val in np.ndenumerate(diff_mat):
        plt.text(j, i, f"{val:.2f}", ha='center', va='center', color='black')
    plt.title("|V_greedy − V_ε| Heatmap")
    plt.colorbar(im)
    plt.xticks(range(env.shape[1])); plt.yticks(range(env.shape[0]))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
