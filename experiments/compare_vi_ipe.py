import numpy as np
import matplotlib.pyplot as plt

from gridworld.env import GridWorld
from algorithms.value_iteration import value_iteration
from algorithms.policy_evaluation.iterative import iterative_policy_evaluation

def main():
    # Setup environment
    env = GridWorld()

    # Value Iteration
    V_vi, pi_vi, deltas_vi = value_iteration(env)

    # Iterative Policy Evaluation on the optimal policy
    V_ipe, deltas_ipe = iterative_policy_evaluation(env, pi_vi)

    # Difference in value functions
    diff = np.abs(V_vi - V_ipe)
    print(f"Max |V_vi - V_ipe| = {np.max(diff):.4f}")

    # Plot convergence
    plt.figure()
    plt.plot(deltas_vi, label='Value Iteration Δ')
    plt.plot(deltas_ipe, label='Iterative PE Δ')
    plt.xlabel("Iteration")
    plt.ylabel("Max Δ")
    plt.title("Convergence Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visual comparisons
    env.plot_grid(V=V_vi, policy=pi_vi, title="Value Iteration: Optimal Policy and Values")
    env.plot_grid(V=V_ipe, policy=pi_vi, title="Iterative PE: Same Policy")

if __name__ == "__main__":
    main()
