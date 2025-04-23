# scripts/run_value_iteration.py
import numpy as np
from gridworld.env import GridWorld
from algorithms.value_iteration import value_iteration
import matplotlib.pyplot as plt


def main():
    # Create the GridWorld environment
    env = GridWorld(shape=(8, 1), goal=(7, 0), start=(0, 0), fake_goal=None)

    print("=== Initial GridWorld ===")
    env.plot_grid(title="Initial GridWorld")

    print("\n=== Running Value Iteration ===")
    V, policy, deltas = value_iteration(env)

    env.plot_grid(V=V, policy=policy, title="Value Iteration: Optimal Policy and Values")

    # Plot convergence of value iteration
    plt.plot(deltas)
    plt.title("Value Iteration Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Max Δ")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
