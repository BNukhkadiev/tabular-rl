import numpy as np
import matplotlib.pyplot as plt
from gridworld.env import GridWorld
from algorithms.value_iteration import value_iteration
from algorithms.policy_evaluation.iterative import iterative_policy_evaluation
from algorithms.policy_evaluation.monte_carlo import mc_policy_evaluation
from algorithms.policy_evaluation.sample_based import sample_based_policy_evaluation
from algorithms.policy_evaluation.mc_q_evaluation import first_visit_mc_q_evaluation
from algorithms.policy_evaluation.finite_horizon import finite_horizon_policy_evaluation

def plot_convergence_curves(results):
    plt.figure(figsize=(12, 6))
    for label, deltas in results.items():
        plt.plot(deltas, label=label)
    plt.xlabel("Iterations / Episodes")
    plt.ylabel("Max Δ or Value Diff")
    plt.title("Convergence Comparison Across Evaluation Methods")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_convergence_experiment():
    env = GridWorld()
    policy = np.random.randint(env.nA, size=env.shape)

    results = {}

    print("Running Value Iteration...")
    V_vi, pi_vi, deltas_vi = value_iteration(env)
    results["Value Iteration"] = deltas_vi

    print("Running Iterative Policy Evaluation...")
    V_ipe, deltas_ipe = iterative_policy_evaluation(env, policy)
    results["Iterative Policy Eval"] = deltas_ipe

    print("Running Monte Carlo Policy Evaluation...")
    V_mc = mc_policy_evaluation(env, policy, episodes=5000)
    deltas_mc = [np.max(np.abs(V_mc - V_vi))] * 100  # Constant for comparison
    results["Monte Carlo Eval"] = deltas_mc

    print("Running Sample-based TD(0) Evaluation...")
    V_td = sample_based_policy_evaluation(env, policy, episodes=5000)
    deltas_td = [np.max(np.abs(V_td - V_vi))] * 100
    results["Sample-based TD(0)"] = deltas_td

    print("Running First-Visit Monte Carlo Q Evaluation...")
    Q_mc = first_visit_mc_q_evaluation(env, policy, episodes=5000)
    V_qmc = Q_mc.max(axis=-1)
    deltas_qmc = [np.max(np.abs(V_qmc - V_vi))] * 100
    results["First-Visit MC-Q"] = deltas_qmc

    print("Running Finite-Horizon Policy Evaluation...")
    V_fh, _ = finite_horizon_policy_evaluation(env, policy, T=15)
    deltas_fh = [np.max(np.abs(V_fh - V_vi))] * 100
    results["Finite-Horizon Eval"] = deltas_fh

    plot_convergence_curves(results)

run_convergence_experiment()
