import numpy as np
import matplotlib.pyplot as plt
from gridworld.env import GridWorld
from algorithms.q_learning import q_learning

def sweep_learning_rate(env, eps_schedule, alphas, episodes=2000):
    results = {}
    for alpha in alphas:
        Q, pi, deltas, rewards = q_learning(
            env, alpha=alpha,
            epsilon_schedule=eps_schedule,
            episodes=episodes
        )
        results[alpha] = {
            'deltas': deltas,
            'cum_reward': np.cumsum(rewards)
        }
    return results

def sweep_epsilon(env, alpha, epsilons, episodes=2000):
    results = {}
    for eps0 in epsilons:
        def eps_schedule(t):
            # exponential decay to 0.01
            return max(0.01, eps0 * np.exp(-t / (episodes/5)))
        Q, pi, deltas, rewards = q_learning(
            env, alpha=alpha,
            epsilon_schedule=eps_schedule,
            episodes=episodes
        )
        results[eps0] = {
            'deltas': deltas,
            'cum_reward': np.cumsum(rewards)
        }
    return results

def plot_sweep(results, title, ylabel):
    plt.figure(figsize=(8,5))
    for param, data in results.items():
        plt.plot(data[ylabel], label=f"{param}")
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel.replace('_',' ').title())
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    env = GridWorld()
    episodes = 2000

    # 1) Sweep alpha with fixed ε‐schedule
    def eps_sched(t): 
        return max(0.01, 0.3 * np.exp(-t/(episodes/5)))
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5]
    alpha_res = sweep_learning_rate(env, eps_sched, alphas, episodes)
    plot_sweep(alpha_res, "Effect of Stepsize α", 'cum_reward')

    # 2) Sweep ε₀ with fixed α
    eps0s = [0.05, 0.1, 0.2, 0.5]
    epsilon_res = sweep_epsilon(env, alpha=0.1, epsilons=eps0s, episodes=episodes)
    plot_sweep(epsilon_res, "Effect of Initial ε", 'cum_reward')

    # 3) Convergence (δ) comparison for best hyperparams
    best_alpha = 0.1
    best_eps0  = 0.3
    eps_sched = lambda t: max(0.01, best_eps0*np.exp(-t/(episodes/5)))
    _, _, deltas_best, _ = q_learning(env, alpha=best_alpha,
                                      epsilon_schedule=eps_sched, episodes=episodes)
    plt.figure()
    plt.plot(deltas_best, label=f"α={best_alpha}, ε0={best_eps0}")
    plt.yscale('log'); plt.xlabel("Episode"); plt.ylabel("Max TD Error")
    plt.title("Convergence with Tuned Hyperparameters")
    plt.legend(); plt.grid(True); plt.show()

if __name__ == "__main__":
    main()
