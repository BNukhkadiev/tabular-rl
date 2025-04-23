from gridworld.env import GridWorld
from algorithms.policy_iteration import greedy_policy_iteration, epsilon_soft_policy_iteration

# 1) Build the same 4×4 GridWorld
env = GridWorld()

# 2) Greedy Policy Iteration (infinite‐horizon, discounted)
V_greedy, pi_greedy = greedy_policy_iteration(env, theta=1e-4)
print("Greedy PI Value Function:\n", V_greedy.round(2))
print("Greedy PI Policy:\n", pi_greedy)

env.plot_grid(
    V=V_greedy,
    policy=pi_greedy,
    title="Greedy Policy Iteration\n(Value & Deterministic π)"
)

# 3) Epsilon‐Soft Policy Iteration
V_eps, pi_eps = epsilon_soft_policy_iteration(env, epsilon=0.1, theta=1e-4)
# Convert the stochastic ε‐soft policy into a point‐policy for plotting
pi_eps_det = pi_eps.argmax(axis=-1)

print("ε-soft PI Value Function:\n", V_eps.round(2))
print("ε-soft PI Deterministic Policy:\n", pi_eps_det)

env.plot_grid(
    V=V_eps,
    policy=pi_eps_det,
    title="ε-Soft Policy Iteration\n(Value & Greedy π from ε-soft)"
)
