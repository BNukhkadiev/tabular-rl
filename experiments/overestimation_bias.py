import numpy as np
import matplotlib.pyplot as plt

def overestimation_env_step(action):
    """Returns next state and reward from state S given action."""
    if action == 0:  # Safe action
        return "T", 10
    elif action == 1:  # Risky action
        return "R", np.random.choice([20, -20])  # 50/50

def q_learning_overestimation(alpha=0.1, episodes=300):
    Q = np.zeros(2)  # 2 actions
    q_vals = []

    for _ in range(episodes):
        a = np.random.choice([0, 1])
        _, r = overestimation_env_step(a)
        Q[a] += alpha * (r - Q[a])
        q_vals.append(Q.copy())

    return np.array(q_vals)


def double_q_learning_overestimation(alpha=0.1, episodes=300):
    QA = np.zeros(2)
    QB = np.zeros(2)
    dq_vals = []

    for _ in range(episodes):
        a = np.random.choice([0, 1])
        _, r = overestimation_env_step(a)
        if np.random.rand() < 0.5:
            a_star = np.argmax(QA)
            td_target = r + 0  # terminal
            QA[a] += alpha * (td_target - QA[a])
        else:
            b_star = np.argmax(QB)
            td_target = r + 0
            QB[a] += alpha * (td_target - QB[a])
        dq_vals.append((QA + QB) / 2)

    return np.array(dq_vals)


def plot_results(q_vals, dq_vals):
    plt.figure(figsize=(10, 5))
    plt.plot(q_vals[:, 0], label="Q-learning: Q[a0] (safe)", color='blue')
    plt.plot(q_vals[:, 1], label="Q-learning: Q[a1] (risky)", color='red')
    plt.plot(dq_vals[:, 0], '--', label="Double Q: Q[a0] (safe)", color='blue')
    plt.plot(dq_vals[:, 1], '--', label="Double Q: Q[a1] (risky)", color='red')
    plt.title("Overestimation Bias in Q-learning vs Double Q-learning")
    plt.xlabel("Episode")
    plt.ylabel("Q-value Estimate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    q_vals = q_learning_overestimation()
    dq_vals = double_q_learning_overestimation()
    plot_results(q_vals, dq_vals)


if __name__ == "__main__":
    main()
