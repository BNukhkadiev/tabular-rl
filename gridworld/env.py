# gridworld/env.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class GridWorld:
    def __init__(self, shape=(4, 4), goal=(0, 1), fake_goal=(3, 0), start=(3, 3),
                 bomb=None,
                 stochastic_region={(0, 2), (0, 3), (1, 2), (1, 3)},
                 wind=None, slip_prob=0.0, gamma=0.9, noise=False):

        self.shape = shape
        self.goal = goal
        self.fake_goal = fake_goal
        self.start = start
        self.bomb = bomb
        self.stochastic_region = stochastic_region
        self.wind = wind or {}
        self.slip_prob = slip_prob
        self.gamma = gamma
        self.noise = noise

        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        self.action_symbols = ['↑', '↓', '←', '→']
        self.nA = len(self.actions)

        # Initialize rewards
        self.rewards = np.zeros(self.shape)
        for s in self.all_states():
            if s == self.goal:
                self.rewards[s] = 1.0
            elif s == self.fake_goal:
                self.rewards[s] = 0.2
            elif s == self.bomb:
                self.rewards[s] = -1.0
            elif s in self.stochastic_region:
                self.rewards[s] = (-2.1 + 2.0) / 2  # = -0.05
            else:
                self.rewards[s] = np.random.choice([-0.05, 0.05]) if noise else -0.05

    def in_bounds(self, state):
        r, c = state
        return 0 <= r < self.shape[0] and 0 <= c < self.shape[1]

    def is_terminal(self, state):
        return state == self.goal or state == self.bomb

    def all_states(self):
        for r in range(self.shape[0]):
            for c in range(self.shape[1]):
                yield (r, c)

    def get_next_states(self, state, action):
        if self.is_terminal(state):
            return [(1.0, state)]

        outcomes = []
        for i, a in enumerate(self.actions):
            prob = self.slip_prob / 3 if i != action else 1 - self.slip_prob
            next_state = tuple(np.add(state, a))
            if self.in_bounds(next_state):
                outcomes.append((prob, next_state))
            else:
                outcomes.append((prob, state))  # bounce back

        if state in self.wind:
            wind_effect = self.wind[state]
            new_outcomes = []
            if state in self.wind:
                wind_effect = self.wind[state]
                new_outcomes = []
                for prob, s in outcomes:
                    if np.random.rand() < 0.8:  # wind chance
                        winded = tuple(np.add(s, wind_effect))
                        if self.in_bounds(winded):
                            new_outcomes.append((prob, winded))
                        else:
                            new_outcomes.append((prob, s))
                    else:
                        new_outcomes.append((prob, s))  # chance: no wind
                return new_outcomes


        return outcomes

    def step(self, state, action):
        transitions = self.get_next_states(state, action)
        probs, next_states = zip(*transitions)
        idx = np.random.choice(len(probs), p=probs)
        next_state = next_states[idx]
        reward = self.rewards[next_state]
        if next_state == state and not self.is_terminal(state):
            reward -= 0.5  # penalize non-movement
        if next_state in self.stochastic_region:
            reward = np.random.choice([-2.1, 2.0])
        return next_state, reward

    def plot_grid(self, V=None, policy=None, title="GridWorld"):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.shape[1])
        ax.set_ylim(0, self.shape[0])
        for r in range(self.shape[0]):
            for c in range(self.shape[1]):
                s = (r, c)
                y = r

                if s == self.goal:
                    ax.add_patch(patches.Rectangle((c, y), 1, 1, color='pink'))
                    ax.text(c + 0.5, y + 0.5, 'G', ha='center', va='center', fontsize=14, weight='bold')
                elif s == self.fake_goal:
                    ax.add_patch(patches.Rectangle((c, y), 1, 1, color='lightgreen'))
                    ax.text(c + 0.5, y + 0.5, 'F', ha='center', va='center', fontsize=14, weight='bold')
                elif s == self.bomb:
                    ax.add_patch(patches.Rectangle((c, y), 1, 1, color='red'))
                    ax.text(c + 0.5, y + 0.5, 'B', ha='center', va='center', fontsize=14, weight='bold')
                elif s == self.start:
                    ax.add_patch(patches.Rectangle((c, y), 1, 1, color='lavender'))
                    ax.text(c + 0.5, y + 0.5, 'S', ha='center', va='center', fontsize=14, weight='bold')
                elif s in self.stochastic_region:
                    ax.add_patch(patches.Rectangle((c, y), 1, 1, color='lightgray'))
                    ax.text(c + 0.5, y + 0.8, 'SR', ha='center', va='center', fontsize=10, weight='bold')
                    if V is not None:
                        val = V[s]
                        ax.text(c + 0.5, y + 0.3, f'{val:+.2f}', ha='center', va='center', fontsize=9)
                    if policy is not None:
                        a = policy[s]
                        symbol = self.action_symbols[a]
                        ax.text(c + 0.5, y + 0.6, symbol, ha='center', va='center', fontsize=12)
                else:
                    if V is not None:
                        val = V[s]
                        ax.text(c + 0.5, y + 0.3, f'{val:+.2f}', ha='center', va='center')
                    if policy is not None:
                        a = policy[s]
                        symbol = self.action_symbols[a]
                        ax.text(c + 0.5, y + 0.7, symbol, ha='center', va='center', fontsize=14)

        # Start border overlay
        start_r, start_c = self.start
        ax.add_patch(patches.Rectangle((start_c, start_r), 1, 1,
                                       fill=False, edgecolor='blue', linewidth=2, linestyle='--'))

        ax.set_xticks(np.arange(self.shape[1] + 1))
        ax.set_yticks(np.arange(self.shape[0] + 1))
        ax.grid(True)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.invert_yaxis()  
        plt.show()