# GridWorld RL Comparison
Author: Bagautdin Nukhkadiev

## Installation
```bash
pip install -r requirements.txt
```

Reproduce any experiment by running command like 
```bash
python -m experiments.<experiment name>
```
For example
```bash
python -m experiments.run_q_learning
```
## GridWorld
The manually implemented simple gridworld environment used in the experiments looks like this:

<img src="images/gridworld.png" alt="GridWorld" width="300"/>


## a) Convergence of Iterative vs Sample-Based Methods
Empirically verify the convergence rates of iterative methods that rely on known transition probabilities, comparing them to the convergence speed of sample-based algorithms.
We conducted an empirical comparison of convergence rates across different value evaluation methods:

<p float="left"> <img src="images/iterative_vs_sample_based.png" alt="Convergence Comparison" width="600"/> </p>
Key Observations:

Value Iteration and Iterative Policy Evaluation — methods that rely on known transition probabilities — converge very quickly.

Value Iteration reaches near-zero Bellman error within just a few iterations (~5–10).

Iterative Policy Evaluation is slower than Value Iteration but still converges steadily.

Sample-Based Methods such as:

Monte Carlo Evaluation

Sample-based TD(0) Evaluation

First-Visit Monte Carlo Q Evaluation

Finite-Horizon Evaluation

show significantly slower convergence.

Their errors plateau at higher values and do not decrease significantly even after many episodes.

Monte Carlo and TD(0) evaluation exhibit large variance, typical for sample-based estimators.

First-Visit MC and Finite-Horizon approaches similarly fail to match the precision of model-based approaches.


## b)
| Concept              | Finite-Time MDP                                       | Discounted MDP                                               |
|----------------------|--------------------------------------------------------|---------------------------------------------------------------|
| Time Horizon         | Fixed, e.g. T = 1000 steps                               | Infinite (potentially infinite steps into the future)         |
| Value Function       | V_t(s): Value depends on time t and state s           | V(s) = E[sum of gamma^t * r_t]: infinite discounted sum       |
| Discounting          | No discounting; horizon defines importance             | Uses gamma in [0, 1) to weigh future rewards less             |
| Optimal Policy       | Non-stationary (depends on remaining time)            | Stationary (same at every timestep for given state)           |
| Typical Use Case     | Games, deadline-based planning, short-term decisions  | Navigation, maintenance, inventory — long-term decisions      |

## c) 
Define and demonstrate the following effects by constructing ”extreme“ MDPs that maximize their visibility, supporting your explanations with appropriate graphs:
• Backpropagation
• Robust Reinforcement Learning
• Overestimation Bias


### Backpropagation
Here the extreme MDP would be a long hallway GridWorld (e.g. 1×8). This allows us to put the goal reward at the end (rightmost state) and observe how values slowly backpropagate from goal to start.

<img src="images/backprop.png" alt="Backprop" width="300"/>


### Robust Reinforcement Learning
The idea that the learned policy should still perform well even if environment dynamics change slightly. For that, we add some wind and slippery floor. 

The experiment setting for this would be to train and evaluate policies under:
- Environment A (no wind)
- Environment B (with wind)
- Plot average episode reward or goal reach rate.

### Overestimation bias
<img src="images/overestimation_bias.png" alt="Bias" width="600"/>


## d) Influence of Stepsize (α) and Exploration (ε) Parameters on Q-Learning
We conducted systematic experiments to understand how the choice of learning rate (α) and exploration rate (ε) affects the convergence and performance of Q-Learning.

### Effect of Learning Rate α
<p float="left"> <img src="images/alphas.png" alt="alphas" width="400"/> </p>
We tested different learning rates. Results show that:

Lower learning rates (e.g., α = 0.01) achieve better cumulative rewards over time.

Larger learning rates lead to more volatile learning and may prevent convergence because Q-updates overshoot the correct value estimates.

Thus, a smaller α promotes stable convergence by making gradual updates to Q-values.

### Effect of Initial Exploration Rate ε
<p float="left"> <img src="images/epsilon_effect.png" alt="epsilon effect" width="400"/> </p>
We also evaluated different initial exploration rates. Key observations:

Lower starting ε (e.g., ε₀ = 0.05) yields better final cumulative rewards.

High ε promotes early exploration but risks excessive randomness, slowing policy improvement.

A moderate initial exploration followed by gradual decay leads to better exploitation of learned policies.

### Convergence Behavior with Tuned Parameters
<p float="left"> <img src="images/tuned_params.png" alt="Convergence with Tuned Hyperparameters" width="400"/> </p>
With tuned hyperparameters (α = 0.01, ε₀ = 0.05), we observed excellent convergence:

TD error reduces smoothly and exponentially over episodes.

Occasional small spikes occur due to stochastic transitions but are quickly corrected.

The plot (log scale) shows convergence below 10⁻⁶, indicating very high precision.


## e) Comparison of Actor-Critic and Q-Learning
Compare the general actor-critic method to direct stochastic control algorithms that utilize
the Bellman Optimality Operator.


We empirically compared the performance of the general Actor-Critic method against Q-learning, a direct control algorithm utilizing the Bellman Optimality Operator.

<p float="left"> <img src="images/actor_critic_q_learning_rewards.png" alt="Actor-critic vs Q-learning Rewards" width="400"/> <img src="images/actor_critic_q_learning_errors.png" alt="Actor-critic vs Q-learning Errors" width="400"/> </p>
As shown in the cumulative reward plot (left), Q-learning accumulates higher rewards over training episodes compared to Actor-Critic. However, the TD-error convergence plot (right) highlights that the Actor-Critic method exhibits larger TD-errors during training, suggesting less stable value estimation. This is expected because Actor-Critic updates are inherently noisier due to stochastic policy gradients and continuous interaction.

Interestingly, despite achieving lower cumulative rewards, Actor-Critic produced a qualitatively better final policy. In our environment containing a fake goal, Q-learning learned a policy that often gets baited into the suboptimal (false) goal. In contrast, Actor-Critic correctly ignored the fake goal and constructed a policy that reliably leads to the true goal.

This behavior highlights a crucial difference: Q-learning overestimates action values, especially under stochastic rewards (overestimation bias), while Actor-Critic’s direct policy learning helps avoid such traps even at the cost of sample inefficiency.

Final learned policies are visualized below:

<p float="left"> <img src="images/final_policy_q_learning.png" alt="Final Policy Q-learning" width="400"/> <img src="images/final_policy_actor_critic.png" alt="Final Policy Actor-Critic" width="400"/> </p>
On the left, Q-learning often navigates toward the false goal; on the right, Actor-Critic steadily reaches the true goal.


# Appendix

## Value Iteration

<img src="images/value_iter_values.png" alt="Value Iteration values" width="300"/>

<img src="images/value_iter_convergence.png" alt="Value Iteration convergence" width="300"/>


## Policy Iteration
We have implemented greedy and epsilon-soft policy iteration algorithms. 



## Policy Evaluation Comparisons
We are comparing the

Monte Carlo evaluation takes very long time to compute compared to other methods. 


## Q-learning

<img src="images/q_learning_values.png" alt="Q-learning solving GridWorld" width="300"/>

<img src="images/q_learning_convergence.png" alt="Q-learning convergence" width="300"/>



## Q-Algos comparison

<img src="images/q_algos_comparison.png" alt="Q-algothims comparison" width="300"/>

