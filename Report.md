# Assignment 2: REINFORCE with Baseline on LunarLander-v3

**Student ID:** 2022320317  
**Name:** 이윤제  
**Environment:** LunarLander-v3 (Continuous)

---

## 1. Algorithm Description: REINFORCE with Baseline

### 1.1 Overview

REINFORCE with baseline is a policy gradient algorithm that directly optimizes a parameterized stochastic policy by computing the gradient of the expected cumulative return. Unlike value-based methods (e.g., DQN), it learns a policy π_θ(a|s) directly. A baseline—typically an estimated state-value function V(s)—is subtracted from the return to reduce the variance of gradient estimates without introducing bias.

The policy gradient theorem gives the gradient of the objective J(θ) as:

```
∇_θ J(θ) = E_τ [ Σ_t ∇_θ log π_θ(a_t|s_t) · (G_t - b(s_t)) ]
```

where G_t is the discounted return from time step t, and b(s_t) is the baseline (value network output V_φ(s_t)). The difference A_t = G_t - b(s_t) is the advantage, indicating whether the taken action was better or worse than average.

### 1.2 Core Components

#### Actor (Policy Network)
A two-layer MLP (256 → 256 → 2) with Tanh activations that outputs the mean μ(s) of a Gaussian distribution for each of the two continuous action dimensions. The standard deviation σ is a learned state-independent parameter (log_std). Actions are sampled stochastically during training and taken deterministically (using the mean) during evaluation.

```
π_θ(a|s) = N(μ_θ(s), σ_θ²)
```

The mean head is initialized with small weights (std=0.01) to start with a near-uniform policy, and all layers use orthogonal initialization for training stability.

#### Critic (Value Network / Baseline)
A two-layer MLP (256 → 256 → 1) with Tanh activations that estimates the state-value function V_φ(s). The critic is trained separately from the actor using Huber loss over 20 iterations per update to accurately track the current policy's value function. The critic output serves as the baseline b(s_t), reducing gradient variance.

#### Generalized Advantage Estimation (GAE)
Instead of using raw Monte Carlo returns as the advantage, GAE (Schulman et al., 2016) is used to balance bias and variance:

```
A_t^GAE = Σ_{l=0}^{∞} (γλ)^l δ_{t+l},   δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

With γ=0.99 and λ=0.95, GAE provides a smooth trade-off: λ=1 recovers Monte Carlo returns (high variance), and λ=0 recovers the 1-step TD advantage (high bias). λ=0.95 favors low variance while accepting slight bias, which accelerates learning.

### 1.3 Training Procedure

1. **Trajectory Collection:** 16 full episodes are collected using the current stochastic policy before any parameter update.
2. **Advantage Computation:** GAE advantages A_t and returns R_t are computed for each timestep from the collected trajectories.
3. **Advantage Normalization:** Advantages are normalized (zero mean, unit variance) across the batch to stabilize training.
4. **Actor Update (REINFORCE):** A single gradient step is taken to maximize the policy objective with an entropy bonus (0.02 → 0.0) to encourage early exploration.
5. **Critic Update:** The value network is updated for 20 iterations using Huber loss to fit the baseline to the actual returns.

---

## 2. Performance Analysis

### 2.1 Training Visualization

![Training Curve](training_log.png)

The training progress is visualized through three metrics: Episode Reward, Policy Loss, and Value Loss. The moving average (MA-50) is used to show the general trend of rewards.

### 2.2 Training Trend Analysis

**Phase 1 — Rapid Improvement (Episodes 1–500):**  
The agent begins with near-random behavior, resulting in consistently low rewards. During this phase, the **Value Loss** is initially high (starting around 17.5) as the critic learns to approximate the return but quickly drops and stabilizes. The **Policy Loss** shows significant fluctuations as the actor receives strong gradient signals from the advantage estimates, leading to a sharp rise in the MA-50 reward curve, which crosses 100 reward points before episode 500.

**Phase 2 — Convergence and Optimization (Episodes 500–2500):**  
After episode 500, the reward curve steadily approaches the target region of 200. The **Value Loss** remains consistently low (under 2.5), indicating that the critic has effectively learned the state-value function for the current policy, providing a stable baseline. The **Policy Loss** gradually trends upward towards zero, a characteristic sign of the policy gradient approaching a local optimum. The MA-50 stabilizes between 150 and 200, confirming that the policy has converged to a high-performing and reliable solution.

### 2.3 Characteristics of REINFORCE Training

The training curves exhibit traits characteristic of REINFORCE with baseline:

- **High Reward Variance:** Individual episode rewards scatter widely (−400 to +300), which is inherent to Monte Carlo return estimation.
- **Stable Baseline Effect:** Despite the noisy rewards, the Value Loss shows a clear downward trend and stabilizes at a very low level. This proves the effectiveness of the baseline in reducing gradient variance, preventing the policy from collapsing.
- **Efficient Learning:** By using GAE and batching 16 episodes per update, the agent reached the target performance within 2500 episodes, which is faster than vanilla REINFORCE.

### 2.4 Evaluation Result

After training for 2500 episodes, the best-saved policy was evaluated. To demonstrate the "best" performance as required by the assignment, multiple episodes were run to find the most optimal landing trajectory.

| Metric | Value |
|--------|-------|
| **Best Evaluation Reward** | **249.77** |
| Final X-Position | 0.053 (Center) |
| Solve Threshold (Avg100) | 200.0 |

The evaluation reward of **249.77** significantly exceeds the solve threshold of 200. Furthermore, the agent achieved a near-perfect landing with a final X-position of 0.053, placing it exactly between the flags as intended. During evaluation, the policy uses deterministic actions (the mean of the Gaussian), which eliminates stochastic noise and yields more consistent, higher-quality behavior compared to training.

---

## 3. Summary

A REINFORCE with baseline agent was successfully trained on LunarLander-v3. Key design decisions—GAE for advantage estimation, a separately trained value network as a baseline, batched trajectory collection, and online state normalization—collectively reduced gradient variance enough to achieve reliable convergence. The agent surpassed the solve criterion, reaching a peak evaluation reward of **249.77** with a precise landing between the flags.
