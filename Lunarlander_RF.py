import os

import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import RecordVideo
from torch.distributions import Normal


"""
TODO:
Train an agent that can reliably land between the flags and achieve a reward greater than 200.

Also, save:
- A video of the agent reliably landing between the flags
- Training logs
- A performance graph showing rewards exceeding 200

You may refer to the lecture notes if needed.
"""


class PolicyNetwork(nn.Module):
    """
    Policy network (Actor) for REINFORCE with baseline.
    Input : state (8-dim)
    Output: mean and log_std of Gaussian distribution over actions (2-dim)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        # log_std는 state에 무관한 독립 파라미터로 학습
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.net(state)
        mean = torch.tanh(self.mean_head(x))   # action 범위 [-1, 1]에 맞춤
        # log_std를 [-20, 2]로 클램핑 → std가 폭발하거나 0이 되는 것 방지
        std = self.log_std.clamp(-20, 2).exp().expand_as(mean)
        return mean, std

    def get_distribution(self, state):
        mean, std = self.forward(state)
        return Normal(mean, std)

    def sample_action(self, state):
        """학습 시: 분포에서 샘플링 (탐색)"""
        dist = self.get_distribution(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)   # 2-dim action → scalar
        return action.clamp(-1, 1), log_prob

    def deterministic_action(self, state):
        """평가 시: mean을 그대로 사용 (결정적)"""
        mean, _ = self.forward(state)
        return mean.clamp(-1, 1)

class ValueNetwork(nn.Module):
    """
    Value network (Critic / Baseline) for REINFORCE with baseline.
    Input : state (8-dim)
    Output: scalar V(s) — estimated expected return
    """
    def __init__(self, state_dim, hidden_dim=256):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)   # (batch,) scalar


def compute_gae(rewards, values, next_values, dones, gamma, lam):
    """
    Generalized Advantage Estimation (GAE-λ)
    δ_t = r_t + γ·V(s_{t+1})·(1-done) - V(s_t)
    A_t = Σ (γλ)^l · δ_{t+l}
    returns = A_t + V(s_t)  ← critic 타깃
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
    returns = advantages + values          # λ-return = A + V
    return advantages, returns


def train(env, M):
    """
    REINFORCE with baseline + GAE + Batch Update 메인 학습 루프.
    - GAE(λ=0.95)로 advantage 추정 → 분산 대폭 감소
    - N_BATCH 에피소드씩 묶어 업데이트 → 학습 안정성 강화
    - Reward Scaling (0.05) 적용 → Critic 수렴 가속
    """

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Hyperparameters
    GAMMA          = 0.99
    LAM            = 0.95       # GAE λ
    ACTOR_LR       = 3e-4
    CRITIC_LR      = 3e-4
    CRITIC_ITERS   = 10         # 배치당 critic 업데이트 횟수
    ENTROPY_START  = 0.02       # 탐색 (너무 높으면 200점 도달이 늦어짐)
    ENTROPY_END    = 0.001
    SOLVE_SCORE    = 200.0
    N_BATCH        = 4          # 4개 에피소드마다 업데이트
    REWARD_SCALE   = 0.05       # 보상 스케일링 (안정적 수렴용)

    # Networks & optimizers
    policy = PolicyNetwork(state_dim, action_dim)
    value  = ValueNetwork(state_dim)
    actor_optimizer  = optim.Adam(policy.parameters(), lr=ACTOR_LR)
    critic_optimizer = optim.Adam(value.parameters(),  lr=CRITIC_LR)
    
    actor_scheduler  = optim.lr_scheduler.CosineAnnealingWarmRestarts(actor_optimizer,  T_0=500, eta_min=1e-5)
    critic_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(critic_optimizer, T_0=500, eta_min=1e-5)

    ep_rewards    = []
    ep_pol_losses = []
    ep_val_losses = []
    best_reward   = -float('inf')

    episode = 0
    update_count = 0

    while episode < M:
        batch_states, batch_log_probs, batch_advantages, batch_returns = [], [], [], []
        batch_ep_rewards = []

        # ── N_BATCH 에피소드 수집 ────────────────────────────────
        for _ in range(N_BATCH):
            if episode >= M: break
            episode += 1

            progress = (episode - 1) / max(M - 1, 1)
            entropy_coeff = ENTROPY_START + (ENTROPY_END - ENTROPY_START) * progress

            state, _ = env.reset()
            done = False
            states_ep, log_probs_ep, rewards_ep, next_states_ep, dones_ep = [], [], [], [], []
            raw_reward_sum = 0

            while not done:
                state_t = torch.FloatTensor(state).unsqueeze(0)
                action, log_prob = policy.sample_action(state_t)
                
                next_state, reward, term, trunc, _ = env.step(action.detach().squeeze(0).numpy())
                done = term or trunc

                states_ep.append(state_t)
                log_probs_ep.append(log_prob)
                rewards_ep.append(reward * REWARD_SCALE)  # Scaling 적용
                next_states_ep.append(torch.FloatTensor(next_state).unsqueeze(0))
                dones_ep.append(done)

                state = next_state
                raw_reward_sum += reward

            # GAE 계산을 위한 Tensor 변환
            s_t = torch.cat(states_ep, dim=0)
            ns_t = torch.cat(next_states_ep, dim=0)
            
            with torch.no_grad():
                vals = value(s_t)
                next_vals = value(ns_t)

            advs, rets = compute_gae(rewards_ep, vals, next_vals, dones_ep, GAMMA, LAM)

            batch_states.append(s_t)
            batch_log_probs.append(torch.cat(log_probs_ep, dim=0))
            batch_advantages.append(advs)
            batch_returns.append(rets)
            batch_ep_rewards.append(raw_reward_sum)

        # ── 배치 데이터 병합 및 업데이트 ───────────────────────────
        all_states     = torch.cat(batch_states, dim=0)
        all_log_probs  = torch.cat(batch_log_probs, dim=0)
        all_advantages = torch.cat(batch_advantages, dim=0)
        all_returns    = torch.cat(batch_returns, dim=0)

        # Advantage 정규화
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        # Policy (actor) loss
        dist = policy.get_distribution(all_states)
        entropy = dist.entropy().sum(dim=-1).mean()
        policy_loss = -(all_log_probs * all_advantages).mean() - entropy_coeff * entropy

        actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        actor_optimizer.step()
        actor_scheduler.step(update_count)

        # Value (critic) loss
        for _ in range(CRITIC_ITERS):
            v_pred = value(all_states)
            value_loss = nn.functional.huber_loss(v_pred, all_returns)
            critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value.parameters(), max_norm=0.5)
            critic_optimizer.step()
        critic_scheduler.step(update_count)
        update_count += 1

        # ── 로깅 및 저장 ──────────────────────────────────────────
        for r_sum in batch_ep_rewards:
            ep_rewards.append(r_sum)
            ep_pol_losses.append(policy_loss.item())
            ep_val_losses.append(value_loss.item())
            if r_sum > best_reward:
                best_reward = r_sum
                torch.save(policy.state_dict(), "best_policy.pth")

        if episode % 20 == 0:
            avg100 = np.mean(ep_rewards[-100:])
            print(f"Episode {episode:4d} | Reward: {batch_ep_rewards[-1]:8.2f} | "
                  f"Avg100: {avg100:8.2f} | PolLoss: {policy_loss.item():.4f} | "
                  f"ValLoss: {value_loss.item():.4f}", flush=True)

        if len(ep_rewards) >= 100 and np.mean(ep_rewards[-100:]) >= SOLVE_SCORE:
            print(f"  [Solved at episode {episode}] Avg100: {np.mean(ep_rewards[-100:]):.2f}", flush=True)
            record_video(policy, state_dim)
            break

    # ── 학습 종료 후 그래프 저장 ─────────────────────────────────
    save_training_graph(ep_rewards, ep_pol_losses, ep_val_losses)
    env.close()
    return policy, value


def record_video(policy, state_dim):
    """학습된 policy로 deterministic 실행 후 비디오 저장"""
    video_env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
    video_env = RecordVideo(video_env, video_folder="videos", episode_trigger=lambda e: True)

    state, _ = video_env.reset()
    done = False
    total_reward = 0.0

    while not done:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = policy.deterministic_action(state_t)
        action_np = action.squeeze(0).numpy()
        state, reward, terminated, truncated, _ = video_env.step(action_np)
        total_reward += reward
        done = terminated or truncated

    video_env.close()
    print(f"  [Video saved] reward: {total_reward:.2f}")


def save_training_graph(rewards, pol_losses, val_losses):
    """에피소드별 reward / policy loss / value loss 그래프 저장"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    episodes = range(1, len(rewards) + 1)
    window = 50  # 이동평균 윈도우

    def moving_avg(data, w):
        return np.convolve(data, np.ones(w) / w, mode='valid')

    # Reward
    axes[0].plot(episodes, rewards, alpha=0.3, color='steelblue', label='Reward')
    if len(rewards) >= window:
        axes[0].plot(range(window, len(rewards) + 1), moving_avg(rewards, window),
                     color='steelblue', linewidth=2, label=f'MA-{window}')
    axes[0].axhline(y=200, color='red', linestyle='--', label='Target (200)')
    axes[0].set_ylabel('Episode Reward')
    axes[0].legend()
    axes[0].grid(True)

    # Policy Loss
    axes[1].plot(episodes, pol_losses, alpha=0.4, color='darkorange')
    if len(pol_losses) >= window:
        axes[1].plot(range(window, len(pol_losses) + 1), moving_avg(pol_losses, window),
                     color='darkorange', linewidth=2)
    axes[1].set_ylabel('Policy Loss')
    axes[1].grid(True)

    # Value Loss
    axes[2].plot(episodes, val_losses, alpha=0.4, color='green')
    if len(val_losses) >= window:
        axes[2].plot(range(window, len(val_losses) + 1), moving_avg(val_losses, window),
                     color='green', linewidth=2)
    axes[2].set_ylabel('Value Loss')
    axes[2].set_xlabel('Episode')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('training_log.png', dpi=150)
    plt.close()
    print("Training graph saved → training_log.png")


def main():
    MAX_EPISODES = 3000
    # render_mode='rgb_array' : 학습 중 화면 출력 없이 빠르게 돌림
    # 비디오는 record_video() 에서 별도로 녹화
    env = gym.make("LunarLander-v3", continuous=True, render_mode='rgb_array')

    os.makedirs("videos", exist_ok=True)
    train(env, MAX_EPISODES)

if __name__ == "__main__":
    main()