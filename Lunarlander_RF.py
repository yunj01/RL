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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count

        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class PolicyNetwork(nn.Module):
    """Actor: Gaussian policy"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.mean_head = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_head(x)
        std = self.log_std.clamp(-20, 2).exp().expand_as(mean)
        return mean, std

    def get_distribution(self, state):
        mean, std = self.forward(state)
        return Normal(mean, std)

    def sample_action(self, state):
        dist = self.get_distribution(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def deterministic_action(self, state):
        mean, _ = self.forward(state)
        return mean.clamp(-1, 1)


class ValueNetwork(nn.Module):
    """Critic: state-value baseline"""
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


def compute_gae(rewards, values, next_values, dones, gamma, lam):
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def train(env, M):
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # REINFORCE with baseline hyperparameters
    GAMMA         = 0.99
    LAM           = 0.95
    ACTOR_LR      = 3e-4
    CRITIC_LR     = 1e-3
    CRITIC_ITERS  = 20
    ENTROPY_START = 0.02
    ENTROPY_END   = 0.0
    N_BATCH       = 16
    MAX_GRAD_NORM = 0.5
    SOLVE_SCORE   = 200.0

    policy = PolicyNetwork(state_dim, action_dim)
    value  = ValueNetwork(state_dim)
    state_rms = RunningMeanStd(shape=(state_dim,))

    if os.path.exists("best_policy_old.pth"):
        try:
            ckpt = torch.load("best_policy_old.pth", map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict) and 'policy' in ckpt:
                policy.load_state_dict(ckpt['policy'])
                if 'value' in ckpt:
                    value.load_state_dict(ckpt['value'])
                if 'rms_mean' in ckpt:
                    state_rms.mean  = ckpt['rms_mean']
                    state_rms.var   = ckpt['rms_var']
                    state_rms.count = ckpt['rms_count']
                print("  [Checkpoint] Loaded best_policy_old.pth (full)", flush=True)
            else:
                policy.load_state_dict(ckpt)
                print("  [Checkpoint] Loaded best_policy_old.pth (policy only)", flush=True)
        except Exception as e:
            print(f"  [Checkpoint] Load failed: {e}", flush=True)

    actor_optimizer  = optim.Adam(policy.parameters(), lr=ACTOR_LR,  eps=1e-5)
    critic_optimizer = optim.Adam(value.parameters(),  lr=CRITIC_LR, eps=1e-5)

    total_updates = max(1, M // N_BATCH)
    actor_scheduler  = optim.lr_scheduler.LambdaLR(
        actor_optimizer,  lambda u: max(0.0, 1.0 - u / total_updates)
    )
    critic_scheduler = optim.lr_scheduler.LambdaLR(
        critic_optimizer, lambda u: max(0.0, 1.0 - u / total_updates)
    )

    ep_rewards = []
    best_avg_reward = -float('inf')
    episode = 0
    update_count = 0

    while episode < M:
        batch_states, batch_log_probs = [], []
        batch_advantages, batch_returns = [], []
        batch_ep_rewards = []

        # --- Trajectory collection ---
        for _ in range(N_BATCH):
            if episode >= M:
                break
            episode += 1

            state, _ = env.reset()
            state_rms.update(state.reshape(1, -1))
            done = False

            states_ep, log_probs_ep = [], []
            rewards_ep, next_states_ep, dones_ep = [], [], []
            raw_reward_sum = 0.0

            while not done:
                state_norm = state_rms.normalize(state).astype(np.float32)
                state_t = torch.from_numpy(state_norm).unsqueeze(0)

                action, log_prob = policy.sample_action(state_t)

                action_np  = action.detach().squeeze(0).numpy()
                action_env = np.clip(action_np, -1.0, 1.0)
                next_state, reward, term, trunc, _ = env.step(action_env)
                done = term or trunc

                states_ep.append(state_t.squeeze(0))
                log_probs_ep.append(log_prob.squeeze(0))
                rewards_ep.append(float(reward))
                dones_ep.append(done)

                state_rms.update(next_state.reshape(1, -1))
                ns_norm = state_rms.normalize(next_state).astype(np.float32)
                next_states_ep.append(torch.from_numpy(ns_norm))

                state = next_state
                raw_reward_sum += float(reward)

            s_t  = torch.stack(states_ep)
            ns_t = torch.stack(next_states_ep)
            lp_t = torch.stack(log_probs_ep)

            with torch.no_grad():
                vals = value(s_t)
                next_vals = value(ns_t)

            advs, rets = compute_gae(rewards_ep, vals, next_vals, dones_ep, GAMMA, LAM)

            batch_states.append(s_t)
            batch_log_probs.append(lp_t)
            batch_advantages.append(advs)
            batch_returns.append(rets)
            batch_ep_rewards.append(raw_reward_sum)

        # --- Prepare batch tensors ---
        all_states     = torch.cat(batch_states)
        all_log_probs  = torch.cat(batch_log_probs)
        all_advantages = torch.cat(batch_advantages)
        all_returns    = torch.cat(batch_returns)

        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        progress = update_count / max(total_updates - 1, 1)
        entropy_coeff = ENTROPY_START + (ENTROPY_END - ENTROPY_START) * progress

        # --- Actor update: REINFORCE with baseline (single step) ---
        dist = policy.get_distribution(all_states)
        entropy = dist.entropy().sum(dim=-1).mean()
        policy_loss = -(all_log_probs * all_advantages).mean() - entropy_coeff * entropy

        actor_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
        actor_optimizer.step()
        actor_scheduler.step()

        # --- Critic update: fit baseline to returns ---
        for _ in range(CRITIC_ITERS):
            v_pred = value(all_states)
            value_loss = nn.functional.huber_loss(v_pred, all_returns)
            critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(value.parameters(), MAX_GRAD_NORM)
            critic_optimizer.step()
        critic_scheduler.step()

        update_count += 1

        for r_sum in batch_ep_rewards:
            ep_rewards.append(r_sum)
            avg100 = np.mean(ep_rewards[-100:]) if len(ep_rewards) >= 100 else np.mean(ep_rewards)
            print(f"Episode {len(ep_rewards):4d} | Reward: {r_sum:8.2f} | Avg100: {avg100:8.2f}", flush=True)
            if len(ep_rewards) >= 100 and avg100 > best_avg_reward:
                best_avg_reward = avg100
                torch.save({
                    'policy':    policy.state_dict(),
                    'value':     value.state_dict(),
                    'rms_mean':  state_rms.mean,
                    'rms_var':   state_rms.var,
                    'rms_count': state_rms.count,
                }, "best_policy.pth")

        if len(ep_rewards) % 200 == 0:
            save_training_graph(ep_rewards)

        if len(ep_rewards) >= 100 and np.mean(ep_rewards[-100:]) >= SOLVE_SCORE:
            print(f"  [Solved at episode {len(ep_rewards)}] Avg100: {np.mean(ep_rewards[-100:]):.2f}", flush=True)
            record_video(policy, state_rms)
            break

    save_training_graph(ep_rewards)
    env.close()
    return policy, value


def record_video(policy, state_rms):
    video_env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
    video_env = RecordVideo(video_env, video_folder="videos", episode_trigger=lambda e: True)
    state, _ = video_env.reset()
    done = False
    total_reward = 0.0
    while not done:
        state_norm = state_rms.normalize(state).astype(np.float32)
        state_t = torch.from_numpy(state_norm).unsqueeze(0)
        with torch.no_grad():
            action = policy.deterministic_action(state_t)
        state, reward, term, trunc, _ = video_env.step(action.squeeze(0).numpy())
        total_reward += reward
        done = term or trunc
    video_env.close()
    print(f"  [Video saved] reward: {total_reward:.2f}")


def save_training_graph(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, color='steelblue', label='Reward')
    if len(rewards) >= 50:
        window = 50
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), ma, color='steelblue', linewidth=2, label=f'MA-{window}')
    plt.axhline(y=200, color='red', linestyle='--', label='Target (200)')
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_log.png', dpi=150)
    plt.close()


def main():
    MAX_EPISODES = 5000
    env = gym.make("LunarLander-v3", continuous=True, render_mode='rgb_array')
    os.makedirs("videos", exist_ok=True)
    train(env, MAX_EPISODES)


if __name__ == "__main__":
    main()
