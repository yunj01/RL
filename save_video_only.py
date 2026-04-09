import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from Lunarlander_RF import PolicyNetwork, RunningMeanStd

def evaluate_and_record(policy, state_rms, num_episodes=10):
    os.makedirs("videos", exist_ok=True)
    best_reward = -float('inf')
    best_episode = -1
    
    # First, run multiple episodes without recording to find the best seed/trajectory
    print(f"[Search] Running {num_episodes} episodes to find the best landing (between flags)...")
    env = gym.make("LunarLander-v3", continuous=True)
    
    best_seed = None
    for i in range(num_episodes):
        seed = int(np.random.randint(0, 100000))
        state, _ = env.reset(seed=seed)
        done = False
        total_reward = 0.0
        final_x = 0.0
        
        while not done:
            state_norm = state_rms.normalize(state).astype(np.float32)
            state_t = torch.from_numpy(state_norm).unsqueeze(0)
            with torch.no_grad():
                action = policy.deterministic_action(state_t)
            state, reward, term, trunc, _ = env.step(action.squeeze(0).numpy())
            total_reward += reward
            done = term or trunc
            final_x = state[0] # x position
            
        # Check if landed safely and between flags (x is roughly between -0.2 and 0.2)
        is_between_flags = -0.25 < final_x < 0.25
        
        print(f"  Trial {i+1}: Reward = {total_reward:.2f}, Final X = {final_x:.3f} (Between flags: {is_between_flags})")
        
        if is_between_flags and total_reward > best_reward:
            best_reward = total_reward
            best_seed = seed
            
    env.close()
    
    if best_seed is None:
        print("\n[Warning] Could not find a perfect landing between flags. Will just record a normal episode.")
        best_seed = int(np.random.randint(0, 100000))
    else:
        print(f"\n[Found] Best episode found with Reward = {best_reward:.2f}. Now recording...")

    # Now record that specific seed
    video_env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
    video_env = RecordVideo(video_env, video_folder="videos", episode_trigger=lambda e: True, name_prefix="best_landing")
    
    state, _ = video_env.reset(seed=best_seed)
    done = False
    final_reward = 0.0
    while not done:
        state_norm = state_rms.normalize(state).astype(np.float32)
        state_t = torch.from_numpy(state_norm).unsqueeze(0)
        with torch.no_grad():
            action = policy.deterministic_action(state_t)
        state, reward, term, trunc, _ = video_env.step(action.squeeze(0).numpy())
        final_reward += reward
        done = term or trunc
        
    video_env.close()
    print(f"[Done] Video saved successfully with reward: {final_reward:.2f}")

def main():
    env = gym.make("LunarLander-v3", continuous=True)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    env.close()

    policy = PolicyNetwork(state_dim, action_dim)
    state_rms = RunningMeanStd(shape=(state_dim,))

    if os.path.exists("best_policy.pth"):
        ckpt = torch.load("best_policy.pth", map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and 'policy' in ckpt:
            policy.load_state_dict(ckpt['policy'])
            if 'rms_mean' in ckpt:
                state_rms.mean  = ckpt['rms_mean']
                state_rms.var   = ckpt['rms_var']
                state_rms.count = ckpt['rms_count']
            print("[Success] 'best_policy.pth' loaded.")
        else:
            policy.load_state_dict(ckpt)
            print("[Success] 'best_policy.pth' (legacy) loaded.")

        evaluate_and_record(policy, state_rms, num_episodes=20)
    else:
        print("[Error] 'best_policy.pth' not found.")

if __name__ == "__main__":
    main()
