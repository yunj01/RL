import os
import numpy as np
import torch
import gymnasium as gym
from Lunarlander_RF import PolicyNetwork, RunningMeanStd, record_video

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

        os.makedirs("videos", exist_ok=True)
        print("[Recording] Starting video recording...")
        record_video(policy, state_rms)
        print("[Done] Check the 'videos' folder.")
    else:
        print("[Error] 'best_policy.pth' not found.")

if __name__ == "__main__":
    main()
