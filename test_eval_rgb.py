import torch
import gymnasium as gym
from Lunarlander_RF import PolicyNetwork
import numpy as np

def evaluate():
    print("Initializing environment with rgb_array...")
    env = gym.make("LunarLander-v3", continuous=True, render_mode='rgb_array')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    policy = PolicyNetwork(state_dim, action_dim)
    try:
        policy.load_state_dict(torch.load("best_policy.pth", map_location='cpu'))
        print("Successfully loaded best_policy.pth")
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = policy.deterministic_action(state_t)
        state, reward, term, trunc, _ = env.step(action.squeeze(0).numpy())
        total_reward += reward
        done = term or trunc
    
    print(f"Evaluation Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    evaluate()
