import gymnasium as gym
import torch

env = gym.make('LunarLander-v3', continuous=True)
print('gym OK')
print('torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('state_dim:', env.observation_space.shape)
print('action_dim:', env.action_space.shape)
env.close()
print('All good!')
