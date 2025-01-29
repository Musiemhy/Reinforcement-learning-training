import gym
import minerl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Load the MineRL Treechop environment
env = gym.make("MineRLTreechop-v0")

# Create a vectorized environment required for Stable-Baselines3
vec_env = make_vec_env(lambda: env, n_envs=1)

# Define the PPO model#, Use a CNN policy for image-based observations
model = PPO(
    "CnnPolicy", 
    vec_env,
    verbose=1,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
)

# Train the model
model.learn(total_timesteps=100000, tb_log_name="ppo_treechop")

# Save the model
model.save("minerl_treechop_ppo")

# Evaluate the model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()