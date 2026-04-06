import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v3")

num_episodes = 100
rewards = []

for ep in range(num_episodes):
    obs, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = env.action_space.sample()  # random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    rewards.append(total_reward)

print(f"Random Agent over {num_episodes} episodes:")
print(f"  Mean reward:  {np.mean(rewards):.2f}")
print(f"  Std reward:   {np.std(rewards):.2f}")
print(f"  Min reward:   {np.min(rewards):.2f}")
print(f"  Max reward:   {np.max(rewards):.2f}")

env.close()