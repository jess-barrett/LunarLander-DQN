import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("rewards_history.npy")

# Compute rolling 100-episode average
avg = np.convolve(rewards, np.ones(100)/100, mode='valid')

plt.figure(figsize=(10, 5))
plt.plot(rewards, alpha=0.3, label="Episode Reward")
plt.plot(range(99, len(rewards)), avg, label="100-Episode Average", linewidth=2)
plt.axhline(y=200, color='g', linestyle='--', label="Solved Threshold (200)")
plt.axhline(y=-150, color='r', linestyle='--', label="Random Baseline (~-150)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Training on LunarLander-v2")
plt.legend()
plt.tight_layout()
plt.savefig("training_curve.png", dpi=150)
plt.show()
print("Saved training_curve.png")