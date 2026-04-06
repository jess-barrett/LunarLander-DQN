import gymnasium as gym
import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

# ── Neural Network ──────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ── Replay Buffer ───────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(s2), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)

# ── DQN Agent ───────────────────────────────────────────────
class DQNAgent:
    def __init__(self, state_dim=8, action_dim=4, lr=1e-3,
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995, batch_size=64,
                 target_update_freq=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon-greedy schedule
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-network and target network
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_net(t).argmax(dim=1).item()

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s  = torch.FloatTensor(s).to(self.device)
        a  = torch.LongTensor(a).to(self.device)
        r  = torch.FloatTensor(r).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        d  = torch.FloatTensor(d).to(self.device)

        # Current Q-values for chosen actions
        q_vals = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Target Q-values from target network
        with torch.no_grad():
            max_next_q = self.target_net(s2).max(dim=1)[0]
            targets = r + self.gamma * max_next_q * (1 - d)

        loss = nn.MSELoss()(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

# ── Training Loop ───────────────────────────────────────────
def train(num_episodes=600, print_every=50):
    env = gym.make("LunarLander-v3")
    agent = DQNAgent()

    rewards_history = []
    best_avg = -float("inf")

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.buffer.push(state, action, reward, next_state, float(done))
            agent.train_step()
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        if ep % agent.target_update_freq == 0:
            agent.update_target()

        rewards_history.append(total_reward)
        avg_100 = np.mean(rewards_history[-100:])

        if ep % print_every == 0:
            print(f"Ep {ep:4d} | Reward: {total_reward:7.1f} | "
                  f"Avg(100): {avg_100:7.1f} | Eps: {agent.epsilon:.3f}")

        # Save best model
        if avg_100 > best_avg and ep >= 100:
            best_avg = avg_100
            torch.save(agent.q_net.state_dict(), "best_dqn.pth")

    env.close()

    # Final summary
    print("\n── Training Complete ──")
    print(f"Best 100-ep average: {best_avg:.1f}")
    print(f"Final 100-ep average: {np.mean(rewards_history[-100:]):.1f}")

    # Save reward history for plotting
    np.save("rewards_history.npy", np.array(rewards_history))
    print("Saved rewards_history.npy and best_dqn.pth")

    return rewards_history

if __name__ == "__main__":
    train()