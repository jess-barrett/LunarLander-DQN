"""
experiment.py — Compares DQN vs DDQN on LunarLander-v3.
Runs 5 trials of each algorithm, then evaluates each trained agent
over 100 test episodes. Saves results for statistical analysis.
"""

import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import time

# ── Q-Network ─────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

# ── Replay Buffer ─────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)
    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(s2), np.array(d, dtype=np.float32))
    def __len__(self):
        return len(self.buf)

# ── Agent (supports both DQN and DDQN) ────────────────────
class Agent:
    def __init__(self, double=False, state_dim=8, action_dim=4,
                 lr=1e-3, gamma=0.99, batch_size=64,
                 target_update_freq=10, seed=0):
        self.double = double
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()

        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995

    def select_action(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_net(t).argmax(dim=1).item()

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s  = torch.FloatTensor(s).to(self.device)
        a  = torch.LongTensor(a).to(self.device)
        r  = torch.FloatTensor(r).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        d  = torch.FloatTensor(d).to(self.device)

        q_vals = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double:
                # DDQN: use online net to select action, target net to evaluate
                next_actions = self.q_net(s2).argmax(dim=1, keepdim=True)
                max_next_q = self.target_net(s2).gather(1, next_actions).squeeze(1)
            else:
                # DQN: use target net for both
                max_next_q = self.target_net(s2).max(dim=1)[0]
            targets = r + self.gamma * max_next_q * (1 - d)

        loss = nn.MSELoss()(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

# ── Training & Evaluation ─────────────────────────────────
def train_agent(double=False, num_episodes=600, seed=0, label=""):
    env = gym.make("LunarLander-v3")
    agent = Agent(double=double, seed=seed)
    rewards_history = []

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset(seed=seed + ep)
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
        if ep % 100 == 0:
            avg = np.mean(rewards_history[-100:])
            print(f"  [{label}] Ep {ep} | Avg(100): {avg:.1f} | Eps: {agent.epsilon:.3f}")
    env.close()
    return agent, rewards_history

def evaluate_agent(agent, num_episodes=100, seed_offset=10000):
    """Evaluate trained agent greedily (no exploration)."""
    env = gym.make("LunarLander-v3")
    eval_rewards = []
    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed_offset + ep)
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        eval_rewards.append(total_reward)
    env.close()
    return np.array(eval_rewards)

def evaluate_random(num_episodes=100, seed_offset=10000):
    env = gym.make("LunarLander-v3")
    eval_rewards = []
    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed_offset + ep)
        total_reward = 0
        done = False
        while not done:
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        eval_rewards.append(total_reward)
    env.close()
    return np.array(eval_rewards)

# ── Main Experiment ────────────────────────────────────────
def main():
    NUM_TRIALS = 5
    NUM_EPISODES = 600

    results = {
        "dqn_eval": [],     # 5 arrays of 100 eval rewards
        "ddqn_eval": [],
        "random_eval": [],
        "dqn_curves": [],   # 5 training curves
        "ddqn_curves": [],
    }

    start_time = time.time()

    # Random baseline (5 trials of 100 eval episodes each)
    print("\n=== RANDOM BASELINE ===")
    for trial in range(NUM_TRIALS):
        random.seed(trial)
        np.random.seed(trial)
        rewards = evaluate_random(num_episodes=100, seed_offset=10000 + trial * 200)
        print(f"  Trial {trial+1}: mean={rewards.mean():.1f}, std={rewards.std():.1f}")
        results["random_eval"].append(rewards)

    # DQN
    print("\n=== DQN TRIALS ===")
    for trial in range(NUM_TRIALS):
        print(f"\n--- DQN Trial {trial+1}/{NUM_TRIALS} ---")
        agent, curve = train_agent(double=False, num_episodes=NUM_EPISODES,
                                    seed=trial, label=f"DQN-{trial+1}")
        eval_rewards = evaluate_agent(agent, num_episodes=100,
                                       seed_offset=10000 + trial * 200)
        print(f"  DQN-{trial+1} EVAL: mean={eval_rewards.mean():.1f}, std={eval_rewards.std():.1f}")
        results["dqn_eval"].append(eval_rewards)
        results["dqn_curves"].append(curve)
        torch.save(agent.q_net.state_dict(), f"dqn_trial{trial+1}.pth")

    # DDQN
    print("\n=== DDQN TRIALS ===")
    for trial in range(NUM_TRIALS):
        print(f"\n--- DDQN Trial {trial+1}/{NUM_TRIALS} ---")
        agent, curve = train_agent(double=True, num_episodes=NUM_EPISODES,
                                    seed=trial + 100, label=f"DDQN-{trial+1}")
        eval_rewards = evaluate_agent(agent, num_episodes=100,
                                       seed_offset=10000 + trial * 200)
        print(f"  DDQN-{trial+1} EVAL: mean={eval_rewards.mean():.1f}, std={eval_rewards.std():.1f}")
        results["ddqn_eval"].append(eval_rewards)
        results["ddqn_curves"].append(curve)
        torch.save(agent.q_net.state_dict(), f"ddqn_trial{trial+1}.pth")

    elapsed = (time.time() - start_time) / 60
    print(f"\n=== Total time: {elapsed:.1f} minutes ===")

    # Save everything
    np.savez("experiment_results.npz",
             dqn_eval=np.array(results["dqn_eval"]),
             ddqn_eval=np.array(results["ddqn_eval"]),
             random_eval=np.array(results["random_eval"]),
             dqn_curves=np.array(results["dqn_curves"]),
             ddqn_curves=np.array(results["ddqn_curves"]))
    print("Saved experiment_results.npz")

if __name__ == "__main__":
    main()
