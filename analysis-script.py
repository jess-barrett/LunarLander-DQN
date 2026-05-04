"""
analysis.py — Statistical analysis and plots for DQN vs DDQN comparison.
Run this AFTER experiment.py finishes.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ── Load results ──────────────────────────────────────────
data = np.load("experiment_results.npz")
dqn_eval = data["dqn_eval"]       # shape: (n_trials, 100)
ddqn_eval = data["ddqn_eval"]
random_eval = data["random_eval"]
dqn_curves = data["dqn_curves"]   # shape: (n_trials, num_episodes)
ddqn_curves = data["ddqn_curves"]

n_trials = dqn_eval.shape[0]
print(f"Loaded {n_trials} trials per condition.\n")

# ── Summary statistics ────────────────────────────────────
print("=" * 60)
print("SUMMARY STATISTICS (mean evaluation reward per trial)")
print("=" * 60)

random_means = random_eval.mean(axis=1)
dqn_means = dqn_eval.mean(axis=1)
ddqn_means = ddqn_eval.mean(axis=1)

print(f"\nRandom Baseline:")
print(f"  Per-trial means: {[f'{m:.1f}' for m in random_means]}")
print(f"  Overall: mean={random_means.mean():.2f}, std={random_means.std():.2f}")

print(f"\nDQN:")
print(f"  Per-trial means: {[f'{m:.1f}' for m in dqn_means]}")
print(f"  Overall: mean={dqn_means.mean():.2f}, std={dqn_means.std():.2f}")

print(f"\nDDQN:")
print(f"  Per-trial means: {[f'{m:.1f}' for m in ddqn_means]}")
print(f"  Overall: mean={ddqn_means.mean():.2f}, std={ddqn_means.std():.2f}")

# ── Statistical tests ─────────────────────────────────────
print("\n" + "=" * 60)
print("STATISTICAL TESTS (paired t-tests + Wilcoxon signed-rank)")
print("=" * 60)

def run_tests(label, sample_a, sample_b):
    """Run both paired t-test and Wilcoxon signed-rank test."""
    t_stat, t_p = stats.ttest_rel(sample_a, sample_b)
    try:
        w_stat, w_p = stats.wilcoxon(sample_a, sample_b)
    except ValueError:
        w_stat, w_p = float('nan'), float('nan')
    print(f"\n[H0] {label}: mean rewards are equal.")
    print(f"  Paired t-test:        t = {t_stat:.3f}, p = {t_p:.4f}")
    print(f"  Wilcoxon signed-rank: W = {w_stat:.3f}, p = {w_p:.4f}")
    decision_t = 'REJECT H0' if t_p < 0.05 else 'FAIL TO REJECT H0'
    decision_w = 'REJECT H0' if w_p < 0.05 else 'FAIL TO REJECT H0'
    print(f"  Decision (t-test):    {decision_t} at α=0.05")
    print(f"  Decision (Wilcoxon):  {decision_w} at α=0.05")
    return t_stat, t_p, w_stat, w_p

run_tests("DQN vs Random",  dqn_means,  random_means)
run_tests("DDQN vs Random", ddqn_means, random_means)
run_tests("DQN vs DDQN",    dqn_means,  ddqn_means)

# Effect size (Cohen's d)
def cohens_d(a, b):
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0

print(f"\nEffect sizes (Cohen's d):")
print(f"  DQN vs Random:  d = {cohens_d(dqn_means, random_means):.3f}")
print(f"  DDQN vs Random: d = {cohens_d(ddqn_means, random_means):.3f}")
print(f"  DDQN vs DQN:    d = {cohens_d(ddqn_means, dqn_means):.3f}")

# ── Plot 1: Training curves ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
window = 50

def smooth(curves):
    return np.array([np.convolve(c, np.ones(window)/window, mode='valid')
                      for c in curves])

dqn_smooth = smooth(dqn_curves)
ddqn_smooth = smooth(ddqn_curves)

x = np.arange(dqn_smooth.shape[1]) + window
ax.plot(x, dqn_smooth.mean(axis=0), label="DQN (mean)", color="C0", linewidth=2)
ax.fill_between(x, dqn_smooth.mean(axis=0) - dqn_smooth.std(axis=0),
                dqn_smooth.mean(axis=0) + dqn_smooth.std(axis=0),
                alpha=0.2, color="C0")

ax.plot(x, ddqn_smooth.mean(axis=0), label="DDQN (mean)", color="C1", linewidth=2)
ax.fill_between(x, ddqn_smooth.mean(axis=0) - ddqn_smooth.std(axis=0),
                ddqn_smooth.mean(axis=0) + ddqn_smooth.std(axis=0),
                alpha=0.2, color="C1")

ax.axhline(y=200, color="g", linestyle="--", label="Solved threshold (200)")
ax.axhline(y=random_means.mean(), color="r", linestyle="--",
           label=f"Random baseline ({random_means.mean():.0f})")
ax.set_xlabel("Episode")
ax.set_ylabel("50-episode rolling reward")
ax.set_title("Training Curves: DQN vs DDQN (mean ± std across trials)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("training_curves_comparison.png", dpi=150)
plt.show()
print("\nSaved training_curves_comparison.png")

# ── Plot 2: Evaluation comparison (box plot) ──────────────
fig, ax = plt.subplots(figsize=(8, 6))
all_rewards = [random_eval.flatten(), dqn_eval.flatten(), ddqn_eval.flatten()]
labels = ["Random\nBaseline", "DQN", "DDQN"]
bp = ax.boxplot(all_rewards, tick_labels=labels, patch_artist=True)
colors = ["#e74c3c", "#3498db", "#f39c12"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(y=200, color="g", linestyle="--", label="Solved threshold")
ax.set_ylabel("Episode Reward (100 evaluation episodes per trial)")
ax.set_title("Evaluation Performance Distribution")
ax.legend()
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("evaluation_comparison.png", dpi=150)
plt.show()
print("Saved evaluation_comparison.png")

# ── Plot 3: Bar chart of trial means ──────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(3)
means = [random_means.mean(), dqn_means.mean(), ddqn_means.mean()]
stds = [random_means.std(), dqn_means.std(), ddqn_means.std()]
ax.bar(x_pos, means, yerr=stds, capsize=8, color=colors, alpha=0.7,
       edgecolor="black", linewidth=1.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylabel("Mean Evaluation Reward")
ax.set_title(f"Mean Evaluation Reward Across {n_trials} Trials (± std)")
ax.axhline(y=200, color="g", linestyle="--", alpha=0.5)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("trial_means_bar.png", dpi=150)
plt.show()
print("Saved trial_means_bar.png")

print("\n=== Analysis complete. ===")

