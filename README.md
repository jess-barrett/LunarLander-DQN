# LunarLander-DQN

Comparing **Deep Q-Network (DQN)** and **Double Deep Q-Network (DDQN)** on the Gymnasium LunarLander-v3 environment.

This is the term project for CIS 530 (Spring 2026) at Kansas State University.

## Project Overview

This project trains reinforcement learning agents to land a spacecraft in the LunarLander-v3 simulation. It compares three approaches:

- **Random baseline** — uniform-random action policy
- **DQN** — Deep Q-Network with experience replay and target network
- **Double DQN (DDQN)** — DQN with decoupled action selection/evaluation to reduce overestimation bias

Each method was run for 5 independent training trials of 600 episodes each, then evaluated greedily over 100 fixed evaluation episodes per trial. Statistical comparisons use paired t-tests and Wilcoxon signed-rank tests with Cohen's *d* effect sizes.

## Headline Results

| Method   | Mean Eval Reward | Std Dev |
|----------|------------------|---------|
| Random   | -189.4           |  7.9    |
| DQN      |  151.5           | 37.9    |
| DDQN     |  117.6           | 71.2    |

Both DQN and DDQN beat the random baseline with very large effect sizes (Cohen's *d* > 5, *p* < 0.005). DDQN did **not** show a statistically significant improvement over vanilla DQN (*p* = 0.46) in this setting.

See `report.pdf` for the full writeup.

## Repository Contents

```
.
├── baseline.py                 # Random-action baseline (single run)
├── dqn_agent.py                # Original single-run DQN trainer
├── experiment.py               # Multi-trial DQN vs DDQN comparison experiment
├── analysis.py                 # Statistical tests and plot generation
├── plot_results.py             # Single-run training curve plot
├── report.pdf                  # Final IEEE-formatted report
├── experiment_results.npz      # Raw experiment data (5 trials × 100 eval episodes per method)
├── best_dqn.pth                # Best DQN model weights from initial training
├── dqn_trial[1-5].pth          # DQN model weights from comparison experiment
├── ddqn_trial[1-5].pth         # DDQN model weights from comparison experiment
├── training_curves_comparison.png
├── evaluation_comparison.png
├── trial_means_bar.png
└── README.md
```

## Setup

### Requirements

- Python 3.10+
- PyTorch
- Gymnasium with Box2D
- NumPy, SciPy, Matplotlib

### Install

```bash
pip install gymnasium[box2d] numpy torch matplotlib scipy
```

### Recommended Environment

This project was developed and tested in **Google Colab** with a T4 GPU runtime. The full experiment (10 training runs) takes approximately 2.5 hours on Colab GPU.

## Reproducing Results

### 1. Run the random baseline

```bash
python baseline.py
```

Expected output: mean reward around -170 to -190.

### 2. Run the full comparison experiment

```bash
python experiment.py
```

This runs 5 trials each of DQN and DDQN, evaluates each, and saves all results to `experiment_results.npz`. Trained model weights are saved to `dqn_trial[N].pth` and `ddqn_trial[N].pth`.

**Expected runtime:** ~2.5 hours on Colab GPU, longer on CPU.

### 3. Generate statistical analysis and plots

```bash
python analysis.py
```

This loads `experiment_results.npz`, runs paired t-tests and Wilcoxon signed-rank tests, computes Cohen's *d* effect sizes, and generates three figures: training curves, evaluation distribution box plot, and trial means bar chart.

## Hyperparameters

| Parameter                | Value     |
|--------------------------|-----------|
| Hidden layers            | 2 × 128 ReLU |
| Replay buffer capacity   | 100,000   |
| Batch size               | 64        |
| Discount factor (γ)      | 0.99      |
| Learning rate            | 1e-3      |
| Optimizer                | Adam      |
| Loss                     | MSE       |
| Target network sync      | every 10 episodes |
| ε start                  | 1.0       |
| ε end                    | 0.01      |
| ε decay                  | 0.995 / episode |
| Episodes per trial       | 600       |
| Trials per method        | 5         |
| Evaluation episodes      | 100 per trial (greedy, ε=0) |

## Citations

The implementation follows:

- Mnih et al. (2013, 2015) — DQN, experience replay, target network
- van Hasselt et al. (2016) — Double DQN
- Sutton & Barto (2018) — Q-learning fundamentals
- Henderson et al. (2018) — Reproducibility concerns informing experimental design

Full citations are in the report bibliography.

## Author

Jess Barrett (EID: 854433853)
CIS 530 — Spring 2026
Kansas State University
