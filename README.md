<div align="center">

```
███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗ ███████╗███╗   ██╗ █████╗ ██╗  ██╗███████╗
████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔════╝████╗  ██║██╔══██╗██║ ██╔╝██╔════╝
██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║███████╗██╔██╗ ██║███████║█████╔╝ █████╗  
██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║╚════██║██║╚██╗██║██╔══██║██╔═██╗ ██╔══╝  
██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝███████║██║ ╚████║██║  ██║██║  ██╗███████╗
╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
```

**An autonomous Snake agent that learns to play from scratch using Deep Q-Network (DQN) reinforcement learning**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Pygame](https://img.shields.io/badge/Pygame-2.x-00b140?logo=python&logoColor=white)](https://pygame.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557c?logo=python&logoColor=white)](https://matplotlib.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 🧠 Overview

**NeuroSnake** teaches a neural network to play Snake from absolute zero — no hardcoded rules, no lookahead algorithms. The agent observes an 11-feature state vector, picks actions using a Deep Q-Network, and learns purely through trial and error guided by a sparse reward signal.

Training leverages two core DRL stabilisation techniques:
- **Experience Replay** — random mini-batches break temporal correlations in training data
- **Target Network** — a periodically-frozen copy of the Q-network prevents oscillating Q-value updates

After ~300–600 episodes the agent reliably collects food and avoids walls. After 1 000 episodes it develops efficient routing strategies entirely on its own.

### 🎯 Project Objectives

| # | Objective |
|---|-----------|
| 1 | Implement a complete, production-quality DQN from scratch using PyTorch |
| 2 | Build a Pygame Snake environment with a clean `reset / step / get_state` API |
| 3 | Demonstrate that compact hand-crafted state features outperform raw pixels for this task |
| 4 | Provide three distinct modes — **train**, **human play**, and **AI eval** — via a unified CLI |
| 5 | Visualise the learning curve live during training and export it as a PNG artifact |

---

## 🏗️ Architecture

### Project Structure

```
NeuroSnake/
├── game/
│   └── environment.py      # Pygame Snake environment (SnakeGameAI + HumanGame)
├── ai/
│   ├── model.py            # DQN neural network  (PyTorch nn.Module)
│   ├── agent.py            # Agent: ε-greedy policy, replay, target net
│   └── replay_buffer.py    # Circular deque-based experience replay buffer
├── training/
│   └── train.py            # Trainer class — full episode loop
├── utils/
│   └── plot.py             # LivePlot — non-blocking Matplotlib learning curve
├── models/
│   └── dqn_snake.pth       # Saved model weights (auto-created on record score)
├── main.py                 # CLI entry point  (train | human | eval)
└── config.py               # All hyperparameters in one place
```

### Neural Network (DQN)

```
                  ┌───────────────────┐
  State (11)  ──► │  Linear  11 → 128 │ ──► ReLU
                  └─────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │  Linear 128 → 256 │ ──► ReLU
                  └─────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │  Linear  256 → 3  │
                  └─────────┬─────────┘
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
         Q(straight)    Q(right)       Q(left)
```

The agent picks the action with the highest Q-value during exploitation, or a random action during exploration (ε-greedy).

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| State representation | 11 binary features | Fast, efficient — no CNN overhead |
| Action space | 3 (straight / right / left) | Relative directions eliminate illegal 180° reversals |
| NN architecture | 11 → 128 → 256 → 3 | Sufficient capacity; no overfitting on compact state |
| Exploration | ε-greedy (exponential decay) | Standard, proven for tabular-to-DQN transfer |
| Stability | Target network (frozen copy) | Breaks the moving-target problem in Q-learning |
| Optimizer | Adam (lr = 0.001) | Adaptive learning rate — robust across hyperparameter ranges |
| Discount factor | γ = 0.99 | High future-reward weight encourages survival |


