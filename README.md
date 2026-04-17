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

