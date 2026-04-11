"""
config.py — NeuroSnake Central Configuration
=============================================
All hyperparameters and constants in one place.
Modify this file to experiment with different settings.
"""

# ─────────────────────────────────────────────
# GAME SETTINGS
# ─────────────────────────────────────────────
GRID_SIZE   = 20          # Number of cells per side (20×20 grid)
CELL_SIZE   = 30          # Pixels per cell
WINDOW_W    = GRID_SIZE * CELL_SIZE   # 600 px
WINDOW_H    = GRID_SIZE * CELL_SIZE   # 600 px
GAME_SPEED  = 40          # FPS during training (higher = faster training)
HUMAN_SPEED = 10          # FPS for human play mode

# ─────────────────────────────────────────────
# STATE & ACTION SPACE
# ─────────────────────────────────────────────
STATE_SIZE  = 11   # [danger×3, direction×4, food×4]
ACTION_SIZE = 3    # [straight, right, left]

# ─────────────────────────────────────────────
# NEURAL NETWORK ARCHITECTURE
# ─────────────────────────────────────────────
HIDDEN_LAYER_1 = 128
HIDDEN_LAYER_2 = 256

# ─────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────
LEARNING_RATE   = 0.001        # Adam optimizer LR
GAMMA           = 0.99         # Discount factor (future reward weight)
BATCH_SIZE      = 64           # Samples per training step
MEMORY_CAPACITY = 100_000      # Max experiences in replay buffer

# ─────────────────────────────────────────────
# EXPLORATION (Epsilon-Greedy)
# ─────────────────────────────────────────────
EPSILON_START = 1.0            # Start fully random
EPSILON_MIN   = 0.01           # Never go below 1% random
EPSILON_DECAY = 0.995          # Multiply epsilon by this each episode

# ─────────────────────────────────────────────
# TARGET NETWORK
# ─────────────────────────────────────────────
TARGET_UPDATE_FREQ = 10        # Copy online → target every N episodes

# ─────────────────────────────────────────────
# REWARDS
# ─────────────────────────────────────────────
REWARD_FOOD      =  10.0
REWARD_DEATH     = -10.0
REWARD_STEP      =  -0.1       # Small penalty per step → encourages efficiency

# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
NUM_EPISODES    = 1000         # Total training episodes
MAX_STEPS       = GRID_SIZE * GRID_SIZE * 2  # Max steps before forced reset
MEAN_WINDOW     = 100          # Rolling mean window size

# ─────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────
MODEL_PATH       = "models/dqn_snake.pth"
PLOT_OUTPUT_PATH = "training_curve.png"

# ─────────────────────────────────────────────
# COLORS  (RGB)
# ─────────────────────────────────────────────
COLOR_BG         = (10,  10,  25)    # Deep navy background
COLOR_GRID       = (20,  20,  45)    # Subtle grid lines
COLOR_SNAKE_HEAD = (0,   220, 150)   # Teal-green head
COLOR_SNAKE_BODY = (0,   170, 100)   # Slightly darker body
COLOR_FOOD       = (255, 80,  100)   # Vivid red-pink food
COLOR_TEXT       = (200, 200, 255)   # Soft lavender text
COLOR_SCORE      = (255, 220, 0)     # Gold score text
