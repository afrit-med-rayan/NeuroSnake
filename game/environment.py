"""
game/environment.py — NeuroSnake Game Environment
==================================================
Implements the Snake game for both AI training and human play.

Classes:
    Direction  — Enum for movement directions
    Point      — Named tuple for 2D grid coordinates
    SnakeGameAI — Game environment used by the DQN agent
    HumanGame  — Keyboard-controlled Snake for human play
"""

import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─────────────────────────────────────────────
# PRIMITIVES
# ─────────────────────────────────────────────

class Direction(Enum):
    RIGHT = 0
    DOWN  = 1
    LEFT  = 2
    UP    = 3

Point = namedtuple("Point", ["x", "y"])

# Clock-wise order matters for action → direction mapping
CLOCK_WISE = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]


# ─────────────────────────────────────────────
# AI GAME ENVIRONMENT
# ─────────────────────────────────────────────

class SnakeGameAI:
    """
    Snake game environment for the DQN agent.

    Action space  : [1,0,0] = straight | [0,1,0] = right | [0,0,1] = left
    State vector  : 11 binary features (danger, direction, food location)
    Reward        : +10 food, -10 death, -0.1 per step
    """

    def __init__(self, render: bool = False):
        """
        Args:
            render: If True, opens a Pygame window for visualisation.
                    Set False during fast headless training.
        """
        self.render_mode = render
        self.w = config.WINDOW_W
        self.h = config.WINDOW_H

        if self.render_mode:
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h + 50))
            pygame.display.set_caption("NeuroSnake — Training")
            self.clock = pygame.time.Clock()
            self._load_fonts()

        self.reset()

    # ── Initialisation ──────────────────────────────────────────────────

    def _load_fonts(self):
        pygame.font.init()
        self.font_large  = pygame.font.SysFont("consolas", 28, bold=True)
        self.font_small  = pygame.font.SysFont("consolas", 18)

    def reset(self):
        """Reset the game to starting state. Returns initial state vector."""
        # Snake starts in the middle, moving right
        mid_x = (config.GRID_SIZE // 2) * config.CELL_SIZE
        mid_y = (config.GRID_SIZE // 2) * config.CELL_SIZE
        self.direction = Direction.RIGHT
        self.head = Point(mid_x, mid_y)
        self.snake = [
            self.head,
            Point(self.head.x - config.CELL_SIZE,     self.head.y),
            Point(self.head.x - 2 * config.CELL_SIZE, self.head.y),
        ]
        self.score      = 0
        self.food       = None
        self.frame_iter = 0
        self._place_food()
        return self.get_state()

    def _place_food(self):
        """Spawn food at a random cell not occupied by the snake."""
        cs = config.CELL_SIZE
        while True:
            x = random.randint(0, config.GRID_SIZE - 1) * cs
            y = random.randint(0, config.GRID_SIZE - 1) * cs
            self.food = Point(x, y)
            if self.food not in self.snake:
                break

    # ── Collision ────────────────────────────────────────────────────────

    def _is_collision(self, point: Point = None) -> bool:
        """Return True if point (default: head) hits a wall or the snake body."""
        if point is None:
            point = self.head

        cs = config.CELL_SIZE
        gs = config.GRID_SIZE

        # Wall collision
        if (point.x < 0 or point.x >= gs * cs or
                point.y < 0 or point.y >= gs * cs):
            return True

        # Self collision (skip the head itself)
        if point in self.snake[1:]:
            return True

        return False

    # ── Movement ─────────────────────────────────────────────────────────

    def _move(self, action: list):
        """
        Translate action [straight, right, left] into a new head position.
        Uses clock-wise rotation relative to current direction.
        """
        idx = CLOCK_WISE.index(self.direction)

        if action[1]:           # Turn right (clock-wise)
            idx = (idx + 1) % 4
        elif action[2]:         # Turn left (counter-clock-wise)
            idx = (idx - 1) % 4
        # action[0] == straight → idx unchanged

        self.direction = CLOCK_WISE[idx]
        cs = config.CELL_SIZE
        x, y = self.head.x, self.head.y

        if   self.direction == Direction.RIGHT: x += cs
        elif self.direction == Direction.LEFT:  x -= cs
        elif self.direction == Direction.DOWN:  y += cs
        elif self.direction == Direction.UP:    y -= cs

        self.head = Point(x, y)

    # ── Core Step ───────────────────────────────────────────────────────

    def step(self, action: list):
        """
        Execute one game step.

        Args:
            action: One-hot list [straight, right, left]

        Returns:
            reward  (float)
            done    (bool)   — True if game over
            score   (int)
        """
        self.frame_iter += 1

        # 1. Handle Pygame quit event (when rendering)
        if self.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        # 2. Move
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. Check collision / timeout
        reward = config.REWARD_STEP
        done   = False

        if self._is_collision() or self.frame_iter > config.MAX_STEPS:
            reward = config.REWARD_DEATH
            done   = True
            return reward, done, self.score

        # 4. Check food
        if self.head == self.food:
            self.score += 1
            reward = config.REWARD_FOOD
            self._place_food()
        else:
            self.snake.pop()   # Remove tail (no growth)

        # 5. Render
        if self.render_mode:
            self.render()

        return reward, done, self.score

