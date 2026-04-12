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

