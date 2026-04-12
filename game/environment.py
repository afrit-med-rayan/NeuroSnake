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
