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
