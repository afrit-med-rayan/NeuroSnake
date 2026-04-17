"""
main.py — NeuroSnake Entry Point
=================================
Launch the project in one of three modes:

    python main.py train   →  Train the DQN agent from scratch (or resume)
    python main.py human   →  Play Snake yourself (Arrow / WASD)
    python main.py eval    →  Watch the trained AI play (no exploration)

Usage
-----
    python main.py [train | human | eval]
"""

import sys
import argparse


# ─────────────────────────────────────────────
# CLI ARGUMENT PARSER
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="NeuroSnake",
        description="Deep Q-Network Snake Agent — choose a mode to run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py train   # Train from scratch\n"
            "  python main.py human   # Play yourself\n"
            "  python main.py eval    # Watch the trained AI\n"
        ),
    )
    parser.add_argument(
        "mode",
        choices=["train", "human", "eval"],
        help="Run mode: train | human | eval",
    )
    return parser


# ─────────────────────────────────────────────
# MODE HANDLERS  (stubs — filled in next steps)
# ─────────────────────────────────────────────

def run_train():
    """Train mode — hand off to the Trainer class."""
    from training.train import Trainer
    trainer = Trainer()
    trainer.run_training()


def run_human():
    """Human mode — launch keyboard-controlled Snake."""
    from game.environment import HumanGame
    game = HumanGame()
    game.run()


def run_eval():
    """Eval mode — load saved weights, watch AI play (no training, no exploration)."""
    import time
    import numpy as np
    import torch
    import pygame

    from game.environment import SnakeGameAI, Direction, CLOCK_WISE
    from ai.model import DQN
    from config import MODEL_PATH, GAME_SPEED

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"[Eval] Loading model from: {MODEL_PATH}")
    model = DQN.load(MODEL_PATH)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ── Set up environment with rendering ────────────────────────────────────
    env = SnakeGameAI(render=True)
    pygame.display.set_caption("NeuroSnake — Eval (AI)")

    episode   = 0
    record    = 0

    print("[Eval] Press Ctrl+C or close the window to quit.\n")

    try:
        while True:
            state = env.reset()
            done  = False
            episode += 1

            while not done:
                # ── Greedy action (ε = 0, pure exploitation) ────────────────
                state_t  = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_vals = model(state_t)
                action_idx = torch.argmax(q_vals).item()

                # Convert scalar index → one-hot list expected by env.step()
                action = [0, 0, 0]
                action[action_idx] = 1

                # ── Step environment ─────────────────────────────────────────
                reward, done, score = env.step(action)
                state = env.get_state()

            if score > record:
                record = score

            print(f"[Eval] Episode {episode:>4}  |  Score: {score:>3}  |  Record: {record}")

    except KeyboardInterrupt:
        print("\n[Eval] Interrupted by user.")
    finally:
        pygame.quit()
        print("[Eval] Pygame window closed. Goodbye!")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = build_parser()

    # Show help if no arguments provided
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    mode_map = {
        "train": run_train,
        "human": run_human,
        "eval":  run_eval,
    }

    try:
        mode_map[args.mode]()
    except KeyboardInterrupt:
        print("\n[NeuroSnake] KeyboardInterrupt — exiting gracefully.")
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass
        sys.exit(0)


if __name__ == "__main__":
    main()
