from game.environment import SnakeGameAI
from ai.agent import Agent

class Trainer:
    """Handles the training loop for the Agent."""
    def __init__(self):
        self.env = SnakeGameAI()
        self.agent = Agent()

    def run_training(self):
        """Execute the training process."""
        print("Starting training...")

if __name__ == '__main__':
    trainer = Trainer()
    trainer.run_training()
