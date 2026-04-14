from game.environment import SnakeGameAI
from ai.agent import Agent
from config import NUM_EPISODES

class Trainer:
    """Handles the training loop for the Agent."""
    def __init__(self):
        self.env = SnakeGameAI()
        self.agent = Agent()

    def run_training(self):
        """Execute the training process."""
        print("Starting training...")
        for episode in range(1, NUM_EPISODES + 1):
            state = self.env.reset()
            # Loop content will go here
            pass

if __name__ == '__main__':
    trainer = Trainer()
    trainer.run_training()
