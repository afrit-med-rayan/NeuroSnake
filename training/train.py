from game.environment import SnakeGameAI
from ai.agent import Agent
from config import NUM_EPISODES, MAX_STEPS

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
            score = 0
            
            for step in range(MAX_STEPS):
                action = self.agent.get_action(state)
                next_state, reward, done, current_score = self.env.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                self.agent.train_step()
                
                state = next_state
                score = current_score
                
                if done:
                    break

if __name__ == '__main__':
    trainer = Trainer()
    trainer.run_training()
