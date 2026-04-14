import numpy as np
from game.environment import SnakeGameAI
from ai.agent import Agent
from config import NUM_EPISODES, MAX_STEPS, MEAN_WINDOW, TARGET_UPDATE_FREQ, MODEL_PATH

class Trainer:
    """Handles the training loop for the Agent."""
    def __init__(self):
        self.env = SnakeGameAI()
        self.agent = Agent()
        self.scores = []
        self.mean_scores = []
        self.record = 0

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
                    
            self.scores.append(score)
            mean_score = np.mean(self.scores[-MEAN_WINDOW:]) if self.scores else 0
            self.mean_scores.append(mean_score)
            
            if score > self.record:
                self.record = score
                self.agent.online_net.save(MODEL_PATH)
                
            self.agent.decay_epsilon()
            if episode % TARGET_UPDATE_FREQ == 0:
                self.agent.update_target()
                
            print(f"Episode: {episode}/{NUM_EPISODES} | Score: {score} | Record: {self.record} | Mean: {mean_score:.2f} | Epsilon: {self.agent.epsilon:.3f}")

if __name__ == '__main__':
    trainer = Trainer()
    trainer.run_training()
