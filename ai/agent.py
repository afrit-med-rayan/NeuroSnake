import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from ai.model import DQN
from ai.replay_buffer import ReplayBuffer
from config import (
    LEARNING_RATE, GAMMA, BATCH_SIZE, MEMORY_CAPACITY,
    EPSILON_START, EPSILON_MIN, EPSILON_DECAY, ACTION_SIZE
)

class Agent:
    def __init__(self):
        """Initialize the DQN agent."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks
        self.online_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval() # Target network is only for inference
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(MEMORY_CAPACITY)
        self.criterion = nn.MSELoss()
        
        # Hyperparameters
        self.epsilon = EPSILON_START
        self.steps_done = 0
        self.episodes_done = 0

    def get_action(self, state):
        """Epsilon-greedy selection (explore vs exploit)."""
        if random.random() < self.epsilon:
            # Explore: random action
            action = random.randint(0, ACTION_SIZE - 1)
        else:
            # Exploit: best action from DQN
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            self.online_net.eval()
            with torch.no_grad():
                q_values = self.online_net(state_tensor)
            self.online_net.train()
            action = torch.argmax(q_values).item()
            
        self.steps_done += 1
        return action

    def remember(self, state, action, reward, next_state, done):
        """Push experience to replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """Sample batch, compute TD target, backprop loss."""
        # Wait until we have enough samples
        if len(self.memory) < BATCH_SIZE:
            return 0.0
            
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        # Move tensors to the designated device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q values (from online network for the chosen actions)
        # gather takes the Q values for the specific actions we played
        current_q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Expected Q values (TD Target)
        # We don't want gradients for the target network
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            
        # Target = Reward + Gamma * max(Next Q) (if not done)
        target_q_values = rewards + GAMMA * max_next_q_values * (1 - dones)

        # Compute Loss
        loss = self.criterion(current_q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """Copy online weights -> target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self):
        """Epsilon decay logic (after each episode)."""
        self.episodes_done += 1
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(EPSILON_MIN, self.epsilon)
