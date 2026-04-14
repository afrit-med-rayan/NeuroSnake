import collections
import random
import torch
import numpy as np

class ReplayBuffer:
    """Circular experience replay buffer for DQN."""
    
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Save a transition to the replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a random batch of transitions and return as PyTorch tensors."""
        transitions = random.sample(self.buffer, batch_size)
        
        # Unzip the batch
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        # Convert to tensors
        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32) # Using float32 for easy multiply (1 - done) in Q-learning eq
        
        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self):
        """Return current size of the buffer."""
        return len(self.buffer)
