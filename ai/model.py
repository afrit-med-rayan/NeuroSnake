import torch
import torch.nn as nn
import os

from config import STATE_SIZE, HIDDEN_LAYER_1, HIDDEN_LAYER_2, ACTION_SIZE


class DQN(nn.Module):
    def __init__(self, input_size=STATE_SIZE, hidden1=HIDDEN_LAYER_1, hidden2=HIDDEN_LAYER_2, output_size=ACTION_SIZE):
        """
        Deep Q-Network for playing Snake.
        Architecture: Linear(11) -> ReLU -> Linear(128) -> ReLU -> Linear(3)
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass through the network."""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save(self, path):
        """Save model weights to the specified path."""
        # Ensure directory exists before saving
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, input_size=STATE_SIZE, hidden1=HIDDEN_LAYER_1, hidden2=HIDDEN_LAYER_2, output_size=ACTION_SIZE):
        """Load model weights (class method) from the specified path."""
        model = cls(input_size, hidden1, hidden2, output_size)
        
        if os.path.exists(path):
            # map_location handles loading a GPU model on a CPU machine
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            model.eval()
        else:
            print(f"Warning: Model file not found at {path}. Returning randomly initialized model.")
            
        return model
