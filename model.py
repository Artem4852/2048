import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(21, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
    def save_model(self, filename):
        with open(filename, "wb") as f:
            torch.save(self.state_dict(), f)

    def load_model(self, filename):
        with open(filename, "rb") as f:
            self.load_state_dict(torch.load(f))
    
class DinoGamer:
    def __init__(self):
        self.model = DQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.criterion = nn.MSELoss()
        self.epsilon = 0.9
        self.gamma = 0.9

    def select_action(self, state):
        self.epsilon *= 0.9999
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 3), True
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float).flatten()
                state = state.view(1, -1)
                q_values = self.model(state)
                return q_values.max(1)[1].item(), False
            
    def train(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
        action = torch.tensor([action], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float)

        current_q = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q = self.model(next_state).max(1)[0].detach()
        expected_q = reward + (self.gamma * next_q * (1 - int(done)))

        loss = self.criterion(current_q, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()