import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class ReplayBuffer:
    """Experience replay buffer for DQN."""
    def __init__(self, capacity: int):
        """Initialize the buffer with a given capacity."""
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a transition in the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Sample a batch of transitions from the buffer."""
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[idx] for idx in indices]
        return batch

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)

class DQNCNN(nn.Module):
    """CNN for processing Atari frames in DQN."""
    def __init__(self, input_shape, n_actions):
        super().__init__()
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Compute conv output size
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.float() / 255.0  # Normalize pixel values
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class DQNAgent:
    """Deep Q-Network agent for Atari."""
    def __init__(self, n_actions: int, state_shape):
        """Initialize DQN agent with networks and replay buffer."""
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.policy_net = DQNCNN(state_shape, n_actions)
        self.target_net = DQNCNN(state_shape, n_actions)
        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.gamma = 0.99
        self.batch_size = 32
        self.learning_rate = 1e-4
        # Device selection
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        # Compile for speed if available and on CUDA only
        if hasattr(torch, 'compile') and self.device.type == 'cuda':
            self.policy_net = torch.compile(self.policy_net)

    def select_action(self, state, epsilon: float) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.rand() < epsilon:
            # Exploration: random action
            return np.random.randint(0, self.n_actions)
        else:
            # Exploitation: greedy action
            state_tensor = torch.from_numpy(state).unsqueeze(0)  # (1, 4, 84, 84)
            if torch.cuda.is_available():
                state_tensor = state_tensor.cuda()
                self.policy_net.cuda()
            elif torch.backends.mps.is_available():
                state_tensor = state_tensor.to('mps')
                self.policy_net.to('mps')
            else:
                state_tensor = state_tensor.cpu()
                self.policy_net.cpu()
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action = int(torch.argmax(q_values, dim=1).item())
            return action

    def optimize_model(self):
        """Sample from replay buffer and optimize policy network."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = self.replay_buffer.sample(self.batch_size)
        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        # Convert to tensors
        states = torch.from_numpy(np.stack(states)).to(self.device).float() / 255.0  # (B, 4, 84, 84)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)  # (B, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)  # (B, 1)
        next_states = torch.from_numpy(np.stack(next_states)).to(self.device).float() / 255.0
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)  # (B, 1)
        self.policy_net.train()
        # Mixed precision if CUDA
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            q_values = self.policy_net(states).gather(1, actions)  # (B, 1)
            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
                target_q = rewards + self.gamma * next_q_values * (1.0 - dones)
            loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
            scaler.step(self.optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
            self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        """Copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict()) 