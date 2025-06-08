import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, extrinsic_reward, intrinsic_reward, next_state, done):
        self.buffer.append((state, action, extrinsic_reward, intrinsic_reward, next_state, done))
    def sample(self, batch_size: int, mode: str = 'exploration', alpha: float = 0.5):
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, extrinsic_rewards, intrinsic_rewards, next_states, dones = zip(*batch)
        if mode == 'exploration':
            rewards = [(1 - alpha) * er + alpha * ir for er, ir in zip(extrinsic_rewards, intrinsic_rewards)]
        else:
            rewards = extrinsic_rewards
        return states, actions, rewards, next_states, dones
    def __len__(self):
        return len(self.buffer)

class DQNCNN(nn.Module):
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
        x = x.float() / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class DQNAgent:
    def __init__(self, n_actions: int, state_shape, replay_buffer=None, prioritized=False, per_alpha=0.6, per_beta=0.4):
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.policy_net = DQNCNN(state_shape, n_actions)
        self.target_net = DQNCNN(state_shape, n_actions)
        self.prioritized = prioritized
        if replay_buffer is not None:
            self.replay_buffer = replay_buffer
        else:
            self.replay_buffer = ReplayBuffer(capacity=1000000)
        self.gamma = 0.99
        self.batch_size = 32
        self.learning_rate = 2.5e-4
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        if hasattr(torch, 'compile') and self.device.type == 'cuda':
            self.policy_net = torch.compile(self.policy_net)
    def select_action(self, state, mode: str = 'greedy', temperature: float = 1.0, epsilon: float = 0.0) -> int:
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            if mode == 'softmax':
                logits = q_values / max(temperature, 1e-6)
                probs = torch.softmax(logits, dim=1)
                action = int(torch.multinomial(probs, num_samples=1).item())
            elif mode == 'epsilon':
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, self.n_actions)
                else:
                    action = int(torch.argmax(q_values, dim=1).item())
            elif mode == 'greedy':
                if np.random.rand() < 0.05:
                    action = np.random.randint(0, self.n_actions)
                else:
                    action = int(torch.argmax(q_values, dim=1).item())
            else:
                action = int(torch.argmax(q_values, dim=1).item())
        return action
    def optimize_model(self, mode: str = 'exploitation', alpha: float = 0.5):
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = self.replay_buffer.sample(self.batch_size, mode=mode, alpha=alpha)
        states, actions, rewards, next_states, dones = batch
        states = torch.from_numpy(np.stack(states)).to(self.device).float()
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.from_numpy(np.stack(next_states)).to(self.device).float()
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        self.policy_net.train()
        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1, keepdim=True)[1]
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q_values * (1.0 - dones)
        td_errors = q_values - target_q
        loss = nn.HuberLoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        if any(torch.isnan(p.grad).any() for p in self.policy_net.parameters() if p.grad is not None):
            print("WARNING: NaN detected in gradients! Skipping update.")
            return None
        self.optimizer.step()
        if any(torch.isnan(p).any() for p in self.policy_net.parameters()):
            print("WARNING: NaN detected in weights! Restoring from target network.")
            self.policy_net.load_state_dict(self.target_net.state_dict())
            return None
        return loss.item(), td_errors.abs().mean().item()
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict()) 