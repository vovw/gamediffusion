"""Deep Q-Network (DQN) implementation with several improvements.

Key Features:
- Double DQN to prevent Q-value overestimation
- Huber Loss for robustness to outliers
- Q-value normalization
- Gradient clipping
- Experience replay for sample efficiency
- Target network for training stability
- Mixed precision training on CUDA
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class ReplayBuffer:
    """Experience replay buffer for DQN with support for intrinsic and extrinsic rewards."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity) # FIFO buffer

    def push(self, state, action, extrinsic_reward, intrinsic_reward, next_state, done):
        """Store a transition in the buffer with both extrinsic and intrinsic rewards."""
        self.buffer.append((state, action, extrinsic_reward, intrinsic_reward, next_state, done))

    def sample(self, batch_size: int, mode: str = 'exploration', alpha: float = 0.5):
        """
        Sample a batch of transitions.
        mode: 'exploration' returns combined reward, 'exploitation' returns extrinsic only.
        alpha: weight for intrinsic vs extrinsic reward (used in exploration mode)
        """
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, extrinsic_rewards, intrinsic_rewards, next_states, dones = zip(*batch)
        if mode == 'exploration':
            # Combined reward
            rewards = [(1 - alpha) * er + alpha * ir for er, ir in zip(extrinsic_rewards, intrinsic_rewards)]
        else:
            # Exploitation: extrinsic only
            rewards = extrinsic_rewards
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class SumTree:
    """SumTree data structure for efficient sampling and priority updates."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.zeros(capacity, dtype=object)
        self.size = 0
        self.write = 0

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s):
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return idx, self.tree[idx], self.data[data_idx]

    def total(self):
        return self.tree[0]

    def __len__(self):
        return self.size

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer using a sum-tree."""
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_initial = beta
        self.epsilon = 1e-6
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def push(self, state, action, extrinsic_reward, intrinsic_reward, next_state, done, priority=None):
        if priority is None:
            priority = self.max_priority
        p = (abs(priority) + self.epsilon) ** self.alpha
        transition = (state, action, extrinsic_reward, intrinsic_reward, next_state, done)
        self.tree.add(p, transition)
        self.max_priority = max(self.max_priority, p)

    def sample(self, batch_size, mode='exploration', alpha=0.5):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)
        states, actions, extrinsic_rewards, intrinsic_rewards, next_states, dones = zip(*batch)
        if mode == 'exploration':
            rewards = [(1 - alpha) * er + alpha * ir for er, ir in zip(extrinsic_rewards, intrinsic_rewards)]
        else:
            rewards = extrinsic_rewards
        priorities = np.array(priorities, dtype=np.float32)
        probs = priorities / (self.tree.total() + 1e-8)
        weights = (len(self.tree) * probs) ** (-self.beta)
        weights /= weights.max() + 1e-8
        weights = weights.astype(np.float32)
        return states, actions, rewards, next_states, dones, weights, idxs

    def update_priorities(self, idxs, priorities):
        for idx, priority in zip(idxs, priorities):
            p = (abs(priority) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def anneal_beta(self, new_beta):
        self.beta = new_beta

    def __len__(self):
        return len(self.tree)

class DQNCNN(nn.Module):
    """CNN architecture for DQN, based on the original DQN paper.
    
    Architecture:
    1. Input: (batch_size, 8, 84, 84) - 8 stacked frames
    2. Conv layers process spatial features
    3. FC layers compute Q-values for each action
    """
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
        """Initialize weights using Kaiming initialization."""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass normalizes inputs and computes Q-values."""
        x = x.float() / 255.0  # Normalize pixel values
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class DQNAgent:
    """Deep Q-Network agent implementing several DQN improvements.
    
    Key Features:
    1. Double DQN: Uses two networks to prevent Q-value overestimation
       - Policy network: Selects actions
       - Target network: Evaluates actions
       This prevents the positive bias in Q-value estimation that occurs
       when the same network both selects and evaluates actions.
       
    2. Experience Replay: Stores and randomly samples transitions
       - Breaks correlation between consecutive samples
       - Allows multiple updates from each experience
       - Improves sample efficiency
       
    3. Q-value normalization: Prevents any action from dominating
       - Subtracts mean and divides by std
       - Helps maintain reasonable Q-value ranges
       
    4. Huber Loss: More robust to outliers than MSE
       - Combines L2 loss for small errors
       - L1 loss for large errors
       - Helps prevent unstable updates
       
    5. Gradient clipping: Prevents explosive gradients
       - Clips gradient norm to 1.0
       - Maintains stable updates
    """
    def __init__(self, n_actions: int, state_shape, replay_buffer=None):
        """Initialize DQN agent with networks and replay buffer."""
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.policy_net = DQNCNN(state_shape, n_actions)
        self.target_net = DQNCNN(state_shape, n_actions)
        self.replay_buffer = replay_buffer if replay_buffer is not None else ReplayBuffer(capacity=1000000)
        self.gamma = 0.99  # Discount factor
        self.batch_size = 128
        self.learning_rate = 3e-4
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

    def select_action(self, state, mode: str = 'greedy', temperature: float = 1.0) -> int:
        """Select action using greedy (argmax) or softmax policy."""
        state_tensor = torch.from_numpy(state).unsqueeze(0)  # (1, 8, 84, 84)
        if torch.cuda.is_available():
            state_tensor = state_tensor.cuda()
            self.policy_net.cuda()
        elif torch.backends.mps.is_available():
            state_tensor = state_tensor.to('mps')
            self.policy_net.to('mps')
        else:
            state_tensor = state_tensor.cpu()
            self.policy_net.cpu()
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            q_values = q_values - q_values.mean(dim=1, keepdim=True)
            q_values = q_values / (q_values.std(dim=1, keepdim=True) + 1e-8)
            if mode == 'softmax':
                # Softmax action selection (Boltzmann exploration)
                logits = q_values / max(temperature, 1e-6)
                probs = torch.softmax(logits, dim=1)
                action = int(torch.multinomial(probs, num_samples=1).item())
            else:
                action = int(torch.argmax(q_values, dim=1).item())
        return action

    def optimize_model(self, mode: str = 'exploration', alpha: float = 0.5):
        """Update policy network using double DQN algorithm. Mode controls reward type."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = self.replay_buffer.sample(self.batch_size, mode=mode, alpha=alpha)
        states, actions, rewards, next_states, dones = batch
        # Convert to tensors
        states = torch.from_numpy(np.stack(states)).to(self.device).float() / 255.0  # (B, 8, 84, 84)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)  # (B, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)  # (B, 1)
        next_states = torch.from_numpy(np.stack(next_states)).to(self.device).float() / 255.0
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)  # (B, 1)
        self.policy_net.train()
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            q_values = self.policy_net(states).gather(1, actions)  # (B, 1)
            with torch.no_grad():
                next_actions = self.policy_net(next_states).max(1, keepdim=True)[1]
                next_q_values = self.target_net(next_states).gather(1, next_actions)
                target_q = rewards + self.gamma * next_q_values * (1.0 - dones)
            loss = nn.HuberLoss()(q_values, target_q)
        self.optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            scaler.step(self.optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        """Copy policy network weights to target network.
        
        This periodic update helps maintain stable Q-learning targets.
        Too frequent updates can lead to unstable training,
        while too infrequent updates can lead to stale targets.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict()) 