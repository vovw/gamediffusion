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
    """Experience replay buffer for DQN.
    
    Stores transitions (state, action, reward, next_state, done) and allows
    random sampling for training. This breaks correlation between consecutive
    samples and improves training stability.
    """
    def __init__(self, capacity: int):
        """Initialize the buffer with a given capacity."""
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity) # FIFO buffer

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
    def __init__(self, n_actions: int, state_shape):
        """Initialize DQN agent with networks and replay buffer."""
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.policy_net = DQNCNN(state_shape, n_actions)
        self.target_net = DQNCNN(state_shape, n_actions)
        self.replay_buffer = ReplayBuffer(capacity=1000000)
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

    def select_action(self, state, epsilon: float, use_softmax: bool = False, temperature: float = 1.0) -> int:
        """Select action using epsilon-greedy policy with either argmax or softmax selection.
        
        Args:
            state: Current state (8 stacked frames)
            epsilon: Exploration probability
            use_softmax: If True, use softmax selection. If False, use argmax
            temperature: Temperature parameter for softmax (higher = more uniform)
            
        The Q-values are normalized before selection to prevent
        any action from dominating due to Q-value scale issues.
        """
        if np.random.rand() < epsilon:
            # Exploration: random action
            return np.random.randint(0, self.n_actions)
        else:
            # Exploitation: greedy action
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
            
            # Set model to eval mode for inference
            self.policy_net.eval()
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                # Normalize Q-values to prevent any action from dominating
                q_values = q_values - q_values.mean(dim=1, keepdim=True)
                q_values = q_values / (q_values.std(dim=1, keepdim=True) + 1e-8)
                
                if use_softmax:
                    # Softmax selection with temperature
                    probabilities = torch.softmax(q_values / temperature, dim=1)
                    action = int(torch.multinomial(probabilities, 1).item())
                else:
                    # Standard argmax selection
                    action = int(torch.argmax(q_values, dim=1).item())
            return action

    def optimize_model(self):
        """Update policy network using double DQN algorithm.
        
        Key Steps:
        1. Sample batch from replay buffer
        2. Compute current Q-values for taken actions
        3. Compute next Q-values using double DQN:
           - Use policy net to select actions
           - Use target net to evaluate those actions
           This prevents overestimation bias
        4. Compute Huber loss and update policy network
        
        Q-value Overestimation Example:
        Consider a state with true Q-values [1.0, 1.0, 1.0, 1.0]
        Policy net estimates: [0.8, 1.2, 0.9, 1.1]
        Target net estimates: [1.1, 0.9, 1.2, 0.8]
        
        Regular DQN would use max(target_net) = 1.2 (overestimated)
        Double DQN:
        1. Policy net selects action (index of 1.2 from its estimates)
        2. Uses target net's value for that action (0.9)
        Result is closer to true Q-value (1.0)
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = self.replay_buffer.sample(self.batch_size)
        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        # Convert to tensors
        states = torch.from_numpy(np.stack(states)).to(self.device).float() / 255.0  # (B, 8, 84, 84)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)  # (B, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)  # (B, 1)
        next_states = torch.from_numpy(np.stack(next_states)).to(self.device).float() / 255.0
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)  # (B, 1)
        
        self.policy_net.train()
        # Mixed precision if CUDA
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            # Get Q-values for taken actions
            # Shape explanation:
            # policy_net(states) -> (B, n_actions)
            # gather(1, actions) -> (B, 1) selects Q-values of taken actions
            q_values = self.policy_net(states).gather(1, actions)  # (B, 1)
            
            with torch.no_grad():
                # Double DQN: Use policy net to select actions, target net to evaluate them
                # This prevents overestimation by decomposing max operation into action selection and evaluation
                next_actions = self.policy_net(next_states).max(1, keepdim=True)[1]  # Select actions using policy net
                next_q_values = self.target_net(next_states).gather(1, next_actions)  # Evaluate using target net
                target_q = rewards + self.gamma * next_q_values * (1.0 - dones)
            
            # Huber loss combines L2 for small errors and L1 for large errors
            # More robust to outliers than MSE loss
            loss = nn.HuberLoss()(q_values, target_q)
        
        self.optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            # Clip gradients to prevent explosive updates
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