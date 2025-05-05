# Part 1: Recording Agent Gameplay Videos

## Overview
In this first part, we'll train an RL agent to play Atari Breakout and record its gameplay at different proficiency levels. This will give us the raw data needed for training our world model in later parts.

## Related Components
- [part2.md](part2.md): Uses these videos to learn latent actions
- [part4.md](part4.md): Uses action labels recorded here to map to latent actions

## Implementation Details

### Environment Setup
1. Set up the Atari Breakout environment using OpenAI Gym/Gymnasium
2. Configure the environment for:
   - Lower resolution (84x84 pixels) for efficiency
   - Grayscale frames to reduce dimensionality
   - Frame skipping (4 frames) to speed up training
   - End-of-episode detection based on lives lost
   - 8-frame stacking for better temporal information

### Agent Training
1. Implement a DQN (Deep Q-Network) agent using PyTorch
   - Use a simple CNN architecture (3 convolutional layers + 2 fully connected)
   - 8-frame stacking for better temporal information
   - Experience replay buffer with capacity of 100,000 transitions
   - Target network update frequency of 1000 steps
   - Epsilon-greedy exploration starting at 1.0 and decaying to 0.1
   - Learning rate of 0.0001 with Adam optimizer
   - Batch size of 128 for training

2. Train the agent in stages, saving checkpoints at:
   - Random play (untrained)
   - Early learning (reaching ~50 points average)
   - Intermediate skill (reaching ~150 points average)
   - Skilled play (reaching ~250+ points average)

### Data Collection
1. For each skill level, record:
   - 50 complete gameplay episodes
   - Store sequential frames as PNG images at 84x84 resolution
   - Save corresponding actions (LEFT, RIGHT, NOTHING) in a JSON file
   - Include reward and episode information for reference

2. Organize data structure as:
data/
  raw_gameplay/
    skill_level_0/
      episode_001/
        frame_00000.png
        frame_00001.png
        ...
      episode_002/
        ...
    skill_level_1/
      ...
    ...
  actions/
    skill_level_0_actions.json
    skill_level_1_actions.json
    ...

3. Create validation splits (10% of episodes) for later evaluation

### Tools and Libraries
- PyTorch for agent implementation
- Gymnasium (OpenAI Gym successor) for Atari environment
- OpenCV for image processing and saving
- Weights & Biases (optional) for experiment tracking

### Success Criteria
- At least 200 total episodes recorded across all skill levels
- Highest skill level should achieve consistent scores >200 points
- Complete and accurate action logs corresponding to all frames

## Progress Overview (Implementation Status)

### Implemented so far
- **AtariBreakoutEnv**: Custom wrapper for Gymnasium's Breakout environment with:
  - 84x84 grayscale preprocessing
  - Frame skipping (4 frames)
  - Minimal action space
- **RandomAgent**: Agent that selects random actions, records frames, actions, and rewards, and saves them to disk as PNGs and JSON.
- **DQNAgent**: Deep Q-Network agent implemented in PyTorch, including:
  - 3-layer CNN + 2 fully connected layers for Q-value prediction
  - Input of 8 stacked frames (instead of traditional 4) for better temporal information
  - Experience replay buffer (capacity 100,000)
  - Double DQN implementation to reduce Q-value overestimation
  - HuberLoss for more robust learning
  - Target network and synchronization logic (every 500 steps)
  - Epsilon-greedy action selection with improved exploration (1.0 to 0.1)
  - Optimized hyperparameters: learning rate 2.5e-4, batch size 128
  - Proper gradient clipping (1.0) for stable training
  - Modular, test-driven design
  - DQN training step (optimize_model) and full training loop
  - Proper device selection (CUDA/MPS/CPU) and torch.compile for speed
- **Training Script**: `train_dqn.py` implements the full DQN training loop, including:
  - Pre-filling of replay buffer with random experiences
  - Multiple optimization steps per environment step (10x)
  - Epsilon decay with faster schedule (200k steps)
  - Target network updates, and checkpointing
  - Running average reward tracking for skill level determination
  - Command-line overrides for max_episodes, min_buffer, save_freq, and no_save (for flexible runs)
  - Loss logging: average loss per episode, running stats for last 10 episodes, and reward tracking
  - Option to skip saving frames/actions for dry runs (`--no_save`)
  - Saving model checkpoints at skill milestones (random, ~50, ~150, ~250 points average)
  - Recording gameplay data (frames, actions, rewards) for each skill level as PNGs and JSON
  - Progress bar and improved logging with formatted metrics
- **Test Suite**: Comprehensive pytest-based tests for environment, agents (random and DQN), replay buffer, target network sync, epsilon-greedy policy, and DQN training/optimization.
- **ROM Installation**: Requirements and instructions updated to ensure ROMs are installed automatically.
- **DQN Training Optimizations**: Implemented various techniques to accelerate learning:
  - Double Q-learning to reduce overestimation bias
  - Increased learning rate and batch size
  - More frequent optimization steps
  - Pre-filling of replay buffer
  - Better exploration strategy
- **Gameplay Video Recording**: Implementation of `record_videos.py` to:
  - Generate videos of gameplay episodes for random and trained agents
  - Save videos as MP4 files at 30fps
  - Include frame number, action taken, and cumulative reward as overlay text
  - Support for different skill level checkpoints
- **Reward Shaping**: Implementation in `AtariBreakoutEnv` including:
  - Living penalty (-0.001 per time step)
  - Life loss penalty (-1.0 for losing a life)
  - Tracking lives between steps to detect life loss

### Remaining tasks
- Train DQN agent to different skill levels and save checkpoints (full run with data saving enabled)
- Record and organize gameplay data for each skill level (as described above)
- Create validation splits for evaluation
- Integrate experiment tracking (e.g., with WandB)

## Planned Improvements Status

1. ✅ **Visualize Current Agent Gameplay**
   - Videos of random and trained agents are implemented in `record_videos.py`
   - Proper overlay with frame number, action, and reward
   - MP4 output at 30fps with correct resolution (160x210 plus text area)

2. ❌ **Upgrade Environment to Full RGB and Higher Resolution**
   - **Status: Not implemented**
   - Need to modify environment wrapper to use RGB instead of grayscale
   - Need to use native Atari resolution (160x210) or higher
   - Need to reduce frame skip from 4 to 2
   - Need to update network architecture to handle larger input size
   - Keep 8-frame stacking for temporal information (currently implemented)

3. ❌ **Implement Prioritized Experience Replay**
   - **Status: Not implemented**
   - Need to implement sum-tree data structure
   - Need to modify sampling to use priorities
   - Need to implement importance sampling weights

4. ✅ **Implement Reward Shaping**
   - Living penalty of -0.001 per time step implemented
   - Life loss penalty of -1.0 implemented
   - Proper tracking of lives between steps implemented

## Environment Setup & Troubleshooting

- After installing dependencies from `requirements.txt`, run:
  
  ```bash
  python -m AutoROM --accept-license
  ```
  This will download and install all required Atari ROMs. If you see errors about missing ROMs, re-run the above command.
- For more details, see the comments in `requirements.txt`.

---