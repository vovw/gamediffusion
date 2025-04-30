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

### Agent Training
1. Implement a DQN (Deep Q-Network) agent using PyTorch
   - Use a simple CNN architecture (3 convolutional layers + 2 fully connected)
   - Experience replay buffer with capacity of 100,000 transitions
   - Target network update frequency of 1000 steps
   - Epsilon-greedy exploration starting at 1.0 and decaying to 0.1
   - Learning rate of 0.0001 with Adam optimizer

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