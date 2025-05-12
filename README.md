# Atari Pixels Project

This project explores deep reinforcement learning for Atari Breakout using PyTorch, with a focus on modern exploration strategies and robust data collection for downstream tasks.

## Part 1: DQN Training & Exploration
- Implements a Deep Q-Network (DQN) agent for Atari Breakout.
- Supports two exploration strategies:
  - **Temperature-based (Boltzmann) exploration** with Prioritized Experience Replay (PER).
  - **Random Network Distillation (RND)** for intrinsic motivation and improved exploration.
- Modular, efficient, and test-driven codebase.
- Current progress: Agent achieves a score up to 20.
- Next steps: Train for longer, increase max steps per episode to 10,000, and target scores of 200+.

See [part1.md](part1.md) for full details, implementation, and next steps.

---

## Getting Started

### 1. Install Dependencies and Atari ROMs
```bash
pip install -r requirements.txt
python -m AutoROM --accept-license
```

### 2. Train the DQN Agent
- **Default (temperature-based exploration):**
  ```bash
  python train_dqn.py
  ```
- **RND-based exploration:**
  ```bash
  python train_dqn.py --exploration_mode rnd
  ```
- Additional arguments (see `python train_dqn.py --help`) allow you to control episodes, buffer size, and more.

### 3. Generate Gameplay Videos
- To record videos of a trained (or random) agent:
  ```bash
  python record_videos.py --checkpoint_path <path_to_checkpoint> --output_dir videos/
  ```
- Replace `<path_to_checkpoint>` with the path to your saved model checkpoint (see `checkpoints/`).
- Videos will be saved as MP4 files in the specified output directory.

---

## Upcoming Parts
- [part2.md](part2.md) _(pending)_
- [part3.md](part3.md) _(pending)_
- [part4.md](part4.md) _(pending)_
- [part5.md](part5.md) _(pending)_

Each part will build on the data and models from Part 1, covering latent action learning, action mapping, and advanced evaluation. 