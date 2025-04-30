import os
import numpy as np
import torch
from tqdm import trange
from atari_env import AtariBreakoutEnv
from dqn_agent import DQNAgent
import cv2
import json
import random
import argparse

# Config
config = {
    'env_name': 'Breakout',
    'n_actions': 4,
    'state_shape': (4, 84, 84),
    'max_episodes': 10000,
    'max_steps': 10000,
    'epsilon_start': 1.0,
    'epsilon_end': 0.1,
    'epsilon_decay': 200000,
    'target_update_freq': 500,
    'checkpoint_dir': 'checkpoints',
    'data_dir': 'data/raw_gameplay',
    'actions_dir': 'data/actions',
    'save_freq': 100,
    'min_buffer': 10000,
    'seed': 42,
    'skill_thresholds': [0, 50, 150, 250],
    'episodes_per_skill': 50,
}

# Set seeds and deterministic flags
np.random.seed(config['seed'])
torch.manual_seed(config['seed'])
random.seed(config['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def save_episode_data(frames, actions, rewards, skill_level, episode_idx, config):
    """Save frames as PNGs and actions/rewards as JSON."""
    skill_dir = os.path.join(config['data_dir'], f'skill_level_{skill_level}', f'episode_{episode_idx:03d}')
    os.makedirs(skill_dir, exist_ok=True)
    actions_json = []
    for i, (frame, action, reward) in enumerate(zip(frames, actions, rewards)):
        frame_path = os.path.join(skill_dir, f'frame_{i:05d}.png')
        cv2.imwrite(frame_path, frame)
        actions_json.append({'step': i, 'action': int(action), 'reward': float(reward)})
    # Save actions JSON
    actions_file = os.path.join(config['actions_dir'], f'skill_level_{skill_level}_actions.json')
    os.makedirs(config['actions_dir'], exist_ok=True)
    if os.path.exists(actions_file):
        with open(actions_file, 'r') as f:
            all_actions = json.load(f)
    else:
        all_actions = []
    all_actions.append({'episode': episode_idx, 'actions': actions_json})
    with open(actions_file, 'w') as f:
        json.dump(all_actions, f, indent=2)


def evaluate_agent(agent, env, n_episodes=10):
    """Evaluate agent and return average reward."""
    rewards = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        state_stack = np.stack([obs] * 4, axis=0)
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state_stack, epsilon=0.0)
            next_obs, reward, terminated, truncated, info = env.step(action)
            state_stack = np.roll(state_stack, shift=-1, axis=0)
            state_stack[-1] = next_obs
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return np.mean(rewards)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_episodes', type=int, default=None)
    parser.add_argument('--min_buffer', type=int, default=None)
    parser.add_argument('--save_freq', type=int, default=None)
    parser.add_argument('--no_save', action='store_true', help='Do not save frames or actions during training')
    args = parser.parse_args()
    # Override config if args provided
    if args.max_episodes is not None:
        config['max_episodes'] = args.max_episodes
    if args.min_buffer is not None:
        config['min_buffer'] = args.min_buffer
    if args.save_freq is not None:
        config['save_freq'] = args.save_freq
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['data_dir'], exist_ok=True)
    os.makedirs(config['actions_dir'], exist_ok=True)
    env = AtariBreakoutEnv()
    agent = DQNAgent(n_actions=config['n_actions'], state_shape=config['state_shape'])
    epsilon = config['epsilon_start']
    epsilon_decay = (config['epsilon_start'] - config['epsilon_end']) / config['epsilon_decay']
    total_steps = 0
    episode_rewards = []
    skill_level = 0
    skill_episodes = 0
    skill_thresholds = config['skill_thresholds']
    skill_counts = [0] * len(skill_thresholds)
    pbar = trange(config['max_episodes'], desc='Training')
    episode_losses = []
    running_avg_rewards = []
    
    # Fill replay buffer with random actions first
    print("Pre-filling replay buffer with random experiences...")
    obs, info = env.reset()
    state_stack = np.stack([obs] * 4, axis=0)
    for step in range(max(5000, config['min_buffer'])):
        action = np.random.randint(0, config['n_actions'])
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state_stack = np.roll(state_stack, shift=-1, axis=0)
        next_state_stack[-1] = next_obs
        agent.replay_buffer.push(state_stack, action, reward, next_state_stack, terminated or truncated)
        state_stack = next_state_stack
        if terminated or truncated:
            obs, info = env.reset()
            state_stack = np.stack([obs] * 4, axis=0)
        if step % 1000 == 0:
            print(f"  {step}/{max(5000, config['min_buffer'])} experiences collected")
    
    for episode in pbar:
        obs, info = env.reset()
        state_stack = np.stack([obs] * 4, axis=0)
        frames, actions, rewards = [obs], [], []
        done = False
        total_reward = 0
        losses = []
        
        # More optimization steps per episode
        for step in range(config['max_steps']):
            action = agent.select_action(state_stack, epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state_stack = np.roll(state_stack, shift=-1, axis=0)
            next_state_stack[-1] = next_obs
            agent.replay_buffer.push(state_stack, action, reward, next_state_stack, terminated or truncated)
            
            # Do multiple optimization steps per environment step
            for _ in range(10):  # Increased optimization frequency
                loss = agent.optimize_model()
                if loss is not None:
                    losses.append(loss)
            
            state_stack = next_state_stack
            frames.append(next_obs)
            actions.append(action)
            rewards.append(reward)
            total_reward += reward
            total_steps += 1
            
            # Epsilon decay
            if epsilon > config['epsilon_end']:
                epsilon -= epsilon_decay
                epsilon = max(config['epsilon_end'], epsilon)  # Ensure it doesn't go below epsilon_end
            
            # Target network update
            if total_steps % config['target_update_freq'] == 0:
                agent.update_target_network()
                
            if terminated or truncated:
                break
                
        # Process episode results
        avg_loss = np.mean(losses) if losses else 0.0
        episode_rewards.append(total_reward)
        episode_losses.append(avg_loss)
        
        # Calculate running average
        running_avg = np.mean(episode_rewards[-min(100, len(episode_rewards)):])
        running_avg_rewards.append(running_avg)
        
        # Logging
        pbar.set_postfix({
            'ep_reward': total_reward, 
            'avg_reward': f"{running_avg:.1f}",
            'epsilon': f"{epsilon:.3f}", 
            'skill': skill_level, 
            'loss': f"{avg_loss:.4f}"
        })
        
        # Print running averages every 10 episodes
        if episode > 0 and episode % 10 == 0:
            last_10_loss = episode_losses[-10:]
            last_10_reward = episode_rewards[-10:]
            print(f"[Stats] Episodes {episode-9}-{episode} | Avg Reward: {np.mean(last_10_reward):.2f} | Avg Loss: {np.mean(last_10_loss):.4f} | Running Avg: {running_avg:.2f}")
        
        # Save episode data if in skill collection phase
        if (not args.no_save and
            skill_level < len(skill_thresholds) and
            running_avg >= skill_thresholds[skill_level]):
            if skill_counts[skill_level] < config['episodes_per_skill']:
                save_episode_data(frames, actions, rewards, skill_level, skill_counts[skill_level]+1, config)
                skill_counts[skill_level] += 1
            if skill_counts[skill_level] >= config['episodes_per_skill']:
                print(f"Skill level {skill_level} ({skill_thresholds[skill_level]}+) complete.")
                # Save checkpoint
                torch.save(agent.policy_net.state_dict(), os.path.join(config['checkpoint_dir'], f'dqn_skill_{skill_level}.pth'))
                skill_level += 1
        # Periodic checkpoint (overwrite previous)
        if episode % config['save_freq'] == 0:
            torch.save(agent.policy_net.state_dict(), os.path.join(config['checkpoint_dir'], 'dqn_latest.pth'))
        # Early stop if all skill levels collected
        if skill_level >= len(skill_thresholds):
            print("All skill levels collected. Training complete.")
            break
    env.close()
    # Save final checkpoint (overwrite previous)
    torch.save(agent.policy_net.state_dict(), os.path.join(config['checkpoint_dir'], 'dqn_latest.pth'))
    print("Training finished.")

if __name__ == '__main__':
    main() 