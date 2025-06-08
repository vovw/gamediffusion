import os
import numpy as np
import torch
from tqdm import trange
from pong_env import AtariPongEnv
from dqn_agent import DQNAgent
import cv2
import json
import ale_py
import random
import argparse
from collections import deque

# Config
config = {
    'env_name': 'Pong',
    'n_actions': 6,  # Pong has 6 actions
    'state_shape': (8, 84, 84),
    'max_episodes': 1000,
    'max_steps': 1000,
    'checkpoint_dir': 'checkpoints',
    'data_dir': 'data/pong_trajectories',
    'save_freq': 1,  # Save every episode
    'seed': 42,
    'epsilon': 0.1,  # Epsilon for epsilon-greedy exploration
}

# config = {
#     'env_name': 'Pong',
#     'n_actions': 6,  # Pong has 6 actions
#     'state_shape': (8, 84, 84),
#     'max_episodes': 10,
#     'max_steps': 10,
#     'checkpoint_dir': 'checkpoints',
#     'data_dir': 'data/pong_trajectories',
#     'save_freq': 1,  # Save every episode
#     'seed': 42,
#     'epsilon': 0.1,  # Epsilon for epsilon-greedy exploration
# }

def save_trajectory(episode, frames, actions, rewards, config):
    """Save a trajectory to disk."""
    episode_dir = os.path.join(config['data_dir'], f'episode_{episode:05d}')
    os.makedirs(episode_dir, exist_ok=True)
    
    # Save frames as PNG files
    for i, frame in enumerate(frames):
        frame_path = os.path.join(episode_dir, f'frame_{i:05d}.png')
        cv2.imwrite(frame_path, frame)
    
    # Save actions and rewards
    metadata = {
        'actions': actions,
        'rewards': rewards,
        'total_reward': sum(rewards),
        'length': len(frames)
    }
    with open(os.path.join(episode_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

def main():
    # Set seeds for reproducibility
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    
    # Create directories
    os.makedirs(config['data_dir'], exist_ok=True)
    
    # Initialize environment and agent
    env = AtariPongEnv(return_rgb=True)  # Get RGB frames for saving
    agent = DQNAgent(
        n_actions=config['n_actions'],
        state_shape=config['state_shape'],
        prioritized=False,
    )
    
    # Load pretrained model if exists
    model_path = os.path.join(config['checkpoint_dir'], 'pong_dqn.pt')
    if os.path.exists(model_path):
        agent.policy_net.load_state_dict(torch.load(model_path))
        print(f"Loaded pretrained model from {model_path}")
    
    pbar = trange(config['max_episodes'], desc='Collecting trajectories')
    for episode in pbar:
        obs, info = env.reset()
        state_stack = np.stack([env.preprocess_observation(obs)] * 8, axis=0)
        
        # Initialize trajectory storage
        frames = [obs]  # Store RGB frames
        actions = []
        rewards = []
        
        for step in range(config['max_steps']):
            # Select action using epsilon-greedy
            action = agent.select_action(state_stack, mode='epsilon', epsilon=config['epsilon'])
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state_stack = np.roll(state_stack, shift=-1, axis=0)
            next_state_stack[-1] = env.preprocess_observation(next_obs)
            
            # Store trajectory data
            frames.append(next_obs)
            actions.append(int(action))
            rewards.append(float(reward))
            
            # Update state
            state_stack = next_state_stack
            
            if terminated or truncated:
                break
        
        # Save trajectory
        if episode % config['save_freq'] == 0:
            save_trajectory(episode, frames, actions, rewards, config)
        
        # Update progress bar
        pbar.set_postfix({
            'length': len(frames),
            'reward': sum(rewards),
        })
    
    env.close()

if __name__ == '__main__':
    main() 
