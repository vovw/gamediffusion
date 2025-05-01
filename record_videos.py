"""Record gameplay videos of trained and random agents playing Breakout.

This script records videos of both a random agent and a trained DQN agent playing Atari Breakout.
Videos include an overlay showing frame number, action taken, and current score.

Usage:
    # Record using latest checkpoint (default)
    python record_videos.py

    # Record using specific skill level checkpoints
    python record_videos.py --skill_level 0  # Random play
    python record_videos.py --skill_level 1  # ~50 points skill
    python record_videos.py --skill_level 2  # ~150 points skill
    python record_videos.py --skill_level 3  # ~250 points skill

    # Record more episodes
    python record_videos.py --skill_level 2 --num_episodes 10

    # Save to different directory
    python record_videos.py --skill_level 2 --output_dir videos_skill2

Arguments:
    --num_episodes: Number of episodes to record for each agent (default: 5)
    --output_dir: Directory to save videos (default: 'videos')
    --skill_level: Skill level checkpoint to use (0-3). If not specified, uses latest checkpoint.

Output:
    - Random agent videos: random_agent_episode_X.mp4
    - Trained agent videos: trained_agent_[checkpoint]_episode_X.mp4
    where [checkpoint] is either 'latest' or 'skill_N' depending on --skill_level
"""

import os
import cv2
import numpy as np
import torch
from atari_env import AtariBreakoutEnv
from dqn_agent import DQNAgent
from random_agent import RandomAgent
import gymnasium as gym
import argparse

def add_text_overlay(frame, frame_num, action, cumulative_reward):
    """Add text overlay to frame with game information."""
    # Convert to RGB if grayscale
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    
    # Create a slightly larger frame to accommodate text
    height, width = frame.shape[:2]
    text_height = 30
    frame_with_text = np.zeros((height + text_height, width, 3), dtype=np.uint8)
    frame_with_text[text_height:, :, :] = frame
    
    # Add black background for text
    cv2.rectangle(frame_with_text, (0, 0), (width, text_height), (0, 0, 0), -1)
    
    # Add text
    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    text = f"{frame_num} | {action_names[action]} | {cumulative_reward:.1f}"
    cv2.putText(frame_with_text, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame_with_text

def record_episode(env, agent, video_writer, epsilon=0.0):
    """Record a single episode."""
    obs, info = env.reset()
    # Initialize state stack for DQN
    state_stack = np.stack([obs] * 4, axis=0)
    cumulative_reward = 0
    frame_num = 0
    max_steps = 10000
    
    while True:
        if frame_num > max_steps:
            break
        
        # Get raw frame from environment (before preprocessing)
        raw_frame = env.env.render()  # This gets the 160x210 RGB frame
        
        # Get action from agent
        if isinstance(agent, DQNAgent):
            action = agent.select_action(state_stack, epsilon)
        else:  # RandomAgent
            action = agent.select_action()
        
        # Add overlay and write frame
        frame_with_overlay = add_text_overlay(raw_frame, frame_num, action, cumulative_reward)
        video_writer.write(frame_with_overlay)
        
        # Take step in environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        frame_num += 1
        
        # Update state stack for DQN
        if isinstance(agent, DQNAgent):
            state_stack = np.roll(state_stack, shift=-1, axis=0)
            state_stack[-1] = next_obs
        
        if terminated or truncated:
            # Write final frame
            raw_frame = env.env.render()
            frame_with_overlay = add_text_overlay(raw_frame, frame_num, action, cumulative_reward)
            video_writer.write(frame_with_overlay)
            break
    
    return cumulative_reward

def record_gameplay_videos(num_episodes=5, output_dir='videos', skill_level=None):
    """Record gameplay videos for both random and trained agents.
    
    Args:
        num_episodes: Number of episodes to record for each agent
        output_dir: Directory to save videos
        skill_level: If specified (0-3), load that skill level's checkpoint.
                    If None, use latest checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    env = AtariBreakoutEnv()
    
    # Initialize agents
    random_agent = RandomAgent(n_actions=4)
    dqn_agent = DQNAgent(n_actions=4, state_shape=(4, 84, 84))
    
    # Load trained model if available
    if skill_level is not None:
        model_path = os.path.join('checkpoints', f'dqn_skill_{skill_level}.pth')
        checkpoint_name = f'skill_{skill_level}'
    else:
        model_path = os.path.join('checkpoints', 'dqn_latest.pth')
        checkpoint_name = 'latest'
        
    if os.path.exists(model_path):
        dqn_agent.policy_net.load_state_dict(torch.load(model_path))
        dqn_agent.policy_net.eval()
        print(f"Model loaded from {model_path}. Device: {next(dqn_agent.policy_net.parameters()).device}")
    else:
        print(f"Warning: No checkpoint found at {model_path}, using untrained model")
    
    # Video writer settings
    fps = 30
    frame_size = (160, 240)  # 160x210 plus space for text overlay
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Record random agent episodes
    print("Recording random agent episodes...")
    for episode in range(num_episodes):
        video_path = os.path.join(output_dir, f'random_agent_episode_{episode+1}.mp4')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
        reward = record_episode(env, random_agent, video_writer)
        video_writer.release()
        print(f"Random Agent Episode {episode+1} - Score: {reward}")
    
    # Record trained agent episodes
    print(f"\nRecording trained agent episodes (checkpoint: {checkpoint_name})...")
    for episode in range(num_episodes):
        video_path = os.path.join(output_dir, f'trained_agent_{checkpoint_name}_episode_{episode+1}.mp4')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
        reward = record_episode(env, dqn_agent, video_writer, epsilon=0.0)
        video_writer.release()
        print(f"Trained Agent Episode {episode+1} - Score: {reward}")
    
    env.close()
    print("\nAll videos recorded successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes to record for each agent')
    parser.add_argument('--output_dir', type=str, default='videos', help='Directory to save videos')
    parser.add_argument('--skill_level', type=int, choices=[0,1,2,3], help='Skill level checkpoint to use (0-3). If not specified, uses latest checkpoint.')
    args = parser.parse_args()
    
    record_gameplay_videos(
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        skill_level=args.skill_level
    ) 