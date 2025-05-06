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
    --no_sync: Set to True to use a non-synced directory for video output (default: False)
    --temperature: Softmax temperature for agent action selection (default: 1.0, inf=greedy/random)

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
import time
import shutil
import tempfile

def add_text_overlay(frame, frame_num, action, cumulative_reward):
    """Add text overlay to frame with game information."""
    # Convert to RGB if grayscale
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    
    # Create a slightly larger frame to accommodate text
    height, width = frame.shape[:2]
    text_height = 20  # Reduced from 30 to 20
    frame_with_text = np.zeros((height + text_height, width, 3), dtype=np.uint8)
    frame_with_text[text_height:, :, :] = frame
    
    # Add black background for text
    cv2.rectangle(frame_with_text, (0, 0), (width, text_height), (0, 0, 0), -1)
    
    # Add text
    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    text = f"{frame_num} | {action_names[action]} | {cumulative_reward:.1f}"
    cv2.putText(frame_with_text, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame_with_text

def record_episode(env, agent, video_writer, temperature=1.0):
    """Record a single episode."""
    obs, info = env.reset()
    # Initialize state stack for DQN
    state_stack = np.stack([obs] * 8, axis=0)
    cumulative_reward = 0
    frame_num = 0
    max_steps = 100
    while True:
        if frame_num > max_steps:
            break
        raw_frame = env.env.render()  # This gets the 160x210 RGB frame
        if raw_frame is None or raw_frame.size == 0:
            print("Warning: Received empty frame from environment. Skipping frame.")
            continue
        # Get action from agent
        if hasattr(agent, 'select_action'):
            if agent.__class__.__name__ == 'DQNAgent':
                mode = 'softmax' if temperature is not None and not np.isinf(temperature) else 'greedy'
                action = agent.select_action(state_stack, mode=mode, temperature=temperature)
            else:
                action = agent.select_action(temperature=temperature)
        else:
            action = agent.select_action()
        frame_with_overlay = add_text_overlay(raw_frame, frame_num, action, cumulative_reward)
        if frame_with_overlay is not None:
            video_writer.write(frame_with_overlay)
        next_obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        frame_num += 1
        if agent.__class__.__name__ == 'DQNAgent':
            state_stack = np.roll(state_stack, shift=-1, axis=0)
            state_stack[-1] = next_obs
        if terminated or truncated:
            raw_frame = env.env.render()
            if raw_frame is not None and raw_frame.size > 0:
                frame_with_overlay = add_text_overlay(raw_frame, frame_num, action, cumulative_reward)
                video_writer.write(frame_with_overlay)
            break
    return cumulative_reward

def record_gameplay_videos(num_episodes=5, output_dir='videos', skill_level=None, no_sync=False, temperature=1.0):
    """Record gameplay videos for both random and trained agents.
    
    Args:
        num_episodes: Number of episodes to record for each agent
        output_dir: Directory to save videos
        skill_level: If specified (0-3), load that skill level's checkpoint.
                    If None, use latest checkpoint.
        no_sync: If True, use a temporary directory and copy files to output_dir after completion
        temperature: Softmax temperature for action selection (default: 1.0)
    """
    # Use a temporary directory if no_sync is True
    temp_dir = None
    working_dir = output_dir
    
    if no_sync:
        temp_dir = tempfile.mkdtemp()
        working_dir = temp_dir
        print(f"Using temporary directory: {temp_dir}")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    env = AtariBreakoutEnv()
    
    # Initialize agents
    random_agent = RandomAgent(n_actions=4)
    dqn_agent = DQNAgent(n_actions=4, state_shape=(8, 84, 84))
    
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
    frame_size = (160, 230)  # 160x210 plus space for text overlay
    
    # Try different codecs in order of preference
    codecs = [
        ('mp4v', '.mp4'),  # MPEG-4 codec
        ('avc1', '.mp4'),  # H.264 codec
        ('XVID', '.avi'),  # XVID codec
        ('MJPG', '.avi')   # Motion JPEG
    ]
    
    # Select a working codec
    working_codec = None
    for codec, ext in codecs:
        try:
            test_path = os.path.join(working_dir, f'test{ext}')
            fourcc = cv2.VideoWriter_fourcc(*codec)
            test_writer = cv2.VideoWriter(test_path, fourcc, fps, frame_size)
            
            # Create a dummy frame and write it
            dummy_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            test_writer.write(dummy_frame)
            test_writer.release()
            
            # Check if file exists and has content
            if os.path.exists(test_path) and os.path.getsize(test_path) > 1000:
                working_codec = (codec, ext)
                os.remove(test_path)
                break
            
            if os.path.exists(test_path):
                os.remove(test_path)
                
        except Exception as e:
            print(f"Codec {codec} failed: {str(e)}")
    
    if working_codec is None:
        print("Error: Could not find a working video codec.")
        return
        
    print(f"Using codec: {working_codec[0]} with extension {working_codec[1]}")
    fourcc = cv2.VideoWriter_fourcc(*working_codec[0])
    file_ext = working_codec[1]
    
    # Record random agent episodes
    print("Recording random agent episodes...")
    for episode in range(num_episodes):
        video_path = os.path.join(working_dir, f'random_agent_episode_{episode+1}{file_ext}')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {video_path}")
            continue
        reward = record_episode(env, random_agent, video_writer, temperature=temperature)
        video_writer.release()
        time.sleep(0.5)
        file_size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
        print(f"Random Agent Episode {episode+1} - Score: {reward} - Video size: {file_size} bytes")
        if file_size < 1000:
            print(f"Warning: Video file may be corrupted (small size): {video_path}")
    
    # Record trained agent episodes
    print(f"\nRecording trained agent episodes (checkpoint: {checkpoint_name})...")
    for episode in range(num_episodes):
        video_path = os.path.join(working_dir, f'trained_agent_{checkpoint_name}_episode_{episode+1}{file_ext}')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {video_path}")
            continue
        reward = record_episode(env, dqn_agent, video_writer, temperature=temperature)
        video_writer.release()
        time.sleep(0.5)
        file_size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
        print(f"Trained Agent Episode {episode+1} - Score: {reward} - Video size: {file_size} bytes")
        if file_size < 1000:
            print(f"Warning: Video file may be corrupted (small size): {video_path}")
    
    env.close()
    
    # If using temporary directory, copy files to the output directory
    if no_sync:
        print(f"\nCopying files from temporary directory to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        
        for filename in os.listdir(temp_dir):
            src_path = os.path.join(temp_dir, filename)
            dst_path = os.path.join(output_dir, filename)
            
            if os.path.getsize(src_path) > 1000:  # Only copy files that have content
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {filename} ({os.path.getsize(dst_path)} bytes)")
            else:
                print(f"Skipped corrupted file: {filename}")
                
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
    
    print("\nAll videos recorded successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes to record for each agent')
    parser.add_argument('--output_dir', type=str, default='videos', help='Directory to save videos')
    parser.add_argument('--skill_level', type=int, choices=[0,1,2,3], help='Skill level checkpoint to use (0-3). If not specified, uses latest checkpoint.')
    parser.add_argument('--no_sync', action='store_true', help='Use a temporary directory to avoid iCloud sync issues')
    parser.add_argument('--temperature', type=float, default=1.0, help='Softmax temperature for agent action selection (default: 1.0, inf=greedy/random)')
    args = parser.parse_args()
    
    record_gameplay_videos(
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        skill_level=args.skill_level,
        no_sync=args.no_sync,
        temperature=args.temperature
    )