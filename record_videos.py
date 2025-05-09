"""Record gameplay videos of trained and random agents playing Atari Breakout.

This script records videos of both a random agent and a trained DQN agent playing Atari Breakout.
Videos include an overlay showing frame number, action taken, current score, and softmax action probabilities.

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
    
    # Enable debug information
    python record_videos.py --debug

    # Adjust action selection randomness (lower values = more deterministic)
    python record_videos.py --temperature 0.5

Arguments:
    --num_episodes: Number of episodes to record for each agent (default: 5)
    --output_dir: Directory to save videos (default: 'videos')
    --skill_level: Skill level checkpoint to use (0-3). If not specified, uses latest checkpoint.
    --no_sync: Set to True to use a non-synced directory for video output (default: False)
    --temperature: Softmax temperature for agent action selection (default: 1.0, inf=greedy/random)
    --debug: Enable debug logging for troubleshooting video recording issues (default: False)

Output:
    - Random agent videos: random_agent_episode_X.mp4
    - Trained agent videos: trained_agent_[checkpoint]_episode_X.mp4
    where [checkpoint] is either 'latest' or 'skill_N' depending on --skill_level
    
Video Features:
    - Frame number, action taken, and score in the first overlay row
    - Softmax probabilities for all 4 actions in the second overlay row (for DQN agent)
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

def add_text_overlay(frame, frame_num, action, cumulative_reward, softmax_probs=None):
    """Add text overlay to frame with game information and (optionally) softmax probabilities."""
    # Convert to RGB if grayscale
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    
    # Create a slightly larger frame to accommodate text
    height, width = frame.shape[:2]
    text_height = 20
    extra_height = text_height
    if softmax_probs is not None:
        extra_height += 20  # Add space for softmax row
    frame_with_text = np.zeros((height + extra_height, width, 3), dtype=np.uint8)
    frame_with_text[extra_height:, :, :] = frame
    
    # Add black background for text
    cv2.rectangle(frame_with_text, (0, 0), (width, extra_height), (0, 0, 0), -1)
    
    # Add main text
    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    text = f"{frame_num} | {action_names[action]} | {cumulative_reward:.1f}"
    cv2.putText(frame_with_text, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Add softmax probabilities if provided
    if softmax_probs is not None:
        probs_str = ' '.join([f"{p:.2f}" for p in softmax_probs])
        cv2.putText(frame_with_text, f"P: {probs_str}", (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 255, 180), 1)
    
    return frame_with_text

def record_episode(env, agent, video_writer, temperature=1.0, frame_size=None, debug=False):
    """Record a single episode.
    
    Args:
        env: Environment to record
        agent: Agent to use for action selection
        video_writer: Video writer object
        temperature: Temperature for softmax action selection
        frame_size: Size of output frame as (width, height)
        debug: If True, print debug information
    """
    obs, info = env.reset()
    # Initialize state stack for DQN
    state_stack = np.stack([obs] * 8, axis=0)
    cumulative_reward = 0
    frame_num = 0
    max_steps = 1000
    frames_written = 0
    
    while True:
        if frame_num > max_steps:
            break
        
        # Get the raw frame from environment
        try:
            raw_frame = env.env.render()  # This gets the 160x210 RGB frame
            if debug:
                print(f"Frame {frame_num} shape: {raw_frame.shape if raw_frame is not None else 'None'}")
        except Exception as e:
            if debug:
                print(f"Error rendering frame {frame_num}: {e}")
            raw_frame = None
        
        if raw_frame is None or raw_frame.size == 0:
            if debug:
                print(f"Warning: Received empty frame from environment at step {frame_num}. Using dummy frame.")
            # Create a dummy frame as fallback
            raw_frame = np.zeros((210, 160, 3), dtype=np.uint8)
        
        # Get action from agent
        softmax_probs = None
        if hasattr(agent, 'select_action'):
            if agent.__class__.__name__ == 'DQNAgent':
                # Use 'greedy' mode by default, 'softmax' only if temperature is not None and not inf
                mode = 'softmax' if (temperature is not None and not np.isinf(temperature)) else 'greedy'
                if mode == 'softmax':
                    softmax_probs = agent.get_action_softmax_probs(state_stack, temperature=temperature)
                action = agent.select_action(state_stack, mode=mode, temperature=temperature)
            else:
                action = agent.select_action(temperature=temperature)
        else:
            action = agent.select_action()
        
        # Add overlay
        frame_with_overlay = add_text_overlay(raw_frame, frame_num, action, cumulative_reward, softmax_probs=softmax_probs)
        
        # Check frame before writing
        if frame_with_overlay is not None:
            # Ensure frame has the right dimensions for the video writer
            if frame_size is not None and hasattr(video_writer, 'write'):
                expected_width, expected_height = frame_size[0], frame_size[1]
                actual_height, actual_width = frame_with_overlay.shape[0], frame_with_overlay.shape[1]
                
                if actual_height != expected_height or actual_width != expected_width:
                    if debug:
                        print(f"Warning: Frame dimensions mismatch. Expected: ({expected_width}, {expected_height}), Got: ({actual_width}, {actual_height})")
                    if expected_width > 0 and expected_height > 0:
                        # Only resize if dimensions are valid
                        frame_with_overlay = cv2.resize(frame_with_overlay, (expected_width, expected_height))
            
            try:
                video_writer.write(frame_with_overlay)
                frames_written += 1
            except Exception as e:
                if debug:
                    print(f"Error writing frame {frame_num}: {e}")
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        frame_num += 1
        if agent.__class__.__name__ == 'DQNAgent':
            state_stack = np.roll(state_stack, shift=-1, axis=0)
            state_stack[-1] = next_obs
        if terminated or truncated:
            try:
                raw_frame = env.env.render()
                if debug:
                    print(f"Final frame shape: {raw_frame.shape if raw_frame is not None else 'None'}")
            except Exception as e:
                if debug:
                    print(f"Error rendering final frame: {e}")
                raw_frame = None
                
            if raw_frame is not None and raw_frame.size > 0:
                frame_with_overlay = add_text_overlay(raw_frame, frame_num, action, cumulative_reward, softmax_probs=softmax_probs)
                
                # Check frame dimensions before writing
                if frame_size is not None and hasattr(video_writer, 'write'):
                    expected_width, expected_height = frame_size[0], frame_size[1]
                    actual_height, actual_width = frame_with_overlay.shape[0], frame_with_overlay.shape[1]
                    
                    if actual_height != expected_height or actual_width != expected_width:
                        if debug:
                            print(f"Warning: Final frame dimensions mismatch. Expected: ({expected_width}, {expected_height}), Got: ({actual_width}, {actual_height})")
                        if expected_width > 0 and expected_height > 0:
                            # Only resize if dimensions are valid
                            frame_with_overlay = cv2.resize(frame_with_overlay, (expected_width, expected_height))
                
                try:
                    video_writer.write(frame_with_overlay)
                    frames_written += 1
                except Exception as e:
                    if debug:
                        print(f"Error writing final frame: {e}")
            break
    
    if debug:
        print(f"Episode completed - wrote {frames_written} frames")
    return cumulative_reward

def record_gameplay_videos(num_episodes=5, output_dir='videos', skill_level=None, no_sync=False, temperature=1.0, debug=False):
    """Record gameplay videos for both random and trained agents.
    
    Args:
        num_episodes: Number of episodes to record for each agent
        output_dir: Directory to save videos
        skill_level: If specified (0-3), load that skill level's checkpoint.
                    If None, use latest checkpoint.
        no_sync: If True, use a temporary directory and copy files to output_dir after completion
        temperature: Softmax temperature for action selection (default: 1.0)
        debug: If True, print debug information
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
    
    # Try to render a single frame to get the actual frame size
    env.reset()
    try:
        test_frame = env.env.render()
        if test_frame is not None:
            if debug:
                print(f"Test frame shape: {test_frame.shape}")
            # Calculate frame size based on actual frame + extra space for text
            frame_height = test_frame.shape[0]
            frame_width = test_frame.shape[1]
            
            # Add space for text overlay
            extra_height = 40  # 20 for each text line
            frame_size = (frame_width, frame_height + extra_height)
        else:
            if debug:
                print("Warning: Test frame is None. Using default size: (160, 230)")
            frame_size = (160, 230)  # 160x210 plus space for text overlay
    except Exception as e:
        if debug:
            print(f"Error rendering test frame: {e}. Using default size: (160, 230)")
        frame_size = (160, 230)
    
    print(f"Using frame size: {frame_size}")
    fps = 30
    
    # Try different codecs in order of preference (reordered for better compatibility)
    codecs = [
        ('MJPG', '.avi'),   # Motion JPEG - most compatible
        ('XVID', '.avi'),   # XVID codec
        ('mp4v', '.mp4'),   # MPEG-4 codec
        ('avc1', '.mp4'),   # H.264 codec
        ('FMP4', '.mp4'),   # FFMPEG codec
        ('H264', '.mp4'),   # H264 codec
        ('X264', '.mp4'),   # x264 codec
        ('WMV1', '.wmv'),   # Windows Media codec
        ('THEO', '.ogv')    # Theora codec
    ]
    
    # Select a working codec
    working_codec = None
    for codec, ext in codecs:
        try:
            test_path = os.path.join(working_dir, f'test{ext}')
            fourcc = cv2.VideoWriter_fourcc(*codec)
            test_writer = cv2.VideoWriter(test_path, fourcc, fps, frame_size)
            
            # Skip if we couldn't open the writer
            if not test_writer.isOpened():
                if debug:
                    print(f"Failed to open video writer with codec {codec}")
                continue
                
            # Create a dummy frame and write it
            dummy_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            # Draw something to ensure frame is not empty
            cv2.rectangle(dummy_frame, (10, 10), (100, 100), (0, 255, 0), -1)
            test_writer.write(dummy_frame)
            test_writer.release()
            
            # Check if file exists and has content
            if os.path.exists(test_path) and os.path.getsize(test_path) > 1000:
                working_codec = (codec, ext)
                if debug:
                    print(f"Successfully tested codec {codec} - file size: {os.path.getsize(test_path)} bytes")
                os.remove(test_path)
                break
            
            if os.path.exists(test_path):
                if debug:
                    print(f"Codec {codec} created file but it was too small: {os.path.getsize(test_path)} bytes")
                os.remove(test_path)
                
        except Exception as e:
            if debug:
                print(f"Codec {codec} failed: {str(e)}")
    
    if working_codec is None:
        print("Error: Could not find a working video codec. Using MJPG with raw frame writing as fallback.")
        working_codec = ('MJPG', '.avi')
    
    # Add PNG fallback for saving individual frames if everything else fails
    use_png_fallback = False
    frame_dir = os.path.join(working_dir, 'frames')
    
    print(f"Using codec: {working_codec[0]} with extension {working_codec[1]}")
    fourcc = cv2.VideoWriter_fourcc(*working_codec[0])
    file_ext = working_codec[1]
    
    # Function to test if a video writer actually works
    def create_and_test_writer(path, fourcc, fps, size):
        try:
            writer = cv2.VideoWriter(path, fourcc, fps, size)
            if not writer.isOpened():
                if debug:
                    print(f"Error: Could not open video writer for {path}")
                return None
            
            # Write a test frame
            test_frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            writer.write(test_frame)
            
            return writer
        except Exception as e:
            if debug:
                print(f"Error creating video writer: {e}")
            return None
    
    # Record random agent episodes
    print("Recording random agent episodes...")
    for episode in range(num_episodes):
        video_path = os.path.join(working_dir, f'random_agent_episode_{episode+1}{file_ext}')
        video_writer = create_and_test_writer(video_path, fourcc, fps, frame_size)
        
        if video_writer is None:
            print(f"Error: Could not create video writer for {video_path}")
            print("Falling back to PNG sequence for this episode")
            use_png_fallback = True
            png_dir = os.path.join(frame_dir, f'random_agent_episode_{episode+1}')
            os.makedirs(png_dir, exist_ok=True)
            # Create a special video writer that saves to PNG
            class PNGWriter:
                def __init__(self, output_dir):
                    self.output_dir = output_dir
                    self.frame_count = 0
                    self.frame_size = frame_size
                    
                def write(self, frame):
                    path = os.path.join(self.output_dir, f'frame_{self.frame_count:05d}.png')
                    cv2.imwrite(path, frame)
                    self.frame_count += 1
                
                def release(self):
                    pass
                
                def isOpened(self):
                    return True
                
                def get(self, propId):
                    if propId == cv2.CAP_PROP_FRAME_WIDTH:
                        return self.frame_size[0]
                    elif propId == cv2.CAP_PROP_FRAME_HEIGHT:
                        return self.frame_size[1]
                    return 0
            
            video_writer = PNGWriter(png_dir)
        
        reward = record_episode(env, random_agent, video_writer, temperature=temperature, frame_size=frame_size, debug=debug)
        video_writer.release()
        time.sleep(0.5)
        
        if not use_png_fallback:
            file_size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
            print(f"Random Agent Episode {episode+1} - Score: {reward} - Video size: {file_size} bytes")
            if file_size < 1000:
                print(f"Warning: Video file may be corrupted (small size): {video_path}")
        else:
            print(f"Random Agent Episode {episode+1} - Score: {reward} - Saved as PNG sequence")
    
    # Record trained agent episodes
    print(f"\nRecording trained agent episodes (checkpoint: {checkpoint_name})...")
    for episode in range(num_episodes):
        video_path = os.path.join(working_dir, f'trained_agent_{checkpoint_name}_episode_{episode+1}{file_ext}')
        video_writer = create_and_test_writer(video_path, fourcc, fps, frame_size)
        
        if video_writer is None:
            print(f"Error: Could not create video writer for {video_path}")
            print("Falling back to PNG sequence for this episode")
            use_png_fallback = True
            png_dir = os.path.join(frame_dir, f'trained_agent_{checkpoint_name}_episode_{episode+1}')
            os.makedirs(png_dir, exist_ok=True)
            
            class PNGWriter:
                def __init__(self, output_dir):
                    self.output_dir = output_dir
                    self.frame_count = 0
                    self.frame_size = frame_size
                    
                def write(self, frame):
                    path = os.path.join(self.output_dir, f'frame_{self.frame_count:05d}.png')
                    cv2.imwrite(path, frame)
                    self.frame_count += 1
                
                def release(self):
                    pass
                
                def isOpened(self):
                    return True
                
                def get(self, propId):
                    if propId == cv2.CAP_PROP_FRAME_WIDTH:
                        return self.frame_size[0]
                    elif propId == cv2.CAP_PROP_FRAME_HEIGHT:
                        return self.frame_size[1]
                    return 0
            
            video_writer = PNGWriter(png_dir)
        
        reward = record_episode(env, dqn_agent, video_writer, temperature=temperature, frame_size=frame_size, debug=debug)
        video_writer.release()
        time.sleep(0.5)
        
        if not use_png_fallback:
            file_size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
            print(f"Trained Agent Episode {episode+1} - Score: {reward} - Video size: {file_size} bytes")
            if file_size < 1000:
                print(f"Warning: Video file may be corrupted (small size): {video_path}")
        else:
            print(f"Trained Agent Episode {episode+1} - Score: {reward} - Saved as PNG sequence")
    
    env.close()
    
    # If using temporary directory, copy files to the output directory
    if no_sync:
        print(f"\nCopying files from temporary directory to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        
        for filename in os.listdir(temp_dir):
            src_path = os.path.join(temp_dir, filename)
            dst_path = os.path.join(output_dir, filename)
            
            if os.path.isdir(src_path):
                # Handle directories (like frame_dir)
                if os.path.basename(src_path) == 'frames':
                    dst_frame_dir = os.path.join(output_dir, 'frames')
                    os.makedirs(dst_frame_dir, exist_ok=True)
                    for frame_dirname in os.listdir(src_path):
                        src_frame_dir = os.path.join(src_path, frame_dirname)
                        dst_subdir = os.path.join(dst_frame_dir, frame_dirname)
                        if os.path.isdir(src_frame_dir):
                            shutil.copytree(src_frame_dir, dst_subdir, dirs_exist_ok=True)
                            print(f"Copied frames directory: {frame_dirname}")
            elif os.path.getsize(src_path) > 1000:  # Only copy files that have content
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {filename} ({os.path.getsize(dst_path)} bytes)")
            else:
                print(f"Skipped corrupted file: {filename}")
                
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
    
    print("\nAll videos recorded successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Record gameplay videos for Atari Breakout with random and trained agents')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes to record for each agent')
    parser.add_argument('--output_dir', type=str, default='videos', help='Directory to save videos')
    parser.add_argument('--skill_level', type=int, choices=[0,1,2,3], help='Skill level checkpoint to use (0-3). If not specified, uses latest checkpoint.')
    parser.add_argument('--no_sync', action='store_true', help='Use a temporary directory to avoid iCloud sync issues')
    parser.add_argument('--temperature', type=float, default=1.0, help='Softmax temperature for agent action selection (default: 1.0, inf=greedy/random)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging for troubleshooting video recording issues')
    
    args = parser.parse_args()
    
    record_gameplay_videos(
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        skill_level=args.skill_level,
        no_sync=args.no_sync,
        temperature=args.temperature,
        debug=args.debug
    )