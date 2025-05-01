import os
import cv2
import numpy as np
import torch
from atari_env import AtariBreakoutEnv
from dqn_agent import DQNAgent
from random_agent import RandomAgent
import gymnasium as gym

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
    text = f"Frame: {frame_num} | Action: {action_names[action]} | Score: {cumulative_reward:.1f}"
    cv2.putText(frame_with_text, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame_with_text

def record_episode(env, agent, video_writer, epsilon=0.0):
    """Record a single episode."""
    obs, info = env.reset()
    # Initialize state stack for DQN
    state_stack = np.stack([obs] * 4, axis=0)
    cumulative_reward = 0
    frame_num = 0
    
    while True:
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

def record_gameplay_videos(num_episodes=5, output_dir='videos'):
    """Record gameplay videos for both random and trained agents."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment with rendering
    env = gym.make(
        "ALE/Breakout-v5",
        render_mode='rgb_array',
        frameskip=4,
        repeat_action_probability=0.0,
        full_action_space=False
    )
    env = AtariBreakoutEnv()  # Wrap it with our preprocessor
    
    # Initialize agents
    random_agent = RandomAgent(n_actions=4)
    dqn_agent = DQNAgent(n_actions=4, state_shape=(4, 84, 84))
    
    # Load trained model if available
    model_path = os.path.join('checkpoints', 'dqn_latest.pth')
    if os.path.exists(model_path):
        dqn_agent.policy_net.load_state_dict(torch.load(model_path))
        dqn_agent.policy_net.eval()
    
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
    print("\nRecording trained agent episodes...")
    for episode in range(num_episodes):
        video_path = os.path.join(output_dir, f'trained_agent_episode_{episode+1}.mp4')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
        reward = record_episode(env, dqn_agent, video_writer, epsilon=0.0)
        video_writer.release()
        print(f"Trained Agent Episode {episode+1} - Score: {reward}")
    
    env.close()
    print("\nAll videos recorded successfully!")

if __name__ == '__main__':
    record_gameplay_videos() 