import pytest
import numpy as np
import os
from atari_env import AtariBreakoutEnv
from random_agent import RandomAgent

def test_env_initialization():
    env = AtariBreakoutEnv()
    assert hasattr(env, 'reset'), "Environment should have reset method"
    assert hasattr(env, 'step'), "Environment should have step method"
    assert hasattr(env, 'close'), "Environment should have close method"

def test_env_reset():
    env = AtariBreakoutEnv()
    obs, info = env.reset()
    
    # Test observation shape (84x84 grayscale)
    assert isinstance(obs, np.ndarray), "Observation should be numpy array"
    assert obs.shape == (84, 84), "Observation should be 84x84"
    assert obs.dtype == np.uint8, "Observation should be uint8"
    
    # Test info dict
    assert isinstance(info, dict), "Info should be a dictionary"
    env.close()

def test_env_step():
    env = AtariBreakoutEnv()
    env.reset()
    
    # Test step with action
    obs, reward, terminated, truncated, info = env.step(0)  # NOOP action
    
    assert isinstance(obs, np.ndarray), "Observation should be numpy array"
    assert obs.shape == (84, 84), "Observation should be 84x84"
    assert isinstance(reward, float), "Reward should be a float"
    assert isinstance(terminated, bool), "Terminated should be a boolean"
    assert isinstance(truncated, bool), "Truncated should be a boolean"
    assert isinstance(info, dict), "Info should be a dictionary"
    env.close()

def test_random_agent_initialization():
    agent = RandomAgent(n_actions=4)
    assert hasattr(agent, 'select_action'), "Agent should have select_action method"
    assert hasattr(agent, 'record_transition'), "Agent should have record_transition method"
    assert hasattr(agent, 'save_episode_data'), "Agent should have save_episode_data method"

def test_random_agent_action_selection():
    agent = RandomAgent(n_actions=4)
    action = agent.select_action()
    assert isinstance(action, int), "Action should be an integer"
    assert 0 <= action < 4, "Action should be in range [0, 3]"

def test_data_recording():
    # Create temporary directories for testing
    os.makedirs("test_data/episode_001", exist_ok=True)
    
    agent = RandomAgent(n_actions=4)
    dummy_frame = np.zeros((84, 84), dtype=np.uint8)
    
    # Record some transitions
    agent.record_transition(dummy_frame, action=1, reward=1.0, episode=1, step=0)
    agent.record_transition(dummy_frame, action=2, reward=0.0, episode=1, step=1)
    
    # Save episode data
    agent.save_episode_data(
        episode_dir="test_data/episode_001",
        actions_file="test_data/actions.json"
    )
    
    # Check if files were created
    assert os.path.exists("test_data/episode_001/frame_00000.png"), "Frame 0 should be saved"
    assert os.path.exists("test_data/episode_001/frame_00001.png"), "Frame 1 should be saved"
    assert os.path.exists("test_data/actions.json"), "Actions file should be saved"
    
    # Clean up
    import shutil
    shutil.rmtree("test_data")

def test_full_episode():
    env = AtariBreakoutEnv()
    agent = RandomAgent(n_actions=4)
    
    obs, info = env.reset()
    done = False
    step = 0
    
    while not done and step < 10:  # Run for max 10 steps for testing
        action = agent.select_action()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.record_transition(obs, action, reward, episode=1, step=step)
        step += 1
    
    assert step > 0, "Should complete at least one step"
    env.close() 