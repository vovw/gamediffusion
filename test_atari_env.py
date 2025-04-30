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

def test_dqn_agent_initialization():
    from dqn_agent import DQNAgent
    agent = DQNAgent(n_actions=4, state_shape=(4, 84, 84))
    assert hasattr(agent, 'select_action'), "DQNAgent should have select_action method"
    assert hasattr(agent, 'optimize_model'), "DQNAgent should have optimize_model method"
    assert hasattr(agent, 'update_target_network'), "DQNAgent should have update_target_network method"
    assert hasattr(agent, 'replay_buffer'), "DQNAgent should have a replay_buffer attribute"

def test_dqn_agent_action_selection():
    from dqn_agent import DQNAgent
    agent = DQNAgent(n_actions=4, state_shape=(4, 84, 84))
    dummy_state = np.zeros((4, 84, 84), dtype=np.uint8)
    action = agent.select_action(dummy_state, epsilon=1.0)
    assert isinstance(action, int), "Action should be an integer"
    assert 0 <= action < 4, "Action should be in range [0, 3]"

def test_dqn_replay_buffer():
    from dqn_agent import ReplayBuffer
    buffer = ReplayBuffer(capacity=100)
    dummy_transition = (np.zeros((4, 84, 84), dtype=np.uint8), 1, 1.0, np.zeros((4, 84, 84), dtype=np.uint8), False)
    buffer.push(*dummy_transition)
    assert len(buffer) == 1, "ReplayBuffer should store transitions"
    sample = buffer.sample(1)
    assert len(sample) == 1, "Sampled batch should have correct size"

def test_dqn_target_network_sync():
    from dqn_agent import DQNAgent
    agent = DQNAgent(n_actions=4, state_shape=(4, 84, 84))
    # Simulate parameter change
    for param in agent.policy_net.parameters():
        param.data += 1.0
    agent.update_target_network()
    for p, t in zip(agent.policy_net.parameters(), agent.target_net.parameters()):
        assert np.allclose(p.detach().cpu().numpy(), t.detach().cpu().numpy()), "Target network should sync with policy network"

def test_dqn_epsilon_greedy_policy():
    from dqn_agent import DQNAgent
    agent = DQNAgent(n_actions=4, state_shape=(4, 84, 84))
    dummy_state = np.zeros((4, 84, 84), dtype=np.uint8)
    actions = [agent.select_action(dummy_state, epsilon=1.0) for _ in range(100)]
    assert len(set(actions)) > 1, "With epsilon=1.0, actions should be random"
    actions = [agent.select_action(dummy_state, epsilon=0.0) for _ in range(10)]
    assert len(set(actions)) == 1, "With epsilon=0.0, actions should be greedy (identical)" 