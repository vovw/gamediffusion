import os
import numpy as np
import torch
from tqdm import trange
from atari_env import AtariBreakoutEnv
from dqn_agent import DQNAgent, ReplayBuffer
import cv2
import json
import random
import argparse
from rnd import RandomNetworkDistillation
import csv
import time

# Config
config = {
    'env_name': 'Breakout',
    'n_actions': 4,
    'state_shape': (8, 84, 84),
    'max_episodes': 10000,
    'max_steps': 1000,
    'target_update_freq': 1000,
    'checkpoint_dir': 'checkpoints',
    'data_dir': 'data/raw_gameplay',
    'actions_dir': 'data/actions',
    'save_freq': 10,
    'min_buffer': 100000,
    'seed': 42,
    'skill_thresholds': [0, 50, 150, 250],
    'episodes_per_skill': 50,
    'epsilon': 0.1,  # Epsilon for epsilon-greedy exploration
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


def evaluate_agent(agent, env, n_episodes=10, log_id=None):
    """Evaluate agent and return average reward. Logs Q-values, softmax probabilities, and actions to CSV."""
    rewards = []
    log_rows = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        state_stack = np.stack([obs] * 8, axis=0)
        done = False
        total_reward = 0
        for step in range(config['max_steps']):
            state_tensor = torch.from_numpy(state_stack).unsqueeze(0).to(agent.device)
            agent.policy_net.eval()
            with torch.no_grad():
                q_values_tensor = agent.policy_net(state_tensor)
                q_values = q_values_tensor.cpu().numpy().flatten()
                if np.random.rand() < 0.05:
                    action = np.random.randint(0, config['n_actions'])
                else:
                    action = int(np.argmax(q_values))
            probs = np.exp(q_values) / np.sum(np.exp(q_values))
            log_rows.append({
                'episode': ep,
                'step': step,
                'q_values': ','.join(f'{q:.4f}' for q in q_values),
                'probabilities': ','.join(f'{p:.4f}' for p in probs),
                'action_selected': action,
                'reward': reward
            })
            next_obs, reward, terminated, truncated, info = env.step(action)
            state_stack = np.roll(state_stack, shift=-1, axis=0)
            state_stack[-1] = next_obs
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        rewards.append(total_reward)
    # Save CSV log
    os.makedirs('eval_logs', exist_ok=True)
    if log_id is None:
        log_id = time.strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join('eval_logs', f'eval_log_{log_id}.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['episode', 'step', 'q_values', 'probabilities', 'action_selected', 'reward']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in log_rows:
            writer.writerow(row)
    return np.mean(rewards)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_episodes', type=int, default=None)
    parser.add_argument('--min_buffer', type=int, default=None)
    parser.add_argument('--save_freq', type=int, default=None)
    parser.add_argument('--no_save', action='store_true', help='Do not save frames or actions during training')
    parser.add_argument('--exploration_mode', type=str, choices=['epsilon'], default='epsilon', help='Exploration mode: epsilon-greedy (default)')
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

    # --- Exploration Mode Setup ---
    exploration_mode = args.exploration_mode
    if exploration_mode == 'epsilon':
        agent = DQNAgent(
            n_actions=config['n_actions'],
            state_shape=config['state_shape'],
            prioritized=False,  # Use random replay buffer
        )
        epsilon_start = 1.0
        epsilon_final = 0.1
        epsilon_decay_steps = 1_000_000
    else:
        shared_replay_buffer = ReplayBuffer(capacity=1000000)
        exploration_agent = DQNAgent(n_actions=config['n_actions'], state_shape=config['state_shape'], replay_buffer=shared_replay_buffer)
        exploitation_agent = DQNAgent(n_actions=config['n_actions'], state_shape=config['state_shape'], replay_buffer=shared_replay_buffer)
        from rnd import RandomNetworkDistillation
        rnd = RandomNetworkDistillation(state_shape=config['state_shape'], output_dim=512, lr=1e-5, reward_scale=0.1)
        alpha = 0.5
        alpha_decay = 0.99995
        min_alpha = 0.05
    # ---

    total_steps = 0
    exploration_rewards = []
    exploitation_rewards = []
    intrinsic_rewards_log = []
    running_avg_rewards = []
    skill_level = 0
    skill_episodes = 0
    skill_thresholds = config['skill_thresholds']
    skill_counts = [0] * len(skill_thresholds)
    pbar = trange(config['max_episodes'], desc='Training')

    # Fill replay buffer with random actions first
    print("Pre-filling replay buffer with random experiences...")
    env = AtariBreakoutEnv()
    obs, info = env.reset()
    state_stack = np.stack([obs] * 8, axis=0)
    replay_buffer = agent.replay_buffer if exploration_mode == 'epsilon' else shared_replay_buffer
    for step in range(max(5000, config['min_buffer'])):
        action = np.random.randint(0, config['n_actions'])
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state_stack = np.roll(state_stack, shift=-1, axis=0)
        next_state_stack[-1] = next_obs
        # Add small random priority variation during prefill
        #random_priority = random.uniform(0.1, 1.0)
        # For prefill, set intrinsic reward to 0
        #replay_buffer.push(state_stack, action, reward, 0.0, next_state_stack, terminated or truncated, random_priority)
        replay_buffer.push(state_stack, action, reward, 0.0, next_state_stack, terminated or truncated)
        state_stack = next_state_stack
        if terminated or truncated:
            obs, info = env.reset()
            state_stack = np.stack([obs] * 8, axis=0)
        if step % 1000 == 0:
            print(f"  {step}/{max(5000, config['min_buffer'])} experiences collected")

    eval_env = AtariBreakoutEnv()

    for episode in pbar:
        # Periodic evaluation
        if episode % 10 == 0:
            log_id = f"ep{episode}"
            eval_reward = evaluate_agent(agent, eval_env, n_episodes=5, log_id=log_id)
            policy_str = f"Policy eval: {eval_reward:.1f}"
        else:
            policy_str = ""

        obs, info = env.reset()
        state_stack = np.stack([obs] * 8, axis=0)
        frames, actions, extrinsic_rewards, intrinsic_rewards = [obs], [], [], []
        total_combined_reward = 0
        total_extrinsic_reward = 0
        losses = []
        done = False
        if exploration_mode == 'epsilon':
            for step in range(config['max_steps']):
                # Epsilon annealing
                epsilon = max(epsilon_final, epsilon_start - (epsilon_start - epsilon_final) * min(1.0, total_steps / epsilon_decay_steps))
                action = agent.select_action(state_stack, mode='epsilon', epsilon=epsilon)
                next_obs, extrinsic_reward, terminated, truncated, info = env.step(action)
                next_state_stack = np.roll(state_stack, shift=-1, axis=0)
                next_state_stack[-1] = next_obs
                agent.replay_buffer.push(state_stack, action, extrinsic_reward, 0.0, next_state_stack, terminated or truncated)
                for _ in range(5):
                    loss = agent.optimize_model(mode='exploitation')
                    if loss is not None:
                        losses.append(loss)
                state_stack = next_state_stack
                frames.append(next_obs)
                actions.append(action)
                extrinsic_rewards.append(extrinsic_reward)
                total_extrinsic_reward += extrinsic_reward
                total_combined_reward += extrinsic_reward
                total_steps += 1
                if total_steps % config['target_update_freq'] == 0:
                    agent.update_target_network()
                if terminated or truncated:
                    break
            avg_loss = np.mean(losses) if losses else 0.0
            exploration_rewards.append(total_combined_reward)
            exploitation_rewards.append(total_extrinsic_reward)
            running_avg = np.mean(exploration_rewards[-min(100, len(exploration_rewards)):])
            running_avg_rewards.append(running_avg)
            pbar.set_postfix({
                'explore_r': total_combined_reward,
                'exploit_r': total_extrinsic_reward,
                'epsilon': f"{epsilon:.3f}",
                'loss': f"{avg_loss:.4f}"
            })
            if episode > 0 and episode % 10 == 0:
                last_10_explore = exploration_rewards[-10:]
                last_10_exploit = exploitation_rewards[-10:]
                print(f"[Stats] Episodes {episode-9}-{episode} | Explore Avg: {np.mean(last_10_explore):.2f} | Exploit Avg: {np.mean(last_10_exploit):.2f} | {policy_str}")
                
                
                
        else:
            # --- RND mode (original logic) ---
            for step in range(config['max_steps']):
                action = exploration_agent.select_action(state_stack, mode='softmax', temperature=1.0, epsilon=config['epsilon'])
                next_obs, extrinsic_reward, terminated, truncated, info = env.step(action)
                next_state_stack = np.roll(state_stack, shift=-1, axis=0)
                next_state_stack[-1] = next_obs
                intrinsic_reward = float(rnd.compute_intrinsic_reward(np.expand_dims(next_state_stack, axis=0)).cpu().numpy()[0])
                shared_replay_buffer.push(state_stack, action, extrinsic_reward, intrinsic_reward, next_state_stack, terminated or truncated)
                for _ in range(5):
                    loss = exploration_agent.optimize_model(mode='exploration', alpha=alpha)
                    if loss is not None:
                        losses.append(loss)
                for _ in range(5):
                    loss = exploitation_agent.optimize_model(mode='exploitation')
                if np.random.rand() < 0.2:
                    if len(shared_replay_buffer) >= 128:
                        batch = shared_replay_buffer.sample(128, mode='exploration', alpha=alpha)
                        states, _, _, _, _ = batch
                        states_np = np.stack(states)
                        rnd.update(states_np)
                state_stack = next_state_stack
                frames.append(next_obs)
                actions.append(action)
                extrinsic_rewards.append(extrinsic_reward)
                intrinsic_rewards.append(intrinsic_reward)
                total_extrinsic_reward += extrinsic_reward
                total_combined_reward += (1 - alpha) * extrinsic_reward + alpha * intrinsic_reward
                total_steps += 1
                if total_steps % config['target_update_freq'] == 0:
                    exploration_agent.update_target_network()
                    exploitation_agent.update_target_network()
                if terminated or truncated:
                    break
            avg_loss = np.mean(losses) if losses else 0.0
            exploration_rewards.append(total_combined_reward)
            exploitation_rewards.append(total_extrinsic_reward)
            intrinsic_rewards_log.append(np.sum(intrinsic_rewards))
            running_avg = np.mean(exploration_rewards[-min(100, len(exploration_rewards)):])
            running_avg_rewards.append(running_avg)
            alpha = max(min_alpha, alpha * alpha_decay)
            if episode % 10 == 0:
                log_id = f"ep{episode}"
                eval_reward = evaluate_agent(exploitation_agent, eval_env, n_episodes=5, log_id=log_id)
                policy_str = f"Policy eval: {eval_reward:.1f}"
            else:
                policy_str = ""
            pbar.set_postfix({
                'explore_r': total_combined_reward,
                'exploit_r': total_extrinsic_reward,
                'alpha': f"{alpha:.3f}",
                'loss': f"{avg_loss:.4f}"
            })
            if episode > 0 and episode % 10 == 0:
                last_10_explore = exploration_rewards[-10:]
                last_10_exploit = exploitation_rewards[-10:]
                last_10_intrinsic = intrinsic_rewards_log[-10:]
                print(f"[Stats] Episodes {episode-9}-{episode} | Explore Avg: {np.mean(last_10_explore):.2f} | Exploit Avg: {np.mean(last_10_exploit):.2f} | {policy_str} | Alpha: {alpha:.3f}")
                print(f"RND Stats: {rnd.get_stats()}")
                intrinsic_ratio = np.mean(last_10_intrinsic) / (np.mean(np.abs(last_10_exploit)) + 1e-8)
                print(f"Intrinsic/Extrinsic ratio: {intrinsic_ratio:.2f}")
        # Save episode data if in skill collection phase
        if (not args.no_save and
            skill_level < len(skill_thresholds) and
            running_avg >= skill_thresholds[skill_level]):
            if skill_counts[skill_level] < config['episodes_per_skill']:
                save_episode_data(frames, actions, extrinsic_rewards, skill_level, skill_counts[skill_level]+1, config)
                skill_counts[skill_level] += 1
            if skill_counts[skill_level] >= config['episodes_per_skill']:
                print(f"Skill level {skill_level} ({skill_thresholds[skill_level]}+) complete.")
                checkpoint_path = os.path.join(config['checkpoint_dir'], f'dqn_skill_{skill_level}.pth')
                torch.save(agent.policy_net.state_dict(), checkpoint_path)
                param_count = sum(p.numel() for p in agent.policy_net.parameters())
                param_sum = sum(p.sum().item() for p in agent.policy_net.parameters())
                print(f"Saved model to {checkpoint_path}")
                skill_level += 1
        if episode % config['save_freq'] == 0:
            checkpoint_path = os.path.join(config['checkpoint_dir'], 'dqn_latest.pth')
            torch.save(agent.policy_net.state_dict(), checkpoint_path)
            param_count = sum(p.numel() for p in agent.policy_net.parameters())
            param_sum = sum(p.sum().item() for p in agent.policy_net.parameters())
        if skill_level >= len(skill_thresholds):
            print("All skill levels collected. Training complete.")
            break
    env.close()
    eval_env.close()
    checkpoint_path = os.path.join(config['checkpoint_dir'], 'dqn_latest.pth')
    torch.save(agent.policy_net.state_dict(), checkpoint_path)
    param_count = sum(p.numel() for p in agent.policy_net.parameters())
    param_sum = sum(p.sum().item() for p in agent.policy_net.parameters())
    print(f"\nSaved final model to {checkpoint_path}")
    print(f"Parameter count: {param_count:,}")
    print(f"Parameter sum: {param_sum:.2f}")
    print("Training finished.")

if __name__ == '__main__':
    main() 