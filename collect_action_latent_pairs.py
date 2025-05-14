"""
Collect (action, latent_code) pairs for Atari Breakout using a trained VQ-VAE and a random agent.

This script runs a random agent in the Atari Breakout environment, processes frame pairs through a trained VQ-VAE model,
and stores (action, latent_code) pairs for use in action-to-latent mapping experiments.

Usage:
    python collect_action_latent_pairs.py --out data/actions/action_latent_pairs.json --n_pairs 100000

Arguments:
    --out: Output path for the JSON file (default: data/actions/action_latent_pairs.json)
    --n_pairs: Number of (action, latent_code) pairs to collect (default: 100000)
    --max_steps_per_episode: Maximum steps per episode (default: 1000)
    --seed: Random seed (default: 42)

Requirements:
    - Trained VQ-VAE checkpoint at checkpoints/latent_action/best.pt
    - AtariBreakoutEnv and RandomAgent classes
    - torch, numpy, tqdm

Example:
    python collect_action_latent_pairs.py --out data/actions/action_latent_pairs.json --n_pairs 50000
"""

# python collect_action_latent_pairs.py --out data/actions/action_latent_pairs.json --n_pairs 100000

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from atari_env import AtariBreakoutEnv
from random_agent import RandomAgent
from latent_action_model import load_latent_action_model

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def flatten_latent_indices(indices):
    # indices: (5, 7) or (B, 5, 7)
    if indices.ndim == 2:
        return indices.flatten().tolist()
    elif indices.ndim == 3:
        return [x.flatten().tolist() for x in indices]
    else:
        raise ValueError(f"Unexpected indices shape: {indices.shape}")

def collect_action_latent_pairs(
    out_path='data/actions/action_latent_pairs.json',
    n_pairs=100_000,
    max_steps_per_episode=1000,
    seed=42
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    device = get_device()
    model, _ = load_latent_action_model('checkpoints/latent_action/best.pt', device)
    model.to(device)
    model.eval()
    if device.type == 'cuda':
        try:
            model = torch.compile(model)
        except Exception:
            pass
    env = AtariBreakoutEnv(return_rgb=True)
    n_actions = env.action_space.n
    agent = RandomAgent(n_actions)
    collected = []
    np.random.seed(seed)
    torch.manual_seed(seed)
    episode = 0
    pbar = tqdm(total=n_pairs, desc='Collecting (action, latent_code) pairs')
    try:
        while len(collected) < n_pairs:
            obs, _ = env.reset()
            frame_t = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            done = False
            steps = 0
            while not done and steps < max_steps_per_episode and len(collected) < n_pairs:
                action = agent.select_action()
                next_obs, reward, terminated, truncated, info = env.step(action)
                frame_tp1 = torch.from_numpy(next_obs).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                try:
                    with torch.no_grad():
                        frame_t = frame_t.to(device)
                        frame_tp1 = frame_tp1.to(device)
                        _, indices, *_ = model(frame_t, frame_tp1)
                except Exception as e:
                    print(f"Error during model call: {e}")
                    print(f"frame_t shape: {frame_t.shape}, dtype: {frame_t.dtype}")
                    print(f"frame_tp1 shape: {frame_tp1.shape}, dtype: {frame_tp1.dtype}")
                    raise
                latent_code = flatten_latent_indices(indices.cpu().squeeze(0))
                collected.append({
                    'action': int(action),
                    'latent_code': latent_code
                })
                pbar.update(1)
                frame_t = frame_tp1.cpu()
                steps += 1
                if terminated or truncated:
                    done = True
            episode += 1
    except KeyboardInterrupt:
        print('Interrupted. Saving collected data...')
    finally:
        env.close()
        pbar.close()
        with open(out_path, 'w') as f:
            json.dump(collected, f, indent=2)
        print(f"Saved {len(collected)} pairs to {out_path}")
        # Log summary statistics
        actions = [d['action'] for d in collected]
        print("Action distribution:", {a: actions.count(a) for a in set(actions)})
        print("Example latent code:", collected[0]['latent_code'] if collected else None)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Collect (action, latent_code) pairs for latent action mapping.')
    parser.add_argument('--out', type=str, default='data/actions/action_latent_pairs.json')
    parser.add_argument('--n_pairs', type=int, default=100_000)
    parser.add_argument('--max_steps_per_episode', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    collect_action_latent_pairs(
        out_path=args.out,
        n_pairs=args.n_pairs,
        max_steps_per_episode=args.max_steps_per_episode,
        seed=args.seed
    )

if __name__ == '__main__':
    main() 