"""
Minimal Neural-Atari world model for Pong.
Trains autoencoder + predictor in latent space for fast video generation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import os
import glob
import cv2
import numpy as np
from pathlib import Path


class PongDataset(Dataset):
    """Custom dataset for loading Pong frames from collected trajectories."""
    def __init__(self, data_dir: str = 'data/pong_trajectories', N: int = None):
        self.data_dir = Path(data_dir)
        
        # Find all frame files
        self.frame_paths = []
        for episode_dir in sorted(self.data_dir.glob('episode_*')):
            frame_paths = sorted(episode_dir.glob('frame_*.png'))
            self.frame_paths.extend(frame_paths)
            
        # Optionally limit dataset size
        if N is not None:
            self.frame_paths = self.frame_paths[:N]
    
    def __len__(self) -> int:
        return len(self.frame_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        # Load frame
        frame = cv2.imread(str(self.frame_paths[idx]), cv2.IMREAD_GRAYSCALE)
        
        # Resize to 84x84 as expected by the encoder
        frame = cv2.resize(frame, (84, 84))
        
        # Convert to tensor and normalize
        frame = torch.from_numpy(frame).float() / 255.0
        frame = frame.unsqueeze(0)  # Add channel dimension: (1, 84, 84)
        
        return frame


def build_encoder(latent_dim: int = 128) -> nn.Sequential:
    """Build CNN encoder: 84x84x1 -> latent_dim."""
    return nn.Sequential(
        nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 42x42
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 21x21
        nn.ReLU(),
        nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 10x10
        nn.ReLU(),
        nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 5x5
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(256 * 5 * 5, latent_dim),
    )


def build_decoder(latent_dim: int = 128) -> nn.Sequential:
    """Build CNN decoder: latent_dim -> 84x84x1."""
    return nn.Sequential(
        nn.Linear(latent_dim, 256 * 5 * 5),
        nn.ReLU(),
        nn.Unflatten(1, (256, 5, 5)),
        nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 5x5 -> 10x10
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, output_padding=1),  # 10x10 -> 21x21
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 21x21 -> 42x42
        nn.ReLU(),
        nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 42x42 -> 84x84
        nn.Sigmoid(),
    )


def build_predictor(latent_dim: int = 128) -> nn.Sequential:
    """Build MLP predictor: latent_t -> latent_t+1."""
    return nn.Sequential(
        nn.Linear(latent_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, latent_dim),
    )


def ae_loss(x_real: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    """Autoencoder reconstruction loss."""
    return F.mse_loss(x_recon, x_real)


def pred_loss(z_real: torch.Tensor, z_pred: torch.Tensor) -> torch.Tensor:
    """Predictor loss in latent space."""
    return F.mse_loss(z_pred, z_real)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_autoencoder(dataloader: DataLoader, steps: int) -> Tuple[nn.Module, nn.Module]:
    """Train encoder+decoder for reconstruction."""
    device = get_device()
    print(f"Using device: {device}")
    
    encoder = build_encoder().to(device)
    decoder = build_decoder().to(device)
    
    # Compile models for better performance
    if device.type == "cuda":
        encoder = torch.compile(encoder)
        decoder = torch.compile(decoder)
    
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3
    )
    
    encoder.train()
    decoder.train()
    
    data_iter = iter(dataloader)
    for step in range(steps):
        try:
            x = next(data_iter).to(device)
        except StopIteration:
            data_iter = iter(dataloader)
            x = next(data_iter).to(device)
        
        z = encoder(x)
        x_recon = decoder(z)
        loss = ae_loss(x, x_recon)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"AE step {step}, loss: {loss.item():.4f}")
    
    return encoder, decoder


def train_predictor(encoder: nn.Module, predictor: nn.Module, dataloader: DataLoader, steps: int) -> nn.Module:
    """Train predictor in latent space (encoder frozen)."""
    device = get_device()
    
    predictor = predictor.to(device)
    encoder.eval()  # Freeze encoder
    predictor.train()
    
    # Compile predictor for better performance
    if device.type == "cuda":
        predictor = torch.compile(predictor)
    
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    
    data_iter = iter(dataloader)
    for step in range(steps):
        try:
            x = next(data_iter).to(device)
        except StopIteration:
            data_iter = iter(dataloader)
            x = next(data_iter).to(device)
        
        # Get consecutive frames for prediction
        if len(x) < 2:
            continue
            
        x_curr = x[:-1]
        x_next = x[1:]
        
        with torch.no_grad():
            z_curr = encoder(x_curr)
            z_next = encoder(x_next)
        
        z_pred = predictor(z_curr)
        loss = pred_loss(z_next, z_pred)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Pred step {step}, loss: {loss.item():.4f}")
    
    return predictor


def sample(encoder: nn.Module, decoder: nn.Module, predictor: nn.Module, 
          seed_frame: torch.Tensor, horizon: int) -> List[torch.Tensor]:
    """Generate video rollout from seed frame."""
    device = get_device()
    
    encoder.eval()
    decoder.eval()
    predictor.eval()
    
    frames = []
    
    with torch.no_grad():
        # Start with seed frame
        current_frame = seed_frame.to(device)
        z = encoder(current_frame.unsqueeze(0))
        
        for _ in range(horizon):
            # Decode current latent to frame
            frame = decoder(z).squeeze(0)
            frames.append(frame.cpu())
            
            # Predict next latent
            z = predictor(z)
    
    return frames


if __name__ == "__main__":
    # Main training pipeline
    print("Loading Pong dataset...")
    dataset = PongDataset(N=None)  # Use all available data (remove limit)
    # Or use a larger number like:
    # dataset = PongDataset(N=50000)  # 50k frames
    # dataset = PongDataset(N=100000)  # 100k frames
    
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    print("Training autoencoder...")
    encoder, decoder = train_autoencoder(dataloader, steps=500)
    
    print("Building predictor...")
    predictor = build_predictor()
    
    print("Training predictor...")
    predictor = train_predictor(encoder, predictor, dataloader, steps=500)
    
    print("Sampling video...")
    seed_frame = dataset[0]
    frames = sample(encoder, decoder, predictor, seed_frame, horizon=200)
    
    print(f"Generated {len(frames)} frames!")
    print("Done. Use train.ipynb for visualization.") 