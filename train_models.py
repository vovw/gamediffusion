#!/usr/bin/env python3
"""
Quick training script for Pong world model.
Creates encoder.pt, decoder.pt, and predictor.pt in models/ directory.
"""

import torch
from torch.utils.data import DataLoader
from world_model import (
    PongDataset, train_autoencoder, train_predictor, 
    build_predictor, get_device
)
import os
from pathlib import Path

def main():
    print("🚀 Training Pong World Model")
    print("=" * 40)
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Load dataset
    print("📚 Loading dataset...")
    dataset = PongDataset('data/pong_trajectories', N=100000)  # Limit for speed
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    print(f"Dataset size: {len(dataset)} frames")
    
    # Train autoencoder
    print("\n🎯 Training autoencoder (1000 steps)...")
    encoder, decoder = train_autoencoder(dataloader, steps=1000)
    
    # Train predictor  
    print("\n🔮 Training predictor (1000 steps)...")
    predictor = build_predictor()
    predictor = train_predictor(encoder, predictor, dataloader, steps=1000)
    
    # Save models
    print("\n💾 Saving models...")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    torch.save(encoder.state_dict(), models_dir / "encoder.pt")
    torch.save(decoder.state_dict(), models_dir / "decoder.pt")
    torch.save(predictor.state_dict(), models_dir / "predictor.pt")
    
    print("✅ Training complete!")
    print("📂 Models saved in models/ directory")
    print("\n🎮 Now run: python interactive_pong_world.py")

if __name__ == "__main__":
    main() 