"""
Interactive Pong World Model - Clean Interface for Neural Pong

A demure and elegant way to interact with the trained Pong world model.
This uses the autoencoder + predictor architecture from your existing codebase.

Controls:
    SPACE: Start/Step prediction
    ESC: Quit
    R: Reset to random frame
    S: Save current frame
    
Features:
    - Real-time prediction rollouts
    - Clean, minimal UI
    - Auto-rollout mode
"""

import os
import sys
import torch
import numpy as np
import pygame
import cv2
import time
import argparse
from pathlib import Path
from datetime import datetime

from world_model import (
    PongDataset, build_encoder, build_decoder, build_predictor,
    get_device
)

class PongWorldPlayer:
    """Clean, elegant interface for Pong world model interaction."""
    
    def __init__(self, window_size=(640, 900), fps=10):
        self.window_size = window_size
        self.fps = fps
        self.device = get_device()
        
        # UI state
        self.frame_count = 0
        self.session_start = datetime.now()
        self.auto_rollout = False
        self.rollout_step = 0
        
        # Game state
        self.current_frame = None
        self.current_latent = None
        
        self._load_models()
        self._setup_pygame()
        self._load_initial_frame()
        
    def _load_models(self):
        """Load trained models or train if needed."""
        print("🔄 Loading Pong world model...")
        
        # Check if models exist
        model_dir = Path("models")
        encoder_path = model_dir / "encoder.pt"
        decoder_path = model_dir / "decoder.pt"
        predictor_path = model_dir / "predictor.pt"
        
        if encoder_path.exists() and decoder_path.exists() and predictor_path.exists():
            # Load existing models
            print("📁 Loading existing models...")
            self.encoder = build_encoder().to(self.device)
            self.decoder = build_decoder().to(self.device)
            self.predictor = build_predictor().to(self.device)
            
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
            self.predictor.load_state_dict(torch.load(predictor_path, map_location=self.device))
            
            print("✅ Models loaded successfully")
        else:
            # Train models
            print("🏋️ Training models (this will take a few minutes)...")
            self._train_models()
        
        self.encoder.eval()
        self.decoder.eval()
        self.predictor.eval()
        
        # Compile for performance
        if self.device.type == 'cuda':
            try:
                self.encoder = torch.compile(self.encoder)
                self.decoder = torch.compile(self.decoder)
                self.predictor = torch.compile(self.predictor)
                print("✅ Models compiled for CUDA")
            except Exception as e:
                print(f"⚠️  Compilation failed: {e}")
        
        print(f"✅ Models ready on {self.device}")
    
    def _train_models(self):
        """Train the world model from scratch."""
        from world_model import train_autoencoder, train_predictor
        from torch.utils.data import DataLoader
        
        # Load dataset
        print("📚 Loading dataset...")
        dataset = PongDataset(data_dir='data/pong_trajectories', N=100000)  # Limit for speed
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        print(f"Dataset size: {len(dataset)}")
        
        # Train autoencoder
        print("🔄 Training autoencoder...")
        self.encoder, self.decoder = train_autoencoder(dataloader, steps=1000)
        
        # Train predictor
        print("🔄 Training predictor...")
        self.predictor = build_predictor().to(self.device)
        self.predictor = train_predictor(self.encoder, self.predictor, dataloader, steps=1000)
        
        # Save models
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        torch.save(self.encoder.state_dict(), model_dir / "encoder.pt")
        torch.save(self.decoder.state_dict(), model_dir / "decoder.pt")
        torch.save(self.predictor.state_dict(), model_dir / "predictor.pt")
        
        print("💾 Models saved to models/ directory")
    
    def _setup_pygame(self):
        """Initialize pygame with clean styling."""
        pygame.init()
        pygame.font.init()
        
        self.window = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Interactive Pong World Model")
        
        # Fonts for UI
        self.font_large = pygame.font.SysFont('Consolas', 24, bold=True)
        self.font_medium = pygame.font.SysFont('Consolas', 18)
        self.font_small = pygame.font.SysFont('Consolas', 14)
        
        self.clock = pygame.time.Clock()
        
        # Colors
        self.colors = {
            'bg': (15, 15, 25),
            'text': (220, 220, 230),
            'accent': (100, 200, 255),
            'warning': (255, 180, 100),
            'success': (100, 255, 150)
        }
    
    def _load_initial_frame(self):
        """Load a random initial frame from dataset."""
        print("🎮 Loading initial frame...")
        dataset = PongDataset(data_dir='data/pong_trajectories', N=10000)
        
        # Get random frame
        idx = np.random.randint(len(dataset))
        self.current_frame = dataset[idx].unsqueeze(0).to(self.device)  # Add batch dim
        
        # Get initial latent
        with torch.no_grad():
            self.current_latent = self.encoder(self.current_frame)
        
        print("✅ Initial frame loaded")
    
    def _predict_next_frame(self):
        """Predict next frame using the world model."""
        with torch.no_grad():
            # Predict next latent
            next_latent = self.predictor(self.current_latent)
            
            # Decode to frame
            next_frame = self.decoder(next_latent)
            
            # Update state
            self.current_frame = next_frame
            self.current_latent = next_latent
            
            return next_frame
    
    def _frame_to_surface(self, frame_tensor):
        """Convert frame tensor to pygame surface."""
        frame_np = frame_tensor.squeeze().cpu().numpy()  # Remove batch and channel dims
        frame_np = (frame_np * 255).astype(np.uint8)
        
        # Resize from 84x84 to larger size for visibility
        frame_np = cv2.resize(frame_np, (400, 400), interpolation=cv2.INTER_NEAREST)
        
        # Convert to RGB (duplicate grayscale across channels)
        frame_rgb = np.stack([frame_np] * 3, axis=-1)
        
        # Create surface
        surface = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
        
        return surface
    
    def _draw_ui(self):
        """Draw clean UI overlay."""
        # Clear UI area
        ui_rect = pygame.Rect(0, self.window_size[1] - 200, self.window_size[0], 200)
        pygame.draw.rect(self.window, self.colors['bg'], ui_rect)
        pygame.draw.line(self.window, self.colors['accent'], 
                        (0, self.window_size[1] - 200), 
                        (self.window_size[0], self.window_size[1] - 200), 2)
        
        # Auto-rollout status
        if self.auto_rollout:
            status_text = f"Auto-rollout active (step {self.rollout_step})"
            status_color = self.colors['success']
        else:
            status_text = "Manual mode"
            status_color = self.colors['text']
        
        status_surface = self.font_medium.render(status_text, True, status_color)
        self.window.blit(status_surface, (20, self.window_size[1] - 180))
        
        # Controls hint
        controls = [
            "SPACE: Predict next frame",
            "A: Toggle auto-rollout",
            "R: Reset to random frame",
            "S: Save frame",
            "ESC: Quit"
        ]
        
        for i, control in enumerate(controls):
            control_surface = self.font_small.render(control, True, self.colors['text'])
            self.window.blit(control_surface, (20, self.window_size[1] - 150 + i * 20))
        
        # Frame counter
        frame_text = f"Frame: {self.frame_count}"
        frame_surface = self.font_small.render(frame_text, True, self.colors['text'])
        frame_rect = frame_surface.get_rect()
        frame_rect.right = self.window_size[0] - 20
        frame_rect.y = self.window_size[1] - 180
        self.window.blit(frame_surface, frame_rect)
    
    def _save_frame(self):
        """Save current frame to disk."""
        os.makedirs('saved_frames', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"saved_frames/pong_frame_{timestamp}.png"
        
        frame_np = self.current_frame.squeeze().cpu().numpy()
        frame_np = (frame_np * 255).astype(np.uint8)
        
        cv2.imwrite(filename, frame_np)
        print(f"💾 Saved frame: {filename}")
    
    def _reset_game(self):
        """Reset to new random frame."""
        self._load_initial_frame()
        self.frame_count = 0
        self.rollout_step = 0
        print("🔄 Reset to random frame")
    
    def run(self):
        """Main game loop."""
        print("\n" + "="*50)
        print("🏓 Interactive Pong World Model")
        print("="*50)
        print("Controls:")
        print("  SPACE - Predict next frame")
        print("  A - Toggle auto-rollout")
        print("  R - Reset to random frame")
        print("  S - Save current frame")
        print("  ESC - Quit")
        print("="*50)
        
        running = True
        last_auto_step = time.time()
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self._predict_next_frame()
                        self.frame_count += 1
                        if self.auto_rollout:
                            self.rollout_step += 1
                    elif event.key == pygame.K_a:
                        self.auto_rollout = not self.auto_rollout
                        self.rollout_step = 0
                        print(f"Auto-rollout: {'ON' if self.auto_rollout else 'OFF'}")
                    elif event.key == pygame.K_r:
                        self._reset_game()
                    elif event.key == pygame.K_s:
                        self._save_frame()
            
            # Auto-rollout
            if self.auto_rollout and time.time() - last_auto_step > 0.5:  # 2 FPS for auto
                self._predict_next_frame()
                self.frame_count += 1
                self.rollout_step += 1
                last_auto_step = time.time()
            
            # Render
            self.window.fill(self.colors['bg'])
            
            # Draw game frame (centered)
            game_surface = self._frame_to_surface(self.current_frame)
            game_rect = game_surface.get_rect()
            game_rect.centerx = self.window_size[0] // 2
            game_rect.y = 50
            self.window.blit(game_surface, game_rect)
            
            # Draw UI
            self._draw_ui()
            
            pygame.display.flip()
            self.clock.tick(self.fps)
        
        # Cleanup
        pygame.quit()
        print("👋 Goodbye!")


def main():
    parser = argparse.ArgumentParser(description="Interactive Pong World Model Player")
    parser.add_argument('--fps', type=int, default=10, 
                        help='Target FPS (default: 10)')
    parser.add_argument('--window-width', type=int, default=640, 
                        help='Window width (default: 640)')
    parser.add_argument('--window-height', type=int, default=900, 
                        help='Window height (default: 900)')
    
    args = parser.parse_args()
    
    try:
        player = PongWorldPlayer(
            window_size=(args.window_width, args.window_height),
            fps=args.fps
        )
        player.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 