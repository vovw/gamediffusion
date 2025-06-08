"""
Interactive World Model - A Clean Interface for Neural Breakout

A demure and elegant way to interact with the trained world model.
This provides a smooth, responsive interface for exploring the neural game world.

Controls:
    SPACE: Fire/Start
    ← →: Move paddle
    ESC: Quit
    R: Reset to initial state
    T: Toggle temperature display
    S: Save current frame
    
Features:
    - Smooth real-time interaction
    - Visual feedback for actions
    - Temperature control via mouse wheel
    - Clean, minimal UI
    - Auto-save interesting moments
"""

import os
import sys
import torch
import numpy as np
import pygame
from PIL import Image
import time
import argparse
from pathlib import Path
import json
from datetime import datetime

from latent_action_model import load_latent_action_model, ActionStateToLatentMLP

class WorldModelPlayer:
    """Clean, elegant interface for world model interaction."""
    
    def __init__(self, temperature=0.01, window_size=(640, 900), fps=20):
        self.temperature = temperature
        self.window_size = window_size
        self.fps = fps
        self.device = self._get_device()
        
        # UI state
        self.show_temperature = True
        self.action_feedback = ""
        self.last_action_time = 0
        self.frame_count = 0
        self.session_start = datetime.now()
        
        # Game state
        self.current_frame = None
        self.frame_history = []
        self.action_history = []
        
        self._load_models()
        self._setup_pygame()
        self._load_initial_frame()
        
    def _get_device(self):
        """Auto-detect best device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _load_models(self):
        """Load and optimize models."""
        print("🔄 Loading world model...")
        self.world_model, _ = load_latent_action_model(
            'checkpoints/latent_action/best.pt', self.device
        )
        self.world_model.to(self.device)
        self.world_model.eval()
        
        print("🔄 Loading action model...")
        self.action_model = ActionStateToLatentMLP().to(self.device)
        ckpt = torch.load(
            'checkpoints/latent_action/action_state_to_latent_best.pt', 
            map_location=self.device
        )
        self.action_model.load_state_dict(ckpt['model_state_dict'])
        self.action_model.eval()
        
        # Compile for performance
        if self.device.type == 'cuda':
            try:
                self.world_model = torch.compile(self.world_model)
                self.action_model = torch.compile(self.action_model)
                print("✅ Models compiled for CUDA")
            except Exception as e:
                print(f"⚠️  Compilation failed: {e}")
        
        print(f"✅ Models loaded on {self.device}")
    
    def _setup_pygame(self):
        """Initialize pygame with clean styling."""
        pygame.init()
        pygame.font.init()
        
        self.window = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Interactive World Model")
        
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
        """Load the initial game frame."""
        init_img = Image.open('data/0.png').convert('RGB')
        init_frame_np = np.array(init_img, dtype=np.float32) / 255.0
        self.current_frame = torch.from_numpy(init_frame_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Initialize frame history for action model
        self.last2_frames = [self.current_frame.clone(), self.current_frame.clone()]
        
        print("✅ Initial frame loaded")
    
    def _action_to_onehot(self, action_idx):
        """Convert action index to one-hot tensor."""
        onehot = torch.zeros(1, 4, device=self.device)
        onehot[0, action_idx] = 1.0
        return onehot
    
    def _generate_next_frame(self, action_idx):
        """Generate next frame using the world model."""
        with torch.no_grad():
            # Prepare inputs
            stacked_frames = torch.cat(self.last2_frames, dim=1)
            onehot = self._action_to_onehot(action_idx)
            
            # Predict latent codes
            logits = self.action_model(onehot, stacked_frames)
            indices = self.action_model.sample_latents(logits, temperature=self.temperature)
            
            # Reshape and get embeddings
            indices = indices.view(1, 5, 7)
            embeddings = self.world_model.vq.embeddings
            indices = indices.to(embeddings.weight.device)
            quantized = embeddings(indices)
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            
            # Generate frame
            frame_in = self.current_frame.permute(0, 1, 3, 2)
            next_frame = self.world_model.decoder(quantized, frame_in)
            next_frame = next_frame.permute(0, 1, 3, 2)
            
            # Update state
            self.last2_frames[0] = self.last2_frames[1]
            self.last2_frames[1] = next_frame.clone()
            self.current_frame = next_frame.clone()
            
            return next_frame
    
    def _frame_to_surface(self, frame_tensor):
        """Convert frame tensor to pygame surface."""
        frame_np = frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
        
        # Create surface
        surface = pygame.surfarray.make_surface(np.transpose(frame_np, (1, 0, 2)))
        
        # Scale to fit window (leave space for UI)
        game_height = self.window_size[1] - 100
        scaled_surface = pygame.transform.scale(surface, (self.window_size[0], game_height))
        
        return scaled_surface
    
    def _draw_ui(self):
        """Draw clean UI overlay."""
        # Clear UI area
        ui_rect = pygame.Rect(0, self.window_size[1] - 100, self.window_size[0], 100)
        pygame.draw.rect(self.window, self.colors['bg'], ui_rect)
        pygame.draw.line(self.window, self.colors['accent'], 
                        (0, self.window_size[1] - 100), 
                        (self.window_size[0], self.window_size[1] - 100), 2)
        
        # Action feedback
        if self.action_feedback and time.time() - self.last_action_time < 0.5:
            action_text = self.font_medium.render(self.action_feedback, True, self.colors['accent'])
            self.window.blit(action_text, (20, self.window_size[1] - 90))
        
        # Temperature display
        if self.show_temperature:
            temp_text = f"Temperature: {self.temperature:.3f}"
            temp_surface = self.font_small.render(temp_text, True, self.colors['text'])
            self.window.blit(temp_surface, (20, self.window_size[1] - 65))
        
        # Controls hint
        controls = "SPACE: Fire  ←→: Move  R: Reset  ESC: Quit"
        controls_surface = self.font_small.render(controls, True, self.colors['text'])
        controls_rect = controls_surface.get_rect()
        controls_rect.centerx = self.window_size[0] // 2
        controls_rect.y = self.window_size[1] - 40
        self.window.blit(controls_surface, controls_rect)
        
        # Frame counter
        frame_text = f"Frame: {self.frame_count}"
        frame_surface = self.font_small.render(frame_text, True, self.colors['text'])
        frame_rect = frame_surface.get_rect()
        frame_rect.right = self.window_size[0] - 20
        frame_rect.y = self.window_size[1] - 65
        self.window.blit(frame_surface, frame_rect)
    
    def _handle_action(self, action_idx, action_name):
        """Handle an action and provide feedback."""
        self._generate_next_frame(action_idx)
        self.action_feedback = f"→ {action_name}"
        self.last_action_time = time.time()
        self.frame_count += 1
        
        # Store in history
        self.action_history.append((action_idx, action_name, time.time()))
        
        # Limit history size
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]
    
    def _save_frame(self):
        """Save current frame to disk."""
        os.makedirs('saved_frames', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"saved_frames/frame_{timestamp}.png"
        
        frame_np = self.current_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
        
        Image.fromarray(frame_np).save(filename)
        print(f"💾 Saved frame: {filename}")
        
        self.action_feedback = "Frame saved!"
        self.last_action_time = time.time()
    
    def _reset_game(self):
        """Reset to initial state."""
        self._load_initial_frame()
        self.frame_count = 0
        self.action_feedback = "Reset!"
        self.last_action_time = time.time()
        print("🔄 Game reset")
    
    def run(self):
        """Main game loop."""
        print("\n" + "="*50)
        print("🎮 Interactive World Model")
        print("="*50)
        print("Controls:")
        print("  SPACE - Fire/Start")
        print("  ← → - Move paddle")
        print("  R - Reset")
        print("  T - Toggle temperature display")
        print("  S - Save current frame")
        print("  Mouse wheel - Adjust temperature")
        print("  ESC - Quit")
        print("="*50)
        
        action_map = {
            pygame.K_SPACE: (1, "FIRE"),
            pygame.K_RIGHT: (2, "RIGHT"), 
            pygame.K_LEFT: (3, "LEFT"),
            pygame.K_PERIOD: (0, "NOOP")
        }
        
        running = True
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in action_map:
                        action_idx, action_name = action_map[event.key]
                        self._handle_action(action_idx, action_name)
                    elif event.key == pygame.K_r:
                        self._reset_game()
                    elif event.key == pygame.K_t:
                        self.show_temperature = not self.show_temperature
                    elif event.key == pygame.K_s:
                        self._save_frame()
                
                elif event.type == pygame.MOUSEWHEEL:
                    # Adjust temperature with mouse wheel
                    delta = event.y * 0.001
                    self.temperature = max(0.001, min(2.0, self.temperature + delta))
            
            # Check for held keys (for continuous movement)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self._handle_action(3, "LEFT")
            elif keys[pygame.K_RIGHT]:
                self._handle_action(2, "RIGHT")
            elif keys[pygame.K_SPACE]:
                self._handle_action(1, "FIRE")
            else:
                # Default to NOOP if no keys pressed
                if time.time() - self.last_action_time > 0.1:  # Slight delay for NOOP
                    self._handle_action(0, "NOOP")
            
            # Render
            self.window.fill(self.colors['bg'])
            
            # Draw game frame
            game_surface = self._frame_to_surface(self.current_frame)
            self.window.blit(game_surface, (0, 0))
            
            # Draw UI
            self._draw_ui()
            
            pygame.display.flip()
            self.clock.tick(self.fps)
        
        # Cleanup
        self._save_session_summary()
        pygame.quit()
    
    def _save_session_summary(self):
        """Save session statistics."""
        duration = (datetime.now() - self.session_start).total_seconds()
        
        summary = {
            'session_start': self.session_start.isoformat(),
            'duration_seconds': duration,
            'total_frames': self.frame_count,
            'total_actions': len(self.action_history),
            'final_temperature': self.temperature,
            'device_used': str(self.device),
            'action_breakdown': {}
        }
        
        # Count actions
        for action_idx, action_name, _ in self.action_history:
            summary['action_breakdown'][action_name] = summary['action_breakdown'].get(action_name, 0) + 1
        
        os.makedirs('session_logs', exist_ok=True)
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        filename = f"session_logs/session_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Session summary saved: {filename}")
        print(f"⏱️  Duration: {duration:.1f}s")
        print(f"🎮 Frames generated: {self.frame_count}")


def main():
    parser = argparse.ArgumentParser(description="Interactive World Model Player")
    parser.add_argument('--temperature', type=float, default=0.01, 
                        help='Initial sampling temperature (default: 0.01)')
    parser.add_argument('--fps', type=int, default=20, 
                        help='Target FPS (default: 20)')
    parser.add_argument('--window-width', type=int, default=640, 
                        help='Window width (default: 640)')
    parser.add_argument('--window-height', type=int, default=900, 
                        help='Window height (default: 900)')
    
    args = parser.parse_args()
    
    try:
        player = WorldModelPlayer(
            temperature=args.temperature,
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