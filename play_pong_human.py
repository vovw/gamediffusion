"""
Human-playable Pong for data collection!

Controls:
- UP ARROW: Move paddle up  
- DOWN ARROW: Move paddle down
- SPACE: Fire/Start
- ESC or Q: Quit and save data

The game will automatically save all frames and actions to expand your dataset!
"""

import pygame
import numpy as np
import os
import cv2
import json
from pong_env import AtariPongEnv
from pathlib import Path
import time

# Game config
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 840
FPS = 60
SAVE_DATA = True

class HumanPongPlayer:
    def __init__(self, data_dir='data/pong_trajectories'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Human Pong - Data Collection")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Initialize Pong environment
        self.env = AtariPongEnv(return_rgb=True)
        
        # Action mapping (Pong actions)
        self.action_map = {
            pygame.K_UP: 2,      # UP
            pygame.K_DOWN: 3,    # DOWN  
            pygame.K_SPACE: 1,   # FIRE
        }
        
        # Data collection
        self.episode_count = self._get_next_episode_number()
        self.frames = []
        self.actions = []
        self.rewards = []
        self.total_episodes_played = 0
        
        print("\n🏓 Human Pong Data Collector")
        print("=" * 40)
        print("Controls:")
        print("  UP ARROW    - Move paddle up")
        print("  DOWN ARROW  - Move paddle down") 
        print("  SPACE       - Fire/Start")
        print("  ESC or Q    - Quit and save")
        print("=" * 40)
        print(f"Starting at episode: {self.episode_count}")
        print("Let's collect some data! 🎮\n")

    def _get_next_episode_number(self):
        """Find the next episode number to avoid overwriting."""
        existing_episodes = list(self.data_dir.glob('episode_*'))
        if not existing_episodes:
            return 0
        
        episode_numbers = []
        for ep_dir in existing_episodes:
            try:
                num = int(ep_dir.name.split('_')[1])
                episode_numbers.append(num)
            except (ValueError, IndexError):
                continue
        
        return max(episode_numbers) + 1 if episode_numbers else 0

    def save_episode_data(self):
        """Save the collected episode data."""
        if not self.frames:
            return
            
        episode_dir = self.data_dir / f'episode_{self.episode_count:05d}'
        episode_dir.mkdir(exist_ok=True)
        
        # Save frames as PNG files
        print(f"💾 Saving {len(self.frames)} frames...")
        for i, frame in enumerate(self.frames):
            frame_path = episode_dir / f'frame_{i:05d}.png'
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Save metadata
        metadata = {
            'actions': self.actions,
            'rewards': self.rewards,
            'total_reward': sum(self.rewards),
            'length': len(self.frames),
            'episode': self.episode_count
        }
        
        with open(episode_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Episode {self.episode_count} saved!")
        print(f"   Frames: {len(self.frames)}, Total reward: {sum(self.rewards):.1f}")
        
        # Reset for next episode
        self.episode_count += 1
        self.total_episodes_played += 1
        self.frames = []
        self.actions = []
        self.rewards = []

    def play(self):
        """Main game loop."""
        running = True
        obs, _ = self.env.reset()
        self.frames = [obs.copy()]
        
        current_action = 0  # NOOP
        game_over = False
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                        running = False
                    elif event.key in self.action_map:
                        current_action = self.action_map[event.key]

            # Check currently pressed keys for continuous movement
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                current_action = 2  # UP
            elif keys[pygame.K_DOWN]:
                current_action = 3  # DOWN
            elif keys[pygame.K_SPACE]:
                current_action = 1  # FIRE
            else:
                current_action = 0  # NOOP
            
            if not game_over:
                # Take action in environment
                obs, reward, terminated, truncated, info = self.env.step(current_action)
                
                # Store data
                self.frames.append(obs.copy())
                self.actions.append(current_action)
                self.rewards.append(reward)
                
                # Check if episode ended
                if terminated or truncated:
                    game_over = True
                    self.save_episode_data()
                    print("🎮 Game Over! Press SPACE to start new episode")
            
            else:
                # Game over - wait for space to restart
                if keys[pygame.K_SPACE]:
                    obs, _ = self.env.reset()
                    self.frames = [obs.copy()]
                    game_over = False
                    print(f"🚀 Starting episode {self.episode_count}")
            
            # Render the game
            self.render(obs, current_action, game_over)
            self.clock.tick(FPS)
        
        # Save final episode if there's data
        if self.frames and not game_over:
            self.save_episode_data()
        
        # Cleanup
        self.env.close()
        pygame.quit()
        
        print(f"\n🎉 Session complete!")
        print(f"Episodes played: {self.total_episodes_played}")
        print(f"Total frames collected: {self._count_total_frames()}")

    def _count_total_frames(self):
        """Count total frames in the dataset."""
        total = 0
        for episode_dir in self.data_dir.glob('episode_*'):
            frames = list(episode_dir.glob('frame_*.png'))
            total += len(frames)
        return total

    def render(self, obs, action, game_over):
        """Render the game to the pygame window."""
        # Scale and display the Pong frame
        if obs is not None:
            # Convert from RGB to pygame surface
            frame_surface = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
            # Scale to fit window (keeping some space for UI)
            scaled_surface = pygame.transform.scale(frame_surface, (WINDOW_WIDTH, WINDOW_HEIGHT - 100))
            
            # Fill background
            self.screen.fill((0, 0, 0))
            self.screen.blit(scaled_surface, (0, 0))
            
            # Display UI info
            y_offset = WINDOW_HEIGHT - 90
            
            # Current action
            action_names = ['NOOP', 'FIRE', 'UP', 'DOWN', 'UPFIRE', 'DOWNFIRE']
            action_text = f"Action: {action_names[action] if action < len(action_names) else action}"
            action_surface = self.font.render(action_text, True, (255, 255, 255))
            self.screen.blit(action_surface, (10, y_offset))
            
            # Episode info
            episode_text = f"Episode: {self.episode_count} | Frames: {len(self.frames)}"
            episode_surface = self.font.render(episode_text, True, (255, 255, 255))
            self.screen.blit(episode_surface, (10, y_offset + 30))
            
            # Game over message
            if game_over:
                game_over_text = "GAME OVER - Press SPACE for new episode"
                text_surface = self.font.render(game_over_text, True, (255, 255, 0))
                text_rect = text_surface.get_rect(center=(WINDOW_WIDTH//2, y_offset + 60))
                self.screen.blit(text_surface, text_rect)
        
        pygame.display.flip()

def main():
    player = HumanPongPlayer()
    player.play()

if __name__ == "__main__":
    main() 