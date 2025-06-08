import gymnasium as gym
import numpy as np
import cv2
from typing import Tuple, Dict, Any

class AtariPongEnv:
    """
    Wrapper for Atari Pong environment with preprocessing.
    If return_rgb=True, returns original RGB frames (210, 160, 3) instead of preprocessed grayscale (84, 84).
    """
    def __init__(self, return_rgb: bool = False):
        self.env = gym.make(
            "ALE/Pong-v5",
            render_mode='rgb_array',
            frameskip=4,
            repeat_action_probability=0.0,
            full_action_space=False
        )
        self.action_space = self.env.action_space
        print(f"Action space: {self.action_space}")
        print(f"Action meanings: {self.env.unwrapped.get_action_meanings()}")
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )
        self.lives = 6  # Pong starts with 6 lives
        self.was_real_done = True
        self.living_penalty = -0.005
        self.life_loss_penalty = 0.0
        self.return_rgb = return_rgb

    def preprocess_observation(self, obs: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset()
        self.lives = info.get('lives', 6)
        if self.return_rgb:
            return obs, info
        processed_obs = self.preprocess_observation(obs)
        return processed_obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        lives = info.get('lives', 0)
        life_loss_reward = 0.0
        if lives < self.lives:
            life_loss_reward = self.life_loss_penalty
            self.lives = lives
        shaped_reward = reward + self.living_penalty + life_loss_reward
        if self.return_rgb:
            return obs, shaped_reward, terminated, truncated, info
        processed_obs = self.preprocess_observation(obs)
        return processed_obs, shaped_reward, terminated, truncated, info

    def close(self):
        self.env.close() 