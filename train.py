#!/usr/bin/env python3
"""
Breakout DQN Training Script
implementation used in the notebook with enhancements
"""

import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback
import torch

class TrainingMonitor(BaseCallback):
    """Custom callback for tracking training progress"""
    def __init__(self, check_freq: int = 10000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.rewards = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            reward_mean = float(np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer]))
            self.rewards.append(reward_mean)
            if self.verbose > 0:
                print(f"Step {self.num_timesteps}: Avg Reward = {reward_mean:.2f}")
        return True

def setup_environment():
    """Create and configure the Breakout environment"""
    env = make_atari_env(
        "BreakoutNoFrameskip-v4", 
        n_envs=1, 
        seed=42,
        env_kwargs={
            'render_mode': 'rgb_array',
            'frameskip': 1,  # Using frame skipping in wrapper instead
        }
    )
    env = VecFrameStack(env, n_stack=4)
    
    # Video recording every 50,000 steps
    env = VecVideoRecorder(
        env,
        "videos/",
        record_video_trigger=lambda x: x % 50000 == 0,
        video_length=1000,
        name_prefix="dqn-breakout"
    )
    return env

def train_dqn_agent():
    """Main training function matching notebook parameters"""
    # Setup
    env = setup_environment()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Model configuration 
    model = DQN(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=100000,
        batch_size=32,
        gamma=0.99,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10000,
        tensorboard_log="./dqn_breakout_tensorboard/",
        device=device
    )
    
    # Callbacks
    callbacks = [TrainingMonitor(check_freq=10000)]
    
    # Training (1M steps)
    print("\nStarting training...")
    model.learn(
        total_timesteps=1_000_000,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    model.save("dqn_breakout_model")
    print("\nTraining completed. Model saved to 'dqn_breakout_model.zip'")
    
    env.close()
    return model

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("videos", exist_ok=True)
    os.makedirs("dqn_breakout_tensorboard", exist_ok=True)
    
    # Start training
    trained_model = train_dqn_agent()