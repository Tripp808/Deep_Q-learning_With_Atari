import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import imageio
import numpy as np
import time

# Load the trained model
model_path = "/kaggle/working/dqn_doubledunk_model.zip"  # Path to saved model
model = DQN.load(model_path)

# Verify model loading
print("✅ Model loaded successfully!")
print("🔹 Model Policy:", model.policy)

# Setup environment for evaluation
env = make_atari_env("ALE/DoubleDunk-v5", n_envs=1, seed=42)
env = VecFrameStack(env, n_stack=4)

# Print action space for debugging
print("🎮 Action Space:", env.action_space)

# Initialize observation and variables
obs = env.reset()
frames = []
total_reward = 0
episode = 1

# Control frame rate (slow down gameplay)
target_fps = 60  
frame_delay = 1.0 / target_fps  

# Run the agent for 10,000 steps
max_steps = 10000
for step in range(max_steps):
    action, _states = model.predict(obs, deterministic=False)  # Allow exploration
    obs, rewards, dones, info = env.step(action)  # Take a step
    total_reward += rewards[0]  # Accumulate rewards

    # Check if agent is on offense or defense
    role = "Offense" if rewards[0] > 0 else "Defense"

    # Render frame and save for video recording
    frame = env.render(mode="rgb_array")
    frames.append(frame)
    
    # Slowing down gameplay for better visualization
    time.sleep(frame_delay)

    # Handle end of episode
    if dones.any():
        print(f"🏆 Episode {episode} finished! Total Reward: {total_reward}")
        obs = env.reset()  # Reset for next episode
        print(f"🔄 Starting Episode {episode + 1}...")
        total_reward = 0
        episode += 1

# Save recorded gameplay as a video
video_path = "/kaggle/working/d_playback.mp4"
imageio.mimsave(
    video_path,
    frames,
    fps=target_fps,  
    quality=10,  
    codec="libx264",  
    pixelformat="yuv420p"
)

print(f"✅ Gameplay video saved to {video_path}")

# Close environment
env.close()
