import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import imageio
import numpy as np
import time

# Load the trained model
model_path = "/kaggle/working/dqn_doubledunk_model.zip" 
model = DQN.load(model_path)

# Verify model loading
print("Model loaded successfully!")
print("Model Policy:", model.policy)

# Set up the environment
env = make_atari_env("ALE/DoubleDunk-v5", n_envs=1, seed=42)
env = VecFrameStack(env, n_stack=4)  # Frame stacking for better performance

# Print action space for debugging
print("Action Space:", env.action_space)

# Initialize observation and variables
obs = env.reset()
frames = []  # To store frames for video recording
total_reward = 0  # for tracking the total reward (score)
episode = 1  # the number of episodes

# Frame rate control
target_fps = 60  # we using this because Atari games typically run at 60 FPS
frame_delay = 1.0 / target_fps  # there'll be a delay between frames in seconds

# Run the agent in the environment
max_steps = 10000  # for a longer gameplay
for step in range(max_steps):
    action, _states = model.predict(obs, deterministic=False)  
    obs, rewards, dones, info = env.step(action)  # To take a step in the environment
    total_reward += rewards[0]  # Accumulated reward

    # Determines if we're on offense or defense
    if rewards[0] > 0:
        role = "Offense"
    else:
        role = "Defense"

    # Render the game as an RGB array and save the frame
    frame = env.render(mode="rgb_array")
    frames.append(frame)

    # Introduce a delay to control the frame rate
    time.sleep(frame_delay)

    # Check if the episode has finished
    if dones.any():
        print(f"Episode {episode} finished! Total Reward: {total_reward}")
        obs = env.reset()  # Reset the environment for the next episode
        print(f"Starting Episode {episode + 1}...")
        total_reward = 0  # Reset the total reward
        episode += 1  # Increment the episode counter

# Save the recorded frames as a video
video_path = "/kaggle/working/d_playback.mp4"  
imageio.mimsave(
    video_path,
    frames,
    fps=target_fps,  # Match the target frame rate
    quality=10,  # Maximum quality
    codec="libx264",  # High-quality codec
    pixelformat="yuv420p",  # Standard pixel format for compatibility
)

print(f"Gameplay video saved to {video_path}")

env.close()