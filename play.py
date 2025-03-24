import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import time
import os
import ale_py

# Load the trained model
model_path = "dqn_breakout_model.zip" 
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model not found at {model_path}. Train the model first.")

model = DQN.load(model_path)
print("✅ Model loaded successfully!")

# Set up the environment for live rendering
env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=1, seed=42)
env = VecFrameStack(env, n_stack=4)

# Printing action space for debugging
print("🎮 Action Space:", env.action_space)

# observation and variables
obs = env.reset()
total_reward = 0
episode = 1

# Frame rate control 
target_fps = 60  
frame_delay = 1.0 / target_fps  

#  with real-time rendering
max_steps = 10000  
for step in range(max_steps):
    action, _states = model.predict(obs, deterministic=True)  # GreedyQPolicy
    obs, rewards, dones, _ = env.step(action)  
    total_reward += rewards[0]

    # Rendering the game live
    env.render(mode="human")

    # a delay to maintain FPS
    time.sleep(frame_delay)

    if dones.any():
        print(f"🏆 Episode {episode} finished! Total Reward: {total_reward}")
        obs = env.reset()
        print(f"🔄 Starting Episode {episode + 1}...")
        total_reward = 0
        episode += 1

# Close the environment
env.close()
