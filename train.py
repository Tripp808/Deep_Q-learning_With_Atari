# imports used
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
import matplotlib.pyplot as plt
import ale_py 

# Create the DoubleDunk environment
env = make_atari_env("ALE/DoubleDunk-v5", n_envs=1, seed=42)
env = VecFrameStack(env, n_stack=4) 

# showing important details of the environment
print("===== Environment Details =====")
print(f"Environment ID: {env.envs[0].unwrapped.spec.id}")
print(f"Observation Space: {env.observation_space}")
print(f"Action Space: {env.action_space}")
print(f"Number of Actions: {env.action_space.n}")
print(f"Reward Range: {env.envs[0].unwrapped.reward_range}")  # Print reward range
print(f"Environment Metadata: {env.envs[0].unwrapped.metadata}")

# # this is oprtional but we used it to record videos every 1k steps
env = VecVideoRecorder(env, "videos/", record_video_trigger=lambda x: x % 1000 == 0, video_length=1000)

# the DQN agent with CNNPolicy
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
    tensorboard_log="./dqn_doubledunk_tensorboard/",
)

# Train the agent
model.learn(total_timesteps=2_000_000)

# Save the trained model

# Save the trained model
MODEL_PATH = "dqn_doubledunk_model"
model.save(MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")

# Close the environment
env.close()
