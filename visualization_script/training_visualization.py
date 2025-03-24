import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt

# Define the TensorBoard log directory
log_dir = "/kaggle/working/dqn_doubledunk_tensorboard/DQN_1"

# Function to extract training rewards from TensorBoard logs
def extract_tb_data(log_dir):
    data = {"step": [], "reward": []}

    # Iterate over event files in the log directory
    for event_file in os.listdir(log_dir):
        event_path = os.path.join(log_dir, event_file)

        # Read and extract scalar values from the logs
        for event in tf.compat.v1.train.summary_iterator(event_path):
            for value in event.summary.value:
                if value.tag == "rollout/ep_rew_mean":  # Extract reward values
                    data["step"].append(event.step)
                    data["reward"].append(value.simple_value)

    return pd.DataFrame(data)

# Extract and save training data
df = extract_tb_data(log_dir)
df.to_csv("progress_agent.csv", index=False)

# Display first few rows of extracted data
print(df.head())

# Plot training rewards over time
plt.figure(figsize=(10, 5))
plt.plot(df["step"], df["reward"], label="Episode Reward", color="blue")
plt.xlabel("Training Steps")
plt.ylabel("Mean Reward")
plt.title("Training Rewards Over Time (DoubleDunk-v5)")
plt.legend()
plt.grid()
plt.show()
