import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

def load_tensorboard_data(log_dir):
    """Load training metrics from TensorBoard logs"""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    rewards = event_acc.Scalars("rollout/ep_rew_mean")
    steps = [x.step for x in rewards]
    values = [x.value for x in rewards]
    
    return pd.DataFrame({"step": steps, "reward": values})

def plot_training_metrics(df):
    """Plot training metrics"""
    plt.figure(figsize=(12, 6))
    
    # Reward plot
    plt.subplot(1, 2, 1)
    plt.plot(df["step"], df["reward"], color="blue")
    plt.title("Training Rewards")
    plt.xlabel("Steps")
    plt.ylabel("Mean Episode Reward")
    plt.grid(True)
    
# Rolling average
    plt.subplot(1, 2, 2)
    df["rolling_avg"] = df["reward"].rolling(100).mean()
    plt.plot(df["step"], df["rolling_avg"], color="red")
    plt.title("Rolling Average (100 episodes)")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()

if _name_ == "_main_":
    log_dir = "./logs/DQN_1"  
    df = load_tensorboard_data(log_dir)
    df.to_csv("training_metrics.csv", index=False)
    plot_training_metrics(df)