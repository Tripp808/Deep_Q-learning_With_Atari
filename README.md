# Deep Q-Learning: Training and Evaluating an Atari RL Agent

## 📌 Project Overview
This project applies **Deep Q-Networks (DQN)** to train an agent to play **Atari's Double Dunk** using **Stable-Baselines3** and **Gymnasium**. The goal was to develop a reinforcement learning agent, fine-tune its hyperparameters, and evaluate its performance through visualization and analysis.

## 🏆 Contributors
- **Oche David Ankeli** – Trained the models and experimented with different hyperparameters.
- **Aime Magnifique** – Created the visualizations used in performance analysis.
- **Esther Mbabazingwe** – Extracted data for reward tracking and episode analysis.

---

## 🎮 Environment Selection
We selected **Double Dunk** from the **Gymnasium Atari collection** because it offers:
- **Dynamic gameplay mechanics** (offense vs. defense)
- **Challenging AI behavior** that requires strategic decision-making
- **Well-defined scoring system**, making it ideal for reinforcement learning

---

## 📜 Training Scripts
### **1️⃣ First Model (Baseline Model)**
| Parameter | Value |
|-----------|-------|
| **Policy** | `CnnPolicy` |
| **Learning Rate** | `1e-4` |
| **Buffer Size** | `100000` |
| **Batch Size** | `32` |
| **Gamma** | `0.99` |
| **Exploration Fraction** | `0.1` |
| **Final Exploration Rate** | `0.01` |
| **Training Timesteps** | `1,000,000` |

🔹 **Observations:**
- The agent learned basic game movements but lacked strong decision-making skills.
- Slow convergence due to **lower exploration fraction (`0.1`)**, resulting in poor exploration of new strategies.
- The **batch size (`32`)** was relatively small, leading to slower policy updates.

---

### **2️⃣ Second Model (Optimized Model)**
| Parameter | Value |
|-----------|-------|
| **Policy** | `CnnPolicy` |
| **Learning Rate** | `5e-4` |
| **Buffer Size** | `100000` |
| **Batch Size** | `64` |
| **Gamma** | `0.99` |
| **Exploration Fraction** | `0.2` |
| **Final Exploration Rate** | `0.01` |
| **Training Timesteps** | `2,000,000` |

🔹 **Improvements:**
- **Increased learning rate (`5e-4`)** led to faster training without instability.
- **Higher batch size (`64`)** provided better updates per step, improving learning efficiency.
- **Higher exploration fraction (`0.2`)** allowed the agent to explore more, leading to better gameplay strategies.
- **Extended training (`2,000,000` steps)** resulted in a more refined policy.

---

## 🤖 Why We Chose `CnnPolicy` Over `MlpPolicy`
| Policy | Description | Suitability for Atari |
|--------|-------------|------------------|
| **CnnPolicy** | Uses convolutional layers to extract spatial features from images | ✅ Ideal for image-based input like Atari frames |
| **MlpPolicy** | Uses fully connected layers with raw pixel input | ❌ Less efficient for processing game screens |

🔹 **Conclusion:** Atari games involve image-based states, making **CNNs much better at recognizing objects and game states** compared to MLPs.

---

## 📊 Hyperparameter Tuning & Documentation
| Hyperparameters | Observed Behavior |
|----------------|------------------|
| `lr=1e-4`, `gamma=0.99`, `batch=32`, `eps=0.1 → 0.01` | Agent learned but was slow to converge due to small batch size and low exploration. |
| `lr=5e-4`, `gamma=0.99`, `batch=64`, `eps=0.2 → 0.01` | Improved performance, better strategy development due to increased exploration. |
| `lr=1e-3`, `gamma=0.95`, `batch=128`, `eps=0.3 → 0.05` | Faster initial learning, but unstable in later episodes due to high learning rate. |
| `lr=2e-4`, `gamma=0.99`, `batch=64`, `eps=0.15 → 0.01` | Balanced learning but not as effective as `lr=5e-4`. |
| `lr=3e-4`, `gamma=0.98`, `batch=32`, `eps=0.2 → 0.02` | Reasonable learning speed, but required longer training for good results. |

🔹 **Key Findings:**
- A **higher learning rate (`5e-4`)** improved convergence speed.
- **Batch size (`64`)** provided the best balance between computation cost and training efficiency.
- **Exploration fraction (`0.2`)** helped the agent discover better strategies early in training.

---

## 🚀 Challenges We Faced
### **1️⃣ GPU Issues**
- Initially, training on CPU was **too slow** (~3 days for `1M` timesteps).
- We attempted to train on **Google Colab GPUs**, but faced frequent disconnects.
- Eventually, we used a **local GPU (RTX 3090)**, which significantly reduced training time.

### **2️⃣ No Live Rendering**
- Atari environments require **OpenGL** for `env.render(mode="human")`.
- On **headless servers (Google Colab)**, rendering **isn't supported**.
- **Solution:** Instead of live rendering, we **recorded gameplay** using:
  ```python
  env.render(mode="rgb_array")
  ```
  - This allowed us to save videos and review performance later.

---

## 🎬 Evaluation: Running `play.py`
After training, we ran `play.py` to load the trained model and evaluate the agent's gameplay. The agent:
✅ Played multiple episodes
✅ Achieved **higher rewards** compared to the baseline model
✅ Successfully switched between **offense and defense** roles
✅ Gameplay was recorded and saved as `doubledunk_playback.mp4`

---

## 📌 Submission Requirements Checklist
✔ **train.py & play.py** scripts included ✅  
✔ **Trained model saved as `dqn_model.zip`** ✅  
✔ **Hyperparameter tuning table added** ✅  
✔ **Gameplay video recorded & included (`doubledunk_playback.mp4`)** ✅  
✔ **Group contributions documented** ✅  

---

## 🎯 Conclusion
Through **DQN training and hyperparameter tuning**, we developed an Atari agent that successfully plays **Double Dunk**. This project demonstrated **how deep reinforcement learning can be optimized for real-world gameplay strategies**.

🚀 **Future Work:** We could explore **PPO (Proximal Policy Optimization)** and **A2C (Advantage Actor-Critic)** to compare performance against DQN.

---
