"""
*******************************************************************************
                            [RL_Train_DDPG_HighwayEnv]
*******************************************************************************
File: RL_Train_DDPG_HighwayEnv.py
Author: Farshad Mirzarazi
Date: 28.01.2024
Description:
------------
This source code is part of the author's Dissertation. (Chapter 8)

DDPG  Reinforcement Learning Highway-Env Environment
Training RL DDPG Agent on Highway-Env Environment: Tensorboard Log Files
Disclaimer:
------------
This code is provided "as is," without warranty of any kind.
The author is not responsible for any consequences arising from the use of this code.
*******************************************************************************
"""
# Library Versions:
# -----------------
# gymnasium                 0.29.1
# matplotlib                3.8.2
# numpy                     1.24.3
# stable-baselines3         2.2.1
# ----------------

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3 import DQN, A2C, PPO, DDPG

# Folder to save model, store TF_Logs, and videos
model_dir = "F:/Brunel/02_SourceCodes/HighwayDriving/Highway_Models/HighwayDDPG"
'''
model_dir = "F:/Brunel/02_SourceCodes/HighwayDriving/Highway_Models/HighwayPPO"
model_dir = "F:/Brunel/02_SourceCodes/HighwayDriving/Highway_Models/HighwayDQN"
model_dir = "F:/Brunel/02_SourceCodes/HighwayDriving/Highway_Models/HighwayA2C"
'''
tensorboard_log = f"{model_dir}/tensorboard_logs"
video_folder = f"{model_dir}/videos/"

# Create environment
env = gym.make("highway-fast-v0", render_mode='rgb_array')
env.config["normalized"] = False
env.configure({"action": {"type": "ContinuousAction"}})
env.reset()
env.render()

# Instantiate the agent
model = DDPG('MlpPolicy', env, verbose=1,tensorboard_log=tensorboard_log)

TIMESTEPS = 2e4
for i in range(1, 3):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DDPGlogs",  progress_bar=True)
    # Save the agent
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model.save(f"{model_dir}/{TIMESTEPS * i}")  # save the model every TIMESTEPS steps
# delete trained model to demonstrate loading
del model
# -------------------------------------------------------------------------------------
env = gym.make("highway-fast-v0", render_mode='rgb_array')
env.config["normalized"] = False
env.configure({"action": {"type": "ContinuousAction"}})
env.reset()

# Load the trained agent
vec_env = DummyVecEnv([lambda: env])
model_path = f"{model_dir}/40000.0"
model = DDPG.load(model_path, env=vec_env)

# Record the video starting at the first step
video_length = 1000

# Generate a timestamp string; Use the timestamp in the video file name
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
video_name = f"DDPG-highway_{timestamp}"
vec_env = VecVideoRecorder(vec_env, video_folder,
                           record_video_trigger=lambda x: x == 0, video_length=video_length,
                           name_prefix=video_name)

# Evaluate the policy and collect per-episode rewards
episode_rewards = []
num_episodes = 100
for _ in range(num_episodes):
    obs = vec_env.reset()
    episode_reward = 0
    Done = False
    while not Done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        episode_reward += rewards
        Done = dones[0]
    episode_rewards.append(episode_reward)
# save the video
vec_env.close()
episode_numbers = np.arange(1, num_episodes + 1)
plt.plot(episode_numbers, episode_rewards, marker='o')
plt.xlabel('Episode')
plt.ylabel('Mean Reward')
plt.title('Mean Reward for Each Episode')
plt.show()