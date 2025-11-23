"""
*******************************************************************************
                            [SL_SafetyMargin_Highway]
*******************************************************************************
File: SL_SafetyMargin_Highway.py
Author: Farshad Mirzarazi
Date: 28.01.2024
Description:
------------
This source code is part of the author's Dissertation. (Chapter 8)

Quantifying and visualizing Safety Layer (SL) safety KPIs and 
RL vs Safety Contexts according to Chapter 7 section 7.3 safety margin scheme in Highway-Env Environment

Disclaimer:
------------
This code is provided "as is," without warranty of any kind.
The author is not responsible for any consequences arising from the use of this code.
*******************************************************************************
"""
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime  # Import the datetime module
import tensorboard
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common import logger

# Folder to save model, store TF_Logs, and videos
model_dir = "F:/Brunel/02_SourceCodes/HighwayDriving/Highway_Models/HighwayDQN"
tensorboard_log = f"{model_dir}/tensorboard_logs"
video_folder = f"{model_dir}/videos/"


# Create environment
env = gym.make("highway-fast-v0", render_mode='rgb_array')
env.config["normalized"] = False
#env.config["absolute"] = True
env.reset()
env.render()
XRange = 100
YRange = 10
VxRange=VyRange= 20

# Tensorboard Callback functions

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record("random_value", value)
        return True

class SafetyMargin:
    def __init__(self, SM_Threshold_RLContext, SM_Threshold_safetyContext, lanespace=3, alpha=2, longitudinalspace=5):
        self.lanespace = lanespace
        self.longitudinalspace = longitudinalspace
        self.alpha = alpha
        self.SM_Threshold_RLContext = SM_Threshold_RLContext # SM_Threshold1=[long_threshold1, lat_threshold1]
        self.SM_Threshold_safetyContext = SM_Threshold_safetyContext # SM_Threshold2=[long_threshold2, lat_threshold2]
    def compute_safety_kpi1(self, D_long, D_lat):
        # Compute Safety_KPI1 based on the absolute value of D_lat
        return np.where(np.abs(D_lat) < self.lanespace, D_long, XRange)
    def compute_safety_kpi2(self, D_lat, D_long):
        # Compute Safety_KPI2 based on the absolute value of D_long
        condition0 = np.abs(D_lat) < self.lanespace
        condition1 = np.abs(D_long) > 2 * self.longitudinalspace
        condition2 = np.abs(D_long) < self.longitudinalspace
        condition3 = np.abs(D_long) > self.longitudinalspace
        return np.where(condition0 | condition1, YRange, np.where(condition2, D_lat, self.alpha * D_lat))

    # Compute safety context based on the minimum Safety_KPI values and SM_Threshold1
    def compute_safety_context(self, Safety_KPI1, Safety_KPI2):
        # Find the index of the minimum value in each Safety_KPI array
        min_index1 = np.argmin(np.abs(Safety_KPI1))
        min_index2 = np.argmin(np.abs(Safety_KPI2))
        # Get the absolute values of the minimum values in each array
        min_value1 = np.abs(Safety_KPI1[min_index1])
        min_value2 = np.abs(Safety_KPI2[min_index2])
        # Compare the absolute values of the argmin values with the thresholds
        sm_safetycontext1 = min_value1 <  self.SM_Threshold_safetyContext[0]
        sm_safetycontext2 = min_value2 <  self.SM_Threshold_safetyContext[1]
        # Return the comparison results as arrays of boolean values
        return sm_safetycontext1, sm_safetycontext2

    def compute_safetyKPI_argmin(self, Safety_KPI1, Safety_KPI2):
        # Find the index of the minimum value in each Safety_KPI array
        min_index1 = np.argmin(np.abs(Safety_KPI1))
        min_index2 = np.argmin(np.abs(Safety_KPI2))
        # Get the absolute values of the minimum values in each array
        min_value1 = np.abs(Safety_KPI1[min_index1])
        min_value2 = np.abs(Safety_KPI2[min_index2])
        return min_value1, min_value2

tensorboard_callback = TensorboardCallback()

# Instantiate the agent
model = DQN('MlpPolicy', env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-4,
            buffer_size=15000,
            learning_starts=200,
            batch_size=32,
            gamma=0.8,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=50,
            verbose=1,
            tensorboard_log=tensorboard_log)

TIMESTEPS = 100 # at least 10000


for i in range(1, 10): # 10
    # model.learn(total_timesteps=100000, reset_num_timesteps=False, tb_log_name="DQN")
    # model.learn(total_timesteps=100, reset_num_timesteps=False, tb_log_name="DQNlogs", callback= tensorboard_callback, progress_bar=True)
    model.learn(total_timesteps=100, reset_num_timesteps=False, tb_log_name="DQNlogs",  progress_bar=True)

    # Save the agent
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # model.save(f"{model_dir}/{TIMESTEPS * i}")  # save the model every TIMESTEPS steps
    model.save(f"{model_dir}/dqn_highway_{timestamp}")  # save the model every TIMESTEPS steps

# delete trained model to demonstrate loading
del model
# -------------------------------------------------------------------------------------

env = gym.make("highway-fast-v0", render_mode='rgb_array')
env.config["normalized"] = False

env.config["duration"] = 60
env.configure({"screen_width": 800})


env.reset()

# Load the trained agent

vec_env = DummyVecEnv([lambda: env])
#vec_env.config["normalized"] = False
#vec_env.reset()
model_path = f"{model_dir}/60000"
model = DQN.load(model_path, env=vec_env)


# model = DQN.load("dqn_highway", env=vec_env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Record the video starting at the first step

video_length = 1000

# Generate a timestamp string
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
# Use the timestamp in the video file name
video_name = f"DQN-highway_{timestamp}"

vec_env = VecVideoRecorder(vec_env, video_folder,
                           record_video_trigger=lambda x: x == 0, video_length=video_length,
                           name_prefix=video_name)


# RL and SafetyContext thresholds
lanespace = 3
alpha = 2
longitudinalspace = 5
SM_Threshold_RLContext = [10, 10]
SM_Threshold_safetyContext = [4, 4]

# Instantiate SafetyMargin with default and specified threshold values
safety_margin = SafetyMargin(
    lanespace=lanespace,
    alpha=alpha,
    longitudinalspace=longitudinalspace,
    SM_Threshold_RLContext=SM_Threshold_RLContext,
    SM_Threshold_safetyContext=SM_Threshold_safetyContext
)



# Evaluate the policy and collect per-episode rewards
episode_rewards = []
num_episodes = 10
for _ in range(num_episodes):
    obs = vec_env.reset()
    episode_reward = 0
    Done = False
    while not Done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        episode_reward += rewards
        Done = dones[0]


        # Extract x,y-coordinates and vx, vy of Ego & target vehicles from Agent observations (Target values relative to Ego!!)
        x_values = obs[0][:, 1]
        y_values = obs[0][:, 2]
        vx_values = obs[0][:, 3]
        vy_values = obs[0][:, 4]

        # Longitudinal and Lateral Distance arrays (values in meter and relative to Ego)
        Ego_Target_Long =x_values[1:5] * XRange
        Ego_Target_Lat = y_values[1:5] * YRange

        # Safety KPI values
        SafetyKPI1 = safety_margin.compute_safety_kpi1(Ego_Target_Long, Ego_Target_Lat)
        SafetyKPI2 = safety_margin.compute_safety_kpi2(Ego_Target_Lat, Ego_Target_Long)

        # Compute safety context based on Safety KPI values
        SM_long_SafetyContext, SM_lat_SafetyContext = safety_margin.compute_safety_context(SafetyKPI1, SafetyKPI2)

    episode_rewards.append(episode_reward)
    print(f"episode: {_}, Safety_KPI1: {SafetyKPI1}, Safety_KPI2: {SafetyKPI2}")


# save the video
vec_env.close()

episode_numbers = np.arange(1, num_episodes + 1)
plt.plot(episode_numbers, episode_rewards, marker='o')
plt.xlabel('Episode')
plt.ylabel('Mean Reward')
plt.title('Mean Reward for Each Episode')
plt.show()
