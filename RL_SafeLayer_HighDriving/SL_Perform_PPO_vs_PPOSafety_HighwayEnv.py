"""
*******************************************************************************
                            [SL_Perform_PPO_vs_PPOSafety_HighwayEnv]
*******************************************************************************
File: SL_Perform_PPO_vs_PPOSafety_HighwayEnv.py
Author: Farshad Mirzarazi
Date: 28.01.2024
Description:
------------
This source code is part of the author's Dissertation. (Chapter 8)

PPO/PPO with safety Penalty  Reinforcement Learning Highway-Env Environment
Performing PPO/PPO with safety penalty Agents on Highway-Env Environment: Tensorboard Log Files
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
import os
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3 import DQN, A2C, PPO, DDPG

# RL and SafetyContext thresholds
lanespace = 2
alpha = 2
longitudinalspace = 5
XRange = 100
YRange = 10
VxRange=VyRange= 20
SM_Threshold_RLContext = [10, 10]
SM_Threshold_safetyContext = [4, 4]

def Plot_Result(PPO_withSP_rewards, PPO_woSP_rewards, PPO_withSP_lengths, PPO_woSP_lengths, PPO_withSP_crashcnt, PPO_woSP_crashcnt,ppo_woSP_avgreward ,PPO_withSP_avgreward ):
    figtext1 = PPO_withSP_crashcnt + "|" + PPO_woSP_crashcnt
    figtext2 =ppo_woSP_avgreward + "|"  + PPO_withSP_avgreward

    figsize = (12, 8)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    ax1.plot(PPO_withSP_rewards, label='PPO_withSP_Rewards', color='red')
    ax1.plot(PPO_woSP_rewards, label='PPO_baseline_Rewards', color='#425066')

    ax1.set_ylabel('Total Reward')
    ax1.set_xlabel('Episode')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    fig.text(0.04, 0.03, figtext1, ha='left', fontsize=10)
    fig.text(0.04, 0.50, figtext2, ha='left', fontsize=10)

    # Plot episode_lengths in another subplot
    ax2.plot(PPO_withSP_lengths, label='PPO_withSP_Lengths', color='red')
    ax2.plot(PPO_woSP_lengths, label='PPO_baseline_Lengths', color='#425066')
    ax2.set_ylabel('Episode Length')
    ax2.set_xlabel('Episode')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    plt.suptitle('PPO_Baseline_Safety Penalty-Reward and Ep. Length Highway Driving')  # Uncomment this line
    # Adjust layout parameters
    plt.subplots_adjust(top=0.934, bottom=0.080, left=0.060, right=0.988, hspace=0.208, wspace=0.200)

    # Save the plot with a timestamp in the filename
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f'plot_{timestamp}.png'
    file_path = os.path.join(plot_save_directory, filename)
    plt.savefig(file_path)

    #plt.show()
def Plot_Result2(PPO_withSP_safetyKPI1, PPO_withSP_safetyKPI2, PPO_withSP_crash_history, PPO_woSP_safetyKPI1, PPO_woSP_safetyKPI2,PPO_woSP_crash_history, PPO_crashcnts, PPO_SafetyMargins):
    figtext1 = PPO_SafetyMargins
    figtext2 = PPO_crashcnts

    figsize = (12, 12)
    fig, (ax1) = plt.subplots(1, 1, figsize=figsize)
    # Plot SafetyKPI1 subplot
    ax1.plot(PPO_withSP_safetyKPI1, label='PPO_with_SP_SafetyKPI1', color='red')
    ax1.plot(PPO_woSP_safetyKPI1, label='PPO_Baseline_SafetyKPI1', color='#425066')
    ax1.set_ylabel('Safety KPI1')
    ax1.set_xlabel('Timestep')
    ax1.axhline(y=4, linestyle='--', color='gray', label='long.Distance = 4[m]')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    fig.text(0.04, 0.03, figtext1, ha='left', fontsize=10)
    #fig.text(0.04, 0.50, figtext2, ha='left', fontsize=10)

    plt.suptitle('PPO_Baseline_PPO_withSP_Safety Margin_Highway Driving')
    # Adjust layout parameters
    plt.subplots_adjust(top=0.934, bottom=0.080, left=0.060, right=0.988, hspace=0.208, wspace=0.200)
    #plt.show()

    # Save the plot with a timestamp in the filename
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f'plot_{timestamp}.png'
    file_path = os.path.join(plot_save_directory, filename)
    plt.savefig(file_path)
    ##########################################################################
    figsize = (12, 12)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
    # Plot SafetyKPI1 subplot
    ax1.plot(PPO_withSP_safetyKPI1, label='PPO_withSP_SafetyKPI1', color='#425066')
    ax1.set_ylabel('Safety KPI1')
    ax1.set_xlabel('Timestep')
    ax1.axhline(y=4, linestyle='--', color='gray', label='long.Distance = 4[m]')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    fig.text(0.04, 0.03, figtext1, ha='left', fontsize=10)
    fig.text(0.04, 0.50, figtext2, ha='left', fontsize=10)

    # Plot SafetyKPI1 subplot
    ax2.plot(PPO_withSP_safetyKPI2, label='A2C_SafetyKPI2', color='#425066')
    ax2.set_ylabel('Safety KPI2')
    ax2.set_xlabel('Timestep')
    ax2.axhline(y=2.5, linestyle='--', color='gray', label='lat.Distance = 2.5[m]')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    # Plot crash_history subplot
    ax3.plot(PPO_withSP_crash_history, label='PPO_withSP_Crashes', color='#425066')
    ax3.set_ylabel('Crash Count')
    ax3.set_xlabel('Timestep')
    ax3.legend(loc='upper right')
    ax3.grid(True)

    plt.suptitle('PPO_withSP Performance: Safety KPIs and Crash Incidents during Highway Driving')
    # Adjust layout parameters
    plt.subplots_adjust(top=0.934, bottom=0.080, left=0.060, right=0.988, hspace=0.208, wspace=0.200)
    #plt.show()
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

    def compute_safety_context(self, Safety_KPI1, Safety_KPI2):
        # Compute safety context based on the minimum Safety_KPI values and SM_Threshold1
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

# Folder to save model, store TF_Logs, and videos
model_dir = "F:/Brunel/02_SourceCodes/HighwayDriving/Highway_Models"
PPO_model_path = f"{model_dir}/HighwayPPO"

tensorboard_log = f"{model_dir}/tensorboard_logs"
plot_save_directory= f"{model_dir}/Plots"
# Record the video starting at the first step
video_length = 1000
# Generate a timestamp string; Use the timestamp in the video file name
env = gym.make("highway-v0", render_mode='rgb_array')
env.config["normalized"] = False
env.configure({"screen_width": 800})
vec_env = DummyVecEnv([lambda: env])

# Load the last trained models  for each algorithm
model1 = PPO.load(f"{PPO_model_path}/SafetyClip_Lambda-0.5_40000.0", env=vec_env)
model2 = PPO.load(f"{PPO_model_path}/SafetyClip_Lambda0.5_40000.0", env=vec_env)

num_episodes = 30
# Initialize lists to store episode_rewards for each agent
PPO_woSP_episode_rewards = []
PPO_withSP_episode_rewards = []

PPO_woSP_episode_lengths = []
PPO_withSP_episode_lengths = []

PPO_woSP_crash_count = 0
PPO_withSP_crash_count = 0

PPO_woSP_SafetyKPI1Margin = 0
PPO_withSP_SafetyKPI1Margin = 0

PPO_woSP_average_reward = 0
PPO_withSP_average_reward = 0

PPO_woSP_SafetyKPI1 = []
PPO_withSP_SafetyKPI1 = []

PPO_woSP_SafetyKPI2 = []
PPO_withSP_SafetyKPI2 = []

PPO_woSP_crash_history = []
PPO_withSP_crash_history = []
# Instantiate SafetyMargin with default and specified threshold values
safety_margin = SafetyMargin(
    lanespace=lanespace,
    alpha=alpha,
    longitudinalspace=longitudinalspace,
    SM_Threshold_RLContext=SM_Threshold_RLContext,
    SM_Threshold_safetyContext=SM_Threshold_safetyContext)
# Create a list of RL models
rl_models = [model1, model2]
for i in range(2):
    video_folder= f"{PPO_model_path}/videos/"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    video_name = f"{timestamp}"
    vec_env = VecVideoRecorder(vec_env, video_folder,
                               record_video_trigger=lambda x: x == 0, video_length=video_length,
                               name_prefix=video_name)
    crash_count = 0
    timestep = 0
    SafetyKPI1Margin = 0
    # Reset the environment
    obs = vec_env.reset()
    # Load the trained agent
    model = rl_models[i]
    # Evaluate the policy and collect per-episode rewards
    episode_rewards = []
    episode_lengths = []
    argmin_safetyKPI1 = []
    argmin_safetyKPI2 = []
    crash_history = []
    agent_cumulative_reward = 0

    for _ in range(num_episodes):
        obs = vec_env.reset()
        episode_reward = 0
        episode_length = 0
        episodeCompleted_count= 0
        done = False
        while not done:
            timestep += 1
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = vec_env.step(action)
            episode_reward += rewards
            episode_length +=1
            done = dones[0]
            # Extract x,y-coordinates and vx, vy of Ego & target vehicles from Agent observations (Target values relative to Ego!!)
            x_values = obs[0][:, 1]
            y_values = obs[0][:, 2]
            vx_values = obs[0][:, 3]
            vy_values = obs[0][:, 4]
            # Longitudinal and Lateral Distance arrays (values in meter and relative to Ego)
            Ego_Target_Long = x_values[1:5] * XRange
            Ego_Target_Lat = y_values[1:5] * YRange
            # Safety KPI values
            SafetyKPI1 = safety_margin.compute_safety_kpi1(Ego_Target_Long, Ego_Target_Lat)
            SafetyKPI2 = safety_margin.compute_safety_kpi2(Ego_Target_Lat, Ego_Target_Long)
            argminKPIs= safety_margin.compute_safetyKPI_argmin(SafetyKPI1, SafetyKPI2)
            argmin_safetyKPI1.append(argminKPIs[0])
            argmin_safetyKPI2.append(argminKPIs[1])

            SafetyKPI1Margin += argminKPIs[0]
            for episode_info in info:
                if 'crashed' in episode_info and episode_info['crashed']:
                    crash_count += 1
                if 'TimeLimit.truncated' in episode_info and episode_info['TimeLimit.truncated']:
                    episodeCompleted_count += 1
            crash_history.append(crash_count)
        agent_cumulative_reward +=episode_reward
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_numbers = np.arange(1, num_episodes + 1)
        print(f"episode: {_}, Safety_KPI1: {argminKPIs[0]}, Safety_KPI2: {argminKPIs[1]}, Agent Action: {action}")

    # Save episode_rewards for each agent
    if i == 0:
        PPO_woSP_episode_rewards = episode_rewards
        PPO_woSP_average_reward = agent_cumulative_reward / num_episodes
        PPO_woSP_episode_lengths = episode_lengths
        PPO_woSP_crash_count = crash_count
        PPO_woSP_SafetyKPI1 = argmin_safetyKPI1
        PPO_woSP_SafetyKPI2 = argmin_safetyKPI2
        PPO_woSP_crash_history = crash_history
        PPO_woSP_SafetyKPI1Margin = SafetyKPI1Margin / timestep
    elif i == 1:
        PPO_withSP_episode_rewards = episode_rewards
        PPO_withSP_average_reward = agent_cumulative_reward / num_episodes
        PPO_withSP_episode_lengths = episode_lengths
        PPO_withSP_crash_count = crash_count
        PPO_withSP_SafetyKPI1 = argmin_safetyKPI1
        PPO_withSP_SafetyKPI2 = argmin_safetyKPI2
        PPO_withSP_crash_history = crash_history
        PPO_withSP_SafetyKPI1Margin = SafetyKPI1Margin / timestep
formatted_ppo_withSP_avg_reward = "{:.3f}".format(PPO_withSP_average_reward[0])
formatted_ppo_woSP_avg_reward = "{:.3f}".format(PPO_woSP_average_reward[0])

Plot_Result(PPO_withSP_episode_rewards, PPO_woSP_episode_rewards, PPO_withSP_episode_lengths, PPO_woSP_episode_lengths,
             f"PPO_withSP Crash Count: {PPO_withSP_crash_count}", f"PPO_woSP Crash Count: {PPO_woSP_crash_count}",
            f"PPO_Baseline avg Reward: {formatted_ppo_woSP_avg_reward}",f"PPO_withSP avg Reward: {formatted_ppo_withSP_avg_reward}")

Plot_Result2(PPO_withSP_SafetyKPI1, PPO_withSP_SafetyKPI2, PPO_withSP_crash_history, PPO_woSP_SafetyKPI1, PPO_woSP_SafetyKPI2, PPO_woSP_crash_history
             , f" PPO_woSP Crash Count: {PPO_woSP_crash_count}|PPO_withSP Crash Count: {PPO_withSP_crash_count}",
             f"PPO_Baseline KPI1 Margin: {PPO_woSP_SafetyKPI1Margin}|PPO_withSP KPI1 Margin: {PPO_withSP_SafetyKPI1Margin}")
plt.show()
