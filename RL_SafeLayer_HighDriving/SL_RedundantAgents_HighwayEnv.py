"""
*******************************************************************************
                            [SL_RedundantAgents_HighwayEnv]
*******************************************************************************
File: SL_RedundantAgents_HighwayEnv.py
Author: Farshad Mirzarazi
Date: 28.01.2024
Description:
------------
This source code is part of the author's Dissertation. (Chapter 8)

DQN and PPO  Reinforcement Learning Highway-Env Environment
Safety Layer (SL) Safety Arbitration between redundant Agents considering Safety KPIs
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

def Plot_Result(redAgent_episode_rewards, redAgent_episode_lengths, redAgent_crashcnt, redAgent_avg_reward):
    figtext1 = redAgent_crashcnt
    figtext2 = redAgent_avg_reward

    figsize = (12, 8)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    ax1.plot(redAgent_episode_rewards, label='Red.Agent_Rewards')
    ax1.set_ylabel('Total Reward')
    ax1.set_xlabel('Episode')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    fig.text(0.04, 0.03, figtext1, ha='left', fontsize=10)
    fig.text(0.04, 0.50, figtext2, ha='left', fontsize=10)

    # Plot episode_lengths in another subplot
    ax2.plot(redAgent_episode_lengths, label='Red.Agent_Lengths')
    ax2.set_ylabel('Episode Length')
    ax2.set_xlabel('Episode')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    plt.suptitle('Redundant DQN & PPO Agents - Reward and Ep. Length- Highway Driving')
    # Adjust layout parameters
    plt.subplots_adjust(top=0.934, bottom=0.080, left=0.060, right=0.988, hspace=0.208, wspace=0.200)
    plt.tight_layout()
    # Remove or comment the plt.savefig line
    # plt.savefig(file_path)
    plt.show()

def Plot_Result2(redAgent_argmin_safetyKPI1, redAgent_argmin_safetyKPI2,dqnAction, ppoAciton, redAgentAction, redAgent_crash_history, DQNArbitcnt, PPOArbitcnt, redAgent_crashcnt, failedArbitrations):
    figtext1 = DQNArbitcnt + "|" + PPOArbitcnt + "|" +redAgent_crashcnt+ "|" + failedArbitrations
    figtext2 = ""# redAgent_avgreward

    figsize = (12, 12)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
    # Plot SafetyKPI1 subplot
    ax1.plot(redAgent_argmin_safetyKPI1, label='Red. Agent_SafetyKPI1')
    ax1.set_ylabel('Safety KPI1')
    ax1.set_xlabel('Timestep')
    ax1.axhline(y=4, linestyle='--', color='gray', label='long.Distance = 4[m]')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    fig.text(0.04, 0.03, figtext1, ha='left', fontsize=10)
    fig.text(0.04, 0.50, figtext2, ha='left', fontsize=10)

    # Plot SafetyKPI2 subplot
    ax2.plot(redAgent_argmin_safetyKPI2, label='Red. Agent_SafetyKPI2')
    ax2.set_ylabel('Safety KPI2')
    ax2.set_xlabel('Timestep')
    ax2.axhline(y=2.5, linestyle='--', color='gray', label='lat.Distance = 2.5[m]')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    # Assuming these are the enumerated values for the actions
    action_labels = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }
    # Plot actions taken by DQN, PPO and redundant agent
    ax3.plot(dqnAction, label='DQN Action')
    ax3.plot(ppoAciton, label='PPO Action')
    ax3.plot(redAgentAction, label='Red. Agent Action')
    ax3.set_ylabel('Action Overrides')
    ax3.set_xlabel('Timestep')
    ax3.legend(loc='upper right')
    ax3.grid(True)

    # Set y-axis ticks and labels
    ax3.set_yticks(list(action_labels.keys()))
    ax3.set_yticklabels(list(action_labels.values()))

    plt.suptitle('Redundant DQN & PPO Agents - Safety KPIs and Action overrides during Highway Driving')
    # Adjust layout parameters
    plt.subplots_adjust(top=0.934, bottom=0.080, left=0.060, right=0.988, hspace=0.208, wspace=0.200)
    # Set x-axis limits to start from zero
    #plt.xlim(-2)

    # Save the plot with a timestamp in the filename
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f'plot_{timestamp}.png'
    file_path = os.path.join(plot_save_directory, filename)
    plt.savefig(file_path)

    plt.show()
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
class RLSafety_layer:
    def __init__(self, latDistThreshold, longDistThreshold):
        self.latDistThreshold = latDistThreshold
        self.longDistThreshold = longDistThreshold

    def safety_arbitration(self, safety_KPI1, safety_KPI2, DQNAction, PPOAction):
        # Rule-based Safety Arbitration
        if (DQNAction == 3 and PPOAction == 1) or (DQNAction == 1 and PPOAction == 3):  # 'FASTER' and 'IDLE' -> idle
            return 1  # 'IDLE'
        elif (DQNAction == 3 and PPOAction == 4) or (DQNAction == 4 and PPOAction == 3):  # 'FASTER', 'SLOWER'
            return 4  # 'SLOWER'
        elif (DQNAction == 4 and PPOAction == 1) or (DQNAction == 1 and PPOAction == 4):  # 'SLOWER', 'IDLE'
            return 4  # 'SLOWER'
        elif (DQNAction == 0 and PPOAction == 4) or (DQNAction == 4 and PPOAction == 0):  # 'LANE_LEFT', 'SLOWER'
            return 4  # 'SLOWER'
        elif (DQNAction == 2 and PPOAction == 4) or (DQNAction == 4 and PPOAction == 2):  # 'LANE_RIGHT', 'SLOWER'
            return 4  # 'SLOWER'
        elif ((DQNAction == 0 and PPOAction == 2) or (DQNAction == 2 and PPOAction == 0)): # 'LANE_LEFT', 'LANE_RIGHT'
            if safety_KPI2[np.argmin(np.abs(safety_KPI1))] > 0:
                return 0  # 'LANE_LEFT'
            else:
                return 2 # 'LANE_RIGHT
        elif ((DQNAction == 0 and PPOAction == 3) or (DQNAction == 3 and PPOAction == 0)):
            if safety_KPI1[np.argmin(np.abs(safety_KPI1))] >= self.longDistThreshold:
                return 3 # 'FASTER'
            # Check if all elements in safety_KPI1 are greater than 0
            elif (all(val > 0 for val in safety_KPI2)) or (all(abs(val) > self.latDistThreshold for val in safety_KPI2)):
                return 0 # 'LANE_LEFT'
        elif ((DQNAction == 0 and PPOAction == 1) or (DQNAction == 1 and PPOAction == 0)):
            if safety_KPI1[np.argmin(np.abs(safety_KPI1))] >= self.longDistThreshold:
                return 1 # 'IDLE'
            elif (all(val > 0 for val in safety_KPI2)) or (all(abs(val) > self.latDistThreshold for val in safety_KPI2)):
                return 0 # 'LANE_LEFT'
        elif ((DQNAction == 2 and PPOAction == 3) or (DQNAction == 3 and PPOAction == 2)):
            if safety_KPI1[np.argmin(np.abs(safety_KPI1))] >= self.longDistThreshold:
                return 3  # 'FASTER'
            elif (all(val < 0 for val in safety_KPI2)) or (all(abs(val) > self.latDistThreshold for val in safety_KPI2)):
                return 2 # 'LANE_RIGHT'
        elif ((DQNAction == 2 and PPOAction == 1) or (DQNAction == 1 and PPOAction == 2)):
            if safety_KPI1[np.argmin(np.abs(safety_KPI1))] >= self.longDistThreshold:
                return 1  # 'IDLE'
            elif (all(val < 0 for val in safety_KPI2)) or (all(abs(val) > self.latDistThreshold for val in safety_KPI2)):
                return 2 # 'LANE_RIGHT'
        else:
            return -1 # Default to Unknown Action


# Folder to save model, store TF_Logs, and videos
model_dir = "F:/Brunel/02_SourceCodes/HighwayDriving/Highway_Models"
PPO_model_path = f"{model_dir}/HighwayPPO"
DQN_model_path = f"{model_dir}/HighwayDQN"
Red_model_path= f"{model_dir}/SL_RedundantAgents"

tensorboard_log = f"{model_dir}/tensorboard_logs"
plot_save_directory= f"{model_dir}/Plots"
video_folder= f"{Red_model_path}/videos/"
# Generate a timestamp string; Use the timestamp in the video file name
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
video_name = f"{timestamp}"
# Record the video starting at the first step
video_length = 10000

env = gym.make("highway-v0", render_mode='rgb_array')
env.config["normalized"] = False
env.configure({"screen_width": 800})
vec_env = DummyVecEnv([lambda: env])
# Reset the environment
obs = vec_env.reset()

vec_env = VecVideoRecorder(vec_env, video_folder,
                               record_video_trigger=lambda x: x == 0, video_length=video_length,
                               name_prefix=video_name)

# Load the last trained models
DQNAgent = DQN.load(f"{DQN_model_path}/40000.0", env= vec_env)
PPOAgent = PPO.load(f"{PPO_model_path}/40000.0", env=vec_env)
# Create a list of RL models
rl_models = [DQNAgent, PPOAgent]

# Instantiate RLSafety_layer & SafetyMargin with default and specified threshold values
safety_margin = SafetyMargin(
    lanespace=lanespace,
    alpha=alpha,
    longitudinalspace=longitudinalspace,
    SM_Threshold_RLContext=SM_Threshold_RLContext,
    SM_Threshold_safetyContext=SM_Threshold_safetyContext)

# Instantiate RLSafety_layer with threshold values
rl_safety_layer = RLSafety_layer(latDistThreshold=0.5, longDistThreshold=10)

crash_count = 0

SL_DQNArbit_count = 0
SL_PPOArbit_count = 0
SL_failedArbit_count = 0


# Evaluate the policy and collect per-episode rewards
episode_rewards = []
episode_lengths = []
argmin_safetyKPI1 = []
argmin_safetyKPI2 = []
crash_history = []

DQNActions = []
PPOActions = []
REDActions = []

agent_cumulative_reward = 0
num_episodes = 10

for _ in range(num_episodes):
    obs = vec_env.reset()
    episode_reward = 0
    episode_length = 0
    episodeCompleted_count= 0

    done = False
    while not done:
        #----------
        # Safety KPIs
        #----------
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
        #----------
        # ACTIONS
        #----------
        # Both DQN and PPO Agents observe the Environment and predict Actions
        DQNAction, _states = DQNAgent.predict(obs, deterministic=True)
        PPOAction, _states = PPOAgent.predict(obs, deterministic=True)
        DQNActions.append(DQNAction)
        PPOActions.append(PPOAction)
        # Check if DQNAction and PPOAction are different
        if DQNAction != PPOAction:
            # Call safety arbitration to compute Red. Agent action
            RedAgentAction = rl_safety_layer.safety_arbitration(SafetyKPI1, SafetyKPI2, DQNAction, PPOAction)
            if RedAgentAction == DQNAction:
                # PPO Action overriden
                SL_DQNArbit_count += 1
            elif RedAgentAction == PPOAction:
                #DQN Action overriden
                SL_PPOArbit_count += 1
        else: # no arbitration required!
            RedAgentAction = DQNAction[0]
        #default DQN action if safety arbitration fails
        if RedAgentAction != DQNAction[0] and RedAgentAction != PPOAction[0]:
            RedAgentAction  = DQNAction[0]
            SL_failedArbit_count += 1

        REDActions.append(RedAgentAction)

        #----------
        # Env step
        #----------
        # take the Redundant agent action
        obs, rewards, dones, info = vec_env.step(np.array([RedAgentAction]))
        episode_reward += rewards
        episode_length +=1
        done = dones[0]

        for episode_info in info:
            if 'crashed' in episode_info and episode_info['crashed']:
                crash_count += 1
            if 'TimeLimit.truncated' in episode_info and episode_info['TimeLimit.truncated']:
                episodeCompleted_count += 1
        crash_history.append(crash_count)

        print(f"episode: {_}, Safety_KPI1: {SafetyKPI1}, Safety_KPI2: {SafetyKPI2}, DQn Action: {DQNAction}, PPO Action: {PPOAction}, Arbitrated Action: {RedAgentAction}")
    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_length)
    agent_cumulative_reward += episode_reward
print(f"Number of failed Arbitration: {SL_failedArbit_count}")
agent_cumulative_reward /= num_episodes
#formatted_redundant_avg_reward = "{:.3f}".format((agent_cumulative_reward))

Plot_Result(episode_rewards, episode_lengths,
            f"Redundant Agent Crash Count: {crash_count}",f"Redundant average Reward: {agent_cumulative_reward}")
Plot_Result2(argmin_safetyKPI1, argmin_safetyKPI2, DQNActions, PPOActions, REDActions, crash_history,
             f"DQN Arbitration Count: {SL_DQNArbit_count}", f"PPO Arbitration Count: {SL_PPOArbit_count}",
             f"Red.Agent Crash Count: {crash_count}", f"Failed Arbitrations:{SL_failedArbit_count}")
