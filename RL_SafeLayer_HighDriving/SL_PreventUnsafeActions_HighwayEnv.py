"""
*******************************************************************************
                            [SL_PreventUnsafeActions_HighwayEnv]
*******************************************************************************
File: SL_PreventUnsafeActions_HighwayEnv.py
Author: Farshad Mirzarazi
Date: 28.01.2024
Description:
------------
This source code is part of the author's Dissertation. (Chapter 8)

DQN  Reinforcement Learning Highway-Env Environment
Safety Layer (SL) prevention of Unsafe Actions considering Safety KPIs
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
longitudinalspace = 8
XRange = 100
YRange = 10
VxRange=VyRange= 20
SM_Threshold_RLContext = [10, 10]
SM_Threshold_safetyContext = [10, 6]
latdistThresh =0.5
longdistThresh = 10

def Plot_Result(Agent_episode_rewards, Agent_episode_lengths, figtext1, figtext2):

    figsize = (12, 8)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    ax1.plot(Agent_episode_rewards, label='Agent_Rewards')
    ax1.set_ylabel('Total Reward')
    ax1.set_xlabel('Episode')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    fig.text(0.04, 0.03, figtext1, ha='left', fontsize=10)
    fig.text(0.04, 0.50, figtext2, ha='left', fontsize=10)

    # Plot episode_lengths in another subplot
    ax2.plot(Agent_episode_lengths, label='Agent_Lengths')
    ax2.set_ylabel('Episode Length')
    ax2.set_xlabel('Episode')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    plt.suptitle('DQN Agent - Prevention of Unsafe Actions - Reward and Ep. Length - Highway Driving')
    plt.subplots_adjust(top=0.934, bottom=0.080, left=0.060, right=0.988, hspace=0.208, wspace=0.200)
    plt.tight_layout()
    #plt.show()
def Plot_Result2(argmin_safetyKPI1, argmin_safetyKPI2, DQNActions, ReplacedActions, redAgent_crash_history, DetectedUnsafeCnt, PreventedUnsafeCnt, AgentCrashCnt):
    figtext1 = ""#DetectedUnsafeCnt + "|" + PreventedUnsafeCnt + "|" + AgentCrashCnt
    figtext2 = ""# redAgent_avgreward

    figsize = (12, 12)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    ax1.plot(argmin_safetyKPI1, label='Agent_SafetyKPI1')
    ax1.set_ylabel('Safety KPI1')
    ax1.set_xlabel('Timestep')
    ax1.axhline(y=4, linestyle='--', color='gray', label='long.Distance = 4[m]')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    fig.text(0.04, 0.03, figtext1, ha='left', fontsize=10)
    fig.text(0.04, 0.50, figtext2, ha='left', fontsize=10)

    ax2.plot(argmin_safetyKPI2, label='Agent_SafetyKPI2')
    ax2.set_ylabel('Safety KPI2')
    ax2.set_xlabel('Timestep')
    ax2.axhline(y=2.5, linestyle='--', color='gray', label='lat.Distance = 2.5[m]')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    action_labels = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }
    # ax3.plot(DQNActions, label='DQN Agent Action')
    # ax3.plot(ReplacedActions, label='Safety Layer Action')
    # ax3.set_xlabel('Timestep')
    # ax3.legend(loc='upper left')
    # ax3.grid(True)
    # ax3.set_yticks(list(action_labels.keys()))
    # ax3.set_yticklabels(list(action_labels.values()))

    plt.suptitle('DQN Agent - Safety KPIs - Prev. unsafe Actions during Highway Driving')
    plt.subplots_adjust(top=0.934, bottom=0.080, left=0.060, right=0.988, hspace=0.208, wspace=0.200)
    plt.xlim(-20)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f'plot_{timestamp}.png'
    file_path = os.path.join(plot_save_directory, filename)
    plt.savefig(file_path)
    #plt.show()
def Plot_Result3(argmin_safetyKPI1, DQNActions, ReplacedActions, crash_history,unsafe_history, DetectedUnsafeCnt, PreventedUnsafeCnt, AgentCrashCnt):
    figtext1 = DetectedUnsafeCnt + "|" + PreventedUnsafeCnt + "|" + AgentCrashCnt
    figtext2 = ""# redAgent_avgreward

    figsize = (12, 12)
    fig, ( ax2, ax3) = plt.subplots(2, 1, figsize=figsize)

    # ax1.plot(argmin_safetyKPI1, label='Red. Agent_SafetyKPI1')
    # ax1.set_ylabel('Safety KPI1')
    # ax1.set_xlabel('Timestep')
    # ax1.axhline(y=4, linestyle='--', color='gray', label='long.Distance = 4[m]')
    # ax1.legend(loc='upper left')
    # ax1.grid(True)

    fig.text(0.04, 0.03, figtext1, ha='left', fontsize=10)
    fig.text(0.04, 0.50, figtext2, ha='left', fontsize=10)

    ax2.plot(crash_history, label='crash_history')
    # ax2.plot(unsafe_history, label='detected unsafe action')
    ax2.set_ylabel('Crash Count')
    ax2.set_xlabel('Timestep')
    ax2.legend(loc='upper left')
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
    ax3.plot(DQNActions, label='DQN Agent Action')
    ax3.plot(ReplacedActions, label='Safety Layer Action', color='red')
    # ax3.set_ylabel('Action Overrides')
    ax3.set_xlabel('Timestep')
    ax3.legend(loc='upper left')
    ax3.grid(True)

    # Set y-axis ticks and labels
    ax3.set_yticks(list(action_labels.keys()))
    ax3.set_yticklabels(list(action_labels.values()))

    plt.suptitle('DQN Agent - Crash Incidents - Prev. unsafe Actions during Highway Driving')
    # Adjust layout parameters
    plt.subplots_adjust(top=0.934, bottom=0.080, left=0.060, right=0.988, hspace=0.208, wspace=0.200)
    # Set x-axis limits to start from zero
    plt.xlim(-20)

    # Save the plot with a timestamp in the filename
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f'plot_{timestamp}.png'
    file_path = os.path.join(plot_save_directory, filename)
    plt.savefig(file_path)

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
class RLSafety_layer:
    def __init__(self, latDistThreshold, longDistThreshold):
        self.latDistThreshold = latDistThreshold
        self.longDistThreshold = longDistThreshold
    def prevent_unsafe_actions(self,action, safety_KPI1, safety_KPI2, ego_lane):
        # Method to replace unsafe actions based on the safety context
        if action == 0:  # 'lane left'
            if safety_KPI1[np.argmin(np.abs(safety_KPI1))] >= self.longDistThreshold:
                return 1  # 'idle'
            elif ((ego_lane!=3) and ((all(val < 0 for val in safety_KPI2)) or (all(abs(val) > self.latDistThreshold for val in safety_KPI2)))):
                return 2  # 'lane right'
            else:
                return 4  # 'slower'

        elif action == 1:  # 'idle'
            # if ((ego_lane!=0) and ((all(val > 0 for val in safety_KPI2)) or (all(abs(val) > self.latDistThreshold for val in safety_KPI2)))):
            #     return 0  # 'lane left'
            # elif ((ego_lane!=3) and ((all(val < 0 for val in safety_KPI2)) or (all(abs(val) > self.latDistThreshold for val in safety_KPI2)))):
            #     return 2  # 'lane right'
            #else:
                return 4  # 'slower'

        elif action == 2:  # 'lane right' 
            if safety_KPI1[np.argmin(np.abs(safety_KPI1))] >= self.longDistThreshold:
                return 1  # 'idle'
            elif ((ego_lane!=0) and ((all(val > 0 for val in safety_KPI2)) or (all(abs(val) > self.latDistThreshold for val in safety_KPI2)))):
                return 0  # 'lane left'
            else:
                return 4  # 'slower'

        elif action == 3:  # 'faster' 
            if safety_KPI1[np.argmin(np.abs(safety_KPI1))] >= self.longDistThreshold:
                return 1  # 'idle'
            # elif ((ego_lane!=0) and ((all(val > 0 for val in safety_KPI2)) or (all(abs(val) > self.latDistThreshold for val in safety_KPI2)))):
            #     return 0  # 'lane left'
            # elif ((ego_lane!=3) and ((all(val < 0 for val in safety_KPI2)) or (all(abs(val) > self.latDistThreshold for val in safety_KPI2)))):
            #     return 2  # 'lane right'
            else:
                return 4  # 'slower'

        elif action == 4:  # 'slower' 
            if ((ego_lane!=0) and ((all(val > 0 for val in safety_KPI2)) or (all(abs(val) > self.latDistThreshold for val in safety_KPI2)))):
                return 0  # 'lane left'
            elif ((ego_lane!=3) and ((all(val < 0 for val in safety_KPI2)) or (all(abs(val) > self.latDistThreshold for val in safety_KPI2)))):
                return 2  # 'lane right'
            else:
                return 4  # 'slower'
        return action

##########################
# Action_unsafe Functions
##########################
def action_unsafe(action, safety_KPI1, safety_KPI2, latdistThresh, longdistThresh, lane_number):
    if action == 0: #Laneleft
        return lane_left_Unsafe(safety_KPI1, safety_KPI2, latdistThresh,longdistThresh, lane_number)
    elif action == 1: # Idle
        return idle_Unsafe(safety_KPI1, longdistThresh)
    elif action == 2: #Laneright
        return lane_right_Unsafe(safety_KPI1, safety_KPI2, latdistThresh,longdistThresh, lane_number)
    elif action == 3: # Faster
        return faster_Unsafe(safety_KPI1, longdistThresh)
    elif action == 4: # Slower
        return slower_Unsafe(safety_KPI1, longdistThresh)
    else:
        return False
def lane_left_Unsafe(safety_KPI1, safety_KPI2, latdistThresh, longdistThresh, lane_number):
    #if lane_number==0: return True
    # Get the indexes where elements in safety_KPI2 are > 0 values
    target_left_indexes = [index for index, val in enumerate(safety_KPI2) if val < 0 and abs(val) < latdistThresh]
    # Iterate through the indexes and check if any satisfy the condition
    for index in target_left_indexes:
        if abs(safety_KPI1[index]) < longdistThresh:
            return True  # If any index satisfies the condition, return True
    # If no index satisfies the condition, return False
    return False
def lane_right_Unsafe(safety_KPI1, safety_KPI2, latdistThresh,longdistThresh, lane_number):
    #if lane_number == 3: return True
    # Get the indexes where elements in safety_KPI2 are > 0 values
    target_right_indexes = [index for index, val in enumerate(safety_KPI2) if val > 0 and abs(val) < latdistThresh]
    # Iterate through the indexes and check if any satisfy the condition
    for index in target_right_indexes:
        if abs(safety_KPI1[index]) < longdistThresh:
            return True  # If any index satisfies the condition, return True
    # If no index satisfies the condition, return False
    return False
def idle_Unsafe(safety_KPI1, longdistThresh):
    if safety_KPI1[np.argmin(np.abs(safety_KPI1))] < longdistThresh:
        return True
    else:
        return False
def slower_Unsafe(safety_KPI1, longdistThresh):
    if safety_KPI1[np.argmin(np.abs(safety_KPI1))] < 0.5 * longdistThresh:
        return True
    else:
        return False
def faster_Unsafe(safety_KPI1, longdistThresh):
    if safety_KPI1[np.argmin(np.abs(safety_KPI1))] < 2 * longdistThresh:
        return True
    else:
        return False

    # Folder to save model, store TF_Logs, and videos

def compute_lane_number(yPosition, lanespace=0.25):
    if yPosition < 0:
        return None  # Negative yPosition is not valid

    lane_number = int(yPosition // lanespace)
    return lane_number

model_dir = "F:/Brunel/02_SourceCodes/HighwayDriving/Highway_Models"
crash_log_directory= "F:/Brunel/02_SourceCodes/HighwayDriving/Highway_Models/CrashLogs"
DQN_model_path = f"{model_dir}/HighwayDQN"
PreventUnsafe_path= f"{model_dir}/SL_PreventUnsafeActions"

tensorboard_log = f"{model_dir}/tensorboard_logs"
plot_save_directory= f"{model_dir}/Plots"
video_folder= f"{PreventUnsafe_path}/videos/"
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
DQNAgent = DQN.load(f"{DQN_model_path}/20000.0", env= vec_env)

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
SL_Action_Prevent_count = 0
unsafe_action_detected_count = 0
Ego_lane_number=0
# Evaluate the policy and collect per-episode rewards
episode_rewards = []
episode_lengths = []
argmin_safetyKPI1 = []
argmin_safetyKPI2 = []
crash_history = []
DQNActions = []
ReplacedActions = []
Unsafe_Action_history = []
agent_cumulative_reward = 0
num_episodes = 30
time_step = -1
for _ in range(num_episodes):
    obs = vec_env.reset()
    episode_reward = 0
    episode_length = 0
    episodeCompleted_count= 0
    done = False
    while not done:
        time_step += 1
        #----------
        # Safety KPIs
        #----------
        # Extract x,y-coordinates and vx, vy of Ego & target vehicles from Agent observations (Target values relative to Ego!!)
        x_values = obs[0][:, 1]
        y_values = obs[0][:, 2]
        vx_values = obs[0][:, 3]
        vy_values = obs[0][:, 4]

        Ego_lane_number = compute_lane_number(y_values[0])
        # Longitudinal and Lateral Distance arrays (values in meter and relative to Ego)
        Ego_Target_Long = x_values[1:5] * XRange
        Ego_Target_Lat = y_values[1:5] * YRange

        # Safety KPI values
        SafetyKPI1 = safety_margin.compute_safety_kpi1(Ego_Target_Long, Ego_Target_Lat)
        SafetyKPI2 = safety_margin.compute_safety_kpi2(Ego_Target_Lat, Ego_Target_Long)

        argminKPIs= safety_margin.compute_safetyKPI_argmin(SafetyKPI1, SafetyKPI2)
        argmin_safetyKPI1.append(argminKPIs[0])
        argmin_safetyKPI2.append(argminKPIs[1])
        # Compute safety context based on Safety KPI values
        SM_long_SafetyContext, SM_lat_SafetyContext = safety_margin.compute_safety_context(SafetyKPI1, SafetyKPI2)
        #----------
        # ACTIONS
        #----------
        # DQN Agent observes the Environment and predicts Actions
        DQNAction, _states = DQNAgent.predict(obs, deterministic=True)
        DQNActions.append(DQNAction)
        # Check if Safety Context is activated
        if SM_long_SafetyContext or SM_lat_SafetyContext:
            # Call safety layer - prev_unsafe_action to prevent unsafe actions
            DQNActionUnsafe= action_unsafe(DQNAction, SafetyKPI1, SafetyKPI2, latdistThresh, longdistThresh, Ego_lane_number)
            if DQNActionUnsafe:
                unsafe_action_detected_count += 1
                ReplacedAction = rl_safety_layer.prevent_unsafe_actions(DQNAction, SafetyKPI1, SafetyKPI2, Ego_lane_number)
                if ReplacedAction != DQNAction [0]:
                    SL_Action_Prevent_count += 1
            else:
                ReplacedAction = DQNAction[0]
        else: # no intervention required!
            ReplacedAction = DQNAction[0]
        ReplacedActions.append(ReplacedAction)
        #----------
        # Env step
        #----------
        # take the Redundant agent action
        obs, rewards, dones, info = vec_env.step(np.array([ReplacedAction]))
        episode_reward += rewards
        episode_length +=1
        done = dones[0]

        for episode_info in info:
            if 'crashed' in episode_info and episode_info['crashed']:
                crash_count += 1
                # Create or open the crash log file in append mode
                crash_log_filepath = os.path.join(crash_log_directory, f'crashlog_{timestamp}_{crash_count}.txt')
                with open(crash_log_filepath, 'a') as log_file:
                    # Write crash information to the file
                    log_file.write(f"Crash {crash_count} at TimeStep: {time_step}:\n")
                    log_file.write(f"Coordinates:\nX: {Ego_Target_Long},\nY:{Ego_Target_Lat}:\n")
                    log_file.write(f"EGO Lane Nr.: {Ego_lane_number}, Agent action {DQNAction},  SL detected unsafe: {DQNActionUnsafe}:\n")
                    log_file.write(f"Episode Information: {episode_info}\n")

            if 'TimeLimit.truncated' in episode_info and episode_info['TimeLimit.truncated']:
                episodeCompleted_count += 1
        crash_history.append(crash_count)
        Unsafe_Action_history.append(unsafe_action_detected_count)

        print(f"episode: {_}, Safety_KPI1: {argminKPIs[0]}, Safety_KPI2: {argminKPIs[1]}, DQN Action: {DQNAction}, Replaced Action: {ReplacedAction}, Actions prevented: {SL_Action_Prevent_count}, SafetyContext2:{SM_lat_SafetyContext}, SafetyContext1:{SM_long_SafetyContext}, lane: {Ego_lane_number}")
    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_length)
    agent_cumulative_reward += episode_reward
print(f"Number of failed Arbitration: {unsafe_action_detected_count}")
agent_cumulative_reward /= num_episodes

Plot_Result(episode_rewards, episode_lengths,
            f"Redundant Agent Crash Count: {crash_count}",f"Redundant average Reward: {agent_cumulative_reward}")

Plot_Result2(argmin_safetyKPI1, argmin_safetyKPI2, DQNActions, ReplacedActions, crash_history, f"Detected Unsafe Actions:{unsafe_action_detected_count}",
             f"DQN Prevented Unsafe Actions: {SL_Action_Prevent_count}", f" Agent Crash Count: {crash_count}")
             
Plot_Result3(argmin_safetyKPI1, DQNActions, ReplacedActions, crash_history, Unsafe_Action_history, f"Detected Unsafe Actions:{unsafe_action_detected_count}",
             f"Prevented Unsafe Actions: {SL_Action_Prevent_count}", f" Agent Crash Count: {crash_count}")
plt.show()
