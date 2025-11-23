"""
*******************************************************************************
                            [RL_Exploration_Metric_with_ExpMaximization_CartPoleEnv]
*******************************************************************************
File: RL_Exploration_Metric_with_ExpMaximization_HighwayEnv.py
Author: Farshad Mirzarazi
Date: 28.01.2024
Description:
------------
This source code is part of the author's Dissertation. (Chapter 7)

Maximizing exploration RL agent in CartPole Environment

Exploration Metric (EM) = (Nr.  of taken Actions)/(Nr.  of all possible Actions)  x 100%
in a discretized segmented state space
each factor s[0], s[1], s[2], and s[3] is uniformly segmented into 10 segments

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
# pytorch                   2.1.2
# ----------------
import os
from collections import namedtuple, deque
from datetime import datetime

import gymnasium as gym
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

variatoin = "Baseline | "
# Specify the directory
plot_save_directory = r'F:\Brunel\MyWritings\Chapter7_RLSafetyLayer\Plots_SafetyLayer'
model_save_directory = r'F:\\Brunel\\02_SourceCodes\\DQNVariations\\online_model.pth'

"""
Hyperparameters: 
"""
num_episodes = 20000

replayMem_size = 10000
# Size of mini-batch
batch_size =256
# update_rates of target network.
TAU_softUpdate = 0.05
TAU_hardUpdate = 10
# Discount factor
gamma = 0.9
# Epsilon greedy parameters
EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 5000
# Optimizer learning rate
LearningRate = 0.0001

max_reward = 500
DecayCount = 0
# Initialize a global variable to count action replacements
action_replacement_count = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#global arrays
episode_rewards = []
epsilon_values = []
epsilon_crossed50Percent = None
exploration_percentage = []
last_50_means = []  # Array to store last 50 means of total rewards
#  a list to store loss values and update values during training
loss_values = []
update_values = []
#------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, 64)
        self.layer3 = nn.Linear(64, 128)
        self.out = nn.Linear(128, n_actions)

    # Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.out(x)
class ReplayMemeory:
    def __init__(self,capacity):
        self.BufferLimit = capacity
        self.memory = deque([], maxlen=capacity) # assign memory to the replay buffer
        self.storagecount = 0

    def storeExperiment(self, experience) -> None:
        if len(self.memory) < self.BufferLimit:
            self.memory.append(experience)
        else:
            self.memory[self.storagecount % self.BufferLimit] = experience
            self.storagecount += 1

    def CreateMiniBatch(self, batch_size: int):
        return random.sample(self.memory, batch_size)
class DQNAgent:
    def __init__(self, input_size, output_size, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # self.target_net.load_state_dict(self.policy_net.state_dict())
        # # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.policy_net = DQN(input_size, output_size)
        self.target_net = DQN(input_size, output_size)
        self._hard_update_target_net()

    def _hard_update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state, current_segment):
        global action_replacement_count  # Referencing the global variable
        randNr = random.random()

        if self.epsilon > randNr:
            # Exploration: Randomly select an action
            selected_action = env.action_space.sample()

            # Check if the selected action has been taken before
            if current_segment[selected_action] > 0:
                available_actions = [action for action in range(env.action_space.n) if current_segment[action] == 0]
                if available_actions:
                    selected_action = random.choice(available_actions)
                    action_replacement_count += 1
        else:
            with torch.no_grad(): # Exploitation from the policy network
                selected_action = self.policy_net(state).max(1)[1].item()

        return torch.tensor([[selected_action]], device=device, dtype=torch.long)
    def decay_action(self):
        # Decay epsilon
        global DecayCount
        global epsilon_crossed50Percent
        self.epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * DecayCount / EPS_DECAY)
        if self.epsilon <= 0.5 and epsilon_crossed50Percent is None:
            epsilon_crossed50Percent = episode
        DecayCount += 1

    def exploit(self, state):
        with torch.no_grad():
            selected_action = self.policy_net(state).max(1)[1].item()
        return torch.tensor([[selected_action]], device=device, dtype=torch.long)

    def update_target_model(self):
        model_state_dict = self.policy_net.state_dict()
        target_model_state_dict = {}
        for key, value in model_state_dict.items():
            # Convert numpy arrays to tensors
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value).float()
            target_model_state_dict[key] = value
        self.target_net.load_state_dict(target_model_state_dict)



class ActionStorage_DataStructure:
    def __init__(self, num_segments, action_space):
        # Initialize the data structure with zeros
        self.segments = [[0] * len(action_space) for _ in range(num_segments)]
        self.action_space = action_space
        self.visited_segments = set()  # Set to keep track of visited segments

    def record_action(self, segment_index, action):
        # Calculate the index of the action in the action_space array
        action_index = self.action_space.index(action)

        # Increment the count for the specified action in the given segment
        self.segments[segment_index][action_index] += 1

        # Check if the segment has been visited before
        if segment_index not in self.visited_segments:
            # Add the segment to the set of visited segments
            self.visited_segments.add(segment_index)

    def get_segment_data(self, segment_index):
        # Retrieve the counts for all actions in the specified segment
        return self.segments[segment_index]

    # Return the count of visited segments
    def get_visited_segments_count(self):
        # Return the count of visited segments
        return len(self.visited_segments)

    def get_taken_actions_count(self):
        # Compute the total number of taken actions over visited segments
        total_actions_count = 0

        for segment_index in self.visited_segments:
            # Increment the count for each action in the current segment
            total_actions_count += sum(1 for action in self.action_space if self.segments[segment_index][self.action_space.index(action)] > 0)

        return total_actions_count

#------------------------------------------------------
# Step 1: Define the Gym environment
env = gym.make('CartPole-v1', render_mode="rgb_array")
input_size = env.observation_space.shape[0]
action_space_size = env.action_space.n
#------------------------------------------------------
Experience = namedtuple('Experience',
                        ('state', 'action', 'next_state', 'reward')
                        )
agent = DQNAgent(input_size, action_space_size)
ExperienceReplayBuffer = ReplayMemeory(replayMem_size)
#------------------------------------------------------
StateSpaceSegments = [10, 10, 10, 10]
NSegmentTotal = np.prod(StateSpaceSegments)

action_data_structure = ActionStorage_DataStructure(NSegmentTotal, [0,1])
# Lists to store data for Exploration Metric plotting
taken_actions_list = []
visited_segments_list = []
exploration_metric1_list = []
exploration_metric2_list = []
#------------------------------------------------------




#------------------------------------------------------
# Instantiate the optimizers with learning rate
# SGD optimizer
optimizer_sgd = optim.SGD(agent.policy_net.parameters(), lr=LearningRate)
# Adam optimizer
optimizer_adam = optim.Adam(agent.policy_net.parameters(), lr=LearningRate)
# RMSprop optimizer
optimizer_rmsprop = optim.RMSprop(agent.policy_net.parameters(), lr=LearningRate)
# agent.optimizer = optimizer_adam
agent.optimizer = optimizer_rmsprop
#------------------------------------------------------
# Helper functions
# Define Plot_Result function
def plot_visits_bar_chart(segments, visited_segments):
    """
    Plot a bar chart of the number of visits over all segments.

    Parameters:
    - segments: List[List[int]], a 2D list representing the action counts for each segment.
    """
    total_segments_count = len(segments)

    # Calculate the total visits for each segment
    total_visits = [sum(segment) for segment in segments]

    # Find the index of the maximum visited segment
    max_visited_index = total_visits.index(max(total_visits)) + 1  # Adding 1 to match the segment index
    Nrofvisited_Segment = len(visited_segments)

    # Plot a bar chart of the number of visits over all segments
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, total_segments_count + 1), [sum(segment) for segment in segments], color='blue', edgecolor='black')
    plt.xlabel('Segment Index')
    plt.ylabel('Number of Visits')
    plt.title('Exploration Patterns: DQN Agent Visits to Segmented State Space')
    plt.grid(True)

    # Set x-axis limits to start from zero
    plt.xlim(0, total_segments_count + 1)

    # Display the maximum visited segment and its count value below the legend
    plt.text(0.2, -0.08, f'{Nrofvisited_Segment} Segments visited | Most Visited: {max_visited_index} ({max(total_visits)} times)', color='black',
             ha='center', transform=plt.gca().transAxes)

    plt.subplots_adjust(left=0.09, bottom=0.095, right=0.97, top=0.92, wspace=0, hspace=.3)
    plt.show()

def plot_exploration_metrics_result(Exploration_Metric1_list, Exploration_Metric2_list, taken_actions_list, visited_segments_list, epsilon_values_list, epsilon_crossed50Percent):
    """
    Plot the exploration metrics result over episodes.

    Parameters:
    - Exploration_Metric1_list: List[float], list of Exploration Metric1 values for each episode.
    - Exploration_Metric2_list: List[float], list of Exploration Metric2 values for each episode.
    - taken_actions_list: List[int], list of counts of taken actions for each episode.
    - visited_segments_list: List[int], list of counts of visited segments for each episode.
    """
    """
    Plot a bar chart of the number of visits over all segments.

    Parameters:
    - segments: List[List[int]], a 2D list representing the action counts for each segment.
    """
    total_episode_count = len(Exploration_Metric1_list)

    # First Plot (Subplots 1 and 2)
    plt.figure(figsize=(10, 8))

    # Plot Subplot 1 for Exploration Metric1
    plt.subplot(2, 1, 1)
    plt.plot(Exploration_Metric1_list, label='Exploration Metric1', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Exploration Metric1')
    plt.legend()
    plt.grid(True)

    # Set x-axis limits to start from zero
    plt.xlim(-100, total_episode_count + 500)

    # Display Min and Max values for Exploration Metric1
    plt.text(0., -0.12, f'Range: {min(Exploration_Metric1_list):.2f}% -- {max(Exploration_Metric1_list):.2f}%', transform=plt.gca().transAxes, color='black')

    # Plot Subplot 2 for Exploration Metric2
    plt.subplot(2, 1, 2)
    plt.plot(Exploration_Metric2_list, label='Exploration Metric2', color='purple')
    plt.xlabel('Episode')
    plt.ylabel('Exploration Metric2')
    plt.legend()
    plt.grid(True)

    # Set x-axis limits to start from zero
    plt.xlim(0, total_episode_count + 500)

    # Display Min and Max values for Exploration Metric2
    plt.text(0.0, -0.12, f'Range: {min(Exploration_Metric2_list):.2f}% -- {max(Exploration_Metric2_list):.2f}%', transform=plt.gca().transAxes, color='black')

    plt.suptitle('Exploration Patterns: DQN Agent Exploration Metrics Over Episodes (with Maximization)')

    plt.subplots_adjust(left=0.07, bottom=0.07, right=0.995, top=0.95, wspace=0, hspace=0.19)
    plt.show()

    # Second Plot (Subplots 3 and 4)
    plt.figure(figsize=(10, 8))

    # Plot Subplot 3 for Taken Actions and Visited Segments
    plt.subplot(2, 1, 1)
    plt.plot(taken_actions_list, label='Taken Actions')
    plt.plot(visited_segments_list, label='Visited Segments', color='green')
    #plt.xlabel('Episode')
    plt.ylabel('Action/Segment Count')
    plt.legend()
    plt.grid(True)

    # Set x-axis limits to start from zero
    plt.xlim(0, total_episode_count + 500)

    # Display maximum number of unique taken actions and visited segments
    max_unique_actions = max(taken_actions_list)
    max_visited_segments = max(visited_segments_list)

    plt.text(0., -0.12, f'{action_replacement_count} action replacements |  {max_unique_actions} Unique Taken Actions in {max_visited_segments} Visited Segments', transform=plt.gca().transAxes, color='black')

    # Plot Subplot 4 for Epsilon Values with vertical line at epsilon_crossed50Percent
    plt.subplot(2, 1, 2)
    plt.plot(epsilon_values_list, label='Epsilon Values', color='brown')
    plt.axvline(x=epsilon_crossed50Percent, color='gray', linestyle='--', label='Epsilon = 0.5')  # Vertical line
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.grid(True)

    # Set x-axis limits to start from zero
    plt.xlim(0, total_episode_count + 500)

    # Display Min and Max values for Epsilon
    plt.text(0.0, -0.12, f'Range: {min(epsilon_values_list):.2f}% -- {max(epsilon_values_list):.2f}%', transform=plt.gca().transAxes, color='black')

    plt.suptitle('Exploration Patterns: DQN Agent Exploration Metrics Over Episodes (with Maximization)')

    plt.subplots_adjust(left=0.07, bottom=0.07, right=0.995, top=0.95, wspace=0, hspace=0.19)
    plt.show()


def Plot_Result(episode_rewards, epsilon_values, epsilon_crossed50Percent, last_50_means, episode_last_50_mean_gt_400, average_reward):
    if episode_rewards:
        figtext = variatoin + "|"  # Initialize with an empty string
        episodes, rewards = zip(*episode_rewards)

        figsize = (12, 4)  # Adjust the width and height as needed
        fig, (ax1) = plt.subplots(1, 1, figsize=figsize)
        #fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize)
        ############ ax1 ############
        ax1.plot(episodes, rewards, label='Total Reward', color='green')
        ax1.axhline(y=max_reward, color='orange', linestyle='--', label='Max Reward')
        ax1.set_ylabel('Total Reward')
        ax1.set_xlabel('Episode')
        ax1.grid(True)

        # Plot Last 50 Mean Reward with label
        ax1.plot(episodes[-len(last_50_means):], last_50_means, color='blue', label='Mean Last 50 Reward')
        ax1.legend(loc='upper left')

        # Add a vertical line at the episode where last_50_mean > 400
        if episode_last_50_mean_gt_400 is not None:
            ax1.axvline(x=episode_last_50_mean_gt_400, color='purple', linestyle='--', label='Mean Last 50 Mean > 400')
            #ax1.text(0.5, -0.1, f'Training stop at episode: {episode_last_50_mean_gt_400}', transform=ax1.transAxes, fontsize=10, ha='center')
            figtext += f'Training stop at episode: {episode_last_50_mean_gt_400} | '

        ############ ax2 ############
        # ax2.plot(episodes, epsilon_values, label='Epsilon', color='red')
        # ax2.axvline(x=epsilon_crossed50Percent, color='gray', linestyle='--', label='Epsilon = 0.5')
        # ax2.set_xlabel('Episode')
        # ax2.set_ylabel('Epsilon')
        # ax2.legend(loc='upper right')
        # ax2.grid(True)

        ############ subplots and subtitle ############
        # Adjust the vertical position of the entire plot
        plt.subplots_adjust(left=0.06, bottom=0.162, right=0.97, top=0.912, wspace=0, hspace=.2)

        # Display average reward below ax2
        figtext += f" Average Reward of Trained Model: {average_reward}"
        fig.text(0.5, 0.01, figtext, ha='center', fontsize=10)
        # fig.text(0.5, 0.01, f'Average Reward of Trained Model: {average_reward}', ha='center', fontsize=10)

        plt.suptitle('DQN Training: Total and Mean Rewards over Episodes')
        # plt.suptitle('DQN Training: Total Reward and Epsilon over Episodes')

        # Save the plot with a timestamp in the filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f'plot_{timestamp}.png'
        file_path = os.path.join(plot_save_directory, filename)
        plt.savefig(file_path)

        plt.show()
    else:
        print("No episode durations recorded. Cannot plot.")
# Define a function to calculate the mean of the last 50 values
def calculate_last_50_mean(episode_rewards):
    if len(episode_rewards) >= 50:
        last_50_rewards = episode_rewards[-50:]
        last_50_mean = sum([reward for _, reward in last_50_rewards]) / 50
        return last_50_mean
    else:
        return None

def compute_segment_number_cartpole(current_state):
    """
    Compute the current segment number from the current state.
    Parameters:
    - current_state: List[float], representing the current state with 4 factors.
    Returns:
    - current segment number [int].
    """
    segment_counts = [10, 10, 10, 10]  # Number of segments for each factor
    factor_ranges = [(-4.8, 4.8), (-5, 5), (-24, 24), (-5, 5)]

    if len(current_state) != len(segment_counts) or len(current_state) != len(factor_ranges):
        raise ValueError("Dimension mismatch between current state, segment counts, and factor ranges.")

    relative_positions = [
        max(1, min(int((current_state[i] - min_value) / (max_value - min_value) * segment_counts[i]) + 1, 10))
        for i, (min_value, max_value) in enumerate(factor_ranges)
    ]

    current_segment = (
        relative_positions[0] +
        (relative_positions[1] - 1) * segment_counts[0] +
        (relative_positions[2] - 1) * segment_counts[0] * segment_counts[1] +
        (relative_positions[3] - 1) * segment_counts[0] * segment_counts[1] * segment_counts[2]
    )

    return current_segment

# Step 5: Train the DQN agent
episode_last_50_mean_gt_400 = None # early stopping of training when mean total reward reaches 400
for episode in range(num_episodes):
    epsilon_values.append(agent.epsilon)  # Record epsilon value
    state, info = env.reset() # initial state: s0 = initial position, velocity of Cart and Pole after reset
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0
    agent.decay_action()        # decay epsilon in each episode
    for time_step in range(500):  # number of time steps
        env.render()
        SegmentNr = compute_segment_number_cartpole(state.squeeze().tolist())
        action = agent.select_action(state, action_data_structure.segments[SegmentNr])  # action selection 0: push left, 1: push right
        action_data_structure.record_action(SegmentNr, action)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        ExperienceReplayBuffer.storeExperiment(Experience(state, action, next_state, reward))
        total_reward += reward.item()

        #if state is not None:
            #SegmentNr= compute_segment_number_cartpole(state.squeeze().tolist())
            #action_data_structure.record_action(SegmentNr, action)

        state = next_state

        if len(ExperienceReplayBuffer.memory) >= batch_size:
            MiniBatch = ExperienceReplayBuffer.CreateMiniBatch(batch_size)
            batch = Experience(*zip(*MiniBatch))
            # Compute a mask of non-terminal states and concatenate the batch elements
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=device, dtype=torch.bool)
            next_states = torch.cat([s for s in batch.next_state
                                               if s is not None])
            states= torch.cat(batch.state)
            actions = torch.cat(batch.action)
            rewards = torch.cat(batch.reward)

            # Compute Q (s_t, a) values for each batch state s_t - according to policy_net
            q_values = agent.policy_net(states).gather(1, actions)

            # Compute Q(s_{t+1}) for all next states.
            next_q_values = torch.zeros(batch_size, device=device)
            with torch.no_grad():
                next_q_values[non_final_mask] = agent.policy_net(next_states).max(1)[0]

            # Compute the target Q-values using the Bellman equation
            expected_q_values = (next_q_values * gamma) + rewards

            #criterion = nn.HuberLoss()    # Huber
            criterion = nn.MSELoss()    # MSE
            loss = criterion(q_values, expected_q_values.unsqueeze(1))
            #loss = criterion(q_values, expected_state_action_values)

            # Optimize the model
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
########## Training loop - End ##########

        # Step 6: Update target network
        target_net_state_dict = agent.target_net.state_dict()
        policy_net_state_dict = agent.policy_net.state_dict()
        if episode % TAU_hardUpdate == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        if done:
            # Plot_Result(episode_rewards, epsilon_values, epsilon_crossed50Percent)
            break

        if terminated or done:
            break

    # Get the count of unique taken actions and the length of visited segments
    unique_actions_count = action_data_structure.get_taken_actions_count()
    visited_segments_count = action_data_structure.get_visited_segments_count()

    # Append data to lists for plotting
    taken_actions_list.append(unique_actions_count)
    visited_segments_list.append(visited_segments_count)
    exploration_metric1_list.append(100*(unique_actions_count)/(10000 * action_space_size))
    exploration_metric2_list.append(100*(unique_actions_count)/(visited_segments_count * action_space_size))

    episode_rewards.append((episode, total_reward))
    last_50_mean= calculate_last_50_mean(episode_rewards)
    last_50_means.append(last_50_mean)
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, last 50 mean reward: {last_50_mean}")

    print(len(action_data_structure.visited_segments))

    if last_50_mean is not None and last_50_mean >= 400:
        episode_last_50_mean_gt_400= episode
        #break

#------------------------------------------------------
# Step 7: save the trained network
# Save the model
torch.save(agent.policy_net.state_dict(), model_save_directory)
print("Model saved.")

# Step 8: reload the trained network
# Load the model
agent.policy_net.load_state_dict(torch.load(model_save_directory))
print("Model loaded.")

# Optional: Move the model to GPU if available
if torch.cuda.is_available():
    agent.policy_net.cuda()

# Step 9: Performance evaluation of DQN Agent
total_reward = 0
for _ in range(100):  # Run the environment 10 times for evaluation
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for _ in range(500):  # You can adjust the maximum number of time steps
        action = agent.exploit(state)  # 100% exploitation
        observation, reward, terminated, truncated, _ = env.step(action.item())
        # reward = torch.tensor([reward], device=device)
        done = terminated or truncated
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        total_reward += reward
        state = next_state

        if done:
            break
average_reward = total_reward / 100
print(f"Average Reward over 100 episodes: {average_reward}")

#Plot_Result(episode_rewards, epsilon_values, epsilon_crossed50Percent, last_50_means, episode_last_50_mean_gt_400, average_reward)
plot_exploration_metrics_result(exploration_metric1_list,exploration_metric2_list,taken_actions_list, visited_segments_list, epsilon_values,epsilon_crossed50Percent)
plot_visits_bar_chart(action_data_structure.segments, action_data_structure.visited_segments)
env.close()
