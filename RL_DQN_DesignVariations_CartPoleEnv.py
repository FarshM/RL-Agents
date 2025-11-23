"""
*******************************************************************************
                            [RL_DQN_DesignVariations_CartPoleEnv]
*******************************************************************************
File: RL_DQN_DesignVariations_CartPoleEnv.py
Author: Farshad Mirzarazi
Date: 28.01.2024
Description:
------------
This source code is part of the author's Dissertation. (Chapter 5)

DQN  Reinforcement Learning CartPole Environment
Design Variations: Basic DQN, MSE/Hoss Loss function, 2 / 3 Hidden layer FC Network,
                   Hard / soft update of target network, with or without gradient clipping, ...
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

variatoin = "ReplayMemory_size 1000 | "
"""
Hyperparameters: 
"""
replayMem_size = 1000
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
EPS_DECAY = 200
# Optimizer learning rate
LearningRate = 0.0001

num_episodes = 1000
max_reward = 500

DecayCount = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


episode_rewards = []
epsilon_values = []
epsilon_crossed50Percent = None
exploration_percentage = []

last_50_means = []  # Array to store last 50 means of total rewards

#  a list to store loss values and update values during training
loss_values = []
update_values = []

# Custom hook function to extract gradients during backward pass
def gradient_hook(module, grad_input, grad_output):
    # Assuming you want to monitor the first layer's weights
    weights_gradients = grad_input[0]
    update_values.append(torch.mean(weights_gradients).item())


# Step 1: Define the Gym environment
env = gym.make('CartPole-v1', render_mode="rgb_array")
"""
Episode Termination:
Pole Angle is more than ±12°
Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
Episode length is greater than 200
Reward :
Reward is 1 for every step taken, including the termination step. The threshold is 475 for v1.
Actions:
Type: Discrete(2)
0	Push cart to the left
1	Push cart to the right
Starting State:
All observations are assigned a uniform random value between ±0.05.
"""

Experience = namedtuple('Experience',
                        ('state', 'action', 'next_state', 'reward')
                        )

input_size = env.observation_space.shape[0]
output_size = env.action_space.n

# Step 2: Define the DQN model
class DQN_BN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.layer2 = nn.Linear(32, 64)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.layer3 = nn.Linear(64, 128)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, n_actions)

    # Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.batch_norm1(self.layer1(x)))
        x = F.relu(self.batch_norm2(self.layer2(x)))
        x = F.relu(self.batch_norm3(self.layer3(x)))
        return self.out(x)
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

# Step 2B: Define NN Initialization
def weights_init_uniform(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
def weights_init_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
def weights_init_he(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# Step 3: Define the Replay Memory
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

# Step 4: Define the DQN agent
class DQNAgent:
    def __init__(self, input_size, output_size, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # self.policy_net = DQN(input_size, output_size)
        # self.target_net = DQN(input_size, output_size)
        # self.target_net.load_state_dict(self.policy_net.state_dict())
        # # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.policy_net = DQN(input_size, output_size)
        self.policy_net.apply(weights_init_he)  # Apply He initialization
        self.target_net = DQN(input_size, output_size)
        self._hard_update_target_net()

    def _hard_update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state):
        # Decay epsilon
        global DecayCount
        global epsilon_crossed50Percent
        randNr = random.random()
        self.epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * DecayCount / EPS_DECAY)
        if self.epsilon <= 0.5 and epsilon_crossed50Percent is None:
            epsilon_crossed50Percent = episode
        DecayCount += 1
        if self.epsilon > randNr:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)

    def update_target_model(self):
        model_state_dict = self.policy_net.state_dict()
        target_model_state_dict = {}
        for key, value in model_state_dict.items():
            # Convert numpy arrays to tensors
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value).float()
            target_model_state_dict[key] = value
        self.target_net.load_state_dict(target_model_state_dict)



# Step 4B: Initialize the DQN agent and Memory
agent = DQNAgent(input_size, output_size)
# Attach the hook to the first layer of your DQN
#hook_handle = agent.policy_net.layer1.register_full_backward_hook(gradient_hook)

# TODO variation of agent optimizer
# Instantiate the optimizers with learning rate
# SGD optimizer
optimizer_sgd = optim.SGD(agent.policy_net.parameters(), lr=LearningRate)
# Adam optimizer
optimizer_adam = optim.Adam(agent.policy_net.parameters(), lr=LearningRate)
# RMSprop optimizer
optimizer_rmsprop = optim.RMSprop(agent.policy_net.parameters(), lr=LearningRate)
# choose which optimizer to use
# agent.optimizer = optimizer_sgd
agent.optimizer = optimizer_adam
# agent.optimizer = optimizer_rmsprop


ExperienceReplayBuffer = ReplayMemeory(replayMem_size)
# Helper functions
# Define Plot_Result function

# Step 10: Plot the Results
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

        # Specify the directory
        save_directory = r'F:\Brunel\MyWritings\Chapter3_LitReview_ReinforcementLearning\Figures_DQN_Variations'

        # Save the plot with a timestamp in the filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f'plot_{timestamp}.png'
        file_path = os.path.join(save_directory, filename)
        plt.savefig(file_path)

        plt.show()
    else:
        print("No episode durations recorded. Cannot plot.")


# Define a function to calculate the mean of the last 50 values
def calculate_last_50_mean_wrong(rewards):
    if len(rewards) >= 50:
        last_50_mean = np.mean(rewards[-50:])
    else:
        last_50_mean = np.mean(rewards)
    return last_50_mean

def calculate_last_50_mean(episode_rewards):
    if len(episode_rewards) >= 50:
        last_50_rewards = episode_rewards[-50:]
        last_50_mean = sum([reward for _, reward in last_50_rewards]) / 50
        return last_50_mean
    else:
        return None
# ...
# Step 5: Train the DQN agent
episode_last_50_mean_gt_400 = None # early stopping of training when mean total reward reaches 400
for episode in range(num_episodes):
    epsilon_values.append(agent.epsilon)  # Record epsilon value
    state, info = env.reset() # initial state: s0 = initial position, velocity of Cart and Pole after reset
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0
    for time_step in range(500):  # number of time steps
        env.render()
        action = agent.select_action(state)  # action selection 0: push left, 1: push right
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        ExperienceReplayBuffer.storeExperiment(Experience(state, action, next_state, reward))
        total_reward += reward.item()

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

            # TODO Loss function variations
            # Compute loss function Var.1 Huber loss, Var.2 MSE
            #criterion = nn.SmoothL1Loss() # Huber loss

            #criterion = nn.HuberLoss()    # Huber
            criterion = nn.MSELoss()    # MSE
            loss = criterion(q_values, expected_q_values.unsqueeze(1))
            #loss = criterion(q_values, expected_state_action_values)

            # Optimize the model
            agent.optimizer.zero_grad()
            loss.backward()
            # TODO with or without gradient clipping
            # In-place gradient clipping
            #torch.nn.utils.clip_grad_value_(agent.policy_net.parameters(), 100)
            agent.optimizer.step()
########## Training loop - End ##########
        # Step 6: Update target network
        # TODO soft or hard update of target network
        # Soft update of the target network's weights
        # Var.1 Soft update
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = agent.target_net.state_dict()
        policy_net_state_dict = agent.policy_net.state_dict()
        # for key in policy_net_state_dict:
        #     target_net_state_dict[key] = policy_net_state_dict[key] * TAU_softUpdate + target_net_state_dict[key] * (1 - TAU_softUpdate)
        # agent.target_net.load_state_dict(target_net_state_dict)
        # Var. 2 Hard update
        if episode % TAU_hardUpdate == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        if done:
            # Plot_Result(episode_rewards, epsilon_values, epsilon_crossed50Percent)
            break

        if terminated or done:
            break

    episode_rewards.append((episode, total_reward))
    last_50_mean= calculate_last_50_mean(episode_rewards)
    last_50_means.append(last_50_mean)
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, last 50 mean reward: {last_50_mean}")

    if last_50_mean is not None and last_50_mean >= 400:
        episode_last_50_mean_gt_400= episode
        break


########################################################################
# Step 7: save the trained network
# Save the model
torch.save(agent.policy_net.state_dict(), 'F:\\Brunel\\02_SourceCodes\\DQNVariations\\online_model.pth')
print("Model saved.")

# Step 8: reload the trained network
# Load the model
agent.policy_net.load_state_dict(torch.load('F:\\Brunel\\02_SourceCodes\\DQNVariations\\online_model.pth'))
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
        action = agent.select_action(state)  # 100% exploitation
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

Plot_Result(episode_rewards, epsilon_values, epsilon_crossed50Percent, last_50_means, episode_last_50_mean_gt_400, average_reward)

# Step 11: Close the environment
env.close()
