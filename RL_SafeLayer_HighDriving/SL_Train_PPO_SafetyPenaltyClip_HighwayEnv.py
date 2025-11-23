"""
*******************************************************************************
                            [SL_Train_PPO_SafetyPenaltyClip_HighwayEnv]
*******************************************************************************
File: SL_Train_PPO_SafetyPenaltyClip_HighwayEnv.py
Author: Farshad Mirzarazi
Date: 28.01.2024
Description:
------------
This source code is part of the author's Dissertation. (Chapter 8)

PPO  Reinforcement Learning Highway-Env Environment
Training RL PPO Agent on Highway-Env Environment: Tensorboard Log Files

This code utilizes Stable-Baselines3 (SB3), an open-source reinforcement learning library.
DLR-RM/stable-baselines3 is licensed under the MIT License
For more information about SB3, visit: https://github.com/DLR-RM/stable-baselines3

# Library Versions:
# -----------------
# gymnasium                 0.29.1
# matplotlib                3.8.2
# numpy                     1.24.3
# stable-baselines3         2.2.1
# pytorch                   2.1.2
Disclaimer:
------------
This code is provided "as is," without warranty of any kind.
The author is not responsible for any consequences arising from the use of this code.
*******************************************************************************
"""
# ----------------
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from stable_baselines3 import DQN, A2C, PPO, DDPG

from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
import torch as th
from torch.nn import functional as F
# Folder to save model, store TF_Logs, and videos
model_dir = "F:/Brunel/02_SourceCodes/HighwayDriving/Highway_Models/HighwayPPO"
'''
model_dir = "F:/Brunel/02_SourceCodes/HighwayDriving/Highway_Models/HighwayA2C"
model_dir = "F:/Brunel/02_SourceCodes/HighwayDriving/Highway_Models/HighwayDQN"
model_dir = "F:/Brunel/02_SourceCodes/HighwayDriving/Highway_Models/HighwayDDPG"
'''
tensorboard_log = f"{model_dir}/tensorboard_logs"
video_folder = f"{model_dir}/videos/"
XRange = 100
YRange = 10

# Create custom PPO model with Safety Penalty Clipped surrogate objective function
class PPO_SL_clipped_Objective_Function(PPO):
    def __init__(self, *args, safety_coeff=0.1, **kwargs):
        super(PPO_SL_clipped_Objective_Function, self).__init__(*args, **kwargs)
        self.safety_coeff = safety_coeff

    def calculate_safety_penalty(self, observations, threshold=10.0):
        # Extract x, y-coordinates of Ego & target vehicles from Agent observations (Target values relative to Ego!!)
        x_values = observations[:, :, 1]  # Assuming the second dimension represents the batch size
        #y_values = observations[:, :, 2]

        Ego_Target_Long = x_values[:, 1:5] * XRange
        #Ego_Target_Lat = y_values[:, 1:5] * YRange

        # Calculate Safety KPI1 , ignoring negative values
        safety_kpi1_values, _ = th.min(th.where(Ego_Target_Long < 0, th.inf, Ego_Target_Long), dim=-1)


        # Calculate Safety Penalty based on Safety KPI1
        safety_penalty = th.where(safety_kpi1_values > threshold, th.tensor(0.0),
                                  (safety_kpi1_values - threshold) * self.safety_coeff)

        return safety_penalty


    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # Access the current observation from the rollout buffer
                current_observations = rollout_data.observations

                #  safety penalty calculation
                safety_penalty = self.calculate_safety_penalty(current_observations, 10)

                # Clipped surrogate loss with safety penalty
                policy_loss_1 = advantages * ratio - safety_penalty
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range) - safety_penalty
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


# Create environment
env = gym.make("highway-fast-v0", render_mode='rgb_array')
env.config["normalized"] = False

# Use DummyVecEnv for a single environment
env = DummyVecEnv([lambda: env])

# Create environment
env = gym.make("highway-fast-v0", render_mode='rgb_array')
env.config["normalized"] = False
env.reset()
env.render()

TIMESTEPS = 2e4

# Instantiate the custom PPO with safety Clipping objective function
model = PPO_SL_clipped_Objective_Function('MlpPolicy', env,
                                                 policy_kwargs=dict(net_arch=[256, 256]),
                                                 learning_rate=2e-3,
                                                 batch_size=64,
                                                 verbose=1,
                                                 tensorboard_log=tensorboard_log,
                                                 safety_coeff=0.5)

# model = PPO('MlpPolicy', env,
#             policy_kwargs=dict(net_arch=[256, 256]),
#             learning_rate=2e-3,
#             batch_size=64,
#             verbose=1,
#             tensorboard_log=tensorboard_log)
# Train the custom PPO

for i in range(1, 3):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPOlogs",  progress_bar=True)
    # Save the agent
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_prefix = "SafetyClip_Lambda-0.5"
    model.save(f"{model_dir}/{model_prefix}_{TIMESTEPS * i}")  # save the model every TIMESTEPS steps
# delete trained model to demonstrate loading
del model
# -------------------------------------------------------------------------------------
env = gym.make("highway-fast-v0", render_mode='rgb_array')
env.config["normalized"] = False
env.reset()

# Load the trained agent
vec_env = DummyVecEnv([lambda: env])
model_path = f"{model_dir}/40000.0"
model = PPO.load(model_path, env=vec_env)

# Record the video starting at the first step

video_length = 1000

# Generate a timestamp string
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
# Use the timestamp in the video file name
video_name = f"PPO-highway_{timestamp}"

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

