import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

# =============================================================================
# Device Setup: Use GPU if available.
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# ContinuousAgent: A network that outputs a scalar action from an observation.
#
# For the main thruster, we use a "sigmoid" transformation to map
# to [0,1]. For the side thruster, we use "tanh" to map to [-1,1].
#
# A (minimal) change-of-variables correction is applied to the log probabilities.
# =============================================================================
class ContinuousAgent(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128, output_transformation="linear"):
        """
        Parameters:
          obs_dim: Dimensionality of environment observation.
          hidden_dim: Size of the hidden layers.
          output_transformation: "sigmoid" (maps to [0,1]),
                                 "tanh" (maps to [-1,1]), or "linear".
        """
        super(ContinuousAgent, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, 1)  # scalar output
        self.log_std = nn.Parameter(torch.zeros(1))
        self.output_transformation = output_transformation

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        return mean, self.log_std.expand_as(mean)

    def get_action(self, obs):
        """
        Args:
          obs: A tensor of shape (batch, obs_dim)
        Returns:
          final_action: A scalar in the desired range (wrapped in a tensor of shape (batch,1)).
          log_prob: The adjusted log probability of that action.
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        base_dist = torch.distributions.Normal(mean, std)
        # Reparameterized sample.
        raw_action = base_dist.rsample()
        base_log_prob = base_dist.log_prob(raw_action)

        if self.output_transformation == "sigmoid":
            # Main thruster: map to [0,1]
            final_action = torch.sigmoid(raw_action)
            # Correction: derivative of sigmoid = s*(1-s)
            correction = torch.log(final_action * (1 - final_action) + 1e-6)
        elif self.output_transformation == "tanh":
            # Side thruster: map to [-1,1]
            final_action = torch.tanh(raw_action)
            # Correction: derivative of tanh = 1 - tanh(x)^2
            correction = torch.log(1 - final_action.pow(2) + 1e-6)
        else:
            final_action = raw_action
            correction = 0.0

        # Adjust log probability with change-of-variable term.
        log_prob = base_log_prob - correction
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return final_action, log_prob

# =============================================================================
# Helper Function: Compute Discounted Returns (normalized)
# =============================================================================
def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    returns = (returns - returns.mean()) / (returns.std() + 1e-6)
    return returns

# =============================================================================
# Training Function: Train agents on LunarLanderContinuous-v2.
#
# The continuous action space of LunarLanderContinuous-v2 is 2 dimensional:
#   - action[0] (main thrust) should be in [0,1]
#   - action[1] (side thrust) should be in [-1,1]
#
# We build two separate agents that share the same observation.
# They learn from the common reward signal using a REINFORCE update.
# =============================================================================
def train_agents(num_episodes=200, gamma=0.99, lr=1e-3):
    env = gym.make("LunarLanderContinuous-v3", render_mode=None)
    obs_dim = env.observation_space.shape[0]  # typically 8 dimensions

    # Instantiate agents.
    main_agent = ContinuousAgent(obs_dim, hidden_dim=128, output_transformation="sigmoid").to(device)
    side_agent = ContinuousAgent(obs_dim, hidden_dim=128, output_transformation="tanh").to(device)

    # Separate optimizers.
    optimizer_main = optim.Adam(main_agent.parameters(), lr=lr)
    optimizer_side = optim.Adam(side_agent.parameters(), lr=lr)

    for episode in range(num_episodes):
        obs, info = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        rewards = []
        log_probs_main = []
        log_probs_side = []
        total_reward = 0
        t = 0

        while not done:
            # Each agent selects an action given the same observation.
            main_action, log_prob_main = main_agent.get_action(obs)
            side_action, log_prob_side = side_agent.get_action(obs)
            # Combine outputs into a 2D action.
            action = torch.cat([main_action, side_action], dim=-1).squeeze(0)
            action_np = action.cpu().detach().numpy()

            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            total_reward += reward
            rewards.append(reward)
            log_probs_main.append(log_prob_main)
            log_probs_side.append(log_prob_side)

            obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            t += 1

        # Compute returns and policy gradients.
        returns = compute_returns(rewards, gamma)
        loss_main = 0
        loss_side = 0
        for lp_main, lp_side, R in zip(log_probs_main, log_probs_side, returns):
            loss_main -= lp_main * R
            loss_side -= lp_side * R

        optimizer_main.zero_grad()
        loss_main.backward()
        optimizer_main.step()

        optimizer_side.zero_grad()
        loss_side.backward()
        optimizer_side.step()

        print(f"Episode {episode+1}/{num_episodes} | Total Reward: {total_reward:.2f} | Length: {t}")

    env.close()
    # Return the trained models.
    return main_agent, side_agent

# =============================================================================
# Recording Function: Run one episode and record a video.
#
# This wraps the continuous Lunar Lander environment with RecordVideo,
# runs a single episode using the provided agents, and saves the video.
# =============================================================================
def record_final_run(main_agent, side_agent):
    # Create folder for storing videos.
    video_folder = "./videos"
    os.makedirs(video_folder, exist_ok=True)
    # Create the environment with "rgb_array" render mode.
    env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True,
                      name_prefix="final_run")

    obs_dim = env.observation_space.shape[0]
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    done = False
    total_reward = 0
    t = 0

    while not done:
        main_action, _ = main_agent.get_action(obs)
        side_action, _ = side_agent.get_action(obs)
        action = torch.cat([main_action, side_action], dim=-1).squeeze(0)
        action_np = action.cpu().detach().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        total_reward += reward
        obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
        t += 1

    env.close()
    print(f"Final Run Recorded | Total Reward: {total_reward:.2f} | Length: {t}")
    print(f"Check the video files in folder: {video_folder}")

# =============================================================================
# Main Execution: Training followed by recording a final run.
# =============================================================================
if __name__ == '__main__':
    # Train the agents.
    print("Starting training...")
    trained_main_agent, trained_side_agent = train_agents(num_episodes=1000, gamma=0.99, lr=1e-3)

    # Optionally, save the trained models here for future use.
    # torch.save(trained_main_agent.state_dict(), "main_agent.pth")
    # torch.save(trained_side_agent.state_dict(), "side_agent.pth")

    # Record a final run using the trained agents.
    print("\nStarting final run recording...")
    record_final_run(trained_main_agent, trained_side_agent)
