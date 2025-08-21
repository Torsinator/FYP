"""
her_ddpg_lunar.py
Goal-conditioned DDPG + HER (future strategy) for LunarLanderContinuous-v2.

Dependencies:
    pip install gym==0.21.0 numpy torch

Notes:
- This is a compact, readable implementation intended as a starting point.
- Tweak hyperparameters, network sizes, and reward shaping for best results.
"""
import gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time

# -------------------------
#  ENV WRAPPER: GoalEnv
# -------------------------
class GoalEnv(gym.Wrapper):
    """
    Wraps an environment so observations are concatenated with a goal.
    - achieved_goal: the underlying observation (we treat whole obs as goal-capable)
    - desired_goal: user-provided; here we sample random goal states from state-space
    Reward is negative weighted Euclidean distance between current achieved and desired goal.
    """
    def __init__(self, env, goal_sampler=None, weights=None):
        super().__init__(env)
        self.env = env
        self.obs_dim = env.observation_space.shape[0]  # 8 for lunar
        self.goal_dim = self.obs_dim
        self.weights = np.array(weights) if weights is not None else np.ones(self.goal_dim)
        # observation is [obs, goal]
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([env.observation_space.low, env.observation_space.low]),
            high=np.concatenate([env.observation_space.high, env.observation_space.high]),
            dtype=np.float32
        )
        self.action_space = env.action_space
        self.goal_sampler = goal_sampler or self._random_goal_from_space

        # bounds for sampling goals (small hack): sample positions within observed ranges
        self.obs_low = env.observation_space.low
        self.obs_high = env.observation_space.high

    def _random_goal_from_space(self):
        return np.random.uniform(self.obs_low, self.obs_high)

    def reset(self, desired_goal=None):
        obs = self.env.reset()
        self.achieved_goal = obs.copy()
        self.desired_goal = desired_goal if desired_goal is not None else self.goal_sampler()
        return np.concatenate([obs, self.desired_goal]).astype(np.float32)

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        self.achieved_goal = obs.copy()
        reward = self.compute_reward(self.achieved_goal, self.desired_goal)
        return np.concatenate([obs, self.desired_goal]).astype(np.float32), reward, done, info

    def compute_reward(self, achieved, desired):
        # Weighted Euclidean distance; sign flipped so higher is better if we negate.
        diff = (achieved - desired) * self.weights
        dist = np.linalg.norm(diff)
        # convert to reward that is denser than pure negative distance:
        # Use -dist (agent minimizes distance) but scaled
        return -dist

    def get_achieved_goal(self):
        return self.achieved_goal.copy()

    def set_goal(self, goal):
        self.desired_goal = goal.copy()


# -------------------------
#  NETWORKS (PyTorch)
# -------------------------
def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes)-1):
        act = activation if i < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, obs_dim, goal_dim, act_dim, hidden=(256,256)):
        super().__init__()
        self.net = mlp([obs_dim + goal_dim] + list(hidden) + [act_dim], activation=nn.ReLU, output_activation=nn.Tanh)

    def forward(self, obs, goal):
        x = torch.cat([obs, goal], dim=-1)
        return self.net(x)  # outputs in [-1,1], scale externally

class Critic(nn.Module):
    def __init__(self, obs_dim, goal_dim, act_dim, hidden=(256,256)):
        super().__init__()
        self.net = mlp([obs_dim + goal_dim + act_dim] + list(hidden) + [1], activation=nn.ReLU, output_activation=nn.Identity)

    def forward(self, obs, goal, act):
        x = torch.cat([obs, goal, act], dim=-1)
        return self.net(x).squeeze(-1)


# -------------------------
#  REPLAY BUFFER + HER
# -------------------------
class HERReplayBuffer:
    def __init__(self, max_episodes=2000, episode_length=1000, her_k=4):
        """
        Stores whole episodes so we can do HER relabeling (future strategy).
        - max_episodes: how many episodes to keep
        - her_k: number of future goals to sample per transition
        """
        self.max_episodes = max_episodes
        self.episode_length = episode_length
        self.her_k = her_k
        self.episodes = deque(maxlen=max_episodes)

    def add_episode(self, episode_transitions):
        """
        episode_transitions: list of dicts with keys:
          obs, action, reward, next_obs, done, achieved_goal
        achieved_goal should be the achieved goal at that timestep (the raw obs)
        """
        self.episodes.append(episode_transitions)

    def sample(self, batch_size):
        """
        Sample a minibatch with HER relabeling applied on the fly.
        """
        states = []
        actions = []
        rewards = []
        next_states = []
        goals = []
        dones = []

        while len(states) < batch_size:
            # pick an episode
            ep = random.choice(self.episodes)
            T = len(ep)
            t = random.randrange(T)
            trans = ep[t]

            # with probability p = her_k/(her_k+1) we relabel with a future achieved goal (future strategy)
            if random.random() < (self.her_k / (self.her_k + 1)):
                # sample one future time index
                if t < T - 1:
                    future_idx = random.randrange(t+1, T)
                    new_goal = ep[future_idx]['achieved_goal'].copy()
                else:
                    new_goal = ep[t]['achieved_goal'].copy()
            else:
                new_goal = ep[t]['achieved_goal'].copy()  # fallback to itself

            # recompute reward relative to new_goal. We'll assume wrapper reward = -distance
            achieved_next = ep[t]['next_achieved'].copy()
            diff = achieved_next - new_goal
            rew = -np.linalg.norm(diff)

            states.append(ep[t]['obs'])
            actions.append(ep[t]['action'])
            rewards.append(rew)
            next_states.append(ep[t]['next_obs'])
            goals.append(new_goal)
            dones.append(ep[t]['done'])

        # to numpy arrays
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.float32),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(goals, dtype=np.float32),
                np.array(dones, dtype=np.float32))


# -------------------------
#  DDPG AGENT
# -------------------------
class DDPGAgent:
    def __init__(self, obs_dim, goal_dim, act_dim, act_limit,
                 actor_hidden=(256,256), critic_hidden=(256,256),
                 actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.005, device='cpu'):
        self.device = device
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.act_dim = act_dim
        self.act_limit = act_limit

        self.actor = Actor(obs_dim, goal_dim, act_dim, actor_hidden).to(self.device)
        self.actor_target = Actor(obs_dim, goal_dim, act_dim, actor_hidden).to(self.device)
        self.critic = Critic(obs_dim, goal_dim, act_dim, critic_hidden).to(self.device)
        self.critic_target = Critic(obs_dim, goal_dim, act_dim, critic_hidden).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau

    def act(self, obs, goal, noise_scale=0.1):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        goal_t = torch.as_tensor(goal, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(obs_t, goal_t).cpu().numpy()[0]
        a = a * self.act_limit
        a = a + noise_scale * np.random.randn(*a.shape)
        return np.clip(a, -self.act_limit, self.act_limit)

    def update(self, batch, batch_size=256):
        states, actions, rewards, next_states, goals, dones = batch
        # states and next_states are concatenated obs+goal in our buffer design; but we stored obs and goals separately
        obs = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rews = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_obs = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        goals_t = torch.as_tensor(goals, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # split obs and old goal (our obs in env = obs_only concatenated with goal at reset time)
        # In buffer we saved obs_only; here obs is obs_only
        obs_only = obs
        next_obs_only = next_obs

        # Critic target:
        with torch.no_grad():
            a_next = self.actor_target(next_obs_only, goals_t)
            a_next = a_next * self.act_limit
            q_next = self.critic_target(next_obs_only, goals_t, a_next).unsqueeze(-1)
            q_target = rews.unsqueeze(-1) + (1.0 - dones_t) * self.gamma * q_next

        q_val = self.critic(obs_only, goals_t, acts).unsqueeze(-1)
        critic_loss = F.mse_loss(q_val, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss (maximize Q)
        a_pred = self.actor(obs_only, goals_t)
        a_pred = a_pred * self.act_limit
        actor_loss = -self.critic(obs_only, goals_t, a_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # soft update targets
        for p, p_target in zip(self.actor.parameters(), self.actor_target.parameters()):
            p_target.data.copy_(self.tau * p.data + (1 - self.tau) * p_target.data)
        for p, p_target in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_target.data.copy_(self.tau * p.data + (1 - self.tau) * p_target.data)

        return critic_loss.item(), actor_loss.item()


# -------------------------
#  TRAINING LOOP
# -------------------------
def train(seed=0,
          num_episodes=2000,
          max_episode_steps=500,
          batch_size=256,
          start_epochs=5,
          updates_per_episode=50,
          her_k=4,
          device='cpu'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    base_env = gym.make('LunarLanderContinuous-v2')
    env = GoalEnv(base_env, weights=[1.0, 1.0, 0.5, 0.5, 0.5, 0.3, 0.0, 0.0])
    obs_dim = base_env.observation_space.shape[0]  # 8
    goal_dim = obs_dim
    act_dim = base_env.action_space.shape[0]
    act_limit = float(base_env.action_space.high[0])

    agent = DDPGAgent(obs_dim, goal_dim, act_dim, act_limit, device=device)

    buffer = HERReplayBuffer(max_episodes=2000, episode_length=max_episode_steps, her_k=her_k)

    total_steps = 0
    print_every = 10
    t0 = time.time()

    for ep in range(1, num_episodes + 1):
        # Sample a new random goal each episode
        desired_goal = env.goal_sampler()
        obs_with_goal = env.reset(desired_goal=desired_goal)
        obs_only = obs_with_goal[:obs_dim]
        episode = []

        ep_reward = 0.0
        for t in range(max_episode_steps):
            noise_scale = max(0.1, 0.4 * (1 - ep / num_episodes))  # decay exploration
            action = agent.act(obs_only, desired_goal, noise_scale=noise_scale)
            # step env
            next_with_goal, reward, done, info = env.step(action)
            next_obs_only = next_with_goal[:obs_dim]
            achieved = env.get_achieved_goal()
            # store transition (we store obs_only and next_only and achieved_goal)
            episode.append({
                'obs': obs_only.copy(),
                'action': action.copy(),
                'reward': reward,
                'next_obs': next_obs_only.copy(),
                'done': float(done),
                'achieved_goal': achieved.copy(),
                'next_achieved': achieved.copy()
            })
            obs_only = next_obs_only
            ep_reward += reward
            total_steps += 1
            if done:
                break

        # add episode to buffer (then HER will relabel on sampling)
        buffer.add_episode(episode)

        # training updates (only after we have some episodes)
        if len(buffer.episodes) > 2:
            for _ in range(updates_per_episode):
                batch = buffer.sample(batch_size)
                critic_loss, actor_loss = agent.update(batch, batch_size=batch_size)

        if ep % print_every == 0:
            elapsed = time.time() - t0
            print(f"Ep {ep}/{num_episodes}  EpRew {ep_reward:.2f}  EpisodesStored {len(buffer.episodes)}  Steps {total_steps}  Time {elapsed/60:.1f}m")

    print("Training finished.")
    return agent, env


if __name__ == "__main__":
    # Run training
    agent, env = train(seed=42, num_episodes=600, max_episode_steps=500, batch_size=256, updates_per_episode=40, her_k=4, device='cpu')
    # Quick demo: pick a random goal and run the policy
    g = env.goal_sampler()
    obs = env.reset(desired_goal=g)
    obs_only = obs[:env.obs_dim]
    total = 0.0
    for _ in range(500):
        a = agent.act(obs_only, g, noise_scale=0.0)
        nxt, r, done, _ = env.step(a)
        obs_only = nxt[:env.obs_dim]
        total += r
        if done:
            break
    print("Demo reward:", total)
