import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random


# ------------------------------
# Networks
# ------------------------------

def mlp(input_dim, output_dim, hidden=[256, 256], activation=nn.ReLU):
    layers = []
    prev = input_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim)
        self.act_limit = act_limit

    def forward(self, obs):
        mu_logstd = self.net(obs)
        mu, log_std = mu_logstd.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        return mu, std

    def sample(self, obs):
        mu, std = self(obs)
        dist = torch.distributions.Normal(mu, std)
        x_t = dist.rsample()  # reparameterized sample
        a = torch.tanh(x_t) * self.act_limit
        log_prob = dist.log_prob(x_t).sum(dim=-1, keepdim=True)
        # Correction for Tanh squashing
        log_prob -= torch.log(1 - torch.tanh(x_t).pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return a, log_prob

    def act(self, obs, deterministic=False):
        mu, std = self(obs)
        if deterministic:
            a = torch.tanh(mu) * self.act_limit
        else:
            dist = torch.distributions.Normal(mu, std)
            x_t = dist.sample()
            a = torch.tanh(x_t) * self.act_limit
        return a.detach().cpu().numpy()


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1)
        self.q2 = mlp(obs_dim + act_dim, 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


# ------------------------------
# Replay Buffer with HER
# ------------------------------

class HERReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim,
                 future_k=4, her_ratio=0.8):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.future_k = future_k
        self.her_ratio = her_ratio

    def add_episode(self, episode):
        """Episode = list of (obs, act, rew, next_obs, done, info)."""
        self.buffer.append(episode)

    def sample(self, batch_size):
        obs, act, rew, next_obs, done = [], [], [], [], []

        while len(obs) < batch_size:
            ep = random.choice(self.buffer)
            t = np.random.randint(len(ep))
            o, a, r, o_next, d, info = ep[t]

            if random.random() < self.her_ratio and not info["success"]:
                # relabel goal
                future_t = np.random.randint(t, len(ep))
                future_state = ep[future_t][0][:self.obs_dim]
                new_goal = future_state[:3]  # x, y, theta

                def relabel(obs_vec):
                    return np.concatenate([obs_vec[:self.obs_dim], new_goal])

                o = relabel(o)
                o_next = relabel(o_next)

                pos_err = o_next[:2] - new_goal[:2]
                ang_err = (o_next[2] - new_goal[2] + np.pi) % (2*np.pi) - np.pi
                r = -(np.linalg.norm(pos_err) + abs(ang_err))
                d = (np.linalg.norm(pos_err) < 0.05 and abs(ang_err) < 0.1)

            obs.append(o)
            act.append(a)
            rew.append(r)
            next_obs.append(o_next)
            done.append(d)

        return (torch.tensor(obs, dtype=torch.float32),
                torch.tensor(act, dtype=torch.float32),
                torch.tensor(rew, dtype=torch.float32).unsqueeze(-1),
                torch.tensor(next_obs, dtype=torch.float32),
                torch.tensor(done, dtype=torch.float32).unsqueeze(-1))


# ------------------------------
# SAC Agent
# ------------------------------

class SAC:
    def __init__(self, obs_dim, act_dim, act_limit,
                 gamma=0.99, tau=0.005, alpha=0.2,
                 lr=3e-4):
        self.actor = Actor(obs_dim, act_dim, act_limit)
        self.critic = Critic(obs_dim, act_dim)
        self.critic_target = Critic(obs_dim, act_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.act_limit = act_limit

    def update(self, buffer, batch_size=256):
        obs, act, rew, next_obs, done = buffer.sample(batch_size)

        # Critic update
        with torch.no_grad():
            next_a, next_logp = self.actor.sample(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_a)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_logp
            target_q = rew + self.gamma * (1 - done) * q_next

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        a, logp = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, a)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * logp - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return dict(actor_loss=actor_loss.item(), critic_loss=critic_loss.item())
