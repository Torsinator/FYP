"""
td3_her_dynamic_weights.py
Goal-Conditioned TD3 + Hindsight Experience Replay (HER) with DYNAMIC STATE WEIGHTS.

Key points:
- Gymnasium API (reset -> (obs, info); step -> (obs, r, terminated, truncated, info))
- Weights are PART OF THE INPUT to actor/critic: policy adapts instantly when change weights.
- Reward uses current weights: r = -|| w ⊙ (g - s_next) ||_2
- HER ("future" strategy) relabels goals and recomputes reward with the SAME weights stored in the transition.
- Feature vector per step: [ state, goal, error=(goal-state), weights, weighted_error=weights*(goal-state) ]
- Optional running normalization per block.

Usage:
    python td3_her_dynamic_weights.py
"""

from __future__ import annotations
import time, random
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import custom_lunar_lander_plain


# =========================
#  Config / Seeding
# =========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
RNG_SEED = 42

# Example default weights (match your state layout). Will be OVERRIDDEN dynamically at train/eval.
DEFAULT_WEIGHTS = np.array([1, 1, 0, 0, 1, 0], dtype=np.float32)


# =========================
#  Normalization utility
# =========================

class RunningNorm:
    """Welford online mean/std tracker."""
    def __init__(self, size: int, eps: float = 1e-8):
        self.size = size
        self.eps = eps
        self.count = 0
        self.mean = np.zeros(size, dtype=np.float64)
        self.M2 = np.zeros(size, dtype=np.float64)

    def update(self, x: np.ndarray):
        x = x.reshape(-1, self.size)
        for row in x:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            delta2 = row - self.mean
            self.M2 += delta * delta2

    @property
    def std(self):
        return np.sqrt((self.M2 / max(self.count - 1, 1)) + self.eps)

    def normalize(self, x: np.ndarray):
        return (x - self.mean) / self.std


# =========================
#  Goal wrapper with dynamic weights
# =========================

class GoalAdapter(gym.Wrapper):
    """
    Wrap any Gymnasium env to:
    - Track a desired goal (vector same size as observation by default).
    - Hold a CURRENT weights vector used for reward (can be changed at any time).
    """
    def __init__(self, env: gym.Env, init_weights: np.ndarray, goal_sampler: Optional[Callable[[], np.ndarray]] = None):
        super().__init__(env)
        self.obs_dim = int(env.observation_space.shape[0])
        self.goal_dim = self.obs_dim
        self.weights = init_weights.astype(np.float32)
        self.goal_sampler = goal_sampler or self._uniform_goal

        # Goal sampling bounds from observation space (fallback to [-1,1] if unbounded)
        low = env.observation_space.low
        high = env.observation_space.high
        self._low = np.where(np.isfinite(low), low, -1.0)
        self._high = np.where(np.isfinite(high), high, 1.0)

        self.desired_goal: Optional[np.ndarray] = None
        self._last_obs: Optional[np.ndarray] = None
        self.steps = 0

    def _uniform_goal(self) -> np.ndarray:
        return np.random.uniform(self._low, self._high).astype(np.float32)

    def set_weights(self, w: np.ndarray):
        """Update weights used for reward immediately (affects ongoing episode)."""
        self.weights = np.asarray(w, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, goal: Optional[np.ndarray] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = obs.astype(np.float32)
        self._last_obs = obs
        self.desired_goal = (goal.astype(np.float32) if goal is not None else self.goal_sampler())
        return obs, info

    def step(self, action):
        prev_dist = np.linalg.norm((self.desired_goal - self._last_obs) * self.weights)
        obs, env_r, terminated, truncated, info = self.env.step(action)
        # print("TW: obs {obs}")
        obs = obs.astype(np.float32)
        self._last_obs = obs
        # print(obs, self.desired_goal, self.weights)
        # Our metric-based reward w.r.t. NEXT state
        r = prev_dist - np.linalg.norm((self.desired_goal - obs) * self.weights)
        r*=100
        # Fail/Crash penalise hard
        if env_r < -999:
            # r = -(5000 - self.steps) * 0.01
            r = -10

        if self.steps == 0:
            r = 0
        r = np.clip(r,-10, 10)
        self.steps += 1
        print(f"step {self.steps} reward {r}")
        return obs, r, terminated, truncated, info

    def achieved(self) -> np.ndarray:
        return self._last_obs.copy()


# =========================
#  Feature builder (dynamic weights)
# =========================

class DynamicFeaturizer:
    """
    Build features that include weights explicitly:
    feat = [ state, goal, error=(goal-state), weights, weighted_error=weights*(goal-state) ]
    Optionally normalize each block with its own RunningNorm.
    """
    def __init__(self, dim: int, normalize: bool = True):
        self.dim = dim
        self.normalize = normalize
        if normalize:
            self.norm_s = RunningNorm(dim)
            self.norm_g = RunningNorm(dim)
            self.norm_e = RunningNorm(dim)
            self.norm_w = RunningNorm(dim)
            self.norm_we = RunningNorm(dim)
        else:
            self.norm_s = self.norm_g = self.norm_e = self.norm_w = self.norm_we = None

    def update_norms(self, S: np.ndarray, G: np.ndarray, W: np.ndarray):
        # S, G, W shapes: [N, dim]
        E = (G - S)
        WE = W * E
        if self.normalize:
            self.norm_s.update(S)
            self.norm_g.update(G)
            self.norm_e.update(E)
            self.norm_w.update(W)
            self.norm_we.update(WE)

    def build(self, s: np.ndarray, g: np.ndarray, w: np.ndarray) -> np.ndarray:
        # print(f's: {s}')
        # print(f'g: {g}')
        # print(f'w: {w}')
        e = g - s
        we = w * e
        if self.normalize and self.norm_s.count > 0:
            s = self.norm_s.normalize(s)
            g = self.norm_g.normalize(g)
            e = self.norm_e.normalize(e)
            w = self.norm_w.normalize(w)
            we = self.norm_we.normalize(we)
        return np.concatenate([s, g, e, w, we], axis=-1).astype(np.float32)

    @property
    def feat_dim(self) -> int:
        # 5 blocks of size dim
        return self.dim * 5


# =========================
#  HER buffer (stores weights)
# =========================

class HERBuffer:
    """
    Stores complete episodes with their per-step weights and goals.
    On sampling, relabel goals with 'future' strategy and recompute reward with the SAME weights.
    Each step record: { s, a, s2, done, achieved, desired, weights }
    """
    def __init__(self, max_episodes: int = 3000, her_k: int = 4):
        self.episodes: List[list] = []
        self.max_episodes = max_episodes
        self.her_k = her_k

    def add_episode(self, ep: list):
        if len(self.episodes) >= self.max_episodes:
            self.episodes.pop(0)
        self.episodes.append(ep)

    def __len__(self):
        return sum(len(ep) for ep in self.episodes)

    def sample(self, batch_size: int):
        S, A, R, S2, G, D, W = [], [], [], [], [], [], []
        while len(S) < batch_size:
            ep = random.choice(self.episodes)
            T = len(ep)
            t = random.randrange(T)
            tr = ep[t]

            # future relabel
            if random.random() < (self.her_k / (self.her_k + 1)) and t < T - 1:
                f = random.randrange(t + 1, T)
                g_new = ep[f]["achieved"].copy()   # achieved state later in the same episode
            else:
                g_new = tr["desired"].copy()

            w = tr["weights"].copy()
            s2 = tr["s2"].copy()
            # recompute reward wrt new goal and SAME weights (dynamic context)
            rew = -np.linalg.norm((g_new - s2) * w)

            S.append(tr["s"])
            A.append(tr["a"])
            R.append(rew)
            S2.append(s2)
            G.append(g_new)
            D.append(tr["done"])
            W.append(w)

        # to tensors
        to_t = lambda x: torch.as_tensor(np.array(x, np.float32), device=DEVICE)
        return to_t(S), to_t(A), to_t(np.array(R, np.float32)).unsqueeze(-1), \
               to_t(S2), to_t(G), to_t(np.array(D, np.float32)).unsqueeze(-1), to_t(W)


# =========================
#  TD3 (actor/critic take FEATURES)
# =========================

def mlp(sizes, activation=nn.ReLU, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else out_act
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, feat_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([feat_dim] + list(hidden) + [act_dim], activation=nn.ReLU, out_act=nn.Tanh)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)  # [-1,1]

class Critic(nn.Module):
    def __init__(self, feat_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.q1 = mlp([feat_dim + act_dim] + list(hidden) + [1], activation=nn.ReLU, out_act=nn.Identity)
        self.q2 = mlp([feat_dim + act_dim] + list(hidden) + [1], activation=nn.ReLU, out_act=nn.Identity)

    def forward(self, feat: torch.Tensor, act: torch.Tensor):
        x = torch.cat([feat, act], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, feat: torch.Tensor, act: torch.Tensor):
        x = torch.cat([feat, act], dim=-1)
        return self.q1(x)

@dataclass
class TD3Config:
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    hidden: Tuple[int, int] = (256, 256)

class TD3Agent:
    def __init__(self, feat_dim: int, act_space: gym.spaces.Box, cfg: TD3Config):
        self.cfg = cfg
        self.act_dim = int(np.prod(act_space.shape))
        self.act_limit = float(act_space.high[0])

        self.actor = Actor(feat_dim, self.act_dim, cfg.hidden).to(DEVICE)
        self.actor_targ = Actor(feat_dim, self.act_dim, cfg.hidden).to(DEVICE)
        self.critic = Critic(feat_dim, self.act_dim, cfg.hidden).to(DEVICE)
        self.critic_targ = Critic(feat_dim, self.act_dim, cfg.hidden).to(DEVICE)

        self.actor_targ.load_state_dict(self.actor.state_dict())
        self.critic_targ.load_state_dict(self.critic.state_dict())

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.update_step = 0

    def select_action(self, feat_np: np.ndarray, noise_std: float = 0.1) -> np.ndarray:
        f = torch.as_tensor(feat_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(f).cpu().numpy()[0]
        a = a * self.act_limit
        if noise_std > 0:
            a = a + noise_std * np.random.randn(*a.shape)
        return np.clip(a, -self.act_limit, self.act_limit).astype(np.float32)

    def update(self, batch, featurizer: DynamicFeaturizer):
        s, a, r, s2, g, d, w = batch

        # Build features (vectorized on CPU for simplicity; you can optimize)
        s_np, s2_np, g_np, w_np = s.cpu().numpy(), s2.cpu().numpy(), g.cpu().numpy(), w.cpu().numpy()
        feat = torch.as_tensor(np.vstack(featurizer.build(s_np, g_np, w_np)),
                               dtype=torch.float32, device=DEVICE)
        feat2 = torch.as_tensor(np.vstack(featurizer.build(s_np, g_np, w_np)),
                                dtype=torch.float32, device=DEVICE)

        # Critic target
        with torch.no_grad():
            noise = (self.cfg.policy_noise * torch.randn_like(a)).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            a2 = (self.actor_targ(feat2) * self.act_limit + noise).clamp(-self.act_limit, self.act_limit)
            q1_t, q2_t = self.critic_targ(feat2, a2)
            q_targ = torch.min(q1_t, q2_t)  # shape: [batch, 1]
            y = r + (1.0 - d) * self.cfg.gamma * q_targ  # y: [batch, 1]

        q1, q2 = self.critic(feat, a)      # q1, q2: [batch, 1]

        # No need to unsqueeze; shapes already match y
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        # Delayed policy update
        actor_loss_val = None
        if self.update_step % self.cfg.policy_delay == 0:
            a_pred = (self.actor(feat) * self.act_limit).clamp(-self.act_limit, self.act_limit)
            actor_loss = -self.critic.q1_only(feat, a_pred).mean()
            self.opt_actor.zero_grad()
            actor_loss.backward()
            self.opt_actor.step()
            actor_loss_val = float(actor_loss.item())

            # Polyak averaging
            with torch.no_grad():
                for p, p_t in zip(self.critic.parameters(), self.critic_targ.parameters()):
                    p_t.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
                for p, p_t in zip(self.actor.parameters(), self.actor_targ.parameters()):
                    p_t.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

        self.update_step += 1
        return float(critic_loss.item()), actor_loss_val


# =========================
#  Training loop
# =========================

def train(
    make_env: Callable[[], gym.Env],
    episodes: int = 700,
    max_ep_steps: int = 600,
    start_random_steps: int = 1_000,
    batch_size: int = 256,
    updates_per_env_step: int = 1,
    her_k: int = 4,
    normalize_features: bool = True,
    seed: int = RNG_SEED,
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    base_env = make_env()
    assert isinstance(base_env.action_space, gym.spaces.Box), "TD3 needs a continuous action space."

    # Wrap with goal + dynamic weights
    env = GoalAdapter(base_env, init_weights=DEFAULT_WEIGHTS)

    obs_dim = env.observation_space.shape[0]
    act_space = env.action_space
    featurizer = DynamicFeaturizer(dim=obs_dim, normalize=normalize_features)
    agent = TD3Agent(feat_dim=featurizer.feat_dim, act_space=act_space, cfg=TD3Config())
    buffer = HERBuffer(max_episodes=3000, her_k=her_k)

    total_steps = 0
    updates = 0
    PRINT_EVERY = 10
    t0 = time.time()

    for ep in range(1, episodes + 1):
        # Sample a goal and a weights vector for THIS EPISODE (teaches the network to condition on weights)
        goal = env.goal_sampler()
        # Example: randomize importance (you can design your own sampler)
        w = np.random.uniform(low=0.0, high=1.0, size=6).astype(np.float32)
        # w *= 10
        w[2] = 0
        w[3] = 0
        w[4] = 0
        w[5] = 0
        env.set_weights(w)

        obs, _ = env.reset(goal=goal)
        ep_buf = []
        ep_ret = 0.0
        env.steps = 0
        for t in range(max_ep_steps):
            # Optional: update norms occasionally using short windows
            if normalize_features and total_steps % 200 == 0 and total_steps > 0 and len(buffer.episodes) > 0:
                # Sample a mini-batch from episodes to refresh normalization
                # Gather up to 1024 random (s,g,w) triplets
                samples = min(1024, sum(len(e) for e in buffer.episodes))
                if samples > 0:
                    S_list, G_list, W_list = [], [], []
                    for _ in range(samples):
                        ep_r = random.choice(buffer.episodes)
                        tr = random.choice(ep_r)
                        S_list.append(tr["s"])
                        G_list.append(tr["desired"])
                        W_list.append(tr["weights"])
                    featurizer.update_norms(np.array(S_list, np.float32), np.array(G_list, np.float32), np.array(W_list, np.float32))

            # Build feature for action selection
            feat_now = featurizer.build(obs, env.desired_goal, env.weights)

            if total_steps < start_random_steps:
                action = np.random.uniform(low=env.action_space.low, high=env.action_space.high).astype(np.float32)
            else:
                noise_std = max(0.1, 0.4 * (1 - ep / max(episodes, 1)))
                action = agent.select_action(feat_now, noise_std=noise_std)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            ep_buf.append({
                "s": obs.copy(),
                "a": action.copy(),
                "s2": next_obs.copy(),
                "done": float(done),
                "achieved": env.achieved(),
                "desired": env.desired_goal.copy(),
                "weights": env.weights.copy(),
            })

            obs = next_obs
            ep_ret += reward
            total_steps += 1

            # Learn
            if len(buffer.episodes) >= 3 and total_steps >= start_random_steps:
                for _ in range(updates_per_env_step):
                    batch = buffer.sample(batch_size)
                    critic_loss, actor_loss = agent.update(batch, featurizer)
                    updates += 1

            if done:
                break

        buffer.add_episode(ep_buf)

        if ep % PRINT_EVERY == 0:
            elapsed_min = (time.time() - t0) / 60.0
            print(f"Ep {ep}/{episodes} | Return {ep_ret:8.2f} | EpisodesInBuf {len(buffer.episodes):4d} | "
                  f"EnvSteps {total_steps:7d} | Updates {updates:7d} | {elapsed_min:4.1f}m")

    print("Training complete.")
    return agent, env, featurizer


# =========================
#  Example usage
# =========================

def make_env() -> gym.Env:
    """
    Replace with your custom Lunar Lander constructor, e.g.:
        from your_pkg import YourLanderEnv
        return YourLanderEnv()
    Must be Gymnasium-compatible.
    """
    # Demo fallback (Gymnasium's Box2D):
    return gym.make("CustomLunarLander-v0", continuous=True)

if __name__ == "__main__":
    np.random.seed(RNG_SEED); random.seed(RNG_SEED); torch.manual_seed(RNG_SEED)

    agent, env, feat = train(
        make_env=make_env,
        episodes=500,
        max_ep_steps=500,
        start_random_steps=10_000,
        batch_size=256,
        updates_per_env_step=1,
        her_k=4,
        normalize_features=True,
        seed=RNG_SEED,
    )

    torch.save(agent.actor.state_dict(), "models/actor.pth")
    torch.save(agent.critic.state_dict(), "models/critic.pth")

    import os
    from gymnasium.wrappers import RecordVideo
    # ---- Inference with new weights (no retraining needed) ----
    # Example: at deployment, prioritize angle & angular velocity more:
    new_weights = np.array([1,1,0,0,0,0], dtype=np.float32)
    goal = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
    # Now, create a new environment for the final demonstration episode.
    video_folder = "./final_video"
    os.makedirs(video_folder, exist_ok=True)

    # For video recording, "render_mode" must be set to "rgb_array".
    demo_env = gym.make("CustomLunarLander-v0", render_mode="rgb_array", continuous=True)
    # Wrap the environment so that it always records the (only) episode.
    demo_env = RecordVideo(demo_env, video_folder=video_folder,
                            episode_trigger=lambda episode: True, name_prefix="HER")

    # Run exactly one episode and record it.
    # state, info = demo_env.reset(options={"target_state" : lunarify_target_state(target_state)})
    env = GoalAdapter(demo_env, new_weights)
    env.desired_goal = goal
    new_goal = env.goal_sampler()

    obs, _ = env.reset(goal=new_goal)
    total = 0.0
    for _ in range(500):
        feat_now = feat.build(obs, env.desired_goal, env.weights)  # dynamic weights used here
        act = agent.select_action(feat_now, noise_std=0.0)
        obs, r, term, trunc, _ = env.step(act)
        total += r
        if term or trunc:
            break
    env.close()
    print("Eval return with NEW weights:", total)
