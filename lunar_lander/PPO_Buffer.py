import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
import custom_lunar_lander_no_target


class MaxRewardRolloutBuffer(RolloutBuffer):
    """
    PPO Rollout buffer that supports:
        - "discounted" (vanilla PPO)
        - "max" (per-episode maximum reward)
        - "hybrid" (weighted combination)
    Alpha can be scheduled via alpha_schedule(step)
    """

    def __init__(self, *args, mode="max", alpha=0.5, alpha_schedule=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert mode in ["discounted", "max", "hybrid"], "Invalid mode"
        self.mode = mode
        self.alpha = alpha
        self.alpha_schedule = alpha_schedule
        self.training_step = 0

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        if self.mode == "discounted":
            return super().compute_returns_and_advantage(last_values, dones)

        # Get current alpha
        alpha = self.alpha_schedule(self.training_step) if self.alpha_schedule else self.alpha

        rewards = self.rewards
        episode_starts = self.episode_starts
        start_idx = 0

        for i in range(len(rewards)):
            # Last step or next step is a new episode
            if i == len(rewards) - 1 or episode_starts[i + 1]:
                # Slice for current episode
                ep_rewards = rewards[start_idx:i + 1]  # shape (episode_length,)
                ep_len = len(ep_rewards)

                # Max reward for episode
                max_r = np.max(ep_rewards)

                # Discounted returns per episode
                discounted = np.zeros(ep_len, dtype=np.float32)
                running_return = 0.0
                for t in reversed(range(ep_len)):
                    running_return = ep_rewards[t] + self.gamma * running_return
                    discounted[t] = running_return

                # Compute final return for each step
                if self.mode == "max":
                    ret = np.full(ep_len, max_r, dtype=np.float32)
                elif self.mode == "hybrid":
                    ret = alpha * discounted + (1 - alpha) * max_r

                # ✅ Assign with correct shape
                self.returns[start_idx:i + 1] = ret.reshape(-1, 1)
                self.advantages[start_idx:i + 1] = (ret - self.values[start_idx:i + 1].flatten()).reshape(-1, 1)

                start_idx = i + 1

class PPOWithCustomBuffer(PPO):
    """
    PPO that supports max, discounted, or hybrid reward optimization.
    """

    def __init__(self, *args, mode="max", alpha=0.5, alpha_schedule=None, **kwargs):
        self.buffer_mode = mode
        self.buffer_alpha = alpha
        self.buffer_alpha_schedule = alpha_schedule
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        super()._setup_model()
        self.rollout_buffer = MaxRewardRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            mode=self.buffer_mode,
            alpha=self.buffer_alpha,
            alpha_schedule=self.buffer_alpha_schedule,
        )

    def learn(self, total_timesteps, *args, **kwargs):
        # Keep track of training step for alpha scheduling
        self.rollout_buffer.training_step = 0
        while self.num_timesteps < total_timesteps:
            super().learn(
                total_timesteps=self.num_timesteps + self.n_steps,
                reset_num_timesteps=False,
                *args,
                **kwargs,
            )
            self.rollout_buffer.training_step = self.num_timesteps
        return self


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    env = gym.make("CustomLunarLander-v0", continuous=True)

    # Example alpha schedule: linear decay from 1.0 (dense) → 0.0 (max) over 200k steps
    def linear_alpha_schedule(step, total_steps=2_000_000):
        return max(0.0, 1.0 - step / total_steps)

    model = PPOWithCustomBuffer(
        "MlpPolicy",
        env,
        verbose=1,
        mode="discounted",  # "max", "discounted", "hybrid"
        alpha=1.0,      # only used if no schedule
        alpha_schedule=linear_alpha_schedule,
        device="cpu",
    )

    model.learn(total_timesteps=2_000_000)
    model.save("models/ppo_lunar_max_reward")
