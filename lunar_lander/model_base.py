import custom_lunar_lander  # Your custom Lunar Lander environment definition.
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# Environment Configuration (from your original Ray + PPO code)
# =============================================================================

env_id = "CustomLunarLander-v0"
env_config = {
    "render_mode": "rgb_array",  # Change as needed for your env.
    "continuous": True           # Your custom flag (e.g. for continuous action space).
}

# In a Ray setup you might register the env with ray.tune.registry.
# Here we simply create an instance using gym.make().
env = gym.make(env_id, **env_config)
obs_space = env.observation_space
act_space = env.action_space

# =============================================================================
# Define a Minimal DreamerV3-Style Agent (using PyTorch)
# =============================================================================

class DreamerV3Agent:
    def __init__(self, obs_space, act_space, config):
        self.obs_space = obs_space
        self.act_space = act_space
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Assume obs_space is a Box; flatten its shape.
        input_dim = int(np.prod(obs_space.shape))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        ).to(self.device)

        # For a discrete action space, the actor outputs logits per action.
        if hasattr(act_space, "n"):
            self.actor = nn.Sequential(
                nn.Linear(64, act_space.n)
            ).to(self.device)
        else:
            # For continuous actions, you might output means (and variances) instead.
            self.actor = nn.Sequential(
                nn.Linear(64, act_space.shape[0])
            ).to(self.device)

        # Simple optimizer over both networks.
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()),
            lr=config.get("learning_rate", 1e-3)
        )

    def act(self, obs):
        """Process the observation and select an action."""
        self.encoder.eval()
        # Convert observation to a flat tensor.
        obs_arr = np.array(obs).flatten()
        obs_tensor = torch.tensor(obs_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            encoded = self.encoder(obs_tensor)
            logits = self.actor(encoded)
            if hasattr(self.act_space, "n"):
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1).item()
            else:
                # For continuous action spaces, a different sampling strategy is recommended.
                action = logits.cpu().numpy()[0]
        return action

    def update(self, batch):
        """
        A dummy update function.
        In a full DreamerV3 implementation you would update the world model,
        actor, and value networks appropriately.
        Here, we perform a simple cross-entropy loss update on the actor.
        """
        # Prepare a batch of flattened observations.
        obs_batch = torch.tensor(
            np.array(batch["obs"]).reshape(len(batch["obs"]), -1),
            dtype=torch.float32, device=self.device
        )
        # Ensure target actions has shape (batch_size,) (not (batch_size, 1)):
        action_batch = torch.tensor(
            np.array(batch["actions"]), dtype=torch.long, device=self.device
        ).squeeze()

        self.encoder.train()
        encoded = self.encoder(obs_batch)
        logits = self.actor(encoded)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, action_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# =============================================================================
# Training Loop
# =============================================================================

def main():
    # A configuration dictionary similar to your Ray PPOConfig (only learning_rate used here)
    agent_config = {
        "learning_rate": 1e-3,
        "batch_size": 32  # Not used in this dummy example but can be integrated.
    }

    agent = DreamerV3Agent(obs_space, act_space, agent_config)

    num_episodes = 10
    max_steps = 200
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        step = 0
        # Collect transitions in a simple memory buffer.
        experience = {"obs": [], "actions": [], "rewards": []}

        while not done and step < max_steps:
            action = agent.act(obs)
            next_obs, reward, done, truncated, info = env.step(action)

            # Store the experience.
            experience["obs"].append(obs)
            experience["actions"].append(action)
            experience["rewards"].append(reward)

            obs = next_obs
            episode_reward += reward
            step += 1

        # Perform an update at the end of the episode.
        loss = agent.update(experience)
        print(f"Episode {ep+1}: Total Reward = {episode_reward:.2f}, Update Loss = {loss:.4f}")

    env.close()

if __name__ == '__main__':
    main()
