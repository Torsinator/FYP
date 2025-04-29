import gymnasium as gym
import numpy as np

class LunarLanderGoalEnv(gym.Env):
    """
    A simplified Lunar Lander environment that uses goal-conditioned RL.
    The observation is augmented to include both the current state and a target state.
    """
    def __init__(self):
        super(LunarLanderGoalEnv, self).__init__()

        # Define the state dimensions: [x, y, v_x, v_y, angle, angular_velocity]
        # For simplicity, we use unbounded spaces here; adjust bounds as needed.
        state_low = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.pi, -np.inf])
        state_high = np.array([np.inf, np.inf, np.inf, np.inf, np.pi, np.inf])
        self.state_space = gym.spaces.Box(low=state_low, high=state_high, dtype=np.float32)

        # Define the goal state space (same dimensions as the state)
        self.goal_space = gym.spaces.Box(low=state_low, high=state_high, dtype=np.float32)

        # The overall observation is the concatenation of both state and goal state
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([state_low, state_low]),
            high=np.concatenate([state_high, state_high]),
            dtype=np.float32
        )

        # Define an action space (could be discrete or continuous depending on your design)
        self.action_space = gym.spaces.Discrete(4)  # For demonstration

        # Initialize the environment state and goal state
        self.reset()

    def reset(self):
        # Reset the current physical state
        self.state = np.array([
            0.0,   # x position
            10.0,  # y position (start 10 units above ground)
            0.0,   # v_x
            0.0,   # v_y
            0.0,   # angle in radians
            0.0    # angular velocity
        ], dtype=np.float32)

        # Set a default goal state (this should be later provided by your higher-level module)
        # For example, a flip might require a change in angle from 0 to 2π,
        # so you might set a goal like: [x_goal, y_goal, v_x_goal, v_y_goal, 2π, 0.0]
        self.goal_state = np.array([
            5.0,    # desired x (landing pad)
            0.0,    # desired y (ground level)
            0.0,    # desired v_x
            0.0,    # desired v_y
            2 * np.pi,  # desired angle (a full flip)
            0.0     # desired angular velocity
        ], dtype=np.float32)

        return self._get_obs()

    def step(self, action):
        dt = 0.1  # time step for simulation

        # Dummy physics update: update state based on action.
        # Replace this with your actual physics model.
        # Example: adjust angular velocity with certain actions
        if action == 1:  # fire left thruster
            self.state[5] -= 0.05
        elif action == 2:  # fire main thruster (affecting velocity)
            self.state[2] += 0.05
            self.state[3] += 0.05
        elif action == 3:  # fire right thruster
            self.state[5] += 0.05

        # Update position and angle based on current velocities.
        self.state[0] += self.state[2] * dt  # x pos update
        self.state[1] += self.state[3] * dt  # y pos update
        self.state[4] += self.state[5] * dt  # angle update

        # Calculate the reward: negative Euclidean distance between current state and goal state.
        # (In practice, you might weight different dimensions differently.)
        error = self.state - self.goal_state
        reward = -np.linalg.norm(error)

        # In a realistic scenario, you would also include termination conditions:
        # e.g., if the agent is close enough to the goal, flag done=True.
        done = np.linalg.norm(error) < 0.5  # arbitrary threshold for task completion

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        # Concatenate current state and goal state into one observation
        return np.concatenate([self.state, self.goal_state]).astype(np.float32)

# Example training loop (you would replace this with your RL algorithm training code)
if __name__ == "__main__":
    env = LunarLanderGoalEnv()
    obs = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        # For the sake of demonstration, sample a random action.
        # In practice, your RL agent's policy will generate this.
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        print(f"Obs: {obs}, Reward: {reward}")

    print(f"Episode finished with total reward: {total_reward}")
