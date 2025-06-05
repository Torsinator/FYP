import custom_lunar_lander
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from ray.tune.registry import register_env
import ray
from ray.rllib.algorithms.dreamerv3 import MBPOConfig

def custom_env_creator(env_config):
    """
    Create an instance of your custom Lunar Lander environment.
    Any items in env_config can be used to pass additional parameters.
    Here we force the render_mode to 'rgb_array' and continuous control to True.
    """
    # This call passes the keyword arguments to your custom environment’s __init__.
    env = gym.make("CustomLunarLander-v0", render_mode="rgb_array", continuous=True)
    # Optionally, wrap the environment. For example, FlattenObservation makes observations flat.
    env = FlattenObservation(env)
    return env

# Register your custom environment with Ray.
register_env("CustomLunarLander-v0", custom_env_creator)

def train_mbpo():
    # Initialize Ray.
    ray.init(ignore_reinit_error=True)

    # Configure the MBPO algorithm.
    config = (
        MBPOConfig()
        .environment(
            env="CustomLunarLander-v0",  # Registered custom environment
            env_config={}                # Additional environment configurations if needed.
        )
        .framework("torch")             # Use "torch" (or "tf" if you prefer TensorFlow)
        # You can adjust training hyperparameters as needed:
        .training(
            train_batch_size=1000,
            rollout_fragment_length=200,
            num_sgd_iter=10
        )
        # MBPO has some model-based specific parameters you can set as well.
        # For example:
        # .training(mbpo_config={"ensemble_size": 5, "rollout_schedule": [0, 20]})
    )

    # Build the MBPO trainer.
    trainer = config.build()

    # Training loop.
    for i in range(50):
        result = trainer.train()
        print(f"Iteration {i}: mean episodic reward = {result.get('episode_reward_mean', 'N/A')}")

    # Save the final checkpoint.
    checkpoint = trainer.save()
    print(f"Checkpoint saved at: {checkpoint}")

    # Shutdown Ray.
    ray.shutdown()

if __name__ == "__main__":
    train_mbpo()
