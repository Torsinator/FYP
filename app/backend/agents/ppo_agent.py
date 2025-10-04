from stable_baselines3.ppo.ppo import PPO
from agents.Agent import Agent

class PPOAgent(Agent):
    def __init__(self, model_path):
        self.model = PPO.load(model_path, device="cpu")

    @staticmethod
    def load(model_path, gym_env=None) -> Agent:
        return PPOAgent(model_path)
    
    def set_env(self, gym_env):
        self.model.set_env(gym_env)
    
    def learn(self, total_timesteps):
        self.model.learn(total_timesteps)
    
    def save(self, path):
        self.model.save(path)

    def predict(self, obs, **kwargs) -> tuple:
        return self.model.predict(obs, **kwargs)