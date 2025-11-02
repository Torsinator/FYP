from stable_baselines3.sac.sac import SAC
from agents.Agent import Agent

class SACAgent(Agent):
    def __init__(self, model_path, gym_env):
        self.model = SAC.load(model_path, device="cpu", env=gym_env)

    @staticmethod
    def load(model_path, gym_env=None) -> Agent:
        return SACAgent(model_path, gym_env)
    
    def set_env(self, gym_env):
        self.model.set_env(gym_env)
    
    def learn(self, total_timesteps):
        self.model.learn(total_timesteps)
    
    def save(self, path):
        self.model.save(path)

    def predict(self, obs, **kwargs) -> tuple:
        return self.model.predict(obs, **kwargs)