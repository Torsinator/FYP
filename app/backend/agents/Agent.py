# Abstract Base Class
from abc import ABC, abstractmethod

class Agent(ABC):

    @abstractmethod
    def __init__(self, model_path):
        raise NotImplementedError
    
    @staticmethod
    def load(model_path, gym_env=None) -> "Agent":
        raise NotImplementedError

    @abstractmethod
    def set_env(self, gym_env):
        raise NotImplementedError

    @abstractmethod
    def learn(self, total_timesteps):
        raise NotImplementedError
    
    @abstractmethod
    def save(self, path):
        raise NotImplementedError

    @abstractmethod
    def predict(self, obs, **kwargs) -> tuple:
        raise NotImplementedError
