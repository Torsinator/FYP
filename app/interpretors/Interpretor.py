# Abstract Base Class
from abc import ABC, abstractmethod

class Interpretor(ABC):

    @abstractmethod
    def __init__(self, config):
        pass
    
    @abstractmethod
    def load(self):
        pass
    
    @abstractmethod
    def give_command(self, user_command, new) -> tuple:
        pass
