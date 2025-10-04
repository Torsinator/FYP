# import models
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent

agents = [PPOAgent, SACAgent]

def get_agent_class(agent_name):
    for cls in agents:
        if cls.__name__ == agent_name:
            return cls
    raise ValueError(f"{agent_name} not registered. Valid options are {[i.__name__ for i in agents]}")
