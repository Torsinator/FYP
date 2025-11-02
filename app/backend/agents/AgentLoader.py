# import models
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent
from agents.mpc_agent import MPCAgent
from agents.sindy_agent import SINDyAgent
from agents.shaped_sindy_agent import ShapedSINDyAgent

agents = [PPOAgent, SACAgent, MPCAgent, SINDyAgent, ShapedSINDyAgent]

def get_agent_class(agent_name):
    for cls in agents:
        if cls.__name__ == agent_name:
            return cls
    raise ValueError(f"{agent_name} not registered. Valid options are {[i.__name__ for i in agents]}")
