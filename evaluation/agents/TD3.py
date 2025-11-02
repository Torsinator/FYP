from stable_baselines3 import TD3

import environments.lunar_lander_customs as envs

class TD3_Unweighted_9_States(TD3):
    @classmethod
    def load_model(cls):
        env = envs.LunarLander9Obs.make_env(False, True)
        model = TD3.load("models/TD3_Unweighted_9_States", env=env)
        return model, env

class TD3_Unweighted_6_States(TD3):
    def __init__(self) -> None:
        env = envs.LunarLander6Obs.make_env()
        self = TD3.load("models/TD3_Unweighted_6_States", env=env)

class TD3_Weighted_9_States(TD3):
    def __init__(self) -> None:
        env = envs.LunarLander9ObsWeighted.make_env()
        self = TD3.load("models/TD3_Unweighted_6_States", env=env)