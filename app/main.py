from json import load
from interpretors.InterpretorLoader import get_interpretor_class
from agents.AgentLoader import get_agent_class
import gymnasium as gym
import numpy as np
import ast
from scipy.interpolate import interp1d
import os
import environments.custom_lunar_lander_no_target


EPISODE_LENGTH_SECONDS = 20  # <-- set this
HZ = 50
DT = 1.0 / HZ

def parseConfig(path):
    with open(path, 'r') as file:
        config = load(file)
    return config

def create_environment(env_name, video_folder):
    env = gym.make(env_name, render_mode="rgb_array", continuous=True)
    # Wrap the environment so that it always records the (only) episode.
    return gym.wrappers.RecordVideo(env, video_folder=video_folder,
                           episode_trigger=lambda episode: True)

def interpolate_states(states, episode_length, hz):
    """
    Interpolate target states so that we have one target per timestep.
    states: array [N, state_dim]
    """
    num_waypoints = len(states)
    state_dim = states.shape[1]

    # Original timepoints (spread across episode)
    t_waypoints = np.linspace(0, episode_length, num_waypoints)

    # New dense timeline at desired resolution
    t_dense = np.linspace(0, episode_length, int(episode_length * hz))

    # Interpolate each dimension separately
    states_interp = np.zeros((len(t_dense), state_dim))
    for d in range(state_dim):
        f = interp1d(t_waypoints, states[:, d], kind="linear")
        states_interp[:, d] = f(t_dense)

    return states_interp

def run_episode(env, model, traj, weights):
    # Interpolate to per-timestep targets
    states_interp = interpolate_states(traj, EPISODE_LENGTH_SECONDS, HZ)

    weights_interp = interpolate_states(weights, EPISODE_LENGTH_SECONDS, HZ)

    # states_interp = traj
    # weights_interp = weights

    # Record video
    video_folder = "./final_video"
    os.makedirs(video_folder, exist_ok=True)

    obs, info = env.reset(options={"target_state": states_interp[0], "weights": weights_interp[0]})

    # Step through dense interpolated targets
    for i in range(len(states_interp)):
        done = False
        env.unwrapped.set_target_state(np.array(states_interp[i], dtype=np.float32))
        env.unwrapped.set_weights(np.array(weights_interp[i], dtype=np.float32))
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    env.close()
    print(f"Final demonstration video recorded and saved in: {video_folder}")

def main():
    # constants
    video_folder = "./final_video"

    # Get config data
    config = parseConfig("config/app_config.json")
    interp_name = config.get("interpretor")
    env_name = config.get("environment").get("name")
    env_cfg = parseConfig(config.get("environment").get("cfg"))
    agent_type = config.get("agent").get("type")
    agent_model_path = config.get("agent").get("model")
    print("config set up")

    # Set up interpretor and agent
    interpretor = get_interpretor_class(interp_name)(env_cfg)    # Get class and call constructor
    print("interpretor set up")
    
    agent = get_agent_class(agent_type).load(agent_model_path)
    print("agent set up")
    
    # load the interpretor model (large)
    interpretor.load()

    print("all set up")

    # Replace with UI code
    while True:
        new_episode = True
        episode_complete = False
        while not episode_complete:

            # Set up environment
            env = create_environment(env_name, video_folder)

            # Get user command
            user_command = input("Enter a command for the Interpretor:\n")
            if (user_command == "new"):
                break

            clarify, reasoning, traj, weights = interpretor.give_command(user_command, new_episode)
            new_episode = False
            if clarify:
                continue

            print(f"reasoning: {reasoning}")
            print(f"trajectory: {traj}")
            print(f"weights: {weights}")

            assert(traj)
            assert(weights)
            traj = ast.literal_eval(traj)
            traj.insert(0, ast.literal_eval(env_cfg.get("current_state")))
            traj = np.array(traj)
            weights = ast.literal_eval(weights)


            weights.insert(0, weights[0])
            weights = np.array(weights)

            # # Patch remove when we can handle velocity
            # weights[:, 2] = 0
            # weights[:, 3] = 0
            # weights[:, 5] = 0
            print(traj, weights)
            
            run_episode(env, agent, traj, weights)



if __name__ == "__main__":
    main()