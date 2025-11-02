# main.py
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from fastapi.responses import FileResponse
from fastapi import HTTPException

from json import load
from interpretors.InterpretorLoader import get_interpretor_class
from agents.AgentLoader import get_agent_class
from agents.Agent import Agent
import gymnasium as gym
import numpy as np
import ast
import os
from environments.env_loader import get_env
from pathlib import Path

import scipy

EPISODE_LENGTH_SECONDS = 20  # <-- set this
HZ = 50
DT = 1.0 / HZ

# constants
video_folder = "./final_video"

# State Variables
new_episode_flag : bool = True

def parseConfig(path):
    with open(path, 'r') as file:
        config = load(file)
    return config

def new_episode(env_name, agent_class:Agent, agent_model_path):
    env = get_env(env_name)
    env = gym.wrappers.RecordVideo(env, video_folder=video_folder,
                           episode_trigger=lambda episode: True)
    agent = agent_class.load(agent_model_path, env)
    return env, agent

def reconfig():
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

    agent_class = get_agent_class(agent_type)

    # load the interpretor model (large)
    interpretor.load()

    return env_cfg, env_name, interpretor, agent_class, agent_model_path

def interpolate(values, episode_length, hz):
    """
    Interpolate target values so that we have one target per timestep.
    values: array [N, state_dim]
    """
    num_waypoints = len(values)
    state_dim = values.shape[1]

    # Original timepoints (spread across episode)
    t_waypoints = np.linspace(0, episode_length, num_waypoints)

    # New dense timeline at desired resolution
    t_dense = np.linspace(0, episode_length, int(episode_length * hz))

    # Interpolate each dimension separately
    values_interp = np.zeros((len(t_dense), state_dim))
    for d in range(state_dim):
        f = scipy.interpolate.interp1d(t_waypoints, values[:, d], kind="linear")
        values_interp[:, d] = f(t_dense)

    return values_interp

def run_episode(env, model, traj, weights):
    # Interpolate to per-timestep targets
    # states_interp = interpolate(traj, EPISODE_LENGTH_SECONDS, HZ)

    # weights_interp = interpolate(weights, EPISODE_LENGTH_SECONDS, HZ)

    states_interp = traj
    # # weights_interp = weights[:, [0,1,4]]
    weights_interp = weights

    print(f"debug weights: {weights}")
    print(f"debug states: {states_interp}")

    # Record video
    video_folder = "./final_video"
    os.makedirs(video_folder, exist_ok=True)

    obs, info = env.reset(options={"target_state": states_interp[0], "weights": weights_interp[0]})

    path = f"{env.name_prefix}-episode"
    if f"{env.episode_id}"[0] != "-":
        path += "-"
    path += f"{env.episode_id}.mp4"

    # Step through dense interpolated targets
    for i in range(len(states_interp)):
        done = False
        print("innnnnnnn")
        env.env.set_target_state(np.array(states_interp[i], dtype=np.float32))
        env.env.set_weights(np.array(weights_interp[i], dtype=np.float32))
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    env.close()
    print(f"Final demonstration video recorded and saved in: {video_folder}")
    return path

# Read config and create necessary objects
env_cfg, env_name, interpretor, agent_class, agent_model_path = reconfig()

env, agent = new_episode(env_name, agent_class, agent_model_path)

app = FastAPI()

# Allow frontend (React) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    clarify: bool
    reasoning: str
    states: str
    weights: str
    video_path: str

# class ConfigMessage(BaseModel):
#     tastes: dict[str, PositiveInt]


@app.get("/")
def root():
    return {"message": "Hello from FastAPI 🚀"}

@app.get("/reset")
def reset():
    global env
    global new_episode_flag
    global agent
    env, agent = new_episode(env_name, agent_class, agent_model_path)
    new_episode_flag = True
    return {}

@app.post("/command", response_model=ChatResponse)
def command(command : ChatRequest):
    global new_episode_flag
    clarify, reasoning, traj, weights = interpretor.give_command(command.message, new_episode_flag)
    new_episode_flag = False
    path = ""
    if not clarify:
        np_traj = ast.literal_eval(traj)
        np_traj.insert(0, ast.literal_eval(env_cfg.get("current_state")))
        np_traj = np.array(np_traj)
        np_weights = ast.literal_eval(weights)


        np_weights.insert(0, np_weights[0])
        np_weights = np.array(np_weights)
        path = run_episode(env, agent, np_traj, np_weights)
        new_episode(env_name, agent_class, agent_model_path)
    response = ChatResponse(clarify=clarify, reasoning=reasoning, states=traj, weights=weights, video_path=path)
    return response

@app.websocket("/ws/generate")
async def generate(websocket: WebSocket):
    await websocket.accept()
    while True:
        while True:
            message = await websocket.receive_text()

            clarify, reasoning, traj, weights = interpretor.give_command(message, new_episode_flag)
            response = ChatResponse(clarify=clarify, reasoning=reasoning, states=traj, weights=weights, video_path="rl-video-episode-0.mp4")

            # Send back text immediately
            await websocket.send_text(response.model_dump_json())

            # if we do not need to clarify, then we can move on to generation
            if not clarify:
                break

            # Simulate video processing delay
            await asyncio.sleep(3)
            await websocket.send_json({"type": "video", "data": "/videos/demo.mp4"})

@app.get("/video/{filename}")
async def get_video(filename: str):
    filepath = Path(video_folder) / filename
    if filepath.is_file():
        return FileResponse(str(filepath), media_type="video/mp4", filename=filename)
    raise HTTPException(status_code=404, detail="File not found")
