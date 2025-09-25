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
import gymnasium as gym
import numpy as np
import ast
from scipy.interpolate import interp1d
import os
from environments.env_loader import get_env
from pathlib import Path

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

    agent = get_agent_class(agent_type).load(agent_model_path)
    print("agent set up")

    # load the interpretor model (large)
    interpretor.load()

    return env_cfg, env_name, interpretor, agent

def new_episode(env_name):
    env = get_env(env_name)
    env = gym.wrappers.RecordVideo(env, video_folder=video_folder,
                           episode_trigger=lambda episode: True)
    return env

def run_episode(env, model, traj, weights):
    # Interpolate to per-timestep targets
    # states_interp = interpolate_states(traj, EPISODE_LENGTH_SECONDS, HZ)

    # weights_interp = interpolate_states(weights, EPISODE_LENGTH_SECONDS, HZ)

    states_interp = traj
    weights_interp = weights[:, [0,1,4]]

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

# Read config and create necessary objects
env_cfg, env_name, interpretor, agent = reconfig()

env = new_episode(env_name)

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
    env = new_episode(env_name)
    new_episode_flag = False
    return {}

@app.post("/command", response_model=ChatResponse)
def command(command : ChatRequest):
    global new_episode_flag
    clarify, reasoning, traj, weights = interpretor.give_command(command.message, new_episode_flag)
    response = ChatResponse(clarify=clarify, reasoning=reasoning, states=traj, weights=weights, video_path=f"{env.name_prefix}-episode{env.episode_id}.mp4")
    if not clarify:
        traj = ast.literal_eval(traj)
        traj.insert(0, ast.literal_eval(env_cfg.get("current_state")))
        traj = np.array(traj)
        weights = ast.literal_eval(weights)


        weights.insert(0, weights[0])
        weights = np.array(weights)
        run_episode(env, agent, traj, weights)
        new_episode_flag = False
        new_episode(env_name)
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
