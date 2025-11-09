# FYP
by Tors Webster

This is the repository containing all code associated with the thesis 

## Setting Up Environment
### System Requirements
- Tested on Ubuntu 24.04.3 LTS
- LLM requires at least 15GB of VRAM
- At least 16GB of RAM

### Conda Environment
- Use: $ conda create --name <env> --file package-list.txt

## Directory Structure
### Important
- app/frontend -> react app UI
- app/backend -> fastAPI Python backend
- evaluation -> evaluation tests for models
- evaluation/video_tests -> video recordings of end to end system
### Testing/Sandbox
- lunar_lander -> code when experimenting with different techniques and envs
- Game -> A small game created to test multiple agents (not relevant now)
- Diagrams -> Images used in report