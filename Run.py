from Game import Game
from Agent import Agent
from Channel import Channel

rows = 20
cols = 20
num_agents = 4
num_resources = 4
max_steps = 200


def main():
    current_timestep = 0
    game = Game(rows, cols, num_resources)
    channel = Channel()
    for i in range(num_agents):
        # For now, just set the goal to be equal to the agent's number
        id = f'A{i}'
        goal = f'R{i}'
        agent = Agent(id, game, channel, goal)
        game.add_agent(agent)
    agents = game.agents
    while True:
        if game.win():
            print("Game won")
            break
        elif current_timestep > max_steps:
            print("Max turns reached")
            break
        for agent in agents:
            agent.make_turn()
        channel.reset_messages()
        current_timestep += 1




if __name__ == '__main__':
    main()