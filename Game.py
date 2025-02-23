import numpy as np
import datetime
import os
import pickle

class Game:
    def __init__(self, length, width, resources, agents = []):
        self.length = length
        self.width = width
        self.resources = resources
        self.grid = np.empty((width, length), dtype='<U2')
        self.agents = agents
        self.logfile = datetime.datetime.now().strftime("logfile_%Y%m%d_%H%M%S.txt")

    def generate(self):
        # Generate positions for the resources
        for i in range(self.resources):
            while True:
                x_pos = np.random.randint(0, self.width)
                y_pos = np.random.randint(0, self.length)
                # Make sure there are no two resources on the same tile
                if self.grid[x_pos][y_pos] == '':
                    self.grid[x_pos][y_pos] = "R" + str(i)
                    print("R" + str(i))
                    break
        print(self.grid)

    def add_agent(self, agent):
        self.agents.append(agent)

    def log_game(self):
        directory = "logs"
        file_path = os.path.join(directory, self.logfile)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path, "ab") as log:
            # agent_positions = np.empty(len(self.agents), dtype=np.float32)
            # for i in range(len(agent_positions)):
            #     agent_positions[i] = str(self.agents(i).pos_x) + " " + str(self.agents(i).pos_y)
            # np.savetxt(log, self.grid, fmt='%s', delimiter=' ')
            # log.write(";")
            # np.savetxt(log, agent_positions, fmt='%d', delimiter=' ')
            # log.write("\n")
            pickle.dump(self, log)

    def read_all_objects(filename):
        objects = []
        with open(filename, 'rb') as file:
            while True:
                try:
                    obj = pickle.load(file)
                    objects.append(obj)
                except EOFError:
                    break
        return

game = Game(5, 5, 3)
game.generate()
game.log_game()