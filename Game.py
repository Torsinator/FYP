import numpy as np
import datetime
import os
import Resource

generator = np.random.default_rng(0)

class Game:
    def __init__(self, rows, cols, num_resources, agents = None):
        if agents is None:
            agents = list()
        self.cols = cols
        self.rows = rows
        self.num_resources = num_resources
        self.resources = list()
        self.grid = np.full((rows, cols), '.', dtype='<U10')
        self.agents = agents
        self.time_step = 0
        self.logfile = datetime.datetime.now().strftime("logfile_%Y%m%d_%H%M%S.txt")

    def generate(self):
        # Generate positions for the resources
        for i in range(1, self.num_resources + 1):
            while True:
                x_pos = generator.integers(0, self.cols)
                y_pos = generator.integers(0, self.rows)
                # Make sure there are no two resources on the same tile
                if self.grid[y_pos][x_pos] == '.':
                    self.grid[y_pos][x_pos] = "R" + str(i)
                    self.resources.append(Resource(i, x_pos, y_pos))
                    break
        print(self.grid)
        self.log_grid()

    def add_agent(self, agent):
        self.agents.append(agent)
        x_pos = generator.integers(0, self.cols)
        y_pos = generator.integers(0, self.rows)
        if self.grid[y_pos][x_pos] == '.':
            self.grid[y_pos][x_pos] = "A" + str(len(self.agents))
        else:
            self.grid[y_pos][x_pos] += "A" + str(len(self.agents))


    def get_log_file(self):
        directory = "logs"
        file_path = os.path.join(directory, self.logfile)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return file_path

    def log_grid(self):
        file = self.get_log_file()
        with open(file, "a") as log:
            log.write(f'{self.time_step} GRID: {self.rows} {self.cols}\n')
            np.savetxt(log, self.grid, "%2s")

    def log_message(self, message):
        file = self.get_log_file()
        with open(file, "a") as log:
            log.write(f'{self.time_step} MESSAGE: {message}\n')

    def normalise_resource(self, resource_id):
        return resource_id/self.num_resources

    def check_resource(self, x_pos, y_pos):
        if x_pos >= self.cols or x_pos < 0 or y_pos >= self.rows or y_pos < 0:
            return 0
        if self.grid[y_pos][x_pos] == ".":
            return 0
        return self.normalise_resource(int(self.grid[y_pos][x_pos][1:]))

    # [timestep, x_pos, y_pos, current, north, south, east, west]
    def get_observations(self, agent):
        x_pos = agent.x
        y_pos = agent.y
        current = self.check_resource(x_pos, y_pos)
        north = self.check_resource(x_pos, y_pos + 1)
        south = self.check_resource(x_pos, y_pos - 1)
        east = self.check_resource(x_pos + 1, y_pos)
        west = self.check_resource(x_pos - 1, y_pos)
        return np.array([self.time_step, x_pos, y_pos, current, north, south, east, west], dtype=np.float32)

    def win(self):
        for agent in self.agents:
            if agent.goal != self.grid[agent.y][agent.x]:
                return False
        return True

# game.log_game()