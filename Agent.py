from Communicator import Communicator
from Action import Action

class Agent:
    def __init__(self, communicator, action, game, channel, goal, id):
        self.communicator = communicator
        self.action = action
        self.game = game
        self.goal = goal
        self.channel = channel
        self.x = 0
        self.y = 0
        self.id = id

    def get_display(self):
        return "A{self.id}"

    def make_turn(self):
        self.communicator.