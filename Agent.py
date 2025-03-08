from Communicator import Communicator
from Action import Action
from Game import Game

class Agent:
    def __init__(self, communicator : Communicator, action, game : Game, channel, goal, id):
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

    # Returns move
    def make_turn(self):
        messages = self.communicator.generate_message_summary()
        observations = self.game.get_observations()
