from Communicator import Communicator
from Action import Action

class Agent:
    def __init__(self, communicator, action, game, channel):
        self.communicator = communicator
        self.action = action
        self.game = game
        self.channel = channel
        self.x = 0
        self.y = 0

    def