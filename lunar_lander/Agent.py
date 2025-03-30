from Communicator import Communicator
from Action import Action
from Channel import Channel

class Agent:
    def __init__(self, id : str, env, channel : Channel, goal : str):
        self.communicator = Communicator(channel)
        self.action = Action()
        self.env = env
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
        observations = self.env.get_observations()
