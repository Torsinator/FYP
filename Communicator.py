from Channel import Message, Channel

class Communicator:
    def __init__(self, agent_id, channel : Channel):
        self.channel = channel
        self.agent_id = agent_id

    def generate_message_summary(self):
        messages = self.channel.get_agent_messages(self.agent_id)
        # Perform message summary stuff

    def communicate(self):
        # Do AI model stuff
        pass
