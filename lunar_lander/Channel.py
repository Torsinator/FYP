class Message:
    def __init__(self, sender, reciever, content):
        self.sender = sender
        self.reciever = reciever
        self.content = content

class Channel:
    def __init__(self):
        self.messages = dict()

    def add_message(self, message : Message):
        reciever = message.reciever
        if self.messages[reciever]:
            self.messages[reciever].append(message)
        else:
            self.messages[reciever] = [message]

    def get_agent_messages(self, agent_id):
        return self.messages[agent_id]

    def reset_messages(self):
        self.messages.clear()