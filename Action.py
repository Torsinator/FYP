import torch

class Action(torch.nn.Module):
    def __init__(self, pos_x, pos_y):
        super.__init__()
        self.pos_x = pos_x
        self.pos_y = pos_y

    def feedback(self):
        pass

    def reward_function(self):
        pass