class Resource:
    def __init__(self, id):
        self.id = id
        self.x = 0
        self.y = 0

    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def get_display(self):
        return f"R{self.id}"