import random

class Model():
    def __init__(self):
        pass

    def select_action(self):
        return random.randint(0, 3)
    
class DinoGamer:
    def __init__(self):
        self.model = Model()

    def select_action(self):
        return random.randint(0, 3)