class HumanController:
    def __init__(self):
        self.reset()

    def reset(self, seed=42):
        self.pending = False

    def flap(self):
        self.pending = True

    def act(self, state):
        action = int(self.pending)
        self.pending = False
        return action
