import random
from flappy_fly.config import Config


class RandomController:
    def __init__(self, config=None):
        self.config = config or Config()
        self.reset()

    def reset(self, seed=42):
        self.rng = random.Random(seed)

    def act(self, state):
        return int(self.rng.random() < self.config.control.random_flap_probability)
