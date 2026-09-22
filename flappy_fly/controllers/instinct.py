from flappy_fly.config import Config
from flappy_fly.brain.adapter import BrainAdapter
from .base import Cooldown


class InstinctController:
    def __init__(self, config=None, adapter=None):
        self.config = config or Config()
        self.adapter = adapter or BrainAdapter(self.config)
        self.cooldown = Cooldown(self.config.control.flap_cooldown)
        self.flap_probability = None  # A thresholded trace is not a probability.

    def reset(self, seed=42):
        self.adapter.reset(seed)
        self.cooldown.reset()

    def act(self, state):
        self.adapter.advance(state)
        return self.cooldown.apply(self.adapter.dnp01_activity > self.config.brain.instinct_threshold,
                                   state.elapsed_time)
