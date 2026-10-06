import numpy as np
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
        wants = self.adapter.dnp01_activity > self.config.brain.instinct_threshold
        # Too high → prefer falling even if DNp01 is loud; too low keeps the escape flap.
        scale = self.config.encoder.position_gate_range
        error = state.bird_y - state.next_gap_center_y
        too_high = float(np.clip((-error) / scale, 0, 1))
        too_low = float(np.clip(error / scale, 0, 1))
        too_high = max(too_high, float(self.adapter.inputs.get('LC10a-R', 0.0)))
        too_low = max(too_low, float(self.adapter.inputs.get('LC10a-L', 0.0)))
        if too_high >= 0.35 and too_low < 0.35:
            wants = False
        return self.cooldown.apply(wants, state.elapsed_time)
