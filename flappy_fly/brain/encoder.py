"""Task adapter, not a biological model of vertical vision.

LC10a L/R carries below/above; LPLC1 L/R carries falling/rising. The choice of
sides is arbitrary. No action, oracle label or cooldown enters the encoder.
"""
import numpy as np
from flappy_fly.config import EncoderConfig

CHANNELS = ('LC4', 'LPLC2', 'LC10a-L', 'LC10a-R', 'LPLC1-L', 'LPLC1-R')


class SensoryEncoder:
    def __init__(self, config=None):
        self.config = config or EncoderConfig()

    def raw(self, state):
        c = self.config
        error = state.bird_y - state.next_gap_center_y
        velocity_scale = c.max_fall_speed if state.bird_velocity_y >= 0 else c.max_rise_speed
        return np.asarray([np.clip(1 - state.distance_to_pipe / c.sensor_range, 0, 1),
                           np.clip(error / c.vertical_error_range, -1, 1),
                           np.clip(state.bird_velocity_y / velocity_scale, -1, 1)], dtype=np.float32)

    def values(self, state):
        c = self.config
        urgency, error, velocity = self.raw(state)
        return dict(zip(CHANNELS, map(float, (urgency * c.looming_gain, urgency * c.looming_gain,
            max(error, 0) * c.target_gain, max(-error, 0) * c.target_gain,
            max(velocity, 0) * c.velocity_gain, max(-velocity, 0) * c.velocity_gain))))

    def encode(self, state, groups):
        return [(groups[name], value) for name, value in self.values(state).items()]
