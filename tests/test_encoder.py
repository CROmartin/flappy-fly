from dataclasses import replace
import numpy as np
from flappy_fly.brain.encoder import SensoryEncoder
from flappy_fly.game.env import FlappyEnv


def test_looming_signed_channels_and_bounds():
    encoder = SensoryEncoder()
    state = FlappyEnv().state
    far = encoder.values(replace(state, distance_to_pipe=600))
    near = encoder.values(replace(state, distance_to_pipe=10))
    assert far['LC4'] == 0 < near['LC4'] <= 1
    for sign in (-1, 1):
        values = encoder.values(replace(state, bird_y=state.next_gap_center_y + sign * 10000,
                                        bird_velocity_y=sign * 10000))
        assert values['LC10a-L' if sign == 1 else 'LC10a-R'] > 0
        assert values['LPLC1-L' if sign == 1 else 'LPLC1-R'] > 0
        assert all(0 <= x <= 1 for x in values.values())
    groups = {name: np.array([i]) for i, name in enumerate(far)}
    assert len(encoder.encode(state, groups)) == 6
