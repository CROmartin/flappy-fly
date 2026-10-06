from dataclasses import replace
import numpy as np
from flappy_fly.brain.encoder import SensoryEncoder
from flappy_fly.game.env import FlappyEnv


def test_wall_threat_channels_and_bounds():
    encoder = SensoryEncoder()
    state = FlappyEnv().state
    far = encoder.values(replace(state, distance_to_pipe=600, next_pipe_x=600 + 170 + 12))
    assert far['LC4'] == 0 and far['LPLC2'] == 0
    centered = replace(state, distance_to_pipe=40, next_pipe_x=40 + 170 + 12,
                       bird_y=state.next_gap_center_y, bird_velocity_y=0.0)
    centered_v = encoder.values(centered)
    assert centered_v['LC4'] < 0.9 and centered_v['LPLC2'] < 0.9
    # Above gap center: wait must NOT look fatal just because falling clips the top lip.
    near_top = replace(centered, bird_y=state.next_gap_center_y - 50)
    flap_risk, wait_risk = encoder.action_risks(near_top)
    assert wait_risk == 0.0
    assert encoder.values(near_top)['LC4'] >= encoder.values(near_top)['LPLC2']
    # Below gap center: flap-upper risk gated off; wait-lower can fire.
    near_bottom = replace(centered, bird_y=state.next_gap_center_y + 50)
    flap_risk_b, wait_risk_b = encoder.action_risks(near_bottom)
    assert flap_risk_b == 0.0
    assert encoder.values(near_bottom)['LPLC2'] >= encoder.values(near_bottom)['LC4']
    for sign in (-1, 1):
        values = encoder.values(replace(state, bird_y=state.next_gap_center_y + sign * 10000,
                                        bird_velocity_y=sign * 10000, distance_to_pipe=10,
                                        next_pipe_x=10 + 170 + 12))
        assert values['LC10a-L' if sign == 1 else 'LC10a-R'] > 0
        assert values['LPLC1-L' if sign == 1 else 'LPLC1-R'] > 0
        assert all(0 <= x <= 1 for x in values.values())
    raw = encoder.raw(centered)
    assert raw.shape == (3,)
    groups = {name: np.array([i]) for i, name in enumerate(far)}
    assert len(encoder.encode(state, groups)) == 6


def test_lookahead_flags_fatal_flap_into_ceiling():
    encoder = SensoryEncoder()
    state = FlappyEnv().state
    doom = replace(state, bird_y=40.0, bird_velocity_y=-200.0,
                   distance_to_pipe=200, next_pipe_x=200 + 170 + 12)
    flap_risk, wait_risk = encoder.action_risks(doom)
    assert flap_risk == 1.0
    assert wait_risk == 0.0


def test_above_gap_wait_does_not_count_upper_clip_as_wait_fatal():
    encoder = SensoryEncoder()
    state = FlappyEnv().state
    # High, near pipe: falling may graze upper geometry, but that is NOT "wait fatal".
    high = replace(state, bird_y=state.next_gap_center_y - 70, bird_velocity_y=50.0,
                   distance_to_pipe=25, next_pipe_x=25 + 170 + 12)
    _, wait_risk = encoder.action_risks(high)
    assert wait_risk == 0.0
    assert encoder.values(high)['LC10a-R'] > 0  # too high channel on
