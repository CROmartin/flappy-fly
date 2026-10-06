"""Task adapter, not a biological model of vertical vision.

LC4 = flap-looks-fatal, gated by being too high (upward danger).
LPLC2 = wait-looks-fatal, gated by being too low (downward danger).
Lookahead only scores the matching collision side: flap→ceiling/upper pipe,
wait→floor/lower pipe. Falling past an upper lip while too high does NOT count
as “wait fatal” (that false alarm made the agent want to flap).
"""
from types import SimpleNamespace
import numpy as np
from flappy_fly.config import EncoderConfig
from flappy_fly.game.physics import integrate, circle_rectangle
from flappy_fly.game.state import FLAP, WAIT

CHANNELS = ('LC4', 'LPLC2', 'LC10a-L', 'LC10a-R', 'LPLC1-L', 'LPLC1-R')


class SensoryEncoder:
    def __init__(self, config=None):
        self.config = config or EncoderConfig()

    def _physics(self):
        c = self.config
        return SimpleNamespace(dt=c.dt, gravity=c.gravity, flap_velocity=c.flap_velocity,
                               max_fall_speed=c.max_fall_speed)

    def _side_gates(self, state):
        """How strongly we are on the high vs low side of the gap (0..1)."""
        error = state.bird_y - state.next_gap_center_y
        scale = self.config.position_gate_range
        too_high = float(np.clip((-error) / scale, 0, 1))
        too_low = float(np.clip(error / scale, 0, 1))
        return too_high, too_low

    def instantaneous_threats(self, state):
        """Upper/lower wall threat from where the fly is right now."""
        c = self.config
        top_lip = state.next_gap_center_y - state.next_gap_size / 2
        bottom_lip = state.next_gap_center_y + state.next_gap_size / 2
        top_clearance = (state.bird_y - c.fly_radius) - top_lip
        bottom_clearance = bottom_lip - (state.bird_y + c.fly_radius)
        proximity = float(np.clip(1 - state.distance_to_pipe / c.sensor_range, 0, 1))
        top_threat = proximity * float(np.clip(1 - top_clearance / c.clearance_margin, 0, 1))
        bottom_threat = proximity * float(np.clip(1 - bottom_clearance / c.clearance_margin, 0, 1))
        return top_threat, bottom_threat

    def _trajectory_risk(self, state, flap_first, side):
        """Risk of collisions on one side only: 'upper' or 'lower'."""
        c = self.config
        physics = self._physics()
        y, velocity = state.bird_y, state.bird_velocity_y
        pipe_x = state.next_pipe_x
        gap_center, gap_size = state.next_gap_center_y, state.next_gap_size
        floor = c.arena_height - c.ground_height
        steps = max(1, int(round(c.look_ahead / c.dt)))
        worst = 0.0
        for step in range(steps):
            action = FLAP if flap_first and step == 0 else WAIT
            y, velocity = integrate(y, velocity, action, physics)
            pipe_x -= c.pipe_speed * c.dt
            top = gap_center - gap_size / 2
            bottom = gap_center + gap_size / 2
            if side == 'upper':
                if y - c.fly_radius <= 0:
                    return 1.0
                if circle_rectangle(c.fly_x, y, c.fly_radius, pipe_x, 0, c.pipe_width, top):
                    return 1.0
            else:
                if y + c.fly_radius >= floor:
                    return 1.0
                if circle_rectangle(c.fly_x, y, c.fly_radius, pipe_x, bottom, c.pipe_width, floor - bottom):
                    return 1.0
            if pipe_x - c.fly_radius <= c.fly_x <= pipe_x + c.pipe_width + c.fly_radius:
                if side == 'upper':
                    top_clear = (y - c.fly_radius) - top
                    worst = max(worst, float(np.clip(1 - top_clear / c.clearance_margin, 0, 1)))
                else:
                    bottom_clear = bottom - (y + c.fly_radius)
                    worst = max(worst, float(np.clip(1 - bottom_clear / c.clearance_margin, 0, 1)))
        return worst

    def action_risks(self, state):
        """flap_risk only when too high; wait_risk only when too low."""
        too_high, too_low = self._side_gates(state)
        flap_upper = self._trajectory_risk(state, True, 'upper')
        wait_lower = self._trajectory_risk(state, False, 'lower')
        return flap_upper * too_high, wait_lower * too_low

    def threats(self, state):
        """LC4←upper-now×too_high + flap risk; LPLC2←lower-now×too_low + wait risk."""
        top_now, bottom_now = self.instantaneous_threats(state)
        flap_risk, wait_risk = self.action_risks(state)
        too_high, too_low = self._side_gates(state)
        # Instantaneous lip threat also gated so "above gap" does not light wait-fatal.
        return (max(top_now * max(too_high, 0.2), flap_risk),
                max(bottom_now * max(too_low, 0.2), wait_risk))

    def raw(self, state):
        """Direct features: gated flap risk, gated wait risk, signed gap error."""
        c = self.config
        flap_risk, wait_risk = self.action_risks(state)
        error = state.bird_y - state.next_gap_center_y
        return np.asarray([flap_risk, wait_risk,
                           np.clip(error / c.vertical_error_range, -1, 1)], dtype=np.float32)

    def values(self, state):
        c = self.config
        flap_side, wait_side = self.threats(state)
        error = float(np.clip((state.bird_y - state.next_gap_center_y) / c.vertical_error_range, -1, 1))
        velocity_scale = c.max_fall_speed if state.bird_velocity_y >= 0 else c.max_rise_speed
        velocity = float(np.clip(state.bird_velocity_y / velocity_scale, -1, 1))
        return dict(zip(CHANNELS, map(float, (
            flap_side * c.looming_gain, wait_side * c.looming_gain,
            max(error, 0) * c.target_gain, max(-error, 0) * c.target_gain,
            max(velocity, 0) * c.velocity_gain, max(-velocity, 0) * c.velocity_gain))))

    def encode(self, state, groups):
        return [(groups[name], value) for name, value in self.values(state).items()]
