"""Short-horizon teacher. query() is side-effect free for DAgger labels."""
from flappy_fly.config import Config
from .base import Cooldown


class OracleController:
    def __init__(self, config=None):
        self.config = config or Config()
        self.cooldown = Cooldown(self.config.control.flap_cooldown)

    def reset(self, seed=42):
        self.cooldown.reset()

    def query(self, state):
        c, p = self.config.control, self.config.physics
        horizon = c.oracle_horizon
        predicted = state.bird_y + state.bird_velocity_y * horizon + 0.5 * p.gravity * horizon**2
        target = state.next_gap_center_y + c.oracle_offset
        safe = state.bird_y > c.ceiling_margin
        return int(safe and self.cooldown.ready(state.elapsed_time)
                   and predicted > target + c.oracle_deadband)

    def observe_action(self, state, actual_action):
        # During learner rollouts the teacher's cooldown follows actions actually taken.
        if actual_action:
            self.cooldown.last_flap = state.elapsed_time

    def act(self, state):
        action = self.query(state)
        self.observe_action(state, action)
        return action
