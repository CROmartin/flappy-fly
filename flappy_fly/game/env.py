"""Deterministic 50 Hz environment. No renderer or brain imports."""
import random
from flappy_fly.config import Config
from .entities import Pipe
from .physics import integrate, circle_rectangle
from .state import FlappyState, WAIT, FLAP


class FlappyEnv:
    def __init__(self, seed=42, config=None):
        self.config = config or Config()
        self.seed = seed
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.seed = int(seed)
        self.rng = random.Random(self.seed)
        g = self.config.game
        self.y = (g.height - g.ground_height) / 2
        self.velocity = 0.0
        self.score = self.steps = 0
        self.alive = True
        self.last_gap = self.y
        self.pipes = []
        for i in range(4):
            self.pipes.append(self._pipe(g.width + i * g.pipe_spacing))
        return self.state

    def _pipe(self, x):
        g = self.config.game
        lo = g.gap_margin + g.gap_size / 2
        hi = g.height - g.ground_height - lo
        self.last_gap = self.rng.uniform(max(lo, self.last_gap - g.max_gap_change),
                                         min(hi, self.last_gap + g.max_gap_change))
        return Pipe(float(x), self.last_gap)

    @property
    def state(self):
        g = self.config.game
        # Keep targeting the current opening until the whole fly clears it.
        pipe = next(p for p in self.pipes if p.x + g.pipe_width >= g.fly_x - g.fly_radius)
        return FlappyState(self.y, self.velocity, pipe.x, pipe.gap_center, g.gap_size,
                           pipe.x - g.fly_x - g.fly_radius, self.score, self.alive,
                           self.steps * self.config.physics.dt)

    def step(self, action):
        if action not in (WAIT, FLAP):
            raise ValueError("Action must be WAIT (0) or FLAP (1)")
        if not self.alive:
            return self.state, 0.0, True, {"reason": "already_dead"}
        g, physics = self.config.game, self.config.physics
        self.y, self.velocity = integrate(self.y, self.velocity, action, physics)
        self.steps += 1
        for pipe in self.pipes:
            pipe.x -= g.pipe_speed * physics.dt
        floor = g.height - g.ground_height
        reason = None
        if self.y - g.fly_radius <= 0 or self.y + g.fly_radius >= floor:
            reason = "boundary"
        for pipe in self.pipes:
            top = pipe.gap_center - g.gap_size / 2
            bottom = pipe.gap_center + g.gap_size / 2
            if (circle_rectangle(g.fly_x, self.y, g.fly_radius, pipe.x, 0, g.pipe_width, top)
                    or circle_rectangle(g.fly_x, self.y, g.fly_radius, pipe.x, bottom, g.pipe_width, floor - bottom)):
                reason = "pipe"
        self.alive = reason is None
        passed = 0
        if self.alive:
            for pipe in self.pipes:
                if not pipe.passed and pipe.x + g.pipe_width < g.fly_x - g.fly_radius:
                    pipe.passed = True
                    self.score += 1
                    passed += 1
        self.pipes = [p for p in self.pipes if p.x + g.pipe_width > -20]
        while len(self.pipes) < 4:
            self.pipes.append(self._pipe(self.pipes[-1].x + g.pipe_spacing))
        return self.state, float(passed) if self.alive else -1.0, not self.alive, {"reason": reason, "passed": passed}
