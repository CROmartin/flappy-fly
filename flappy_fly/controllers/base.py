from typing import Protocol
from flappy_fly.game.state import FlappyState


class Controller(Protocol):
    def reset(self, seed: int = 42) -> None: ...
    def act(self, state: FlappyState) -> int: ...


class Cooldown:
    def __init__(self, seconds):
        self.seconds = seconds
        self.reset()

    def reset(self):
        self.last_flap = float('-inf')

    def ready(self, time):
        return time - self.last_flap >= self.seconds - 1e-9

    def apply(self, wants_flap, time):
        action = int(wants_flap and self.ready(time))
        if action:
            self.last_flap = time
        return action
