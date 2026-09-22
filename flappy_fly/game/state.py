from dataclasses import dataclass

WAIT = 0
FLAP = 1


@dataclass(frozen=True)
class FlappyState:
    bird_y: float
    bird_velocity_y: float
    next_pipe_x: float
    next_gap_center_y: float
    next_gap_size: float
    distance_to_pipe: float
    score: int
    alive: bool
    elapsed_time: float
