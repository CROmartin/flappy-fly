from dataclasses import dataclass


@dataclass
class Pipe:
    x: float
    gap_center: float
    passed: bool = False
