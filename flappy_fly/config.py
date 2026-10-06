"""Immutable experiment settings, serialized beside every dataset/model/run."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass(frozen=True)
class GameConfig:
    width: int = 720
    height: int = 600
    ground_height: int = 40
    fly_x: float = 170.0
    fly_radius: float = 12.0
    pipe_width: float = 72.0
    pipe_speed: float = 155.0
    pipe_spacing: float = 280.0
    gap_size: float = 190.0
    gap_margin: float = 55.0
    max_gap_change: float = 95.0


@dataclass(frozen=True)
class PhysicsConfig:
    dt: float = 0.020
    gravity: float = 850.0
    flap_velocity: float = -270.0
    max_fall_speed: float = 480.0


@dataclass(frozen=True)
class EncoderConfig:
    looming_gain: float = 0.65
    target_gain: float = 1.0
    velocity_gain: float = 0.55
    sensor_range: float = 450.0
    vertical_error_range: float = 160.0
    max_fall_speed: float = 480.0
    max_rise_speed: float = 270.0
    fly_radius: float = 12.0
    clearance_margin: float = 95.0
    # Lookahead “will this flap / this wait kill me?” (hand-designed, not biological).
    look_ahead: float = 0.24
    dt: float = 0.020
    gravity: float = 850.0
    flap_velocity: float = -270.0
    fly_x: float = 170.0
    pipe_width: float = 72.0
    pipe_speed: float = 155.0
    arena_height: float = 600.0
    ground_height: float = 40.0
    map_version: int = 6  # bump when sensory semantics change (not just gains)
    # Smaller = too-high / too-low reach full strength sooner (more impact).
    position_gate_range: float = 160.0


@dataclass(frozen=True)
class BrainConfig:
    device: str = "auto"
    dt: float = 0.020
    trace_tau: float = 0.1
    brain_steps_per_action: int = 1
    threads: int = 4
    instinct_threshold: float = 0.8


@dataclass(frozen=True)
class ControlConfig:
    decision_threshold: float = 0.62
    flap_cooldown: float = 0.14
    oracle_horizon: float = 0.20
    oracle_offset: float = 20.0
    oracle_deadband: float = 12.0
    ceiling_margin: float = 65.0
    random_flap_probability: float = 0.06
    # When too high, shrink P(FLAP); when too low, push it up. Actuator-only.
    too_high_flap_dampen: float = 0.92
    too_low_flap_boost: float = 0.40


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    max_steps: int = 1500
    components: int = 20
    regularization: float = 0.1


@dataclass(frozen=True)
class UIConfig:
    panel_width: int = 460
    history_steps: int = 100
    smoothing: float = 0.85


@dataclass(frozen=True)
class Config:
    game: GameConfig = field(default_factory=GameConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    def __post_init__(self):
        if self.physics.dt <= 0 or self.brain.dt != self.physics.dt:
            raise ValueError("Brain and physics timesteps must match and be positive")
        if self.brain.brain_steps_per_action < 1 or self.brain.trace_tau <= 0:
            raise ValueError("Invalid brain timing")
        if self.game.gap_size <= 2 * self.game.fly_radius:
            raise ValueError("Gap must fit the fly")
        if self.game.gap_size + 2 * self.game.gap_margin >= self.game.height - self.game.ground_height:
            raise ValueError("Gap and margins must fit the arena")
        for gain in (self.encoder.looming_gain, self.encoder.target_gain, self.encoder.velocity_gain):
            if not 0 <= gain <= 1:
                raise ValueError("Stimulus gains must be in [0, 1]")
        if min(self.encoder.sensor_range, self.encoder.vertical_error_range,
               self.encoder.max_fall_speed, self.encoder.max_rise_speed,
               self.encoder.fly_radius, self.encoder.clearance_margin,
               self.encoder.look_ahead, self.encoder.dt, self.encoder.gravity,
               self.encoder.pipe_width, self.encoder.pipe_speed,
               self.encoder.arena_height, self.encoder.ground_height,
               self.encoder.position_gate_range) <= 0:
            raise ValueError("Encoder normalization ranges must be positive")
        if self.encoder.flap_velocity >= 0:
            raise ValueError("Encoder flap_velocity must be negative (upward)")
        if self.encoder.dt != self.physics.dt:
            raise ValueError("Encoder lookahead dt must match physics dt")
        if not 0 <= self.control.decision_threshold <= 1:
            raise ValueError("Decision threshold must be in [0, 1]")
        for gain in (self.control.too_high_flap_dampen, self.control.too_low_flap_boost):
            if not 0 <= gain <= 1:
                raise ValueError("Height bias gains must be in [0, 1]")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        types = {"game": GameConfig, "physics": PhysicsConfig, "encoder": EncoderConfig,
                 "brain": BrainConfig, "control": ControlConfig, "training": TrainingConfig, "ui": UIConfig}
        return cls(**{key: types[key](**fields) for key, fields in value.items()})

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text())) if path else cls()
