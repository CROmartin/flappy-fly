import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from flappy_fly.config import Config, EncoderConfig
from flappy_fly.brain.encoder import SensoryEncoder
from .base import Cooldown

# Hand-tuned at play time; not part of the fitted readout.
ACTUATOR_KEYS = {'decision_threshold', 'flap_cooldown', 'too_high_flap_dampen', 'too_low_flap_boost'}


def _encoder_dict(section):
    fields = {k: v for k, v in section.items() if k in EncoderConfig.__dataclass_fields__}
    return asdict(EncoderConfig(**fields))


def load_model(path, kind, config):
    from flybrain.reservoir import Readout
    import flybrain
    from flappy_fly.training.dataset import digest
    if not path:
        raise ValueError(f'{kind} mode requires --model (train one first)')
    path = Path(path)
    metadata = json.loads(path.with_suffix('.json').read_text())
    if metadata['model_kind'] != kind:
        raise ValueError(f'Expected a {kind} model, got {metadata["model_kind"]}')
    if digest(path) != metadata['sha256']:
        raise ValueError('Model checksum mismatch')
    if metadata['flybrain_version'] != flybrain.__version__:
        raise ValueError('flybrain version differs from training')
    # Reject silent physics/encoder/trace changes. Threshold/cooldown/height bias are
    # actuator rules on top of the fitted readout, so they may be retuned at runtime.
    for section in ('game', 'physics', 'brain'):
        if config.to_dict()[section] != metadata['config'][section]:
            raise ValueError(f'Model/runtime {section} configuration mismatch; use the training config')
    if _encoder_dict(config.to_dict()['encoder']) != _encoder_dict(metadata['config']['encoder']):
        raise ValueError('Model/runtime encoder configuration mismatch; use the training config')
    trained_control = {k: v for k, v in metadata['config']['control'].items() if k not in ACTUATOR_KEYS}
    runtime_control = {k: v for k, v in config.to_dict()['control'].items() if k not in ACTUATOR_KEYS}
    if trained_control != runtime_control:
        raise ValueError('Model/runtime control configuration mismatch; use the training config')
    model = Readout.load(path)
    if len(model.basis[0]) != metadata['feature_count']:
        raise ValueError('Model feature count mismatch')
    return model, metadata


def apply_height_bias(probability, state, config, inputs=None):
    """Too high → brake flaps; too low → encourage flaps. Lets gravity win when high."""
    c = config.control
    scale = config.encoder.position_gate_range
    error = state.bird_y - state.next_gap_center_y
    too_high = float(np.clip((-error) / scale, 0, 1))
    too_low = float(np.clip(error / scale, 0, 1))
    if inputs:
        too_high = max(too_high, float(inputs.get('LC10a-R', 0.0)))
        too_low = max(too_low, float(inputs.get('LC10a-L', 0.0)))
    probability = float(probability) * (1.0 - c.too_high_flap_dampen * too_high)
    probability = probability + (1.0 - probability) * c.too_low_flap_boost * too_low
    return float(np.clip(probability, 0.0, 1.0))


class TrainedBrainController:
    def __init__(self, model, config=None, adapter=None):
        from flappy_fly.brain.adapter import BrainAdapter
        self.config = config or Config()
        self.model, self.metadata = load_model(model, 'brain', self.config)
        self.adapter = adapter or BrainAdapter(self.config)
        if self.adapter.feature_indices.tolist() != self.metadata['feature_indices']:
            raise ValueError('Descending neuron feature order differs from training')
        if self.adapter.brain.device != self.metadata['device']:
            raise ValueError('Brain backend differs from training; run a separately recorded experiment')
        self.cooldown = Cooldown(self.config.control.flap_cooldown)
        self.flap_probability = 0.0

    def reset(self, seed=42):
        self.adapter.reset(seed)
        self.cooldown.reset()
        self.flap_probability = 0.0

    def act(self, state):
        features = self.adapter.advance(state)
        raw_p = float(self.model.predict(features))
        self.flap_probability = apply_height_bias(raw_p, state, self.config, self.adapter.inputs)
        return self.cooldown.apply(self.flap_probability >= self.config.control.decision_threshold, state.elapsed_time)


class DirectInputBaseline:
    def __init__(self, model, config=None):
        self.config = config or Config()
        self.model, self.metadata = load_model(model, 'direct', self.config)
        self.encoder = SensoryEncoder(self.config.encoder)
        self.cooldown = Cooldown(self.config.control.flap_cooldown)
        self.flap_probability = 0.0

    def reset(self, seed=42):
        self.cooldown.reset()
        self.flap_probability = 0.0

    def act(self, state):
        raw_p = float(self.model.predict(self.encoder.raw(state)))
        self.flap_probability = apply_height_bias(raw_p, state, self.config, self.encoder.values(state))
        return self.cooldown.apply(self.flap_probability >= self.config.control.decision_threshold, state.elapsed_time)
