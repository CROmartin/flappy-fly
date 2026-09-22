import json
from pathlib import Path
import numpy as np
from flappy_fly.config import Config
from flappy_fly.brain.encoder import SensoryEncoder
from .base import Cooldown


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
    # Reject silent physics/encoder/trace/control changes. UI and training options
    # have no effect on inference, so they need not match.
    for section in ('game', 'physics', 'encoder', 'brain', 'control'):
        if config.to_dict()[section] != metadata['config'][section]:
            raise ValueError(f'Model/runtime {section} configuration mismatch; use the training config')
    model = Readout.load(path)
    if len(model.basis[0]) != metadata['feature_count']:
        raise ValueError('Model feature count mismatch')
    return model, metadata


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
        self.flap_probability = float(self.model.predict(features))
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
        self.flap_probability = float(self.model.predict(self.encoder.raw(state)))
        return self.cooldown.apply(self.flap_probability >= self.config.control.decision_threshold, state.elapsed_time)
