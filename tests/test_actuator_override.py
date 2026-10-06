"""Threshold/cooldown may differ from the training sidecar; other control fields may not."""
import json
from pathlib import Path
import numpy as np
import pytest
from flappy_fly.config import Config
from flappy_fly.controllers.trained import load_model, DirectInputBaseline
from flappy_fly.training.dataset import save_dataset
from flappy_fly.training.train import train
from test_dataset import sample_arrays


def _train_direct(tmp_path):
    arrays = sample_arrays()
    arrays['raw_inputs'][:, 1] = arrays['oracle_action'] * 2 - 1
    dataset, output = tmp_path / 'sample.npz', tmp_path / 'model.npz'
    save_dataset(dataset, arrays, {'config': Config().to_dict(), 'flybrain_version': '0.1.0',
                                  'feature_indices': list(range(4)), 'device': 'cpu'})
    train(dataset, output, direct=True)
    return output


def test_threshold_override_allowed(tmp_path):
    output = _train_direct(tmp_path)
    tuned = Config.from_dict({**Config().to_dict(),
                              'control': {**Config().to_dict()['control'], 'decision_threshold': 0.8}})
    model, _ = load_model(output, 'direct', tuned)
    assert model is not None
    controller = DirectInputBaseline(output, tuned)
    assert controller.config.control.decision_threshold == 0.8


def test_non_actuator_control_mismatch_rejected(tmp_path):
    output = _train_direct(tmp_path)
    bad = Config.from_dict({**Config().to_dict(),
                            'control': {**Config().to_dict()['control'], 'oracle_horizon': 0.3}})
    with pytest.raises(ValueError, match='control configuration mismatch'):
        load_model(output, 'direct', bad)
