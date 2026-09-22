import os
import numpy as np
import pytest
from flappy_fly.game.env import FlappyEnv

pytestmark = [pytest.mark.brain, pytest.mark.skipif(os.environ.get('RUN_BRAIN_TESTS') != '1', reason='opt-in real connectome')]


def test_real_brain_trace_reset_frozen_weights():
    from flappy_fly.brain.adapter import BrainAdapter
    adapter = BrainAdapter()
    weights = adapter.brain.weights.copy()
    state = FlappyEnv().state
    adapter.reset(15)
    first = np.stack([adapter.advance(state) for _ in range(12)])
    assert first.shape == (12, adapter.feature_count)
    assert adapter.feature_count > 0
    adapter.reset(15)
    second = np.stack([adapter.advance(state) for _ in range(12)])
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(weights, adapter.brain.weights)
    assert all(len(group) for group in adapter.groups.values())


def test_model_import_before_brain_is_supported():
    # Regression: do not mutate NUMBA_NUM_THREADS after importing flybrain.
    from flybrain.reservoir import Readout
    from flappy_fly.brain.adapter import BrainAdapter
    adapter = BrainAdapter()
    assert adapter.advance(FlappyEnv().state).shape == (adapter.feature_count,)
