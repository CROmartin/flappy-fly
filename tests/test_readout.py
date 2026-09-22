import numpy as np
import pytest
from flappy_fly.config import Config
from flappy_fly.training.dataset import save_dataset
from flappy_fly.training.train import train
from flappy_fly.controllers.trained import DirectInputBaseline
from flappy_fly.game.env import FlappyEnv
from test_dataset import sample_arrays


def test_library_model_roundtrip_and_integrity(tmp_path):
    from flybrain.reservoir import Readout
    arrays = sample_arrays()
    arrays['raw_inputs'][:, 1] = arrays['oracle_action'] * 2 - 1
    dataset, output = tmp_path / 'sample.npz', tmp_path / 'model.npz'
    save_dataset(dataset, arrays, {'config': Config().to_dict(), 'flybrain_version': '0.1.0',
                                  'feature_indices': list(range(4)), 'device': 'cpu'})
    meta = train(dataset, output, direct=True)
    assert meta['metrics']['test']['f1_flap'] == 1
    controller = DirectInputBaseline(output)
    loaded = Readout.load(output)
    np.testing.assert_allclose(controller.model.predict(arrays['raw_inputs']), loaded.predict(arrays['raw_inputs']))
    assert controller.act(FlappyEnv().state) in (0, 1)
    with open(output, 'ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='checksum'):
        DirectInputBaseline(output)
