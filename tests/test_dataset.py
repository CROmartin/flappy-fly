import numpy as np
import pytest
from flappy_fly.training.dataset import episode_splits, validate, save_dataset, load_dataset, merge_datasets
from flappy_fly.training.train import balanced_indices
from flappy_fly.training.metrics import classification


def sample_arrays():
    return {'features': np.ones((10, 4)), 'raw_inputs': np.ones((10, 3)), 'states': np.zeros((10, 9)),
            'encoder_values': np.zeros((10, 6)), 'oracle_action': np.tile([0, 1], 5),
            'actual_action': np.tile([0, 1], 5), 'episode_id': np.repeat(np.arange(5), 2),
            'step': np.tile([0, 1], 5), 'seed': np.repeat(np.arange(5), 2), 'score': np.zeros(10),
            'split': np.repeat(episode_splits(5), 2), 'dnp01': np.zeros(10)}


def test_roundtrip_and_leakage_detection(tmp_path):
    arrays = sample_arrays()
    output = tmp_path / 'data.npz'
    save_dataset(output, arrays, {})
    loaded, meta = load_dataset(output)
    np.testing.assert_array_equal(loaded['features'], arrays['features'])
    assert meta['episodes'] == 5
    arrays['split'][0] = 'test'
    with pytest.raises(ValueError, match='boundaries'):
        validate(arrays)


def test_split_and_balancing():
    assert episode_splits(50).count('train') == 35
    with pytest.raises(ValueError):
        episode_splits(2)
    labels = np.array([0] * 90 + [1] * 10)
    idx = balanced_indices(labels, np.arange(100), np.random.default_rng(42))
    assert labels[idx].sum() == 10 and len(idx) == 20
    with pytest.raises(ValueError):
        balanced_indices(labels, np.arange(20), np.random.default_rng(42))
    m = classification(labels, np.zeros(100))
    assert m['accuracy'] == .9 and m['f1_flap'] == 0
    assert m['confusion_matrix'] == [[90, 0], [10, 0]]


def test_merge_keeps_holdouts_and_rejects_reused_seeds(tmp_path):
    a = sample_arrays()
    b = sample_arrays()
    b['seed'] += 100
    b['split'][:] = 'train'
    save_dataset(tmp_path / 'a.npz', a, {})
    save_dataset(tmp_path / 'b.npz', b, {})
    merge_datasets([tmp_path / 'a.npz', tmp_path / 'b.npz'], tmp_path / 'c.npz')
    c, _ = load_dataset(tmp_path / 'c.npz')
    assert len(np.unique(c['episode_id'])) == 10
    np.testing.assert_array_equal(c['seed'][c['split'] == 'test'], a['seed'][a['split'] == 'test'])
    with pytest.raises(ValueError, match='fresh'):
        merge_datasets([tmp_path / 'a.npz'] * 2, tmp_path / 'bad.npz')
