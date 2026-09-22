"""Non-pickled arrays plus auditable JSON. Splits belong to episodes, never rows."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

STATE_FIELDS = ('bird_y', 'bird_velocity_y', 'next_pipe_x', 'next_gap_center_y', 'next_gap_size',
                'distance_to_pipe', 'score', 'alive', 'elapsed_time')
REQUIRED = ('features', 'raw_inputs', 'states', 'encoder_values', 'oracle_action', 'actual_action',
            'episode_id', 'step', 'seed', 'score', 'split', 'dnp01')


def provenance():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, True
    return {'created_at': datetime.now(timezone.utc).isoformat(), 'git_commit': commit, 'git_dirty': dirty}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def class_counts(labels):
    n = len(labels)
    flaps = int(np.count_nonzero(labels))
    return {'WAIT': n - flaps, 'FLAP': flaps, 'percent_flap': 100 * flaps / max(n, 1)}


def episode_splits(episodes):
    if episodes < 5:
        raise ValueError('Collect at least 5 episodes for disjoint train/validation/test sets')
    train = min(episodes - 2, max(2, int(episodes * 0.7)))
    validation = max(1, int(episodes * 0.15))
    return ['train'] * train + ['validation'] * validation + ['test'] * (episodes - train - validation)


def validate(arrays):
    if any(k not in arrays for k in REQUIRED):
        raise ValueError('Dataset is missing required arrays')
    n = len(arrays['oracle_action'])
    if n == 0 or any(len(arrays[k]) != n for k in REQUIRED):
        raise ValueError('Dataset arrays must have equal, nonzero row counts')
    if arrays['features'].ndim != 2 or arrays['raw_inputs'].shape != (n, 3):
        raise ValueError('Invalid feature dimensions')
    for key in ('features', 'raw_inputs', 'states', 'encoder_values', 'dnp01'):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f'Nonfinite data: {key}')
    for key in ('oracle_action', 'actual_action'):
        if not np.isin(arrays[key], [0, 1]).all():
            raise ValueError('Actions must be binary')
    if not np.isin(arrays['split'], ['train', 'validation', 'test']).all():
        raise ValueError('Invalid split name')
    for key in ('episode_id', 'seed'):
        for value in np.unique(arrays[key]):
            if len(np.unique(arrays['split'][arrays[key] == value])) != 1:
                raise ValueError(f'{key} crosses split boundaries')


def save_dataset(path, arrays, metadata):
    path = Path(path)
    if path.suffix != '.npz':
        raise ValueError('Dataset output must end in .npz')
    validate(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    metadata = {**metadata, **provenance(), 'schema_version': 1, 'state_fields': STATE_FIELDS,
                'samples': len(arrays['oracle_action']), 'feature_count': arrays['features'].shape[1],
                'episodes': len(np.unique(arrays['episode_id'])), 'seeds': np.unique(arrays['seed']).tolist(),
                'class_counts': class_counts(arrays['oracle_action']), 'sha256': digest(path)}
    write_json(path.with_suffix('.json'), metadata)
    return metadata


def load_dataset(path):
    path = Path(path)
    metadata = json.loads(path.with_suffix('.json').read_text())
    if metadata['sha256'] != digest(path):
        raise ValueError('Dataset checksum mismatch')
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    validate(arrays)
    if arrays['features'].shape[1] != metadata['feature_count']:
        raise ValueError('Metadata feature count mismatch')
    return arrays, metadata


def merge_datasets(paths, output):
    loaded = [load_dataset(p) for p in paths]
    first = loaded[0][1]
    chunks, offset, used_seeds = [], 0, set()
    for arrays, metadata in loaded:
        for key in ('config', 'feature_indices', 'flybrain_version', 'device', 'numba_threads'):
            if metadata.get(key) != first.get(key):
                raise ValueError(f'Cannot aggregate incompatible {key}')
        seeds = set(map(int, np.unique(arrays['seed'])))
        if used_seeds & seeds:
            raise ValueError('Aggregation requires fresh episode seeds')
        used_seeds |= seeds
        arrays['episode_id'] = arrays['episode_id'] + offset
        offset = int(arrays['episode_id'].max()) + 1
        chunks.append(arrays)
    combined = {k: np.concatenate([a[k] for a in chunks]) for k in REQUIRED}
    return save_dataset(output, combined, {**first, 'collection_policy': 'aggregated',
                         'source_datasets': [str(p) for p in paths]})
