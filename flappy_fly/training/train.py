"""Fit only a library logistic readout; the connectome is never loaded here."""
from pathlib import Path
import numpy as np
from .dataset import load_dataset, class_counts, provenance, write_json, digest
from .metrics import classification


def balanced_indices(labels, indices, rng):
    a = indices[labels[indices] == 0]
    b = indices[labels[indices] == 1]
    size = min(len(a), len(b))
    if size == 0:
        raise ValueError('Training needs both WAIT and FLAP examples')
    result = np.concatenate([rng.choice(a, size, replace=False), rng.choice(b, size, replace=False)])
    rng.shuffle(result)
    return result


def train(dataset, output, direct=False):
    from flybrain.reservoir import Readout
    arrays, metadata = load_dataset(dataset)
    settings = metadata['config']['training']
    labels = arrays['oracle_action']
    X = arrays['raw_inputs'] if direct else arrays['features']
    roles = arrays['split']
    if any(not np.any(roles == role) for role in ('train', 'validation', 'test')):
        raise ValueError('Training requires train, validation, and test episodes; aggregate DAgger with the original dataset')
    selected = balanced_indices(labels, np.flatnonzero(roles == 'train'), np.random.default_rng(settings['seed']))
    # Fold groups combine whole episodes, avoiding 35 redundant SVDs. Each episode
    # remains entirely in one fold; validation/test never enter this fit.
    episodes = np.unique(arrays['episode_id'][selected])
    if len(episodes) < 2:
        raise ValueError('Need at least two training episodes for grouped fitting')
    group_map = {int(e): i % min(5, len(episodes)) for i, e in enumerate(episodes)}
    groups = np.asarray([group_map[int(e)] for e in arrays['episode_id'][selected]])
    print('Natural class counts:', class_counts(labels), flush=True)
    print('Balanced fitting subset:', class_counts(labels[selected]), flush=True)
    # Fixed regularization/rank are predeclared; CV is diagnostic, not test tuning.
    model = Readout.fit(X[selected], labels[selected], kind='logistic', groups=groups,
                        components=(None,) if direct else (settings['components'],),
                        lambdas=(settings['regularization'],), verbose=True)
    threshold = metadata['config']['control']['decision_threshold']
    metrics = {role: classification(labels[roles == role], model.predict(X[roles == role]), threshold)
               for role in ('validation', 'test')}
    output = Path(output)
    if output.suffix != '.npz':
        raise ValueError('Model output must end in .npz')
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    model_metadata = {**provenance(), 'schema_version': 1, 'model_kind': 'direct' if direct else 'brain',
        'flybrain_version': metadata['flybrain_version'], 'config': metadata['config'],
        'feature_count': X.shape[1], 'feature_indices': [] if direct else metadata['feature_indices'],
        'device': metadata['device'], 'training_dataset': str(dataset), 'dataset_sha256': metadata['sha256'],
        'dataset_seeds': metadata['seeds'], 'training_samples': len(selected),
        'class_counts': class_counts(labels), 'balanced_class_counts': class_counts(labels[selected]),
        'split_seeds': {role: np.unique(arrays['seed'][roles == role]).tolist() for role in ('train', 'validation', 'test')},
        'metrics': metrics, 'cv_auc_balanced': float(model.cv_score) if np.isfinite(model.cv_score) else None,
        'sha256': digest(output)}
    write_json(output.with_suffix('.json'), model_metadata)
    print(metrics, flush=True)
    return model_metadata
