from dataclasses import astuple
import numpy as np
from flappy_fly.config import Config
from flappy_fly.game.env import FlappyEnv
from flappy_fly.controllers.oracle import OracleController
from flappy_fly.brain.adapter import BrainAdapter
from flappy_fly.brain.encoder import CHANNELS
from .dataset import REQUIRED, episode_splits, save_dataset


def collect(output, episodes=50, seed=0, config=None, max_steps=None, model=None):
    config = config or Config()
    max_steps = max_steps or config.training.max_steps
    if episodes <= 0 or max_steps <= 0:
        raise ValueError('episodes and max_steps must be positive')
    roles = ['train'] * episodes if model else episode_splits(episodes)
    adapter = BrainAdapter(config)
    teacher = OracleController(config)
    learner = None
    if model:
        from flappy_fly.controllers.trained import TrainedBrainController
        learner = TrainedBrainController(model, config, adapter=adapter)
        if set(range(seed, seed + episodes)) & set(learner.metadata['dataset_seeds']):
            raise ValueError('DAgger collection requires new seeds')
    rows = {key: [] for key in REQUIRED}
    results = []
    for episode in range(episodes):
        episode_seed = seed + episode
        env = FlappyEnv(episode_seed, config)
        teacher.reset(episode_seed)
        if learner:
            learner.reset(episode_seed)
        else:
            adapter.reset(episode_seed)
        for step in range(max_steps):
            state = env.state
            label = teacher.query(state)
            if learner:
                action = learner.act(state)  # Exactly one advance; reuse its features below.
                features = adapter.features.copy()
            else:
                features = adapter.advance(state)
                action = label
            teacher.observe_action(state, action)
            values = (features, adapter.encoder.raw(state), astuple(state),
                      [adapter.inputs[name] for name in CHANNELS], label, action,
                      episode, step, episode_seed, state.score, roles[episode], adapter.dnp01_activity)
            for key, value in zip(REQUIRED, values):
                rows[key].append(value)
            _, _, done, _ = env.step(action)
            if done:
                break
        results.append({'seed': episode_seed, 'score': env.score, 'steps': env.steps, 'truncated': env.alive})
        print(f'episode {episode + 1}/{episodes} seed={episode_seed} score={env.score} '
              f'samples={len(rows["oracle_action"])} brain={adapter.mean_step_ms:.2f}ms', flush=True)
    arrays = {}
    for key, values in rows.items():
        dtype = 'U10' if key == 'split' else (np.float32 if key in ('features', 'raw_inputs', 'encoder_values', 'dnp01')
                                             else np.float64 if key == 'states' else np.int64)
        arrays[key] = np.asarray(values, dtype=dtype)
    metadata = save_dataset(output, arrays, {**adapter.metadata(), 'config': config.to_dict(),
        'encoder_channels': CHANNELS, 'collection_policy': 'dagger' if learner else 'oracle',
        'learner_model': str(model) if model else None, 'max_steps': max_steps, 'episode_results': results})
    print(f'Saved {output}: {metadata["class_counts"]}', flush=True)
    return metadata
