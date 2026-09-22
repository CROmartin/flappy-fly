from time import perf_counter
from flappy_fly.config import Config
from flappy_fly.controllers.factory import make_controller
from flappy_fly.game.env import FlappyEnv
from .dataset import write_json, provenance
from .metrics import episode_metrics


def evaluate(model=None, direct_model=None, episodes=30, seed=10000, config=None,
             max_steps=None, modes=None, output='data/runs/evaluation.json'):
    config = config or Config()
    if episodes <= 0:
        raise ValueError('episodes must be positive')
    max_steps = config.training.max_steps if max_steps is None else max_steps
    if max_steps <= 0:
        raise ValueError('max_steps must be positive')
    modes = modes or (['random', 'oracle', 'instinct', 'brain'] + (['direct'] if direct_model else []))
    seeds = list(range(seed, seed + episodes))
    from flappy_fly.controllers.trained import load_model
    from flappy_fly.training.dataset import digest
    model_hashes = {}
    for mode, path in (('brain', model), ('direct', direct_model)):
        if mode in modes:
            _, metadata = load_model(path, mode, config)
            overlap = set(seeds) & set(metadata['dataset_seeds'])
            if overlap:
                raise ValueError(f'Evaluation seeds overlap dataset seeds: {sorted(overlap)}')
            model_hashes[mode] = digest(path)
    results = {}
    adapter = None
    print(f'{"Controller":<16} {"Mean":>8} {"Median":>8} {"Max":>6} {"Survival":>10}', flush=True)
    for mode in modes:
        controller = make_controller(mode, config, direct_model if mode == 'direct' else model, adapter)
        if hasattr(controller, 'metadata'):
            overlap = set(seeds) & set(controller.metadata['dataset_seeds'])
            if overlap:
                raise ValueError(f'Evaluation seeds overlap dataset seeds: {sorted(overlap)}')
        if hasattr(controller, 'adapter'):
            adapter = controller.adapter  # Reuse graph; reset public dynamic state for each episode.
        rows = []
        start = perf_counter()
        neural_seconds = neural_steps = 0
        for episode_seed in seeds:
            env = FlappyEnv(episode_seed, config)
            controller.reset(episode_seed)
            for _ in range(max_steps):
                _, _, done, _ = env.step(controller.act(env.state))
                if done:
                    break
            rows.append({'seed': episode_seed, 'score': env.score, 'survival_time': env.state.elapsed_time,
                         'truncated': env.alive})
            if hasattr(controller, 'adapter'):
                neural_seconds += controller.adapter.total_step_seconds
                neural_steps += controller.adapter.neural_steps
        summary = episode_metrics(rows)
        summary['wall_seconds'] = perf_counter() - start
        summary['mean_brain_step_ms'] = 1000 * neural_seconds / neural_steps if neural_steps else None
        results[mode] = {'summary': summary, 'episodes': rows}
        print(f'{mode:<16} {summary["mean_score"]:8.2f} {summary["median_score"]:8.1f} '
              f'{summary["max_score"]:6} {summary["mean_survival_time"]:9.2f}s', flush=True)
    report = {**provenance(), 'config': config.to_dict(), 'seeds': seeds, 'max_steps': max_steps,
              'model': str(model) if model else None, 'direct_model': str(direct_model) if direct_model else None,
              'results': results, 'model_sha256': model_hashes, 'note': 'Survival and score are capped for episodes marked truncated.'}
    write_json(output, report)
    return report
