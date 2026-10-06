"""Retune flap threshold/cooldown on an existing readout (actuator rules only)."""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flappy_fly.config import Config
from flappy_fly.training.dataset import load_dataset, write_json
from flappy_fly.training.evaluate import evaluate
from flappy_fly.training.metrics import classification


def offline_sweep(model_path, dataset_path, thresholds):
    from flybrain.reservoir import Readout
    arrays, _ = load_dataset(dataset_path)
    model = Readout.load(model_path)
    mask = arrays['split'] == 'validation'
    labels = arrays['oracle_action'][mask]
    probs = model.predict(arrays['features'][mask])
    rows = [classification(labels, probs, float(t)) for t in thresholds]
    best = max(rows, key=lambda row: row['f1_flap'])
    return rows, best


def write_config(base, threshold, cooldown, path):
    cfg = deepcopy(base)
    cfg['control']['decision_threshold'] = float(threshold)
    cfg['control']['flap_cooldown'] = float(cooldown)
    Path(path).write_text(json.dumps(cfg, indent=2) + '\n')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Tune flap threshold on a frozen readout')
    parser.add_argument('--model', default='models/flap_readout_v1.npz')
    parser.add_argument('--dataset', default='data/datasets/oracle_v1.npz')
    parser.add_argument('--base-config', default='configs/baseline_v1.json')
    parser.add_argument('--cooldown', type=float, default=0.12)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--max-steps', type=int, default=750)
    parser.add_argument('--output-config', default='configs/tuned_threshold_v1.json')
    parser.add_argument('--output-report', default='data/runs/threshold_tune_v1.json')
    parser.add_argument('--skip-closed-loop', action='store_true')
    args = parser.parse_args(argv)

    thresholds = np.round(np.arange(0.50, 0.96, 0.05), 2)
    offline, best_offline = offline_sweep(args.model, args.dataset, thresholds)
    print('Offline validation (threshold → F1 / precision / recall):', flush=True)
    for row in offline:
        print(f"  {row['threshold']:.2f}  F1={row['f1_flap']:.3f}  "
              f"P={row['precision_flap']:.3f}  R={row['recall_flap']:.3f}", flush=True)
    print(f"Best offline F1 at threshold={best_offline['threshold']:.2f}", flush=True)

    base = json.loads(Path(args.base_config).read_text())
    closed_loop = []
    if not args.skip_closed_loop:
        # Shortlist: baseline, best offline F1, and a couple of higher precision picks.
        candidates = sorted({0.5, float(best_offline['threshold']), 0.70, 0.80, 0.90})
        print(f'Closed-loop brain eval on thresholds {candidates}...', flush=True)
        for threshold in candidates:
            cfg_path = Path('data/runs') / f'_tmp_threshold_{threshold:.2f}.json'
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            write_config(base, threshold, args.cooldown, cfg_path)
            report = evaluate(
                model=args.model, episodes=args.episodes, seed=args.seed,
                config=Config.load(cfg_path), max_steps=args.max_steps, modes=['brain'],
                output=str(Path('data/runs') / f'_tmp_eval_threshold_{threshold:.2f}.json'),
            )
            summary = report['results']['brain']['summary']
            closed_loop.append({'threshold': threshold, 'cooldown': args.cooldown, **summary})
            print(f"  thr={threshold:.2f} mean_pipes={summary['mean_score']:.2f} "
                  f"survival={summary['mean_survival_time']:.2f}s", flush=True)
            cfg_path.unlink(missing_ok=True)
        winner = max(closed_loop, key=lambda row: (row['mean_score'], row['mean_survival_time']))
    else:
        winner = {'threshold': float(best_offline['threshold']), 'cooldown': args.cooldown}

    write_config(base, winner['threshold'], winner.get('cooldown', args.cooldown), args.output_config)
    write_json(args.output_report, {
        'model': args.model, 'dataset': args.dataset, 'cooldown': args.cooldown,
        'offline_validation': offline, 'best_offline': best_offline,
        'closed_loop': closed_loop, 'selected': winner, 'config': args.output_config,
    })
    print(f"Wrote {args.output_config} with threshold={winner['threshold']:.2f} "
          f"cooldown={winner.get('cooldown', args.cooldown):.2f}", flush=True)


if __name__ == '__main__':
    main()
