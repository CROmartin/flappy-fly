"""All heavy imports are lazy: ordinary gameplay never loads a connectome."""
import argparse
from flappy_fly.config import Config


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description='Flappy Fly: frozen real connectome, trained output readout')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('inspect-brain')
    p = sub.add_parser('play')
    p.add_argument('--mode', choices=['human', 'random', 'oracle', 'instinct', 'brain', 'direct'], default='human')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model')
    p.add_argument('--config')
    p.add_argument('--max-steps', type=positive)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--monitor-brain', action='store_true', help='Run brain telemetry while human/oracle/random controls')
    p = sub.add_parser('collect')
    p.add_argument('--episodes', type=positive, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', default='data/datasets/oracle_v1.npz')
    p.add_argument('--config')
    p.add_argument('--max-steps', type=positive)
    p.add_argument('--model', help='DAgger: learner controls, oracle labels; all new episodes are training-only')
    p.add_argument('--headless', action='store_true', help='Collection is always headless')
    p = sub.add_parser('train')
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', default='models/flap_readout_v1.npz')
    p.add_argument('--direct', action='store_true', help='Fit the three-input logistic baseline')
    p = sub.add_parser('merge')
    p.add_argument('--datasets', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('evaluate')
    p.add_argument('--model')
    p.add_argument('--direct-model')
    p.add_argument('--episodes', type=positive, default=30)
    p.add_argument('--seed', type=int, default=10000)
    p.add_argument('--config')
    p.add_argument('--max-steps', type=positive)
    p.add_argument('--modes', nargs='+', choices=['random', 'oracle', 'instinct', 'brain', 'direct'])
    p.add_argument('--output', default='data/runs/evaluation.json')
    p.add_argument('--headless', action='store_true', help='Evaluation is always headless')
    args = parser.parse_args(argv)
    try:
        if args.command == 'inspect-brain':
            import runpy
            from pathlib import Path
            runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/inspect_brain.py'), run_name='__main__')
        elif args.command == 'play':
            from flappy_fly.ui.play import play
            play(args.mode, args.seed, Config.load(args.config), args.model, args.max_steps, args.headless, args.monitor_brain)
        elif args.command == 'collect':
            from flappy_fly.training.collector import collect
            collect(args.output, args.episodes, args.seed, Config.load(args.config), args.max_steps, args.model)
        elif args.command == 'train':
            from flappy_fly.training.train import train
            train(args.dataset, args.output, args.direct)
        elif args.command == 'merge':
            from flappy_fly.training.dataset import merge_datasets
            merge_datasets(args.datasets, args.output)
        elif args.command == 'evaluate':
            from flappy_fly.training.evaluate import evaluate
            evaluate(args.model, args.direct_model, args.episodes, args.seed, Config.load(args.config), args.max_steps, args.modes, args.output)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
