import argparse
from flappy_fly.config import Config


def main(argv=None):
    parser = argparse.ArgumentParser(description='Flappy Fly: a frozen real connectome and a trained readout')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('inspect-brain')
    p = sub.add_parser('play')
    p.add_argument('--mode', choices=['human', 'random', 'oracle', 'instinct', 'brain', 'direct'], default='human')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model')
    p.add_argument('--config')
    p.add_argument('--max-steps', type=int)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--monitor-brain', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'inspect-brain':
        import runpy
        from pathlib import Path
        runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/inspect_brain.py'), run_name='__main__')
    elif args.command == 'play':
        from flappy_fly.ui.play import play
        play(args.mode, args.seed, Config.load(args.config), args.model, args.max_steps, args.headless, args.monitor_brain)


if __name__ == '__main__':
    main()
