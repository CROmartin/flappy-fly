import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flappy_fly.cli import main

if __name__ == '__main__':
    main(['evaluate', *sys.argv[1:]])
