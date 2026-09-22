from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from flappy_fly.brain.encoder import SensoryEncoder
from flappy_fly.training.dataset import write_json, provenance


class DecisionHistory:
    def __init__(self, config):
        self.config = config
        self.rows = deque(maxlen=config.ui.history_steps)
        self.encoder = SensoryEncoder(config.encoder)

    def clear(self):
        self.rows.clear()

    def append(self, state, controller, action, oracle_action):
        adapter = getattr(controller, 'adapter', None)
        self.rows.append({'time': state.elapsed_time, 'vertical_error': state.bird_y - state.next_gap_center_y,
            'velocity': state.bird_velocity_y, 'distance': state.distance_to_pipe,
            'looming': float(self.encoder.raw(state)[0]),
            'dnp01': adapter.dnp01_activity if adapter else None,
            'descending': adapter.descending_activity if adapter else None,
            'flap_probability': getattr(controller, 'flap_probability', None),
            'action': action, 'oracle_action': oracle_action})

    def save(self, mode, seed, model):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        path = Path('data/runs') / f'death_{mode}_{seed}_{stamp}.json'
        write_json(path, {**provenance(), 'mode': mode, 'seed': seed, 'model': str(model) if model else None,
                          'config': self.config.to_dict(), 'history': list(self.rows)})
        return path
