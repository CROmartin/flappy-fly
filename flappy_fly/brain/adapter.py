"""Public flybrain API only. Biological weights are never assigned or trained."""
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from flappy_fly.config import Config
from .encoder import SensoryEncoder, CHANNELS
from .neuron_groups import resolve_groups


class BrainAdapter:
    def __init__(self, config=None, data=None):
        self.config = config or Config()
        c = self.config.brain
        from flybrain import FlyBrain
        from flybrain.reservoir import Trace
        import numba
        numba.set_num_threads(c.threads)
        data = data or os.environ.get('FLY_DATA', str(Path(__file__).resolve().parents[2] / 'data/brain'))
        self.brain = FlyBrain(data=data, device=c.device, dt=c.dt, sensory_input=True)
        self.groups = resolve_groups(self.brain)
        self.encoder = SensoryEncoder(self.config.encoder)
        self.trace = Trace(self.brain, idx=self.groups['descending_neuron'], tau=c.trace_tau)
        self.dnp01_trace = Trace(self.brain, idx=self.groups['DNp01'], tau=c.trace_tau)
        self.input_traces = {name: Trace(self.brain, idx=self.groups[name], tau=c.trace_tau) for name in CHANNELS}
        self.feature_indices = self.groups['descending_neuron'].copy()
        self.feature_count = len(self.feature_indices)
        self.reset()

    def reset(self, seed=42):
        self.brain.reset(seed=seed)
        self.trace.reset()
        self.dnp01_trace.reset()
        for trace in self.input_traces.values():
            trace.reset()
        self.features = self.trace.features()
        self.inputs = dict.fromkeys(CHANNELS, 0.0)
        self.sensory_activity = dict.fromkeys(CHANNELS, 0.0)
        self.dnp01_activity = self.descending_activity = 0.0
        self.total_step_seconds = 0.0
        self.neural_steps = 0

    def advance(self, state):
        self.inputs = self.encoder.values(state)
        inject = [(self.groups[name], value) for name, value in self.inputs.items()]
        start = perf_counter()
        for _ in range(self.config.brain.brain_steps_per_action):
            fired = self.brain.step(inject=inject)
            self.features = self.trace.observe(fired)
            self.dnp01_activity = float(self.dnp01_trace.observe(fired).mean())
            for name, trace in self.input_traces.items():
                self.sensory_activity[name] = float(trace.observe(fired).mean())
            self.neural_steps += 1
        self.total_step_seconds += perf_counter() - start
        self.descending_activity = float(self.features.mean())
        return self.features.copy()

    @property
    def mean_step_ms(self):
        return 1000 * self.total_step_seconds / max(1, self.neural_steps)

    def metadata(self):
        import flybrain
        return {'flybrain_version': flybrain.__version__, 'device': self.brain.device,
                'neurons': self.brain.n, 'connections': len(self.brain.weights),
                'feature_count': self.feature_count, 'feature_indices': self.feature_indices.tolist(),
                'mean_brain_step_ms': self.mean_step_ms, 'brain_seed_policy': 'episode game seed',
                'reset_policy': 'public FlyBrain.reset(seed) and Trace.reset; no washout',
                'numba_threads': self.config.brain.threads}
