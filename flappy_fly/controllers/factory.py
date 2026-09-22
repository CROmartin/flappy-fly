from .human import HumanController
from .random_controller import RandomController
from .oracle import OracleController


def make_controller(mode, config, model=None, adapter=None):
    if mode == 'human':
        return HumanController()
    if mode == 'random':
        return RandomController(config)
    if mode == 'oracle':
        return OracleController(config)
    if mode == 'instinct':
        from .instinct import InstinctController
        return InstinctController(config, adapter=adapter)
    if mode == 'brain':
        from .trained import TrainedBrainController
        return TrainedBrainController(model, config, adapter=adapter)
    if mode == 'direct':
        from .trained import DirectInputBaseline
        return DirectInputBaseline(model, config)
    raise ValueError(f'Unknown controller: {mode}')
