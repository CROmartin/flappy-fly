import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
from flappy_fly.config import Config
from flappy_fly.game.env import FlappyEnv
from flappy_fly.controllers.human import HumanController
from flappy_fly.ui.history import DecisionHistory


def test_human_input_and_death_render():
    from flappy_fly.game.renderer import Renderer
    env = FlappyEnv()
    controller = HumanController()
    controller.flap()
    assert controller.act(env.state) == 1
    assert controller.act(env.state) == 0
    renderer = Renderer(Config())
    history = DecisionHistory(Config())
    try:
        for _ in range(100):
            history.append(env.state, controller, 0, 0)
            env.step(0)
            if not env.alive:
                break
        renderer.draw(env, 'human', 0, 50, 0, controller, history.rows)
        assert renderer.screen.get_size() == (1120, 600)
    finally:
        renderer.close()
