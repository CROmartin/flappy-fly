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
        assert renderer.screen.get_size() == (1180, 600)
        assert tuple(renderer.screen.get_at((960, 10)))[:3] == (12, 23, 32)
    finally:
        renderer.close()


def test_play_loop_space_restart_and_escape(monkeypatch):
    import pygame
    from flappy_fly.ui.play import play
    from flappy_fly.game.renderer import Renderer
    seen = []
    seeds = []
    events = iter([[pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)],
                   [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)],
                   [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)]])
    monkeypatch.setattr(pygame.event, 'get', lambda: next(events))
    original = Renderer.draw
    def capture(self, env, *args, **kwargs):
        seen.append(env.state)
        seeds.append(kwargs.get('seed'))
        original(self, env, *args, **kwargs)
    monkeypatch.setattr(Renderer, 'draw', capture)
    play(max_steps=3, seed=42)
    assert seen[0].bird_velocity_y < 0
    assert seeds[0] == 42
    assert seen[1].elapsed_time == Config().physics.dt
    assert seen[1].bird_velocity_y > 0
    assert seeds[1] == 43  # R advances to a new episode seed
