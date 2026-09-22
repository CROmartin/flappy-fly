from flappy_fly.config import Config
from flappy_fly.game.env import FlappyEnv
from flappy_fly.controllers.factory import make_controller


def play(mode='human', seed=42, config=None, model=None, max_steps=None, headless=False, monitor_brain=False):
    config = config or Config()
    env = FlappyEnv(seed, config)
    controller = make_controller(mode, config, model)
    controller.reset(seed)
    if headless:
        if mode == 'human':
            raise ValueError('Human play requires a window')
        for _ in range(max_steps or config.training.max_steps):
            state, _, done, _ = env.step(controller.act(env.state))
            if done:
                break
        print(f'{mode}: score={state.score} survival={state.elapsed_time:.2f}s')
        return
    import pygame
    from flappy_fly.game.renderer import Renderer
    renderer = Renderer(config)
    clock = pygame.time.Clock()
    best = 0
    running = True
    frames = 0
    action = 0
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and mode == 'human':
                        controller.flap()
                    if event.key == pygame.K_r:
                        env.reset(seed)
                        controller.reset(seed)
            if env.alive and running:
                action = controller.act(env.state)
                env.step(action)
                best = max(best, env.score)
            renderer.draw(env, mode, best, clock.get_fps(), action, controller)
            frames += 1
            if max_steps and frames >= max_steps:
                running = False
            clock.tick(round(1 / config.physics.dt))
    finally:
        renderer.close()
