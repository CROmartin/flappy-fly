from time import perf_counter
from flappy_fly.config import Config
from flappy_fly.game.env import FlappyEnv
from flappy_fly.controllers.factory import make_controller
from flappy_fly.controllers.oracle import OracleController
from .history import DecisionHistory


def play(mode='human', seed=42, config=None, model=None, max_steps=None, headless=False, monitor_brain=False):
    config = config or Config()
    env = FlappyEnv(seed, config)
    controller = make_controller(mode, config, model)
    controller.reset(seed)
    monitor = None
    if monitor_brain and not hasattr(controller, 'adapter'):
        from flappy_fly.brain.adapter import BrainAdapter
        monitor = BrainAdapter(config)
        monitor.reset(seed)
        controller.adapter = monitor
    if headless:
        if mode == 'human':
            raise ValueError('Human play requires a window')
        for _ in range(max_steps or config.training.max_steps):
            state = env.state
            action = controller.act(state)
            if monitor:
                monitor.advance(state)
            state, _, done, _ = env.step(action)
            if done:
                break
        print(f'{mode}: score={state.score} survival={state.elapsed_time:.2f}s')
        return
    import pygame
    from flappy_fly.game.renderer import Renderer
    renderer = Renderer(config)
    clock = pygame.time.Clock()
    history = DecisionHistory(config)
    teacher = OracleController(config)
    teacher.reset(seed)
    best = {}
    running = True
    frames = action = 0
    frame_seconds = 0.0
    notice = ''
    mode_keys = {pygame.K_1: 'human', pygame.K_2: 'random', pygame.K_3: 'oracle',
                 pygame.K_4: 'instinct', pygame.K_5: 'brain'}

    def reset_episode():
        env.reset(seed)
        controller.reset(seed)
        teacher.reset(seed)
        if monitor:
            monitor.reset(seed)
        history.clear()
        renderer.dashboard.reset()

    try:
        while running:
            start = perf_counter()
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and mode == 'human':
                        controller.flap()
                    if event.key == pygame.K_r:
                        reset_episode()
                        action = 0
                    if event.key in mode_keys:
                        requested = mode_keys[event.key]
                        try:
                            new = make_controller(requested, config, model, getattr(controller, 'adapter', None))
                        except (ValueError, FileNotFoundError) as error:
                            notice = str(error)
                        else:
                            controller, mode = new, requested
                            monitor = None
                            notice = ''
                            if monitor_brain and not hasattr(controller, 'adapter'):
                                from flappy_fly.brain.adapter import BrainAdapter
                                monitor = BrainAdapter(config)
                                controller.adapter = monitor
                            reset_episode()
                            action = 0
            if env.alive and running:
                state = env.state
                label = teacher.query(state)
                action = controller.act(state)
                if monitor:
                    monitor.advance(state)
                teacher.observe_action(state, action)
                history.append(state, controller, action, label)
                _, _, done, _ = env.step(action)
                best[mode] = max(best.get(mode, 0), env.score)
                if done:
                    print(f'Death history saved: {history.save(mode, seed, model)}', flush=True)
            renderer.draw(env, mode, best.get(mode, 0), clock.get_fps(), action, controller,
                          history.rows, notice=notice, mean_frame_ms=1000 * frame_seconds / max(1, frames))
            frame_seconds += perf_counter() - start
            frames += 1
            if max_steps and frames >= max_steps:
                running = False
            # Fixed simulation dt even if wall time is slower. Never skip neural decisions.
            clock.tick(round(1 / config.physics.dt))
    finally:
        renderer.close()
