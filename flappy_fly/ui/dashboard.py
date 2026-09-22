from flappy_fly.brain.encoder import CHANNELS


class Dashboard:
    def __init__(self, config):
        self.config = config
        self.smoothed = {}

    def reset(self):
        self.smoothed.clear()

    def draw(self, renderer, env, controller, action, history, mode):
        import pygame
        x = self.config.game.width + 24
        s = env.state
        adapter = getattr(controller, 'adapter', None)
        renderer.text('CONNECTOME ONLINE' if adapter else 'CONNECTOME OFFLINE', (x, 28), (133, 229, 189))
        renderer.text(f'{adapter.brain.n:,} simulated neurons' if adapter else 'Human / conventional controller', (x, 59), font=renderer.small)
        role = {'brain': 'READOUT CONTROLS FLAPS', 'instinct': 'DNp01 REFLEX CONTROLS FLAPS'}.get(mode, 'TELEMETRY ONLY' if adapter else 'NO BRAIN REQUIRED')
        renderer.text(role, (x, 80), (179, 197, 205), font=renderer.small)
        renderer.text('SENSORY INJECTION  /  SPIKE TRACE', (x, 103), font=renderer.small)
        for i, name in enumerate(CHANNELS):
            y = 137 + i * 31
            value = adapter.inputs[name] if adapter else 0
            self.smoothed[name] = max(value, self.smoothed.get(name, 0) * self.config.ui.smoothing)
            renderer.text(name, (x, y), font=renderer.small)
            pygame.draw.rect(renderer.screen, (35, 51, 59), (x + 92, y, 155, 13), border_radius=5)
            pygame.draw.rect(renderer.screen, (102, 222, 171), (x + 92, y, int(155 * self.smoothed[name]), 13), border_radius=5)
            actual = adapter.sensory_activity[name] if adapter else 0
            renderer.text(f'{value:.2f} / {actual:.1f}', (x + 258, y - 1), font=renderer.small)
        renderer.text('DESCENDING OUTPUT', (x, 334), (133, 229, 189))
        renderer.text(f'DNp01 trace     {adapter.dnp01_activity:.2f}' if adapter else 'DNp01 trace     --', (x, 365))
        renderer.text(f'DN mean         {adapter.descending_activity:.3f}' if adapter else 'DN mean         --', (x, 393))
        probability = getattr(controller, 'flap_probability', None)
        renderer.text(f'FLAP probability   {probability:.0%}' if probability is not None else 'FLAP probability   --', (x, 427))
        renderer.text('FLAP' if action else 'WAIT', (x + 240, 460), (255, 198, 110))
        renderer.text(f'Distance {s.distance_to_pipe:+.0f}px', (x, 471), font=renderer.small)
        renderer.text(f'Error {s.bird_y - s.next_gap_center_y:+.0f}px    Velocity {s.bird_velocity_y:+.0f}', (x, 495), font=renderer.small)
        renderer.text(f'Brain step {adapter.mean_step_ms:.1f} ms' if adapter else 'No neural simulation running', (x, 525), font=renderer.small)
        renderer.text('Frozen wiring. Trained output only.', (x, 565), font=renderer.small)
