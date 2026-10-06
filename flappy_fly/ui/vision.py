"""Draw what the fake eyes care about onto the game view. UI-only."""
import pygame
from flappy_fly.brain.encoder import SensoryEncoder


def draw_fly_vision(renderer, env, controller):
    """Highlight the next pipe: red upper threat, amber lower threat, green safe gap."""
    g = renderer.config.game
    s = env.state
    encoder = SensoryEncoder(renderer.config.encoder)
    adapter = getattr(controller, 'adapter', None)
    if adapter:
        top_threat = adapter.inputs.get('LC4', 0.0)
        bottom_threat = adapter.inputs.get('LPLC2', 0.0)
    else:
        top_threat, bottom_threat = encoder.threats(s)

    # Next relevant pipe: first pipe whose trailing edge is still ahead of the fly.
    pipes = [p for p in env.pipes if p.x + g.pipe_width >= g.fly_x - 4]
    if not pipes:
        return
    pipe = min(pipes, key=lambda p: p.x)
    top = pipe.gap_center - g.gap_size / 2
    bottom = pipe.gap_center + g.gap_size / 2

    overlay = pygame.Surface((g.width, g.height), pygame.SRCALPHA)
    # Upper solid: red intensity = top threat
    if top > 0:
        alpha = int(40 + 140 * max(0.0, min(1.0, top_threat)))
        pygame.draw.rect(overlay, (232, 120, 103, alpha), (pipe.x, 0, g.pipe_width, top))
    # Lower solid: amber intensity = bottom threat
    lower_h = g.height - g.ground_height - bottom
    if lower_h > 0:
        alpha = int(40 + 140 * max(0.0, min(1.0, bottom_threat)))
        pygame.draw.rect(overlay, (249, 193, 111, alpha), (pipe.x, bottom, g.pipe_width, lower_h))
    # Safe opening
    pygame.draw.rect(overlay, (102, 222, 171, 55), (pipe.x, top, g.pipe_width, bottom - top))
    pygame.draw.rect(overlay, (102, 222, 171, 200), (pipe.x, top, g.pipe_width, bottom - top), width=2)
    renderer.screen.blit(overlay, (0, 0))

    # Aim line from fly to gap center
    fx, fy = int(g.fly_x), int(s.bird_y)
    gx = int(pipe.x + g.pipe_width / 2)
    gy = int(pipe.gap_center)
    pygame.draw.line(renderer.screen, (156, 223, 169), (fx + 12, fy), (gx, gy), 1)
    pygame.draw.circle(renderer.screen, (156, 223, 169), (gx, gy), 4, width=1)

    # Tiny legend on the playfield
    renderer.text('red=flap fatal  amber=wait fatal  green=hole', (24, g.height - 52),
                  (140, 160, 170), font=renderer.tiny)
