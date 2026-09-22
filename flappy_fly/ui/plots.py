"""UI-only trace scaling; never feeds values back into the brain."""
import pygame


def death_analysis(renderer, history):
    rows = list(history)
    if not rows:
        return
    last = rows[-1]
    x, y = 56, 366
    renderer.text(f'Error {last["vertical_error"]:+.0f}px  |  Velocity {last["velocity"]:+.0f}px/s  |  Pipe {last["distance"]:.0f}px',
                  (x, y), font=renderer.small)
    probability = last['flap_probability']
    dn = last['dnp01']
    renderer.text(f'Last P(FLAP): {probability:.0%}' if probability is not None else 'Last P(FLAP): --', (x, y + 23), font=renderer.small)
    renderer.text(f'DNp01 trace: {dn:.2f}' if dn is not None else 'DNp01 trace: --', (x + 210, y + 23), font=renderer.small)
    chart = pygame.Rect(x, y + 56, 595, 62)
    pygame.draw.rect(renderer.screen, (24, 38, 46), chart, border_radius=5)
    series = [('flap_probability', 1, (249, 193, 111)), ('looming', 1, (100, 220, 180)), ('dnp01', 6, (183, 154, 250)),
              ('vertical_error', 320, (117, 196, 255)), ('velocity', 480, (242, 143, 183))]
    for key, scale, color in series:
        points = [(chart.x + int(i * chart.width / max(1, len(rows) - 1)),
                   chart.bottom - 2 - int(min(1, max(0, (0.5 + r[key] / (2 * scale)) if key in ('vertical_error', 'velocity') else r[key] / scale)) * (chart.height - 4)))
                  for i, r in enumerate(rows) if r[key] is not None]
        if len(points) > 1:
            pygame.draw.lines(renderer.screen, color, False, points, 2)
    for i, row in enumerate(rows):
        px = chart.x + int(i * chart.width / max(1, len(rows) - 1))
        if row['action']:
            pygame.draw.line(renderer.screen, (249, 193, 111), (px, chart.bottom + 3), (px, chart.bottom + 9), 2)
        if row['oracle_action']:
            pygame.draw.line(renderer.screen, (210, 230, 235), (px, chart.bottom + 12), (px, chart.bottom + 18), 2)
    renderer.text(f'Last {len(rows)} steps: gold P/flap | green looming | violet DN | white teacher',
                  (x, chart.bottom + 22), font=renderer.small)
    renderer.text('Blue gap error | pink velocity (signed; zero at plot midpoint)',
                  (x, chart.bottom + 40), font=renderer.small)
