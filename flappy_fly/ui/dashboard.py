"""Beginner-friendly side panel: what the fly sees, what fires, what that means."""
import pygame
from flappy_fly.brain.encoder import CHANNELS, SensoryEncoder
from .guide import CHANNEL_GUIDE, OUTPUT_GUIDE, narrate, mismatch_warning


class Dashboard:
    def __init__(self, config):
        self.config = config
        self.smoothed = {}
        self.encoder = SensoryEncoder(config.encoder)

    def reset(self):
        self.smoothed.clear()

    def _bar(self, screen, x, y, width, height, value, color):
        pygame.draw.rect(screen, (35, 51, 59), (x, y, width, height), border_radius=4)
        fill = int(width * max(0.0, min(1.0, value)))
        if fill:
            pygame.draw.rect(screen, color, (x, y, fill, height), border_radius=4)

    def draw(self, renderer, env, controller, action, history, mode, mean_frame_ms=0):
        x = self.config.game.width + 18
        width = self.config.ui.panel_width - 36
        s = env.state
        adapter = getattr(controller, 'adapter', None)
        values = adapter.inputs if adapter else self.encoder.values(s)
        probability = getattr(controller, 'flap_probability', None)
        threshold = self.config.control.decision_threshold
        y = 14

        renderer.text('HOW THIS WORKS', (x, y), (133, 229, 189))
        y += 22
        renderer.text('Fake eyes → frozen fly brain → flap / wait guess',
                      (x, y), (179, 197, 205), font=renderer.small)
        y += 16
        renderer.text('Wiring never learns. Only a tiny readout is trained.',
                      (x, y), (140, 160, 170), font=renderer.tiny)
        y += 22

        renderer.text('1. WHAT THE FLY "SEES"', (x, y), (133, 229, 189))
        y += 18
        renderer.text('Pipe colors: red=flap fatal, amber=wait fatal, green=hole.',
                      (x, y), (140, 160, 170), font=renderer.tiny)
        y += 16

        channel_colors = {
            'LC4': (232, 120, 103), 'LPLC2': (249, 193, 111),
            'LC10a-L': (117, 196, 255), 'LC10a-R': (117, 196, 255),
            'LPLC1-L': (242, 143, 183), 'LPLC1-R': (242, 143, 183),
        }
        for name in CHANNELS:
            guide = CHANNEL_GUIDE[name]
            value = float(values.get(name, 0.0))
            self.smoothed[name] = max(value, self.smoothed.get(name, 0) * self.config.ui.smoothing)
            count = len(adapter.groups[name]) if adapter else None
            suffix = f' · {count} cells' if count is not None else ''
            renderer.text(f'{guide["title"]}{suffix}', (x, y), font=renderer.small)
            renderer.text(f'{value:.0%}', (x + width - 36, y), font=renderer.small)
            y += 15
            self._bar(renderer.screen, x, y, width - 44, 8, self.smoothed[name], channel_colors[name])
            y += 10
            renderer.text(guide['role'], (x, y), (140, 160, 170), font=renderer.tiny)
            y += 15

        y += 4
        renderer.text('2. WHAT IS FIRING', (x, y), (133, 229, 189))
        y += 18
        if adapter:
            renderer.text(f'Whole brain: {adapter.brain.n:,} neurons (frozen connectome)',
                          (x, y), font=renderer.small)
            y += 16
            dn = OUTPUT_GUIDE['descending_neuron']
            renderer.text(f'{dn["title"]}: {adapter.feature_count}  mean {adapter.descending_activity:.3f}',
                          (x, y), font=renderer.small)
            y += 14
            renderer.text(dn['role'], (x, y), (140, 160, 170), font=renderer.tiny)
            y += 15
            giant = OUTPUT_GUIDE['DNp01']
            self._bar(renderer.screen, x, y + 1, width - 120, 9,
                      min(1.0, adapter.dnp01_activity / 6.0), (183, 154, 250))
            renderer.text(f'DNp01 {adapter.dnp01_activity:.2f}', (x + width - 110, y - 1), font=renderer.small)
            y += 14
            renderer.text(giant['role'], (x, y), (140, 160, 170), font=renderer.tiny)
            y += 15
            hottest = max(CHANNELS, key=lambda n: adapter.sensory_activity.get(n, 0.0))
            spike = adapter.sensory_activity.get(hottest, 0.0)
            renderer.text(f'Loudest eye group: {CHANNEL_GUIDE[hottest]["title"]} ({spike:.1f})',
                          (x, y), font=renderer.small)
            y += 18
        else:
            renderer.text('Brain offline here — press 4 or 5 to load it.', (x, y), font=renderer.small)
            y += 20

        renderer.text('3. DECISION', (x, y), (133, 229, 189))
        y += 18
        role = {
            'brain': 'Readout votes FLAP from all descending traces.',
            'instinct': 'Flaps when the giant DNp01 neuron is loud.',
            'oracle': 'Perfect teacher (cheats). Compare against this.',
            'direct': 'No fly brain — only wall/gap numbers.',
            'human': 'You press SPACE.',
            'random': 'Random flaps (dumb baseline).',
        }.get(mode, '')
        renderer.text(role, (x, y), (179, 197, 205), font=renderer.small)
        y += 18
        decision_color = (255, 198, 110) if action else (179, 197, 205)
        line = 'FLAP' if action else 'WAIT'
        if probability is not None:
            line += f'   P={probability:.0%} (need ≥ {threshold:.0%})'
        renderer.text(line, (x, y), decision_color)
        y += 20
        story = narrate(values, action, probability, mode, threshold)
        for start in range(0, len(story), 50):
            renderer.text(story[start:start + 50], (x, y), (221, 236, 229), font=renderer.small)
            y += 15
        tip = mismatch_warning(values, action)
        if tip:
            warn = tip.startswith('Mismatch')
            color = (255, 170, 130) if warn else (133, 229, 189)
            for start in range(0, len(tip), 50):
                renderer.text(tip[start:start + 50], (x, y), color, font=renderer.small)
                y += 15
        else:
            renderer.text('Too high brakes FLAP (fall). Too low pushes FLAP.',
                          (x, y), (140, 160, 170), font=renderer.tiny)
            y += 14

        y = max(y + 4, self.config.game.height - 36)
        renderer.text(f'Pipe {s.distance_to_pipe:.0f}px  err {s.bird_y - s.next_gap_center_y:+.0f}  '
                      f'frame {mean_frame_ms:.0f}ms',
                      (x, y), (140, 160, 170), font=renderer.tiny)
