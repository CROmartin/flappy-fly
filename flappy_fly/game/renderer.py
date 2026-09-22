"""Original vector-like shapes drawn with Pygame; no external artwork."""
import math
import pygame


class Renderer:
    def __init__(self, config):
        self.config = config
        pygame.init()
        g = config.game
        self.screen = pygame.display.set_mode((g.width + config.ui.panel_width, g.height))
        pygame.display.set_caption('Flappy Fly | Frozen connectome, trained readout')
        self.font = pygame.font.Font(None, 25)
        self.small = pygame.font.Font(None, 20)
        self.title = pygame.font.Font(None, 48)
        from flappy_fly.ui.dashboard import Dashboard
        self.dashboard = Dashboard(config)

    def text(self, text, position, color=(221, 236, 229), font=None):
        self.screen.blit((font or self.font).render(str(text), True, color), position)

    def draw(self, env, mode, best, fps, action, controller, history=(), notice='', mean_frame_ms=0):
        g = self.config.game
        s = env.state
        self.screen.fill((12, 23, 32))
        self.screen.set_clip(pygame.Rect(0, 0, g.width, g.height))
        pygame.draw.rect(self.screen, (19, 42, 49), (0, 0, g.width, g.height))
        for i in range(12):
            x = int((i * 83 - s.elapsed_time * 10) % g.width)
            pygame.draw.circle(self.screen, (30, 62, 65), (x, 140 + i * 37 % 330), 3)
        for pipe in env.pipes:
            top = pipe.gap_center - g.gap_size / 2
            bottom = pipe.gap_center + g.gap_size / 2
            for rect in ((pipe.x, 0, g.pipe_width, top), (pipe.x, bottom, g.pipe_width, g.height - bottom)):
                pygame.draw.rect(self.screen, (61, 137, 107), rect, border_radius=7)
                pygame.draw.rect(self.screen, (100, 196, 140), rect, width=2, border_radius=7)
        pygame.draw.rect(self.screen, (29, 62, 52), (0, g.height - g.ground_height, g.width, g.ground_height))
        x, y = int(g.fly_x), int(s.bird_y)
        if action and s.alive:
            for i in range(5):
                pygame.draw.circle(self.screen, (160, 231, 187), (x - 24 - i * 7, y + 9 + i * 3), max(1, 4 - i))
        wing = int(5 * math.sin(s.elapsed_time * 75))
        pygame.draw.ellipse(self.screen, (188, 231, 228), (x - 18, y - 23 - wing, 25, 16))
        pygame.draw.ellipse(self.screen, (159, 210, 214), (x - 4, y - 26 + wing, 26, 15))
        pygame.draw.ellipse(self.screen, (46, 47, 56), (x - 17, y - 10, 30, 23))
        for dy in (-3, 4):
            pygame.draw.line(self.screen, (115, 110, 105), (x - 12, y + dy), (x - 4, y + dy), 2)
        pygame.draw.circle(self.screen, (232, 120, 103), (x + 10, y - 4), 10)
        pygame.draw.circle(self.screen, (255, 238, 189), (x + 14, y - 6), 3)
        for dx in (-9, -1, 7):
            pygame.draw.line(self.screen, (25, 28, 37), (x + dx, y + 8), (x + dx - 5, y + 17), 2)
        self.text('FLAPPY FLY', (24, 18), font=self.title)
        self.text(f'SCORE {s.score:02d}   BEST {best:02d}', (26, 62), (156, 223, 169))
        self.text(f'{mode.upper()}  |  {fps:.0f} FPS', (26, g.height - 29), font=self.small)
        self.text('SPACE flap    R restart    ESC quit', (280, g.height - 29), font=self.small)
        self.screen.set_clip(None)
        self.dashboard.draw(self, env, controller, action, history, mode)
        self.text(f'Frame compute {mean_frame_ms:.1f} ms', (g.width + 24, 545), font=self.small)
        self.text('1 human  2 random  3 oracle  4 instinct  5 brain', (26, 91), font=self.small)
        if notice:
            self.text(notice[:85], (26, 116), (255, 185, 130), font=self.small)
        if not s.alive:
            overlay = pygame.Surface((g.width - 60, 330), pygame.SRCALPHA)
            overlay.fill((10, 18, 26, 235))
            self.screen.blit(overlay, (30, 208))
            self.text('THE FLY HAS MADE A DECISION.', (56, 260))
            self.text('Unfortunately, it was wrong.', (56, 295))
            self.text('Press R to try the same seed again.', (56, 344), font=self.small)
        if not s.alive:
            from flappy_fly.ui.plots import death_analysis
            death_analysis(self, history)
        pygame.display.flip()

    def close(self):
        pygame.quit()
