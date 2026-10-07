"""
PROJECT ECHO Tactical Squad Leader HUD
Pygame-based tactical CQB visualization with squad units and gesture-controlled formations.

Usage (standalone test):
    python tactical_hud.py

Keyboard controls (standalone):
    1 = IDLE (Squad Standby)
    2 = HALT (Hold Position — clenched fist)
    3 = ADVANCE (Assault Forward — sweep arms)
    ESC = Quit
"""

import pygame
import sys
import math
import random
import time

import numpy as np
from collections import deque

WIDTH, HEIGHT = 1200, 700
FPS = 60
NUM_UNITS = 5
DOPPLER_HISTORY = 120  # frames of spectrogram history
DOPPLER_BINS = 24

BG_COLOR = (10, 12, 20)
GRID_COLOR = (25, 30, 45)
UNIT_COLORS = [
    (0, 200, 255),
    (0, 255, 180),
    (130, 100, 255),
    (255, 180, 0),
    (255, 80, 100),
]
TRAIL_ALPHA = 40
HUD_COLOR = (0, 200, 255)
HUD_BG = (10, 12, 20, 180)
LABEL_COLORS = {
    "empty_room": (100, 100, 100),
    "idle": (100, 200, 100),
    "gesture_halt": (255, 60, 60),
    "gesture_advance": (0, 200, 255),
}


class SquadUnit:
    def __init__(self, x, y, color, unit_id):
        self.x = x
        self.y = y
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)
        self.target_x = x
        self.target_y = y
        self.color = color
        self.unit_id = unit_id
        self.trail = []
        self.pulse = 0
        self.size = 8

    def update_target(self, tx, ty):
        self.target_x = tx
        self.target_y = ty

    def update(self, units, mode):
        if mode in ["idle", "empty_room"]:
            self._flock(units)
        elif mode == "gesture_halt":
            self._freeze()
        elif mode == "gesture_advance":
            self._move_to_target(speed=0.10)

        self.x += self.vx
        self.y += self.vy

        self.vx *= 0.95
        self.vy *= 0.95

        margin = 40
        if self.x < margin:
            self.vx += 0.5
        if self.x > WIDTH - margin:
            self.vx -= 0.5
        if self.y < margin:
            self.vy += 0.5
        if self.y > HEIGHT - margin:
            self.vy -= 0.5

        self.trail.append((self.x, self.y))
        if len(self.trail) > 30:
            self.trail.pop(0)

        self.pulse = (self.pulse + 0.05) % (2 * math.pi)

    def _flock(self, units):
        cx, cy = WIDTH / 2, HEIGHT / 2
        sep_x, sep_y = 0, 0
        align_x, align_y = 0, 0
        coh_x, coh_y = 0, 0
        neighbors = 0

        for other in units:
            if other is self:
                continue
            dx = other.x - self.x
            dy = other.y - self.y
            dist = math.sqrt(dx**2 + dy**2) + 0.001

            if dist < 60:
                sep_x -= dx / dist * 2
                sep_y -= dy / dist * 2

            if dist < 150:
                align_x += other.vx
                align_y += other.vy
                neighbors += 1

            coh_x += other.x
            coh_y += other.y

        if neighbors > 0:
            align_x /= neighbors
            align_y /= neighbors
            coh_x = coh_x / (len(units) - 1) - self.x
            coh_y = coh_y / (len(units) - 1) - self.y

        center_pull_x = (cx - self.x) * 0.003
        center_pull_y = (cy - self.y) * 0.003

        wander_x = random.uniform(-0.3, 0.3)
        wander_y = random.uniform(-0.3, 0.3)

        self.vx += sep_x * 0.05 + align_x * 0.02 + coh_x * 0.005 + center_pull_x + wander_x
        self.vy += sep_y * 0.05 + align_y * 0.02 + coh_y * 0.005 + center_pull_y + wander_y

        speed = math.sqrt(self.vx**2 + self.vy**2)
        if speed > 3:
            self.vx = self.vx / speed * 3
            self.vy = self.vy / speed * 3

    def _freeze(self):
        """HALT: rapidly decelerate and hold position."""
        self.vx *= 0.8
        self.vy *= 0.8
        # Drift slowly toward hold target
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        self.vx += dx * 0.02
        self.vy += dy * 0.02

    def _move_to_target(self, speed=0.10):
        """ADVANCE: move toward spread-out forward positions."""
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        self.vx += dx * speed
        self.vy += dy * speed

        spd = math.sqrt(self.vx**2 + self.vy**2)
        if spd > 6:
            self.vx = self.vx / spd * 6
            self.vy = self.vy / spd * 6

    def draw(self, surface):
        for i, (tx, ty) in enumerate(self.trail):
            alpha = int(255 * (i / len(self.trail)) * 0.3)
            r, g, b = self.color
            trail_surf = pygame.Surface((6, 6), pygame.SRCALPHA)
            pygame.draw.circle(trail_surf, (r, g, b, alpha), (3, 3), 3)
            surface.blit(trail_surf, (int(tx) - 3, int(ty) - 3))

        glow_size = int(self.size + 12 + math.sin(self.pulse) * 4)
        glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        r, g, b = self.color
        pygame.draw.circle(glow_surf, (r, g, b, 30), (glow_size, glow_size), glow_size)
        surface.blit(glow_surf, (int(self.x) - glow_size, int(self.y) - glow_size))

        pulse_size = self.size + math.sin(self.pulse) * 2
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(pulse_size))
        pygame.draw.circle(surface, (255, 255, 255), (int(self.x), int(self.y)), int(pulse_size * 0.4))


def get_formation_targets(mode, num_units):
    targets = []
    cx, cy = WIDTH / 2, HEIGHT / 2

    if mode == "gesture_halt":
        for i in range(num_units):
            angle = (2 * math.pi / num_units) * i
            r = 35
            targets.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))

    elif mode == "gesture_advance":
        positions = [
            (cx, 100),
            (cx - 180, 220),
            (cx + 180, 220),
            (cx - 350, 350),
            (cx + 350, 350),
        ]
        for i in range(num_units):
            targets.append(positions[i % len(positions)])

    else:
        for i in range(num_units):
            targets.append((cx + random.uniform(-50, 50), cy + random.uniform(-50, 50)))

    return targets


def draw_hud(surface, mode, confidence, font, small_font):
    bar_rect = pygame.Rect(0, 0, WIDTH, 50)
    bar_surf = pygame.Surface((WIDTH, 50), pygame.SRCALPHA)
    bar_surf.fill((0, 0, 0, 160))
    surface.blit(bar_surf, (0, 0))

    title = font.render("PROJECT ECHO  ·  Tactical Silent Command Communication (TSCC)", True, HUD_COLOR)
    surface.blit(title, (20, 12))

    panel_h = 100
    panel_w = 350
    panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel_surf.fill((0, 0, 0, 160))
    surface.blit(panel_surf, (15, HEIGHT - panel_h - 15))

    label_color = LABEL_COLORS.get(mode, (200, 200, 200))
    display_name = {
        "empty_room": "👻 NO COMMANDER DETECTED",
        "idle": "IDLE — Squad Standby",
        "gesture_halt": "✊ HALT — HOLD POSITION",
        "gesture_advance": "🫸 ADVANCE — ASSAULT FORWARD",
    }.get(mode, mode.upper())

    label_text = font.render(f"▶ {display_name}", True, label_color)
    surface.blit(label_text, (25, HEIGHT - panel_h))

    conf_text = small_font.render(f"Confidence: {confidence:.0%}", True, (180, 180, 180))
    surface.blit(conf_text, (25, HEIGHT - panel_h + 35))

    bar_x, bar_y = 25, HEIGHT - panel_h + 58
    bar_w, bar_h = 300, 14
    pygame.draw.rect(surface, (40, 40, 60), (bar_x, bar_y, bar_w, bar_h), border_radius=7)
    fill_w = int(bar_w * confidence)
    if fill_w > 0:
        pygame.draw.rect(surface, label_color, (bar_x, bar_y, fill_w, bar_h), border_radius=7)

    cmd_text = small_font.render(f"Squad Order → {display_name}", True, (120, 130, 150))
    surface.blit(cmd_text, (25, HEIGHT - panel_h + 78))

    count_text = small_font.render(f"Squad: {NUM_UNITS} Active Units", True, (120, 130, 150))
    surface.blit(count_text, (WIDTH - 200, 15))


def draw_grid(surface):
    spacing = 50
    for x in range(0, WIDTH - 300, spacing):
        pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, spacing):
        pygame.draw.line(surface, GRID_COLOR, (0, y), (WIDTH - 300, y))


def doppler_color(value, max_val):
    """Convert a Doppler magnitude to a heat color (black → blue → cyan → green → yellow → red → white)."""
    if max_val < 1e-6:
        return (0, 0, 0)
    t = min(value / (max_val + 1e-9), 1.0)
    if t < 0.2:
        r, g, b = 0, 0, int(t / 0.2 * 255)
    elif t < 0.4:
        r, g, b = 0, int((t - 0.2) / 0.2 * 255), 255
    elif t < 0.6:
        r, g, b = 0, 255, int((1 - (t - 0.4) / 0.2) * 255)
    elif t < 0.8:
        r, g, b = int((t - 0.6) / 0.2 * 255), 255, 0
    else:
        r, g, b = 255, int((1 - (t - 0.8) / 0.2) * 255), 0
    return (min(r, 255), min(g, 255), min(b, 255))


def draw_doppler_spectrogram(surface, doppler_history, font, small_font):
    """Draw a scrolling Doppler spectrogram on the right panel."""
    panel_x = WIDTH - 290
    panel_y = 60
    panel_w = 270
    panel_h = HEIGHT - 80

    # Background
    panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel_surf.fill((0, 0, 0, 200))
    surface.blit(panel_surf, (panel_x, panel_y))

    # Title
    title = font.render("DOPPLER RADAR", True, (0, 200, 255))
    surface.blit(title, (panel_x + 10, panel_y + 5))
    subtitle = small_font.render("Velocity Spectrum (DC Removed)", True, (100, 120, 140))
    surface.blit(subtitle, (panel_x + 10, panel_y + 25))

    # Spectrogram area
    spec_x = panel_x + 10
    spec_y = panel_y + 45
    spec_w = panel_w - 20
    spec_h = panel_h - 90

    if len(doppler_history) == 0:
        no_data = small_font.render("Awaiting CSI data...", True, (80, 80, 80))
        surface.blit(no_data, (spec_x + 20, spec_y + spec_h // 2))
        return

    # Find global max for color scaling
    all_vals = np.array(list(doppler_history))
    global_max = np.max(all_vals) if all_vals.size > 0 else 1.0

    # Draw spectrogram: each column = one time step, each row = one Doppler bin
    n_cols = len(doppler_history)
    col_w = max(1, spec_w // DOPPLER_HISTORY)
    row_h = max(1, spec_h // DOPPLER_BINS)

    for col_idx, spectrum in enumerate(doppler_history):
        x = spec_x + col_idx * col_w
        if x > spec_x + spec_w:
            break
        for bin_idx in range(min(DOPPLER_BINS, len(spectrum))):
            y = spec_y + (DOPPLER_BINS - 1 - bin_idx) * row_h
            color = doppler_color(spectrum[bin_idx], global_max)
            pygame.draw.rect(surface, color, (x, y, col_w, row_h))

    # Border
    pygame.draw.rect(surface, (0, 200, 255), (spec_x, spec_y, spec_w, spec_h), 1)

    # Axis labels
    high_label = small_font.render("Fast ↑", True, (100, 120, 140))
    low_label = small_font.render("Slow ↓", True, (100, 120, 140))
    surface.blit(high_label, (spec_x, spec_y - 15))
    surface.blit(low_label, (spec_x, spec_y + spec_h + 3))

    time_label = small_font.render("← Time", True, (100, 120, 140))
    surface.blit(time_label, (spec_x + spec_w - 55, spec_y + spec_h + 3))

    # Motion energy bar
    total_energy = np.sum(list(doppler_history)[-1]) if len(doppler_history) > 0 else 0
    energy_label = small_font.render(f"Motion Energy: {total_energy:.0f}", True, (0, 200, 255))
    surface.blit(energy_label, (panel_x + 10, panel_y + panel_h - 35))


class TacticalHUD:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("PROJECT ECHO — Tactical Silent Command HUD (CQB / Through-Wall)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 18, bold=True)
        self.small_font = pygame.font.SysFont("Consolas", 14)

        cx, cy = WIDTH / 2, HEIGHT / 2
        self.units = []
        for i in range(NUM_UNITS):
            x = cx + random.uniform(-80, 80)
            y = cy + random.uniform(-80, 80)
            self.units.append(SquadUnit(x, y, UNIT_COLORS[i % len(UNIT_COLORS)], i))

        self.current_mode = "idle"
        self.current_confidence = 0.0
        self.current_doppler = np.zeros(DOPPLER_BINS)
        self.doppler_history = deque(maxlen=DOPPLER_HISTORY)
        self.running = True

    def set_mode(self, mode, confidence=1.0):
        if mode != self.current_mode:
            self.current_mode = mode
            targets = get_formation_targets(mode, NUM_UNITS)
            for i, unit in enumerate(self.units):
                unit.update_target(*targets[i])
        self.current_confidence = confidence

    def run(self, get_external_state=None):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_1:
                        self.set_mode("idle")
                    elif event.key == pygame.K_2:
                        self.set_mode("gesture_halt")
                    elif event.key == pygame.K_3:
                        self.set_mode("gesture_advance")

            if get_external_state:
                state = get_external_state()
                if len(state) == 3:
                    label, confidence, doppler = state
                    self.set_mode(label, confidence)
                    self.current_doppler = doppler
                    self.doppler_history.append(doppler)
                else:
                    label, confidence = state
                    self.set_mode(label, confidence)

            for unit in self.units:
                unit.update(self.units, self.current_mode)

            self.screen.fill(BG_COLOR)
            draw_grid(self.screen)

            for unit in self.units:
                unit.draw(self.screen)

            draw_hud(self.screen, self.current_mode, self.current_confidence,
                     self.font, self.small_font)

            draw_doppler_spectrogram(self.screen, self.doppler_history,
                                     self.font, self.small_font)

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   PROJECT ECHO TSCC — Squad Leader HUD (Test Mode)       ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Press 1 = IDLE (Squad Standby)                          ║")
    print("║  Press 2 = HALT (Hold Position - Clenched Fist)          ║")
    print("║  Press 3 = ADVANCE (Assault Forward - Sweep Arms)        ║")
    print("║  ESC = Quit                                              ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    demo = TacticalHUD()
    demo.run()


if __name__ == "__main__":
    main()
