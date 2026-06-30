"""Fullscreen projector rendering with pygame.

Draws the marbles, optional debug overlays (the object outlines physics is using
and a heads-up display), and the calibration markers. Everything is drawn in
display-pixel coordinates, matching the physics world.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import pygame

import config


class Display:
    def __init__(
        self,
        width: int = config.DISPLAY_WIDTH,
        height: int = config.DISPLAY_HEIGHT,
        fullscreen: bool = config.FULLSCREEN,
    ) -> None:
        pygame.init()
        pygame.display.set_caption("Virtual Marble Run")
        self.width = width
        self.height = height
        self.fullscreen = fullscreen
        self._make_surface()
        self.font = pygame.font.SysFont("consolas", 22)
        self.big_font = pygame.font.SysFont("consolas", 40, bold=True)
        self.clock = pygame.time.Clock()

    def _make_surface(self) -> None:
        flags = pygame.FULLSCREEN | pygame.SCALED if self.fullscreen else 0
        self.screen = pygame.display.set_mode((self.width, self.height), flags)

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        self._make_surface()

    def clear(self) -> None:
        self.screen.fill(config.BACKGROUND_COLOR)

    def draw_marbles(self, marbles: Iterable) -> None:
        for marble in marbles:
            x, y = marble.position
            ix, iy = int(x), int(y)
            r = int(marble.radius)
            pygame.draw.circle(self.screen, marble.color, (ix, iy), r)
            # Glossy highlight for a marble-like look.
            hl = (min(marble.color[0] + 70, 255),
                  min(marble.color[1] + 70, 255),
                  min(marble.color[2] + 70, 255))
            pygame.draw.circle(self.screen, hl, (ix - r // 3, iy - r // 3), max(2, r // 3))

    @staticmethod
    def _outline_color(pts: np.ndarray) -> Tuple[int, int, int]:
        """A bright, random-looking color that stays stable for a given object.

        Derived from the object's (quantised) centroid so the same shape keeps
        the same color across frames instead of flickering, while different
        objects get different colors. Full saturation/value for visibility.
        """
        cx, cy = pts.mean(axis=0)
        seed = int(cx) // 12 * 73856093 ^ int(cy) // 12 * 19349663
        hue = (seed % 360) / 360.0
        col = pygame.Color(0)
        col.hsva = (hue * 360.0, 90, 100, 100)
        return (col.r, col.g, col.b)

    def draw_object_outlines(
        self,
        polygons: Sequence[np.ndarray],
        color: Tuple[int, int, int] | None = None,
        width: int = 3,
        filled: bool = False,
    ) -> None:
        for poly in polygons:
            pts = np.asarray(poly, dtype=int).reshape(-1, 2)
            if len(pts) < 3:
                continue
            c = color if color is not None else self._outline_color(pts)
            verts = [tuple(p) for p in pts]
            if filled:
                pygame.draw.polygon(self.screen, c, verts, 0)  # 0 = filled
            else:
                pygame.draw.polygon(self.screen, c, verts, width)

    def draw_calibration_markers(
        self,
        points: np.ndarray,
        labels: Sequence[str],
    ) -> None:
        """Draw numbered registration targets the operator clicks in the camera.

        Concentric rings plus a crosshair and a single bright center pixel make
        the exact click point unambiguous, which is what makes alignment precise.
        """
        for (px, py), label in zip(np.asarray(points, dtype=int), labels):
            px, py = int(px), int(py)
            center = (px, py)
            pygame.draw.circle(self.screen, (255, 255, 255), center, 28, 2)
            pygame.draw.circle(self.screen, (255, 255, 255), center, 14, 2)
            pygame.draw.line(self.screen, (255, 255, 255), (px - 40, py), (px + 40, py), 1)
            pygame.draw.line(self.screen, (255, 255, 255), (px, py - 40), (px, py + 40), 1)
            # Bright center dot = the precise point to click.
            pygame.draw.circle(self.screen, (255, 60, 60), center, 4)
            self.screen.set_at(center, (255, 255, 0))
            text = self.font.render(label.split(":")[0], True, (255, 255, 0))
            self.screen.blit(text, (px + 18, py - 40))

    def frame_to_surface(self, frame_bgr: np.ndarray) -> pygame.Surface:
        """Convert a BGR OpenCV frame to a pygame Surface (size = camera res)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        return pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")

    def draw_loupe(
        self,
        frame_surf: pygame.Surface,
        cam_xy: Tuple[float, float],
        mouse_xy: Tuple[int, int],
        src: int = 70,
        out: int = 220,
    ) -> None:
        """Draw a magnified inset of `frame_surf` around `cam_xy` for precision.

        The loupe is placed in the corner opposite the cursor so it never hides
        what is being clicked.
        """
        cw, ch = frame_surf.get_size()
        cx, cy = cam_xy
        x0 = int(max(0, min(cw - src, cx - src / 2)))
        y0 = int(max(0, min(ch - src, cy - src / 2)))
        try:
            sub = frame_surf.subsurface((x0, y0, src, src)).copy()
        except ValueError:
            return
        mag = pygame.transform.scale(sub, (out, out))
        rel_x = int((cx - x0) / src * out)
        rel_y = int((cy - y0) / src * out)
        pygame.draw.line(mag, (0, 255, 255), (rel_x, 0), (rel_x, out), 1)
        pygame.draw.line(mag, (0, 255, 255), (0, rel_y), (out, rel_y), 1)
        pygame.draw.rect(mag, (0, 255, 255), (0, 0, out, out), 2)
        mx, my = mouse_xy
        px = 20 if mx > self.width // 2 else self.width - out - 20
        py = 20 if my > self.height // 2 else self.height - out - 20
        self.screen.blit(mag, (px, py))

    def draw_center_text(self, lines: Sequence[str]) -> None:
        total_h = len(lines) * 48
        y = self.height // 2 - total_h // 2
        for line in lines:
            surf = self.big_font.render(line, True, (240, 240, 240))
            rect = surf.get_rect(center=(self.width // 2, y))
            self.screen.blit(surf, rect)
            y += 48

    def draw_hud(self, lines: Sequence[str]) -> None:
        y = 12
        for line in lines:
            surf = self.font.render(line, True, (200, 255, 200))
            # subtle shadow for readability over busy backgrounds
            shadow = self.font.render(line, True, (0, 0, 0))
            self.screen.blit(shadow, (13, y + 1))
            self.screen.blit(surf, (12, y))
            y += 26

    def draw_badges(self, badges: Sequence[Tuple[str, Tuple[int, int, int]]]) -> None:
        """Draw compact status labels at the top-center (e.g. PAUSED, LOCKED).

        These stay visible even when the full HUD text is hidden, so operators
        always know the current mode.
        """
        y = 10
        for text, color in badges:
            surf = self.big_font.render(text, True, color)
            rect = surf.get_rect(center=(self.width // 2, y + surf.get_height() // 2))
            shadow = self.big_font.render(text, True, (0, 0, 0))
            self.screen.blit(shadow, (rect.x + 2, rect.y + 2))
            self.screen.blit(surf, rect)
            y += surf.get_height() + 6

    def flip(self) -> None:
        pygame.display.flip()

    def tick(self, fps: int = config.TARGET_FPS) -> float:
        """Advance the clock; return dt in seconds (clamped for stability)."""
        ms = self.clock.tick(fps)
        return min(ms / 1000.0, 1.0 / 20.0)

    def quit(self) -> None:
        pygame.quit()
