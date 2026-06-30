"""2D physics for the marble run, powered by pymunk (Chipmunk).

Coordinate system: we work entirely in *display pixels*. Y grows downward to
match pygame's screen coordinates, so gravity is a positive Y acceleration.

Two kinds of bodies live in the world:
  * Static "object outlines": each detected object's polygon becomes a closed
    loop of static segments. Marbles bounce and roll along these edges, which is
    exactly what we want for a marble run (vs. solid blobs they'd sink into).
  * Dynamic marbles: circles spawned at top-center that fall, bounce and roll
    until they leave the play area, at which point they are removed.
"""

from __future__ import annotations

import logging
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pymunk

import config

log = logging.getLogger("marblerun.physics")


def _signed_area(loop) -> float:
    """Shoelace signed area of a point list (loop assumed, no closing dup needed)."""
    s = 0.0
    n = len(loop)
    for i in range(n):
        x1, y1 = loop[i]
        x2, y2 = loop[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _dedupe(pts: List[Tuple[float, float]], min_dist: float = 1.0) -> List[Tuple[float, float]]:
    """Drop consecutive points closer than `min_dist` (incl. the wrap-around)."""
    out: List[Tuple[float, float]] = []
    for p in pts:
        if not out or (abs(p[0] - out[-1][0]) + abs(p[1] - out[-1][1])) >= min_dist:
            out.append(p)
    if len(out) >= 2 and (abs(out[0][0] - out[-1][0]) + abs(out[0][1] - out[-1][1])) < min_dist:
        out.pop()
    return out


def _point_in_tri(p, a, b, c) -> bool:
    """True if point p is strictly inside triangle abc."""
    d1 = (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
    d2 = (p[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (p[1] - c[1])
    d3 = (p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _convex_pieces(pts: np.ndarray, tolerance: float = 0.0) -> List[List[Tuple[float, float]]]:
    """Triangulate a simple polygon into convex (triangle) pieces.

    This is a pure-Python ear-clipping triangulation. Unlike calling into native
    convex-decomposition code, it can never segfault on a degenerate or
    self-touching contour - the worst case is that it returns fewer triangles.
    Each returned piece is a CCW triangle suitable for a pymunk.Poly.
    """
    poly = _dedupe([(float(x), float(y)) for x, y in pts])
    n = len(poly)
    if n < 3:
        return []
    if _signed_area(poly) < 0:  # ensure counter-clockwise
        poly.reverse()

    idx = list(range(len(poly)))
    tris: List[List[Tuple[float, float]]] = []
    guard = 0
    max_guard = 3 * len(idx) + 10
    while len(idx) > 3 and guard < max_guard:
        guard += 1
        m = len(idx)
        ear = False
        for i in range(m):
            i0, i1, i2 = idx[(i - 1) % m], idx[i], idx[(i + 1) % m]
            a, b, c = poly[i0], poly[i1], poly[i2]
            # Convex corner for CCW winding has positive cross product.
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 0:
                continue
            if any(
                _point_in_tri(poly[idx[j]], a, b, c)
                for j in range(m) if idx[j] not in (i0, i1, i2)
            ):
                continue
            tris.append([a, b, c])
            del idx[i]
            ear = True
            break
        if not ear:
            break  # numerical trouble; stop with what we have
    if len(idx) == 3:
        tris.append([poly[idx[0]], poly[idx[1]], poly[idx[2]]])
    return tris


class Marble:
    """A single dynamic marble plus its render color."""

    __slots__ = ("body", "shape", "color", "radius")

    def __init__(self, body: pymunk.Body, shape: pymunk.Circle, color, radius: float):
        self.body = body
        self.shape = shape
        self.color = color
        self.radius = radius

    @property
    def position(self) -> Tuple[float, float]:
        return self.body.position.x, self.body.position.y


class World:
    def __init__(
        self,
        width: int = config.DISPLAY_WIDTH,
        height: int = config.DISPLAY_HEIGHT,
    ) -> None:
        self.width = width
        self.height = height

        self.space = pymunk.Space()
        self.space.gravity = (0.0, config.GRAVITY)
        # A little global damping keeps things calm and prevents jitter buildup.
        self.space.damping = 0.98

        self.marbles: List[Marble] = []
        self._static_shapes: List[pymunk.Shape] = []
        self._static_signature: Optional[Tuple] = None
        self._mask_signature: Optional[Tuple] = None
        self._color_idx = 0
        self.palette = list(config.MARBLE_COLORS)
        # Display-space polygons currently realised as static geometry (for debug
        # drawing of exactly what physics "sees").
        self.static_polygons: List[np.ndarray] = []

        if config.SIDE_WALLS:
            self._add_side_walls()

    # ------------------------------------------------------------------ #
    # Static geometry (detected objects)
    # ------------------------------------------------------------------ #
    def _add_side_walls(self) -> None:
        body = self.space.static_body
        pts = [
            ((0, 0), (0, self.height)),                       # left
            ((self.width, 0), (self.width, self.height)),     # right
        ]
        for a, b in pts:
            seg = pymunk.Segment(body, a, b, config.SEGMENT_THICKNESS)
            seg.friction = config.OBJECT_FRICTION
            seg.elasticity = config.OBJECT_ELASTICITY
            self.space.add(seg)

    @staticmethod
    def _signature(polygons: List[np.ndarray]) -> Tuple:
        """Cheap fingerprint of the object set, used to debounce rebuilds.

        Combines the count, total perimeter and centroid sum (quantised) so that
        tiny vision jitter does not trigger a rebuild, but real movement does.
        """
        if not polygons:
            return (0,)
        total_len = 0.0
        cx = cy = 0.0
        n = 0
        for p in polygons:
            d = np.diff(np.vstack([p, p[:1]]), axis=0)
            total_len += float(np.sqrt((d ** 2).sum(axis=1)).sum())
            cx += float(p[:, 0].sum())
            cy += float(p[:, 1].sum())
            n += len(p)
        q = 1.0 / 8.0  # quantise to ~8px buckets
        return (
            len(polygons),
            round(total_len * q),
            round(cx * q),
            round(cy * q),
            n,
        )

    def maybe_rebuild_static(self, polygons: List[np.ndarray]) -> bool:
        """Rebuild static geometry only if the object set changed meaningfully.

        Returns True if a rebuild happened.
        """
        sig = self._signature(polygons)
        if self._static_signature is not None:
            old_total = self._static_signature[1] if len(self._static_signature) > 1 else 0
            new_total = sig[1] if len(sig) > 1 else 0
            same_count = sig[0] == self._static_signature[0]
            denom = max(1.0, float(old_total))
            changed_frac = abs(new_total - old_total) / denom
            centroid_moved = (
                len(sig) > 3
                and len(self._static_signature) > 3
                and (
                    sig[2] != self._static_signature[2]
                    or sig[3] != self._static_signature[3]
                )
            )
            if same_count and changed_frac < config.REBUILD_CHANGE_FRAC and not centroid_moved:
                return False
        self.rebuild_static(polygons)
        self._static_signature = sig
        return True

    def rebuild_static(self, polygons: List[np.ndarray]) -> None:
        """Replace all object outlines with segment chains from `polygons`."""
        for shp in self._static_shapes:
            self.space.remove(shp)
        self._static_shapes.clear()

        body = self.space.static_body
        clean_polys: List[np.ndarray] = []
        for poly in polygons:
            pts = np.asarray(poly, dtype=float).reshape(-1, 2)
            if len(pts) < 3:
                continue
            clean_polys.append(pts)
            n = len(pts)
            for i in range(n):
                a = tuple(pts[i])
                b = tuple(pts[(i + 1) % n])  # close the loop
                seg = pymunk.Segment(body, a, b, config.SEGMENT_THICKNESS)
                seg.friction = config.OBJECT_FRICTION
                seg.elasticity = config.OBJECT_ELASTICITY
                self.space.add(seg)
                self._static_shapes.append(seg)
        self.static_polygons = clean_polys

    # ------------------------------------------------------------------ #
    # Static geometry from a mask (white pixels = hard surfaces)
    # ------------------------------------------------------------------ #
    def maybe_rebuild_from_mask(self, mask: Optional[np.ndarray]) -> bool:
        """Rebuild collision geometry from a display-space mask if it changed.

        `mask` is a uint8 image (display resolution) where white (>127) pixels
        are solid. Uses area + centroid as a cheap fingerprint so we only rebuild
        when objects actually appear, move or change size.
        """
        if mask is None:
            return False
        m = cv2.moments(mask, binaryImage=True)
        area = m["m00"]
        if area <= 0:
            sig: Tuple = (0,)
        else:
            cx = m["m10"] / area
            cy = m["m01"] / area
            sig = (round(area / 600.0), round(cx / 8.0), round(cy / 8.0))
        if sig == self._mask_signature:
            return False
        self._mask_signature = sig
        self.rebuild_static_from_mask(mask)
        return True

    def rebuild_static_from_mask(self, mask: Optional[np.ndarray]) -> None:
        """Replace collision geometry with shapes traced from a mask.

        Marbles bounce off the exact shape that was detected (pixel-accurate,
        only lightly simplified for performance). When `OBJECT_SOLID` is set the
        shapes are filled solid bodies (triangulated) so a marble can never get
        stuck inside; otherwise they are hollow segment outlines.
        """
        for shp in self._static_shapes:
            self.space.remove(shp)
        self._static_shapes.clear()

        polys: List[np.ndarray] = []
        if mask is not None:
            # Solid shapes fill holes (RETR_EXTERNAL); hollow outlines keep
            # interior holes as their own loops (RETR_CCOMP).
            retr = cv2.RETR_EXTERNAL if config.OBJECT_SOLID else cv2.RETR_CCOMP
            contours, _ = cv2.findContours(mask, retr, cv2.CHAIN_APPROX_SIMPLE)
            body = self.space.static_body
            eps = config.MASK_COLLISION_EPSILON
            for c in contours:
                if cv2.contourArea(c) < config.MASK_MIN_OBJECT_AREA:
                    continue
                if len(self._static_shapes) >= config.MAX_STATIC_SHAPES:
                    log.warning("static shape cap (%d) reached; skipping remaining objects",
                                config.MAX_STATIC_SHAPES)
                    break
                pts = self._simplify_contour(c, eps)
                if pts is None or len(pts) < 3:
                    continue
                polys.append(pts)
                try:
                    if config.OBJECT_SOLID:
                        self._add_solid(body, pts)
                    else:
                        self._add_outline(body, pts)
                except Exception:
                    # Never let one bad shape take down the whole run.
                    log.exception("failed to build collision shape for a contour")
        self.static_polygons = polys

    @staticmethod
    def _simplify_contour(c: np.ndarray, eps: float) -> Optional[np.ndarray]:
        """approxPolyDP with enough simplification to stay under the vertex cap."""
        e = eps if eps and eps > 0 else 1.0
        approx = cv2.approxPolyDP(c, e, True)
        # Keep simplifying (coarser) until the vertex count is bounded.
        for _ in range(8):
            if len(approx) <= config.MAX_CONTOUR_VERTS:
                break
            e *= 1.8
            approx = cv2.approxPolyDP(c, e, True)
        return approx.reshape(-1, 2).astype(float)

    def _add_outline(self, body: pymunk.Body, pts: np.ndarray) -> None:
        n = len(pts)
        for i in range(n):
            a = tuple(pts[i])
            b = tuple(pts[(i + 1) % n])  # close the loop
            seg = pymunk.Segment(body, a, b, config.SEGMENT_THICKNESS)
            seg.friction = config.OBJECT_FRICTION
            seg.elasticity = config.OBJECT_ELASTICITY
            self.space.add(seg)
            self._static_shapes.append(seg)

    def _add_solid(self, body: pymunk.Body, pts: np.ndarray) -> None:
        pieces = _convex_pieces(pts, config.SOLID_DECOMP_TOLERANCE)
        if not pieces:
            # Fall back to a hollow outline rather than losing the object.
            self._add_outline(body, pts)
            return
        added = 0
        for piece in pieces:
            if len(self._static_shapes) >= config.MAX_STATIC_SHAPES:
                break
            if len(piece) < 3 or abs(_signed_area(piece)) < 1.0:
                continue  # skip degenerate / zero-area triangles
            try:
                poly = pymunk.Poly(body, piece)
                poly.friction = config.OBJECT_FRICTION
                poly.elasticity = config.OBJECT_ELASTICITY
                self.space.add(poly)
                self._static_shapes.append(poly)
                added += 1
            except Exception:
                continue
        if added == 0:
            self._add_outline(body, pts)

    # ------------------------------------------------------------------ #
    # Marbles
    # ------------------------------------------------------------------ #
    def spawn_marble(self) -> Optional[Marble]:
        if len(self.marbles) >= config.MAX_MARBLES:
            return None
        radius = config.MARBLE_RADIUS
        moment = pymunk.moment_for_circle(config.MARBLE_MASS, 0.0, radius)
        body = pymunk.Body(config.MARBLE_MASS, moment)
        half_span = self.width * config.SPAWN_WIDTH_FRAC / 2.0
        x = self.width / 2 + random.uniform(-half_span, half_span)
        body.position = (x, config.SPAWN_Y)
        body.velocity = (random.uniform(-config.SPAWN_VX_RANGE, config.SPAWN_VX_RANGE), 0.0)

        shape = pymunk.Circle(body, radius)
        shape.friction = config.MARBLE_FRICTION
        shape.elasticity = config.MARBLE_ELASTICITY

        self.space.add(body, shape)
        color = self.palette[self._color_idx % len(self.palette)]
        self._color_idx += 1
        marble = Marble(body, shape, color, radius)
        self.marbles.append(marble)
        return marble

    def _cull_offscreen(self) -> None:
        m = config.CULL_MARGIN
        survivors: List[Marble] = []
        for marble in self.marbles:
            x, y = marble.position
            if y > self.height + m or x < -m or x > self.width + m or y < -10 * self.height:
                self.space.remove(marble.body, marble.shape)
            else:
                survivors.append(marble)
        self.marbles = survivors

    def step(self, dt: float) -> None:
        substeps = max(1, config.PHYSICS_SUBSTEPS)
        h = dt / substeps
        for _ in range(substeps):
            self.space.step(h)
        self._cull_offscreen()

    def clear_marbles(self) -> None:
        for marble in self.marbles:
            self.space.remove(marble.body, marble.shape)
        self.marbles.clear()

    def set_palette(self, colors, recolor: bool = True) -> None:
        """Switch the marble color palette. Optionally recolor existing marbles."""
        self.palette = list(colors)
        if recolor:
            n = len(self.palette)
            for i, marble in enumerate(self.marbles):
                marble.color = self.palette[i % n]
