"""Object detection by background differencing.

We snapshot the empty wall once ("background"). Each subsequent camera frame is
compared against that snapshot: any region that differs strongly is something a
kid added (a sticky note, a hand-drawn shape, a toy). We turn those regions into
polygon outlines in *camera* pixel space; main.py warps them to display space
and physics.py turns them into things marbles bounce off.

A fixed reference background (rather than an adaptive subtractor) is deliberate:
objects that sit still must NOT fade into the background, otherwise marbles would
stop colliding with a sticky note that has been on the wall for a few seconds.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config

log = logging.getLogger("marblerun.vision")

# Detection modes that key on color directly and need no background snapshot.
# For these, peeling tape off never leaves a phantom object.
NO_BACKGROUND_MODES = frozenset({"saturation", "tape", "learned"})


def _blur_bgr(frame: np.ndarray, blur: int) -> np.ndarray:
    """Blur a BGR frame to suppress sensor noise before analysis."""
    if blur and blur >= 3:
        k = blur | 1  # force odd kernel
        return cv2.GaussianBlur(frame, (k, k), 0)
    return frame.copy()


# Backwards-compatible helper: some callers expect a single prepared image.
def _prepare(frame: np.ndarray, blur: int) -> np.ndarray:
    """Return a blurred BGR frame (the reference background representation)."""
    return _blur_bgr(frame, blur)


def _gray_diff_mask(frame_bgr: np.ndarray, bg_bgr: np.ndarray, threshold: int) -> np.ndarray:
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g, b)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return mask


def _lab_diff_mask(
    frame_bgr: np.ndarray, bg_bgr: np.ndarray, threshold: int, l_weight: float
) -> np.ndarray:
    """Color-aware background diff: distance in Lab space.

    Detects objects whose *color* differs from the background even when their
    *brightness* is similar (e.g. blue/green tape on grey drywall). The L
    (lightness) channel is down-weighted so shadows and projector light matter
    less than actual color (a/b) differences.
    """
    lab_f = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = lab_f - lab_b
    dl = d[:, :, 0] * float(l_weight)
    da = d[:, :, 1]
    db = d[:, :, 2]
    dist = np.sqrt(dl * dl + da * da + db * db)
    mask = (dist > float(threshold)).astype(np.uint8) * 255
    return mask


def _tape_mask(
    frame_bgr: np.ndarray,
    hue_ranges,
    min_sat: int,
    min_value: int,
) -> np.ndarray:
    """Detect only specific colors (by HSV hue), independent of any background.

    Because this keys on the tape's color rather than on change-from-background,
    removing tape simply removes its detection - the revealed wall is grey, not
    one of the target hues, so it never registers as a phantom object.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    out: Optional[np.ndarray] = None
    for h_lo, h_hi in hue_ranges:
        lower = np.array([h_lo, min_sat, min_value], dtype=np.uint8)
        upper = np.array([h_hi, 255, 255], dtype=np.uint8)
        m = cv2.inRange(hsv, lower, upper)
        out = m if out is None else cv2.bitwise_or(out, m)
    if out is None:
        return np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    return out


def _learned_mask(frame_bgr: np.ndarray, ranges) -> np.ndarray:
    """Detect colors learned during setup. Each range is [h_lo,h_hi,s_min,v_min].

    Unlike the fixed "tape" hues, these bands are sampled from the tape under the
    real projector lighting, so they track exactly what the camera sees.
    """
    if not ranges:
        return np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    out: Optional[np.ndarray] = None
    for h_lo, h_hi, s_min, v_min in ranges:
        lower = np.array([int(h_lo), int(s_min), int(v_min)], dtype=np.uint8)
        upper = np.array([int(h_hi), 255, 255], dtype=np.uint8)
        m = cv2.inRange(hsv, lower, upper)
        out = m if out is None else cv2.bitwise_or(out, m)
    if out is None:
        return np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    return out


def sample_hsv(frame_bgr: np.ndarray, x: int, y: int,
               patch: int = config.LEARN_PATCH) -> Tuple[int, int, int]:
    """Return the median HSV of a small patch around (x, y) in a BGR frame."""
    h, w = frame_bgr.shape[:2]
    r = max(1, patch // 2)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    region = frame_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    med = np.median(hsv, axis=0)
    return int(med[0]), int(med[1]), int(med[2])


def build_ranges_from_samples(samples) -> List[List[int]]:
    """Cluster HSV samples by hue and turn each cluster into an inRange band."""
    if not samples:
        return []
    pts = sorted(samples, key=lambda s: s[0])  # sort by hue
    clusters: List[List[Tuple[int, int, int]]] = [[pts[0]]]
    for s in pts[1:]:
        if s[0] - clusters[-1][0][0] <= config.LEARN_HUE_CLUSTER:
            clusters[-1].append(s)
        else:
            clusters.append([s])

    ranges: List[List[int]] = []
    for cl in clusters:
        hs = [c[0] for c in cl]
        ss = [c[1] for c in cl]
        vs = [c[2] for c in cl]
        h_lo = max(0, min(hs) - config.LEARN_H_MARGIN)
        h_hi = min(179, max(hs) + config.LEARN_H_MARGIN)
        s_min = max(config.LEARN_S_FLOOR, min(ss) - config.LEARN_S_MARGIN)
        v_min = max(config.LEARN_V_FLOOR, min(vs) - config.LEARN_V_MARGIN)
        ranges.append([int(h_lo), int(h_hi), int(s_min), int(v_min)])
    return ranges


def save_learned_colors(ranges, path: str = config.TAPE_COLOR_FILE) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"ranges": ranges}, f, indent=2)


def load_learned_colors(path: str = config.TAPE_COLOR_FILE) -> List[List[int]]:
    """Load saved learned-color bands, or [] if none/invalid."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ranges = data.get("ranges", [])
        return [[int(v) for v in r] for r in ranges if len(r) == 4]
    except (json.JSONDecodeError, ValueError, TypeError):
        return []


def _saturation_mask(frame_bgr: np.ndarray, sat_thresh: int, min_value: int) -> np.ndarray:
    """Find colorful regions: high HSV saturation and not too dark.

    Grey drywall has near-zero saturation; colored tape is highly saturated, so
    this isolates the tape without even needing a clean background snapshot.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = ((s > sat_thresh) & (v > min_value)).astype(np.uint8) * 255
    return mask


def detect_objects(
    frame: np.ndarray,
    background: Optional[np.ndarray],
    threshold: int = config.DIFF_THRESHOLD,
    blur: int = config.BLUR_KERNEL,
    morph: int = config.MORPH_KERNEL,
    min_area: float = config.MIN_OBJECT_AREA,
    epsilon_frac: float = config.CONTOUR_EPSILON_FRAC,
    mode: Optional[str] = None,
) -> tuple[List[np.ndarray], np.ndarray]:
    """Return (polygons, mask).

    polygons: list of (N, 2) float32 arrays of camera-space contour points.
    mask: the cleaned binary foreground mask (handy for a debug window).

    `background` is a blurred BGR reference frame (from `_blur_bgr`). It may be
    None when `mode == "saturation"`, which does not need one.
    """
    mode = mode or config.DETECTION_MODE
    frame_bgr = _blur_bgr(frame, blur)

    masks: List[np.ndarray] = []
    needs_bg = mode in ("gray", "lab", "combo")
    if needs_bg and background is not None:
        if mode in ("gray",):
            masks.append(_gray_diff_mask(frame_bgr, background, threshold))
        if mode in ("lab", "combo"):
            masks.append(
                _lab_diff_mask(frame_bgr, background, config.COLOR_DIFF_THRESHOLD,
                               config.LAB_L_WEIGHT)
            )
    if mode in ("saturation", "combo"):
        masks.append(
            _saturation_mask(frame_bgr, config.SATURATION_THRESHOLD,
                             config.SATURATION_MIN_VALUE)
        )
    if mode == "tape":
        masks.append(
            _tape_mask(frame_bgr, config.TAPE_HUE_RANGES,
                       config.TAPE_MIN_SATURATION, config.TAPE_MIN_VALUE)
        )
    if mode == "learned":
        masks.append(_learned_mask(frame_bgr, config.LEARNED_COLORS))

    if not masks:
        # No usable signal yet (e.g. background-based mode without a background).
        empty = np.zeros(frame.shape[:2], dtype=np.uint8)
        return [], empty

    mask = masks[0]
    for m in masks[1:]:
        mask = cv2.bitwise_or(mask, m)

    if morph and morph >= 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
        # Open removes speckle noise; close fills small holes inside objects.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons: List[np.ndarray] = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        eps = epsilon_frac * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) >= 3:
            polygons.append(approx.reshape(-1, 2).astype(np.float32))
    return polygons, mask


class VisionThread:
    """Runs object detection on a background thread at a fixed rate."""

    def __init__(self, camera, fps: int = config.VISION_FPS) -> None:
        self.camera = camera
        self.period = 1.0 / max(1, fps)

        self._background: Optional[np.ndarray] = None
        self._polygons: List[np.ndarray] = []
        self._mask: Optional[np.ndarray] = None
        self._version = 0  # bumped each time polygons are refreshed

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def has_background(self) -> bool:
        with self._lock:
            return self._background is not None

    def capture_background(self) -> bool:
        """Snapshot the current camera frame as the empty-wall reference."""
        _id, frame = self.camera.read()
        if frame is None:
            return False
        bg = _prepare(frame, config.BLUR_KERNEL)
        with self._lock:
            self._background = bg
        return True

    def start(self) -> "VisionThread":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            with self._lock:
                bg = self._background
            # Color-keyed modes detect directly and need no captured background.
            ready = bg is not None or config.DETECTION_MODE in NO_BACKGROUND_MODES
            if ready:
                _id, frame = self.camera.read()
                if frame is not None:
                    try:
                        polys, mask = detect_objects(frame, bg)
                    except Exception:
                        # Keep the thread alive no matter what; just log it.
                        log.exception("detect_objects failed")
                        polys, mask = [], None
                    with self._lock:
                        self._polygons = polys
                        self._mask = mask
                        self._version += 1
            # Sleep to hit the target rate without busy-waiting.
            elapsed = time.time() - t0
            time.sleep(max(0.0, self.period - elapsed))

    def get_objects(self) -> tuple[int, List[np.ndarray]]:
        """Return (version, polygons). Compare version to skip unchanged work."""
        with self._lock:
            return self._version, [p.copy() for p in self._polygons]

    def get_mask(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._mask is None else self._mask.copy()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
