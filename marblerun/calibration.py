"""Projector-to-camera calibration via a manual registration-grid homography.

The projector and camera see the play area from different positions, scales and
angles. A homography is a 3x3 matrix that maps any point in the *camera* image
to the corresponding point on the *display* canvas (and vice versa via its
inverse). With it, an object the camera detects at camera-pixel (cx, cy) can be
placed in the physics world at display-pixel (dx, dy) so the marble bounces off
exactly where the projected light lands.

Workflow (driven by main.py):
  1. The projector draws a grid of numbered registration marks at known display
     positions (returned by `display_grid_points`).
  2. The operator clicks those marks, in order, in the live camera window opened
     by `collect_camera_points` (a magnifier loupe aids precision).
  3. `compute_homography` fits H (camera -> display) by least squares over all
     points and `reprojection_error` reports the quality; H is saved to JSON.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config

# Corner order used everywhere: top-left, top-right, bottom-right, bottom-left.
CORNER_LABELS = ("1: TOP-LEFT", "2: TOP-RIGHT", "3: BOTTOM-RIGHT", "4: BOTTOM-LEFT")


def display_target_points(
    width: int = config.DISPLAY_WIDTH,
    height: int = config.DISPLAY_HEIGHT,
    margin_frac: float = 0.12,
) -> np.ndarray:
    """Marker positions on the display, inset from the edges by `margin_frac`.

    Insetting matters because projector edges are often clipped or keystoned;
    keeping the markers inside the frame makes them easy to click accurately.
    """
    mx = width * margin_frac
    my = height * margin_frac
    return np.array(
        [
            [mx, my],                      # top-left
            [width - mx, my],              # top-right
            [width - mx, height - my],     # bottom-right
            [mx, height - my],             # bottom-left
        ],
        dtype=np.float32,
    )


def display_grid_points(
    rows: int = 3,
    cols: int = 3,
    width: int = config.DISPLAY_WIDTH,
    height: int = config.DISPLAY_HEIGHT,
    margin_frac: float = 0.12,
) -> Tuple[np.ndarray, List[str]]:
    """A grid of registration marks (row-major) plus their numeric labels.

    Using more than 4 points lets the homography be solved by least squares, so
    small clicking errors at each mark average out and the overall alignment is
    much more accurate than 4 corners alone.
    """
    mx = width * margin_frac
    my = height * margin_frac
    xs = np.linspace(mx, width - mx, cols)
    ys = np.linspace(my, height - my, rows)
    pts = []
    labels = []
    n = 1
    for y in ys:
        for x in xs:
            pts.append([x, y])
            labels.append(str(n))
            n += 1
    return np.array(pts, dtype=np.float32), labels


def default_homography(
    camera_size: Tuple[int, int],
    display_size: Tuple[int, int] = (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT),
) -> np.ndarray:
    """A fallback mapping that stretches the whole camera frame onto the display.

    Used before the operator has calibrated so detected objects still reach the
    physics world (alignment will be approximate until a real 4-corner
    calibration is done). `camera_size` is (width, height).
    """
    cw, ch = camera_size
    dw, dh = display_size
    cam = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float32)
    disp = np.array([[0, 0], [dw, 0], [dw, dh], [0, dh]], dtype=np.float32)
    return cv2.getPerspectiveTransform(cam, disp)


def compute_homography(
    camera_points: np.ndarray, display_points: np.ndarray
) -> np.ndarray:
    """Return the 3x3 homography mapping camera pixels -> display pixels.

    With exactly 4 points an exact transform is used; with more, a least-squares
    fit (averaging out per-click error) is used.
    """
    cam = np.asarray(camera_points, dtype=np.float32).reshape(-1, 2)
    disp = np.asarray(display_points, dtype=np.float32).reshape(-1, 2)
    if cam.shape[0] != disp.shape[0] or cam.shape[0] < 4:
        raise ValueError("Need at least 4 matched camera/display points.")
    if cam.shape[0] == 4:
        return cv2.getPerspectiveTransform(cam, disp)
    H, _ = cv2.findHomography(cam, disp, method=0)
    if H is None:
        raise ValueError("Homography could not be computed from these points.")
    return H


def reprojection_error(
    H: np.ndarray, camera_points: np.ndarray, display_points: np.ndarray
) -> float:
    """Mean pixel distance between warped camera points and their display targets.

    A quality score for the calibration: lower is better (a few px is excellent).
    """
    warped = warp_points(H, camera_points)
    disp = np.asarray(display_points, dtype=np.float32).reshape(-1, 2)
    return float(np.sqrt(((warped - disp) ** 2).sum(axis=1)).mean())


def warp_points(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply homography H to an (N, 2) array of points, returning (N, 2)."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if pts.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)


def save_homography(H: np.ndarray, path: str = config.CALIBRATION_FILE) -> None:
    data = {
        "homography": np.asarray(H, dtype=float).tolist(),
        "display_width": config.DISPLAY_WIDTH,
        "display_height": config.DISPLAY_HEIGHT,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_homography(path: str = config.CALIBRATION_FILE) -> Optional[np.ndarray]:
    """Load a saved homography, or None if no valid calibration exists."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        H = np.array(data["homography"], dtype=np.float64)
        if H.shape == (3, 3):
            return H
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return None


# Arrow-key codes from cv2.waitKeyEx on Windows (for fine nudging).
_ARROW_LEFT = 2424832
_ARROW_UP = 2490368
_ARROW_RIGHT = 2555904
_ARROW_DOWN = 2621440


def _draw_loupe(view: np.ndarray, center: Tuple[int, int], zoom: int = 6,
                size: int = 70, out: int = 200) -> None:
    """Draw a magnified inset of the area around `center` for precise clicking."""
    h, w = view.shape[:2]
    cx, cy = center
    x0 = max(0, min(w - size, cx - size // 2))
    y0 = max(0, min(h - size, cy - size // 2))
    roi = view[y0:y0 + size, x0:x0 + size].copy()
    if roi.size == 0:
        return
    mag = cv2.resize(roi, (out, out), interpolation=cv2.INTER_NEAREST)
    # Crosshair at the magnified cursor location.
    rel_x = int((cx - x0) / size * out)
    rel_y = int((cy - y0) / size * out)
    cv2.line(mag, (rel_x, 0), (rel_x, out), (0, 255, 255), 1)
    cv2.line(mag, (0, rel_y), (out, rel_y), (0, 255, 255), 1)
    cv2.rectangle(mag, (0, 0), (out - 1, out - 1), (0, 255, 255), 2)
    # Place the loupe in the corner opposite the cursor so it never hides it.
    px = 10 if cx > w // 2 else w - out - 10
    py = 10 if cy > h // 2 else h - out - 10
    view[py:py + out, px:px + out] = mag


def collect_camera_points(
    camera,
    n_points: int,
    labels: Optional[List[str]] = None,
    window_name: str = "Calibration - click each numbered mark (magnifier aids precision)",
) -> Optional[np.ndarray]:
    """Collect `n_points` clicked points (with a magnifier) in label order.

    Click each numbered registration mark in order. A magnifier loupe follows the
    cursor for pixel-precise clicks. After placing a point you can nudge the most
    recent one with the arrow keys. Returns an (N, 2) float32 array, or None if
    cancelled. `camera` must expose `read() -> (frame_id, frame)`.
    """
    clicks: List[List[int]] = []
    mouse = [0, 0]

    def on_mouse(event: int, x: int, y: int, flags: int, _param) -> None:
        mouse[0], mouse[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < n_points:
            clicks.append([x, y])

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            _id, frame = camera.read()
            if frame is None:
                continue
            view = frame.copy()

            for i, (px, py) in enumerate(clicks):
                cv2.circle(view, (px, py), 6, (0, 255, 0), -1)
                cv2.circle(view, (px, py), 12, (0, 255, 0), 1)
                lbl = labels[i] if labels else str(i + 1)
                cv2.putText(view, lbl, (px + 12, py + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if len(clicks) < n_points:
                nxt = labels[len(clicks)] if labels else str(len(clicks) + 1)
                prompt = f"Click mark {nxt}  ({len(clicks)}/{n_points})"
            else:
                prompt = "ENTER=confirm  R=redo last  arrows=nudge  ESC=cancel"
            cv2.putText(view, prompt, (20, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            _draw_loupe(view, (mouse[0], mouse[1]))

            cv2.imshow(window_name, view)
            key = cv2.waitKeyEx(20)
            if key == -1:
                continue
            low = key & 0xFF
            if low == 27:  # ESC
                return None
            if low in (ord("r"), ord("R")) and clicks:
                clicks.pop()
            elif low in (13, 10) and len(clicks) == n_points:  # ENTER
                return np.array(clicks, dtype=np.float32)
            elif clicks:  # arrow-key nudge of the most recent point
                if key == _ARROW_LEFT:
                    clicks[-1][0] -= 1
                elif key == _ARROW_RIGHT:
                    clicks[-1][0] += 1
                elif key == _ARROW_UP:
                    clicks[-1][1] -= 1
                elif key == _ARROW_DOWN:
                    clicks[-1][1] += 1
    finally:
        cv2.destroyWindow(window_name)


def calibrate(
    camera,
    display_points: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    save: bool = True,
) -> Optional[Tuple[np.ndarray, float]]:
    """Full interactive calibration.

    Returns (homography, reprojection_error_px), or None if cancelled. Assumes
    the projector is already showing the registration marks at `display_points`.
    """
    if display_points is None:
        display_points, labels = display_grid_points()
    cam_pts = collect_camera_points(camera, len(display_points), labels)
    if cam_pts is None:
        return None
    H = compute_homography(cam_pts, display_points)
    err = reprojection_error(H, cam_pts, display_points)
    if save:
        save_homography(H)
    return H, err
