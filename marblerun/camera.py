"""Threaded webcam capture.

Grabbing frames from a webcam can block for tens of milliseconds. To keep the
60 FPS render/physics loop smooth we read frames on a background thread and just
hand the newest one to whoever asks for it.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

import config

log = logging.getLogger("marblerun.camera")


def _resolve_backend(name: Optional[str]) -> int:
    if name == "dshow":
        return cv2.CAP_DSHOW
    if name == "msmf":
        return cv2.CAP_MSMF
    return cv2.CAP_ANY


class CameraThread:
    """Continuously reads frames from a camera in a background thread."""

    def __init__(
        self,
        index: int = config.CAMERA_INDEX,
        width: int = config.CAMERA_WIDTH,
        height: int = config.CAMERA_HEIGHT,
        backend: Optional[str] = config.CAMERA_BACKEND,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.backend = backend

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "CameraThread":
        self._cap = cv2.VideoCapture(self.index, _resolve_backend(self.backend))
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {self.index}. "
                "Check it is connected and not in use by another app."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Prime the pump: block until we have the first frame so callers do not
        # have to special-case a None frame on startup.
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("Camera opened but returned no frames.")
        self._frame = frame
        self._frame_id = 1

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        assert self._cap is not None
        while not self._stop.is_set():
            try:
                ok, frame = self._cap.read()
            except Exception:
                log.exception("camera read failed")
                time.sleep(0.05)
                continue
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self._frame_id += 1

    @property
    def resolution(self) -> tuple[int, int]:
        """Actual capture resolution (may differ from requested)."""
        if self._cap is None:
            return self.width, self.height
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        return w, h

    def read(self) -> tuple[int, Optional[np.ndarray]]:
        """Return (frame_id, frame_copy). frame_id lets callers skip stale frames."""
        with self._lock:
            if self._frame is None:
                return 0, None
            return self._frame_id, self._frame.copy()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
