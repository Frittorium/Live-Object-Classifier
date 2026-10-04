import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np

from logger import get_logger

log = get_logger("camera")


class CameraState(Enum):
    CLOSED = auto()
    OPENING = auto()
    OPEN = auto()
    READ_FAILURE = auto()
    RECOVERING = auto()
    FAILED = auto()
    RELEASING = auto()


@dataclass
class FrameResult:
    success: bool
    frame: Optional[np.ndarray] = None
    reason: str = ""


class Camera:
    def __init__(self, cfg, sleep=time.sleep):
        self._cfg = cfg
        self._sleep = sleep
        self._cap = None
        self.state = CameraState.CLOSED
        self.consecutive_read_failures = 0

    def open(self) -> bool:
        self.state = CameraState.OPENING
        try:
            cap = cv2.VideoCapture(self._cfg.camera_index)
            if not cap.isOpened():
                cap.release()
                log.error("Camera %d could not be opened (missing or in use)", self._cfg.camera_index)
                self.state = CameraState.FAILED
                return False
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.camera_height)
            if self._cfg.camera_fps > 0:
                cap.set(cv2.CAP_PROP_FPS, self._cfg.camera_fps)
            self._cap = cap
            self.consecutive_read_failures = 0
            self.state = CameraState.OPEN
            log.info("Camera %d opened (%dx%d)", self._cfg.camera_index,
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            return True
        except cv2.error as e:
            log.error("Camera initialization exception: %s", e, exc_info=True)
            self._safe_release_capture()
            self.state = CameraState.FAILED
            return False

    def is_open(self) -> bool:
        return self._cap is not None and self.state == CameraState.OPEN and self._cap.isOpened()

    def read(self) -> FrameResult:
        if self._cap is None:
            return FrameResult(False, None, "camera not open")
        try:
            ok, frame = self._cap.read()
        except cv2.error as e:
            ok, frame = False, None
            log.debug("cv2 error during read: %s", e)
        if ok and frame is not None:
            return FrameResult(True, frame)
        self.consecutive_read_failures += 1
        self.state = CameraState.READ_FAILURE
        return FrameResult(False, None, f"read failed ({self.consecutive_read_failures} consecutive)")

    def note_read_success(self) -> None:
        """Called by the controller after a frame is confirmed usable."""
        if self.consecutive_read_failures:
            self.consecutive_read_failures = 0
        if self.state == CameraState.READ_FAILURE:
            self.state = CameraState.OPEN

    def read_failure_limit_reached(self) -> bool:
        return self.consecutive_read_failures >= self._cfg.camera_max_read_failures

    def retry_delay(self) -> None:
        self._sleep(self._cfg.camera_retry_delay)

    def reconnect(self) -> bool:
        """Bounded reconnection. Returns True only once a valid frame is confirmed."""
        self.state = CameraState.RECOVERING
        self._safe_release_capture()
        for attempt in range(1, self._cfg.camera_reconnect_attempts + 1):
            log.debug("Camera reconnect attempt %d/%d", attempt, self._cfg.camera_reconnect_attempts)
            self._sleep(self._cfg.camera_reconnect_delay)
            if self.open():
                res = self.read()
                if res.success:
                    self.consecutive_read_failures = 0
                    self.state = CameraState.OPEN
                    log.info("Camera recovered after %d attempt(s)", attempt)
                    return True
                self._safe_release_capture()
        log.error("Camera recovery failed after %d attempts", self._cfg.camera_reconnect_attempts)
        self.state = CameraState.FAILED
        return False

    def release(self) -> None:
        if self._cap is None and self.state in (CameraState.CLOSED, CameraState.FAILED):
            return
        self.state = CameraState.RELEASING
        self._safe_release_capture()
        self.state = CameraState.CLOSED
        log.info("Camera released")

    def _safe_release_capture(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:  # cleanup must not raise
                log.error("Camera release error: %s", type(e).__name__)
            finally:
                self._cap = None