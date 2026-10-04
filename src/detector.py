import time
from enum import Enum, auto
from pathlib import Path

from detection import Detection
from errors import InferenceError, ModelLoadError
from logger import get_logger

log = get_logger("detector")


class DetectorState(Enum):
    UNINITIALIZED = auto()
    READY = auto()
    FAILED = auto()
    RELEASED = auto()


class Detector:
    def __init__(self, cfg):
        self._cfg = cfg
        self._model = None
        self._device = None
        self.state = DetectorState.UNINITIALIZED
        self.consecutive_failures = 0
        self.last_inference_ms = 0.0

    def load(self) -> None:
        path = Path(self._cfg.model_path)
        try:
            from ultralytics import YOLO      # vendor import isolated here
            import torch

            self._device = self._resolve_device(torch)
            # Ultralytics auto-downloads known weight names (e.g. yolov8n.pt) if absent;
            # for any other path we require the file to exist.
            if not path.exists() and path.name not in _KNOWN_WEIGHTS:
                raise FileNotFoundError(f"Model file not found: {path}")
            self._model = YOLO(str(path))
            self._model.to(self._device)
            # warm-up so the first live frame isn't slow
            import numpy as np
            self._model.predict(np.zeros((self._cfg.camera_height, self._cfg.camera_width, 3), np.uint8),
                                device=self._device, verbose=False)
        except Exception as e:
            self.state = DetectorState.FAILED
            self._model = None
            log.critical("Model initialization failed: %s: %s", type(e).__name__, e, exc_info=True)
            raise ModelLoadError(str(e)) from e
        self.state = DetectorState.READY
        log.info("YOLO model loaded: %s", path.name)
        log.info("Using device: %s", self._device)

    def _resolve_device(self, torch) -> str:
        want = self._cfg.device
        if want == "cpu":
            return "cpu"
        cuda_ok = torch.cuda.is_available()
        if want == "cuda":
            if not cuda_ok:
                raise RuntimeError("CUDA requested but unavailable")
            return "cuda"
        return "cuda" if cuda_ok else "cpu"      # auto (deterministic and logged)

    def is_ready(self) -> bool:
        return self.state == DetectorState.READY and self._model is not None

    def detect(self, frame) -> list[Detection]:
        """Returns application-owned Detections. Raises InferenceError on failure."""
        if not self.is_ready():
            raise InferenceError("detector not ready")
        t0 = time.perf_counter()
        try:
            # Low model-level threshold; app-level threshold is applied in detection.py.
            results = self._model.predict(frame, device=self._device, conf=0.01, verbose=False)
            dets = self._convert(results)
        except Exception as e:
            self.consecutive_failures += 1
            log.error("Inference failed (%d consecutive): %s: %s",
                      self.consecutive_failures, type(e).__name__, e)
            if self.consecutive_failures >= self._cfg.max_inference_failures:
                self.state = DetectorState.FAILED
                log.critical("Detector unusable after %d consecutive failures", self.consecutive_failures)
            raise InferenceError(str(e)) from e
        self.consecutive_failures = 0
        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
        return dets

    def _convert(self, results) -> list[Detection]:
        out: list[Detection] = []
        if not results:
            return out
        r = results[0]
        names = r.names
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            k = int(k)
            out.append(Detection(k, str(names.get(k, "")), float(c),
                                 float(x1), float(y1), float(x2), float(y2)))
        return out

    def release(self) -> None:
        if self.state == DetectorState.RELEASED:
            return
        try:
            self._model = None
            if self._device == "cuda":
                import torch
                torch.cuda.empty_cache()
        except Exception as e:
            log.error("Detector release error: %s", type(e).__name__)
        self.state = DetectorState.RELEASED
        log.info("Detector released")


_KNOWN_WEIGHTS = {f"yolov8{s}.pt" for s in "nmlxs"} | {f"yolo11{s}.pt" for s in "nmlxs"}