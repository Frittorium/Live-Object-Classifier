import numpy as np


def is_valid_frame(frame) -> bool:
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return False
    if frame.ndim != 3 or frame.shape[2] != 3:
        return False
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return False
    return frame.dtype == np.uint8


def preprocess(frame: np.ndarray) -> np.ndarray:
    """V1: Ultralytics handles resize/normalization internally, so this is a pass-through."""
    return frame