import math
from dataclasses import dataclass
from typing import Iterable

from logger import get_logger

log = get_logger("detection")


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


def _finite_numbers(*vals) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)


def process_detections(raw: Iterable[Detection], frame_width: int, frame_height: int,
                       confidence_threshold: float) -> list[Detection]:
    out: list[Detection] = []
    for d in raw:
        if not isinstance(d.class_id, int) or d.class_id < 0 or not d.class_name:
            log.debug("Discarded detection: invalid class info")
            continue
        if not _finite_numbers(d.confidence) or not 0.0 <= d.confidence <= 1.0:
            log.debug("Discarded detection: invalid confidence")
            continue
        if d.confidence < confidence_threshold:
            continue
        if not _finite_numbers(d.x1, d.y1, d.x2, d.y2) or d.x1 > d.x2 or d.y1 > d.y2:
            log.debug("Discarded detection: malformed box")
            continue
        x1, y1 = max(0.0, d.x1), max(0.0, d.y1)
        x2, y2 = min(float(frame_width), d.x2), min(float(frame_height), d.y2)
        if x2 <= x1 or y2 <= y1:
            log.debug("Discarded detection: zero-area or fully out-of-frame box")
            continue
        out.append(Detection(d.class_id, d.class_name, d.confidence, x1, y1, x2, y2))
    return out