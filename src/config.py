from dataclasses import dataclass
import logging

from errors import ConfigurationError


@dataclass(frozen=True)
class Config:
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 0                 # 0 = camera default

    model_path: str = "models/yolov8n.pt"
    device: str = "cpu"                 # cpu | cuda | auto

    confidence_threshold: float = 0.50
    exit_key: str = "q"
    window_name: str = "YOLO Object Detection"

    camera_max_read_failures: int = 5
    camera_retry_delay: float = 0.1
    camera_reconnect_attempts: int = 3
    camera_reconnect_delay: float = 1.0
    max_invalid_frames: int = 30

    max_inference_failures: int = 5

    log_level: str = "INFO"
    log_directory: str = "logs"
    log_file: str = "app.log"
    log_max_bytes: int = 5_000_000
    log_backup_count: int = 3
    log_suppress_interval: float = 10.0

    performance_logging_enabled: bool = True
    performance_log_interval: float = 5.0
    debug_detection_logging: bool = False


def validate(c: Config) -> None:
    errs = []
    if c.camera_index < 0: errs.append("camera_index must be >= 0")
    if c.camera_width <= 0 or c.camera_height <= 0: errs.append("camera resolution must be > 0")
    if not 0.0 <= c.confidence_threshold <= 1.0: errs.append("confidence_threshold must be in [0, 1]")
    for name in ("camera_max_read_failures", "camera_reconnect_attempts",
                 "max_invalid_frames", "max_inference_failures"):
        if getattr(c, name) < 0: errs.append(f"{name} must be >= 0")
    for name in ("camera_retry_delay", "camera_reconnect_delay", "performance_log_interval"):
        if getattr(c, name) < 0: errs.append(f"{name} must be >= 0")
    if not c.model_path.strip(): errs.append("model_path must not be empty")
    if c.device not in ("cpu", "cuda", "auto"): errs.append("device must be cpu, cuda or auto")
    if not isinstance(logging.getLevelName(c.log_level.upper()), int): errs.append("invalid log_level")
    if len(c.exit_key) != 1: errs.append("exit_key must be a single character")
    if errs:
        raise ConfigurationError("; ".join(errs))