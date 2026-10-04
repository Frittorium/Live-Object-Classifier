import sys
import time
from collections import deque
from enum import Enum, auto

import config as config_mod
import logger as logger_mod
from camera import Camera
from config import Config
from detection import process_detections
from detector import Detector
from errors import (ApplicationError, ConfigurationError, InferenceError,
                    ModelLoadError, RenderingError)
from preprocessing import is_valid_frame, preprocess
from renderer import Renderer

log = logger_mod.get_logger("application")


class AppState(Enum):
    STARTING = auto()
    INITIALIZING_CAMERA = auto()
    INITIALIZING_MODEL = auto()
    RUNNING = auto()
    RECOVERING = auto()
    SHUTTING_DOWN = auto()
    STOPPED = auto()
    FAILED = auto()


class Application:
    def __init__(self, cfg: Config, camera=None, detector=None, renderer=None, clock=time.perf_counter):
        self.cfg = cfg
        self.camera = camera or Camera(cfg)
        self.detector = detector or Detector(cfg)
        self.renderer = renderer or Renderer(cfg)
        self._clock = clock
        self.state = AppState.STARTING
        self._shutdown_requested = False
        self._failed = False
        self._invalid_frames = 0
        self._rl = logger_mod.RateLimitedLogger(log, cfg.log_suppress_interval)
        # metrics
        self._frames = 0
        self._skipped = 0
        self._cam_failures = 0
        self._inf_failures = 0
        self._frame_times = deque(maxlen=300)
        self._inf_times = deque(maxlen=300)
        self._last_perf = clock()
        self._last_det_count = 0

    # ---- lifecycle -------------------------------------------------
    def start(self) -> None:
        self.state = AppState.INITIALIZING_CAMERA
        if not self.camera.open():
            self._failed = True
            raise ApplicationError("camera initialization failed")
        self.state = AppState.INITIALIZING_MODEL
        self.detector.load()                      # raises ModelLoadError
        self.renderer.init()                      # raises RenderingError
        self.state = AppState.RUNNING
        log.info("Application started")

    def request_shutdown(self, reason: str) -> None:
        if not self._shutdown_requested:
            log.info("Shutdown requested: %s", reason)
        self._shutdown_requested = True

    def run(self) -> None:
        cfg = self.cfg
        while self.state == AppState.RUNNING and not self._shutdown_requested:
            t_start = self._clock()

            res = self.camera.read()
            if not res.success:
                self._cam_failures += 1
                self._rl.failure("camera_read", f"Camera read failure: {res.reason}")
                if self.camera.read_failure_limit_reached():
                    self.state = AppState.RECOVERING
                    log.info("Beginning camera recovery")
                    if self.camera.reconnect():
                        self._rl.resolved("camera_read")
                        self.state = AppState.RUNNING
                    else:
                        self._failed = True
                        self.request_shutdown("camera recovery failed")
                        break
                else:
                    self.camera.retry_delay()
                continue

            frame = res.frame
            if not is_valid_frame(frame):
                self._skipped += 1
                self._invalid_frames += 1
                self._rl.failure("invalid_frame", "Invalid frame skipped")
                if self._invalid_frames >= cfg.max_invalid_frames:
                    log.error("Persistent invalid frames (%d); treating as camera failure", self._invalid_frames)
                    self._invalid_frames = 0
                    if not self.camera.reconnect():
                        self._failed = True
                        self.request_shutdown("persistent invalid input")
                        break
                continue
            self._invalid_frames = 0
            self._rl.resolved("invalid_frame")
            self.camera.note_read_success()
            self._rl.resolved("camera_read")

            processed = preprocess(frame)

            try:
                raw = self.detector.detect(processed)
                self._rl.resolved("inference")
            except InferenceError as e:
                self._inf_failures += 1
                self._rl.failure("inference", f"Inference failed: {e}", level=40)
                if not self.detector.is_ready():
                    self._failed = True
                    self.request_shutdown("detector unusable")
                    break
                continue

            h, w = frame.shape[:2]
            dets = process_detections(raw, w, h, cfg.confidence_threshold)
            self._last_det_count = len(dets)
            if cfg.debug_detection_logging:
                log.debug("%d valid detections", len(dets))

            try:
                self.renderer.show(self.renderer.render(frame, dets))
            except RenderingError:
                self._failed = True
                self.request_shutdown("display failure")
                break
            if self.renderer.exit_requested():
                self.request_shutdown("exit key / window closed")

            self._frames += 1
            self._frame_times.append(self._clock() - t_start)
            self._inf_times.append(self.detector.last_inference_ms / 1000.0)
            self._maybe_log_perf()

    def shutdown(self) -> None:
        if self.state == AppState.STOPPED:
            return
        self.state = AppState.SHUTTING_DOWN
        log.info("Shutdown initiated")
        for name, fn in (("camera", self.camera.release),
                         ("renderer", self.renderer.close),
                         ("detector", self.detector.release)):
            try:
                fn()
            except Exception as e:
                log.error("Cleanup error in %s: %s", name, e)
        log.info("Shutdown completed")
        self.state = AppState.FAILED if self._failed else AppState.STOPPED
        # Allow idempotence: mark stopped after final state recorded
        self._final_state = self.state
        self.state = AppState.STOPPED

    @property
    def failed(self) -> bool:
        return self._failed

    # ---- metrics ---------------------------------------------------
    def _maybe_log_perf(self) -> None:
        cfg = self.cfg
        now = self._clock()
        if not cfg.performance_logging_enabled or now - self._last_perf < cfg.performance_log_interval:
            return
        n = len(self._frame_times)
        if n:
            avg = sum(self._frame_times) / n
            fps = 1.0 / avg if avg > 0 else 0.0
            inf = sum(self._inf_times) / len(self._inf_times) * 1000
            log.info("perf: fps=%.1f frame=%.1fms inference=%.1fms detections=%d skipped=%d "
                     "cam_failures=%d inf_failures=%d",
                     fps, avg * 1000, inf, self._last_det_count,
                     self._skipped, self._cam_failures, self._inf_failures)
        self._last_perf = now


def main() -> int:
    logger_mod.init_minimal()
    app = None
    exit_code = 0
    try:
        cfg = Config()
        config_mod.validate(cfg)
        logger_mod.initialize(cfg)
        app = Application(cfg)
        app.start()
        app.run()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt: controlled shutdown")
    except ConfigurationError as e:
        log.critical("Invalid configuration: %s", e)
        exit_code = 2
    except (ModelLoadError, RenderingError, ApplicationError) as e:
        log.error("Fatal application error: %s", e)
        exit_code = 1
    except Exception:
        log.critical("Unexpected exception", exc_info=True)
        exit_code = 1
    finally:
        if app is not None:
            app.shutdown()
            if app.failed:
                exit_code = exit_code or 1
        logger_mod.shutdown()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())