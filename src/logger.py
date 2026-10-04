import logging
import logging.handlers
import time
from pathlib import Path

ROOT = "yolo_app"
_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(component: str) -> logging.Logger:
    return logging.getLogger(f"{ROOT}.{component}")


def init_minimal() -> None:
    root = logging.getLogger(ROOT)
    if not root.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
        root.addHandler(h)
        root.setLevel(logging.INFO)


def initialize(cfg) -> None:
    """Full logging: console + rotating file. File logging is non-critical."""
    root = logging.getLogger(ROOT)
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    root.setLevel(cfg.log_level.upper())
    root.propagate = False
    fmt = logging.Formatter(_FORMAT, _DATEFMT)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    try:
        d = Path(cfg.log_directory)
        d.mkdir(parents=True, exist_ok=True)
        fh = _SafeRotatingHandler(d / cfg.log_file, maxBytes=cfg.log_max_bytes,
                                  backupCount=cfg.log_backup_count, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError as e:
        root.warning("File logging disabled: %s", type(e).__name__)


class _SafeRotatingHandler(logging.handlers.RotatingFileHandler):
    """On write failure, disable itself and warn once via the other handlers."""
    def handleError(self, record):
        root = logging.getLogger(ROOT)
        root.removeHandler(self)
        try:
            self.close()
        except Exception:
            pass
        root.warning("File log handler failed and was disabled")


def shutdown() -> None:
    root = logging.getLogger(ROOT)
    for h in list(root.handlers):
        try:
            h.flush()
            h.close()
        except Exception:
            pass
        root.removeHandler(h)


class RateLimitedLogger:
    """Logs the first occurrence immediately, suppresses repeats, emits periodic
    summaries, and reports a summary when the condition ends."""
    def __init__(self, logger: logging.Logger, interval: float, clock=time.monotonic):
        self._log, self._interval, self._clock = logger, interval, clock
        self._state: dict[str, dict] = {}

    def failure(self, key: str, msg: str, level=logging.WARNING) -> None:
        now = self._clock()
        s = self._state.get(key)
        if s is None:
            self._state[key] = dict(count=1, first=now, last_logged=now, suppressed=0)
            self._log.log(level, msg)
            return
        s["count"] += 1
        if now - s["last_logged"] >= self._interval:
            self._log.log(level, "%s (still failing: %d occurrences over %.1fs)",
                          msg, s["count"], now - s["first"])
            s["last_logged"] = now
        else:
            s["suppressed"] += 1

    def resolved(self, key: str) -> None:
        s = self._state.pop(key, None)
        if s:
            self._log.info("%s condition ended. Occurrences: %d. Suppressed: %d. Duration: %.1fs",
                           key, s["count"], s["suppressed"], self._clock() - s["first"])