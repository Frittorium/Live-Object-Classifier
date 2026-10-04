import cv2

from errors import RenderingError
from logger import get_logger

log = get_logger("renderer")

_BOX_COLOR = (0, 200, 0)
_TEXT_COLOR = (0, 0, 0)


class Renderer:
    def __init__(self, cfg):
        self._cfg = cfg
        self._ready = False
        self._exit = False
        self._window_closed = False

    def init(self) -> None:
        try:
            cv2.namedWindow(self._cfg.window_name, cv2.WINDOW_NORMAL)
            self._ready = True
            log.info("Display initialized")
        except cv2.error as e:
            log.critical("Display initialization failed: %s", e, exc_info=True)
            raise RenderingError(str(e)) from e

    def render(self, frame, detections):
        out = frame.copy()
        for d in detections:
            p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
            cv2.rectangle(out, p1, p2, _BOX_COLOR, 2)
            label = f"{d.class_name} {d.confidence:.2f}"
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(p1[1], th + base)
            cv2.rectangle(out, (p1[0], top - th - base), (p1[0] + tw, top), _BOX_COLOR, -1)
            cv2.putText(out, label, (p1[0], top - base), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, _TEXT_COLOR, 1, cv2.LINE_AA)
        return out

    def show(self, frame) -> None:
        if not self._ready:
            raise RenderingError("renderer not initialized")
        try:
            cv2.imshow(self._cfg.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(self._cfg.exit_key.lower()) or key == ord(self._cfg.exit_key.upper()):
                self._exit = True
            if cv2.getWindowProperty(self._cfg.window_name, cv2.WND_PROP_VISIBLE) < 1:
                self._window_closed = True
                self._exit = True
                log.info("Display window closed by user")
        except cv2.error as e:
            log.error("Display failure: %s", e, exc_info=True)
            raise RenderingError(str(e)) from e

    def exit_requested(self) -> bool:
        return self._exit

    def close(self) -> None:
        if not self._ready:
            return
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except cv2.error as e:
            log.error("Window destroy error: %s", e)
        finally:
            self._ready = False
            log.info("Display closed")