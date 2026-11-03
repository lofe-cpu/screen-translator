from __future__ import annotations

import numpy as np

_cams: dict[int, object] = {}
_last: dict[int, np.ndarray] = {}


class ScreenCapture:
    def __init__(self, monitor: int = 0):
        self.monitor = monitor
        self._fresh = True
        try:
            import dxcam
            if monitor not in _cams:
                _cams[monitor] = dxcam.create(output_idx=monitor, output_color="BGR")
            self._cam = _cams[monitor]
            self.width, self.height = self._cam.width, self._cam.height
            self.backend = "dxcam"
        except Exception as e:
            import mss
            print(f"[capture] dxcam недоступен ({e}), использую mss")
            self._cam = None
            self._sct = mss.mss()
            self._mon = self._sct.monitors[monitor + 1]
            self.width, self.height = self._mon["width"], self._mon["height"]
            self.backend = "mss"

    def _grab_full(self) -> tuple[np.ndarray | None, bool]:
        if self._cam is not None:
            frame = self._cam.grab()
            if frame is not None:
                _last[self.monitor] = frame
                return frame, True
            return _last.get(self.monitor), False
        frame = np.asarray(self._sct.grab(self._mon))[:, :, :3]
        return frame, True

    def grab(self, region: tuple[int, int, int, int] | None = None) -> tuple[np.ndarray | None, bool]:
        frame, changed = self._grab_full()
        if frame is None:
            return None, False
        if self._fresh:
            self._fresh, changed = False, True
        if region:
            l, t, r, b = region
            frame = frame[max(t, 0):min(b, frame.shape[0]), max(l, 0):min(r, frame.shape[1])]
        return frame, changed

    def close(self):
        pass
