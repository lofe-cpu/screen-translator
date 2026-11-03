from __future__ import annotations

import ctypes

from PySide6.QtCore import QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ocr import Block

user32 = ctypes.windll.user32
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
WDA_EXCLUDEFROMCAPTURE = 0x11


def make_click_through(hwnd: int) -> bool:
    style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE,
                          style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)
    return bool(user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE))


class Overlay(QWidget):
    def __init__(self, screen, opacity: float = 0.92):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                         | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setScreen(screen)
        self.setGeometry(screen.geometry())
        self._dpr = screen.devicePixelRatio()
        self._alpha = int(255 * opacity)
        self._blocks: list[Block] = []
        self._status = ""
        self.font_family = "Segoe UI"
        self._fit_cache: dict[tuple[str, int, int], tuple[int, float]] = {}

    def showEvent(self, e):
        super().showEvent(e)
        if not make_click_through(int(self.winId())):
            print("[overlay] SetWindowDisplayAffinity не сработал — оверлей может попадать в захват")

    def set_blocks(self, blocks: list[Block]):
        self._blocks = blocks
        self.update()

    def set_opacity(self, opacity: float):
        self._alpha = int(255 * opacity)
        self.update()

    def set_status(self, text: str):
        self._status = text
        self.update()

    def _fit_font(self, text: str, rect: QRectF) -> tuple[QFont, QRectF]:
        key = (text, round(rect.width()), round(rect.height()))
        hit = self._fit_cache.get(key)
        if hit is None:
            hit = self._fit_cache[key] = self._compute_fit(text, QRectF(0, 0, rect.width(), rect.height()))
            if len(self._fit_cache) > 2000:
                self._fit_cache.clear()
        font = QFont(self.font_family)
        font.setPixelSize(hit[0])
        return font, QRectF(rect.x(), rect.y(), rect.width(), hit[1])

    def _compute_fit(self, text: str, rect: QRectF) -> tuple[int, float]:
        font = QFont(self.font_family)
        flags = int(Qt.TextWordWrap)
        lo, hi = 6.0, max(6.0, min(rect.height() * 0.8, 72.0))
        best = lo
        while hi - lo > 0.5:
            mid = (lo + hi) / 2
            font.setPixelSize(int(mid))
            br = QFontMetricsF(font).boundingRect(rect, flags, text)
            if br.height() <= rect.height() and br.width() <= rect.width() + 1:
                best, lo = mid, mid
            else:
                hi = mid
        font.setPixelSize(int(best))
        br = QFontMetricsF(font).boundingRect(rect, flags, text)
        return int(best), max(rect.height(), br.height())

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        k = 1.0 / self._dpr
        for b in self._blocks:
            if not b.translated:
                continue
            pad = 2
            rect = QRectF(b.x * k - pad, b.y * k - pad, b.w * k + 2 * pad, b.h * k + 2 * pad)
            font, rect = self._fit_font(b.translated, rect)
            bg = QColor(*b.bg)
            bg.setAlpha(self._alpha)
            p.fillRect(rect, bg)
            p.setFont(font)
            p.setPen(QPen(QColor(*b.fg)))
            p.drawText(rect, int(Qt.TextWordWrap | Qt.AlignLeft | Qt.AlignVCenter), b.translated)
        if self._status:
            f = QFont(self.font_family)
            f.setPixelSize(13)
            p.setFont(f)
            r = QRect(8, 8, 420, 22)
            p.fillRect(r, QColor(20, 20, 20, 200))
            p.setPen(QColor(120, 255, 160))
            p.drawText(r.adjusted(6, 0, 0, 0), int(Qt.AlignVCenter), self._status)
        p.end()
