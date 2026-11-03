from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


class RegionSelector(QWidget):
    selected = Signal(tuple)
    cancelled = Signal()

    def __init__(self, screen):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setScreen(screen)
        self.setGeometry(screen.geometry())
        self.setCursor(Qt.CrossCursor)
        self._dpr = screen.devicePixelRatio()
        self._a: QPoint | None = None
        self._b: QPoint | None = None

    def start(self):
        self._a = self._b = None
        self.show()
        self.activateWindow()
        self.raise_()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0, 90))
        if self._a and self._b:
            r = QRect(self._a, self._b).normalized()
            p.setCompositionMode(QPainter.CompositionMode_Clear)
            p.fillRect(r, Qt.transparent)
            p.setCompositionMode(QPainter.CompositionMode_SourceOver)
            p.setPen(QPen(QColor(80, 200, 255), 2))
            p.drawRect(r)
        p.setPen(QColor(255, 255, 255))
        p.drawText(self.rect().adjusted(0, 20, 0, 0), int(Qt.AlignHCenter | Qt.AlignTop),
                   "Выделите область для перевода  ·  Esc — отмена  ·  ПКМ — весь экран")
        p.end()

    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            self.hide()
            self.selected.emit(())
            return
        self._a = self._b = e.position().toPoint()
        self.update()

    def mouseMoveEvent(self, e):
        if self._a:
            self._b = e.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.LeftButton or not self._a:
            return
        r = QRect(self._a, e.position().toPoint()).normalized()
        self.hide()
        if r.width() < 10 or r.height() < 10:
            self.cancelled.emit()
            return
        d = self._dpr
        self.selected.emit((int(r.left() * d), int(r.top() * d), int(r.right() * d), int(r.bottom() * d)))

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.hide()
            self.cancelled.emit()
