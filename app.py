from __future__ import annotations

import importlib.util

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QMenu, QMessageBox, QPushButton, QRadioButton, QSlider,
                               QSystemTrayIcon, QToolButton, QVBoxLayout, QWidget)

from config import Settings, save_settings
from overlay import Overlay
from pipeline import Pipeline
from selector import RegionSelector

SRC_LANGS = [
    ("auto", "Автоопределение"),
    ("en", "Английский"), ("ja", "Японский"), ("zh", "Китайский"), ("ko", "Корейский"),
    ("de", "Немецкий"), ("fr", "Французский"), ("es", "Испанский"), ("it", "Итальянский"),
    ("pt", "Португальский"), ("pl", "Польский"), ("tr", "Турецкий"), ("nl", "Нидерландский"),
    ("vi", "Вьетнамский"), ("id", "Индонезийский"), ("ru", "Русский"), ("uk", "Украинский"),
    ("kk", "Казахский"), ("ar", "Арабский"), ("hi", "Хинди"), ("th", "Тайский"), ("el", "Греческий"),
]
DST_LANGS = [
    ("ru", "Русский"), ("en", "Английский"), ("uk", "Украинский"), ("kk", "Казахский"),
    ("ky", "Кыргызский"), ("uz", "Узбекский"), ("de", "Немецкий"), ("fr", "Французский"),
    ("es", "Испанский"), ("it", "Итальянский"), ("pt", "Португальский"), ("pl", "Польский"),
    ("tr", "Турецкий"), ("ja", "Японский"), ("zh-CN", "Китайский"), ("ko", "Корейский"),
    ("ar", "Арабский"),
]
HAS_ARGOS = importlib.util.find_spec("argostranslate") is not None


def app_icon() -> QIcon:
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor(40, 140, 255))
        p.setPen(Qt.NoPen)
        m = size / 16
        p.drawRoundedRect(m, m, size - 2 * m, size - 2 * m, size / 5, size / 5)
        f = QFont("Segoe UI", 1, QFont.Bold)
        f.setPixelSize(int(size * 0.42))
        p.setFont(f)
        p.setPen(QColor(255, 255, 255))
        p.drawText(pm.rect(), int(Qt.AlignCenter), "文A")
        p.end()
        icon.addPixmap(pm)
    return icon


class Hotkeys(QObject):
    toggle = Signal()
    region = Signal()
    full = Signal()
    peek = Signal(bool)

    def __init__(self, s: Settings):
        super().__init__()
        try:
            import keyboard
            keyboard.add_hotkey(s.hotkey_toggle, self.toggle.emit)
            keyboard.add_hotkey(s.hotkey_region, self.region.emit)
            keyboard.add_hotkey(s.hotkey_full, self.full.emit)
            keyboard.add_hotkey(s.hotkey_peek, lambda: self.peek.emit(True), trigger_on_release=False)
            keyboard.add_hotkey(s.hotkey_peek, lambda: self.peek.emit(False), trigger_on_release=True)
        except Exception as e:
            print(f"[hotkeys] горячие клавиши недоступны: {e}")


class Controller(QObject):
    state_changed = Signal(str, str)
    region_changed = Signal()
    region_cancelled = Signal()

    def __init__(self, s: Settings):
        super().__init__()
        self.s = s
        self.pipe: Pipeline | None = None
        self._dying: list[Pipeline] = []
        self.overlay: Overlay | None = None
        self.selector: RegionSelector | None = None
        self._build_screen_widgets()

    @property
    def running(self) -> bool:
        return self.pipe is not None

    def _screen(self):
        screens = QApplication.screens()
        return screens[self.s.monitor] if self.s.monitor < len(screens) else QApplication.primaryScreen()

    def _build_screen_widgets(self):
        for w in (self.overlay, self.selector):
            if w is not None:
                w.close()
                w.deleteLater()
        screen = self._screen()
        self.overlay = Overlay(screen, self.s.opacity)
        self.selector = RegionSelector(screen)
        self.selector.selected.connect(self._region_selected)
        self.selector.cancelled.connect(self._region_cancelled)

    def save(self):
        save_settings(self.s)

    def start(self):
        if self.pipe is not None:
            return
        self.pipe = p = Pipeline(self.s)
        p.blocks_ready.connect(self.overlay.set_blocks)
        p.status.connect(lambda t: self.state_changed.emit("loading", t) if t else None)
        p.ready.connect(lambda dev: self.state_changed.emit("on", f"Перевод включён · распознавание: {dev}"))
        p.failed.connect(self._failed)
        self.overlay.show()
        self.state_changed.emit("loading", "Загрузка…")
        p.start()

    def stop(self, message: str = "Перевод выключен"):
        p, self.pipe = self.pipe, None
        if p is not None:
            p.blocks_ready.disconnect()
            p._running = False
            self._dying.append(p)
            p.finished.connect(lambda p=p: self._dying.remove(p) if p in self._dying else None)
        self.overlay.set_blocks([])
        self.overlay.hide()
        self.state_changed.emit("off", message)

    def toggle(self):
        self.stop() if self.running else self.start()

    def restart(self):
        self.save()
        if self.running:
            self.stop()
            self.start()

    def _failed(self, err: str):
        self.stop()
        self.state_changed.emit("error", f"Не удалось запустить: {err}")

    def set_opacity(self, value: float):
        self.s.opacity = value
        self.overlay.set_opacity(value)
        self.save()

    def set_monitor(self, idx: int):
        if idx == self.s.monitor:
            return
        was = self.running
        self.stop()
        self.s.monitor, self.s.region = idx, None
        self._build_screen_widgets()
        self.save()
        self.region_changed.emit()
        if was:
            self.start()

    def full_screen(self):
        self.s.region = None
        self.save()
        self.region_changed.emit()
        if self.pipe:
            self.pipe.set_region(None)
        else:
            self.start()

    def pick_region(self):
        self.overlay.hide()
        self.selector.start()

    def _region_selected(self, region):
        self.s.region = tuple(region) if region else None
        self.save()
        self.region_changed.emit()
        if self.pipe:
            self.pipe.set_region(self.s.region)
            self.overlay.show()
        else:
            self.start()

    def _region_cancelled(self):
        if self.running:
            self.overlay.show()
        self.region_cancelled.emit()

    def peek(self, down: bool):
        if self.running:
            self.overlay.setVisible(not down)

    def shutdown(self):
        self.stop()
        for p in list(self._dying):
            p.wait(5000)


class MainWindow(QWidget):
    def __init__(self, ctl: Controller, quit_app):
        super().__init__()
        self.ctl, self.s = ctl, ctl.s
        self._quit_app = quit_app
        self._tray_hint_shown = False
        self.setWindowTitle("Экранный переводчик")
        self.setWindowIcon(app_icon())
        self.setMinimumWidth(420)

        root = QVBoxLayout(self)
        root.setSpacing(12)

        self.btn = QPushButton()
        self.btn.setMinimumHeight(48)
        f = self.btn.font()
        f.setPointSize(12)
        f.setBold(True)
        self.btn.setFont(f)
        self.btn.clicked.connect(ctl.toggle)
        root.addWidget(self.btn)
        self.status = QLabel()
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        langs = QGroupBox("Языки")
        lay = QHBoxLayout(langs)
        self.src = QComboBox()
        for code, name in SRC_LANGS:
            self.src.addItem(name, code)
        self.src.setToolTip("Язык текста на экране. «Автоопределение» понимает китайский, японский, "
                            "английский и европейские языки на латинице.\nДля русского, корейского, "
                            "арабского и т.п. выберите язык явно.")
        swap = QToolButton()
        swap.setText("⇄")
        swap.setToolTip("Поменять местами")
        swap.clicked.connect(self._swap)
        self.dst = QComboBox()
        for code, name in DST_LANGS:
            self.dst.addItem(name, code)
        lay.addWidget(self.src, 1)
        lay.addWidget(swap)
        lay.addWidget(self.dst, 1)
        root.addWidget(langs)

        area = QGroupBox("Что переводить")
        lay = QVBoxLayout(area)
        self.rb_full = QRadioButton("Весь экран")
        self.rb_region = QRadioButton("Только выбранную область")
        grp = QButtonGroup(self)
        grp.addButton(self.rb_full)
        grp.addButton(self.rb_region)
        self.rb_full.clicked.connect(ctl.full_screen)
        self.rb_region.clicked.connect(self._region_clicked)
        row = QHBoxLayout()
        row.addWidget(self.rb_region)
        self.btn_region = QPushButton("Выбрать область…")
        self.btn_region.clicked.connect(self._pick_region)
        row.addWidget(self.btn_region)
        row.addStretch()
        lay.addWidget(self.rb_full)
        lay.addLayout(row)
        self.monitor = QComboBox()
        screens = QApplication.screens()
        for i, sc in enumerate(screens):
            g = sc.geometry()
            self.monitor.addItem(f"Монитор {i + 1}  ({g.width()}×{g.height()})", i)
        if len(screens) > 1:
            lay.addWidget(self.monitor)
        root.addWidget(area)

        more = QGroupBox("Настройки")
        form = QFormLayout(more)
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(40, 100)
        self.opacity.valueChanged.connect(lambda v: ctl.set_opacity(v / 100))
        form.addRow("Непрозрачность перевода", self.opacity)
        self.translator = QComboBox()
        self.translator.addItem("Google (нужен интернет)", "google")
        if HAS_ARGOS:
            self.translator.addItem("Argos (без интернета)", "argos")
        form.addRow("Переводчик", self.translator)
        self.device = QComboBox()
        self.device.addItem("Видеокарта, если есть (быстрее)", True)
        self.device.addItem("Только процессор", False)
        form.addRow("Распознавание", self.device)
        self.autostart = QCheckBox("Включать перевод сразу при запуске")
        form.addRow(self.autostart)
        root.addWidget(more)

        keys = QLabel(
            f"<b>Горячие клавиши</b><br>"
            f"{self._k(self.s.hotkey_toggle)} — включить / выключить<br>"
            f"{self._k(self.s.hotkey_region)} — выбрать область<br>"
            f"{self._k(self.s.hotkey_full)} — весь экран<br>"
            f"{self._k(self.s.hotkey_peek)} (удерживать) — показать оригинал")
        keys.setStyleSheet("color: #8a8a8a;")
        root.addWidget(keys)

        bottom = QHBoxLayout()
        bottom.addStretch()
        quit_btn = QPushButton("Выйти из программы")
        quit_btn.clicked.connect(quit_app)
        bottom.addWidget(quit_btn)
        root.addLayout(bottom)

        self._load_values()
        self.src.currentIndexChanged.connect(self._lang_changed)
        self.dst.currentIndexChanged.connect(self._lang_changed)
        self.translator.currentIndexChanged.connect(self._engine_changed)
        self.device.currentIndexChanged.connect(self._engine_changed)
        self.monitor.currentIndexChanged.connect(lambda: ctl.set_monitor(self.monitor.currentData()))
        self.autostart.toggled.connect(self._autostart_changed)
        ctl.state_changed.connect(self._on_state)
        ctl.region_changed.connect(self._sync_region)
        self._was_visible = False
        ctl.region_cancelled.connect(lambda: (self._sync_region(), self._was_visible and self.show_front()))
        self._on_state("off", "Перевод выключен. Нажмите кнопку выше или "
                       f"{self._k(self.s.hotkey_toggle)}.")

    @staticmethod
    def _k(hotkey: str) -> str:
        return "+".join(p.capitalize() for p in hotkey.split("+"))

    @staticmethod
    def _select(combo: QComboBox, value):
        i = combo.findData(value)
        if i >= 0:
            combo.setCurrentIndex(i)

    def _load_values(self):
        self._select(self.src, self.s.src)
        self._select(self.dst, self.s.dst)
        self._select(self.translator, self.s.translator)
        self._select(self.device, self.s.gpu)
        self._select(self.monitor, self.s.monitor)
        self.opacity.setValue(int(self.s.opacity * 100))
        self.autostart.setChecked(self.s.autostart)
        self._sync_region()

    def _sync_region(self):
        (self.rb_region if self.s.region else self.rb_full).setChecked(True)

    def _swap(self):
        src, dst = self.src.currentData(), self.dst.currentData()
        dst_as_src = "zh" if dst == "zh-CN" else dst
        src_as_dst = "zh-CN" if src == "zh" else src
        if self.src.findData(dst_as_src) < 0 or self.dst.findData(src_as_dst) < 0:
            return
        self.src.blockSignals(True)
        self._select(self.src, dst_as_src)
        self.src.blockSignals(False)
        self._select(self.dst, src_as_dst)

    def _lang_changed(self):
        self.s.src, self.s.dst = self.src.currentData(), self.dst.currentData()
        if self.s.translator == "argos" and self.s.src == "auto":
            QMessageBox.information(self, "Экранный переводчик",
                                    "Офлайн-переводчик Argos не умеет определять язык сам — "
                                    "выберите язык текста явно.")
        self.ctl.restart()

    def _engine_changed(self):
        self.s.translator = self.translator.currentData()
        self.s.gpu = self.device.currentData()
        self.ctl.restart()

    def _autostart_changed(self, on: bool):
        self.s.autostart = on
        self.ctl.save()

    def _region_clicked(self):
        if not self.s.region:
            self._pick_region()

    def _pick_region(self):
        self._was_visible = self.isVisible()
        self.hide()
        QTimer.singleShot(150, self.ctl.pick_region)

    def _on_state(self, state: str, text: str):
        running = state in ("on", "loading")
        self.btn.setText("■  Остановить перевод" if running else "▶  Начать перевод")
        bg, hover = ("#c0392b", "#d6493a") if running else ("#1f7ae0", "#3a8ef0")
        self.btn.setStyleSheet(f"QPushButton {{ background: {bg}; color: white; border: none; border-radius: 6px; }}"
                               f"QPushButton:hover {{ background: {hover}; }}")
        color = {"on": "#1a9f4b", "loading": "#c58a00", "error": "#d03030"}.get(state, "#8a8a8a")
        self.status.setStyleSheet(f"color: {color};")
        self.status.setText(text)

    def closeEvent(self, e):
        e.ignore()
        self.hide()
        if not self._tray_hint_shown and self.ctl.tray is not None:
            self._tray_hint_shown = True
            self.ctl.tray.showMessage("Экранный переводчик",
                                      "Программа продолжает работать в трее. "
                                      "Чтобы выйти — правый клик по значку → «Выход».",
                                      app_icon(), 4000)

    def show_front(self):
        self.show()
        self.setWindowState(self.windowState() & ~Qt.WindowMinimized)
        self.raise_()
        self.activateWindow()


def run_app(app: QApplication, s: Settings) -> int:
    app.setQuitOnLastWindowClosed(False)
    app.setWindowIcon(app_icon())
    ctl = Controller(s)
    ctl.tray = None

    def quit_app():
        ctl.shutdown()
        app.quit()

    win = MainWindow(ctl, quit_app)

    menu = QMenu()
    act_toggle = QAction("Перевод включён", menu, checkable=True)
    act_toggle.triggered.connect(lambda: ctl.toggle())
    menu.addAction(act_toggle)
    menu.addAction("Выбрать область…", win._pick_region)
    menu.addAction("Весь экран", ctl.full_screen)
    menu.addSeparator()
    menu.addAction("Открыть окно", win.show_front)
    menu.addAction("Выход", quit_app)
    ctl.state_changed.connect(lambda st, _: act_toggle.setChecked(st in ("on", "loading")))

    tray = QSystemTrayIcon(app_icon())
    tray.setToolTip("Экранный переводчик")
    tray.setContextMenu(menu)
    tray.activated.connect(lambda reason: win.show_front()
                           if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick) else None)
    tray.show()
    ctl.tray = tray

    keys = Hotkeys(s)
    keys.toggle.connect(ctl.toggle)
    keys.region.connect(win._pick_region)
    keys.full.connect(ctl.full_screen)
    keys.peek.connect(ctl.peek)

    win.show_front()
    if s.autostart:
        ctl.start()
    app.aboutToQuit.connect(ctl.shutdown)
    return app.exec()
