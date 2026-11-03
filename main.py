from __future__ import annotations

import sys

from config import APP_DIR, load_settings


def _setup_log():
    if sys.stdout is not None and sys.stderr is not None:
        return
    APP_DIR.mkdir(parents=True, exist_ok=True)
    log = open(APP_DIR / "log.txt", "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log


def main():
    _setup_log()
    from PySide6.QtCore import QLockFile
    from PySide6.QtWidgets import QApplication, QMessageBox

    from app import run_app

    app = QApplication(sys.argv)
    app.setApplicationName("Экранный переводчик")

    APP_DIR.mkdir(parents=True, exist_ok=True)
    lock = QLockFile(str(APP_DIR / "running.lock"))
    if not lock.tryLock(100):
        QMessageBox.information(None, "Экранный переводчик",
                                "Программа уже запущена — её значок в трее рядом с часами.")
        return 0
    return run_app(app, load_settings())


if __name__ == "__main__":
    sys.exit(main())
