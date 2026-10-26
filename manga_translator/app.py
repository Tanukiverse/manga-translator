"""Entry point: python -m manga_translator.app [image folder]"""
from __future__ import annotations

import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from .core.typeset import ensure_qt
from .ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Manga Translator")
    app.setFont(QFont("Leelawadee UI", 10))
    ensure_qt()  # register bundled fonts
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1:
        win.open_folder(sys.argv[1])
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
