"""Middle-button drag pans the page (real mouse events on the canvas).

Usage: .venv\\Scripts\\python tools\\ui_pan_smoke.py <folder with page images>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _isolate import isolate_settings  # noqa: E402

isolate_settings()
import tempfile  # noqa: E402

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from manga_translator.ui.canvas import MODE_SELECT  # noqa: E402
from manga_translator.ui.main_window import MainWindow  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.settings.work_dir = tempfile.mkdtemp()
    win.resize(1400, 900)
    win.show()
    win.open_folder(sys.argv[1])
    QTest.qWait(200)
    c = win.canvas
    c.fit()
    c.scale(3, 3)  # zoom in so there is room to pan
    QTest.qWait(100)
    h, v = c.horizontalScrollBar(), c.verticalScrollBar()
    h.setValue((h.minimum() + h.maximum()) // 2)
    v.setValue((v.minimum() + v.maximum()) // 2)
    before = (h.value(), v.value())
    vp = c.viewport()
    start = QPoint(vp.width() // 2, vp.height() // 2)
    QTest.mousePress(vp, Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, start)
    assert vp.cursor().shape() == Qt.CursorShape.ClosedHandCursor
    for i in range(1, 11):
        QTest.mouseMove(vp, start + QPoint(-12 * i, -8 * i))  # drag up-left by 120, 80
    QTest.mouseRelease(vp, Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, start + QPoint(-120, -80))
    after = (h.value(), v.value())
    # dragging the page up-left reveals what lies down-right: scroll values grow
    assert after[0] - before[0] >= 100 and after[1] - before[1] >= 60, (before, after)
    assert c._pan_last is None and c.mode == MODE_SELECT
    assert vp.cursor().shape() == Qt.CursorShape.ArrowCursor
    print("pan ok:", before, "->", after)
    win.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
