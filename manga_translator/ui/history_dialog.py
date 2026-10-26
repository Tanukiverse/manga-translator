"""Page history: pick an earlier version of the page and put it back."""
from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ..core import versions


class HistoryDialog(QDialog):
    def __init__(self, win, doc):
        super().__init__(win)
        self.win, self.doc = win, doc
        self.restored = None
        self.setWindowTitle(f"ประวัติของหน้า {doc.folder.name}")
        self.resize(560, 640)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("โปรแกรมเก็บสภาพหน้าไว้ก่อนทุกครั้งที่รันขั้นตอน รีเซ็ต หรือคืนค่า "
                             f"(เก็บล่าสุด {win.settings.versions_keep} รายการ)"))
        self.list = QListWidget()
        self.list.setIconSize(QSize(110, 150))
        self.list.setSpacing(4)
        self.list.itemDoubleClicked.connect(lambda _: self._restore())
        lay.addWidget(self.list, 1)
        row = QHBoxLayout()
        row.addStretch()
        self.restore_btn = QPushButton("คืนค่าเวอร์ชันนี้")
        self.restore_btn.clicked.connect(self._restore)
        close = QPushButton("ปิด")
        close.clicked.connect(self.reject)
        row.addWidget(self.restore_btn)
        row.addWidget(close)
        lay.addLayout(row)
        self._fill()

    def _fill(self):
        self.list.clear()
        self.items = versions.list_versions(self.doc)
        for v in self.items:
            when = datetime.fromtimestamp(v.time).strftime("%d/%m %H:%M:%S")
            what = "มีผลลัพธ์" if v.has_final else "ยังไม่ได้ใส่คำแปล"
            item = QListWidgetItem(f"{when}\n{v.label}\n{v.regions} กล่อง · {what}")
            if v.preview:
                item.setIcon(QIcon(QPixmap(str(v.preview)).scaled(110, 150, Qt.AspectRatioMode.KeepAspectRatio)))
            self.list.addItem(item)
        if not self.items:
            self.list.addItem("ยังไม่มีเวอร์ชันที่เก็บไว้ (จะเริ่มเก็บเมื่อรันขั้นตอนกับหน้านี้)")
            self.list.setEnabled(False)
        else:
            self.list.setCurrentRow(0)
        self.restore_btn.setEnabled(bool(self.items))

    def _restore(self):
        row = self.list.currentRow()
        if not self.items or row < 0:
            return
        v = self.items[row]
        when = datetime.fromtimestamp(v.time).strftime("%d/%m %H:%M")
        if QMessageBox.question(self, "คืนค่าเวอร์ชัน",
                                f"คืนหน้านี้เป็นสภาพเมื่อ {when} ({v.label})?\n"
                                "สภาพตอนนี้จะถูกเก็บไว้ในประวัติก่อน จึงย้อนกลับมาได้") != QMessageBox.StandardButton.Yes:
            return
        self.restored = versions.restore(self.doc, v, self.win.settings.versions_keep)
        self.accept()
