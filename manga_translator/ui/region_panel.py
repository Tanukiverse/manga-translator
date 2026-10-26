"""Right-hand panel: list of text boxes and the editor for the selected one."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFontComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

ALIGNS = [("center", "กึ่งกลาง"), ("left", "ชิดซ้าย"), ("right", "ชิดขวา")]
WRAPS = [("auto", "ตัดบรรทัดอัตโนมัติ"), ("manual", "ตัดเฉพาะที่กด Enter")]


class ColorButton(QPushButton):
    changed = Signal(str)

    def __init__(self, allow_none: bool = False):
        super().__init__()
        self.allow_none = allow_none
        self.value = ""
        self.clicked.connect(self._pick)
        self.setFixedWidth(70)

    def set_value(self, value: str):
        self.value = value or ""
        if self.value:
            self.setText("")
            self.setStyleSheet(f"background:{self.value}; border:1px solid #888;")
        else:
            self.setText("ไม่มี")
            self.setStyleSheet("")

    def _pick(self):
        c = QColorDialog.getColor(QColor(self.value or "#000000"), self, "เลือกสี")
        if c.isValid():
            self.set_value(c.name())
            self.changed.emit(self.value)

    def contextMenuEvent(self, e):
        if self.allow_none:
            self.set_value("")
            self.changed.emit("")


class RegionPanel(QWidget):
    region_selected = Signal(int)
    region_edited = Signal(int, dict)  # id, changed fields
    edit_started = Signal()  # before the first change of an edit burst (for undo)
    delete_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current: int | None = None
        self._loading = False
        self._burst = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)

        lay.addWidget(QLabel("<b>กล่องข้อความ</b>"))
        self.list = QListWidget()
        self.list.setMinimumHeight(160)
        self.list.currentItemChanged.connect(self._on_list)
        lay.addWidget(self.list, 2)

        form = QFormLayout()
        self.source = QPlainTextEdit()
        self.source.setPlaceholderText("ข้อความต้นฉบับ")
        self.source.setFixedHeight(70)
        self.translation = QPlainTextEdit()
        self.translation.setPlaceholderText("คำแปล (กด Enter เพื่อขึ้นบรรทัดใหม่)")
        self.translation.setFixedHeight(90)
        form.addRow("ต้นฉบับ", self.source)
        form.addRow("คำแปล", self.translation)

        self.font = QFontComboBox()
        self.font_default = QCheckBox("ค่าเริ่มต้น")
        frow = QHBoxLayout()
        frow.addWidget(self.font, 1)
        frow.addWidget(self.font_default)
        form.addRow("ฟอนต์", frow)

        self.size = QSpinBox()
        self.size.setRange(0, 200)
        self.size.setSpecialValueText("อัตโนมัติ")
        self.align = QComboBox()
        for key, label in ALIGNS:
            self.align.addItem(label, key)
        srow = QHBoxLayout()
        srow.addWidget(self.size)
        srow.addWidget(self.align)
        form.addRow("ขนาด", srow)

        self.wrap = QComboBox()
        for key, label in WRAPS:
            self.wrap.addItem(label, key)
        self.wrap.setToolTip("อัตโนมัติ: โปรแกรมตัดคำให้พอดีกรอบ และขึ้นบรรทัดใหม่ทุกครั้งที่กด Enter\n"
                             "ตัดเฉพาะที่กด Enter: ใช้บรรทัดตามที่พิมพ์ทุกประการ ตัวอักษรย่อลงจนบรรทัดยาวสุดพอดีกรอบ")
        form.addRow("บรรทัด", self.wrap)

        self.color = ColorButton()
        self.outline = ColorButton(allow_none=True)
        crow = QHBoxLayout()
        crow.addWidget(QLabel("ตัวอักษร"))
        crow.addWidget(self.color)
        crow.addWidget(QLabel("ขอบ"))
        crow.addWidget(self.outline)
        crow.addStretch()
        form.addRow("สี", crow)

        self.locked = QCheckBox("ล็อก (ขั้นอัตโนมัติจะไม่แก้กล่องนี้)")
        form.addRow("", self.locked)
        self.skip = QCheckBox("ไม่ใช่ข้อความ (ไม่ลบ ไม่แปล ไม่วาดทับ)")
        self.skip.setToolTip("เช่น เสียงประกอบหรือลายเส้นที่อ่านออกมาเป็นตัวมั่ว: ภาพเดิมตรงนี้จะคงไว้")
        form.addRow("", self.skip)
        lay.addLayout(form)

        self.auto_place = QPushButton("วางข้อความอัตโนมัติ")
        self.auto_place.setToolTip("ยกเลิกตำแหน่งที่ย้ายเอง ให้โปรแกรมจัดคำแปลในบอลลูนให้")
        self.auto_place.clicked.connect(lambda: self._emit({"text_rect": None}))
        lay.addWidget(self.auto_place)
        self.delete = QPushButton("ลบกล่องนี้")
        self.delete.clicked.connect(lambda: self.current is not None and self.delete_requested.emit(self.current))
        lay.addWidget(self.delete)
        lay.addStretch()

        self._text_timer = QTimer(self, singleShot=True, interval=500)
        self._text_timer.timeout.connect(self._emit_text)
        self.source.textChanged.connect(self._text_changed)
        self.translation.textChanged.connect(self._text_changed)
        self.font.currentFontChanged.connect(lambda f: self._emit({"font_family": f.family()}))
        self.font_default.toggled.connect(self._font_default_toggled)
        self.size.valueChanged.connect(lambda v: self._emit({"font_size": v}))
        self.align.currentIndexChanged.connect(lambda _: self._emit({"align": self.align.currentData()}))
        self.wrap.currentIndexChanged.connect(lambda _: self._emit({"wrap": self.wrap.currentData()}))
        self.color.changed.connect(lambda v: self._emit({"color": v}))
        self.outline.changed.connect(lambda v: self._emit({"outline": v}))
        self.locked.toggled.connect(lambda v: self._emit({"locked": v}))
        # deciding by hand locks the box, so re-running OCR keeps the choice
        self.skip.toggled.connect(lambda v: self._emit({"skip": v, "locked": True}))
        self.set_enabled(False)

    # ---- data in -----------------------------------------------------
    def set_regions(self, regions, keep: int | None = None):
        self._loading = True
        self.list.clear()
        for r in regions:
            item = QListWidgetItem(self._label(r))
            item.setData(Qt.ItemDataRole.UserRole, r.id)
            self.list.addItem(item)
        self._loading = False
        self._regions = {r.id: r for r in regions}
        if keep in self._regions:
            self.select(keep)
        else:
            self.current = None
            self.set_enabled(False)

    @staticmethod
    def _label(r) -> str:
        src = (r.source_text or "…").replace("\n", " ")[:24]
        tr = (r.translation or "").replace("\n", " ")[:24]
        lock = ("⊘ " if r.skip else "") + ("🔒 " if r.locked else "")
        return f"{lock}{r.id}. {src}\n    → {tr}"

    def refresh_label(self, r):
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == r.id:
                item.setText(self._label(r))

    def select(self, rid: int):
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.ItemDataRole.UserRole) == rid:
                self.list.setCurrentRow(i)
                return

    def _on_list(self, item, _prev):
        if self._loading or item is None:
            return
        rid = item.data(Qt.ItemDataRole.UserRole)
        self._show(self._regions[rid])
        self.region_selected.emit(rid)

    def _show(self, r):
        self._commit_pending()
        self._loading = True
        self.current = r.id
        self.source.setPlainText(r.source_text)
        self.translation.setPlainText(r.translation)
        self.font_default.setChecked(not r.font_family)
        self.font.setEnabled(bool(r.font_family))
        if r.font_family:
            self.font.setCurrentFont(r.font_family)
        self.size.setValue(r.font_size)
        self.align.setCurrentIndex(max(0, [a for a, _ in ALIGNS].index(r.align) if r.align in dict(ALIGNS) else 0))
        self.wrap.setCurrentIndex(max(0, [k for k, _ in WRAPS].index(r.wrap) if r.wrap in dict(WRAPS) else 0))
        self.color.set_value(r.color or "#000000")
        self.outline.set_value(r.outline)
        self.locked.setChecked(r.locked)
        self.skip.setChecked(r.skip)
        self.set_enabled(True)
        self.auto_place.setEnabled(bool(r.text_rect))
        self._loading = False
        self._burst = False

    def set_enabled(self, on: bool):
        for w in (self.source, self.translation, self.font_default, self.size, self.align, self.wrap,
                  self.color, self.outline, self.locked, self.skip, self.delete, self.auto_place):
            w.setEnabled(on)
        self.font.setEnabled(on and not self.font_default.isChecked())

    # ---- data out ----------------------------------------------------
    def _font_default_toggled(self, on: bool):
        self.font.setEnabled(not on)
        self._emit({"font_family": "" if on else self.font.currentFont().family()})

    def _text_changed(self):
        if self._loading or self.current is None:
            return
        if not self._burst:
            self._burst = True
            self.edit_started.emit()
        self._text_timer.start()

    def _emit_text(self):
        if self.current is None:
            return
        # editing text by hand locks the box so re-running OCR/translate keeps it
        self.region_edited.emit(self.current, {
            "source_text": self.source.toPlainText(),
            "translation": self.translation.toPlainText(),
            "locked": True,
        })
        self._loading = True
        self.locked.setChecked(True)
        self._loading = False
        self._burst = False

    def _commit_pending(self):
        if self._text_timer.isActive():
            self._text_timer.stop()
            self._emit_text()

    def _emit(self, fields: dict):
        if self._loading or self.current is None:
            return
        self.edit_started.emit()
        self.region_edited.emit(self.current, fields)
