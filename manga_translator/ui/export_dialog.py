"""Save many pages at once with new names (prefix + number), as PNG/JPG/WEBP, with a live preview."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..core.export import FORMATS, ExportError, export_pages, make_prefix, open_folder, output_name


class ExportDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        s = win.settings
        self.chapter = win.folder.name
        self.selected_rows = sorted({i.row() for i in win.pages.selectedIndexes()})
        self.setWindowTitle("เซฟหลายหน้าพร้อมเปลี่ยนชื่อ")
        self.resize(720, 640)
        lay = QVBoxLayout(self)
        form = QFormLayout()

        self.all_pages = QRadioButton("ทุกหน้าที่ใส่คำแปลแล้ว")
        self.only_selected = QRadioButton(f"เฉพาะหน้าที่เลือกในรายการ ({len(self.selected_rows)} หน้า)")
        self.only_selected.setEnabled(len(self.selected_rows) > 1)
        (self.only_selected if len(self.selected_rows) > 1 else self.all_pages).setChecked(True)
        prow = QHBoxLayout()
        prow.addWidget(self.all_pages)
        prow.addWidget(self.only_selected)
        prow.addStretch()
        form.addRow("หน้า", prow)

        self.prefix = QLineEdit(s.save_prefix)
        self.prefix.setToolTip("{ตอน} = ชื่อโฟลเดอร์ต้นฉบับ · เว้นว่างได้")
        form.addRow("คำนำหน้า", self.prefix)
        self.start = QSpinBox()
        self.start.setRange(0, 99999)
        self.start.setValue(s.save_start)
        self.digits = QSpinBox()
        self.digits.setRange(1, 6)
        self.digits.setValue(s.save_digits)
        self.renumber = QCheckBox("เรียงเลขใหม่ต่อกัน")
        self.renumber.setToolTip("ไม่ติ๊ก: ใช้ลำดับของหน้าในรายการ (ตรงกับเซฟอัตโนมัติ)\nติ๊ก: นับ 1, 2, 3 … เฉพาะหน้าที่เซฟ")
        nrow = QHBoxLayout()
        nrow.addWidget(QLabel("เริ่มที่"))
        nrow.addWidget(self.start)
        nrow.addWidget(QLabel("จำนวนหลัก"))
        nrow.addWidget(self.digits)
        nrow.addWidget(self.renumber)
        nrow.addStretch()
        form.addRow("เลขลำดับ", nrow)
        self.fmt = QComboBox()
        for key in FORMATS:
            self.fmt.addItem(key.upper(), key)
        self.fmt.setCurrentIndex(max(0, self.fmt.findData(s.save_format)))
        self.quality = QSpinBox()
        self.quality.setRange(50, 100)
        self.quality.setValue(s.save_quality)
        frow = QHBoxLayout()
        frow.addWidget(self.fmt)
        frow.addWidget(QLabel("คุณภาพ"))
        frow.addWidget(self.quality)
        frow.addStretch()
        form.addRow("ชนิดไฟล์", frow)
        default_dir = (str(Path(s.save_dir) / self.chapter) if s.save_dir else s.export_dir)
        self.folder = QLineEdit(default_dir)
        browse = QPushButton("เลือก…")
        browse.clicked.connect(self._pick)
        drow = QHBoxLayout()
        drow.addWidget(self.folder, 1)
        drow.addWidget(browse)
        form.addRow("เซฟไปที่", drow)
        lay.addLayout(form)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["ไฟล์ต้นฉบับ", "ชื่อใหม่", "สถานะ"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lay.addWidget(self.table, 1)
        self.summary = QLabel()
        lay.addWidget(self.summary)

        self.remember = QCheckBox("ใช้ค่านี้เป็นค่าเริ่มต้นการเซฟ (รวมเซฟอัตโนมัติ)")
        lay.addWidget(self.remember)
        brow = QHBoxLayout()
        brow.addStretch()
        self.save_btn = QPushButton("เซฟ")
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self._save)
        close = QPushButton("ปิด")
        close.clicked.connect(self.reject)
        brow.addWidget(self.save_btn)
        brow.addWidget(close)
        lay.addLayout(brow)

        for w in (self.prefix, self.folder):
            w.textChanged.connect(self._refresh)
        for w in (self.start, self.digits, self.quality):
            w.valueChanged.connect(self._refresh)
        for w in (self.all_pages, self.only_selected, self.renumber):
            w.toggled.connect(self._refresh)
        self.fmt.currentIndexChanged.connect(self._refresh)
        self._refresh()

    def _pick(self):
        d = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์ที่จะเซฟ", self.folder.text())
        if d:
            self.folder.setText(d)

    def plan(self) -> list[tuple[Path, Path | None, str | None]]:
        """[(source image, final.png or None if not translated yet, new name or None)] in list order."""
        win = self.win
        rows = self.selected_rows if self.only_selected.isChecked() else list(range(len(win.images)))
        fmt = self.fmt.currentData()
        prefix = make_prefix(self.prefix.text(), self.chapter)
        out, n = [], 0
        for r in rows:
            path = win.images[r]
            final = win.work_dir / path.stem / "final.png"
            if not final.exists():
                out.append((path, None, None))
                continue
            number = self.start.value() + (n if self.renumber.isChecked() else r)
            out.append((path, final, output_name(prefix, number, self.digits.value(), fmt)))
            n += 1
        return out

    def _refresh(self):
        self.quality.setEnabled(self.fmt.currentData() != "png")
        try:
            plan = self.plan()
        except ExportError as e:
            self.summary.setText(str(e))
            self.save_btn.setEnabled(False)
            return
        folder = Path(self.folder.text().strip()) if self.folder.text().strip() else None
        self.table.setRowCount(len(plan))
        ready = exists = 0
        for i, (src, final, name) in enumerate(plan):
            if final is None:
                status = "ยังไม่ได้ใส่คำแปล (ข้าม)"
            elif folder and (folder / name).exists():
                status = "มีไฟล์ชื่อนี้อยู่แล้ว"
                exists += 1
                ready += 1
            else:
                status = "พร้อม"
                ready += 1
            for c, text in enumerate([src.name, name or "—", status]):
                self.table.setItem(i, c, QTableWidgetItem(text))
        text = f"จะเซฟ {ready} หน้า"
        if exists:
            text += f" · ชื่อซ้ำกับไฟล์ที่มีอยู่ {exists} ไฟล์ (จะถามก่อนเขียนทับ)"
        if not folder:
            text += " · ยังไม่ได้เลือกโฟลเดอร์"
        self.summary.setText(text)
        self.save_btn.setEnabled(bool(ready and folder))

    def _save(self):
        folder = Path(self.folder.text().strip())
        items = [(final, folder / name) for _src, final, name in self.plan() if final is not None]
        clash = [d for _f, d in items if d.exists()]
        overwrite = False
        if clash:
            box = QMessageBox(QMessageBox.Icon.Question, "มีไฟล์ชื่อซ้ำ",
                              f"มีไฟล์ชื่อเดียวกันอยู่แล้ว {len(clash)} ไฟล์ เช่น {clash[0].name}", parent=self)
            over = box.addButton("เขียนทับ", QMessageBox.ButtonRole.DestructiveRole)
            skip = box.addButton("ข้ามไฟล์ที่ซ้ำ", QMessageBox.ButtonRole.AcceptRole)
            box.addButton("ยกเลิก", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() not in (over, skip):
                return
            overwrite = box.clickedButton() is over
        try:
            written, skipped = export_pages(items, self.fmt.currentData(), self.quality.value(), overwrite)
        except (OSError, ExportError) as e:
            QMessageBox.warning(self, "เซฟไม่สำเร็จ", str(e))
            return
        s = self.win.settings
        s.export_dir = str(folder)
        if self.remember.isChecked():
            s.save_prefix, s.save_start, s.save_digits = self.prefix.text(), self.start.value(), self.digits.value()
            s.save_format, s.save_quality = self.fmt.currentData(), self.quality.value()
        s.save()
        msg = f"เซฟแล้ว {len(written)} ไฟล์ที่\n{folder}"
        if skipped:
            msg += f"\nข้ามไฟล์ที่ชื่อซ้ำ {len(skipped)} ไฟล์"
        done = QMessageBox(QMessageBox.Icon.Information, "เซฟเสร็จแล้ว", msg, parent=self)
        open_btn = done.addButton("เปิดโฟลเดอร์", QMessageBox.ButtonRole.ActionRole)
        done.addButton("ปิด", QMessageBox.ButtonRole.AcceptRole)
        done.exec()
        if done.clickedButton() is open_btn:
            open_folder(folder)
        self.win.statusBar().showMessage(f"💾 เซฟ {len(written)} หน้าไปที่ {folder}", 8000)
        self.accept()
