"""Main window: pages on the left, the page canvas in the middle, text boxes on the right."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QActionGroup, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..core.pipeline import STEP_LABELS, STEPS, Models, run_typeset
from ..core.project import PageDoc, Region, copy_export
from ..core.settings import Settings
from ..core.translate import LANG_NAMES
from .canvas import MODE_ADD, MODE_ERASE, MODE_PAINT, MODE_SELECT, Canvas, to_qimage
from .region_panel import RegionPanel
from .settings_dialog import SettingsDialog
from .worker import StepWorker, start_worker

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SOURCE_LANGS = ["ja", "en"]
TARGET_LANGS = ["th", "en", "ja", "zh", "ko", "vi", "id"]
LAYERS = [("original", "ต้นฉบับ"), ("cleaned", "ลบคำแล้ว"), ("final", "ผลลัพธ์")]
UNDO_LIMIT = 40


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manga Translator")
        self.resize(1500, 950)
        self.settings = Settings.load()
        self.models = Models(self.settings)
        self.images: list[Path] = []
        self.docs: dict[Path, PageDoc] = {}
        self.doc: PageDoc | None = None
        self.layer = "original"
        self.selected: int | None = None
        self.undo_stack: list[tuple] = []
        self.redo_stack: list[tuple] = []
        self.thread = None
        self.worker = None
        self._build()
        self._update_actions()

    # ================================================================ UI
    def _build(self):
        tb = QToolBar("หลัก")
        tb.setIconSize(QSize(18, 18))
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(tb)

        open_act = QAction("📂 เปิดโฟลเดอร์ภาพ", self, shortcut=QKeySequence.StandardKey.Open, triggered=self.open_folder)
        tb.addAction(open_act)
        tb.addSeparator()
        self.src_combo = QComboBox()
        for code in SOURCE_LANGS:
            self.src_combo.addItem(LANG_NAMES[code], code)
        self.tgt_combo = QComboBox()
        for code in TARGET_LANGS:
            self.tgt_combo.addItem(LANG_NAMES[code], code)
        self.src_combo.setCurrentIndex(max(0, SOURCE_LANGS.index(self.settings.source_lang) if self.settings.source_lang in SOURCE_LANGS else 0))
        self.tgt_combo.setCurrentIndex(max(0, TARGET_LANGS.index(self.settings.target_lang) if self.settings.target_lang in TARGET_LANGS else 0))
        self.src_combo.currentIndexChanged.connect(self._langs_changed)
        self.tgt_combo.currentIndexChanged.connect(self._langs_changed)
        tb.addWidget(QLabel(" จาก "))
        tb.addWidget(self.src_combo)
        tb.addWidget(QLabel(" เป็น "))
        tb.addWidget(self.tgt_combo)
        tb.addSeparator()

        self.step_actions: dict[str, QAction] = {}
        for i, step in enumerate(STEPS, 1):
            act = QAction(f"{i}. {STEP_LABELS[step]}", self, triggered=lambda _=False, s=step: self.run_steps([s]))
            self.step_actions[step] = act
            tb.addAction(act)
        tb.addSeparator()
        self.all_act = QAction("▶ ทำทั้งหมด", self, shortcut="F5", triggered=lambda: self.run_steps(STEPS))
        self.all_pages_act = QAction("▶▶ ทุกหน้า", self, triggered=self.run_all_pages)
        self.cancel_act = QAction("■ หยุด", self, triggered=self.cancel)
        tb.addAction(self.all_act)
        tb.addAction(self.all_pages_act)
        tb.addAction(self.cancel_act)
        tb.addSeparator()
        self.export_act = QAction("💾 Export", self, shortcut=QKeySequence.StandardKey.Save, triggered=self.export)
        tb.addAction(self.export_act)
        tb.addAction(QAction("⚙ ตั้งค่า", self, triggered=self.open_settings))

        undo = QAction("Undo", self, shortcut=QKeySequence.StandardKey.Undo, triggered=self.undo)
        redo = QAction("Redo", self, shortcut=QKeySequence.StandardKey.Redo, triggered=self.redo)
        delete = QAction("Delete", self, shortcut=QKeySequence.StandardKey.Delete, triggered=self._delete_shortcut)
        for a in (undo, redo, delete):
            self.addAction(a)

        # ---- left: pages
        self.pages = QListWidget()
        self.pages.setIconSize(QSize(72, 100))
        self.pages.setMinimumWidth(170)
        self.pages.currentRowChanged.connect(self._page_selected)

        # ---- centre: view bar + canvas
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 0)
        self.layer_group = QButtonGroup(self)
        for key, label in LAYERS:
            b = QPushButton(label)
            b.setCheckable(True)
            b.setProperty("layer", key)
            self.layer_group.addButton(b)
            bar.addWidget(b)
            if key == "original":
                b.setChecked(True)
        self.layer_group.buttonClicked.connect(lambda b: self.show_layer(b.property("layer")))
        bar.addSpacing(16)
        self.mode_group = QButtonGroup(self)
        for mode, label, tip in [
            (MODE_SELECT, "เลือก/ย้าย", "ลากกล่องเพื่อย้าย ลากมุมขวาล่างเพื่อปรับขนาด"),
            (MODE_ADD, "+ กล่อง", "ลากบนภาพเพื่อสร้างกล่องข้อความใหม่"),
            (MODE_PAINT, "ระบายส่วนที่จะลบ", "ระบายทับตัวอักษรที่ยังลบไม่หมด"),
            (MODE_ERASE, "ยางลบ", "เอาส่วนที่ไม่ใช่ตัวอักษรออกจากพื้นที่ลบ"),
        ]:
            b = QPushButton(label)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setProperty("mode", mode)
            self.mode_group.addButton(b)
            bar.addWidget(b)
            if mode == MODE_SELECT:
                b.setChecked(True)
        self.mode_group.buttonClicked.connect(lambda b: self.set_mode(b.property("mode")))
        self.brush = QSlider(Qt.Orientation.Horizontal)
        self.brush.setRange(2, 60)
        self.brush.setValue(14)
        self.brush.setFixedWidth(100)
        self.brush.setToolTip("ขนาดแปรง")
        self.brush.valueChanged.connect(lambda v: setattr(self.canvas, "brush", v))
        bar.addWidget(QLabel("แปรง"))
        bar.addWidget(self.brush)
        bar.addStretch()
        fit = QPushButton("พอดีจอ")
        fit.clicked.connect(lambda: self.canvas.fit())
        bar.addWidget(fit)
        cl.addLayout(bar)
        self.canvas = Canvas()
        self.canvas.region_selected.connect(self._canvas_selected)
        self.canvas.region_changed.connect(self._canvas_moved)
        self.canvas.region_added.connect(self._canvas_added)
        self.canvas.mask_edit_started.connect(self.push_undo)
        self.canvas.mask_edited.connect(self._mask_edited)
        cl.addWidget(self.canvas, 1)
        self.hint = QLabel("เปิดโฟลเดอร์ภาพมังงะเพื่อเริ่ม  ·  ล้อเมาส์ = ซูม  ·  ปุ่มกลางเมาส์ = เลื่อนภาพ")
        self.hint.setStyleSheet("color:#888; padding:2px 8px;")
        cl.addWidget(self.hint)

        # ---- right: regions
        self.panel = RegionPanel()
        self.panel.setMinimumWidth(320)
        self.panel.region_selected.connect(self._panel_selected)
        self.panel.region_edited.connect(self._region_edited)
        self.panel.edit_started.connect(self.push_undo)
        self.panel.delete_requested.connect(self.delete_region)

        split = QSplitter()
        split.addWidget(self.pages)
        split.addWidget(center)
        split.addWidget(self.panel)
        split.setStretchFactor(1, 1)
        split.setSizes([190, 1000, 340])
        self.setCentralWidget(split)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedWidth(160)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)

    # ============================================================ pages
    def open_folder(self, folder: str | None = None):
        folder = folder or QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์ภาพมังงะ")
        if not folder:
            return
        folder = Path(folder)
        self.images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)
        if not self.images:
            QMessageBox.information(self, "ไม่พบภาพ", "โฟลเดอร์นี้ไม่มีไฟล์ภาพ (png, jpg, webp)")
            return
        self.work_dir = Path(self.settings.work_dir) / folder.name
        self.docs.clear()
        self.pages.clear()
        for p in self.images:
            item = QListWidgetItem(QIcon(QPixmap(str(p)).scaled(72, 100, Qt.AspectRatioMode.KeepAspectRatio)), p.name)
            self.pages.addItem(item)
        self.setWindowTitle(f"Manga Translator — {folder}")
        self.pages.setCurrentRow(0)

    def _doc_for(self, row: int) -> PageDoc:
        path = self.images[row]
        if path not in self.docs:
            self.docs[path] = PageDoc.create(path, self.work_dir, self._src(), self._tgt())
        return self.docs[path]

    def _page_selected(self, row: int):
        if row < 0 or row >= len(self.images):
            return
        self.doc = self._doc_for(row)
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._loading_langs = True
        self.src_combo.setCurrentIndex(SOURCE_LANGS.index(self.doc.source_lang) if self.doc.source_lang in SOURCE_LANGS else 0)
        self.tgt_combo.setCurrentIndex(TARGET_LANGS.index(self.doc.target_lang) if self.doc.target_lang in TARGET_LANGS else 0)
        self._loading_langs = False
        best = "final" if self.doc.final_path.exists() else "original"
        self.selected = None
        self.reload(fit=True, layer=best)

    def _src(self) -> str:
        return self.src_combo.currentData()

    def _tgt(self) -> str:
        return self.tgt_combo.currentData()

    def _langs_changed(self):
        self.settings.source_lang, self.settings.target_lang = self._src(), self._tgt()
        self.settings.save()
        if self.doc and not getattr(self, "_loading_langs", False):
            self.doc.source_lang, self.doc.target_lang = self._src(), self._tgt()
            self.doc.save()

    # ========================================================== display
    def reload(self, fit: bool = False, layer: str | None = None):
        if not self.doc:
            return
        self.canvas.set_regions(self.doc.regions, self.selected)
        self.canvas.set_mask(self.doc.load_layer("mask"))
        self.panel.set_regions(self.doc.regions, self.selected)
        self.show_layer(layer or self.layer, fit=fit)
        self._update_actions()

    def show_layer(self, layer: str, fit: bool = False):
        if not self.doc:
            return
        img = self.doc.load_layer(layer)
        if img is None:
            img = self.doc.load_original()
            layer = "original"
        self.layer = layer
        for b in self.layer_group.buttons():
            b.setChecked(b.property("layer") == layer)
        self.canvas.set_image(img, fit=fit)
        # boxes clutter the finished page; show them on the working layers
        self.canvas.set_show_boxes(layer != "final" or self.canvas.mode != MODE_SELECT)
        self.canvas.set_show_mask(self.canvas.mode in (MODE_PAINT, MODE_ERASE))

    def set_mode(self, mode: str):
        self.canvas.set_mode(mode)
        if mode in (MODE_PAINT, MODE_ERASE):
            if self.doc and self.canvas.mask is None:
                self.canvas.set_mask(np.zeros(self.doc.load_original().shape[:2], np.uint8))
            if self.layer == "final":
                self.show_layer("original")
        self.canvas.set_show_mask(mode in (MODE_PAINT, MODE_ERASE))
        self.canvas.set_show_boxes(self.layer != "final" or mode != MODE_SELECT)

    def _update_actions(self):
        busy = self.thread is not None
        has = self.doc is not None
        for a in self.step_actions.values():
            a.setEnabled(has and not busy)
        self.all_act.setEnabled(has and not busy)
        self.all_pages_act.setEnabled(has and not busy and len(self.images) > 1)
        self.cancel_act.setEnabled(busy)
        self.export_act.setEnabled(has and not busy)
        self.progress.setVisible(busy)
        # edits while a step runs would be overwritten when the step saves the page
        self.panel.setEnabled(not busy)
        self.canvas.setInteractive(not busy)
        self.pages.setEnabled(not busy)
        if has:
            done = [f"{'✓' if self.doc.done.get(s) else '·'} {STEP_LABELS[s]}" for s in STEPS]
            self.hint.setText("   ".join(done))

    # ========================================================== regions
    def _canvas_selected(self, rid: int):
        self.selected = rid
        self.panel.select(rid)

    def _panel_selected(self, rid: int):
        self.selected = rid
        self.canvas.select_region(rid)

    def _canvas_moved(self, rid: int, bbox: list):
        r = self.doc.region(rid)
        if not r:
            return
        self.push_undo(snapshot=self._snapshot(regions_override=self._regions_with(rid, bbox_before=r.bbox)))
        r.bbox = bbox
        self.doc.save()

    def _regions_with(self, rid, bbox_before):
        regs = copy.deepcopy(self.doc.regions)
        for r in regs:
            if r.id == rid:
                r.bbox = list(bbox_before)
        return regs

    def _canvas_added(self, bbox: list):
        self.push_undo()
        r = Region(id=self.doc.next_region_id(), bbox=bbox, locked=True,
                   color=self.settings.text_color or "#000000", outline=self.settings.outline_color)
        self.doc.regions.append(r)
        self.doc.save()
        self.selected = r.id
        self.reload()
        self.set_mode(MODE_SELECT)
        for b in self.mode_group.buttons():
            b.setChecked(b.property("mode") == MODE_SELECT)
        self.statusBar().showMessage("เพิ่มกล่องแล้ว: พิมพ์ข้อความต้นฉบับหรือกด 'อ่านข้อความ' (กล่องที่ล็อกจะไม่ถูกอ่านซ้ำ ปลดล็อกก่อน)", 8000)

    def _region_edited(self, rid: int, fields: dict):
        r = self.doc.region(rid)
        if not r:
            return
        for k, v in fields.items():
            setattr(r, k, v)
        self.doc.save()
        self.panel.refresh_label(r)
        self.canvas.items_by_id.get(rid) and self.canvas.items_by_id[rid].set_style(r.locked)
        # live preview: re-typeset when the cleaned layer exists (fast, no AI)
        if self.doc.cleaned_path.exists() and any(k in fields for k in
                                                  ("translation", "font_family", "font_size", "color", "outline", "align")):
            run_typeset(self.doc, self.models, self.statusBar().showMessage)
            if self.layer == "final":
                self.show_layer("final")

    def delete_region(self, rid: int):
        self.push_undo()
        self.doc.regions = [r for r in self.doc.regions if r.id != rid]
        self.doc.save()
        self.selected = None
        self.reload()

    def _delete_shortcut(self):
        if self.selected is not None and self.canvas.hasFocus():
            self.delete_region(self.selected)

    def _mask_edited(self):
        self.doc.save_layer("mask", self.canvas.mask)
        self.doc.done["clean"] = False
        self.doc.save()
        self.statusBar().showMessage("แก้พื้นที่ลบแล้ว กด 'ลบคำ' เพื่อใช้ผล", 5000)

    # ============================================================= undo
    def _snapshot(self, regions_override=None):
        mask = self.canvas.mask.copy() if self.canvas.mask is not None else None
        return (copy.deepcopy(regions_override if regions_override is not None else self.doc.regions), mask)

    def push_undo(self, snapshot=None):
        if not self.doc:
            return
        self.undo_stack.append(snapshot or self._snapshot())
        del self.undo_stack[:-UNDO_LIMIT]
        self.redo_stack.clear()

    def _restore(self, snap):
        regions, mask = snap
        self.doc.regions = regions
        self.doc.save()
        if mask is not None:
            self.doc.save_layer("mask", mask)
        self.reload()

    def undo(self):
        if self.doc and self.undo_stack:
            self.redo_stack.append(self._snapshot())
            self._restore(self.undo_stack.pop())

    def redo(self):
        if self.doc and self.redo_stack:
            self.undo_stack.append(self._snapshot())
            self._restore(self.redo_stack.pop())

    # ========================================================= pipeline
    def run_steps(self, steps: list[str], docs: list[PageDoc] | None = None):
        if not self.doc or self.thread:
            return
        self.panel._commit_pending()
        docs = docs or [self.doc]
        for d in docs:
            d.source_lang, d.target_lang = self._src(), self._tgt()
        self.worker = StepWorker(self.models, docs, steps)
        self.worker.progress.connect(lambda m: self.statusBar().showMessage(m))
        self.worker.step_done.connect(self._step_done)
        self.worker.finished.connect(self._finished)
        self.thread = start_worker(self, self.worker)
        self._update_actions()

    def run_all_pages(self):
        docs = [self._doc_for(i) for i in range(len(self.images))]
        self.run_steps(STEPS, docs)

    def cancel(self):
        if self.worker:
            self.worker.cancel()
            self.statusBar().showMessage("กำลังหยุดหลังขั้นตอนปัจจุบัน …")

    def _step_done(self, step: str):
        layer = {"clean": "cleaned", "typeset": "final"}.get(step, self.layer)
        self.doc = PageDoc.load(self.doc.folder) if self.doc else None
        if self.doc:
            self.docs[self.images[self.pages.currentRow()]] = self.doc
        self.reload(layer=layer)

    def _finished(self, ok: bool, error: str):
        self.thread = None
        self.worker = None
        # worker threads saved their own PageDoc copies; reload every open page from disk
        for p, d in list(self.docs.items()):
            self.docs[p] = PageDoc.load(d.folder)
        if self.pages.currentRow() >= 0:
            self.doc = self.docs[self.images[self.pages.currentRow()]]
            self.reload()
        self._update_actions()
        if ok:
            self.statusBar().showMessage("เสร็จแล้ว", 5000)
        elif error:
            self.statusBar().showMessage(error, 10000)
            if error != "ยกเลิกแล้ว":
                QMessageBox.warning(self, "เกิดข้อผิดพลาด", error)

    # =========================================================== export
    def export(self):
        if not self.doc:
            return
        ready = [(p, d) for p, d in self.docs.items() if d.final_path.exists()]
        if not ready:
            QMessageBox.information(self, "ยังไม่มีผลลัพธ์", "ยังไม่มีหน้าที่ใส่คำแปลแล้ว")
            return
        folder = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์ที่จะบันทึกภาพแปล")
        if not folder:
            return
        for p, d in ready:
            copy_export(d.final_path, Path(folder) / f"{p.stem}.png")
        self.statusBar().showMessage(f"บันทึก {len(ready)} หน้าไปที่ {folder}", 8000)

    def open_settings(self):
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec():
            gpu_before = self.models.settings.use_gpu
            self.settings = dlg.apply()
            self.models.settings = self.settings
            if gpu_before != self.settings.use_gpu:
                self.models.release_all()
            self.statusBar().showMessage("บันทึกการตั้งค่าแล้ว", 4000)

    def closeEvent(self, e):
        if self.thread:
            self.worker.cancel()
            self.thread.quit()
            self.thread.wait(3000)
        super().closeEvent(e)
