"""Main window: pages on the left, the page canvas in the middle, text boxes on the right."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QLocale, QSize, Qt, QTimer, Slot
from PySide6.QtGui import QAction, QActionGroup, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
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

from ..core import importer
from ..core.export import AutoSaver
from ..core.pages import SORT_KEYS, SORT_LABELS, load_order, save_order, sort_pages
from ..core.pipeline import STEP_LABELS, STEPS, Models, run_typeset
from ..core.typeset import place
from ..core.project import PageDoc, Region
from ..core.settings import Settings
from ..core.translate import LANG_NAMES
from .canvas import MODE_ADD, MODE_ERASE, MODE_PAINT, MODE_SELECT, MODE_TEXT, Canvas, to_qimage
from .region_panel import RegionPanel
from .settings_dialog import SettingsDialog
from .worker import AIWorker

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SOURCE_LANGS = ["ja", "en"]
TARGET_LANGS = ["th", "en", "ja", "zh", "ko", "vi", "id"]
LAYERS = [("original", "ต้นฉบับ"), ("cleaned", "ลบคำแล้ว"), ("final", "ผลลัพธ์")]
UNDO_LIMIT = 40


class MainWindow(QMainWindow):
    def __init__(self):
        # Windows set to Thai with native digits made spin boxes and progress show ๑๒๓;
        # an explicit Thai locale keeps Thai formatting but uses 0-9
        QLocale.setDefault(QLocale(QLocale.Language.Thai, QLocale.Country.Thailand))
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
        self.busy = False
        self.folder: Path | None = None
        self.work_dir: Path | None = None
        self.failed: dict[str, str] = {}  # page name -> error from the last run
        self._run_failed: list[str] = []
        self._current_path: Path | None = None
        self.autosaver = AutoSaver(self.settings)
        self._autosave_timer = QTimer(self, singleShot=True, interval=1500)  # after the last edit
        self._autosave_timer.timeout.connect(self._autosave_current)
        self.ai = AIWorker(self.models)
        # bound methods, not lambdas: a lambda has no receiver object, so Qt would call it on
        # the AI thread and touch widgets there (crash in Qt6Core.dll)
        self.ai.runner.progress.connect(self._on_progress)
        self.ai.runner.step_done.connect(self._step_done)
        self.ai.runner.finished.connect(self._finished)
        self.ai.runner.page_started.connect(self._page_started)
        self.ai.runner.page_failed.connect(self._page_failed)
        self.ai.runner.page_saved.connect(self._page_saved)
        self._build()
        self.setAcceptDrops(True)
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
        tb.addAction(QAction("📦 เปิดไฟล์บีบอัด", self, triggered=self.open_archive_dialog,
                             toolTip="เปิดตอนจากไฟล์ ZIP / CBZ / RAR / CBR / 7Z"))
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
        self.all_act = QAction("▶ ทำหน้าที่เลือก", self, shortcut="F5", triggered=self.run_selected)
        self.all_act.setToolTip("ทำครบ 5 ขั้นกับหน้าที่เลือกในรายการ (Ctrl/Shift+คลิกเพื่อเลือกหลายหน้า) ตามลำดับ")
        self.all_pages_act = QAction("▶▶ ทำทุกหน้า", self, triggered=self.run_all_pages)
        self.cancel_act = QAction("■ หยุด", self, triggered=self.cancel)
        tb.addAction(self.all_act)
        tb.addAction(self.all_pages_act)
        tb.addAction(self.cancel_act)
        tb.addSeparator()
        self.export_act = QAction("💾 เซฟหลายหน้า", self, shortcut=QKeySequence.StandardKey.Save, triggered=self.export)
        tb.addAction(self.export_act)
        tb.addAction(QAction("⚙ ตั้งค่า", self, triggered=self.open_settings))
        tb.addAction(QAction("ℹ ข้อมูลและสถานะ", self, shortcut="F1", triggered=self.open_info))

        undo = QAction("Undo", self, shortcut=QKeySequence.StandardKey.Undo, triggered=self.undo)
        redo = QAction("Redo", self, shortcut=QKeySequence.StandardKey.Redo, triggered=self.redo)
        delete = QAction("Delete", self, shortcut=QKeySequence.StandardKey.Delete, triggered=self._delete_shortcut)
        for a in (undo, redo, delete):
            self.addAction(a)

        # ---- left: sort bar + pages
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 0, 0)
        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("เรียง"))
        self.sort_combo = QComboBox()
        for key in SORT_KEYS:
            self.sort_combo.addItem(SORT_LABELS[key], key)
        self.sort_combo.setCurrentIndex(SORT_KEYS.index(self.settings.page_sort) if self.settings.page_sort in SORT_KEYS else 0)
        self.sort_combo.currentIndexChanged.connect(self._sort_changed)
        sort_row.addWidget(self.sort_combo, 1)
        self.sort_dir = QPushButton()
        self.sort_dir.setCheckable(True)
        self.sort_dir.setChecked(self.settings.page_sort_desc)
        self.sort_dir.setFixedWidth(34)
        self.sort_dir.setToolTip("สลับทิศการเรียง")
        self.sort_dir.toggled.connect(self._sort_changed)
        sort_row.addWidget(self.sort_dir)
        ll.addLayout(sort_row)
        self.pages = QListWidget()
        self.pages.setIconSize(QSize(72, 100))
        self.pages.setMinimumWidth(170)
        self.pages.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.pages.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.pages.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.pages.setToolTip("ลากเพื่อเรียงเอง · Ctrl/Shift+คลิกเพื่อเลือกหลายหน้า")
        self.pages.currentRowChanged.connect(self._page_selected)
        self.pages.model().rowsMoved.connect(self._pages_reordered)
        ll.addWidget(self.pages, 1)
        self.skip_done = QCheckBox("ข้ามหน้าที่ทำครบแล้ว")
        self.skip_done.setToolTip("ตอนทำหลายหน้า ข้ามหน้าที่ทำครบ 5 ขั้นแล้ว")
        self.skip_done.setChecked(self.settings.skip_done)
        self.skip_done.toggled.connect(self._skip_done_changed)
        ll.addWidget(self.skip_done)
        self._update_sort_dir_label()

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
            (MODE_TEXT, "ย้ายข้อความ", "ลากคำแปลไปวางตรงไหนก็ได้ ลากมุมขวาล่างเพื่อปรับพื้นที่ (ตัวอักษรปรับขนาดตาม)"),
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
        self.canvas.text_moved.connect(self._text_moved)
        self.canvas.region_added.connect(self._canvas_added)
        self.canvas.mask_edit_started.connect(self.push_undo)
        self.canvas.mask_edited.connect(self._mask_edited)
        # drops of files land on the window (handled below), not on the page canvas
        self.canvas.setAcceptDrops(False)
        self.canvas.viewport().setAcceptDrops(False)
        cl.addWidget(self.canvas, 1)
        self.hint = QLabel("ลากโฟลเดอร์ ภาพ หรือไฟล์ ZIP/CBZ/CBR มาวางในหน้าต่างเพื่อเริ่ม  ·  ล้อเมาส์ = ซูม  ·  ปุ่มกลางเมาส์ = เลื่อนภาพ")
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
        split.addWidget(left)
        split.addWidget(center)
        split.addWidget(self.panel)
        split.setStretchFactor(1, 1)
        split.setSizes([190, 1000, 340])
        self.setCentralWidget(split)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedWidth(160)
        self.progress.setFormat("%v/%m หน้า")
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)

    # ============================================================ pages
    def open_folder(self, folder: str | None = None):
        folder = folder or QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์ภาพมังงะ")
        if not folder:
            return
        folder = Path(folder)
        work_dir = Path(self.settings.work_dir) / folder.name
        files = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT]
        added = work_dir / importer.ADDED  # images dropped onto this chapter earlier (copies)
        if added.is_dir():
            files += [p for p in added.iterdir() if importer.is_image(p)]
        if not files:
            QMessageBox.information(self, "ไม่พบภาพ", "โฟลเดอร์นี้ไม่มีไฟล์ภาพ (png, jpg, webp)")
            return
        self.folder = folder
        self.work_dir = Path(self.settings.work_dir) / folder.name
        self.docs.clear()
        self.failed.clear()
        self._current_path = None
        self.doc = None
        self.images = self._sorted(files)
        self._populate(keep=None)
        self.setWindowTitle(f"Manga Translator — {folder}")

    # ---- bringing pages in: drag & drop, archives
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(u.isLocalFile() for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        self.dragEnterEvent(e)

    def dropEvent(self, e):
        paths = [Path(u.toLocalFile()) for u in e.mimeData().urls() if u.isLocalFile()]
        e.acceptProposedAction()
        self.open_paths(paths)

    def open_archive_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "เลือกไฟล์บีบอัดของตอน", "",
                                              "ไฟล์บีบอัด (*.zip *.cbz *.rar *.cbr *.7z *.cb7)")
        if path:
            self.open_paths([Path(path)])

    def open_paths(self, paths: list[Path]):
        """Open what was dropped (or passed on the command line): a folder, an archive, or images.
        Images go into the open chapter, or into a new one when none is open. Only copies are made."""
        if self.busy:
            self.statusBar().showMessage("กำลังประมวลผลอยู่ รอให้เสร็จก่อนแล้วค่อยลากไฟล์มาใหม่", 6000)
            return
        folders, archives, images, other = importer.classify(paths)
        extra = len(folders) + len(archives) - 1
        try:
            if folders:
                self.open_folder(str(folders[0]))
            elif archives:
                self.statusBar().showMessage(f"กำลังแตกไฟล์ {archives[0].name} …")
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    chapter = importer.import_archive(archives[0], Path(self.settings.work_dir))
                finally:
                    QApplication.restoreOverrideCursor()
                self.open_folder(str(chapter))
            elif images:
                if self.folder and self.work_dir:
                    importer.copy_images(images, self.work_dir / importer.ADDED)
                    keep = self._current_path
                    self.open_folder(str(self.folder))
                    if keep in self.images:
                        self.pages.setCurrentRow(self.images.index(keep))
                    self.statusBar().showMessage(f"เพิ่ม {len(images)} ภาพเข้าตอนนี้แล้ว (คัดลอก ไม่แตะไฟล์ต้นฉบับ)", 6000)
                else:
                    self.open_folder(str(importer.new_chapter_from_images(images, Path(self.settings.work_dir))))
            elif other:
                QMessageBox.information(self, "เปิดไฟล์นี้ไม่ได้",
                                        "ลากโฟลเดอร์ ภาพ (png, jpg, webp) หรือไฟล์ ZIP/CBZ/RAR/CBR/7Z มาวางได้")
                return
        except (importer.ImportFailed, OSError) as err:
            QMessageBox.warning(self, "เปิดไม่สำเร็จ", str(err))
            return
        if extra > 0:
            self.statusBar().showMessage(f"เปิดได้ทีละตอน: เปิดตอนแรกแล้ว อีก {extra} รายการไม่ได้เปิด", 8000)

    def _sorted(self, files: list[Path]) -> list[Path]:
        key = self.sort_combo.currentData()
        manual = load_order(self.work_dir) if key == "manual" and self.work_dir else None
        return sort_pages(files, key, self.sort_dir.isChecked(), manual)

    def _populate(self, keep: Path | None):
        """Fill the page list from self.images, keeping `keep` (or the first page) selected."""
        self.pages.blockSignals(True)
        self.pages.clear()
        for p in self.images:
            item = QListWidgetItem(QIcon(QPixmap(str(p)).scaled(72, 100, Qt.AspectRatioMode.KeepAspectRatio)), "")
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            self.pages.addItem(item)
        self.pages.blockSignals(False)
        self._refresh_page_labels()
        row = self.images.index(keep) if keep in self.images else 0
        self.pages.setCurrentRow(row)
        self._page_selected(row)  # no-op when it is already the open page

    def _page_label(self, path: Path) -> str:
        """Status marks: ✓ every step done · ⚠ failed in the last run · 💾 auto-saved."""
        doc = self.docs.get(path)
        if doc is None and self.work_dir and (self.work_dir / path.stem / "page.json").exists():
            doc = PageDoc.load(self.work_dir / path.stem)
        marks = ""
        if path.stem in self.failed:
            marks += "⚠ "
        elif doc and all(doc.done.get(s) for s in STEPS):
            marks += "✓ "
        if self.work_dir and path.stem in self.autosaver.saved_pages(self.work_dir):
            marks += "💾 "
        return f"{marks}{path.name}"

    def _refresh_page_labels(self):
        for i in range(self.pages.count()):
            item = self.pages.item(i)
            path = Path(item.data(Qt.ItemDataRole.UserRole))
            item.setText(self._page_label(path))
            err = self.failed.get(path.stem)
            item.setToolTip(f"พังรอบล่าสุด: {err}" if err else "")

    def _update_sort_dir_label(self):
        manual = self.sort_combo.currentData() == "manual"
        self.sort_dir.setEnabled(not manual)
        self.sort_dir.setText("↑" if self.sort_dir.isChecked() else "↓")

    def _sort_changed(self, *_):
        self.settings.page_sort = self.sort_combo.currentData()
        self.settings.page_sort_desc = self.sort_dir.isChecked()
        self.settings.save()
        self._update_sort_dir_label()
        if not self.images:
            return
        self.images = self._sorted(self.images)
        self._populate(self._current_path)
        self._resave_all()

    def _pages_reordered(self, *_):
        """The user dragged pages: their order becomes the "manual" order and is remembered."""
        self.images = [Path(self.pages.item(i).data(Qt.ItemDataRole.UserRole)) for i in range(self.pages.count())]
        save_order(self.work_dir, self.images)
        if self.sort_combo.currentData() != "manual":
            self.sort_combo.blockSignals(True)
            self.sort_combo.setCurrentIndex(SORT_KEYS.index("manual"))
            self.sort_combo.blockSignals(False)
            self.settings.page_sort = "manual"
            self.settings.save()
            self._update_sort_dir_label()
        self._resave_all()

    def _skip_done_changed(self, on: bool):
        self.settings.skip_done = on
        self.settings.save()

    def _doc_for(self, row: int) -> PageDoc:
        path = self.images[row]
        if path not in self.docs:
            self.docs[path] = PageDoc.create(path, self.work_dir, self._src(), self._tgt())
        return self.docs[path]

    def _page_selected(self, row: int):
        if row < 0 or row >= len(self.images):
            return
        if self.images[row] == self._current_path and self.doc is not None:
            return  # same page after re-sorting: keep zoom, selection and undo history
        try:
            doc = self._doc_for(row)
        except (OSError, ValueError) as e:
            self.failed[self.images[row].stem] = f"เปิดภาพไม่ได้: {e}"
            self._refresh_page_labels()
            self.statusBar().showMessage(f"เปิดภาพ {self.images[row].name} ไม่ได้: {e}", 8000)
            return
        self._current_path = self.images[row]
        self.doc = doc
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
        self._refresh_boxes()
        self.canvas.set_mask(self.doc.load_layer("mask"))
        self.panel.set_regions(self.doc.regions, self.selected)
        self.show_layer(layer or self.layer, fit=fit)
        self._update_actions()

    def show_layer(self, layer: str, fit: bool = False):
        if not self.doc:
            return
        try:
            img = self.doc.load_layer(layer)
            if img is None:
                img = self.doc.load_original()
                layer = "original"
        except (OSError, ValueError) as e:  # the page's work files are missing or damaged
            self.canvas.set_image(None)
            self.statusBar().showMessage(f"เปิดภาพของหน้านี้ไม่ได้: {e}", 8000)
            return
        self.layer = layer
        for b in self.layer_group.buttons():
            b.setChecked(b.property("layer") == layer)
        self.canvas.set_image(img, fit=fit)
        # boxes clutter the finished page; show them on the working layers
        self.canvas.set_show_boxes(layer != "final" or self.canvas.mode != MODE_SELECT)
        self.canvas.set_show_mask(self.canvas.mode in (MODE_PAINT, MODE_ERASE))

    def _text_rects(self) -> dict[int, list[int]]:
        """Where each translation is drawn now (the user's box or the automatic fit)."""
        cleaned = self.doc.load_layer("cleaned")
        if cleaned is None:
            return {}
        boxes = {r.id: r.bbox for r in self.doc.regions}
        return {r.id: list(place(cleaned, r, self.doc.target_lang, self.settings.font_family,
                                 [b for i, b in boxes.items() if i != r.id])[0])
                for r in self.doc.regions if (r.translation or "").strip()}

    def _refresh_boxes(self):
        if self.canvas.mode == MODE_TEXT:
            self.canvas.set_regions(self.doc.regions, self.selected, text_rects=self._text_rects())
        else:
            self.canvas.set_regions(self.doc.regions, self.selected)
        self.canvas.set_mode(self.canvas.mode)  # new items take the mode's movable flags

    def set_mode(self, mode: str):
        if mode == MODE_TEXT and (not self.doc or not self.doc.final_path.exists()):
            QMessageBox.information(self, "ยังไม่มีคำแปลบนภาพ", "กด '5. ใส่คำแปล' ก่อน แล้วค่อยย้ายข้อความ")
            mode = MODE_SELECT
            for b in self.mode_group.buttons():
                b.setChecked(b.property("mode") == MODE_SELECT)
        was_text = self.canvas.mode == MODE_TEXT
        for b in self.mode_group.buttons():
            b.setChecked(b.property("mode") == mode)
        self.canvas.set_mode(mode)
        if self.doc and (was_text or mode == MODE_TEXT):
            self._refresh_boxes()
        if mode == MODE_TEXT:
            self.show_layer("final")
        if mode in (MODE_PAINT, MODE_ERASE):
            if self.doc and self.canvas.mask is None:
                self.canvas.set_mask(np.zeros(self.doc.load_original().shape[:2], np.uint8))
            if self.layer == "final":
                self.show_layer("original")
        self.canvas.set_show_mask(mode in (MODE_PAINT, MODE_ERASE))
        self.canvas.set_show_boxes(self.layer != "final" or mode != MODE_SELECT)

    def _update_actions(self):
        busy = self.busy
        has = self.doc is not None
        for a in self.step_actions.values():
            a.setEnabled(has and not busy)
        self.all_act.setEnabled(has and not busy)
        self.all_pages_act.setEnabled(has and not busy and len(self.images) > 1)
        self.sort_combo.setEnabled(not busy)
        self.sort_dir.setEnabled(not busy and self.sort_combo.currentData() != "manual")
        self.pages.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop if busy
                                   else QAbstractItemView.DragDropMode.InternalMove)
        self.cancel_act.setEnabled(busy)
        self.export_act.setEnabled(has and not busy)
        self.progress.setVisible(busy)
        # edits while a step runs would be overwritten when the step saves the page
        self.panel.setEnabled(not busy)
        self.canvas.setInteractive(not busy)
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
        if "text_rect" in fields and self.panel.current == rid:
            self.panel.auto_place.setEnabled(bool(r.text_rect))
        self.canvas.items_by_id.get(rid) and self.canvas.items_by_id[rid].set_style(r.locked, r.skip)
        if "skip" in fields and self.doc.cleaned_path.exists():
            # the removal area changed: redo "remove text" and "typeset" on the AI thread
            if "locked" in fields and self.panel.current == rid:
                self.panel.locked.blockSignals(True)
                self.panel.locked.setChecked(True)
                self.panel.locked.blockSignals(False)
            QTimer.singleShot(0, lambda: self.run_steps(["clean", "typeset"]))
            return
        # live preview: re-typeset when the cleaned layer exists (fast, no AI)
        if self.doc.cleaned_path.exists() and any(k in fields for k in ("translation", "font_family", "font_size", "color",
                                                                        "outline", "align", "text_rect", "wrap")):
            run_typeset(self.doc, self.models, self.statusBar().showMessage)
            self._autosave_timer.start()
            if self.layer == "final":
                self.show_layer("final")
            if self.canvas.mode == MODE_TEXT:
                self._refresh_boxes()  # the automatic box follows the new text

    def _text_moved(self, rid: int, rect: list):
        r = self.doc.region(rid)
        if not r:
            return
        self.push_undo()
        r.text_rect = rect
        r.locked = True  # keep the user's placement when steps are re-run
        self.doc.save()
        self.panel.refresh_label(r)
        if self.panel.current == rid:
            self.panel.locked.blockSignals(True)
            self.panel.locked.setChecked(True)
            self.panel.locked.blockSignals(False)
            self.panel.auto_place.setEnabled(True)
        run_typeset(self.doc, self.models, self.statusBar().showMessage)
        self._autosave_timer.start()
        # only the image changes: the dragged box is already where the user dropped it
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
        if not self.doc or self.busy:
            return
        self.panel._commit_pending()
        docs = docs or [self.doc]
        for d in docs:
            d.source_lang, d.target_lang = self._src(), self._tgt()
            self.failed.pop(d.folder.name, None)
        self._run_failed = []
        self.ai.runner.save_hook = self._make_save_hook()
        self.busy = True
        self.progress.setRange(0, len(docs))
        self.progress.setValue(0)
        self.progress.setTextVisible(len(docs) > 1)
        self._update_actions()
        self._refresh_page_labels()
        self.ai.submit(docs, list(steps))

    def _batch_docs(self, rows: list[int]) -> list[PageDoc] | None:
        docs = []
        for r in rows:
            try:
                docs.append(self._doc_for(r))
            except (OSError, ValueError) as e:  # unreadable image: report it, keep the others
                self.failed[self.images[r].stem] = f"เปิดภาพไม่ได้: {e}"
        self._refresh_page_labels()
        if len(docs) > 1 and self.skip_done.isChecked():
            docs = [d for d in docs if not all(d.done.get(s) for s in STEPS)]
            if not docs:
                QMessageBox.information(self, "ไม่มีหน้าที่ต้องทำ",
                                        "ทุกหน้าที่เลือกทำครบแล้ว (ยกเลิกติ๊ก 'ข้ามหน้าที่ทำครบแล้ว' เพื่อทำซ้ำ)")
                return None
        return docs

    def run_selected(self):
        rows = sorted({i.row() for i in self.pages.selectedIndexes()}) or [self.pages.currentRow()]
        docs = self._batch_docs([r for r in rows if r >= 0])
        if docs:
            self.run_steps(STEPS, docs)

    def run_all_pages(self):
        docs = self._batch_docs(list(range(len(self.images))))
        if docs:
            self.run_steps(STEPS, docs)

    def cancel(self):
        if self.busy:
            self.ai.cancel()
            self.statusBar().showMessage("กำลังหยุดหลังขั้นตอนปัจจุบัน …")

    # ---- auto-save
    def _make_save_hook(self):
        """A plain function for the AI thread: saves a finished page under its number in the list."""
        if not self.autosaver.enabled or not self.folder:
            return None
        chapter, work_dir = self.folder.name, self.work_dir
        index = {p.stem: i for i, p in enumerate(self.images)}
        saver = self.autosaver

        def hook(doc):
            i = index.get(doc.folder.name)
            return None if i is None else saver.save(work_dir, chapter, doc.folder.name, i, doc.final_path)
        return hook

    def _autosave_current(self):
        if not self.doc or not self.autosaver.enabled or self._current_path not in self.images:
            return
        name = self.autosaver.save(self.work_dir, self.folder.name, self.doc.folder.name,
                                   self.images.index(self._current_path), self.doc.final_path)
        if name:
            self.statusBar().showMessage(f"💾 เซฟอัตโนมัติ {name}", 4000)
        elif self.autosaver.status.last_error:
            self.statusBar().showMessage(self.autosaver.status.last_error, 8000)
        self._refresh_page_labels()

    def _resave_all(self):
        """Page numbers follow the list order: after re-ordering, re-save every finished page."""
        if not self.autosaver.enabled or not self.folder or not self.images:
            return
        jobs = [(p.stem, i, self.work_dir / p.stem / "final.png") for i, p in enumerate(self.images)
                if (self.work_dir / p.stem / "final.png").exists()]
        if not jobs:
            return
        chapter, work_dir, saver, runner = self.folder.name, self.work_dir, self.autosaver, self.ai.runner

        def resave():
            # swapped numbers are safe: AutoSaver may overwrite a file only if it wrote it,
            # and it hands the name over to the page now using it
            for stem, i, final in jobs:
                name = saver.save(work_dir, chapter, stem, i, final)
                if name:
                    runner.page_saved.emit(stem, name)
        self.statusBar().showMessage(f"💾 เซฟใหม่ตามลำดับใหม่ {len(jobs)} หน้า …", 4000)
        self.ai.run_call(resave)

    @Slot(str)
    def _on_progress(self, msg: str):
        self.statusBar().showMessage(msg)

    @Slot(int, int, str)
    def _page_started(self, n: int, total: int, name: str):
        self.progress.setValue(n - 1)

    @Slot(str, str)
    def _page_failed(self, name: str, error: str):
        self.failed[name] = error
        self._run_failed.append(name)
        self._refresh_page_labels()

    @Slot(str, str)
    def _page_saved(self, name: str, file: str):
        self._refresh_page_labels()

    @Slot(str, str)
    def _step_done(self, step: str, folder: str):
        if self.doc and str(self.doc.folder) == folder:
            layer = {"clean": "cleaned", "typeset": "final"}.get(step, self.layer)
            self.doc = PageDoc.load(self.doc.folder)
            self.docs[self._current_path] = self.doc
            self.reload(layer=layer)
        if step == "typeset":
            self._refresh_page_labels()

    @Slot(bool, str)
    def _finished(self, ok: bool, message: str):
        self.busy = False
        self.progress.setValue(self.progress.maximum())
        # the AI thread saved its own PageDoc copies; reload every open page from disk
        for p, d in list(self.docs.items()):
            self.docs[p] = PageDoc.load(d.folder)
        if self._current_path in self.docs:
            self.doc = self.docs[self._current_path]
            self.reload()
        self._update_actions()
        self._refresh_page_labels()
        if message == "ยกเลิกแล้ว":
            self.statusBar().showMessage("หยุดแล้ว", 6000)
            return
        if not self._run_failed:
            self.statusBar().showMessage(message or "เสร็จแล้ว", 6000)
            return
        self.statusBar().showMessage(message, 10000)
        lines = "\n".join(f"• {n}: {self.failed.get(n, '')[:200]}" for n in self._run_failed)
        box = QMessageBox(QMessageBox.Icon.Warning, "บางหน้าทำไม่สำเร็จ", f"{message}\n\n{lines}", parent=self)
        retry = box.addButton("ลองใหม่เฉพาะหน้าที่พัง", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("ปิด", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is retry:
            rows = [i for i, p in enumerate(self.images) if p.stem in self._run_failed]
            self.run_steps(STEPS, [self._doc_for(r) for r in rows])

    # =========================================================== export
    def export(self):
        if not self.doc or not self.folder:
            return
        from .export_dialog import ExportDialog

        ExportDialog(self).exec()
        self._refresh_page_labels()

    def open_info(self):
        from .info_dialog import InfoDialog

        if getattr(self, "_info", None) is None:
            self._info = InfoDialog(self)
            self._info.finished.connect(lambda _: setattr(self, "_info", None))
        self._info.show()
        self._info.raise_()

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
        self.ai.stop()
        super().closeEvent(e)
