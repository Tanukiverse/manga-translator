"""Settings: Ollama model, default font and colours, glossary, story notes, and saving presets."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFontComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QFileDialog,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.export import CHAPTER_TOKEN, FORMATS, ExportError, make_prefix, output_name
from ..core.settings import Settings
from ..core.translate import OllamaTranslator, TranslatorConfig
from .region_panel import ColorButton


def parse_glossary(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            if k.strip() and v.strip():
                out[k.strip()] = v.strip()
    return out


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ตั้งค่า")
        self.setMinimumWidth(520)
        self.s = settings
        lay = QVBoxLayout(self)
        tabs = QTabWidget()
        lay.addWidget(tabs)
        general = QWidget()
        form = QFormLayout(general)
        tabs.addTab(general, "ทั่วไป")
        tabs.addTab(self._build_save_tab(settings), "การเซฟ")

        self.url = QLineEdit(settings.ollama_url)
        form.addRow("Ollama URL", self.url)
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.setCurrentText(settings.ollama_model)
        refresh = QPushButton("โหลดรายชื่อโมเดล")
        refresh.clicked.connect(self._load_models)
        mrow = QHBoxLayout()
        mrow.addWidget(self.model, 1)
        mrow.addWidget(refresh)
        form.addRow("โมเดลแปล", mrow)
        self.model_status = QLabel("")
        form.addRow("", self.model_status)
        self.keep_alive = QLineEdit(settings.ollama_keep_alive)
        self.keep_alive.setToolTip("เวลาที่ Ollama ค้างโมเดลไว้ใน VRAM หลังแปล เช่น 5m หรือ 0 = ปล่อยทันที")
        form.addRow("ค้างโมเดลไว้", self.keep_alive)

        self.font_default = QCheckBox("ใช้ฟอนต์เริ่มต้นตามภาษา (ไทย: Leelawadee UI)")
        self.font = QFontComboBox()
        self.font_default.setChecked(not settings.font_family)
        self.font.setEnabled(bool(settings.font_family))
        if settings.font_family:
            self.font.setCurrentFont(settings.font_family)
        self.font_default.toggled.connect(lambda on: self.font.setEnabled(not on))
        form.addRow("ฟอนต์", self.font_default)
        form.addRow("", self.font)

        self.color = ColorButton()
        self.color.set_value(settings.text_color)
        self.outline = ColorButton(allow_none=True)
        self.outline.set_value(settings.outline_color)
        crow = QHBoxLayout()
        crow.addWidget(QLabel("ตัวอักษร"))
        crow.addWidget(self.color)
        crow.addWidget(QLabel("ขอบ (คลิกขวา = ไม่มี)"))
        crow.addWidget(self.outline)
        crow.addStretch()
        form.addRow("สีเริ่มต้น", crow)

        self.versions_keep = QSpinBox()
        self.versions_keep.setRange(1, 50)
        self.versions_keep.setValue(settings.versions_keep)
        self.versions_keep.setSuffix(" รายการต่อหน้า")
        self.versions_keep.setToolTip("ประวัติที่ใช้ย้อนกลับไปผลก่อนหน้า ไฟล์ที่ไม่เปลี่ยนไม่กินพื้นที่ซ้ำ")
        form.addRow("เก็บประวัติ", self.versions_keep)

        self.ui_size = QSpinBox()
        self.ui_size.setRange(9, 18)
        self.ui_size.setValue(settings.ui_font_size)
        self.ui_size.setSuffix(" pt")
        self.ui_size.setToolTip("ขนาดตัวอักษร ปุ่ม และเมนูของโปรแกรม (มีผลทันทีที่กด OK)")
        form.addRow("ขนาดตัวอักษรโปรแกรม", self.ui_size)

        self.gpu = QCheckBox("ใช้ GPU สำหรับ OCR ญี่ปุ่นและ LaMa")
        self.gpu.setChecked(settings.use_gpu)
        form.addRow("", self.gpu)

        self.glossary = QPlainTextEdit("\n".join(f"{k} = {v}" for k, v in settings.glossary.items()))
        self.glossary.setPlaceholderText("ชื่อเฉพาะที่ต้องแปลแบบเดิมทุกครั้ง บรรทัดละคำ เช่น\nナルミ = นารุมิ")
        self.glossary.setFixedHeight(110)
        form.addRow("อภิธานศัพท์", self.glossary)
        self.notes = QPlainTextEdit(settings.story_notes)
        self.notes.setPlaceholderText("บริบทของเรื่อง เช่น แนวตลก ตัวละครเป็นนักเรียนมัธยม")
        self.notes.setFixedHeight(70)
        form.addRow("บริบทเรื่อง", self.notes)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _build_save_tab(self, s: Settings) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.autosave = QCheckBox("เซฟอัตโนมัติทุกครั้งที่ภาพผลลัพธ์เปลี่ยน")
        self.autosave.setChecked(s.autosave)
        self.autosave.setToolTip("หลังขั้น 'ใส่คำแปล' และหลังแก้คำแปลหรือย้ายข้อความเอง (รอ 1.5 วินาทีหลังแก้ครั้งสุดท้าย)")
        form.addRow("", self.autosave)
        self.save_dir = QLineEdit(s.save_dir)
        self.save_dir.setPlaceholderText("เช่น D:\\มังงะแปล")
        browse = QPushButton("เลือก…")
        browse.clicked.connect(self._pick_save_dir)
        row = QHBoxLayout()
        row.addWidget(self.save_dir, 1)
        row.addWidget(browse)
        form.addRow("โฟลเดอร์หลัก", row)
        form.addRow("", QLabel("<span style='color:#888'>แต่ละตอนจะอยู่ในโฟลเดอร์ย่อยชื่อเดียวกับโฟลเดอร์ต้นฉบับ</span>"))
        self.save_prefix = QLineEdit(s.save_prefix)
        self.save_prefix.setToolTip(f"{CHAPTER_TOKEN} = ชื่อโฟลเดอร์ต้นฉบับ (ชื่อตอน) · เว้นว่างได้")
        form.addRow("คำนำหน้า", self.save_prefix)
        self.save_start = QSpinBox()
        self.save_start.setRange(0, 99999)
        self.save_start.setValue(s.save_start)
        self.save_digits = QSpinBox()
        self.save_digits.setRange(1, 6)
        self.save_digits.setValue(s.save_digits)
        nrow = QHBoxLayout()
        nrow.addWidget(QLabel("เริ่มที่"))
        nrow.addWidget(self.save_start)
        nrow.addWidget(QLabel("จำนวนหลัก"))
        nrow.addWidget(self.save_digits)
        nrow.addStretch()
        form.addRow("เลขลำดับ", nrow)
        self.save_format = QComboBox()
        for key in FORMATS:
            self.save_format.addItem(key.upper(), key)
        self.save_format.setCurrentIndex(max(0, self.save_format.findData(s.save_format)))
        self.save_quality = QSpinBox()
        self.save_quality.setRange(50, 100)
        self.save_quality.setValue(s.save_quality)
        frow = QHBoxLayout()
        frow.addWidget(self.save_format)
        frow.addWidget(QLabel("คุณภาพ"))
        frow.addWidget(self.save_quality)
        frow.addStretch()
        form.addRow("ชนิดไฟล์", frow)
        self.save_preview = QLabel()
        self.save_preview.setWordWrap(True)
        form.addRow("ตัวอย่าง", self.save_preview)
        for w in (self.save_dir, self.save_prefix):
            w.textChanged.connect(self._update_save_preview)
        for w in (self.save_start, self.save_digits, self.save_quality):
            w.valueChanged.connect(self._update_save_preview)
        self.save_format.currentIndexChanged.connect(self._update_save_preview)
        self.autosave.toggled.connect(self._update_save_preview)
        self._update_save_preview()
        return page

    def _pick_save_dir(self):
        d = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์หลักสำหรับเซฟภาพแปล", self.save_dir.text())
        if d:
            self.save_dir.setText(d)

    def _update_save_preview(self):
        fmt = self.save_format.currentData()
        self.save_quality.setEnabled(fmt != "png")
        try:
            first = output_name(make_prefix(self.save_prefix.text(), "ตอน5"), self.save_start.value(),
                                self.save_digits.value(), fmt)
            second = output_name(make_prefix(self.save_prefix.text(), "ตอน5"), self.save_start.value() + 1,
                                 self.save_digits.value(), fmt)
        except ExportError as e:
            self.save_preview.setText(str(e))
            return
        root = self.save_dir.text().strip() or "<โฟลเดอร์หลัก>"
        text = f"{root}\\ตอน5\\{first}, {second}, …"
        if self.autosave.isChecked() and not self.save_dir.text().strip():
            text += "<br><span style='color:#c62828'>ต้องเลือกโฟลเดอร์หลักก่อน เซฟอัตโนมัติถึงจะทำงาน</span>"
        self.save_preview.setText(text)

    def _load_models(self):
        try:
            names = OllamaTranslator(TranslatorConfig(url=self.url.text().strip())).list_models()
        except Exception as e:
            self.model_status.setText(f"ติดต่อ Ollama ไม่ได้: {e}")
            return
        current = self.model.currentText()
        self.model.clear()
        self.model.addItems(names)
        self.model.setCurrentText(current)
        self.model_status.setText(f"พบ {len(names)} โมเดล")

    def apply(self) -> Settings:
        s = self.s
        s.ollama_url = self.url.text().strip()
        s.ollama_model = self.model.currentText().strip()
        s.ollama_keep_alive = self.keep_alive.text().strip() or "5m"
        s.font_family = "" if self.font_default.isChecked() else self.font.currentFont().family()
        s.text_color = self.color.value or "#000000"
        s.outline_color = self.outline.value
        s.use_gpu = self.gpu.isChecked()
        s.ui_font_size = self.ui_size.value()
        s.versions_keep = self.versions_keep.value()
        s.glossary = parse_glossary(self.glossary.toPlainText())
        s.story_notes = self.notes.toPlainText().strip()
        s.autosave = self.autosave.isChecked()
        s.save_dir = self.save_dir.text().strip()
        s.save_prefix = self.save_prefix.text()
        s.save_start = self.save_start.value()
        s.save_digits = self.save_digits.value()
        s.save_format = self.save_format.currentData()
        s.save_quality = self.save_quality.value()
        s.save()
        return s
