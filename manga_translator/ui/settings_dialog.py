"""Settings: Ollama model, default font and colours, glossary, story notes."""
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
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

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
        form = QFormLayout()

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

        lay.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

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
        s.glossary = parse_glossary(self.glossary.toPlainText())
        s.story_notes = self.notes.toPlainText().strip()
        s.save()
        return s
