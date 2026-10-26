"""Info & status page: what the app uses and what is running right now. Refreshes itself."""
from __future__ import annotations

import html
import os

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QTextBrowser, QVBoxLayout

from ..core import status
from ..core.paths import MODELS_DIR, ROOT, SETTINGS_PATH
from ..crashlog import LOG_PATH

REFRESH_MS = 3000

OK, BAD, IDLE = "#2e7d32", "#c62828", "#757575"


class _Probe(QObject):
    """Runs the status probes off the UI thread (Ollama/nvidia-smi can take a moment)."""

    done = Signal(object, object)

    @Slot(str)
    def run(self, url: str):
        self.done.emit(status.ollama_status(url), status.gpu_status())


class InfoDialog(QDialog):
    _probe_request = Signal(str)

    def __init__(self, main_window):
        super().__init__(main_window)
        self.win = main_window
        self.setWindowTitle("ข้อมูลและสถานะ")
        self.resize(760, 820)
        lay = QVBoxLayout(self)
        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(False)
        self.view.anchorClicked.connect(self._open_link)
        lay.addWidget(self.view, 1)
        row = QHBoxLayout()
        self.stamp = QLabel("")
        self.stamp.setStyleSheet("color:#888")
        row.addWidget(self.stamp, 1)
        refresh = QPushButton("รีเฟรช")
        refresh.clicked.connect(self.refresh)
        row.addWidget(refresh)
        close = QPushButton("ปิด")
        close.clicked.connect(self.close)
        row.addWidget(close)
        lay.addLayout(row)

        self._ollama = None
        self._gpu = None
        self._versions = status.versions()
        self._files = status.model_files()
        self._probing = False
        self._thread = QThread(self)
        self._probe = _Probe()
        self._probe.moveToThread(self._thread)
        self._probe_request.connect(self._probe.run)
        self._probe.done.connect(self._probed)
        self._thread.start()
        self._timer = QTimer(self, interval=REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        self.refresh()
        self._timer.start()

    # ---- data -------------------------------------------------------------
    def refresh(self):
        self._render()  # app-side state is instant; probes fill in when they return
        if not self._probing:
            self._probing = True
            self._probe_request.emit(self.win.settings.ollama_url)

    @Slot(object, object)
    def _probed(self, ollama, gpu):
        self._probing = False
        self._ollama, self._gpu = ollama, gpu
        self._render()

    def closeEvent(self, e):
        self._timer.stop()
        self._thread.quit()
        self._thread.wait(3000)
        super().closeEvent(e)

    def _open_link(self, url):
        path = url.toLocalFile() or url.toString()
        if os.path.exists(path):
            os.startfile(path)  # opens the folder in Explorer
        self.view.setSource(self.view.source())  # stay on this page

    # ---- page -------------------------------------------------------------
    def _render(self):
        from datetime import datetime

        scroll = self.view.verticalScrollBar().value()
        self.view.setHtml(self._html())
        self.view.verticalScrollBar().setValue(scroll)
        self.stamp.setText(f"อัปเดตทุก {REFRESH_MS // 1000} วินาที · ล่าสุด {datetime.now():%H:%M:%S}")

    def _html(self) -> str:
        w, s = self.win, self.win.settings
        e = html.escape
        dot = lambda color, text: f'<span style="color:{color}">●</span> {e(text)}'  # noqa: E731
        parts = ["<style>td{padding:2px 10px 2px 0} h3{margin:14px 0 4px 0}</style>"]

        # ---- now
        parts.append("<h3>สถานะตอนนี้</h3><table>")
        busy = dot(OK, "กำลังทำงาน: " + (w.statusBar().currentMessage() or "…")) if w.busy else dot(IDLE, "ว่าง")
        parts.append(f"<tr><td>งาน AI</td><td>{busy}</td></tr>")
        loaded = w.models.loaded()
        parts.append(f"<tr><td>โมเดลที่โปรแกรมโหลดไว้</td><td>{e(', '.join(loaded)) if loaded else 'ยังไม่ได้โหลด (โหลดเมื่อใช้ครั้งแรก)'}</td></tr>")
        g = self._gpu
        if g is None:
            gpu = "กำลังตรวจ…"
        elif g.ok:
            pct = 100 * g.used_mb / max(1, g.total_mb)
            color = BAD if pct > 92 else OK
            gpu = dot(color, f"{g.name} · VRAM {g.used_mb / 1024:.1f}/{g.total_mb / 1024:.1f} GB ({pct:.0f}%) · ใช้งาน {g.util}% · {g.temp}°C")
        else:
            gpu = dot(BAD, f"อ่านสถานะ GPU ไม่ได้ ({g.error})")
        parts.append(f"<tr><td>GPU</td><td>{gpu}</td></tr>")
        o = self._ollama
        if o is None:
            oll = "กำลังตรวจ…"
        elif o.ok:
            oll = dot(OK, f"เชื่อมต่อได้ · เวอร์ชัน {o.version} · {s.ollama_url}")
        else:
            oll = dot(BAD, f"เชื่อมต่อไม่ได้ที่ {s.ollama_url} (เปิดโปรแกรม Ollama หรือยัง?)")
        parts.append(f"<tr><td>Ollama</td><td>{oll}</td></tr>")
        if o and o.ok:
            running = ", ".join(f"{m['name']} ({m['vram_gb']:.1f} GB)" for m in o.running) or "ไม่มี"
            parts.append(f"<tr><td>โมเดลแปลที่อยู่ใน VRAM</td><td>{e(running)}</td></tr>")
        st, sv = w.settings, w.autosaver.status
        if not st.autosave:
            save = dot(IDLE, "ปิดอยู่ (เปิดได้ใน ⚙ ตั้งค่า → การเซฟ)")
        elif not st.save_dir:
            save = dot(BAD, "เปิดอยู่แต่ยังไม่ได้เลือกโฟลเดอร์หลัก")
        else:
            folder = sv.folder or (str(w.autosaver.chapter_dir(w.folder.name)) if w.folder else st.save_dir)
            save = dot(OK, f"เปิดอยู่ · {folder}")
            if w.work_dir:
                save += f"<br>เซฟแล้ว {len(w.autosaver.saved_pages(w.work_dir))} หน้า"
            if sv.last_file:
                save += f" · ล่าสุด {e(sv.last_file)}"
            if sv.last_error:
                save += "<br>" + dot(BAD, sv.last_error)
        parts.append(f"<tr><td valign=top>เซฟอัตโนมัติ</td><td>{save}</td></tr>")
        if w.images:
            failed = len(w.failed)
            line = f"{len(w.images)} หน้า · เรียงตาม{e(w.sort_combo.currentText())}"
            if failed:
                line += " · " + dot(BAD, f"พังรอบล่าสุด {failed} หน้า")
            parts.append(f"<tr><td>ตอนนี้</td><td>{e(w.folder.name)} · {line}</td></tr>")
        if w.doc:
            done = " · ".join(("✓ " if w.doc.done.get(k) else "· ") + v for k, v in _step_labels())
            parts.append(f"<tr><td>หน้าปัจจุบัน</td><td>{e(w.doc.folder.name)} · {len(w.doc.regions)} กล่อง<br>{e(done)}</td></tr>")
        parts.append("</table>")

        # ---- translator
        parts.append("<h3>ตัวแปล</h3><table>")
        parts.append(f"<tr><td>โมเดลที่เลือก</td><td><b>{e(s.ollama_model)}</b></td></tr>")
        if o and o.ok:
            names = [m["name"] for m in o.installed]
            if s.ollama_model not in names:
                parts.append(f"<tr><td></td><td>{dot(BAD, 'ยังไม่มีโมเดลนี้ใน Ollama')}</td></tr>")
            rows = "<br>".join(("<b>▶ </b>" if m["name"] == s.ollama_model else "&nbsp;&nbsp;&nbsp;")
                               + f"{e(m['name'])} ({m['size_gb']:.1f} GB)" for m in o.installed)
            parts.append(f"<tr><td valign=top>โมเดลใน Ollama</td><td>{rows or 'ไม่มี'}</td></tr>")
        parts.append(f"<tr><td>ภาษา</td><td>{e(s.source_lang)} → {e(s.target_lang)}</td></tr>")
        parts.append(f"<tr><td>อภิธานศัพท์</td><td>{len(s.glossary)} คำ</td></tr>")
        parts.append(f"<tr><td>ค้างโมเดลไว้</td><td>{e(s.ollama_keep_alive)}</td></tr>")
        parts.append("</table>")

        # ---- model files
        parts.append("<h3>ไฟล์โมเดลของโปรแกรม</h3><table>")
        for f in self._files:
            mark = dot(OK, f"{f['size_mb']:.0f} MB") if f["present"] else dot(BAD, "ไม่พบ (รัน tools\\download_models.py)")
            parts.append(f"<tr><td>{e(f['purpose'])}</td><td>{mark}</td></tr>")
        parts.append("</table>")
        parts.append("<p style='color:#888'>ตัดคำไทย: pythainlp (newmm) · วาดตัวอักษร: Qt · "
                     f"ฟอนต์: {e(s.font_family or 'ตามภาษา (ไทย: Leelawadee UI)')} · GPU สำหรับ OCR ญี่ปุ่น/LaMa: "
                     f"{'เปิด' if s.use_gpu else 'ปิด'}</p>")

        # ---- versions
        parts.append("<h3>เวอร์ชันซอฟต์แวร์</h3><table>")
        for k, v in self._versions.items():
            parts.append(f"<tr><td>{e(k)}</td><td>{e(v)}</td></tr>")
        parts.append("</table>")

        # ---- folders
        work = getattr(w, "work_dir", None) or s.work_dir
        parts.append("<h3>ไฟล์และโฟลเดอร์</h3><table>")
        for label, path in [("โฟลเดอร์งาน", work), ("โมเดล", MODELS_DIR), ("ตั้งค่า", SETTINGS_PATH),
                            ("บันทึก error", LOG_PATH), ("โปรแกรม", ROOT)]:
            target = path if os.path.isdir(path) else os.path.dirname(path)
            parts.append(f"<tr><td>{label}</td><td><a href='file:///{e(str(target))}'>{e(str(path))}</a></td></tr>")
        parts.append("</table>")
        return "".join(parts)


def _step_labels():
    from ..core.pipeline import STEP_LABELS, STEPS

    return [(k, STEP_LABELS[k]) for k in STEPS]
