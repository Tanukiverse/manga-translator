"""The five editable steps. Each step reads/writes the PageDoc and its image layers, so any
step can be re-run after the user edits the previous one. Locked regions are never overwritten.
"""
from __future__ import annotations

import gc
from typing import Callable

import cv2
import numpy as np

from .project import PageDoc, Region
from .settings import Settings
from .translate import OllamaTranslator, TranslatorConfig

LIGHT_TEXT_GROW = 6  # extra mask growth (px) for light lettering; 0 left a white smear, 12 a small dent

STEPS = ["detect", "ocr", "translate", "clean", "typeset"]
STEP_LABELS = {
    "detect": "ตรวจจับข้อความ",
    "ocr": "อ่านข้อความ",
    "translate": "แปล",
    "clean": "ลบคำ",
    "typeset": "ใส่คำแปล",
}

Progress = Callable[[str], None]


class Models:
    """Lazily loaded models, one GPU model at a time to stay inside 8 GB VRAM."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._detector = None
        self._ocr: dict[str, object] = {}
        self._lama = None

    @property
    def detector(self):
        if self._detector is None:
            from .detect import TextDetector

            self._detector = TextDetector()
        return self._detector

    def ocr(self, lang: str):
        if lang not in self._ocr:
            if lang == "ja":
                self.release_lama()
                from .ocr import JapaneseOcr

                self._ocr[lang] = JapaneseOcr(force_cpu=not self.settings.use_gpu)
            else:
                from .ocr import EnglishOcr

                self._ocr[lang] = EnglishOcr()
        return self._ocr[lang]

    @property
    def lama(self):
        if self._lama is None:
            self._ocr.pop("ja", None)
            self._free_cuda()
            from .inpaint import Lama

            self._lama = Lama(device=None if self.settings.use_gpu else "cpu")
        return self._lama

    def release_lama(self) -> None:
        if self._lama is not None:
            self._lama.release()
            self._lama = None

    def loaded(self) -> list[str]:
        """Models currently held in memory by this app (for the info page)."""
        names = []
        if self._detector is not None:
            names.append("ตรวจจับข้อความ")
        if "ja" in self._ocr:
            names.append("OCR ญี่ปุ่น (GPU)" if self.settings.use_gpu else "OCR ญี่ปุ่น (CPU)")
        if "en" in self._ocr:
            names.append("OCR อังกฤษ")
        if self._lama is not None:
            names.append(f"LaMa ({self._lama.device})")
        return names

    def release_all(self) -> None:
        self._detector = None
        self._ocr.clear()
        self.release_lama()
        self._free_cuda()

    @staticmethod
    def _free_cuda() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def translator(self) -> OllamaTranslator:
        s = self.settings
        return OllamaTranslator(
            TranslatorConfig(
                url=s.ollama_url,
                model=s.ollama_model,
                keep_alive=s.ollama_keep_alive,
                glossary=s.glossary,
                notes=s.story_notes,
            )
        )


def _log(progress: Progress | None, msg: str) -> None:
    if progress:
        progress(msg)


def keep_skipped(mask: np.ndarray, regions, pad: int = 6) -> np.ndarray:
    """Remove "not text" boxes from the removal mask (but not where a real box overlaps them)."""
    skipped = [r for r in regions if r.skip]
    if not skipped:
        return mask
    mask = mask.copy()
    keep = np.zeros(mask.shape, bool)
    for r in regions:
        if not r.skip:
            x, y, w, h = r.bbox
            keep[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad] = True
    for r in skipped:
        x, y, w, h = r.bbox
        sl = (slice(max(0, y - pad), y + h + pad), slice(max(0, x - pad), x + w + pad))
        mask[sl][~keep[sl]] = 0
    return mask


def _invalidate_after(doc: PageDoc, step: str) -> None:
    for s in STEPS[STEPS.index(step):]:
        doc.done[s] = False


def run_detect(doc: PageDoc, models: Models, progress: Progress | None = None) -> None:
    from .detect import sort_reading_order

    img = doc.load_original()
    dets, mask = models.detector(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if doc.source_lang != "ja":  # horizontal text: split boxes holding two blocks side by side
        from .detect import Detection, split_side_by_side

        dets = [Detection(b, d.score) for d in dets for b in split_side_by_side(d.bbox, mask, gray)]
    locked = [r for r in doc.regions if r.locked]
    # keep locked regions; drop new detections that overlap them
    fresh = [d for d in dets if not any(_iou(d.bbox, r.bbox) > 0.3 for r in locked)]
    order = sort_reading_order([d.bbox for d in fresh], right_to_left=doc.source_lang == "ja")
    regions = list(locked)
    next_id = max((r.id for r in locked), default=0) + 1
    from .detect import grow_mask_in, is_light_text

    light = 0
    for i in order:
        x, y, w, h = fresh[i].bbox
        region = Region(
            id=next_id,
            bbox=[x, y, w, h],
            vertical=doc.source_lang == "ja" and h > w * 1.2,
            color=models.settings.text_color or "#000000",
            outline=models.settings.outline_color,
        )
        if is_light_text(gray, mask, region.bbox):
            # white lettering on dark art: remove its halo too, and draw the translation white
            grow_mask_in(mask, region.bbox, LIGHT_TEXT_GROW)
            region.color, region.outline = "#ffffff", "#000000"
            light += 1
        regions.append(region)
        next_id += 1
    doc.regions = regions
    old_mask = doc.load_layer("mask")
    if old_mask is not None and locked:
        # keep the user's mask strokes inside locked regions
        for r in locked:
            x, y, w, h = r.bbox
            mask[y:y + h, x:x + w] = old_mask[y:y + h, x:x + w]
    doc.save_layer("mask", mask)
    _invalidate_after(doc, "detect")
    doc.done["detect"] = True
    doc.save()
    _log(progress, f"พบข้อความ {len(fresh)} กล่อง" + (f" (ตัวหนังสือสว่างบนพื้นมืด {light})" if light else ""))


def run_ocr(doc: PageDoc, models: Models, progress: Progress | None = None) -> None:
    img = doc.load_original()
    mask = doc.load_layer("mask")
    engine = models.ocr(doc.source_lang)
    todo = [r for r in doc.regions if not r.locked]
    for n, r in enumerate(todo, 1):
        r.source_text = engine(img, r.bbox, mask)
        _log(progress, f"อ่านข้อความ {n}/{len(todo)}")
    if doc.source_lang == "en":
        from .ocr import uppercase_page

        for r, t in zip(todo, uppercase_page([r.source_text for r in todo])):
            r.source_text = t
    for r in todo:
        r.skip_reason = _skip_reason(doc.source_lang, r, engine, img)
        r.skip = bool(r.skip_reason)
    skipped = [r for r in todo if r.skip]
    if skipped:
        foreign = sum(r.skip_reason == "foreign" for r in skipped)
        _log(progress, f"ข้าม {len(skipped)} กล่อง (ภาษาอื่น {foreign}, ไม่ใช่ข้อความ {len(skipped) - foreign}) ไม่ลบ ไม่แปล")
    _invalidate_after(doc, "ocr")
    doc.done["ocr"] = True
    doc.save()


def _skip_reason(lang: str, r: Region, engine, img) -> str:
    """Why a box should be left untouched: "foreign" (not the source language), "noise"
    (OCR read art or sound effects), "empty", or "" to translate it."""
    from .ocr import is_junk_english, is_latin_only

    text = r.source_text.strip()
    if lang == "ja":
        if not text:
            return "empty"
        return "foreign" if is_latin_only(text) else ""
    if lang == "en":
        junk = not text or is_junk_english(text)
        # a Japanese box read by the English recognizer comes out as junk or with low confidence;
        # only then is the slower kanji/kana check worth it
        if (junk or getattr(engine, "last_score", 1.0) < 0.85) and engine.looks_cjk(img, r.bbox):
            return "foreign"
        if junk:
            return "noise" if text else "empty"
    return ""


def run_translate(doc: PageDoc, models: Models, progress: Progress | None = None) -> None:
    translator = models.translator()
    lines = [(r.id, r.source_text) for r in doc.regions if r.source_text.strip() and not r.skip]
    _log(progress, f"กำลังแปล {len(lines)} กล่องด้วย {translator.cfg.model} …")
    result = translator.translate(lines, doc.source_lang, doc.target_lang)
    for r in doc.regions:
        if not r.locked and r.id in result:
            r.translation = result[r.id]
    _invalidate_after(doc, "translate")
    doc.done["translate"] = True
    doc.save()
    msg = f"แปลแล้ว {len(result)}/{len(lines)} กล่อง"
    if translator.wrong_language:
        msg += (f" (กล่อง {', '.join(map(str, translator.wrong_language))} ยังไม่ใช่ภาษาเป้าหมาย"
                f" โมเดล {translator.cfg.model} อาจไม่ทำตามคำสั่งเรื่องภาษา)")
    _log(progress, msg)


def run_clean(doc: PageDoc, models: Models, progress: Progress | None = None) -> None:
    from .inpaint import clean

    img = doc.load_original()
    mask = doc.load_layer("mask")
    if mask is None:
        mask = np.zeros(img.shape[:2], np.uint8)
    mask = keep_skipped(mask, doc.regions)
    # load LaMa only when some blob actually needs it
    _, probe = clean(img, mask, lama=None)
    lama = models.lama if probe["telea"] else None
    cleaned, stats = clean(img, mask, lama=lama)
    doc.save_layer("cleaned", cleaned)
    _invalidate_after(doc, "clean")
    doc.done["clean"] = True
    doc.save()
    _log(progress, f"ลบคำแล้ว (พื้นเรียบ {stats['flat']}, LaMa {stats['lama']})")


def run_typeset(doc: PageDoc, models: Models, progress: Progress | None = None) -> list[int]:
    from .typeset import render

    cleaned = doc.load_layer("cleaned")
    if cleaned is None:
        cleaned = doc.load_original()
    final, overflow = render(cleaned, doc.regions, doc.target_lang, models.settings.font_family)
    doc.save_layer("final", final)
    doc.done["typeset"] = True
    doc.save()
    msg = "ใส่คำแปลแล้ว"
    if overflow:
        msg += f" (ข้อความล้นกรอบ: กล่อง {', '.join(map(str, overflow))})"
    _log(progress, msg)
    return overflow


RUNNERS = {
    "detect": run_detect,
    "ocr": run_ocr,
    "translate": run_translate,
    "clean": run_clean,
    "typeset": run_typeset,
}


def run_all(doc: PageDoc, models: Models, progress: Progress | None = None, start: str = "detect") -> None:
    for step in STEPS[STEPS.index(start):]:
        _log(progress, f"[{STEP_LABELS[step]}]")
        RUNNERS[step](doc, models, progress)


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0
