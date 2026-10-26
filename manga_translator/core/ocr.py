"""OCR per region: manga-ocr for Japanese (handles vertical text), RapidOCR for English."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .paths import MODELS_DIR


def crop(img_bgr: np.ndarray, bbox, pad: int = 4) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x, y, bw, bh = bbox
    return img_bgr[max(0, y - pad):min(h, y + bh + pad), max(0, x - pad):min(w, x + bw + pad)]


class JapaneseOcr:
    def __init__(self, force_cpu: bool = False):
        from manga_ocr import MangaOcr

        self.model = MangaOcr(pretrained_model_name_or_path=str(MODELS_DIR / "manga-ocr-base"), force_cpu=force_cpu)

    def __call__(self, img_bgr: np.ndarray, bbox) -> str:
        part = crop(img_bgr, bbox)
        if part.size == 0:
            return ""
        return self.model(Image.fromarray(part[:, :, ::-1])).strip()


class EnglishOcr:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR

        self.engine = RapidOCR()

    def __call__(self, img_bgr: np.ndarray, bbox) -> str:
        part = crop(img_bgr, bbox, pad=6)
        if part.size == 0:
            return ""
        # a margin around tight crops makes the line detector keep words together
        part = cv2.copyMakeBorder(part, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
        result, _ = self.engine(part)
        if not result:
            return ""
        # result: [[box(4 points), text, score], ...]; group boxes into rows, rows top to bottom
        items = []
        for box, text, _score in result:
            ys = [p[1] for p in box]
            items.append(((min(ys) + max(ys)) / 2, max(ys) - min(ys), min(p[0] for p in box), text.strip()))
        items.sort()
        rows: list[list] = []
        for it in items:
            if rows and abs(it[0] - rows[-1][0][0]) <= 0.5 * max(it[1], rows[-1][0][1]):
                rows[-1].append(it)
            else:
                rows.append([it])
        words = [it[3] for row in rows for it in sorted(row, key=lambda i: i[2]) if it[3]]
        return normalize_english(" ".join(words))


def normalize_english(text: str) -> str:
    return " ".join(text.split())
