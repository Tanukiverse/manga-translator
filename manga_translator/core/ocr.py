"""OCR per region: manga-ocr for Japanese (handles vertical text), English PP-OCR for English."""
from __future__ import annotations

import re
import unicodedata

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

    def __call__(self, img_bgr: np.ndarray, bbox, mask: np.ndarray | None = None) -> str:
        part = crop(img_bgr, bbox)
        if part.size == 0:
            return ""
        return self.model(Image.fromarray(part[:, :, ::-1])).strip()


EN_REC_MODEL = MODELS_DIR / "en_PP-OCRv3_rec_infer.onnx"


class EnglishOcr:
    """Lines are split from the text mask (comic-text-detector), then each line is read by the
    English PP-OCR recognizer. On a real page this halved the error rate of RapidOCR's own
    line detector + default Chinese/English recognizer, which also dropped whole lines."""

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR

        kw = {"rec_model_path": str(EN_REC_MODEL)} if EN_REC_MODEL.exists() else {}
        self.engine = RapidOCR(**kw)
        self._cjk = None  # RapidOCR's bundled Chinese/English model, loaded on first need
        self.last_score = 1.0  # mean recognition confidence of the last box read

    def __call__(self, img_bgr: np.ndarray, bbox, mask: np.ndarray | None = None) -> str:
        pad = 6
        part = crop(img_bgr, bbox, pad=pad)
        if part.size == 0:
            return ""
        if mask is not None:
            lines = text_lines(cv2.cvtColor(part, cv2.COLOR_BGR2GRAY), crop(mask, bbox, pad=pad))
            if lines:
                texts, scores = [], []
                for y0, y1, x0, x1 in lines:
                    lc = part[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2]
                    lc = cv2.copyMakeBorder(lc, 4, 4, 8, 8, cv2.BORDER_REPLICATE)
                    res, _ = self.engine(lc, use_det=False, use_cls=False)
                    if res and res[0][0].strip():
                        texts.append(res[0][0].strip())
                        scores.append(float(res[0][1]))
                if texts:
                    self.last_score = sum(scores) / len(scores)
                    return normalize_english(" ".join(texts))
        self.last_score = 0.0
        return self._detect_and_read(part)

    def looks_cjk(self, img_bgr: np.ndarray, bbox) -> bool:
        """True when the box holds Japanese/Chinese lettering. The English recognizer cannot read
        it and guesses Latin noise, so a recognizer that knows kanji/kana looks again."""
        if self._cjk is None:
            from rapidocr_onnxruntime import RapidOCR

            self._cjk = RapidOCR()
        part = cv2.copyMakeBorder(crop(img_bgr, bbox, pad=6), 20, 20, 20, 20, cv2.BORDER_REPLICATE)
        res, _ = self._cjk(part)
        text = "".join(r[1] for r in res or [])
        cjk = sum(is_cjk(c) for c in text)
        latin = sum(c.isascii() and c.isalpha() for c in text)
        return cjk >= 1 and cjk >= latin

    def _detect_and_read(self, part: np.ndarray) -> str:
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


def _runs(flags: np.ndarray, min_gap: int = 1) -> list[list[int]]:
    """[start, end) runs of True, joining runs separated by gaps shorter than min_gap."""
    runs: list[list[int]] = []
    start = None
    for i, v in enumerate(flags):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(flags)])
    merged: list[list[int]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < min_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return merged


def _row_lines(ink: np.ndarray) -> list[tuple[int, int]]:
    runs = _runs(ink.any(1))
    if not runs:
        return []
    med = sorted(b - a for a, b in runs)[len(runs) // 2]
    lines: list[list[int]] = []
    for a, b in runs:
        # glue thin fragments (dots, accents, the bottom of "!") to the line they touch
        close = lines and a - lines[-1][1] <= 0.3 * med
        thin = b - a < 0.4 * med or (lines and lines[-1][1] - lines[-1][0] < 0.4 * med)
        if close and thin:
            lines[-1][1] = b
        else:
            lines.append([a, b])
    return [(a, b) for a, b in lines]


def text_lines(gray: np.ndarray, mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Horizontal text lines (y0, y1, x0, x1) of dark lettering inside the text mask, in reading
    order. Side-by-side blocks in one bubble (e.g. "WAH!" next to a name) are split into columns
    first, so their lines are not glued together."""
    ink = (gray < 140) & (mask > 0)
    if not ink.any():
        return []
    rows = _row_lines(ink)
    line_h = sorted(b - a for a, b in rows)[len(rows) // 2] if rows else 10
    blocks = []
    for c0, c1 in _runs(ink.any(0), min_gap=max(8, int(1.2 * line_h))):
        sub = ink[:, c0:c1]
        ys = np.nonzero(sub.any(1))[0]
        blocks.append((ys.min(), c0, c1))
    blocks.sort(key=lambda b: (b[0] // max(1, line_h * 2), b[1]))  # top to bottom, then left to right
    out = []
    for _top, c0, c1 in blocks:
        sub = ink[:, c0:c1]
        for a, b in _row_lines(sub):
            if b - a > 1.6 * line_h:
                # two blocks offset vertically ("THE / INFANTRY" beside "CAPTAIN!") merge into one
                # tall row; split it at a narrower column gap and read each part separately
                parts = []
                for p0, p1 in _runs(sub[a:b].any(0), min_gap=max(4, int(0.5 * line_h))):
                    for a2, b2 in _row_lines(sub[a:b, p0:p1]):
                        parts.append((a + a2, a + b2, c0 + p0, c0 + p1))
                if len(parts) > 1:
                    parts.sort(key=lambda t: (t[0] // max(1, line_h), t[2]))
                    out.extend(_tight(ink, t) for t in parts)
                    continue
            cols = np.nonzero(sub[a:b].any(0))[0]
            if len(cols):
                out.append((a, b, c0 + int(cols.min()), c0 + int(cols.max()) + 1))
    return out


def _tight(ink: np.ndarray, line: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    a, b, c0, c1 = line
    cols = np.nonzero(ink[a:b, c0:c1].any(0))[0]
    return (a, b, c0 + int(cols.min()), c0 + int(cols.max()) + 1) if len(cols) else line


# interjections that are real words without a vowel
_VOWELLESS = re.compile(r"^(H+M+|M+|S+H+|Z+|G+R+|B+R+|T+S+K+|P+S+T+|N+|T+C+H+|P+F+T+|H+N+|C+H+)$")
_SYMBOLS = set("$#@%^&*=+<>|{}[]~\\_")


def is_junk_english(text: str) -> bool:
    """True when English OCR read noise (sound effects, art, a Japanese page) instead of lettering,
    e.g. "$7R6AH-7." or "S 6 MT 1 1 2 5". Punctuation-only bubbles ("?", "...") are real text."""
    t = re.sub(r"[\^`´]", "'", text.strip())  # misread apostrophes: "YOU^VE"
    if not any(c.isalnum() for c in t):
        return False
    tokens = re.findall(r"[A-Za-z0-9']+", t)
    words = [w.strip("'").upper() for w in tokens if w.strip("'").isalpha()]
    real = [w for w in words if (len(w) >= 2 and (re.search(r"[AEIOUY]", w) or _VOWELLESS.match(w)))
            or w in ("I", "A", "O")]
    if sum(len(w) >= 2 for w in real) >= 2:
        return False  # a sentence with a stray misread symbol ("RETAIN_MY") is still lettering
    if any(c in _SYMBOLS for c in t):
        return True
    if any(re.search(r"[A-Za-z]", w) and re.search(r"\d", w) for w in tokens):
        return True  # letters and digits mixed in one word
    if real and all(len(w) == 1 for w in real) and sum(c.isdigit() for c in t) > sum(c.isalpha() for c in t):
        return True  # "11 I 1 0 6 3": digits with a stray letter, e.g. vertical Japanese page numbers
    return not real


def is_cjk(c: str) -> bool:
    return "\u3040" <= c <= "\u30ff" or "\u4e00" <= c <= "\u9fff" or "\uff66" <= c <= "\uff9f"


def is_latin_only(text: str) -> bool:
    """manga-ocr output with Latin letters and no kana/kanji: an English (or other Latin) box
    inside a Japanese page."""
    t = unicodedata.normalize("NFKC", text)
    return any(c.isascii() and c.isalpha() for c in t) and not any(is_cjk(c) for c in t)


def uppercase_page(texts: list[str]) -> list[str]:
    """Comic lettering is usually all caps for the whole page: a short bubble read as "KOu-uu!"
    is upper-cased when most letters on the page are capitals."""
    letters = [c for t in texts for c in t if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) >= 0.85 * len(letters):
        return [t.upper() for t in texts]
    return texts


def normalize_english(text: str) -> str:
    """Clean OCR noise typical of comic lettering (all caps, hyphenated line breaks)."""
    text = unicodedata.normalize("NFKC", " ".join(text.split()))
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) >= 0.7 * len(letters):
        text = text.upper()  # lettering is all caps; stray lowercase is misread
        # O read for 0 inside numbers ("3OOO" -> "3000")
        text = re.sub(r"(?<=\d)O+|O+(?=\d)", lambda m: "0" * len(m.group()), text)
    # "COM- MAN- DER" (line breaks inside a word) -> "COM-MAN-DER"
    text = re.sub(r"(?<=\w)- (?=\w)", "-", text)
    return text
