"""Draw translations into the cleaned page with Qt (HarfBuzz shapes Thai vowels and tone marks).

Thai has no spaces between words, so text is split into words with pythainlp and wrapped
only at word boundaries. The font size is the largest that fits the bubble interior.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

from .paths import FONTS_DIR

MIN_SIZE = 8
MAX_SIZE = 72
LINE_SPACING = 1.12

DEFAULT_FONTS = {"th": "Leelawadee UI", "en": "Comic Sans MS", "ja": "Yu Gothic"}

_app = None
_fonts_loaded = False


def ensure_qt():
    """Qt fonts need a QGuiApplication; create one for CLI/tests if the UI has not."""
    global _app, _fonts_loaded
    from PySide6.QtGui import QFontDatabase, QGuiApplication

    if QGuiApplication.instance() is None:
        _app = QGuiApplication([])
    if not _fonts_loaded:
        for f in list(FONTS_DIR.glob("*.ttf")) + list(FONTS_DIR.glob("*.otf")):
            QFontDatabase.addApplicationFont(str(f))
        _fonts_loaded = True


def font_families() -> list[str]:
    ensure_qt()
    from PySide6.QtGui import QFontDatabase

    return QFontDatabase.families()


def tokenize(text: str, lang: str) -> list[str]:
    """Split into wrap units. Spaces are kept as their own tokens."""
    text = " ".join(text.split())
    if lang == "th":
        from pythainlp.tokenize import word_tokenize

        # tokenize Thai runs only: punctuation inside a run confuses newmm ("เหรอ..." -> "เห|รอ.")
        out: list[str] = []
        for run in re.findall(r"[฀-๿]+|\s+|[^฀-๿\s]+", text):
            if run.isspace():
                out.append(" ")
            elif "฀" <= run[0] <= "๿":
                out.extend(t for t in word_tokenize(run, custom_dict=_thai_dict(), engine="newmm") if t)
            elif out and out[-1] != " ":
                out[-1] += run  # keep "?!…" on the word they follow, never at a line start
            else:
                out.append(run)
        return out
    if lang in ("ja", "zh"):
        return list(text)
    out: list[str] = []
    for i, word in enumerate(text.split(" ")):
        if i:
            out.append(" ")
        out.append(word)
    return out


# colloquial words common in manga dialogue that the default dictionary splits (เหรอ -> เห|รอ)
THAI_EXTRA_WORDS = [
    "เหรอ", "หรอ", "หรือเปล่า", "รึเปล่า", "เปล่า", "ป่ะ", "มั้ย", "ไหมล่ะ", "จ้ะ", "จ๊ะ", "จ้า", "ค่ะ", "คะ", "นะคะ",
    "นะครับ", "ครับผม", "ล่ะ", "เนอะ", "เนี่ย", "นี่", "ไง", "งั้น", "งี้", "อ่ะ", "อะ", "เว้ย", "ว่ะ", "โว้ย", "ฮะ",
    "คับ", "เฮ้ย", "เอ๊ะ", "อ้าว", "โอ้", "ว้าว", "อืม", "ฮึ", "เฮ้อ", "แหละ", "หรอก", "สิ", "ซะ", "เถอะ", "น่ะ",
]
_dict = None


def _thai_dict():
    global _dict
    if _dict is None:
        from pythainlp.corpus import thai_words
        from pythainlp.util import Trie

        _dict = Trie(set(thai_words()) | set(THAI_EXTRA_WORDS))
    return _dict


def max_size_for(page_shape) -> int:
    """Upper bound for auto font size so short lines do not become poster-sized."""
    return max(14, min(MAX_SIZE, round(min(page_shape[:2]) * 0.045)))


def _split_long(token: str, fm, width: float) -> list[str]:
    """Break a single token wider than the box at grapheme boundaries (never inside a Thai cluster)."""
    from PySide6.QtCore import QTextBoundaryFinder

    finder = QTextBoundaryFinder(QTextBoundaryFinder.BoundaryType.Grapheme, token)
    bounds = [0]
    while (p := finder.toNextBoundary()) != -1:
        bounds.append(p)
    pieces, start = [], 0
    for a, b in zip(bounds, bounds[1:]):
        if fm.horizontalAdvance(token[start:b]) > width and a > start:
            pieces.append(token[start:a])
            start = a
    pieces.append(token[start:])
    return pieces


def wrap(tokens: list[str], fm, width: float) -> tuple[list[str], bool]:
    """Greedy wrap at token boundaries. Returns (lines, broke_a_word)."""
    lines: list[str] = []
    broke = False
    cur = ""
    for tok in tokens:
        if tok == " " and not cur:
            continue
        cand = cur + tok
        if fm.horizontalAdvance(cand.rstrip()) <= width:
            cur = cand
            continue
        if cur.strip():
            lines.append(cur.rstrip())
        cur = "" if tok == " " else tok
        if fm.horizontalAdvance(cur) > width:
            parts = _split_long(cur, fm, width)
            broke = broke or len(parts) > 1
            lines.extend(parts[:-1])
            cur = parts[-1]
    if cur.strip():
        lines.append(cur.rstrip())
    return lines, broke


@dataclass
class Layout:
    size: int
    lines: list[str]
    line_height: float
    fits: bool
    broke: bool = False  # a word had to be split across lines


def paragraphs(text: str) -> list[str]:
    """Lines the user ended with Enter; each is wrapped on its own."""
    return [" ".join(p.split()) for p in text.strip().splitlines()] or [""]


def fit_text(text: str, lang: str, family: str, rect_w: float, rect_h: float, fixed_size: int = 0,
             max_size: int = MAX_SIZE, min_size: int = MIN_SIZE, manual: bool = False) -> Layout:
    """Largest font size that fits. A line break typed by the user always starts a new line;
    manual=True breaks only there (the font shrinks until the longest line fits)."""
    ensure_qt()
    from PySide6.QtGui import QFont, QFontMetricsF

    paras = paragraphs(text)
    para_tokens = [tokenize(p, lang) for p in paras]

    def layout(size: int, allow_break: bool) -> Layout:
        font = QFont(family)
        font.setPixelSize(size)
        fm = QFontMetricsF(font)
        lines, broke = [], False
        for para, tokens in zip(paras, para_tokens):
            if manual or not para:
                lines.append(para)
                continue
            plines, pbroke = wrap(tokens, fm, rect_w)
            lines.extend(plines)
            broke = broke or pbroke
        lh = fm.height() * LINE_SPACING
        widest = max((fm.horizontalAdvance(l) for l in lines), default=0)
        fits = widest <= rect_w + 0.5 and lh * len(lines) <= rect_h and (allow_break or not broke)
        return Layout(size, lines, lh, fits, broke)

    if fixed_size:
        return layout(fixed_size, True)

    def search(allow_break: bool) -> Layout | None:
        lo, hi = min_size, max(min_size, min(max_size, int(rect_h)))
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            lay = layout(mid, allow_break)
            if lay.fits:
                best, lo = lay, mid + 1
            else:
                hi = mid - 1
        return best

    # prefer whole words; break inside a word only when nothing else fits
    return search(False) or search(True) or layout(min_size, True)


def _bubble_mask(img_bgr: np.ndarray, bbox, tol: int = 24) -> np.ndarray | None:
    """Interior of the bubble the box sits in (flood fill from the box centre on the cleaned page),
    or None when there is no closed light bubble (the fill leaks into the page)."""
    x, y, w, h = bbox
    H, W = img_bgr.shape[:2]
    cx, cy = min(W - 1, x + w // 2), min(H - 1, y + h // 2)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if gray[cy, cx] < 128:
        return None
    ff = np.zeros((H + 2, W + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    cv2.floodFill(gray, ff, (cx, cy), 0, tol, tol, flags)
    region = ff[1:-1, 1:-1]
    ys, xs = np.nonzero(region)
    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    bw, bh = bx1 - bx0, by1 - by0
    # a short word in a big bubble is normal; a fill covering much of the page is a leak
    if bw * bh > 0.25 * W * H or bw >= W * 0.9 or bh >= H * 0.9 or bx0 > x + 2 or by0 > y + 2:
        return None
    return region


def _inscribed_rect(region: np.ndarray, cx: int, cy: int) -> tuple[int, int, int, int]:
    """Grow a rectangle from (cx, cy) one side at a time while it stays inside the region.
    Unlike the bounding box, this ignores spikes and tails of jagged bubbles."""
    outside = cv2.integral((region == 0).astype(np.uint8))

    def bad(x0, y0, x1, y1):
        return outside[y1, x1] - outside[y0, x1] - outside[y1, x0] + outside[y0, x0]

    H, W = region.shape
    x0, y0, x1, y1 = cx, cy, cx + 1, cy + 1
    for step in (8, 2, 1):
        grew = True
        while grew:
            grew = False
            if x0 - step >= 0 and bad(x0 - step, y0, x0, y1) <= 0.02 * step * (y1 - y0):
                x0 -= step
                grew = True
            if x1 + step <= W and bad(x1, y0, x1 + step, y1) <= 0.02 * step * (y1 - y0):
                x1 += step
                grew = True
            if y0 - step >= 0 and bad(x0, y0 - step, x1, y0) <= 0.02 * step * (x1 - x0):
                y0 -= step
                grew = True
            if y1 + step <= H and bad(x0, y1, x1, y1 + step) <= 0.02 * step * (x1 - x0):
                y1 += step
                grew = True
    return x0, y0, x1 - x0, y1 - y0


def _shrink(rect, ratio: float, min_px: int = 2) -> tuple[int, int, int, int]:
    x, y, w, h = rect
    dx, dy = max(min_px, int(w * ratio)), max(min_px, int(h * ratio))
    if w - 2 * dx < 4 or h - 2 * dy < 4:
        return (int(x), int(y), int(w), int(h))
    return (int(x + dx), int(y + dy), int(w - 2 * dx), int(h - 2 * dy))


def bubble_rects(img_bgr: np.ndarray, bbox) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """(inner, loose) text rectangles for a region.

    inner: largest rectangle grown inside the bubble from the box centre, with a margin.
    loose: the bubble's bounding box with a small margin; used when text will not fit inside.
    Both fall back to the box itself when there is no closed bubble.
    """
    region = _bubble_mask(img_bgr, bbox)
    if region is None:
        return tuple(bbox), tuple(bbox)
    x, y, w, h = bbox
    ys, xs = np.nonzero(region)
    loose = _shrink((xs.min(), ys.min(), xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min()), 0.04)
    cx, cy = x + w // 2, y + h // 2
    if not region[cy, cx]:
        return tuple(bbox), loose
    inner = _shrink(_inscribed_rect(region, cx, cy), 0.06)
    if inner[2] * inner[3] < 0.5 * w * h:
        inner = tuple(bbox)
    return inner, loose


def bubble_rect(img_bgr: np.ndarray, bbox) -> tuple[int, int, int, int]:
    return bubble_rects(img_bgr, bbox)[0]


def min_size_for(page_shape) -> int:
    """Smallest auto font size that stays readable on this page."""
    return max(MIN_SIZE, round(min(page_shape[:2]) * 0.011))


def choose_layout(text: str, lang: str, family: str, inner, loose, fixed_size: int, page_shape, manual: bool = False):
    """Pick (rect, layout, overflow): inside the bubble at a readable size if possible, then the
    looser bubble box, then a readable size spilling past the bubble (drawn with a halo)."""
    floor, cap = min_size_for(page_shape), max_size_for(page_shape)
    if fixed_size:
        lay = fit_text(text, lang, family, inner[2], inner[3], fixed_size, manual=manual)
        return inner, lay, not lay.fits
    for rect in (inner, loose):
        lay = fit_text(text, lang, family, rect[2], rect[3], 0, cap, floor, manual)
        if lay.fits and not lay.broke:
            return rect, lay, False
    # too long for the bubble: keep it readable and let it spill sideways, centred on the bubble
    x, y, w, h = loose
    for widen in (1.3, 1.6, 2.0, 2.6, 3.2):
        nw = int(w * widen)
        rect = (int(x + w / 2 - nw / 2), y, nw, h)
        lay = fit_text(text, lang, family, rect[2], rect[3], 0, cap, floor, manual)
        if lay.fits and not lay.broke:
            return rect, lay, True
    lay = fit_text(text, lang, family, loose[2], loose[3], 0, cap, manual=manual)
    return loose, lay, True


def place(cleaned_bgr: np.ndarray, r, lang: str, default_family: str = ""):
    """(rect, layout, spilled) for one region: the user's own text box when they moved it,
    otherwise the automatic fit inside the bubble."""
    family = r.font_family or default_family or DEFAULT_FONTS.get(lang, "Arial")
    text = (r.translation or "").strip()
    manual = r.wrap == "manual"
    if r.text_rect:
        x, y, w, h = r.text_rect
        lay = fit_text(text, lang, family, w, h, r.font_size, max_size_for(cleaned_bgr.shape), manual=manual)
        return (x, y, w, h), lay, False
    inner, loose = bubble_rects(cleaned_bgr, r.bbox)
    return choose_layout(text, lang, family, inner, loose, r.font_size, cleaned_bgr.shape, manual)


def render(cleaned_bgr: np.ndarray, regions, lang: str, default_family: str = "") -> tuple[np.ndarray, list[int]]:
    """Draw every region's translation. Returns (image, ids of regions whose text did not fit)."""
    ensure_qt()
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen

    h, w = cleaned_bgr.shape[:2]
    rgb = np.ascontiguousarray(cleaned_bgr[:, :, ::-1])
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    painter = QPainter(qimg)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    overflow: list[int] = []
    family_default = default_family or DEFAULT_FONTS.get(lang, "Arial")

    for r in regions:
        text = (r.translation or "").strip()
        if not text:
            continue
        family = r.font_family or family_default
        (rx, ry, rw, rh), lay, spilled = place(cleaned_bgr, r, lang, default_family)
        if spilled or not lay.fits:
            overflow.append(r.id)
        font = QFont(family)
        font.setPixelSize(lay.size)
        from PySide6.QtGui import QFontMetricsF

        fm = QFontMetricsF(font)
        total = lay.line_height * len(lay.lines)
        top = ry + (rh - total) / 2
        path = QPainterPath()
        for i, line in enumerate(lay.lines):
            lw = fm.horizontalAdvance(line)
            if r.align == "left":
                lx = rx
            elif r.align == "right":
                lx = rx + rw - lw
            else:
                lx = rx + (rw - lw) / 2
            baseline = top + i * lay.line_height + (lay.line_height - fm.height()) / 2 + fm.ascent()
            path.addText(QPointF(lx, baseline), font, line)
        halo = r.outline or ("#ffffff" if spilled else "")  # text spilling onto artwork needs a halo
        if halo:
            pen = QPen(QColor(halo), max(2.0, lay.size / 7))
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.strokePath(path, pen)
        painter.fillPath(path, QColor(r.color or "#000000"))
    painter.end()

    qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
    ptr = qimg.constBits()
    arr = np.frombuffer(ptr, np.uint8).reshape(h, qimg.bytesPerLine())[:, : 3 * w].reshape(h, w, 3)
    return np.ascontiguousarray(arr[:, :, ::-1]), overflow
