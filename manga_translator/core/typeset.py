"""Draw translations into the cleaned page with Qt (HarfBuzz shapes Thai vowels and tone marks).

Thai has no spaces between words, so text is split into words with pythainlp and wrapped
only at word boundaries. The font size is the largest that fits the bubble interior.
"""
from __future__ import annotations

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

        return [t for t in word_tokenize(text, engine="newmm", keep_whitespace=True) if t]
    if lang in ("ja", "zh"):
        return list(text)
    out: list[str] = []
    for i, word in enumerate(text.split(" ")):
        if i:
            out.append(" ")
        out.append(word)
    return out


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


def fit_text(text: str, lang: str, family: str, rect_w: float, rect_h: float, fixed_size: int = 0) -> Layout:
    ensure_qt()
    from PySide6.QtGui import QFont, QFontMetricsF

    tokens = tokenize(text, lang)

    def layout(size: int, allow_break: bool) -> Layout:
        font = QFont(family)
        font.setPixelSize(size)
        fm = QFontMetricsF(font)
        lines, broke = wrap(tokens, fm, rect_w)
        lh = fm.height() * LINE_SPACING
        widest = max((fm.horizontalAdvance(l) for l in lines), default=0)
        fits = widest <= rect_w + 0.5 and lh * len(lines) <= rect_h and (allow_break or not broke)
        return Layout(size, lines, lh, fits)

    if fixed_size:
        return layout(fixed_size, True)

    def search(allow_break: bool) -> Layout | None:
        lo, hi = MIN_SIZE, max(MIN_SIZE, min(MAX_SIZE, int(rect_h)))
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
    return search(False) or search(True) or layout(MIN_SIZE, True)


def bubble_rect(img_bgr: np.ndarray, bbox, tol: int = 24) -> tuple[int, int, int, int]:
    """Usable text rectangle: the bubble the box sits in (flood fill from its centre), shrunk to its interior.

    Falls back to the box itself when the fill leaks into the page (no closed bubble).
    """
    x, y, w, h = bbox
    H, W = img_bgr.shape[:2]
    cx, cy = min(W - 1, x + w // 2), min(H - 1, y + h // 2)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if gray[cy, cx] < 128:
        return tuple(bbox)
    ff = np.zeros((H + 2, W + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    cv2.floodFill(gray.copy(), ff, (cx, cy), 0, tol, tol, flags)
    region = ff[1:-1, 1:-1]
    ys, xs = np.nonzero(region)
    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    bw, bh = bx1 - bx0, by1 - by0
    if bw * bh > 20 * max(1, w * h) or bw >= W * 0.9 or bh >= H * 0.9 or bx0 > x + 2 or by0 > y + 2:
        return tuple(bbox)
    # an ellipse's largest inscribed rectangle is ~70% of its bounding box per side
    fill_ratio = region.sum() / 255 / (bw * bh)
    k = 0.15 if fill_ratio < 0.9 else 0.06
    return (int(bx0 + bw * k), int(by0 + bh * k), int(bw * (1 - 2 * k)), int(bh * (1 - 2 * k)))


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
        rx, ry, rw, rh = bubble_rect(cleaned_bgr, r.bbox)
        family = r.font_family or family_default
        lay = fit_text(text, lang, family, rw, rh, r.font_size)
        if not lay.fits:
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
        if r.outline:
            pen = QPen(QColor(r.outline), max(2.0, lay.size / 7))
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.strokePath(path, pen)
        painter.fillPath(path, QColor(r.color or "#000000"))
    painter.end()

    qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
    ptr = qimg.constBits()
    arr = np.frombuffer(ptr, np.uint8).reshape(h, qimg.bytesPerLine())[:, : 3 * w].reshape(h, w, 3)
    return np.ascontiguousarray(arr[:, :, ::-1]), overflow
