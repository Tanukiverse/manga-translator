import numpy as np

from manga_translator.core.project import Region
from manga_translator.core.typeset import bubble_rect, fit_text, render, tokenize

TEXT = "วันนี้อากาศดีมากเลยนะ ไปเดินเล่นที่สวนสาธารณะกันไหม"


def _word_boundaries(text):
    pos, out = 0, {0}
    for tok in tokenize(text, "th"):
        pos += len(tok)
        out.add(pos)
    return out


def test_thai_wraps_only_between_words():
    lay = fit_text(TEXT, "th", "Leelawadee UI", 160, 220)
    assert lay.fits and len(lay.lines) > 1
    assert "".join(lay.lines).replace(" ", "") == TEXT.replace(" ", "")
    bounds = _word_boundaries(TEXT.replace(" ", ""))
    pos = 0
    for line in lay.lines[:-1]:
        pos += len(line.replace(" ", ""))
        assert pos in bounds, f"line break inside a word after {line!r}"


def test_colloquial_words_not_split():
    toks = tokenize("จริงเหรอ...?", "th")
    assert toks == ["จริง", "เหรอ...?"]


def test_fit_shrinks_for_small_box():
    big = fit_text(TEXT, "th", "Leelawadee UI", 300, 300)
    small = fit_text(TEXT, "th", "Leelawadee UI", 120, 120)
    assert small.size < big.size and small.fits


def test_english_tokens_keep_spaces():
    assert tokenize("HELLO  THERE", "en") == ["HELLO", " ", "THERE"]


def test_bubble_rect_inside_bubble(ja_page):
    img, truth = ja_page
    for t in truth:
        rx, ry, rw, rh = bubble_rect(img, t.bbox)
        bx, by, bw, bh = t.bubble
        assert bx <= rx and by <= ry and rx + rw <= bx + bw and ry + rh <= by + bh
        assert rw > t.bbox[2]  # wider than the vertical Japanese text box


def test_render_draws_inside_box_only():
    img = np.full((300, 300, 3), 255, np.uint8)
    r = Region(id=1, bbox=[50, 50, 200, 120], translation="สวัสดีครับ")
    out, overflow = render(img, [r], "th")
    assert overflow == []
    ink = np.argwhere(out.min(axis=2) < 128)
    assert len(ink) > 50
    assert ink[:, 0].min() >= 50 and ink[:, 0].max() < 170
    assert ink[:, 1].min() >= 50 and ink[:, 1].max() < 250
