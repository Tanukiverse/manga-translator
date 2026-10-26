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
    img = img.copy()
    for t in truth:  # layout runs on the cleaned page: remove the source text first
        x, y, w, h = t.bbox
        img[y:y + h, x:x + w] = 255
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


def _spiky_bubble():
    import cv2

    img = np.full((1000, 1000, 3), 120, np.uint8)  # page-sized: a bubble is a small part of it
    pts = []
    for i in range(24):  # star-shaped "shout" bubble: long spikes, round core
        ang = i * np.pi / 12
        rad = 170 if i % 2 else 110
        pts.append([500 + rad * np.cos(ang), 500 + rad * np.sin(ang)])
    cv2.fillPoly(img, [np.array(pts, np.int32)], (255, 255, 255))
    cv2.polylines(img, [np.array(pts, np.int32)], True, (0, 0, 0), 3)
    return img


def test_inner_rect_stays_inside_spiky_bubble():
    from manga_translator.core.typeset import bubble_rects

    img = _spiky_bubble()
    inner, loose = bubble_rects(img, [470, 480, 60, 40])
    x, y, w, h = inner
    assert (img[y:y + h, x:x + w] == 255).all()  # no outline or background inside
    assert w > 100 and h > 100  # but uses the round core, not just the text box
    assert loose[2] > w  # loose box reaches into the spikes


def test_long_text_in_tiny_bubble_stays_readable_and_whole():
    from manga_translator.core.typeset import choose_layout, min_size_for

    shape = (1761, 1200)
    rect, lay, spilled = choose_layout("ผู้บัญชาการ!", "th", "Leelawadee UI", (0, 0, 30, 50), (0, 0, 37, 60), 0, shape)
    assert lay.size >= min_size_for(shape)
    assert lay.lines == ["ผู้บัญชาการ!"]  # not broken inside the word
    assert spilled


def test_user_text_rect_overrides_bubble():
    from manga_translator.core.typeset import place

    img = np.full((600, 600, 3), 255, np.uint8)
    r = Region(id=1, bbox=[50, 50, 200, 120], translation="สวัสดีครับ", text_rect=[300, 400, 220, 90])
    rect, lay, spilled = place(img, r, "th")
    assert rect == (300, 400, 220, 90) and lay.fits and not spilled
    out, _ = render(img, [r], "th")
    ink = np.argwhere(out.min(axis=2) < 128)
    assert ink[:, 0].min() >= 400 and ink[:, 1].min() >= 300  # drawn where the user put it


def test_enter_forces_a_line_break():
    lay = fit_text("สวัสดีครับ\nวันนี้อากาศดีมากเลยนะ", "th", "Leelawadee UI", 400, 300)
    assert lay.lines[0] == "สวัสดีครับ"  # would fit on one line with the rest, but Enter was pressed
    assert "".join(lay.lines[1:]) == "วันนี้อากาศดีมากเลยนะ"


def test_manual_wrap_keeps_typed_lines_and_shrinks():
    text = "ไปกันเถอะ\nทุกคนรออยู่แล้ว"
    auto = fit_text("ไปกันเถอะทุกคนรออยู่แล้ว", "th", "Leelawadee UI", 90, 300)
    manual = fit_text(text, "th", "Leelawadee UI", 90, 300, manual=True)
    assert manual.lines == ["ไปกันเถอะ", "ทุกคนรออยู่แล้ว"]
    assert manual.fits and len(auto.lines) > 2


def test_model_line_breaks_are_not_forced():
    from manga_translator.core.translate import parse_response

    assert parse_response('{"translations":[{"id":1,"text":"บรรทัดหนึ่ง\\nบรรทัดสอง"}]}') == {1: "บรรทัดหนึ่ง บรรทัดสอง"}


def test_joined_bubble_is_shared_between_its_boxes():
    import cv2

    from manga_translator.core.typeset import bubble_rects

    img = np.full((1000, 1000, 3), 120, np.uint8)
    for cx in (400, 560):  # two lobes joined into one white shape, like a double speech bubble
        cv2.ellipse(img, (cx, 500), (110, 150), 0, 0, 360, (255, 255, 255), -1)
    left, right = [350, 450, 60, 80], [520, 450, 60, 80]
    li, ll = bubble_rects(img, left, [right])
    ri, rl = bubble_rects(img, right, [left])
    assert li[0] + li[2] <= ri[0] and ll[0] + ll[2] <= rl[0] + 2  # no overlap between the halves
    alone, _ = bubble_rects(img, left)
    assert alone[2] > li[2]  # without the other box it would have taken the whole shape
