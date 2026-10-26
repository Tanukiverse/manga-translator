import numpy as np
import pytest

from manga_translator.core.detect import split_side_by_side
from manga_translator.core.ocr import is_junk_english
from manga_translator.core.pipeline import keep_skipped
from manga_translator.core.project import Region
from manga_translator.core.translate import pick_one
from manga_translator.core.typeset import render


@pytest.mark.parametrize("text", ["$7R6AH-7.", "S 6 MT 1 1 2 5", "S", "X9Q", "Q#Z"])
def test_junk_is_detected(text):
    assert is_junk_english(text)


@pytest.mark.parametrize("text", ["?", "...", "HM?", "I SEE...", "OK", "A-AH!", "MM...", "3000-MAN ARMY", "WHOA?!", "NO!"])
def test_real_lettering_is_kept(text):
    assert not is_junk_english(text)


def test_pick_one_particle():
    assert pick_one("ไปกันเลยค่ะ/ครับ") == "ไปกันเลยค่ะ"
    assert pick_one("ขอบคุณ (ครับ/ค่ะ)") == "ขอบคุณครับ"
    assert pick_one("ผม/ฉันไม่รู้") == "ผมไม่รู้"
    assert pick_one("หนึ่ง/สอง") == "หนึ่ง/สอง"  # other slashes are left alone


def test_skipped_box_is_not_erased_but_overlapping_real_box_is():
    mask = np.zeros((100, 200), np.uint8)
    mask[10:40, 10:60] = 255  # junk box
    mask[10:40, 120:180] = 255  # real box
    regions = [Region(1, [10, 10, 50, 30], skip=True), Region(2, [120, 10, 60, 30])]
    out = keep_skipped(mask, regions)
    assert out[10:40, 10:60].max() == 0 and out[10:40, 120:180].min() == 255
    assert keep_skipped(mask, [Region(1, [10, 10, 50, 30])]) is mask  # nothing skipped: unchanged


def test_skipped_box_is_not_drawn():
    img = np.full((200, 300, 3), 255, np.uint8)
    out, _ = render(img, [Region(1, [20, 20, 200, 100], translation="ทดสอบ", skip=True)], "th")
    assert out.min() == 255


def _block(mask, x, y, lines, width, line_h=20, gap=8):
    for i in range(lines):
        mask[y + i * (line_h + gap):y + i * (line_h + gap) + line_h, x:x + width] = 255


def test_two_blocks_side_by_side_are_split():
    mask = np.zeros((300, 400), np.uint8)
    _block(mask, 20, 40, 6, 150)  # left speech beat
    _block(mask, 200, 20, 5, 150)  # right one, starting higher
    parts = split_side_by_side((10, 10, 360, 260), mask)
    assert len(parts) == 2
    (lx, _ly, lw, _lh), (rx, ry, _rw, _rh) = parts
    assert lx + lw <= rx and ry == 20


def test_one_block_with_word_spaces_is_not_split():
    mask = np.zeros((300, 400), np.uint8)
    for i in range(5):  # words separated by narrow spaces, aligned on every line
        mask[20 + i * 28:40 + i * 28, 20:120] = 255
        mask[20 + i * 28:40 + i * 28, 128:250] = 255
    assert len(split_side_by_side((10, 10, 300, 200), mask)) == 1


def test_paragraph_with_aligned_word_spaces_is_not_split():
    mask = np.zeros((200, 600), np.uint8)
    for i in range(2):  # two lines, a wide space at the same place on both
        mask[20 + i * 30:42 + i * 30, 20:250] = 255
        mask[20 + i * 30:42 + i * 30, 290:560] = 255
    assert len(split_side_by_side((10, 10, 570, 80), mask)) == 1
