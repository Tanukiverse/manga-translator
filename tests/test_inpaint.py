import cv2
import numpy as np
import pytest

from manga_translator.core.inpaint import clean


def test_flat_bubble_restored_and_outside_untouched(detector, ja_page):
    img, truth = ja_page
    _, mask = detector(img)
    out, stats = clean(img, mask)
    assert stats["flat"] > 0
    outside = mask == 0
    assert np.array_equal(out[outside], img[outside])
    for t in truth:
        x, y, w, h = t.bbox
        assert out[y:y + h, x:x + w].min() > 200, f"ink left in {t.text}"


def _textured():
    rng = np.random.default_rng(1)
    img = (rng.integers(0, 2, (256, 256)) * 120 + 60).astype(np.uint8)
    img = cv2.cvtColor(cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    mask = np.zeros((512, 512), np.uint8)
    cv2.putText(img, "SFX", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 12)
    cv2.putText(mask, "SFX", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 20)
    return img, mask


def test_textured_falls_back_to_telea_without_lama():
    img, mask = _textured()
    out, stats = clean(img, mask, lama=None)
    assert stats["flat"] == 0 and stats["telea"] > 0
    assert np.array_equal(out[mask == 0], img[mask == 0])
    assert out[mask > 0].mean() < 200  # white letters gone


@pytest.mark.gpu
def test_textured_uses_lama():
    from manga_translator.core.inpaint import Lama

    img, mask = _textured()
    lama = Lama()
    out, stats = clean(img, mask, lama=lama)
    lama.release()
    assert stats["lama"] > 0
    assert np.array_equal(out[mask == 0], img[mask == 0])
    assert out[mask > 0].mean() < 200
