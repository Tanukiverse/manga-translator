import pytest
import synth


def test_english_ocr(en_page):
    from manga_translator.core.ocr import EnglishOcr

    img, truth = en_page
    ocr = EnglishOcr()
    letters = lambda s: "".join(c for c in s.upper() if c.isalnum())  # noqa: E731
    for t in truth:
        assert synth.cer(letters(t.text), letters(ocr(img, t.bbox))) < 0.1


@pytest.mark.gpu
def test_japanese_ocr(ja_page):
    from manga_translator.core.ocr import JapaneseOcr

    img, truth = ja_page
    ocr = JapaneseOcr()
    for t in truth:
        assert synth.cer(t.text, ocr(img, t.bbox)) < 0.15
