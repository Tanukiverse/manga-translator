import numpy as np

from manga_translator.core.project import PageDoc, Region, imread, imwrite


def test_roundtrip_with_thai_path(tmp_path):
    src = tmp_path / "ต้นฉบับ_ページ1.png"
    imwrite(src, np.full((40, 30, 3), 200, np.uint8))
    doc = PageDoc.create(src, tmp_path / "งาน", "ja", "th")
    doc.regions = [Region(id=1, bbox=[1, 2, 3, 4], source_text="こんにちは", translation="สวัสดี", locked=True)]
    doc.done["detect"] = True
    doc.save()

    again = PageDoc.load(doc.folder)
    assert again.regions[0].translation == "สวัสดี"
    assert again.regions[0].locked
    assert again.done == {"detect": True}
    assert again.load_original().shape == (40, 30, 3)
    assert not list(doc.folder.glob("*.tmp"))


def test_create_reopens_existing_page(tmp_path):
    src = tmp_path / "p.png"
    imwrite(src, np.zeros((10, 10, 3), np.uint8))
    doc = PageDoc.create(src, tmp_path / "w")
    doc.regions = [Region(id=7, bbox=[0, 0, 5, 5])]
    doc.save()
    assert PageDoc.create(src, tmp_path / "w").regions[0].id == 7


def test_unknown_fields_ignored(tmp_path):
    (tmp_path / "page.json").write_text('{"regions":[{"id":1,"bbox":[0,0,1,1],"future":1}]}', "utf-8")
    assert PageDoc.load(tmp_path).regions[0].id == 1


def test_imread_gray(tmp_path):
    imwrite(tmp_path / "m.png", np.full((5, 5), 255, np.uint8))
    assert imread(tmp_path / "m.png", gray=True).shape == (5, 5)
