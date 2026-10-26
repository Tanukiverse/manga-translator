import numpy as np

from manga_translator.core import versions
from manga_translator.core.project import PageDoc, Region, imread, imwrite


def _page(tmp_path):
    src = tmp_path / "p1.png"
    imwrite(src, np.full((40, 30, 3), 100, np.uint8))
    doc = PageDoc.create(src, tmp_path / "work")
    doc.regions = [Region(1, [1, 1, 10, 10], source_text="HI", translation="แรก")]
    doc.save_layer("final", np.full((40, 30, 3), 10, np.uint8))
    doc.save()
    return doc


def test_snapshot_keeps_the_old_picture_after_it_is_rewritten(tmp_path):
    doc = _page(tmp_path)
    v = versions.snapshot(doc, "ก่อน: แปล")
    assert v is not None
    doc.save_layer("final", np.full((40, 30, 3), 200, np.uint8))  # a new run replaces the file
    doc.regions[0].translation = "สอง"
    doc.save()
    saved = versions.list_versions(doc)[0]
    assert imread(saved.path / "final.png")[0, 0, 0] == 10
    assert saved.label == "ก่อน: แปล" and saved.regions == 1


def test_unchanged_page_is_not_saved_twice_and_fresh_page_not_at_all(tmp_path):
    doc = _page(tmp_path)
    assert versions.snapshot(doc, "a") is not None
    assert versions.snapshot(doc, "b") is None  # nothing changed since
    src = tmp_path / "p2.png"
    imwrite(src, np.zeros((10, 10, 3), np.uint8))
    assert versions.snapshot(PageDoc.create(src, tmp_path / "work"), "c") is None  # nothing done yet


def test_restore_brings_back_files_and_text_and_is_itself_undoable(tmp_path):
    doc = _page(tmp_path)
    versions.snapshot(doc, "ก่อน: แปลใหม่")
    doc.save_layer("final", np.full((40, 30, 3), 200, np.uint8))
    doc.save_layer("cleaned", np.full((40, 30, 3), 250, np.uint8))  # did not exist in the version
    doc.regions[0].translation = "ใหม่แต่แย่กว่า"
    doc.save()
    old = versions.list_versions(doc)[0]
    restored = versions.restore(doc, old)
    assert restored.regions[0].translation == "แรก"
    assert imread(restored.final_path)[0, 0, 0] == 10
    assert not restored.cleaned_path.exists()
    # the state just replaced was saved, so the restore can be undone
    newest = versions.list_versions(doc)[0]
    assert newest.label == "ก่อนคืนค่าเวอร์ชัน" and imread(newest.path / "final.png")[0, 0, 0] == 200
    assert imread(old.path / "final.png")[0, 0, 0] == 10  # the version itself is unchanged


def test_reset_clears_the_page_but_keeps_the_original_and_a_way_back(tmp_path):
    doc = _page(tmp_path)
    fresh = versions.reset(doc)
    assert fresh.regions == [] and fresh.done == {}
    assert not fresh.final_path.exists() and fresh.original_path.exists()
    assert PageDoc.load(doc.folder).regions == []
    back = versions.restore(fresh, versions.list_versions(fresh)[0])
    assert back.regions[0].translation == "แรก"


def test_only_the_newest_versions_are_kept(tmp_path):
    doc = _page(tmp_path)
    for i in range(5):
        doc.regions[0].translation = f"รอบ {i}"
        doc.save()
        versions.snapshot(doc, f"รอบ {i}", keep=3)
    vs = versions.list_versions(doc)
    assert len(vs) == 3 and vs[0].label == "รอบ 4"
