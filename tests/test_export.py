import numpy as np
import pytest

from manga_translator.core.export import (AutoSaver, ExportError, export_pages, make_prefix, output_name,
                                          plan_names)
from manga_translator.core.project import imread, imwrite
from manga_translator.core.settings import Settings


def _final(path, value=200):
    img = np.full((40, 30, 3), value, np.uint8)
    img[5:10, 5:10] = (0, 0, 255)
    imwrite(path, img)
    return path


def test_names():
    assert plan_names(3, "ตอน5_", 1, 3, "jpg") == ["ตอน5_001.jpg", "ตอน5_002.jpg", "ตอน5_003.jpg"]
    assert output_name("", 7, 2, "png") == "07.png"
    assert make_prefix("{ตอน}:ฉบับ?แปล_", "Ch5") == "Ch5ฉบับแปล_"  # Windows-forbidden characters removed
    with pytest.raises(ExportError):
        output_name("a", 1, 3, "gif")


@pytest.mark.parametrize("fmt", ["png", "jpg", "webp"])
def test_export_formats_and_no_silent_overwrite(tmp_path, fmt):
    src = _final(tmp_path / "final.png")
    dest = tmp_path / "out" / f"a_001.{'jpg' if fmt == 'jpg' else fmt}"
    written, skipped = export_pages([(src, dest)], fmt, 90, overwrite=False)
    assert written == [dest] and imread(dest).shape == (40, 30, 3)
    before = dest.read_bytes()
    _final(src, 50)
    written, skipped = export_pages([(src, dest)], fmt, 90, overwrite=False)
    assert skipped == [dest] and dest.read_bytes() == before


def _saver(tmp_path, **kw):
    s = Settings(save_dir=str(tmp_path / "out"), autosave=True, save_format="png", **kw)
    return AutoSaver(s)


def test_autosave_path_and_rename_on_reorder(tmp_path):
    saver = _saver(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    f1, f2 = _final(tmp_path / "f1.png", 10), _final(tmp_path / "f2.png", 90)
    assert saver.save(work, "ตอน5", "p1", 0, f1) == "ตอน5_001.png"
    assert saver.save(work, "ตอน5", "p2", 1, f2) == "ตอน5_002.png"
    out = tmp_path / "out" / "ตอน5"
    # pages swapped: p2 is now first
    assert saver.save(work, "ตอน5", "p2", 0, f2) == "ตอน5_001.png"
    assert saver.save(work, "ตอน5", "p1", 1, f1) == "ตอน5_002.png"
    assert sorted(p.name for p in out.iterdir()) == ["ตอน5_001.png", "ตอน5_002.png"]
    assert imread(out / "ตอน5_001.png")[20, 20, 0] == 90  # p2's image now has number 1


def test_autosave_never_touches_foreign_files(tmp_path):
    saver = _saver(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    out = tmp_path / "out" / "ตอน5"
    out.mkdir(parents=True)
    (out / "ตอน5_001.png").write_bytes(b"user file")
    assert saver.save(work, "ตอน5", "p1", 0, _final(tmp_path / "f.png")) is None
    assert (out / "ตอน5_001.png").read_bytes() == b"user file"
    assert "ไม่เขียนทับ" in saver.status.last_error


def test_autosave_keeps_old_file_the_user_edited(tmp_path):
    saver = _saver(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    f = _final(tmp_path / "f.png")
    saver.save(work, "c", "p1", 0, f)
    out = tmp_path / "out" / "c"
    (out / "c_001.png").write_bytes(b"edited by user")
    saver.save(work, "c", "p1", 4, f)  # page moved to position 5
    assert (out / "c_001.png").read_bytes() == b"edited by user"
    assert (out / "c_005.png").exists()


def test_autosave_off_writes_nothing(tmp_path):
    s = Settings(save_dir=str(tmp_path / "out"), autosave=False)
    assert AutoSaver(s).save(tmp_path, "c", "p1", 0, _final(tmp_path / "f.png")) is None
    assert not (tmp_path / "out").exists()
