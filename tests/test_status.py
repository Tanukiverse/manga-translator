from manga_translator.core import status


def test_ollama_down_is_reported_not_raised():
    st = status.ollama_status("http://127.0.0.1:9", timeout=0.5)
    assert not st.ok and st.error


def test_model_files_listed():
    files = status.model_files()
    assert {f["name"] for f in files} >= {"comictextdetector.pt.onnx", "big-lama.pt"}
    assert all(f["size_mb"] > 0 for f in files if f["present"])


def test_versions_include_core_packages():
    v = status.versions()
    assert v["Python"] and v["PySide6"] != "ไม่ได้ติดตั้ง"


def test_gpu_status_never_raises():
    g = status.gpu_status()
    assert g.ok or g.error
