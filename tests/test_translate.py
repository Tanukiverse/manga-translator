import json

import pytest
import requests

from manga_translator.core.translate import OllamaTranslator, TranslatorConfig, build_messages, parse_response


class FakeResp:
    def __init__(self, status, payload):
        self.status_code = status
        self._p = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, replies):
        self.replies = list(replies)
        self.bodies = []

    def post(self, url, json=None, timeout=None):
        self.bodies.append(json)
        return self.replies.pop(0)


def _reply(items):
    return FakeResp(200, {"message": {"content": json.dumps({"translations": items}, ensure_ascii=False)}})


def test_translate_page_in_one_request():
    s = FakeSession([_reply([{"id": 1, "text": "สวัสดี"}, {"id": 2, "text": "ลาก่อน"}])])
    out = OllamaTranslator(TranslatorConfig(), s).translate([(1, "こんにちは"), (2, "さようなら")], "ja", "th")
    assert out == {1: "สวัสดี", 2: "ลาก่อน"}
    assert len(s.bodies) == 1 and s.bodies[0]["think"] is False


def test_missing_id_retried_alone():
    s = FakeSession([_reply([{"id": 1, "text": "หนึ่ง"}]), _reply([{"id": 2, "text": "สอง"}])])
    out = OllamaTranslator(TranslatorConfig(), s).translate([(1, "a"), (2, "b")], "en", "th")
    assert out == {1: "หนึ่ง", 2: "สอง"}


def test_model_without_thinking_retries_without_field():
    s = FakeSession([FakeResp(400, {"error": "model does not support thinking"}), _reply([{"id": 1, "text": "x"}])])
    OllamaTranslator(TranslatorConfig(), s).translate([(1, "a")], "en", "th")
    assert "think" not in s.bodies[1]


def test_glossary_in_prompt():
    msgs = build_messages([(1, "a")], "ja", "th", TranslatorConfig(glossary={"ナルミ": "นารุมิ"}))
    assert "ナルミ => นารุมิ" in msgs[0]["content"]


def test_parse_code_fence():
    assert parse_response('```json\n{"translations":[{"id":3,"text":"ok"}]}\n```') == {3: "ok"}


def _ollama_up():
    try:
        return requests.get("http://localhost:11434/api/tags", timeout=2).ok
    except requests.RequestException:
        return False


@pytest.mark.ollama
@pytest.mark.skipif(not _ollama_up(), reason="Ollama not running")
def test_real_ollama():
    out = OllamaTranslator().translate([(1, "おはようございます"), (2, "今日はいい天気だね")], "ja", "th")
    assert set(out) == {1, 2}
    assert all(any("฀" <= c <= "๿" for c in t) for t in out.values())
