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
    s = FakeSession([FakeResp(400, {"error": "model does not support thinking"}), _reply([{"id": 1, "text": "ก"}])])
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


def test_wrong_language_is_retried_with_a_reminder():
    s = FakeSession([_reply([{"id": 1, "text": "こんにちは"}, {"id": 2, "text": "?"}]),
                     _reply([{"id": 1, "text": "สวัสดี"}, {"id": 2, "text": "?"}])])
    tr = OllamaTranslator(TranslatorConfig(), s)
    out = tr.translate([(1, "HELLO"), (2, "?")], "en", "th")
    assert out == {1: "สวัสดี", 2: "?"}
    retry = s.bodies[1]["messages"]
    assert "never in the source language" in retry[-1]["content"]
    assert all(m["role"] != "assistant" for m in retry)  # fresh request, not shown its wrong answer
    assert s.bodies[1]["think"] is False  # the retry keeps thinking off
    assert s.bodies[1]["format"] == "json"  # and drops the strict schema
    assert tr.wrong_language == []


def test_wrong_language_reported_when_retry_fails():
    s = FakeSession([_reply([{"id": 1, "text": "こんにちは"}]), _reply([{"id": 1, "text": "こんにちは"}])])
    tr = OllamaTranslator(TranslatorConfig(), s)
    tr.translate([(1, "HELLO")], "en", "th")
    assert tr.wrong_language == [1]


def test_target_language_named_in_request():
    msgs = build_messages([(1, "a")], "en", "th", TranslatorConfig())
    assert "ภาษาไทย" in msgs[1]["content"]


def test_parse_free_json_mapping():
    assert parse_response('{"1": "สวัสดี", "2": "ลาก่อน\\nนะ"}') == {1: "สวัสดี", 2: "ลาก่อน นะ"}


def test_empty_answer_is_asked_again():
    s = FakeSession([FakeResp(200, {"message": {"content": ""}}), _reply([{"id": 1, "text": "สวัสดี"}])])
    out = OllamaTranslator(TranslatorConfig(), s).translate([(1, "HELLO")], "en", "th")
    assert out == {1: "สวัสดี"} and "format" not in s.bodies[1]  # asked again without a forced format


def test_json_inside_free_text_reply():
    assert parse_response('Here you go:\n{"translations":[{"id":2,"text":"ได้เลย"}]}\nDone.') == {2: "ได้เลย"}


def test_single_bubble_plain_text_reply():
    # schema/JSON answers empty, free-form answer is just the translation
    s = FakeSession([FakeResp(200, {"message": {"content": ""}}),
                     FakeResp(200, {"message": {"content": "  \"ไปกันเถอะค่ะ/ครับ\"\n"}})])
    out = OllamaTranslator(TranslatorConfig(), s).translate([(5, "LET'S GO.")], "en", "th")
    assert out == {5: "ไปกันเถอะค่ะ"}


def test_free_form_reply_one_line_per_bubble():
    s = FakeSession([FakeResp(200, {"message": {"content": ""}}),
                     FakeResp(200, {"message": {"content": "สวัสดี\nลาก่อนนะ\n"}})])
    out = OllamaTranslator(TranslatorConfig(), s).translate([(1, "HI"), (2, "BYE")], "en", "th")
    assert out == {1: "สวัสดี", 2: "ลาก่อนนะ"}
