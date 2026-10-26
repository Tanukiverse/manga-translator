"""Translation through a local Ollama model. One request per page so the model sees the whole dialogue."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import requests

LANG_NAMES = {
    "ja": "Japanese",
    "en": "English",
    "th": "Thai",
    "zh": "Simplified Chinese",
    "ko": "Korean",
    "vi": "Vietnamese",
    "id": "Indonesian",
}


# target language in its own script: put in the request so models do not drift to another language
LANG_NATIVE = {"th": "ภาษาไทย", "ja": "日本語", "zh": "简体中文", "ko": "한국어", "en": "English",
               "vi": "Tiếng Việt", "id": "Bahasa Indonesia"}
# a translation with letters must contain at least one of these characters
SCRIPT_RANGES = {
    "th": [("฀", "๿")],
    "ja": [("぀", "ヿ"), ("一", "鿿")],
    "zh": [("一", "鿿")],
    "ko": [("가", "힯"), ("ᄀ", "ᇿ")],
}


def in_target_script(text: str, tgt: str) -> bool:
    """False when a translation is plainly in the wrong language (e.g. Japanese asked for Thai)."""
    if not any(c.isalpha() for c in text):
        return True  # "?", "...", "!!" are fine in any language
    ranges = SCRIPT_RANGES.get(tgt)
    if ranges is None:  # Latin-script targets: reject text with no Latin letters at all
        return any("a" <= c.lower() <= "z" or "À" <= c <= "ɏ" for c in text)
    return any(lo <= c <= hi for c in text for lo, hi in ranges)


SHORT_CHARS = 10  # source bubbles with at most this many letters get a "short" hint


@dataclass
class TranslatorConfig:
    url: str = "http://localhost:11434"
    model: str = "hf.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive:Q4_K_M"
    temperature: float = 0.3
    keep_alive: str = "5m"
    timeout: float = 300.0
    glossary: dict[str, str] = field(default_factory=dict)
    notes: str = ""  # free-form context, e.g. "Comedy, high-school characters"


class TranslationError(RuntimeError):
    pass


def build_messages(lines: list[tuple[int, str]], src: str, tgt: str, cfg: TranslatorConfig) -> list[dict]:
    src_name, tgt_name = LANG_NAMES.get(src, src), LANG_NAMES.get(tgt, tgt)
    system = (
        # not "manga": that word alone made a Qwen3.5-9B finetune answer in Japanese whatever the target
        f"You are a professional comic translator. Translate speech bubbles from {src_name} into natural, "
        f"conversational {tgt_name} that fits each character's voice. The bubbles are given in reading order "
        "and belong to the same page, so use the surrounding lines as context. Keep sound effects short. "
        "Do not add explanations, notes or romanization. Translate every id exactly once. "
        "Keep each translation about as long as the original: bubbles marked \"short\": true are tiny, "
        "so answer those in one to three words."
    )
    if tgt == "th":
        system += (" Use Thai particles (ครับ, ค่ะ, นะ, จ้ะ) only where they fit the speaker; keep lines short so they "
                   "fit a bubble. Never write alternatives with a slash such as ค่ะ/ครับ or ผม/ฉัน: decide who is "
                   "speaking from the page, and if you cannot tell, end the line without ครับ or ค่ะ.")
    if cfg.glossary:
        terms = "\n".join(f"- {k} => {v}" for k, v in cfg.glossary.items())
        system += f"\nAlways use these fixed translations:\n{terms}"
    if cfg.notes:
        system += f"\nStory context: {cfg.notes}"
    bubbles = []
    for i, t in lines:
        b = {"id": i, "text": t}
        if sum(c.isalnum() for c in t) <= SHORT_CHARS:
            b["short"] = True
        bubbles.append(b)
    target = f"{tgt_name} ({LANG_NATIVE.get(tgt, tgt_name)})"
    user = json.dumps({"translate_into": target, "bubbles": bubbles}, ensure_ascii=False)
    user += f"\n\nAnswer in {target} only."  # last thing the model reads before answering
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def language_reminder(tgt: str) -> dict:
    name = f"{LANG_NAMES.get(tgt, tgt)} ({LANG_NATIVE.get(tgt, tgt)})"
    return {"role": "user", "content": f"Important: write every translation in {name} only, never in the "
                                       "source language or in Japanese. Answer as JSON."}


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["id", "text"],
            },
        }
    },
    "required": ["translations"],
}


class OllamaTranslator:
    def __init__(self, cfg: TranslatorConfig | None = None, session: requests.Session | None = None):
        self.cfg = cfg or TranslatorConfig()
        self.http = session or requests.Session()
        self.wrong_language: list[int] = []  # ids still not in the target language after a retry
        self._think: bool | None = False  # becomes None for models that reject the "think" field

    def list_models(self) -> list[str]:
        r = self.http.get(f"{self.cfg.url}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def _chat(self, messages: list[dict], think: bool | None, fmt=RESPONSE_SCHEMA) -> str:
        body = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "format": fmt,
            "keep_alive": self.cfg.keep_alive,
            "options": {"temperature": self.cfg.temperature},
        }
        if think is not None:
            body["think"] = think
        try:
            r = self.http.post(f"{self.cfg.url}/api/chat", json=body, timeout=self.cfg.timeout)
        except requests.ConnectionError as e:
            raise TranslationError(f"เชื่อมต่อ Ollama ไม่ได้ที่ {self.cfg.url} (เปิด Ollama แล้วหรือยัง?)") from e
        if r.status_code >= 400:
            raise TranslationError(f"Ollama {r.status_code}: {r.text[:300]}")
        return r.json()["message"]["content"]

    def _ask(self, messages: list[dict], fmt=RESPONSE_SCHEMA) -> str:
        """Chat with thinking off. Every follow-up must keep it off too: a thinking model asked
        again without the flag spent its answer on thinking and returned empty content."""
        try:
            return self._chat(messages, think=self._think, fmt=fmt)
        except TranslationError as e:
            # models without a thinking mode reject the "think" field
            if self._think is None or "think" not in str(e).lower():
                raise
            self._think = None
            return self._chat(messages, think=None, fmt=fmt)

    def translate(self, lines: list[tuple[int, str]], src: str, tgt: str) -> dict[int, str]:
        """lines: [(region id, source text)] in reading order. Returns {id: translation}."""
        lines = [(i, t) for i, t in lines if t.strip()]
        if not lines:
            return {}
        messages = build_messages(lines, src, tgt, self.cfg)
        content = self._ask(messages)
        try:
            result = parse_response(content)
        except TranslationError:
            # an empty or broken answer happens now and then (1 page in 38 with Gemma): ask once
            # more in plain JSON mode instead of failing the whole page
            content = self._ask(messages, fmt="json")
            result = parse_response(content)
        wrong = [i for i, t in result.items() if not in_target_script(t, tgt)]
        if wrong:
            # some models (e.g. a 9B uncensored Qwen) answer in the source language: say so and ask again
            # ask afresh in plain JSON mode: under the strict schema a Qwen3.5-9B finetune kept
            # answering in Japanese, and showing it its own Japanese answer anchored it there too,
            # while free JSON from the same prompt came back in Thai
            again = messages[:-1] + [{**messages[-1], "content": messages[-1]["content"] + "\n\n"
                                      + language_reminder(tgt)["content"]}]
            for i, t in parse_response(self._ask(again, fmt="json")).items():
                if i in wrong and in_target_script(t, tgt):
                    result[i] = t
        self.wrong_language = [i for i, t in result.items() if not in_target_script(t, tgt)]
        missing = [(i, t) for i, t in lines if not result.get(i, "").strip()]
        # retry missing ids one by one rather than failing the whole page
        for i, t in missing:
            single = parse_response(self._ask(build_messages([(i, t)], src, tgt, self.cfg)))
            if single:
                result[i] = next(iter(single.values()))
        return {i: result[i] for i, _ in lines if i in result}

    def unload(self) -> None:
        """Free the model's VRAM immediately."""
        try:
            self.http.post(f"{self.cfg.url}/api/generate", json={"model": self.cfg.model, "keep_alive": 0}, timeout=30)
        except requests.RequestException:
            pass


_ALTERNATIVES = re.compile(r"\s*\(?\s*(ครับ|ค่ะ|คะ|ผม|ฉัน|ดิฉัน|นะครับ|นะคะ)\s*/\s*(ครับ|ค่ะ|คะ|ผม|ฉัน|ดิฉัน|นะครับ|นะคะ)\s*\)?")


def pick_one(text: str) -> str:
    """'ไปกันเถอะค่ะ/ครับ' -> 'ไปกันเถอะค่ะ': a speech bubble has one speaker."""
    return _ALTERNATIVES.sub(lambda m: m.group(1), text)


def parse_response(content: str) -> dict[int, str]:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise TranslationError(f"model returned invalid JSON: {content[:200]}") from e
    out: dict[int, str] = {}
    if isinstance(data, dict) and "translations" not in data:
        # free JSON mode may answer {"1": "...", "2": "..."} instead of the schema
        for k, v in data.items():
            if str(k).isdigit() and isinstance(v, str):
                out[int(k)] = pick_one(" ".join(v.split()))
        return out
    for item in data.get("translations", []):
        try:
            # the model's own line breaks would act as forced breaks; layout wraps by itself
            out[int(item["id"])] = pick_one(" ".join(str(item["text"]).split()))
        except (KeyError, TypeError, ValueError):
            continue
    return out
