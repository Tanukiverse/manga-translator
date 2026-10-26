"""Translation through a local Ollama model. One request per page so the model sees the whole dialogue."""
from __future__ import annotations

import json
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


@dataclass
class TranslatorConfig:
    url: str = "http://localhost:11434"
    model: str = "qwen3.5:4b"
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
        f"You are a professional manga translator. Translate speech bubbles from {src_name} into natural, "
        f"conversational {tgt_name} that fits each character's voice. The bubbles are given in reading order "
        "and belong to the same page, so use the surrounding lines as context. Keep sound effects short. "
        "Do not add explanations, notes or romanization. Translate every id exactly once."
    )
    if tgt == "th":
        system += " Use Thai particles (ครับ/ค่ะ/นะ/จ้ะ) only where they fit the speaker; keep lines short so they fit a bubble."
    if cfg.glossary:
        terms = "\n".join(f"- {k} => {v}" for k, v in cfg.glossary.items())
        system += f"\nAlways use these fixed translations:\n{terms}"
    if cfg.notes:
        system += f"\nStory context: {cfg.notes}"
    user = json.dumps({"bubbles": [{"id": i, "text": t} for i, t in lines]}, ensure_ascii=False)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


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

    def list_models(self) -> list[str]:
        r = self.http.get(f"{self.cfg.url}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def _chat(self, messages: list[dict], think: bool | None) -> str:
        body = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "format": RESPONSE_SCHEMA,
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

    def translate(self, lines: list[tuple[int, str]], src: str, tgt: str) -> dict[int, str]:
        """lines: [(region id, source text)] in reading order. Returns {id: translation}."""
        lines = [(i, t) for i, t in lines if t.strip()]
        if not lines:
            return {}
        messages = build_messages(lines, src, tgt, self.cfg)
        try:
            content = self._chat(messages, think=False)
        except TranslationError as e:
            # models without a thinking mode reject the "think" field
            if "think" not in str(e).lower():
                raise
            content = self._chat(messages, think=None)
        result = parse_response(content)
        missing = [(i, t) for i, t in lines if not result.get(i, "").strip()]
        # retry missing ids one by one rather than failing the whole page
        for i, t in missing:
            single = parse_response(self._chat(build_messages([(i, t)], src, tgt, self.cfg), think=None))
            if single:
                result[i] = next(iter(single.values()))
        return {i: result[i] for i, _ in lines if i in result}

    def unload(self) -> None:
        """Free the model's VRAM immediately."""
        try:
            self.http.post(f"{self.cfg.url}/api/generate", json={"model": self.cfg.model, "keep_alive": 0}, timeout=30)
        except requests.RequestException:
            pass


def parse_response(content: str) -> dict[int, str]:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise TranslationError(f"model returned invalid JSON: {content[:200]}") from e
    out: dict[int, str] = {}
    for item in data.get("translations", []):
        try:
            out[int(item["id"])] = str(item["text"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
    return out
