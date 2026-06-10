from __future__ import annotations

import json
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .observer import ObservationBuilder
from .prompts import REPAIR_PROMPT, SYSTEM_PROMPT
from .schema import ActionParseError, parse_action_text
from spl.core.actions import GameAction


@dataclass(frozen=True)
class Cassette:
    name: str
    base_url: str
    model: str
    api_key: str = "local"
    temperature: float = 0.7
    max_tokens: int = 256
    json_mode: bool = True
    persona: str = ""


def load_cassettes(path: Path) -> list[Cassette]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    cassettes = []
    for row in data.get("cassette", []):
        cassettes.append(
            Cassette(
                name=row["name"],
                base_url=row.get("base_url", ""),
                model=row.get("model", ""),
                api_key=row.get("api_key", "local"),
                temperature=float(row.get("temperature", 0.7)),
                max_tokens=int(row.get("max_tokens", 256)),
                json_mode=bool(row.get("json_mode", True)),
                persona=row.get("persona", ""),
            )
        )
    return cassettes


def find_cassette(path: Path, name: str | None) -> Cassette:
    cassettes = load_cassettes(path)
    if not cassettes:
        raise ValueError("No cassettes configured.")
    if not name:
        return cassettes[0]
    for cassette in cassettes:
        if cassette.name == name:
            return cassette
    names = ", ".join(c.name for c in cassettes)
    raise ValueError(f"Cassette not found: {name}. Available: {names}")


class OpenAICompatibleBrain:
    def __init__(self, cassette: Cassette, timeout: float = 45.0) -> None:
        self.cassette = cassette
        self.timeout = timeout
        self.observer = ObservationBuilder()

    def choose(self, sim: object) -> GameAction:
        obs = self.observer.build(sim)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n" + self.cassette.persona},
            {"role": "user", "content": json.dumps(obs, ensure_ascii=False)},
        ]
        first = self._chat(messages)
        try:
            return parse_action_text(first).to_game_action()
        except ActionParseError as exc:
            repair_messages = messages + [
                {"role": "assistant", "content": first},
                {"role": "user", "content": REPAIR_PROMPT + f"\nError: {exc}"},
            ]
            repaired = self._chat(repair_messages)
            try:
                return parse_action_text(repaired).to_game_action()
            except ActionParseError as repair_exc:
                return GameAction(
                    action="invalid_llm_output",
                    args={},
                    think=f"Could not repair LLM JSON: {repair_exc}",
                    say="The words came apart in my hands.",
                )

    def _chat(self, messages: list[dict[str, str]]) -> str:
        if not self.cassette.base_url:
            raise RuntimeError("Cassette has no base_url.")
        payload: dict[str, Any] = {
            "model": self.cassette.model,
            "messages": messages,
            "temperature": self.cassette.temperature,
            "max_tokens": self.cassette.max_tokens,
        }
        if self.cassette.json_mode:
            payload["response_format"] = {"type": "json_object"}
        url = self.cassette.base_url.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cassette.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        data = json.loads(body)
        return data["choices"][0]["message"]["content"]
