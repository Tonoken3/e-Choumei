from __future__ import annotations

import json
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spl.core.actions import ACTION_WORDS, GameAction

from .observer import ObservationBuilder
from .prompts import DIARY_PROMPT, REPAIR_PROMPT, SYSTEM_PROMPT
from .schema import (
    ActionParseError,
    _iter_balanced_objects,
    _remove_trailing_commas,
    parse_action_text,
)


@dataclass(frozen=True)
class Cassette:
    name: str
    base_url: str
    model: str = "auto"
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
                # "auto" (or empty) means: ask the server what is loaded.
                model=row.get("model", "auto"),
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


def _action_schema() -> dict[str, Any]:
    # maxLength on think/say is load-bearing: a grammar-constrained server forces
    # the strings to *close*, so a reasoning model cannot ramble forever inside
    # the "think" field and never emit a valid object (observed on LM Studio).
    return {
        "name": "hero_action",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "think": {"type": "string", "maxLength": 240},
                "action": {"type": "string", "enum": sorted(ACTION_WORDS)},
                "args": {"type": "object", "additionalProperties": True},
                "say": {"type": "string", "maxLength": 160},
            },
            "required": ["think", "action", "args", "say"],
        },
    }


def _diary_schema() -> dict[str, Any]:
    return {
        "name": "hero_diary",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"diary": {"type": "string", "maxLength": 240}},
            "required": ["diary"],
        },
    }


class _ResponseFormatRejected(Exception):
    """Raised when a server refuses a response_format we sent (older backends)."""


class OpenAICompatibleBrain:
    def __init__(self, cassette: Cassette, timeout: float = 45.0) -> None:
        self.cassette = cassette
        self.timeout = timeout
        self.observer = ObservationBuilder()
        self._model: str | None = None
        self._schema_supported = True

    def choose(self, sim: object) -> GameAction:
        obs = self.observer.build(sim)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n" + self.cassette.persona},
            {"role": "user", "content": json.dumps(obs, ensure_ascii=False)},
        ]
        first = self._chat(messages, schema=_action_schema())
        try:
            return parse_action_text(first).to_game_action()
        except ActionParseError as exc:
            repair_messages = messages + [
                {"role": "assistant", "content": first},
                {"role": "user", "content": REPAIR_PROMPT + f"\nError: {exc}"},
            ]
            repaired = self._chat(repair_messages, schema=_action_schema())
            try:
                return parse_action_text(repaired).to_game_action()
            except ActionParseError as repair_exc:
                # The sim turns this unknown action into Confusion (spec §4.3).
                return GameAction(
                    action="invalid_llm_output",
                    args={},
                    think=f"Could not repair LLM JSON: {repair_exc}",
                    say="The words came apart in my hands.",
                )

    def write_diary(self, sim: object, season: str, weather: str) -> str | None:
        """Author tonight's diary with a separate, bounded call (spec §5)."""
        day_log = list(getattr(sim, "day_log", []) or [])
        hero = sim.hero
        context = {
            "day": sim.world.day,
            "season": season,
            "weather": weather,
            "stats": {"hp": hero.hp, "hunger": hero.hunger, "water": hero.water, "sanity": hero.sanity},
            "today": day_log[-8:],
        }
        messages = [
            {"role": "system", "content": DIARY_PROMPT + "\n" + self.cassette.persona},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]
        raw = self._chat(messages, schema=_diary_schema()).strip()
        if not raw:
            return None
        for candidate in _iter_balanced_objects(raw):
            try:
                obj = json.loads(_remove_trailing_commas(candidate))
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("diary"):
                return str(obj["diary"]).strip() or None
        cleaned = raw.strip("`").strip()
        return cleaned or None

    def _resolve_model(self) -> str:
        if self._model:
            return self._model
        wanted = (self.cassette.model or "").strip()
        if wanted and wanted.lower() != "auto":
            self._model = wanted
            return self._model
        self._model = self._discover_model()
        return self._model

    def _discover_model(self) -> str:
        url = self.cassette.base_url.rstrip("/") + "/models"
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {self.cassette.api_key}"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read models from {url}: {exc}") from exc
        models = [str(m.get("id", "")) for m in data.get("data", []) if m.get("id")]
        chat = [m for m in models if "embed" not in m.lower()]
        if not chat:
            raise RuntimeError(f"No chat model is loaded at {url}.")
        return chat[0]

    def _chat(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if not self.cassette.base_url:
            raise RuntimeError("Cassette has no base_url.")
        payload: dict[str, Any] = {
            "model": self._resolve_model(),
            "messages": messages,
            "temperature": self.cassette.temperature,
            # Reasoning models spend tokens thinking before the answer; give a
            # floor so the bounded JSON answer is never truncated away.
            "max_tokens": max(max_tokens or self.cassette.max_tokens, 384),
        }
        if schema is not None and self.cassette.json_mode and self._schema_supported:
            payload["response_format"] = {"type": "json_schema", "json_schema": schema}
        try:
            return self._post_chat(payload)
        except _ResponseFormatRejected:
            # Older backend without json_schema support: drop it and let the
            # tolerant parser recover the object from free text.
            self._schema_supported = False
            payload.pop("response_format", None)
            return self._post_chat(payload)

    def _post_chat(self, payload: dict[str, Any]) -> str:
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
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                pass
            if exc.code == 400 and "response_format" in payload:
                raise _ResponseFormatRejected(detail) from exc
            raise RuntimeError(f"LLM request failed: HTTP {exc.code}: {detail[:200]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        data = json.loads(body)
        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        if not content.strip():
            # Reasoning models (e.g. Qwen3.x on LM Studio) leave content empty and
            # put the schema-constrained answer in reasoning_content instead.
            content = message.get("reasoning_content") or ""
        return content
