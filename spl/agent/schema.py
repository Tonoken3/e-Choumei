from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from spl.core.actions import ACTION_WORDS, GameAction


class ActionParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedAction:
    think: str
    action: str
    args: dict[str, Any]
    say: str

    def to_game_action(self) -> GameAction:
        return GameAction(action=self.action, args=self.args, think=self.think, say=self.say)


def parse_action_text(text: str) -> ParsedAction:
    raw = _strip_code_fences(text.strip())
    raw = _extract_json_object(raw)
    raw = _remove_trailing_commas(raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ActionParseError(f"Invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ActionParseError("Action must be a JSON object.")
    action = str(data.get("action", "")).strip()
    if action not in ACTION_WORDS:
        raise ActionParseError(f"Unknown action: {action}")
    args = data.get("args", {})
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ActionParseError("args must be an object.")
    return ParsedAction(
        think=str(data.get("think", ""))[:500],
        action=action,
        args=args,
        say=str(data.get("say", ""))[:160],
    )


def _strip_code_fences(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)

