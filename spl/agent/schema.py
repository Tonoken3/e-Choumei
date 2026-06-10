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
    stripped = _strip_code_fences(text.strip())
    data = None
    last_error: Exception | None = None
    # Try every balanced {...} candidate in order; prose braces (e.g. "{note}")
    # simply fail to parse and we move on to the real object.
    for candidate in _iter_balanced_objects(stripped):
        try:
            parsed = json.loads(_remove_trailing_commas(candidate))
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(parsed, dict):
            data = parsed
            break
    if data is None:
        try:
            data = json.loads(_remove_trailing_commas(stripped))
        except json.JSONDecodeError as exc:
            raise ActionParseError(f"Invalid JSON: {last_error or exc}") from (last_error or exc)
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


def _iter_balanced_objects(text: str):
    """Yield each balanced ``{...}`` substring, in start order.

    Small models love to wrap their answer in prose, and that prose often
    contains braces (``Sure! {note}: {"action": ...}``). A naive first-``{``/
    last-``}`` slice captures the wrong span and parsing dies, needlessly
    inflating the confusion rate. We brace-count from each ``{`` (ignoring
    braces inside JSON strings) so the caller can try candidates until one is a
    real object.
    """
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_string = False
        escaped = False
        for j in range(i, n):
            ch = text[j]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[i : j + 1]
                    break
        i += 1


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)

