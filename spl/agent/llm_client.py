from __future__ import annotations

import json
import time
import tomllib
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spl.core.actions import ACTION_WORDS, GameAction

from .observer import ObservationBuilder
from .prompts import (
    DIARY_PROMPT,
    MOTTO_PROMPT,
    REPAIR_PROMPT,
    SYSTEM_PROMPT,
    VERIFY_PROMPT,
)
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
    # 思考予算: 0 (or absent) = auto-measure the serving stack's tokens/sec from
    # each completion; >0 = force a constant TPS (measurement ignored), so a slow
    # or fast rig can be simulated without changing the hardware.
    tps: float = 0.0


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
                tps=float(row.get("tps", 0.0)),
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


# ===========================================================================
# 思考予算 (Thinking Budget): inference speed becomes survival capability.
#
# Every hermit gets the same wall-clock "instinct time" per action. How much
# THINKING fits in it depends on the serving stack's tokens/sec — model × quant
# × engine × hardware. A faster rig "thinks" deeper (more tokens, a verify pass,
# even multiple candidates) and so trips fewer world-rejects (fumbles) and lives
# longer. SPL thus doubles as an endurance race for the whole serving stack.
# ===========================================================================
@dataclass(frozen=True)
class ThinkingBudget:
    name: str          # tier name (雲水 / 行者 / 羅漢 / 仙界)
    tps_label: str     # human-readable TPS band, e.g. "<30" or "100–300"
    max_tokens: int    # completion budget for the action call
    think_len: int     # maxLength of the "think" field
    say_len: int       # maxLength of the "say" field
    repair: bool       # allow a repair round when parsing fails
    verify: bool       # run a second VERIFY pass over the proposal(s)
    candidates: int    # how many action proposals to generate


# The four tiers, by ascending tokens/sec. The bands are picked so a sluggish
# CPU build (雲水) gets a terse, no-repair, no-verify budget, while a fast NVFP4
# rig (仙界) gets room to propose twice and verify.
_TIERS: tuple[ThinkingBudget, ...] = (
    ThinkingBudget("雲水", "<30", 192, 60, 80, repair=False, verify=False, candidates=1),
    ThinkingBudget("行者", "30–100", 384, 240, 160, repair=True, verify=False, candidates=1),
    ThinkingBudget("羅漢", "100–300", 512, 280, 160, repair=True, verify=True, candidates=1),
    ThinkingBudget("仙界", "300+", 640, 320, 160, repair=True, verify=True, candidates=2),
)


def tier_for_tps(tps: float) -> ThinkingBudget:
    """Map a measured tokens/sec to a 思考予算 tier.

    Boundaries (inclusive lower): <30 雲水, 30–100 行者, 100–300 羅漢, 300+ 仙界.
    """
    if tps < 30:
        return _TIERS[0]
    if tps < 100:
        return _TIERS[1]
    if tps < 300:
        return _TIERS[2]
    return _TIERS[3]


def _same_action(a: GameAction, b: GameAction) -> bool:
    """True when two actions are effectively identical (the verify pass made no
    correction). Compares the verb and its args, ignoring think/say flavour."""
    return a.action == b.action and a.args == b.args


def _action_schema(budget: ThinkingBudget | None = None) -> dict[str, Any]:
    # maxLength on think/say is load-bearing: a grammar-constrained server forces
    # the strings to *close*, so a reasoning model cannot ramble forever inside
    # the "think" field and never emit a valid object (observed on LM Studio).
    # The bounds are 思考予算-parameterized: a slow tier gets a terser think/say.
    think_len = budget.think_len if budget else 240
    say_len = budget.say_len if budget else 160
    return {
        "name": "hero_action",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "think": {"type": "string", "maxLength": think_len},
                "action": {"type": "string", "enum": sorted(ACTION_WORDS)},
                "args": {"type": "object", "additionalProperties": True},
                "say": {"type": "string", "maxLength": say_len},
            },
            "required": ["think", "action", "args", "say"],
        },
    }


def _motto_schema() -> dict[str, Any]:
    return {
        "name": "hermit_motto",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "motto": {"type": "string", "maxLength": 90},
                "words": {"type": "string", "maxLength": 160},
                "highlights": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 5,
                    "items": {"type": "string", "maxLength": 120},
                },
            },
            "required": ["motto", "words", "highlights"],
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
    # Until this many measured samples exist, assume 行者 (80 TPS) so the first
    # turns are not stuck in the slowest tier while the rolling average warms up.
    _DEFAULT_TPS = 80.0
    _TPS_WINDOW = 8

    def __init__(self, cassette: Cassette, timeout: float = 45.0) -> None:
        self.cassette = cassette
        self.timeout = timeout
        self.observer = ObservationBuilder()
        self._model: str | None = None
        self._schema_supported = True
        # 思考予算 telemetry the UIs read.
        self._tps_samples: deque[float] = deque(maxlen=self._TPS_WINDOW)
        self.calls = 0                  # total action-choosing turns
        self.verify_corrections = 0     # times the VERIFY pass changed the action
        self.tier_history: list[str] = []  # tier name chosen per turn

    # -- 思考予算: tokens/sec → tier ----------------------------------------
    @property
    def forced_tps(self) -> float:
        return max(0.0, float(getattr(self.cassette, "tps", 0.0) or 0.0))

    def avg_tps(self) -> float:
        """Rolling average TPS over the last calls. Forced TPS overrides it.
        Until 2 real samples exist, fall back to the 行者 default (80)."""
        if self.forced_tps > 0:
            return self.forced_tps
        if len(self._tps_samples) < 2:
            return self._DEFAULT_TPS
        return sum(self._tps_samples) / len(self._tps_samples)

    def current_tier(self) -> ThinkingBudget:
        return tier_for_tps(self.avg_tps())

    def status_line(self) -> str:
        """e.g. '羅漢 212t/s 検証修正3' for the UIs."""
        tier = self.current_tier()
        return f"{tier.name} {self.avg_tps():.0f}t/s 検証修正{self.verify_corrections}"

    def _record_tps(self, completion_tokens: int, seconds: float) -> None:
        if self.forced_tps > 0 or seconds <= 0 or completion_tokens <= 0:
            return
        self._tps_samples.append(completion_tokens / seconds)

    def choose(self, sim: object) -> GameAction:
        self.calls += 1
        budget = self.current_tier()
        self.tier_history.append(budget.name)
        obs = self.observer.build(sim)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n" + self.cassette.persona},
            {"role": "user", "content": json.dumps(obs, ensure_ascii=False)},
        ]

        # 1) Generate `candidates` action proposals (sequential calls).
        proposals: list[GameAction] = []
        for _ in range(max(1, budget.candidates)):
            try:
                proposals.append(self._propose(messages, budget))
            except ActionParseError as exc:
                # A proposal that will not parse becomes Confusion if we cannot
                # repair (雲水) — but only if NO valid proposal exists at all.
                if not proposals:
                    return GameAction(
                        action="invalid_llm_output",
                        args={},
                        think=f"Could not parse LLM JSON: {exc}",
                        say="The words came apart in my hands.",
                    )

        # 2) Optional VERIFY pass: a second thought that catches would-be
        #    world-rejects and returns a corrected (or best-of) action.
        if budget.verify and proposals:
            verified = self._verify(messages, proposals, obs, budget)
            if verified is not None:
                if not _same_action(verified, proposals[0]):
                    self.verify_corrections += 1
                return verified
        return proposals[0]

    def _propose(self, messages: list[dict[str, str]], budget: ThinkingBudget) -> GameAction:
        """One action proposal, with a repair round when the budget allows it."""
        first = self._chat(messages, schema=_action_schema(budget), max_tokens=budget.max_tokens)
        try:
            return parse_action_text(first).to_game_action()
        except ActionParseError as exc:
            if not budget.repair:
                # 雲水: no repair budget — straight to invalid_llm_output.
                raise
            repair_messages = messages + [
                {"role": "assistant", "content": first},
                {"role": "user", "content": REPAIR_PROMPT + f"\nError: {exc}"},
            ]
            repaired = self._chat(
                repair_messages, schema=_action_schema(budget), max_tokens=budget.max_tokens
            )
            return parse_action_text(repaired).to_game_action()

    def _verify(
        self,
        messages: list[dict[str, str]],
        proposals: list[GameAction],
        obs: dict[str, Any],
        budget: ThinkingBudget,
    ) -> GameAction | None:
        """One VERIFY call: re-show the observation and the proposal(s) and ask
        for a corrected / best-valid action under the same JSON contract."""
        proposal_json = [
            {"action": p.action, "args": p.args, "think": p.think, "say": p.say}
            for p in proposals
        ]
        verify_messages = [
            {"role": "system", "content": VERIFY_PROMPT + "\n" + self.cassette.persona},
            {
                "role": "user",
                "content": json.dumps(
                    {"observation": obs, "proposals": proposal_json},
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            raw = self._chat(
                verify_messages, schema=_action_schema(budget), max_tokens=budget.max_tokens
            )
            return parse_action_text(raw).to_game_action()
        except (ActionParseError, RuntimeError):
            # A flaky verify must never sink the turn: keep the first proposal.
            return None

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

    def write_motto(self, sim: object) -> dict[str, object] | None:
        """After the year ends, the hermit reads their own five best 銘言 and the
        watcher's chronicle, then distills a 座右の銘 plus the heaven's-voice
        highlights — the final result screen's crown."""
        from spl.agent.chronicle import extract_milestones
        from spl.arena.leaderboard import select_meigen

        hero = sim.hero
        context = {
            "ending": sim.result_reason or ("survived" if sim.completed else "fell"),
            "completed": bool(sim.completed),
            "days_survived": hero.days_survived,
            "score": sim.score(),
            "confusions": hero.confusion_count,
            "final_stats": {"hp": hero.hp, "hunger": hero.hunger, "water": hero.water, "sanity": hero.sanity},
            "best_lines": select_meigen(hero.spoken_lines, 5),
            "chronicle": [ms["text"] for ms in extract_milestones(sim)],
            "diary_tail": [entry.text for entry in sim.memory.diary[-4:]],
        }
        messages = [
            {"role": "system", "content": MOTTO_PROMPT + "\n" + self.cassette.persona},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]
        raw = self._chat(messages, schema=_motto_schema(), max_tokens=768).strip()
        if not raw:
            return None
        for candidate in _iter_balanced_objects(raw):
            try:
                obj = json.loads(_remove_trailing_commas(candidate))
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("motto"):
                highlights = obj.get("highlights") or []
                if not isinstance(highlights, list):
                    highlights = []
                return {
                    "motto": str(obj["motto"]).strip(),
                    "words": str(obj.get("words", "")).strip(),
                    "highlights": [str(h).strip() for h in highlights if str(h).strip()][:5],
                }
        return None

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
        # Respect a 思考予算 max_tokens when one is explicitly passed (so 雲水's
        # terse 192-token budget is not silently raised); only the unbounded
        # callers (diary/motto) keep the 384 floor that protects reasoning models.
        if max_tokens is not None:
            budget_tokens = max(96, int(max_tokens))
        else:
            budget_tokens = max(self.cassette.max_tokens, 384)
        payload: dict[str, Any] = {
            "model": self._resolve_model(),
            "messages": messages,
            "temperature": self.cassette.temperature,
            "max_tokens": budget_tokens,
        }
        if schema is not None and self.cassette.json_mode and self._schema_supported:
            payload["response_format"] = {"type": "json_schema", "json_schema": schema}
        started = time.monotonic()
        try:
            content, completion_tokens = self._post_chat(payload)
        except _ResponseFormatRejected:
            # Older backend without json_schema support: drop it and let the
            # tolerant parser recover the object from free text.
            self._schema_supported = False
            payload.pop("response_format", None)
            content, completion_tokens = self._post_chat(payload)
        # 思考予算 measurement: completion tokens / wall seconds → rolling TPS.
        elapsed = time.monotonic() - started
        if completion_tokens <= 0:
            completion_tokens = max(1, len(content) // 3)  # rough fallback
        self._record_tps(completion_tokens, elapsed)
        return content

    def _post_chat(self, payload: dict[str, Any]) -> tuple[str, int]:
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
        # 思考予算: completion tokens drive the TPS measurement; many reasoning
        # backends report only total tokens, so derive completion if missing.
        usage = data.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        if completion_tokens <= 0:
            total = int(usage.get("total_tokens") or 0)
            prompt = int(usage.get("prompt_tokens") or 0)
            completion_tokens = max(0, total - prompt)
        return content, completion_tokens
