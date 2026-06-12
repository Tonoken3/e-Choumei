from __future__ import annotations

import json
import time
import tomllib
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spl.core.actions import ACTION_WORDS, GameAction

from .observer import ObservationBuilder
from .prompts import (
    AGGREGATE_PROMPT,
    COMPILE_PROMPT,
    DIARY_PROMPT,
    MOTTO_PROMPT,
    OVERFLOW_REPAIR_PROMPT,
    REPAIR_PROMPT,
    SYSTEM_PROMPT,
    VERIFY_PROMPT,
    lens_prompt,
    lenses_for,
    system_prompt_for_difficulty,
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
    # Reasoning no-count (2026-06-12 ruling): hidden reasoning tokens are NOT
    # taxed by the tier budget — the budget binds the visible answer; deep
    # thought pays only in wall-clock time. True for models that emit
    # reasoning_content (Step3.7 etc.); grants a 4096-token safety ceiling.
    reasoning: bool = False
    # 八識熟考: N (0=off) concurrent inference streams over ONE observation, each a
    # themed lens (八識), aggregated into one action. A silicon mind is PARALLEL —
    # with continuous batching (vLLM) N thoughts cost ~1 thought of wall-clock.
    parallel: int = 0


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
                reasoning=bool(row.get("reasoning", False)),
                parallel=int(row.get("parallel", 0)),
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


def condition_cap_index(hero: object) -> tuple[int, str | None]:
    """内省は満腹の上に立つ — the body gates the mind (自然の摂理).

    Returns (max tier index, reason). A starving, parched or broken hermit
    cannot run a careful verify pass no matter how fast the rig is; only a
    nourished one can introspect. Revelation (天の声) still reaches them —
    that is what the watcher is for.
    """
    if hero.hp <= 25 or hero.sanity <= 20:
        return 0, "心身衰弱"
    if hero.hunger <= 0 or hero.water <= 10:
        return 0, "飢渇"
    if hero.hunger <= 25 or hero.water <= 25 or hero.sanity <= 40 or hero.stamina <= 15:
        return 1, "疲弊"
    return len(_TIERS) - 1, None


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


def _has_cjk(s: str) -> bool:
    return any("぀" <= ch <= "鿿" or "ｦ" <= ch <= "ﾟ" for ch in s)


def _clean_counsel(content: str | None) -> str | None:
    """Pull a usable two-sentence counsel out of a lens reply.

    A clean model returns the two Japanese sentences directly (passed through). A
    reasoning model (e.g. the abliterated Qwen on vLLM) prefixes an English
    'Here's a thinking process:' trace; we drop strip-prefix think-fences and, if
    the head is an English trace, keep only the Japanese tail lines. Returns None
    when nothing usable remains (the lens is then skipped)."""
    if not content:
        return None
    text = content.strip()
    # strip a leading <think>...</think> fence if present
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if not text:
        return None
    # If the whole reply carries CJK and no obvious English reasoning preamble,
    # keep it as-is (the clean-model path).
    head = text[:40].lower()
    leaked = ("thinking process" in text.lower()[:80]) or head.startswith(
        ("here's", "here is", "okay", "let me", "first", "1.")
    )
    if not leaked:
        return text or None
    # Reasoning leaked: keep only the Japanese lines (the actual counsel, if any
    # was emitted before truncation).
    jp_lines = [ln.strip() for ln in text.splitlines() if _has_cjk(ln) and ln.strip()]
    # drop lines that are clearly part of the English analysis (key: value echoes)
    jp_lines = [ln for ln in jp_lines if not ln.lower().startswith(("**", "- ", "* "))]
    tail = " ".join(jp_lines[-2:]).strip()
    return tail or None


def _parse_counsel(content: str | None) -> str | None:
    """Extract the 'counsel' field from a lens reply. The schema makes the server
    emit {"counsel": "..."}; we read the first balanced object that carries it.
    Falls back to _clean_counsel for a backend that dropped the schema (older
    servers) and returned free text instead."""
    if not content:
        return None
    for candidate in _iter_balanced_objects(content):
        try:
            obj = json.loads(_remove_trailing_commas(candidate))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("counsel"):
            return str(obj["counsel"]).strip() or None
    return _clean_counsel(content)


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


def _counsel_schema() -> dict[str, Any]:
    """八識の進言 schema: one short Japanese field. maxLength is load-bearing —
    a grammar-constrained server is forced to CLOSE the string, so a reasoning
    model cannot burn its whole budget on a 'thinking process' and emit only a
    truncated trace (observed on the abliterated Qwen). The lens counsel becomes
    the actual two-sentence advice, not a leaked analysis."""
    return {
        "name": "lens_counsel",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"counsel": {"type": "string", "maxLength": 160}},
            "required": ["counsel"],
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
                # ぼうけんのしょ: exactly three imperative lessons for the NEXT life.
                "lessons": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {"type": "string", "maxLength": 80},
                },
            },
            "required": ["motto", "words", "highlights", "lessons"],
        },
    }


def _compile_schema() -> dict[str, Any]:
    # 家訓: EXACTLY 5 articles, each ≤80 chars. min==max==5 forces a fixed-size
    # canon — the book is revised, never grown.
    return {
        "name": "house_code",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "lessons": {
                    "type": "array",
                    "minItems": 5,
                    "maxItems": 5,
                    "items": {"type": "string", "maxLength": 80},
                },
            },
            "required": ["lessons"],
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
        # 入植者の来歴: at settlement the WATCHER (not the model) may write the
        # hermit's self-introduction, which becomes part of the system prompt —
        # prompt engineering as a first-class game mechanic and benchmark axis
        # (same model + island, different player-authored souls). Plain attributes,
        # set after construction. ``append`` grafts the来歴 onto the cassette's own
        # persona; ``replace`` makes the来歴 the WHOLE persona (cassette's is
        # ignored unless the player left the来歴 empty).
        self.player_persona: str = ""
        self.persona_mode: str = "append"  # "append" | "replace"
        self.observer = ObservationBuilder()
        self._model: str | None = None
        self._schema_supported = True
        # 落丁: the finish_reason of the most recent completion. "length" means the
        # answer was CUT OFF by max_tokens (a long-reasoning model thought too long
        # and the JSON never closed). Only the serial propose path reads it.
        self._last_finish_reason: str | None = None
        # 思考予算 telemetry the UIs read.
        self._tps_samples: deque[float] = deque(maxlen=self._TPS_WINDOW)
        self.calls = 0                  # total action-choosing turns
        self.verify_corrections = 0     # times the VERIFY pass changed the action
        self.tier_history: list[str] = []  # tier name chosen per turn
        # 八識熟考 telemetry.
        self.deliberations = 0          # times deliberate() fanned out the eight識
        self.last_counsels: list[tuple[str, str]] = []  # last fan-out's counsels
        # UI toggle mirror: when True, choose_or_deliberate fans out even with no
        # body scream (the [熟考] watch button / CLI --deliberate set this).
        self.deliberate_forced = False

    # -- 思考予算: tokens/sec → tier ----------------------------------------
    @property
    def forced_tps(self) -> float:
        return max(0.0, float(getattr(self.cassette, "tps", 0.0) or 0.0))

    @property
    def parallel(self) -> int:
        """八識熟考 fan-out width (0 = off)."""
        return max(0, int(getattr(self.cassette, "parallel", 0) or 0))

    # -- 入植者の来歴 (player-written persona) -------------------------------
    def effective_persona(self) -> str:
        """The persona actually fed into every system prompt.

        replace → the watcher's 来歴 IS the persona (falls back to the cassette's
        own persona when the 来歴 is empty). append → the cassette's persona with
        the 来歴 grafted on under a header that tells the model this is who it is.
        With no 来歴 set, both modes return the cassette persona unchanged — so a
        run with no --persona behaves exactly as before."""
        own = self.cassette.persona or ""
        player = (self.player_persona or "").strip()
        if self.persona_mode == "replace":
            return player or own
        if player:
            return own + (
                "\n[入植者の来歴 — the watcher wrote this about you. "
                "It is who you are.]\n" + player
            )
        return own

    def system_for(self, base: str) -> str:
        """Assemble a system message. The player-written 来歴 (≤140 chars, X-post
        sized) is injected at the HEAD of EVERY call — identity precedes law —
        then the base prompt, then the cassette persona (unless replace mode)."""
        player = (self.player_persona or "").strip()[:140]
        parts: list[str] = []
        if player:
            parts.append("[入植者の来歴 — the watcher wrote this about you. It is who you are.]\n" + player)
        parts.append(base)
        if not (player and self.persona_mode == "replace"):
            if self.cassette.persona:
                parts.append(self.cassette.persona)
        return "\n".join(parts)

    def system_prompt_for(self, sim: object) -> str:
        """The full action system message for THIS run — with the settler's
        briefing told truthfully for the sim's きびしさ (so a 修羅 hermit reads the
        修羅 arithmetic). Falls back to the 修羅 SYSTEM_PROMPT when the sim has no
        difficulty (older fakes / a None sim), so back-compat holds."""
        difficulty = getattr(sim, "difficulty", None)
        base = system_prompt_for_difficulty(difficulty) if difficulty else SYSTEM_PROMPT
        return self.system_for(base)

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
        """e.g. '羅漢 212t/s 検証修正3', or '羅漢→雲水(飢渇) ...' when the
        hermit's condition has capped the effective tier."""
        tier = self.current_tier()
        name = tier.name
        eff = getattr(self, "effective_tier_name", None)
        note = getattr(self, "condition_note", None)
        if eff and note and eff != name:
            name = f"{name}→{eff}({note})"
        line = f"{name} {self.avg_tps():.0f}t/s 検証修正{self.verify_corrections}"
        # 八識熟考: surface how many parallel deliberations have fired this run.
        if self.deliberations > 0:
            line += f" 熟考{self.deliberations}"
        return line

    def _record_tps(self, completion_tokens: int, seconds: float) -> None:
        if self.forced_tps > 0 or seconds <= 0 or completion_tokens <= 0:
            return
        self._tps_samples.append(completion_tokens / seconds)

    def _record_tps_aggregate(self, completion_tokens: int, seconds: float) -> None:
        """八識熟考-honest TPS: feed ONE rolling sample whose tokens are the SUM of
        every call's completion tokens across the fan-out and whose seconds are the
        WHOLE fan-out's wall-clock. That ratio is the serving stack's real
        throughput under continuous batching, so batching legitimately EARNS a
        higher tier. deliberate() suppresses per-call recording (see ``_chat``'s
        ``record_tps`` flag) so nothing is double-counted."""
        self._record_tps(completion_tokens, seconds)

    def choose(
        self, sim: object, extra: list[dict[str, str]] | None = None
    ) -> GameAction:
        """Full action-choosing turn with the 思考予算 tiers, repair round and
        VERIFY pass. ``extra`` messages (e.g. the MAGI 評定 for the day) are
        appended after the observation, before the model answers, and are carried
        through every proposal / repair / verify call so the standing counsel
        steers the whole turn without disturbing the tier machinery."""
        self.calls += 1
        budget = self.current_tier()
        # 内省は満腹の上に立つ: the hermit's condition caps the effective tier.
        cap_idx, cap_reason = condition_cap_index(sim.hero)
        hw_idx = _TIERS.index(budget)
        if cap_idx < hw_idx:
            budget = _TIERS[cap_idx]
            self.condition_note = cap_reason
        else:
            self.condition_note = None
        self.effective_tier_name = budget.name
        self.tier_history.append(budget.name)
        obs = self.observer.build(sim)
        messages = [
            # きびしさ: the briefing numbers match the island actually being played.
            {"role": "system", "content": self.system_prompt_for(sim)},
            {"role": "user", "content": json.dumps(obs, ensure_ascii=False)},
        ]
        if extra:
            messages.extend(extra)

        # 1) Generate `candidates` action proposals. When the budget wants more
        #    than one AND the cassette declares parallel slots (LM Studio serves 4
        #    by default; vLLM 8), issue them CONCURRENTLY so the second thought
        #    costs ~no extra wall-clock under continuous batching. Otherwise the
        #    original sequential path. Failures skip; the first-index successful
        #    proposal stays the deterministic preference (order-stable).
        n = max(1, budget.candidates)
        proposals: list[GameAction] = []
        last_exc: ActionParseError | None = None
        if n > 1 and self.parallel > 1:
            # One thread per candidate (bounded by the parallel slot count). The
            # results are reassembled IN INDEX ORDER so the first valid proposal
            # is preferred deterministically, exactly like the sequential path.
            workers = min(n, self.parallel)
            results: list[GameAction | None] = [None] * n
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._propose, messages, budget): i for i in range(n)
                }
                for fut in futures:
                    i = futures[fut]
                    try:
                        results[i] = fut.result()
                    except ActionParseError as exc:
                        last_exc = exc  # this candidate did not parse — skip it
            proposals = [r for r in results if r is not None]
        else:
            for _ in range(n):
                try:
                    proposals.append(self._propose(messages, budget))
                except ActionParseError as exc:
                    last_exc = exc
                    # A proposal that will not parse becomes Confusion if we
                    # cannot repair (雲水) — but only if NO valid proposal exists.
                    if not proposals:
                        break
        if not proposals:
            # 落丁: if the parse failure was a max_tokens cut, name the cause so
            # sim.step() can log "思考が長すぎて言葉にならなかった" — the hermit
            # reads it next turn and learns to think shorter.
            args = {"cause": "overflow"} if getattr(last_exc, "overflow", False) else {}
            return GameAction(
                action="invalid_llm_output",
                args=args,
                think=f"Could not parse LLM JSON: {last_exc}",
                say="The words came apart in my hands.",
            )

        # 2) Optional VERIFY pass: a second thought that catches would-be
        #    world-rejects and returns a corrected (or best-of) action.
        if budget.verify and proposals:
            verified = self._verify(messages, proposals, obs, budget, extra=extra)
            if verified is not None:
                if not _same_action(verified, proposals[0]):
                    self.verify_corrections += 1
                return verified
        return proposals[0]

    # -- 八識熟考 (parallel deliberation) -------------------------------------
    def _lens_counsel(
        self, lens: str, theme: str, obs_json: str, tokens_out: dict[str, int]
    ) -> str | None:
        """One識's counsel: a SHORT schema-forced call reading the same observation
        through one themed lens. The one-field {counsel} schema makes a grammar-
        constrained server CLOSE the string, so even a reasoning model emits the
        two-sentence advice instead of a truncated thinking trace. Records the
        completion tokens into ``tokens_out`` (keyed by lens) for the aggregate-TPS
        sum. Per-call TPS is OFF; failures return None so a dead lens is simply
        skipped, never crashing the turn. ``max_tokens`` gives reasoning models
        breathing space before the JSON closes — paid honestly in wall-clock."""
        messages = [
            {"role": "system", "content": self.system_for(lens_prompt(lens, theme))},
            {"role": "user", "content": obs_json},
        ]
        cap = max(160, self.cassette.max_tokens)
        try:
            content, tokens, _elapsed = self._chat_timed(
                messages, schema=_counsel_schema(), max_tokens=cap, record_tps=False
            )
        except Exception:  # noqa: BLE001 - one dead stream must not sink the turn
            return None
        tokens_out[lens] = tokens
        return _parse_counsel(content)

    def deliberate(self, sim: object) -> GameAction:
        """八識熟考: build the observation ONCE, fan out N CONCURRENT lens counsels
        (八識), then synthesize ONE schema-forced action (阿頼耶識) that reuses the
        normal _propose machinery (tier think/say bounds, breathing space, repair).

        The whole fan-out is timed as one unit: the aggregate TPS = (sum of all
        N+1 completion tokens) / (wall-clock of the entire deliberation), fed once
        to the rolling samples — so batching legitimately earns its tier. A body
        is serial; a silicon mind is parallel."""
        self.calls += 1
        self.deliberations += 1
        # Effective tier is condition-capped exactly like choose() — a starving
        # mind fans out but synthesizes poorly.
        budget = self.current_tier()
        cap_idx, cap_reason = condition_cap_index(sim.hero)
        hw_idx = _TIERS.index(budget)
        if cap_idx < hw_idx:
            budget = _TIERS[cap_idx]
            self.condition_note = cap_reason
        else:
            self.condition_note = None
        self.effective_tier_name = budget.name
        self.tier_history.append(budget.name)

        obs = self.observer.build(sim)
        obs_json = json.dumps(obs, ensure_ascii=False)
        lenses = lenses_for(self.parallel)

        # The deliberation's wall-clock spans the whole fan-out + the aggregate.
        started = time.monotonic()
        token_sum = 0
        counsels: list[tuple[str, str]] = []
        lens_tokens: dict[str, int] = {}
        # Fan out CONCURRENTLY: one request per thread (urllib is thread-safe per
        # request). vLLM continuous batching makes the eight cost ~one wall-clock.
        if lenses:
            with ThreadPoolExecutor(max_workers=len(lenses)) as pool:
                futures = [
                    (lens, pool.submit(self._lens_counsel, lens, theme, obs_json, lens_tokens))
                    for lens, theme in lenses
                ]
                for lens, fut in futures:
                    try:
                        text = fut.result(timeout=self.timeout + 5.0)
                    except Exception:  # noqa: BLE001 - per-thread timeout/error → skip
                        text = None
                    if text:
                        counsels.append((lens, text))
        token_sum += sum(lens_tokens.values())

        # 阿頼耶識: one schema-forced action with the observation + the N counsels.
        counsel_block = "\n".join(f"【{lens}】{text}" for lens, text in counsels)
        messages = [
            {"role": "system", "content": self.system_prompt_for(sim)},
            {"role": "user", "content": obs_json},
            {
                "role": "user",
                "content": AGGREGATE_PROMPT
                + ("\n八識の進言:\n" + counsel_block if counsel_block else "")
                + "\n八識の進言を統合し、最善の一手を返せ。",
            },
        ]
        try:
            action, agg_tokens = self._propose_timed(messages, budget, record_tps=False)
        except ActionParseError as exc:
            action = GameAction(
                action="invalid_llm_output",
                args={},
                think=f"Could not parse LLM JSON: {exc}",
                say="The words came apart in my hands.",
            )
            agg_tokens = 0
        token_sum += agg_tokens

        # Aggregate-TPS honesty: one rolling sample for the WHOLE deliberation.
        elapsed = time.monotonic() - started
        self._record_tps_aggregate(token_sum, elapsed)
        # Stash the counsels so a UI/smoke can show what the eight識 said.
        self.last_counsels = counsels
        return action

    def choose_or_deliberate(self, sim: object) -> GameAction:
        """The single entry both UIs call. 八識熟考 fires when the cassette has
        ``parallel > 0`` AND (the toggle/CLI forced it OR the flesh is screaming:
        a ``body`` block in the observation summons full attention — biology).
        Otherwise the normal serial choose()."""
        if self.parallel > 0 and (self.deliberate_forced or self._body_screams_present(sim)):
            return self.deliberate(sim)
        return self.choose(sim)

    def _body_screams_present(self, sim: object) -> bool:
        """True when the observation would carry a ``body`` block (the flesh is
        screaming). Reuses the observer's own interoception so the trigger and the
        prompt agree exactly."""
        try:
            return bool(self.observer._body_screams(sim))
        except Exception:  # noqa: BLE001
            return False

    # -- MAGI seams ----------------------------------------------------------
    # The MAGI council reuses one OpenAICompatibleBrain per seat (sharing all of
    # the schema / parse / TPS machinery here). These two methods are the minimal
    # public surface the council drives: one schema-forced action call from a
    # prebuilt observation, and one free-text "think" call (no schema).
    def propose_action(
        self,
        obs: dict[str, Any],
        budget: ThinkingBudget | None = None,
        extra: list[dict[str, str]] | None = None,
    ) -> GameAction:
        """One schema-forced action call from an already-built observation,
        optionally with extra messages (a plan or a moderator ruling) appended
        before the model answers. Raises ActionParseError on unparseable output."""
        budget = budget or self.current_tier()
        messages = [
            {"role": "system", "content": self.system_for(SYSTEM_PROMPT)},
            {"role": "user", "content": json.dumps(obs, ensure_ascii=False)},
        ]
        if extra:
            messages.extend(extra)
        raw = self._chat(messages, schema=_action_schema(budget), max_tokens=budget.max_tokens)
        return parse_action_text(raw).to_game_action()

    def think_freetext(self, system: str, user: str, max_tokens: int = 128) -> str:
        """One free-text (no-schema) call — the 'think' half of a relay. Returns
        the raw content (think tags and all); callers use it as plan/ruling text."""
        messages = [
            {"role": "system", "content": self.system_for(system)},
            {"role": "user", "content": user},
        ]
        return self._chat(messages, schema=None, max_tokens=max_tokens).strip()

    def _propose(self, messages: list[dict[str, str]], budget: ThinkingBudget) -> GameAction:
        """One action proposal, with a repair round when the budget allows it.

        max_tokens honours the cassette's declared breathing space when it
        exceeds the tier budget: long-reasoning models (hidden reasoning_content
        before any JSON) suffocate under small caps; their real cost is paid in
        wall-clock time, which the TPS measurement keeps honest."""
        action, _tokens = self._propose_timed(messages, budget, record_tps=True)
        return action

    def _completion_cap(self, budget: ThinkingBudget) -> int:
        """Tokens allowed for one completion. The tier budget binds the VISIBLE
        answer; a declared reasoning model additionally gets a 4096 safety
        ceiling for its hidden chain of thought — reasoning tokens are
        no-count (the ruling: thought is taxed in wall-clock, not tokens)."""
        cap = max(budget.max_tokens, self.cassette.max_tokens)
        if self.cassette.reasoning:
            cap = max(cap, 4096)
        return cap

    def _propose_timed(
        self, messages: list[dict[str, str]], budget: ThinkingBudget,
        record_tps: bool = True,
    ) -> tuple[GameAction, int]:
        """``_propose`` plus the summed completion tokens of its call(s), and a
        ``record_tps`` switch. 八識熟考's aggregator calls this with
        ``record_tps=False`` so the per-call TPS is not double-counted against the
        single aggregate sample deliberate() records for the whole fan-out."""
        cap = self._completion_cap(budget)
        schema = _action_schema(budget)
        first, tokens, _elapsed = self._chat_timed(
            messages, schema=schema, max_tokens=cap, record_tps=record_tps
        )
        # 落丁: did this completion get CUT OFF by max_tokens? Captured on the
        # SERIAL propose path right after _post_chat stashed it. A length cut means
        # the model thought too long and the JSON never closed.
        overflowed = self._last_finish_reason == "length"
        try:
            return parse_action_text(first).to_game_action(), tokens
        except ActionParseError as exc:
            if not budget.repair:
                # 雲水: no repair budget — straight to invalid_llm_output. Mark the
                # length cut so choose() can name the cause in the world log.
                exc.overflow = overflowed
                raise
            # Overflow-aware repair: when the first answer was a length cut, give
            # direct meta-feedback (think in three sentences, then close the JSON)
            # instead of the generic "not valid JSON" nudge.
            repair_prompt = OVERFLOW_REPAIR_PROMPT if overflowed else REPAIR_PROMPT
            repair_messages = messages + [
                {"role": "assistant", "content": first},
                {"role": "user", "content": repair_prompt + f"\nError: {exc}"},
            ]
            repaired, rtokens, _e = self._chat_timed(
                repair_messages, schema=schema, max_tokens=cap, record_tps=record_tps
            )
            # If the repair ALSO got cut off, the overflow signal persists so the
            # raised error still names the cause as "overflow".
            repair_overflowed = self._last_finish_reason == "length"
            try:
                return parse_action_text(repaired).to_game_action(), tokens + rtokens
            except ActionParseError as rexc:
                rexc.overflow = overflowed or repair_overflowed
                raise

    def _verify(
        self,
        messages: list[dict[str, str]],
        proposals: list[GameAction],
        obs: dict[str, Any],
        budget: ThinkingBudget,
        extra: list[dict[str, str]] | None = None,
    ) -> GameAction | None:
        """One VERIFY call: re-show the observation and the proposal(s) and ask
        for a corrected / best-valid action under the same JSON contract. Any
        ``extra`` messages (the day's 評定) are carried so verify weighs the same
        standing counsel as the proposal calls did."""
        proposal_json = [
            {"action": p.action, "args": p.args, "think": p.think, "say": p.say}
            for p in proposals
        ]
        verify_messages = [
            {"role": "system", "content": self.system_for(VERIFY_PROMPT)},
            {
                "role": "user",
                "content": json.dumps(
                    {"observation": obs, "proposals": proposal_json},
                    ensure_ascii=False,
                ),
            },
        ]
        if extra:
            verify_messages.extend(extra)
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
            {"role": "system", "content": self.system_for(DIARY_PROMPT)},
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
            {"role": "system", "content": self.system_for(MOTTO_PROMPT)},
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
                lessons = obj.get("lessons") or []
                if not isinstance(lessons, list):
                    lessons = []
                return {
                    "motto": str(obj["motto"]).strip(),
                    "words": str(obj.get("words", "")).strip(),
                    "highlights": [str(h).strip() for h in highlights if str(h).strip()][:5],
                    # ぼうけんのしょ: the three lessons the next life inherits.
                    "lessons": [str(s).strip() for s in lessons if str(s).strip()][:3],
                }
        return None

    def compile_canon(self, book: object) -> list[str] | None:
        """家訓の編纂: revise the lineage's fixed 5-article canon from the current
        canon + the full history (each life's lifespan beside its lessons). The
        编纂者 merges duplicates, keeps what long lives carried, rewrites what
        short lives carried, and adds what the newest death teaches. Returns the
        5 articles, or None on any failure (the caller then uses fallback_compile)."""
        context = {
            "canon": list(getattr(book, "canon", []) or []),
            "history": book.history_table(),
        }
        messages = [
            {"role": "system", "content": self.system_for(COMPILE_PROMPT)},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]
        try:
            raw = self._chat(messages, schema=_compile_schema(), max_tokens=768).strip()
        except (RuntimeError, urllib.error.URLError):
            return None
        if not raw:
            return None
        for candidate in _iter_balanced_objects(raw):
            try:
                obj = json.loads(_remove_trailing_commas(candidate))
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("lessons"), list):
                lessons = [str(s).strip() for s in obj["lessons"] if str(s).strip()]
                if lessons:
                    return lessons[:5]
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
        record_tps: bool = True,
    ) -> str:
        """One chat completion. ``record_tps`` feeds the per-call 思考予算
        measurement; deliberate() turns it OFF on the fan-out calls and records a
        single aggregate sample instead (so batching is not double-counted)."""
        content, _tokens, _elapsed = self._chat_timed(
            messages, schema=schema, max_tokens=max_tokens, record_tps=record_tps
        )
        return content

    def _chat_timed(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        record_tps: bool = True,
    ) -> tuple[str, int, float]:
        """Like ``_chat`` but also returns (completion_tokens, wall_seconds) so the
        八識熟考 fan-out can sum the tokens and time the whole batch itself."""
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
        if record_tps:
            self._record_tps(completion_tokens, elapsed)
        return content, completion_tokens, elapsed

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
        choice = data["choices"][0]
        # 落丁検出: stash WHY the completion stopped. Only meaningful on the SERIAL
        # propose path (choose()): the threaded 八識 lens calls may overwrite this
        # concurrently, but lenses never repair, so the race is acceptable.
        self._last_finish_reason = choice.get("finish_reason")
        message = choice["message"]
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
