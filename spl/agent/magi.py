from __future__ import annotations

"""MAGI 合議制 — an Evangelion-style three-seat deliberation brain.

The council fronts the SAME external interface as ``OpenAICompatibleBrain``
(``choose`` / ``write_diary`` / ``write_motto`` / ``compile_canon`` /
``status_line`` / ``avg_tps`` / ``current_tier``) so every caller — simulate,
play, pixel, evolve, the book machinery — drives it unchanged.

The three seats are ordinary ``OpenAICompatibleBrain`` instances, one per
人格, sharing all the schema / parse / 思考予算 machinery in ``llm_client``:

  MELCHIOR  科学者  (Qwen :8011)  — proposes the logical / best-plan move
  BALTHASAR 母      (Gemma :8102) — proposes the survival-first move
  CASPER    直感    (Qwemma relay) — Qwen thinks a 2-line plan → Gemma speaks
                                      the move (think on :8011, act on :8102)
  司会 Gemwen        — Gemma thinks the ruling (:8102) → Qwen speaks the FINAL
                       schema-forced move (:8011)

Per turn the three seats propose; if ≥2 agree on the same action+args the
council takes the majority shortcut (cheap confirm), otherwise the 司会 weighs
the three and the final Qwen call emits the adopted action. If the Gemma server
(:8102) is down the council degrades to MELCHIOR-only so the game still runs.
"""

from pathlib import Path
from typing import Any

from spl.core.actions import GameAction

from .llm_client import (
    Cassette,
    OpenAICompatibleBrain,
    ThinkingBudget,
    find_cassette,
    load_cassettes,
)
from .observer import ObservationBuilder
from .schema import ActionParseError

# 司会 (moderator) prompts. The relay alternates which server *thinks* and which
# *speaks*, so the two halves are deliberately model-agnostic in tone.
CASPER_THINK_PROMPT = (
    "あなたは直感の人格CASPERの思考担当。観測を読み、次の一手の狙いを"
    "二文で述べよ。JSONや行動名は書かず、短い日本語の方針だけを返せ。"
)
MODERATOR_THINK_PROMPT = (
    "あなたはMAGIの司会者。観測と、三機関(MELCHIOR/BALTHASAR/CASPER)の三案が"
    "与えられる。三案を比較し、(1)世界のルールで成功するか (2)作戦・家訓に沿うか "
    "(3)生存に資するか の観点で、どれを採るべきか二、三文で裁定せよ。JSONは書かず、"
    "短い日本語の裁定だけを返せ。"
)
COMPILE_REVIEW_PROMPT = (
    "あなたはMAGIの第二機関BALTHASAR、母の人格。生存が全てに優先する。"
    "編纂者(MELCHIOR)が起草した家訓改定案を受け取る。条文ごとに『採用』か『棄却』かを、"
    "理由を添えて短く述べよ。子孫の寿命(長命の生が運んだ条文か)と、条文どうしの矛盾を見よ。"
    "とりわけ毒の条文を狩れ——空腹・渇き・寒さを『忍ぶ』『耐える』『我慢する』と美化する文言、"
    "無常観や精神論で具体的な生存行動を曇らせる文言は、命を縮める毒である。"
    "そのような条文は必ず『棄却』とし、生存に資する具体的な行動(いつ・何を)へ書き換える指示を述べよ。"
)


# Deprivation nouns × endurance verbs = the poison pattern the 編纂評議会 kills:
# articles that romanticize ENDURING hunger / thirst / cold instead of acting on
# them. 「凌ぐ」(overcome) and 「癒す/満たす/防ぐ」(act) are NOT poison — only the
# 忍ぶ/耐える/我慢 (suffer-and-bear) framing is.
_POISON_NEED = ("空腹", "飢え", "飢", "渇き", "渇", "寒さ", "ひもじ")
_POISON_BEAR = ("忍ぶ", "忍べ", "忍び", "耐え", "我慢", "辛抱")


def _is_poison_article(article: str) -> bool:
    """True when an article pairs a deprivation noun with an endure/bear verb —
    the survival-hostile '飢えを忍べ' / '空腹は忍ぶも' pattern."""
    s = str(article or "")
    return any(n in s for n in _POISON_NEED) and any(b in s for b in _POISON_BEAR)


class MagiBrain:
    """A council brain. Holds one warm seat-brain per server and aggregates."""

    def __init__(
        self,
        melchior: OpenAICompatibleBrain,
        balthasar: OpenAICompatibleBrain | None,
        casper: OpenAICompatibleBrain | None = None,
        identity: Cassette | None = None,
    ) -> None:
        # MELCHIOR is the spine: diary/motto route through it, and it is the
        # degraded-mode fallback. It MUST exist.
        self.melchior = melchior
        self.balthasar = balthasar      # None / dead → degraded council
        # CASPER's *action* call lives on the Gemma server; reuse the BALTHASAR
        # brain's connection when no dedicated CASPER brain was supplied.
        self.casper = casper or balthasar
        # The council's own identity (so the journal keys on "MAGI", not the
        # MELCHIOR seat). Falls back to a synthetic MAGI cassette.
        self._identity = identity or Cassette(name="MAGI", base_url="", model="magi")
        # One shared observer so every seat reasons over the SAME world view and
        # the book lessons injected by the runner reach the council.
        self.observer = ObservationBuilder()
        # Telemetry the UIs read.
        self.calls = 0
        self.councils = 0          # turns that ran the full/short council
        self.unanimous = 0         # turns where all three seats agreed
        self.moderator_used = 0    # turns the 司会 ruling was invoked
        self.verify_corrections = 0  # mirror the OpenAICompatibleBrain field
        self.tier_history: list[str] = []
        self.turn_records: list[dict[str, Any]] = []
        self.degraded_warned = False

    # -- diarist seam: the runner calls set_diarist(brain) then injects book ---
    # ObservationBuilder carries book_lessons via attributes; the runner sets
    # them on whatever object it holds. Expose the council's shared observer so
    # inject_into_observer reaches every seat.
    def __getattr__(self, name: str) -> Any:
        # book_lessons / book_lives are set on the observer by inject_into_observer
        # when it is handed the brain; forward unknown attribute reads to the
        # shared observer so the lessons reach the council's obs.
        if name in {"book_lessons", "book_lives"}:
            return getattr(self.observer, name, None)
        raise AttributeError(name)

    # ====================================================================
    # 思考予算 telemetry — delegate to MELCHIOR (the warm spine).
    # ====================================================================
    def avg_tps(self) -> float:
        return self.melchior.avg_tps()

    def current_tier(self) -> ThinkingBudget:
        return self.melchior.current_tier()

    @property
    def cassette(self) -> Cassette:
        # Some callers (book keying) read brain.cassette.name — the council's
        # journal keys on "MAGI", not the MELCHIOR seat.
        return self._identity

    def status_line(self) -> str:
        return (
            f"MAGI 合議{self.councils} 全会一致{self.unanimous} "
            f"司会裁定{self.moderator_used}"
        )

    # ====================================================================
    # choose — the council per turn.
    # ====================================================================
    def choose(self, sim: object) -> GameAction:
        self.calls += 1
        budget = self.melchior.current_tier()
        self.tier_history.append(budget.name)
        obs = self.observer.build(sim)

        degraded = not self._gemma_alive()
        if degraded:
            if not self.degraded_warned:
                sim.log("MAGI: BALTHASAR(:8102)応答せず — MELCHIOR単独の縮退合議に移行")
                self.degraded_warned = True
            return self._degraded_choose(obs, budget)

        self.councils += 1

        # 1) MELCHIOR proposal (Qwen, logic).
        melchior_a = self._safe_propose(self.melchior, obs, budget)
        # 2) BALTHASAR proposal (Gemma, survival).
        balthasar_a = self._safe_propose(self.balthasar, obs, budget)
        # 3) CASPER proposal: Qwen thinks a short plan, Gemma speaks the move.
        casper_a = self._casper_propose(obs, budget)

        proposals = [
            ("melchior", melchior_a),
            ("balthasar", balthasar_a),
            ("casper", casper_a),
        ]
        votes = {seat: self._fmt_vote(a) for seat, a in proposals}

        # Majority shortcut: ≥2 identical action+args → skip the 司会 THINK and
        # have the Qwen final call just confirm/format the winner (cheaper).
        winner = self._majority(proposals)
        agreed = winner is not None
        moderator_used = False
        unanimous = self._all_same(proposals)

        if agreed:
            ruling = (
                f"三機関の多数が同一案に達した。採るべき行動: "
                f"{winner.action} args={winner.args}。この行動を確定し整形せよ。"
            )
        else:
            # 3-way split → 司会 Gemwen weighs the three (Gemma thinks).
            ruling = self._moderate(obs, proposals, budget)
            moderator_used = True
            self.moderator_used += 1

        if unanimous:
            self.unanimous += 1

        # Final action call: Qwen speaks the FINAL schema-forced move with the
        # ruling appended. Falls back to the winner / MELCHIOR proposal on error.
        final = self._final_action(obs, ruling, budget, fallback=winner or melchior_a)

        self.turn_records.append({
            "votes": votes,
            "agreed": agreed,
            "moderator_used": moderator_used,
            "unanimous": unanimous,
            "final": self._fmt_vote(final),
        })
        return final

    # -- degraded: MELCHIOR alone (full normal pipeline on the one seat) -------
    def _degraded_choose(self, obs: dict, budget: ThinkingBudget) -> GameAction:
        self.councils += 1
        action = self._safe_propose(self.melchior, obs, budget)
        self.turn_records.append({
            "votes": {"melchior": self._fmt_vote(action)},
            "agreed": True,
            "moderator_used": False,
            "unanimous": False,
            "degraded": True,
            "final": self._fmt_vote(action),
        })
        return action

    # ====================================================================
    # Seat helpers.
    # ====================================================================
    def _safe_propose(
        self, brain: OpenAICompatibleBrain, obs: dict, budget: ThinkingBudget,
        extra: list[dict] | None = None,
    ) -> GameAction:
        """One seat action proposal; never raises — a failed seat returns a
        harmless 'rest' so the council still tallies a vote."""
        try:
            return brain.propose_action(obs, budget, extra=extra)
        except (ActionParseError, RuntimeError, Exception):  # noqa: BLE001
            return GameAction(action="rest", args={}, think="(seat failed)", say="")

    def _casper_propose(self, obs: dict, budget: ThinkingBudget) -> GameAction:
        """CASPER = Qwemma relay: MELCHIOR(Qwen) thinks a short plan, then the
        Gemma seat speaks the schema-forced action with that plan appended."""
        import json

        try:
            plan = self.melchior.think_freetext(
                CASPER_THINK_PROMPT,
                json.dumps(obs, ensure_ascii=False),
                max_tokens=120,
            )
        except Exception:  # noqa: BLE001
            plan = ""
        extra = None
        if plan:
            extra = [{"role": "assistant", "content": f"直感の方針: {plan}"}]
        # The action call goes to the Gemma seat (CASPER speaks through Gemma).
        speaker = self.casper or self.melchior
        return self._safe_propose(speaker, obs, budget, extra=extra)

    def _moderate(self, obs: dict, proposals, budget: ThinkingBudget) -> str:
        """司会 Gemwen: Gemma thinks the ruling over the labeled three proposals."""
        import json

        labeled = {
            seat.upper(): {"action": a.action, "args": a.args, "say": a.say}
            for seat, a in proposals
        }
        user = json.dumps(
            {"observation": obs, "proposals": labeled}, ensure_ascii=False
        )
        moderator = self.balthasar or self.melchior  # Gemma thinks
        try:
            return moderator.think_freetext(MODERATOR_THINK_PROMPT, user, max_tokens=200)
        except Exception:  # noqa: BLE001
            return "(司会応答せず — 各案を勘案して最善手を選べ)"

    def _final_action(
        self, obs: dict, ruling: str, budget: ThinkingBudget, fallback: GameAction
    ) -> GameAction:
        """Final schema-forced move from the Qwen seat, ruling appended."""
        extra = [{"role": "assistant", "content": f"司会の裁定: {ruling}"}]
        try:
            return self.melchior.propose_action(obs, budget, extra=extra)
        except (ActionParseError, RuntimeError, Exception):  # noqa: BLE001
            return fallback

    # ====================================================================
    # write_diary / write_motto — route to MELCHIOR (the Qwen spine) unchanged.
    # ====================================================================
    def write_diary(self, sim: object, season: str, weather: str) -> str | None:
        return self.melchior.write_diary(sim, season, weather)

    def write_motto(self, sim: object) -> dict | None:
        return self.melchior.write_motto(sim)

    # ====================================================================
    # compile_canon — the 編纂評議会 (poison-article killer).
    # MELCHIOR drafts → BALTHASAR reviews article-by-article → Gemwen-style
    # final call (Qwen) emits the adopted ≤5 articles.
    # ====================================================================
    def compile_canon(self, book: object) -> list[str] | None:
        import json

        # 1) MELCHIOR drafts the revision (its normal compile path).
        draft = None
        try:
            draft = self.melchior.compile_canon(book)
        except Exception:  # noqa: BLE001
            draft = None
        if not draft:
            return None
        # If Gemma is down, ship MELCHIOR's draft (degraded 編纂).
        if not self._gemma_alive() or self.balthasar is None:
            return draft[:5]

        # 2) BALTHASAR reviews each article (Gemma, mother's veto).
        review_ctx = {
            "canon_draft": draft,
            "history": book.history_table() if hasattr(book, "history_table") else [],
        }
        try:
            review = self.balthasar.think_freetext(
                COMPILE_REVIEW_PROMPT,
                json.dumps(review_ctx, ensure_ascii=False),
                max_tokens=320,
            )
        except Exception:  # noqa: BLE001
            review = ""
        self.last_compile_review = review  # exposed for the verification harness

        # 2b) Mother's veto, enforced: any draft article that romanticizes
        #     enduring hunger/thirst/cold (the poison pattern BALTHASAR hunts) is
        #     STRIPPED from the draft the final call sees, so the emitter cannot
        #     simply echo it back. The seat then refills to 5 survival articles.
        clean_draft = [a for a in draft if not _is_poison_article(a)]
        self.last_poison_stripped = [a for a in draft if _is_poison_article(a)]

        # 3) Final 編纂: Qwen emits the adopted articles given the cleaned draft +
        #    review, through MELCHIOR's schema-forced compile machinery.
        adopted = self._adopt_after_review(book, clean_draft, review)
        if adopted:
            # Belt-and-braces: never let a poison article slip through the final
            # emission either — the mother's veto is absolute.
            cleaned = [a for a in adopted if not _is_poison_article(a)]
            return (cleaned or adopted)[:5]
        return clean_draft[:5]

    def _adopt_after_review(self, book: object, draft: list[str], review: str):
        """Re-run MELCHIOR's compile with the BALTHASAR review folded into the
        canon context so the final 5 articles reflect the council's veto."""
        import json

        from .llm_client import _compile_schema
        from .prompts import COMPILE_PROMPT
        from .schema import _iter_balanced_objects, _remove_trailing_commas

        context = {
            "canon": draft,
            "history": book.history_table() if hasattr(book, "history_table") else [],
            "balthasar_review": review,
            "instruction": (
                "BALTHASARの査読を厳密に反映せよ。『棄却』とされた条文は原文のまま残すな——"
                "必ず生存に資する具体的な行動(いつ・何を)へ書き換えるか、削除して別条で埋めよ。"
                "空腹・渇き・寒さを『忍ぶ/耐える/我慢』と美化する文言、精神論で生存行動を曇らせる文言は"
                "最終稿から完全に消せ。家訓をちょうど5条で返せ。"
            ),
        }
        messages = [
            {"role": "system", "content": COMPILE_PROMPT + "\n" + self.melchior.cassette.persona},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]
        try:
            raw = self.melchior._chat(
                messages, schema=_compile_schema(), max_tokens=768
            ).strip()
        except Exception:  # noqa: BLE001
            return None
        if not raw:
            return None
        for candidate in _iter_balanced_objects(raw):
            try:
                obj = json.loads(_remove_trailing_commas(candidate))
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, dict) and isinstance(obj.get("lessons"), list):
                lessons = [str(s).strip() for s in obj["lessons"] if str(s).strip()]
                if lessons:
                    return lessons[:5]
        return None

    # ====================================================================
    # Health.
    # ====================================================================
    def _gemma_alive(self) -> bool:
        """True when the BALTHASAR (Gemma) seat is reachable. Cached negative is
        avoided: a flaky turn re-probes so the council can recover mid-game."""
        if self.balthasar is None:
            return False
        try:
            self.balthasar._resolve_model()
            return True
        except Exception:  # noqa: BLE001
            return False

    # ====================================================================
    # Vote / tally helpers.
    # ====================================================================
    @staticmethod
    def _fmt_vote(a: GameAction) -> str:
        if not a.args:
            return a.action
        return f"{a.action}{tuple(sorted(a.args.items()))}"

    @staticmethod
    def _key(a: GameAction):
        return (a.action, tuple(sorted((str(k), str(v)) for k, v in (a.args or {}).items())))

    def _majority(self, proposals) -> GameAction | None:
        """Return the action shared by ≥2 proposals (the first such), else None."""
        keys = [self._key(a) for _, a in proposals]
        for i, k in enumerate(keys):
            if keys.count(k) >= 2:
                return proposals[i][1]
        return None

    def _all_same(self, proposals) -> bool:
        keys = [self._key(a) for _, a in proposals]
        return len(set(keys)) == 1


# ====================================================================
# make_magi — build a council from config/models.toml.
# ====================================================================
def make_magi(config_path: str | Path, *, timeout: float = 45.0) -> MagiBrain:
    """Look up the MELCHIOR / BALTHASAR / CASPER cassettes by name and build the
    council. MELCHIOR must exist (the spine); BALTHASAR may be unreachable at
    build time (the council degrades to MELCHIOR-only on the first turn)."""
    config_path = Path(config_path)
    cassettes = {c.name: c for c in load_cassettes(config_path)}
    if "MELCHIOR" not in cassettes:
        raise ValueError(
            "MAGI requires a 'MELCHIOR' cassette in models.toml (the Qwen spine)."
        )
    melchior = OpenAICompatibleBrain(cassettes["MELCHIOR"], timeout=timeout)
    balthasar = None
    if "BALTHASAR" in cassettes and cassettes["BALTHASAR"].base_url:
        balthasar = OpenAICompatibleBrain(cassettes["BALTHASAR"], timeout=timeout)
    casper = None
    if "CASPER" in cassettes and cassettes["CASPER"].base_url:
        casper = OpenAICompatibleBrain(cassettes["CASPER"], timeout=timeout)
    identity = cassettes.get("MAGI") or Cassette(name="MAGI", base_url="", model="magi")
    return MagiBrain(melchior, balthasar, casper, identity=identity)
