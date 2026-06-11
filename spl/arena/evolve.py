from __future__ import annotations

"""家訓の編纂 — the evolve harness.

Run N lives sequentially on one seed, REVISING the lineage's fixed 5-article
家訓 (canon) after each life rather than growing the journal. Each life inherits
the canon via ``lessons_for`` (canon wins); after it ends its record is appended
and the canon is re-compiled (LLM 編纂者 if a brain is available, else the
deterministic fallback). The lifespan beside each life's lessons is the selection
pressure: articles carried by long lives earn their place, articles carried by
short lives get rewritten.

The brain is reused across lives (one warm serving connection); the Simulation is
fresh each life. An exception in one life logs ERROR and continues to the next —
the lineage must outlive any single death.
"""

import sys
import traceback

from spl.agent.bouken import (
    BoukenNoSho,
    book_path_for,
    build_entry,
    fallback_compile,
    inject_into_observer,
)
from spl.agent.llm_client import OpenAICompatibleBrain
from spl.agent.policy import LocalPolicyAgent
from spl.arena.leaderboard import fallback_motto
from spl.core.sim import Simulation


def _say(line: str) -> None:
    """Print and flush immediately — the architect monitors via tail -f."""
    print(line, flush=True)


def _compile(book: BoukenNoSho, brain: object | None) -> list[str]:
    """Revise the canon at revision+1 (LLM 編纂者 or deterministic fallback)."""
    articles: list[str] | None = None
    if brain is not None and hasattr(brain, "compile_canon"):
        try:
            articles = brain.compile_canon(book)
        except Exception:  # noqa: BLE001 - the 編纂 must never sink a life
            articles = None
    if not articles:
        articles = fallback_compile(book)
    return book.set_canon(articles, book.canon_revision + 1)


def _final_motto(sim: Simulation, brain: object | None) -> dict:
    motto = None
    if brain is not None and sim.done:
        try:
            motto = brain.write_motto(sim)
        except Exception:  # noqa: BLE001 - the ending must never crash
            motto = None
    return motto or fallback_motto(sim)


def _run_one_life(
    sim: Simulation,
    brain: object | None,
    local: LocalPolicyAgent,
    use_llm: bool,
    max_turns: int,
) -> None:
    """Drive a single life to a terminal state, reusing the simulate machinery."""
    if use_llm and brain is not None:
        sim.set_diarist(brain)
    turns = 0
    while not sim.done and turns < max_turns:
        if use_llm and brain is not None:
            try:
                action = brain.choose(sim)
            except Exception:  # noqa: BLE001 - a flaky brain falls back this turn
                action = local.choose(sim)
        else:
            action = local.choose(sim)
        sim.step(action, confuse_on_invalid=use_llm)
        turns += 1
    if turns >= max_turns and not sim.done:
        sim.result_reason = "Stopped by max turn guard."


def run_evolve(
    lives: int,
    seed: int,
    days: int,
    cassette: object | None,
    use_llm: bool,
    book_dir_cassette: str | None = None,
) -> int:
    """Run ``lives`` lives, revising the 家訓 after each. Returns 0 always (the
    lineage is resilient: a dead life is logged and the next is born)."""
    # The book is keyed by cassette name (or a stable local key); the SPL_BOOK_DIR
    # env override decides the directory.
    key = book_dir_cassette or (getattr(cassette, "name", None) if cassette else None) or "Evolve仙人"
    book = BoukenNoSho.load(book_path_for(key))

    brain: object | None = None
    if use_llm and cassette is not None and getattr(cassette, "base_url", ""):
        brain = OpenAICompatibleBrain(cassette)

    local = LocalPolicyAgent()
    max_turns = days * 50
    days_list: list[int] = []

    _say(f"=== 家訓の編纂: {lives} lives on seed {seed} ({days}日) "
         f"brain={'LLM ' + getattr(cassette, 'name', '?') if brain else 'local'} ===")
    if book.canon:
        _say(f"開始時の家訓 (rev {book.canon_revision}):")
        for i, art in enumerate(book.canon, 1):
            _say(f"  第{i}条 {art}")

    for n in range(1, lives + 1):
        try:
            sim = Simulation(seed=seed, max_days=days)
            # This life inherits the current canon (canon wins in lessons_for).
            if brain is not None:
                inject_into_observer(getattr(brain, "observer", None), book, seed)

            _run_one_life(sim, brain, local, use_llm and brain is not None, max_turns)

            motto = _final_motto(sim, brain)
            entry = book.append(build_entry(sim, seed, motto))
            articles = _compile(book, brain)

            d = int(entry.get("days") or 0)
            days_list.append(d)
            _say(
                f"life {n}: {d}日 score {sim.score()} "
                f"ending={sim.result_reason or '—'} 介入なし canon_rev={book.canon_revision}"
            )
            for i, art in enumerate(articles, 1):
                _say(f"  第{i}条 {art}")
        except Exception:  # noqa: BLE001 - one death must not end the lineage
            _say(f"ERROR life {n}: {traceback.format_exc().strip().splitlines()[-1]}")
            traceback.print_exc(file=sys.stderr)
            continue

    _say("=== 編纂おわり ===")
    if days_list:
        mean = sum(days_list) / len(days_list)
        _say(f"days: {days_list}")
        _say(f"mean {mean:.1f}日 / max {max(days_list)}日 / lives {len(days_list)}")
    _say(f"最終家訓 (rev {book.canon_revision}):")
    for i, art in enumerate(book.canon, 1):
        _say(f"  第{i}条 {art}")
    return 0
