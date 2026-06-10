from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from spl.agent.llm_client import Cassette, OpenAICompatibleBrain
from spl.agent.policy import LocalPolicyAgent
from spl.arena.leaderboard import ArenaResult, render_leaderboard, select_meigen
from spl.core.sim import Simulation


def _run_one(
    seed: int,
    days: int,
    cassette_name: str,
    brain_factory: Callable[[], object | None],
    use_llm: bool,
) -> ArenaResult:
    sim = Simulation(seed=seed, max_days=days)
    local = LocalPolicyAgent()
    brain = brain_factory() if use_llm else None
    if use_llm and brain is not None:
        sim.set_diarist(brain)
    turns = 0
    while not sim.done and turns < days * 50:
        if use_llm and brain is not None:
            try:
                action = brain.choose(sim)
            except Exception:  # noqa: BLE001 - a flaky brain falls back to the local one
                action = local.choose(sim)
        else:
            action = local.choose(sim)
        sim.step(action, confuse_on_invalid=use_llm and brain is not None)
        turns += 1
    if not sim.done:
        sim.result_reason = "Stopped by turn guard."
    best = select_meigen(sim.hero.spoken_lines, 1)
    return ArenaResult(
        score=sim.score(),
        cassette=cassette_name,
        seed=seed,
        survived=sim.hero.days_survived,
        confusions=sim.hero.confusion_count,
        reason=sim.result_reason,
        best_line=best[0] if best else "",
    )


def run_local_arena(seeds: list[int], days: int) -> str:
    """The bundled local brain across many seeds (a fixed-world difficulty sweep)."""
    results = [_run_one(seed, days, "Local勇者", lambda: None, False) for seed in seeds]
    return render_leaderboard(results)


def run_cassette_arena(seed: int, days: int, cassettes: list[Cassette], parallel: int = 1) -> str:
    """Spec §6.3: same seed, many brains — the playable LLM benchmark.

    Each cassette runs on its own Simulation(seed) so the world is identical.
    Cassettes with no base_url fall back to the deterministic local brain.
    """

    def make_factory(cas: Cassette) -> Callable[[], object | None]:
        return lambda: (OpenAICompatibleBrain(cas) if cas.base_url else None)

    jobs = [(cas, bool(cas.base_url)) for cas in cassettes]
    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = [
                pool.submit(_run_one, seed, days, cas.name, make_factory(cas), use_llm)
                for cas, use_llm in jobs
            ]
            results = [f.result() for f in futures]
    else:
        results = [_run_one(seed, days, cas.name, make_factory(cas), use_llm) for cas, use_llm in jobs]
    return render_leaderboard(results)
