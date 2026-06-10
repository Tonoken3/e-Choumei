from __future__ import annotations

from spl.agent.policy import LocalPolicyAgent
from spl.arena.leaderboard import ArenaResult, render_leaderboard
from spl.core.sim import Simulation


def run_local_arena(seeds: list[int], days: int) -> str:
    results: list[ArenaResult] = []
    for seed in seeds:
        sim = Simulation(seed=seed, max_days=days)
        agent = LocalPolicyAgent()
        turns = 0
        while not sim.done and turns < days * 50:
            sim.step(agent.choose(sim))
            turns += 1
        if not sim.done:
            sim.result_reason = "Stopped by turn guard."
        results.append(
            ArenaResult(
                score=sim.score(),
                cassette="Local勇者",
                seed=seed,
                survived=sim.hero.days_survived,
                confusions=sim.hero.confusion_count,
                reason=sim.result_reason,
            )
        )
    return render_leaderboard(results)
