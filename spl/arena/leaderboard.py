from __future__ import annotations

from dataclasses import dataclass


@dataclass(order=True)
class ArenaResult:
    score: int
    cassette: str
    seed: int
    survived: int
    confusions: int
    reason: str
    best_line: str = ""


def select_meigen(lines: list[str], n: int = 5) -> list[str]:
    """Pick the hero's most memorable lines (迷言ベスト, spec §2.5/§6.3).

    Deduplicate (keeping first appearance), then favour the longer, more
    distinctive lines — they read better than the short repeated filler. Pure
    function of the input, so it stays deterministic.
    """
    seen: list[str] = []
    for line in lines:
        text = (line or "").strip()
        if text and text not in seen:
            seen.append(text)
    seen.sort(key=len, reverse=True)
    return seen[:n]


def render_leaderboard(results: list[ArenaResult]) -> str:
    ordered = sorted(results, reverse=True)
    header = f"{'Cassette':20} {'Seed':>5} {'Days':>5} {'Score':>6} {'Conf':>5}  {'Result':24} 迷言"
    lines = [header, "-" * 96]
    for row in ordered:
        lines.append(
            f"{row.cassette[:20]:20} {row.seed:5d} {row.survived:5d} {row.score:6d} "
            f"{row.confusions:5d}  {row.reason[:24]:24} {row.best_line}"
        )
    return "\n".join(lines)
