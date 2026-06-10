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


def render_leaderboard(results: list[ArenaResult]) -> str:
    ordered = sorted(results, reverse=True)
    lines = ["Cassette              Seed  Days  Score  Conf  Result", "-" * 62]
    for row in ordered:
        lines.append(
            f"{row.cassette[:20]:20} {row.seed:5d} {row.survived:5d} {row.score:6d} {row.confusions:5d}  {row.reason}"
        )
    return "\n".join(lines)

