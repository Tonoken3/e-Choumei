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
    tier: str = ""  # 思考予算 tier the LLM brain ran at (empty for local)


def select_meigen(lines: list[str], n: int = 5) -> list[str]:
    """Pick the hero's most memorable lines (銘言ベスト, spec §2.5/§6.3).

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


def fallback_motto(sim: object) -> dict[str, object]:
    """Deterministic 座右の銘 for runs without an LLM brain, read from the
    journey's actual ending (the LLM hermit writes its own via write_motto)."""
    from spl.agent.chronicle import jp_chronicle

    hero = sim.hero
    tail = " ".join(sim.full_log[-12:])
    chronicle = jp_chronicle(sim)
    if sim.completed:
        return {"motto": "足るを知る者、冬を越ゆ。", "words": "島は何も約束せなんだが、裏切りもせなんだ。", "highlights": chronicle}
    if "Dehydration" in tail or hero.water <= 0:
        return {"motto": "水を恃む前に、水を掘れ。", "words": "渇きは、いつも計画より一日早い。", "highlights": chronicle}
    if "Starvation" in tail or hero.hunger <= 0:
        return {"motto": "飯は思想に先んず。", "words": "明日の備えを、今日の腹が追い越した。", "highlights": chronicle}
    if "Winter" in tail or "snow" in tail.lower():
        return {"motto": "冬は約束どおりに来る。壁は約束より早く。", "words": "白い静寂に、火がひとつ足りなかった。", "highlights": chronicle}
    return {"motto": "島は非情、されど公平なり。", "words": "敗れたのではない、学びが一年に間に合わなんだだけ。", "highlights": chronicle}


def render_leaderboard(results: list[ArenaResult]) -> str:
    ordered = sorted(results, reverse=True)
    header = (
        f"{'Cassette':20} {'Seed':>5} {'Days':>5} {'Score':>6} {'Conf':>5} "
        f"{'Tier':>6}  {'Result':24} 銘言"
    )
    lines = [header, "-" * 104]
    for row in ordered:
        lines.append(
            f"{row.cassette[:20]:20} {row.seed:5d} {row.survived:5d} {row.score:6d} "
            f"{row.confusions:5d} {(row.tier or '-'):>6}  {row.reason[:24]:24} {row.best_line}"
        )
    return "\n".join(lines)
