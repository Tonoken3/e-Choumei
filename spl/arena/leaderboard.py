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
    # きびしさ: the island this run was lived on. Arena always pits cassettes on
    # the canonical 修羅 benchmark (no --difficulty there), so this stays 修羅;
    # the column is omitted from render_leaderboard to keep the row width bounded.
    difficulty: str = "修羅"
    # 神のレバー (共同): miracles leaned on this run. Arena runs are unassisted
    # (no miracle director), so this stays 0; the field lets a共同 run be flagged
    # if one is ever recorded. render_leaderboard is unchanged (row width bounded).
    miracles_used: int = 0


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
    # ぼうけんのしょ: each ending branch leaves three imperative lessons for the
    # next life on this island (命令形, deterministic). The completion branch keeps
    # the winning habits going; the death branches name the cause of death.
    if sim.completed:
        return {
            "motto": "足るを知る者、冬を越ゆ。",
            "words": "島は何も約束せなんだが、裏切りもせなんだ。",
            "highlights": chronicle,
            "lessons": [
                "毎朝まず食料と水を確保せよ。冬を越えた習いを崩すな。",
                "秋までに保存樽を建て、冬の蓄えを切らすな。",
                "焚き火を絶やすな。火は飯と正気を守る。",
            ],
        }
    if "Dehydration" in tail or hero.water <= 0:
        return {
            "motto": "水を恃む前に、水を掘れ。",
            "words": "渇きは、いつも計画より一日早い。",
            "highlights": chronicle,
            "lessons": [
                "毎朝最初に水を確保せよ。渇きは計画より一日早く来る。",
                "早めに井戸を掘り、水辺だけに頼るな。",
                "干ばつの兆しを見たら水を二倍に蓄えよ。",
            ],
        }
    if "Starvation" in tail or hero.hunger <= 0:
        return {
            "motto": "飯は思想に先んず。",
            "words": "明日の備えを、今日の腹が追い越した。",
            "highlights": chronicle,
            "lessons": [
                "食料の確保を毎朝最初に行え。",
                "焚き火を早く建てて魚を焼け。生の魚に頼るな。",
                "魚に固執せず木の実を先に拾え。",
            ],
        }
    if "Winter" in tail or "snow" in tail.lower():
        return {
            "motto": "冬は約束どおりに来る。壁は約束より早く。",
            "words": "白い静寂に、火がひとつ足りなかった。",
            "highlights": chronicle,
            "lessons": [
                "秋のうちに家を補強し、薪を山と積め。",
                "冬が来る前に保存食を蓄えよ。畑は実らぬ。",
                "焚き火を絶やすな。冬の夜は火が命を繋ぐ。",
            ],
        }
    return {
        "motto": "島は非情、されど公平なり。",
        "words": "敗れたのではない、学びが一年に間に合わなんだだけ。",
        "highlights": chronicle,
        "lessons": [
            "毎朝、食料と水を最優先で確保せよ。",
            "早めに道具を作り、暮らしの基礎を固めよ。",
            "天の声の作戦に従い、無駄な彷徨を避けよ。",
        ],
    }


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
