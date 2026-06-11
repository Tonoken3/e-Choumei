from __future__ import annotations

import re

"""The heaven's-voice chronicle: deterministic milestones mined from the run log.

This is the 天の声目線 "logger" — what the watcher saw, day by day. It feeds the
result screens directly (no LLM needed) and gives the LLM epilogue call real
material to look back on.
"""

_JP_ITEM = {
    "Stone Axe": "石の斧",
    "Hoe": "くわ",
    "Fishing Rod": "釣り竿",
    "Campfire": "焚き火",
    "Stove": "かまど",
    "Well": "井戸",
    "Storage Barrel": "保存樽",
    "House Upgrade": "家の改築",
    "Fence": "柵",
    "Turnip": "カブ",
    "Wheat": "小麦",
    "Tomato": "トマト",
    "Pumpkin": "カボチャ",
}

_LINE = re.compile(r"^D(\d{3}) AP\d{2}: (.*)$")


def extract_milestones(sim: object) -> list[dict[str, object]]:
    """Scan the full log once and keep the moments a watcher would remember."""
    made_seen: set[str] = set()
    first_harvest = True
    winter_logged = False
    milestones: list[dict[str, object]] = []

    def add(day: int, kind: str, text: str) -> None:
        milestones.append({"day": day, "kind": kind, "text": text})

    for raw in sim.full_log:
        m = _LINE.match(raw)
        if not m:
            continue
        day = int(m.group(1))
        msg = m.group(2)
        made = re.match(r"Made (.+)\.$", msg)
        if made and made.group(1) not in made_seen:
            made_seen.add(made.group(1))
            name = _JP_ITEM.get(made.group(1), made.group(1))
            add(day, "made", f"{day}日目、{name}を作り上げた。")
            continue
        harvest = re.match(r"Harvested (\d+) (.+)\.$", msg)
        if harvest and first_harvest:
            first_harvest = False
            name = _JP_ITEM.get(harvest.group(2), harvest.group(2))
            add(day, "harvest", f"{day}日目、初めての収穫——{name}が{harvest.group(1)}つ。")
            continue
        if msg.startswith("Wild dogs raided"):
            add(day, "dogs", f"{day}日目、野犬に蓄えを荒らされる。")
        elif "Bellyache ends the day" in msg:
            cause = "きのこ" if "mushroom" in msg.lower() else "生の魚"
            add(day, "bellyache", f"{day}日目、{cause}にあたって一日伏せる。")
        elif msg.startswith("Storm damage"):
            add(day, "storm", f"{day}日目、嵐が庵を揺らす。")
        elif msg.startswith("The merchant trade is done"):
            add(day, "trade", f"{day}日目、行商人との取引が成る。")
        elif msg.startswith("Starvation bites"):
            add(day, "hunger", f"{day}日目、飢えが骨を噛む。")
        elif msg.startswith("Dehydration bites"):
            add(day, "thirst", f"{day}日目、渇きが牙を剥く。")
        elif "begins: Winter" in msg and not winter_logged:
            winter_logged = True
            add(day, "winter", f"{day}日目、冬が島に降りる。")

    # The same crisis repeating daily reads as noise; keep each kind's first
    # occurrence per 7-day week so long famines compress to their beats.
    kept: list[dict[str, object]] = []
    seen_weeks: set[tuple[str, int]] = set()
    for ms in milestones:
        key = (str(ms["kind"]), int(ms["day"]) // 7)
        if ms["kind"] in {"hunger", "thirst", "dogs", "storm"} and key in seen_weeks:
            continue
        seen_weeks.add(key)
        kept.append(ms)
    return kept[-24:]


def jp_chronicle(sim: object, limit: int = 6) -> list[str]:
    """The watcher's chronicle as display lines (deterministic fallback)."""
    milestones = extract_milestones(sim)
    if len(milestones) <= limit:
        lines = [str(ms["text"]) for ms in milestones]
    else:
        # Keep the beginning, the crises, and the end of the story.
        head = milestones[:2]
        tail = milestones[-(limit - 2):]
        lines = [str(ms["text"]) for ms in head + tail]
    if not lines:
        lines = ["静かな日々だった。島と仙人のほかに、語る者はない。"]
    return lines
