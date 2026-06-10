from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DiaryEntry:
    day: int
    text: str


# Salient log fragments worth remembering, in rough order of "this is what the
# day was really about". Used to pick a diary highlight instead of always the
# last (usually "The hero sleeps.") line.
_HIGHLIGHT_KEYS = (
    "Harvested",
    "is ready to harvest",
    "Made ",
    "trade is done",
    "Stored",
    "Cooked",
    "Wild dogs",
    "Storm",
    "Caught",
    "withered",
    "Bellyache",
    "Planted",
)


class Memory:
    def __init__(self) -> None:
        self.diary: list[DiaryEntry] = []
        self.notes: dict[int, list[str]] = {}
        self.rolling_summary = "No older memories yet."

    def add_note(self, day: int, text: str) -> None:
        self.notes.setdefault(day, []).append(text.strip())

    def nightly_entry(
        self,
        day: int,
        season: str,
        weather: str,
        log_lines: list[str],
        hp: int,
        llm_line: str | None = None,
    ) -> DiaryEntry:
        header = f"Day {day} ({season}, {weather})"
        if llm_line and llm_line.strip():
            # The hero's own (LLM-authored) words; keep them, just stamp the date.
            text = header + "\n" + llm_line.strip()
        else:
            notes = self.notes.get(day, [])
            highlight = notes[-1] if notes else self._highlight(log_lines)
            text = (
                f"{header}\n"
                f"- {highlight}\n"
                f"- {self._middle_line(season, weather, hp, log_lines)}\n"
                f"- {self._small_feeling(day, season, weather, hp, log_lines)}"
            )
        entry = DiaryEntry(day=day, text=text)
        self.diary.append(entry)
        if len(self.diary) > 14:
            older = self.diary[:-7]
            self.rolling_summary = (
                f"Older diary: survived through day {older[-1].day}; "
                "kept farming, gathering, and preparing for the cold."
            )
        return entry

    def recent_context(self, days: int = 7) -> str:
        recent = self.diary[-days:]
        if not recent:
            return self.rolling_summary
        return self.rolling_summary + "\n" + "\n".join(entry.text for entry in recent)

    def _highlight(self, log_lines: list[str]) -> str:
        for key in _HIGHLIGHT_KEYS:
            for line in reversed(log_lines):
                if key in line:
                    return line
        for line in reversed(log_lines):
            if "begins:" not in line and "sleeps" not in line:
                return line
        return "The island was quiet, and I kept moving."

    def _middle_line(self, season: str, weather: str, hp: int, log_lines: list[str]) -> str:
        if weather in {"Storm", "Snow"}:
            return f"HP {hp}; the {weather.lower()} pressed on the walls, but I held."
        if season == "Winter":
            return f"HP {hp}; nothing grows now, so I lean on what I put away."
        if season == "Autumn":
            return f"HP {hp}; I keep one eye on the cold that is coming."
        return f"HP {hp}; I will plan for tomorrow before the dark gets too loud."

    def _small_feeling(
        self, day: int, season: str, weather: str, hp: int, log_lines: list[str]
    ) -> str:
        if hp < 35:
            return "I am afraid, but fear still means I am alive."
        day_text = " ".join(log_lines)
        if "Harvested" in day_text:
            return "The soil paid me back today. Small mercies count."
        if "Wild dogs" in day_text:
            return "The dogs took a little. I will build a better fence."
        if "trade is done" in day_text:
            return "A voice that was not the wind. I had almost forgotten them."
        if "Made " in day_text:
            return "One more thing my hands can do that they could not yesterday."
        if season == "Winter":
            return "The white quiet is heavy, but I keep my small fire."
        if season == "Autumn":
            return "Every full barrel is a promise I make to my winter self."
        if day % 7 == 0:
            return "A week leaves marks on the hands and names in the soil."
        return "Tomorrow, I will make one more useful thing."
