from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DiaryEntry:
    day: int
    text: str


class Memory:
    def __init__(self) -> None:
        self.diary: list[DiaryEntry] = []
        self.notes: dict[int, list[str]] = {}
        self.rolling_summary = "No older memories yet."

    def add_note(self, day: int, text: str) -> None:
        self.notes.setdefault(day, []).append(text.strip())

    def nightly_entry(self, day: int, season: str, weather: str, log_lines: list[str], hp: int) -> DiaryEntry:
        notes = self.notes.get(day, [])
        fragments = []
        if log_lines:
            fragments.append(log_lines[-1])
        if notes:
            fragments.append(notes[-1])
        if not fragments:
            fragments.append("The island was quiet, and I kept moving.")
        text = (
            f"Day {day} ({season}, {weather})\n"
            f"- {fragments[0]}\n"
            f"- HP {hp}; I will plan for tomorrow before the dark gets too loud.\n"
            f"- {self._small_feeling(day, hp)}"
        )
        entry = DiaryEntry(day=day, text=text)
        self.diary.append(entry)
        if len(self.diary) > 14:
            older = self.diary[:-7]
            self.rolling_summary = f"Older diary: survived through day {older[-1].day}; kept farming, gathering, and preparing."
        return entry

    def recent_context(self, days: int = 7) -> str:
        recent = self.diary[-days:]
        if not recent:
            return self.rolling_summary
        return self.rolling_summary + "\n" + "\n".join(entry.text for entry in recent)

    def _small_feeling(self, day: int, hp: int) -> str:
        if hp < 35:
            return "I am afraid, but fear still means I am alive."
        if day % 7 == 0:
            return "A week leaves marks on the hands and names in the soil."
        return "Tomorrow, I will make one more useful thing."

