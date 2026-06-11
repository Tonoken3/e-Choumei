from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DiaryEntry:
    day: int
    text: str


# Salient log fragments worth remembering, in rough order of "this is what the
# day was really about". Used to pick a diary highlight instead of always the
# last (usually "The hero sleeps.") line.
_JP_SEASON = {"Spring": "春", "Summer": "夏", "Autumn": "秋", "Winter": "冬"}
_JP_WEATHER = {"Sunny": "晴れ", "Rain": "雨", "Storm": "嵐", "Drought": "旱", "Snow": "雪"}

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
        self.rolling_summary = "まだ古い記憶はない。"

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
        header = f"{day}日目（{_JP_SEASON.get(season, season)}・{_JP_WEATHER.get(weather, weather)}）"
        if llm_line and llm_line.strip():
            # The hermit's own (LLM-authored) words; keep them, just stamp the date.
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
                f"古い日記の要約: {older[-1].day}日目までを生き延びた。"
                "耕し、拾い、来たる寒さに備え続けている。"
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
        return "島は静かで、我はただ動いていた。"

    def _middle_line(self, season: str, weather: str, hp: int, log_lines: list[str]) -> str:
        if weather == "Storm":
            return f"体の力は{hp}。嵐が庵の壁を叩いたが、屈しはせぬ。"
        if weather == "Snow":
            return f"体の力は{hp}。雪が音を消し、世界は白い紙のようだ。"
        if season == "Winter":
            return f"体の力は{hp}。育つもののなき季節、蓄えに寄り掛かって生きる。"
        if season == "Autumn":
            return f"体の力は{hp}。来たる寒さを、片目で見据えている。"
        return f"体の力は{hp}。闇が濃くなる前に、明日の段取りを思う。"

    def _small_feeling(
        self, day: int, season: str, weather: str, hp: int, log_lines: list[str]
    ) -> str:
        if hp < 35:
            return "恐ろしい。だが恐れているうちは、まだ生きている。"
        day_text = " ".join(log_lines)
        if "Harvested" in day_text:
            return "土が応えてくれた。小さき恵みも、数えれば灯になる。"
        if "Wild dogs" in day_text:
            return "犬に少し持っていかれた。柵をもっと固く編もう。"
        if "trade is done" in day_text:
            return "風ではない声を聞いた。人というものを忘れかけていた。"
        if "Made " in day_text:
            return "昨日の手にできなかったことが、今日の手にはできる。"
        if season == "Winter":
            return "白い静寂は重い。されど我が火は、小さくとも消えぬ。"
        if season == "Autumn":
            return "樽が満ちるたび、冬の我にひとつ約束をする。"
        if day % 7 == 0:
            return "また七日が過ぎ、手に跡を、土に名を残した。"
        return "明日もまた、役に立つものをひとつ作ろう。"
