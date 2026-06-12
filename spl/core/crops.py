from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CropDef:
    key: str
    name: str
    seed: str
    grow_days: int
    seasons: tuple[str, ...]
    needs_water: bool
    food: int
    crop_yield: int


class CropBook:
    def __init__(self, crops: list[CropDef]) -> None:
        self._by_key = {crop.key: crop for crop in crops}
        self._by_seed = {crop.seed: crop for crop in crops}

    @classmethod
    def load(cls, path: Path) -> "CropBook":
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        crops = []
        for row in data.get("crop", []):
            crops.append(
                CropDef(
                    key=row["key"],
                    name=row.get("name", row["key"]),
                    seed=row["seed"],
                    grow_days=int(row["grow_days"]),
                    seasons=tuple(row["seasons"]),
                    needs_water=bool(row["needs_water"]),
                    food=int(row["food"]),
                    crop_yield=int(row.get("yield", 1)),
                )
            )
        return cls(crops)

    def __contains__(self, key: str) -> bool:
        return key in self._by_key

    def get(self, key: str) -> CropDef:
        return self._by_key[key]

    def from_seed(self, seed: str) -> CropDef | None:
        return self._by_seed.get(seed)

    def seasonal(self, season: str) -> list[CropDef]:
        return [crop for crop in self._by_key.values() if season in crop.seasons]

    def all(self) -> list[CropDef]:
        return list(self._by_key.values())


# Crop key -> JP name. The core stays English everywhere else, but the 石碑
# inscription and the 行商人's small-talk are diegetic Japanese the brain reads
# in its "recent" window, so the JP names live next to the agronomy that
# produces them (single source of truth, mirrored by the UI layer's CROP_JP).
CROP_JP: dict[str, str] = {
    "turnip": "カブ",
    "wheat": "小麦",
    "tomato": "トマト",
    "pumpkin": "カボチャ",
}

# The settlers' arithmetic: winter lands on this day every year (season_length
# 28 -> spring/summer/autumn fill days 1-84, winter begins day 85). The stone
# and the merchant both reckon planting deadlines against it. Mirrors the day
# the observer's 体の予感 already counts down to (85 - day).
WINTER_DAY = 85


def crop_jp(key: str) -> str:
    """JP name for a crop key, falling back to the key itself."""
    return CROP_JP.get(key, key)


def monument_inscription(crop_book: "CropBook") -> str:
    """The agronomy carved on the settlers' stone, built from the REAL loaded
    crop data: each main crop's true growth days, then winter's fixed date and
    the settlers' warning. Deterministic (no RNG); the numbers are whatever the
    crops.toml says, so the stone can never drift from the world's rules."""
    parts = [f"{crop_jp(c.key)}は{c.grow_days}日" for c in crop_book.all()]
    crops_text = "、".join(parts)
    return (
        "庵の傍らに古い石碑が立つ。刻まれた文字: "
        f"『{crops_text}…冬は{WINTER_DAY}日目に来る。実りより先に、種を数えよ。』"
    )


def monument_agronomy_line(crop_book: "CropBook") -> str:
    """A compact (~120 char) standing version of the stone's agronomy for the
    observation's permanent ``monument`` key — background knowledge the brain
    always carries, not an alarm. Same numbers as the inscription, terse."""
    parts = [f"{crop_jp(c.key)}{c.grow_days}日" for c in crop_book.all()]
    return f"石碑の知恵: {'／'.join(parts)}。冬は{WINTER_DAY}日目。実りより先に種を数えよ。"


def merchant_planting_hint(crop_book: "CropBook", day: int, season: str) -> str:
    """The 行商人's deterministic small-talk planting hint, reckoned from the
    CURRENT day + the real crop data (no RNG, so the determinism test is safe):

    * winter: there is nothing to plant — remind the hermit to keep seed for
      the 28-day spring;
    * else if some crop can still mature before winter (day + grow_days <=
      WINTER_DAY - 1), recommend the most-nourishing such crop;
    * else the planting window has closed — tell them to lay in stores and walls.
    """
    if season == "winter":
        return "春は二十八日続きます。種を残しておきなさい"
    deadline = WINTER_DAY - 1  # last day a planting can still finish by winter
    fits = [c for c in crop_book.all() if day + c.grow_days <= deadline]
    if fits:
        best = max(fits, key=lambda c: (c.food, -c.grow_days, c.key))
        return f"今から{crop_jp(best.key)}を植えれば、冬の前に実りますよ"
    return "もう種時は過ぎましたなぁ。蓄えと壁の支度をなさい"


FOOD_VALUES: dict[str, int] = {
    "berries": 10,
    "mushroom": 12,
    "fish": 12,
    "cooked_fish": 30,
    "turnip": 18,
    "wheat": 12,
    "bread": 45,
    "tomato": 20,
    "pumpkin": 35,
    "stew": 55,
    "dried_berries": 18,
    "preserved_pumpkin": 45,
}

