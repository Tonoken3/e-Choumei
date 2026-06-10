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

