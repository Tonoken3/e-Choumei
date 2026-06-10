from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Recipe:
    key: str
    name: str
    kind: str
    requires: dict[str, int]
    station: str | None = None


class RecipeBook:
    def __init__(self, recipes: list[Recipe]) -> None:
        self._recipes = {recipe.key: recipe for recipe in recipes}

    @classmethod
    def load(cls, path: Path) -> "RecipeBook":
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        recipes = []
        for row in data.get("recipe", []):
            recipes.append(
                Recipe(
                    key=row["key"],
                    name=row.get("name", row["key"]),
                    kind=row.get("kind", "item"),
                    requires=dict(row.get("requires", {})),
                    station=row.get("station"),
                )
            )
        return cls(recipes)

    def __contains__(self, key: str) -> bool:
        return key in self._recipes

    def get(self, key: str) -> Recipe:
        return self._recipes[key]

    def all(self) -> list[Recipe]:
        return list(self._recipes.values())

