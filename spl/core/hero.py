from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, order=True)
class Position:
    x: int
    y: int

    def step(self, dx: int, dy: int) -> "Position":
        return Position(self.x + dx, self.y + dy)


@dataclass
class Hero:
    pos: Position
    hp: int = 100
    hunger: int = 78
    water: int = 82
    stamina: int = 92
    sanity: int = 85
    ap_left: int = 12
    inventory: dict[str, int] = field(default_factory=dict)
    confusion_count: int = 0
    days_survived: int = 0
    total_actions: int = 0
    spoken_lines: list[str] = field(default_factory=list)

    @property
    def alive(self) -> bool:
        return self.hp > 0

    def item_count(self, item: str) -> int:
        return self.inventory.get(item, 0)

    def has(self, item: str, amount: int = 1) -> bool:
        return self.item_count(item) >= amount

    def add_item(self, item: str, amount: int = 1) -> None:
        if amount <= 0:
            return
        self.inventory[item] = self.item_count(item) + amount

    def remove_item(self, item: str, amount: int = 1) -> bool:
        if amount <= 0:
            return True
        if not self.has(item, amount):
            return False
        new_value = self.item_count(item) - amount
        if new_value:
            self.inventory[item] = new_value
        else:
            self.inventory.pop(item, None)
        return True

    def adjust(self, stat: str, delta: int) -> None:
        value = getattr(self, stat)
        if stat == "hp":
            setattr(self, stat, max(0, min(100, value + delta)))
        else:
            setattr(self, stat, max(0, min(100, value + delta)))

    def inventory_summary(self, limit: int = 14) -> dict[str, int]:
        items = [(key, value) for key, value in self.inventory.items() if value > 0]
        items.sort(key=lambda pair: (-pair[1], pair[0]))
        return dict(items[:limit])

    def civilization_points(self) -> int:
        points = 0
        weights = {
            "stone_axe": 8,
            "hoe": 6,
            "fishing_rod": 6,
            "campfire": 8,
            "stove": 16,
            "well": 24,
            "storage_barrel": 18,
            "house_upgrade": 30,
            "fence": 12,
            "preserved_pumpkin": 4,
            "bread": 3,
            "stew": 5,
        }
        for item, weight in weights.items():
            points += self.item_count(item) * weight
        points += sum(self.item_count(item) for item in ("wood", "stone", "clay", "iron_ore"))
        return points

