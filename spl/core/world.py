from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .hero import Position
from .rng import GameRng

SEASONS = ("spring", "summer", "autumn", "winter")
SEASON_NAMES = {
    "spring": "Spring",
    "summer": "Summer",
    "autumn": "Autumn",
    "winter": "Winter",
}
WEATHER_NAMES = {
    "sunny": "Sunny",
    "rain": "Rain",
    "storm": "Storm",
    "drought": "Drought",
    "snow": "Snow",
}
TILE_NAMES = {
    "grass": "Grass",
    "forest": "Forest",
    "rock": "Rock",
    "water": "Water",
    "beach": "Beach",
    "field": "Field",
    "home": "Home",
    "workshop": "Workshop",
}
TILE_CHARS = {
    "grass": ".",
    "forest": "F",
    "rock": "^",
    "water": "~",
    "beach": ",",
    "field": "#",
    "home": "H",
    "workshop": "W",
}


@dataclass
class Plot:
    crop: str
    days_left: int
    water_level: int = 0
    age: int = 0

    @property
    def ready(self) -> bool:
        return self.days_left <= 0


@dataclass
class World:
    width: int
    height: int
    season_length: int = 28
    day: int = 1
    weather: str = "sunny"
    tiles: list[list[str]] = field(default_factory=list)
    plots: dict[Position, Plot] = field(default_factory=dict)

    @classmethod
    def generate(cls, width: int, height: int, season_length: int, rng: GameRng) -> "World":
        world = cls(width=width, height=height, season_length=season_length)
        tiles: list[list[str]] = []
        center = Position(width // 2, height // 2)
        for y in range(height):
            row = []
            for x in range(width):
                edge = x in (0, width - 1) or y in (0, height - 1)
                pos = Position(x, y)
                if edge:
                    row.append("water" if rng.chance(0.55) else "beach")
                elif abs(x - center.x) <= 1 and abs(y - center.y) <= 1:
                    row.append("grass")
                else:
                    roll = rng.random()
                    if roll < 0.30:
                        row.append("forest")
                    elif roll < 0.45:
                        row.append("rock")
                    elif roll < 0.54 and (x < 4 or y < 4 or x > width - 5 or y > height - 5):
                        row.append("water")
                    elif roll < 0.60:
                        row.append("beach")
                    else:
                        row.append("grass")
                if pos == center:
                    row[-1] = "home"
                elif pos == Position(center.x + 1, center.y):
                    row[-1] = "workshop"
            tiles.append(row)
        world.tiles = tiles
        world.weather = world.next_weather(rng, current=None)
        return world

    @property
    def season(self) -> str:
        index = ((self.day - 1) // self.season_length) % len(SEASONS)
        return SEASONS[index]

    @property
    def day_in_season(self) -> int:
        return ((self.day - 1) % self.season_length) + 1

    @property
    def start_pos(self) -> Position:
        return Position(self.width // 2, self.height // 2)

    def in_bounds(self, pos: Position) -> bool:
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height

    def tile_at(self, pos: Position) -> str:
        if not self.in_bounds(pos):
            return "water"
        return self.tiles[pos.y][pos.x]

    def set_tile(self, pos: Position, tile: str) -> None:
        if self.in_bounds(pos):
            self.tiles[pos.y][pos.x] = tile

    def is_passable(self, pos: Position) -> bool:
        return self.in_bounds(pos)

    def neighbors(self, pos: Position) -> Iterable[tuple[str, Position]]:
        directions = {
            "north": (0, -1),
            "south": (0, 1),
            "west": (-1, 0),
            "east": (1, 0),
        }
        for name, (dx, dy) in directions.items():
            new_pos = pos.step(dx, dy)
            if self.is_passable(new_pos):
                yield name, new_pos

    def is_near(self, pos: Position, tile: str) -> bool:
        return self.tile_at(pos) == tile or any(self.tile_at(npos) == tile for _, npos in self.neighbors(pos))

    def positions(self) -> Iterable[Position]:
        for y in range(self.height):
            for x in range(self.width):
                yield Position(x, y)

    def find_nearest(self, start: Position, predicate: Callable[[Position], bool]) -> Position | None:
        seen = {start}
        queue = deque([start])
        while queue:
            pos = queue.popleft()
            if predicate(pos):
                return pos
            for _, npos in self.neighbors(pos):
                if npos not in seen:
                    seen.add(npos)
                    queue.append(npos)
        return None

    def target_position(self, start: Position, target: str) -> Position | None:
        aliases: dict[str, Callable[[Position], bool]] = {
            "home": lambda pos: self.tile_at(pos) == "home",
            "workshop": lambda pos: self.tile_at(pos) == "workshop",
            "water": lambda pos: self.tile_at(pos) == "water",
            "forest": lambda pos: self.tile_at(pos) == "forest",
            "rock": lambda pos: self.tile_at(pos) == "rock",
            "grass": lambda pos: self.tile_at(pos) == "grass",
            "field": lambda pos: self.tile_at(pos) == "field",
            "empty_field": lambda pos: self.tile_at(pos) == "field" and pos not in self.plots,
            "ready_field": lambda pos: pos in self.plots and self.plots[pos].ready,
            "dry_field": lambda pos: pos in self.plots and not self.plots[pos].ready,
            "crop_field": lambda pos: pos in self.plots,
        }
        predicate = aliases.get(target)
        if predicate is None:
            return None
        return self.find_nearest(start, predicate)

    def step_toward(self, start: Position, goal: Position) -> Position:
        if start == goal:
            return start
        dx = goal.x - start.x
        dy = goal.y - start.y
        candidates: list[Position] = []
        if abs(dx) >= abs(dy) and dx:
            candidates.append(start.step(1 if dx > 0 else -1, 0))
        if dy:
            candidates.append(start.step(0, 1 if dy > 0 else -1))
        if dx and (not candidates or candidates[0].x == start.x):
            candidates.append(start.step(1 if dx > 0 else -1, 0))
        for candidate in candidates:
            if self.is_passable(candidate):
                return candidate
        return start

    def next_weather(self, rng: GameRng, current: str | None = None) -> str:
        season = self.season
        if season == "spring":
            weights = [("sunny", 0.55), ("rain", 0.34), ("storm", 0.11)]
        elif season == "summer":
            weights = [("sunny", 0.58), ("rain", 0.18), ("storm", 0.09), ("drought", 0.15)]
        elif season == "autumn":
            weights = [("sunny", 0.48), ("rain", 0.34), ("storm", 0.18)]
        else:
            weights = [("sunny", 0.44), ("snow", 0.38), ("storm", 0.18)]
        if current in {"rain", "storm", "drought", "snow"}:
            weights = [(name, weight * (1.35 if name == current else 0.92)) for name, weight in weights]
        return rng.weighted_choice(weights)

    def grow_crops(self, crop_book: object) -> list[str]:
        messages: list[str] = []
        season = self.season
        required_water = 2 if self.weather == "drought" else 1
        for pos, plot in list(self.plots.items()):
            crop = crop_book.get(plot.crop)
            if season == "winter":
                plot.water_level = 0
                continue
            if season not in crop.seasons:
                if plot.age > 1:
                    messages.append(f"{crop.name} withered out of season.")
                    self.plots.pop(pos, None)
                else:
                    plot.age += 1
                continue
            watered = self.weather == "rain" or not crop.needs_water or plot.water_level >= required_water
            if watered and plot.days_left > 0:
                plot.days_left -= 1
                plot.age += 1
                if plot.days_left == 0:
                    messages.append(f"{crop.name} is ready to harvest.")
            plot.water_level = 0
        return messages

    def render_map(self, hero_pos: Position, radius: int | None = None) -> str:
        rows = []
        y_range = range(self.height)
        x_range = range(self.width)
        if radius is not None:
            y_range = range(max(0, hero_pos.y - radius), min(self.height, hero_pos.y + radius + 1))
            x_range = range(max(0, hero_pos.x - radius), min(self.width, hero_pos.x + radius + 1))
        for y in y_range:
            chars = []
            for x in x_range:
                pos = Position(x, y)
                if pos == hero_pos:
                    chars.append("@")
                elif pos in self.plots:
                    chars.append("*" if self.plots[pos].ready else "#")
                else:
                    chars.append(TILE_CHARS.get(self.tile_at(pos), "?"))
            rows.append("".join(chars))
        return "\n".join(rows)

