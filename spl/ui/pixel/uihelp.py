from __future__ import annotations

"""Mouse-first UI helpers for the Island Diorama.

Pure-data tables and tiny functions shared by the HUD button bar, the
click-to-act popup, tooltips and the guide strip. Nothing here mutates the
``Simulation`` — callers turn the returned menu/action descriptors into
``GameAction`` objects and hand them to ``sim.step(...)`` on the main thread.

All player-facing strings are plain JP (with an ASCII fallback chosen by the
``Fonts`` helper) — never emoji, since the pixel/CJK fonts may lack them.
"""

from dataclasses import dataclass, field
from typing import Callable

from spl.core.crops import FOOD_VALUES
from spl.core.hero import Position
from spl.core.world import TILE_NAMES

# --- Japanese names (UI layer only; core stays English) ---------------------
# Tile names.
TILE_JP = {
    "grass": "草地",
    "forest": "森",
    "rock": "岩場",
    "water": "水辺",
    "beach": "砂浜",
    "field": "畑",
    "home": "我が家",
    "workshop": "工房",
}

# Crop keys + their seeds -> JP crop name.
CROP_JP = {
    "turnip": "カブ",
    "wheat": "小麦",
    "tomato": "トマト",
    "pumpkin": "カボチャ",
}
SEED_JP = {
    "turnip_seed": "カブ",
    "wheat_seed": "小麦",
    "tomato_seed": "トマト",
    "pumpkin_seed": "カボチャ",
}

# Edible inventory items -> JP name (for the eat popup).
FOOD_JP = {
    "berries": "木の実",
    "mushroom": "きのこ",
    "fish": "魚",
    "cooked_fish": "焼き魚",
    "turnip": "カブ",
    "wheat": "小麦",
    "bread": "パン",
    "tomato": "トマト",
    "pumpkin": "カボチャ",
    "stew": "シチュー",
    "dried_berries": "干し木の実",
    "preserved_pumpkin": "保存カボチャ",
}

# Recipe keys -> JP name (for the craft overlay).
RECIPE_JP = {
    "stone_axe": "石の斧",
    "hoe": "くわ",
    "fishing_rod": "釣り竿",
    "campfire": "焚き火",
    "stove": "かまど",
    "well": "井戸",
    "storage_barrel": "保存樽",
    "house_upgrade": "家の改築",
    "fence": "柵",
}

# Materials -> JP (for craft material lists).
MAT_JP = {
    "wood": "木材",
    "stone": "石",
    "fiber": "繊維",
    "clay": "粘土",
    "iron_ore": "鉄鉱石",
}


def tile_name(fonts, tile: str) -> str:
    return fonts.jp(TILE_JP.get(tile, tile), TILE_NAMES.get(tile, tile))


def crop_name(fonts, crop_key: str) -> str:
    return fonts.jp(CROP_JP.get(crop_key, crop_key), crop_key.title())


def seed_crop_name(fonts, seed_key: str) -> str:
    return fonts.jp(SEED_JP.get(seed_key, seed_key), seed_key.replace("_seed", "").title())


def food_name(fonts, item: str) -> str:
    return fonts.jp(FOOD_JP.get(item, item), item.replace("_", " ").title())


def recipe_name(fonts, recipe) -> str:
    return fonts.jp(RECIPE_JP.get(recipe.key, recipe.name), recipe.name)


def station_name(fonts, station_key: str) -> str:
    return fonts.jp(RECIPE_JP.get(station_key, station_key), station_key)


def mat_name(fonts, item: str) -> str:
    return fonts.jp(MAT_JP.get(item, item), item)


# --- Geometry ----------------------------------------------------------------
def manhattan(a: Position, b: Position) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


# --- Menu / action descriptors ----------------------------------------------
@dataclass
class MenuItem:
    """One clickable row in the click-to-act popup.

    ``action`` + ``args`` describe the ``GameAction`` to issue (or a special
    UI verb like ``"move_to"`` / ``"close"`` / ``"submenu"`` / ``"switch_mode"``).
    ``cost`` is the base AP hint shown in the row (None = no pip).
    """

    label: str
    action: str
    args: dict = field(default_factory=dict)
    cost: int | None = None
    enabled: bool = True


@dataclass
class Button:
    """A HUD bar button. ``key`` identifies it for click dispatch."""

    key: str
    label: str
    tooltip: str
    rect: object = None  # pygame.Rect, assigned at layout time
    enabled: bool = True
    active: bool = False  # toggled-on state (e.g. 追従 follow), drawn highlighted


def crop_tooltip(fonts, sim, pos: Position) -> str | None:
    """Crop status for a tile: 'カブ: あと2日' / 'カブ: 収穫できる' / 'カブ: 水が要る'."""
    plot = sim.world.plots.get(pos)
    if plot is None:
        return None
    name = crop_name(fonts, plot.crop)
    if plot.ready:
        return f"{name}: " + fonts.jp("収穫できる", "ready")
    crop = sim.crop_book.get(plot.crop)
    needed = 2 if sim.world.weather == "drought" else 1
    dry = crop.needs_water and sim.world.weather != "rain" and plot.water_level < needed
    days = fonts.jp(f"あと{plot.days_left}日", f"{plot.days_left}d left")
    if dry:
        return f"{name}: {days}・" + fonts.jp("水が要る", "needs water")
    return f"{name}: {days}"


def plantable_seeds(sim) -> list[str]:
    """Seed-crop keys the hero can plant *this season* from inventory."""
    hero = sim.hero
    out = []
    for crop in sim.crop_book.seasonal(sim.world.season):
        if hero.has(crop.seed):
            out.append(crop.key)
    return out


def edible_items(sim) -> list[tuple[str, int]]:
    """(item, food_value) for edible inventory the hero actually holds,
    best first."""
    hero = sim.hero
    foods = [
        (item, FOOD_VALUES[item])
        for item, amount in hero.inventory.items()
        if amount > 0 and item in FOOD_VALUES
    ]
    foods.sort(key=lambda p: p[1], reverse=True)
    return foods


def build_tile_menu(fonts, sim, pos: Position) -> list[MenuItem]:
    """All valid clickable actions for the tile at ``pos``, mirroring the
    engine's validity rules + LocalPolicyAgent's context priorities.

    The hero only acts on its own tile or an adjacent one (the engine's
    ``_nearby_tile`` / ``is_near`` rules), so for a *distant* passable tile the
    only entry is 'move here'. The caller decides whether the popup is
    interactive (manual) or info-only (watch).
    """
    hero = sim.hero
    world = sim.world
    items: list[MenuItem] = []
    here = pos == hero.pos
    adjacent = manhattan(pos, hero.pos) == 1

    # Merchant trade (global while an offer is active) — surfaced on the
    # hero's own tile so it is always reachable.
    if here and sim.current_offer is not None:
        items.append(MenuItem(
            fonts.jp("商人と取引…", "Trade with merchant…"),
            "submenu", {"kind": "trade"},
        ))

    # Move to a different passable tile.
    if not here and world.is_passable(pos):
        items.append(MenuItem(
            fonts.jp("ここへ移動", "Walk here"),
            "move_to", {"x": pos.x, "y": pos.y}, cost=1,
        ))

    if here:
        plot = world.plots.get(pos)
        if plot is not None and plot.ready:
            items.append(MenuItem(fonts.jp("収穫する", "Harvest"), "harvest", cost=1))
        if plot is not None and not plot.ready:
            crop = sim.crop_book.get(plot.crop)
            needed = 2 if world.weather == "drought" else 1
            if crop.needs_water and world.weather != "rain" and plot.water_level < needed:
                items.append(MenuItem(fonts.jp("水をやる", "Water"), "water", cost=1))
        tile = world.tile_at(pos)
        if tile == "field" and pos not in world.plots:
            for crop_key in plantable_seeds(sim):
                label = fonts.jp(f"{CROP_JP.get(crop_key, crop_key)}を植える",
                                 f"Plant {crop_key}")
                items.append(MenuItem(label, "plant", {"crop": crop_key}, cost=1))
        if tile in {"grass", "beach"}:
            items.append(MenuItem(fonts.jp("耕す", "Till"),
                                  "till", cost=1 if hero.has("hoe") else 2))

    # Forest work: on or adjacent to a forest tile.
    if (here or adjacent) and (world.tile_at(pos) == "forest"):
        items.append(MenuItem(fonts.jp("木を切る", "Chop wood"),
                              "chop", cost=1 if hero.has("stone_axe") else 2))
        items.append(MenuItem(fonts.jp("採集する", "Forage"), "forage", cost=2))

    # Mining: on or adjacent to a rock tile.
    if (here or adjacent) and world.tile_at(pos) == "rock":
        items.append(MenuItem(fonts.jp("採掘する", "Mine"), "mine", cost=2))

    # Water work: on/adjacent water, or anywhere if the hero owns a well.
    water_here = world.tile_at(pos) == "water"
    if (here or adjacent) and water_here:
        items.append(MenuItem(fonts.jp("釣りをする", "Fish"), "fish", cost=2))
        items.append(MenuItem(fonts.jp("水を飲む", "Drink"), "drink", cost=1))
    elif here and hero.has("well"):
        items.append(MenuItem(fonts.jp("水を飲む（井戸）", "Drink (well)"), "drink", cost=1))

    # Home comforts.
    if here and world.tile_at(pos) == "home":
        items.append(MenuItem(fonts.jp("休む", "Rest"), "rest", cost=2))
        items.append(MenuItem(fonts.jp("日記を書く", "Write diary"), "write_diary", cost=1))
        items.append(MenuItem(fonts.jp("寝る（1日を終える）", "Sleep (end day)"), "sleep", cost=0))

    return items


def build_trade_menu(fonts, sim) -> list[MenuItem]:
    offer = sim.current_offer
    items: list[MenuItem] = []
    if offer is None:
        return items
    items.append(MenuItem(fonts.jp("受ける", "Accept"),
                          "trade_accept", {"id": offer.id}, cost=1))
    items.append(MenuItem(fonts.jp("断る", "Decline"), "trade_decline", cost=0))
    return items
