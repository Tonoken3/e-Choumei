from __future__ import annotations

"""Colour tables for the Island Diorama.

Everything here is plain RGB tuples. Tile palettes are keyed per season so the
SpriteFactory can rebuild and recache surfaces when the season turns. Weather
and time-of-day tints are returned as RGBA fills painted over the finished map.
"""

from dataclasses import dataclass

RGB = tuple[int, int, int]
RGBA = tuple[int, int, int, int]


@dataclass(frozen=True)
class TilePalette:
    """The colours a single tile type needs: a lit top face plus darker left and
    right side faces (consistent sun) so each block reads as a flat-shaded
    voxel, an edge line for definition, and an accent colour for texture.

    Stage-4 voxel shading derives the side faces from ``top`` with a fixed sun:
    left ≈ 72% brightness, right ≈ 52%. ``side`` is kept for back-compat (it
    equals the left face) so older callers still work."""

    top: RGB
    side: RGB
    edge: RGB
    accent: RGB

    @property
    def left(self) -> RGB:
        return shade(self.top, 0.72)

    @property
    def right(self) -> RGB:
        return shade(self.top, 0.52)

    @property
    def line(self) -> RGB:
        """Thin darker edge line between faces, for definition."""
        return shade(self.top, 0.40)


# Voxel face brightness (consistent sun): top is the lit base.
FACE_LEFT = 0.72
FACE_RIGHT = 0.52


def shade(color: RGB, factor: float) -> RGB:
    """Multiply an RGB toward black (factor<1) or white-ish (factor>1, clamped)."""
    return tuple(max(0, min(255, int(round(c * factor)))) for c in color)  # type: ignore[return-value]


def lighten(color: RGB, amount: int) -> RGB:
    return tuple(max(0, min(255, c + amount)) for c in color)  # type: ignore[return-value]


# --- UI ---------------------------------------------------------------------
UI_BG: RGB = (24, 22, 34)
UI_PANEL: RGB = (38, 34, 52)
UI_PANEL_LIGHT: RGB = (52, 48, 70)
UI_BORDER: RGB = (96, 90, 120)
UI_TEXT: RGB = (232, 230, 240)
UI_TEXT_DIM: RGB = (158, 154, 178)
UI_GOLD: RGB = (236, 200, 96)
UI_BUBBLE: RGB = (248, 248, 244)
UI_BUBBLE_BORDER: RGB = (60, 56, 70)
UI_BUBBLE_TEXT: RGB = (28, 26, 36)
PARCHMENT: RGB = (228, 214, 176)
PARCHMENT_LINE: RGB = (180, 162, 122)
PARCHMENT_TEXT: RGB = (58, 44, 24)
# Bright cursor-hover tile outline.
HILITE: RGB = (255, 244, 180)

# Stat-bar colours (HP / 満腹 / 水 / 体力 / 正気)
STAT_COLORS: dict[str, RGB] = {
    "hp": (212, 72, 72),
    "hunger": (228, 148, 56),
    "water": (72, 140, 220),
    "stamina": (96, 196, 104),
    "sanity": (176, 116, 220),
}
STAT_BG: RGB = (44, 40, 58)
AP_PIP: RGB = (236, 222, 120)
AP_PIP_EMPTY: RGB = (60, 56, 76)


# --- Per-season tile palettes ----------------------------------------------
def _grass(season: str) -> TilePalette:
    if season == "spring":
        return TilePalette((118, 188, 86), (78, 142, 58), (60, 116, 46), (150, 210, 110))
    if season == "summer":
        return TilePalette((78, 156, 64), (50, 116, 44), (38, 92, 36), (104, 180, 80))
    if season == "autumn":
        return TilePalette((170, 156, 70), (132, 116, 48), (104, 90, 38), (196, 176, 86))
    # winter: snow-covered ground
    return TilePalette((226, 232, 240), (188, 198, 214), (158, 168, 188), (244, 248, 252))


def _beach(season: str) -> TilePalette:
    if season == "winter":
        return TilePalette((222, 222, 226), (190, 190, 198), (162, 162, 172), (236, 236, 240))
    return TilePalette((226, 208, 150), (196, 176, 118), (168, 148, 94), (238, 224, 176))


def _field(season: str) -> TilePalette:
    if season == "winter":
        return TilePalette((150, 130, 112), (118, 100, 84), (92, 78, 66), (170, 150, 132))
    return TilePalette((146, 104, 66), (114, 78, 48), (88, 58, 34), (96, 64, 38))


def _rock(season: str) -> TilePalette:
    if season == "winter":
        return TilePalette((150, 154, 166), (110, 114, 128), (84, 88, 102), (210, 216, 226))
    return TilePalette((140, 138, 146), (102, 100, 110), (76, 74, 84), (170, 168, 178))


def _water(season: str) -> TilePalette:
    if season == "winter":
        # frozen: pale ice with crack lines (accent = crack colour)
        return TilePalette((196, 216, 230), (160, 188, 210), (132, 162, 190), (118, 150, 184))
    return TilePalette((56, 110, 178), (40, 86, 150), (30, 70, 128), (140, 188, 226))


def _forest_ground(season: str) -> TilePalette:
    # forest sits on a slightly darker grass base
    base = _grass(season)
    return TilePalette(
        tuple(max(0, c - 18) for c in base.top),  # type: ignore[arg-type]
        tuple(max(0, c - 18) for c in base.side),  # type: ignore[arg-type]
        base.edge,
        base.accent,
    )


def _home(_season: str) -> TilePalette:
    # the ground patch under the house (trodden grass)
    return TilePalette((120, 156, 92), (86, 120, 64), (66, 96, 50), (150, 120, 84))


def _workshop(_season: str) -> TilePalette:
    return TilePalette((128, 132, 140), (94, 98, 108), (72, 76, 86), (108, 120, 140))


def season_tile_palettes(season: str) -> dict[str, TilePalette]:
    return {
        "grass": _grass(season),
        "beach": _beach(season),
        "field": _field(season),
        "rock": _rock(season),
        "water": _water(season),
        "forest": _forest_ground(season),
        "home": _home(season),
        "workshop": _workshop(season),
    }


# --- Crop colours (per growth stage tint base) ------------------------------
CROP_COLORS: dict[str, RGB] = {
    "turnip": (224, 214, 240),   # pale violet-white
    "wheat": (226, 196, 86),     # gold
    "tomato": (212, 70, 60),     # red
    "pumpkin": (228, 142, 52),   # orange
}
# (CROP_LEAF / CROP_SPARKLE and per-object voxel colours live near the bottom,
#  after the atmosphere tints — see the "Voxel object face colours" block.)


# --- Atmosphere tints (RGBA, painted over the finished map) ------------------
def time_of_day_tint(ap_left: int, ap_per_day: int = 12) -> RGBA | None:
    """A multiply-ish overlay that warms then darkens the world as the hero's
    daylight (AP) runs out. None = full daylight (no overlay)."""
    if ap_left >= 8:
        return None  # full daylight
    if ap_left >= 4:
        return (240, 170, 90, 46)   # late afternoon, warm
    if ap_left >= 2:
        return (210, 110, 60, 86)   # dusk, deeper orange
    return (40, 36, 96, 120)        # last light -> indigo


WEATHER_TINT: dict[str, RGBA] = {
    "rain": (40, 70, 120, 46),
    "storm": (24, 30, 64, 96),
    "drought": (236, 206, 96, 40),
    "snow": (210, 218, 232, 30),
}

RAIN_COLOR: RGB = (150, 184, 224)
STORM_FLASH: RGBA = (255, 255, 255, 150)
SNOW_COLOR: RGB = (244, 248, 252)


# --- Sea / seabed backdrop (the slab floats in the sea) ---------------------
def sea_backdrop(season: str) -> RGB:
    """The open-sea colour filling the map band behind the island slab."""
    if season == "winter":
        return (74, 96, 124)
    return (38, 84, 140)


def seabed_color(season: str) -> RGB:
    """The dark seabed slab the island's outer block sides drop down to."""
    if season == "winter":
        return (44, 60, 82)
    return (24, 56, 96)


# --- Voxel object face colours (derived top -> shaded faces in sprites) ------
# Tree
TREE_TRUNK_TOP: RGB = (132, 92, 58)
TREE_SNOW: RGB = (236, 240, 246)
TREE_SNOW_SIDE: RGB = (206, 212, 224)


def tree_leaf_top(season: str) -> RGB:
    return {
        "spring": (110, 192, 92),
        "summer": (70, 158, 72),
        "autumn": (224, 150, 56),
        "winter": (58, 104, 78),  # dark pine
    }[season]


def tree_leaf_accent(season: str) -> RGB:
    # autumn gets a red-orange mix dabbed on; others a brighter highlight
    return {
        "spring": (150, 214, 120),
        "summer": (104, 184, 96),
        "autumn": (210, 86, 56),
        "winter": (84, 132, 104),
    }[season]


# Rock voxel
ROCK_TOP: RGB = (158, 158, 168)
ROCK_MOSS: RGB = (108, 156, 86)       # summer mossy top dab
ROCK_SNOW: RGB = (232, 238, 246)      # winter snow-dusted top

# House voxel cottage
HOUSE_WALL: RGB = (228, 210, 170)
HOUSE_ROOF: RGB = (176, 84, 60)
HOUSE_ROOF_DARK: RGB = (132, 58, 40)
HOUSE_DOOR: RGB = (104, 70, 44)
HOUSE_WINDOW: RGB = (150, 204, 226)

# Workshop voxel: a light stone base, a warm wooden bench, a dark iron anvil so
# the three cuboids read as distinct elements.
WORKSHOP_BLOCK: RGB = (150, 156, 170)
WORKSHOP_BENCH: RGB = (146, 108, 66)
WORKSHOP_ANVIL: RGB = (58, 62, 76)

# Hero (mini Roblox figure)
HERO_SKIN: RGB = (240, 200, 162)
HERO_HAIR: RGB = (66, 46, 38)
HERO_TUNIC: RGB = (208, 76, 76)
HERO_LEG: RGB = (66, 78, 110)
HERO_OUTLINE: RGB = (30, 24, 28)

# Merchant (blue robe + pack)
MERCHANT_SKIN: RGB = (228, 196, 156)
MERCHANT_ROBE: RGB = (74, 110, 196)
MERCHANT_PACK: RGB = (150, 110, 70)

# Field furrow + crop leaf
FIELD_FURROW_DARK: int = 26  # how much darker the furrow grooves are
CROP_LEAF: RGB = (118, 196, 96)   # bright green so sprouts pop on brown soil
CROP_SPARKLE: RGB = (255, 248, 196)
