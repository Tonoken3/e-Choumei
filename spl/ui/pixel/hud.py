from __future__ import annotations

"""HUD, speech/thought bubbles and overlay panels for the diorama.

Two-layer rendering (stage 4: neo-retro voxel world):

* **Layer 1 — the voxel world** is drawn *directly on the window* at native
  resolution into the map band (no low-res upscale any more — clean flat-shaded
  edges are the look). The :mod:`sprites` factory pre-renders each block/object
  variant; the static terrain is composited into one cached full-map surface.
* **Layer 2 — all text & UI chrome** (this module) is drawn on the same window
  surface at native resolution with antialiased fonts, so the (mostly Japanese)
  text stays crisp.

All measurements here are in *window* pixels. A :class:`Layout` carries the
window geometry (the map band, the HUD region), a float ``ui_scale`` for
font/spacing sizing, and the ``sprite_scale`` (iso zoom) used to draw the world.
Default window is Full HD 1920x1080; presets are small(1280x720) /
fhd(1920x1080) / large(2560x1440), and the window stays resizable. A CJK font
is located at construction; if none is found the JP labels degrade to ASCII so
rendering never crashes.
"""

import glob
from dataclasses import dataclass
from typing import TYPE_CHECKING

from spl.arena.leaderboard import select_meigen
from spl.core.crops import FOOD_VALUES
from spl.core.world import SEASON_NAMES, WEATHER_NAMES

from . import palette as pal

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pygame

# -- Window presets -----------------------------------------------------------
# Reinterpreted --scale: 0=auto, 1=small, 2=fhd (default), 3=large.
WINDOW_PRESETS = {
    1: (1280, 720),
    2: (1920, 1080),   # Full HD — the default
    3: (2560, 1440),
}
DEFAULT_WINDOW = WINDOW_PRESETS[2]

# ``ui_scale`` is a float derived from the window height: 1080p -> ~2.4 (about
# +20% over the old stage-3 "scale 2" reference), so the Layout.px() reference
# measurements and the fonts grow with the bigger window. The reference height
# for ui_scale == 2.0 is 900px.
_UI_REF_H = 900.0


def ui_scale_for(win_h: int) -> float:
    """Float UI scale from window height. 1080p -> ~2.4, 720p -> ~1.6,
    1440p -> ~3.2. Floored so the smallest window stays readable."""
    return max(1.4, win_h / _UI_REF_H * 2.0)


# -- Reference (ui_scale==2) sizes --------------------------------------------
# HUD height and button-bar height at ui_scale 2; the real per-window values
# live on the Layout and scale with ui_scale.
HUD_H = 168
BUTTON_BAR_H = 30
# Legacy aliases (kept so older imports don't break).
VIEW_W = DEFAULT_WINDOW[0]
MAP_H = DEFAULT_WINDOW[1] - HUD_H  # nominal map-band height at 1080p

# Base font sizes (pre-multiply by ui_scale*0.9). At ui_scale 2.4 (1080p) this
# lands ~+20% over the old scale-2 art-reviewed targets; floor keeps 720p legible.
_FONT_BASE = {
    "label": 11,   # guide / buttons / HUD labels / small text
    "body": 10,    # menu rows, bubbles, overlay body
    "num": 12,     # stat numbers / score line
    "big": 12,     # overlay headers
    "tip": 9,      # tooltips
}

_FONT_CANDIDATES = (
    "notosanscjkjp",
    "notosanscjkjpregular",
    "notosanmonocjkjp",
    "ipagothic",
    "vlgothic",
    "takaogothic",
)
_FONT_GLOBS = (
    "/usr/share/fonts/**/NotoSansCJK*.otf",
    "/usr/share/fonts/**/NotoSansCJK*.ttc",
)

# JP -> ASCII fallback labels (used only when no CJK font is found).
_STAT_LABELS_JP = {"hp": "HP", "hunger": "満腹", "water": "水", "stamina": "体力", "sanity": "正気"}
_STAT_LABELS_ASCII = {"hp": "HP", "hunger": "Hun", "water": "Wat", "stamina": "Sta", "sanity": "San"}


def _font_size_for(base: int, scale: float) -> int:
    """Slightly sub-linear scaling with a readable floor (spec: never < 14px)."""
    return max(14, int(round(base * scale * 0.9)))


class Fonts:
    """Resolves a CJK-capable font (or falls back) and builds an antialiased
    font set sized for a given window scale. Layer-2 text is *not* pixel-scaled
    any more, so antialiasing is correct here. Call :meth:`for_scale` to (re)build
    the font set when the window scale changes."""

    def __init__(self, pygame_module: "pygame", scale: float = 2.0) -> None:
        self.pg = pygame_module
        self._path = self._find_font_path()
        self.has_cjk = self._path is not None
        self.scale = 0.0
        self.for_scale(scale)

    def for_scale(self, scale: float) -> "Fonts":
        if scale == self.scale:
            return self
        self.scale = scale
        mk = (lambda s: self.pg.font.Font(self._path, s)) if self._path else \
             (lambda s: self.pg.font.Font(None, s + 2))
        self.label = mk(_font_size_for(_FONT_BASE["label"], scale))
        self.body = mk(_font_size_for(_FONT_BASE["body"], scale))
        self.num = mk(_font_size_for(_FONT_BASE["num"], scale))
        self.big = mk(_font_size_for(_FONT_BASE["big"], scale))
        self.tip = mk(_font_size_for(_FONT_BASE["tip"], scale))
        return self

    def _find_font_path(self) -> str | None:
        for name in _FONT_CANDIDATES:
            try:
                found = self.pg.font.match_font(name)
            except Exception:  # noqa: BLE001
                found = None
            if found:
                return found
        for pattern in _FONT_GLOBS:
            matches = sorted(glob.glob(pattern, recursive=True))
            # prefer a Regular weight if present
            matches.sort(key=lambda p: ("Regular" not in p, p))
            if matches:
                return matches[0]
        return None

    def stat_labels(self) -> dict[str, str]:
        return _STAT_LABELS_JP if self.has_cjk else _STAT_LABELS_ASCII

    def jp(self, jp_text: str, ascii_text: str) -> str:
        return jp_text if self.has_cjk else ascii_text


# -- Window geometry ----------------------------------------------------------
@dataclass
class Layout:
    """Where everything lives, in *window* pixels.

    The voxel world is drawn directly into ``map_rect`` (the full-width map band
    across the top of the window) at native resolution; the HUD panel fills the
    remaining height below it. ``ui_scale`` (float) sizes fonts and the
    ``px()`` reference measurements; ``sprite_scale`` (float) is the iso zoom the
    world is drawn at.
    """

    win_w: int
    win_h: int
    ui_scale: float
    sprite_scale: float
    map_rect: "pygame.Rect"     # the full-width map band (window coords)
    hud_top: int                # y where the HUD panel begins (window px)
    hud_h: int                  # HUD panel height (window px)
    button_bar_h: int           # button-bar strip height inside the HUD

    @property
    def scale(self) -> float:
        """Back-compat alias: callers that read ``lay.scale`` get the UI scale."""
        return self.ui_scale

    @property
    def s(self) -> float:
        return float(self.ui_scale)

    def px(self, v: float) -> int:
        """Scale a (ui_scale==2) reference measurement to this window."""
        return int(round(v * self.ui_scale / 2))


def _stat_row_h(scale: float) -> int:
    """Pitch of one stat row: the (floored) num font height plus breathing room."""
    return _font_size_for(_FONT_BASE["num"], scale) + max(2, int(round(4 * scale / 2)))


def hud_height_px(ui_scale: float) -> int:
    """HUD panel height in window px. Must fit the button bar + five stat rows +
    an AP row on the left (the tallest column)."""
    bar = button_bar_height_px(ui_scale)
    row = _stat_row_h(ui_scale)
    pad = max(6, int(round(12 * ui_scale / 2)))
    content = bar + 5 * row + row + pad
    return max(int(round(HUD_H * ui_scale / 2)), content)


def button_bar_height_px(ui_scale: float) -> int:
    return int(round(BUTTON_BAR_H * ui_scale / 2))


def compute_layout(pygame_module, win_w: int, win_h: int,
                   sprite_scale: float = 1.0) -> Layout:
    """Build a :class:`Layout` for a window of ``win_w x win_h``.

    The map band is the full window width and fills the height above the HUD.
    The HUD takes its natural height at the window's ``ui_scale`` (derived from
    the window height)."""
    pg = pygame_module
    ui_scale = ui_scale_for(win_h)
    hud_h = hud_height_px(ui_scale)
    map_h = max(120, win_h - hud_h)
    map_rect = pg.Rect(0, 0, win_w, map_h)
    hud_top = map_h
    hud_h = max(hud_h, win_h - hud_top)
    return Layout(win_w, win_h, ui_scale, sprite_scale, map_rect, hud_top, hud_h,
                  button_bar_height_px(ui_scale))


def _render(font, text: str, color) -> "pygame.Surface":
    # Layer 2 is native-resolution now, so antialiasing is correct.
    return font.render(text, True, color)


def _wrap(text: str, max_chars: int, max_lines: int) -> list[str]:
    """Wrap on width *and* whitespace. CJK has no spaces, so we wrap on a hard
    character count as well; max_chars is a coarse character-count proxy."""
    lines: list[str] = []
    current = ""
    for word in text.replace("\n", " ").split(" "):
        if not word:
            continue
        candidate = (current + " " + word).strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = word
        # hard-break a single long token (e.g. a CJK run with no spaces)
        while len(current) > max_chars:
            lines.append(current[:max_chars])
            current = current[max_chars:]
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: max_chars - 1] + "…"
    return lines


class Hud:
    def __init__(self, pygame_module: "pygame", fonts: Fonts) -> None:
        self.pg = pygame_module
        self.f = fonts

    # -- bottom stat HUD -----------------------------------------------------
    def draw(self, surf, sim, score: int, lay: Layout) -> None:
        pg = self.pg
        f = self.f
        hero = sim.hero
        world = sim.world
        top = lay.hud_top
        pg.draw.rect(surf, pal.UI_PANEL, (0, top, lay.win_w, lay.hud_h))
        pg.draw.line(surf, pal.UI_BORDER, (0, top), (lay.win_w, top), max(1, lay.px(2)))

        # All content is positioned relative to the map band's left edge.
        ox = lay.map_rect.x
        # The button bar (drawn by the app) occupies the first button_bar_h px;
        # stat/text content starts below it.
        content_top = top + lay.button_bar_h + lay.px(4)

        # stat bars (left column). Row pitch tracks the (floored) font height so
        # rows never overlap at scale 1.
        labels = f.stat_labels()
        order = ("hp", "hunger", "water", "stamina", "sanity")
        lab_x = ox + lay.px(6)
        bar_x = ox + lay.px(64)
        bar_w = lay.px(120)
        row_h = _stat_row_h(lay.s)
        bar_h = max(8, lay.px(11))
        bar_dy = (row_h - bar_h) // 2
        y = content_top
        for stat in order:
            value = getattr(hero, stat)
            lab = _render(f.label, labels[stat], pal.UI_TEXT_DIM)
            surf.blit(lab, (lab_x, y + (row_h - lab.get_height()) // 2))
            pg.draw.rect(surf, pal.STAT_BG, (bar_x, y + bar_dy, bar_w, bar_h))
            fill = int(bar_w * max(0, min(100, value)) / 100)
            pg.draw.rect(surf, pal.STAT_COLORS[stat], (bar_x, y + bar_dy, fill, bar_h))
            pg.draw.rect(surf, pal.UI_BORDER, (bar_x, y + bar_dy, bar_w, bar_h), max(1, lay.px(2)))
            val = _render(f.num, str(value), pal.UI_TEXT)
            surf.blit(val, (bar_x + bar_w + lay.px(8), y + (row_h - val.get_height()) // 2))
            y += row_h

        # AP pips (12 small diamonds) below the bars
        ap_y = y + lay.px(2)
        ap_lab = _render(f.label, f.jp("行動", "AP"), pal.UI_TEXT_DIM)
        surf.blit(ap_lab, (lab_x, ap_y))
        pip = max(2, lay.px(3))
        px = bar_x
        pip_gap = lay.px(13)
        for i in range(sim.ap_per_day):
            color = pal.AP_PIP if i < hero.ap_left else pal.AP_PIP_EMPTY
            cx, cy = px + pip, ap_y + ap_lab.get_height() // 2
            pg.draw.polygon(surf, color,
                            [(cx, cy - pip), (cx + pip, cy), (cx, cy + pip), (cx - pip, cy)])
            px += pip_gap

        # right column: day / season / weather, inventory, log, score. Starts
        # well clear of the stat values (bar at 64, width 120, then a 3-digit
        # value ~ +30px) so the two columns never overlap.
        rx = ox + lay.px(232)
        ry = content_top
        season = SEASON_NAMES[world.season]
        weather = WEATHER_NAMES[world.weather]
        day_line = (
            f"{f.jp('日', 'Day')} {world.day}/{sim.max_days}  "
            f"{season} {world.day_in_season}  {weather}"
        )
        surf.blit(_render(f.body, day_line, pal.UI_GOLD), (rx, ry))
        ry += f.body.get_height() + lay.px(3)

        inv = hero.inventory_summary(limit=6)
        inv_text = "  ".join(f"{k}:{v}" for k, v in inv.items()) or f.jp("持ち物なし", "(empty)")
        surf.blit(_render(f.label, inv_text[:80], pal.UI_TEXT), (rx, ry))
        ry += f.label.get_height() + lay.px(2)

        for line in sim.full_log[-2:]:
            surf.blit(_render(f.label, line[:78], pal.UI_TEXT_DIM), (rx, ry))
            ry += f.label.get_height() + lay.px(1)

        # Score line: anchored to the bottom of the panel in the legible num font.
        food = sum(FOOD_VALUES[i] * a for i, a in hero.inventory.items() if i in FOOD_VALUES)
        score_line = (
            f"{f.jp('得点', 'Score')} {score}   "
            f"{f.jp('食料', 'Food')} {food}   "
            f"{f.jp('混乱', 'Conf')} {hero.confusion_count}"
        )
        score_surf = _render(f.num, score_line, pal.UI_GOLD)
        surf.blit(score_surf, (rx, top + lay.hud_h - score_surf.get_height() - lay.px(6)))

        if sim.current_offer is not None:
            mark = _render(f.label, f.jp("商人来訪", "Merchant"), pal.UI_GOLD)
            surf.blit(mark, (lay.win_w - mark.get_width() - lay.px(6),
                             top + lay.hud_h - mark.get_height() - lay.px(4)))

    # -- bubbles -------------------------------------------------------------
    def draw_speech(self, surf, text: str, anchor: tuple[int, int], lay: Layout) -> None:
        if not text:
            return
        pg = self.pg
        f = self.f
        lines = _wrap(text, 26, 3)
        if not lines:
            return
        lh = f.body.get_height()
        widths = [f.body.size(ln)[0] for ln in lines]
        pad = lay.px(6)
        w = max(widths) + pad * 2
        h = lh * len(lines) + pad * 2
        ax, ay = anchor
        bx = max(2, min(lay.win_w - w - 2, ax - w // 2))
        by = max(2, ay - h - lay.px(8))
        rect = pg.Rect(bx, by, w, h)
        rad = lay.px(5)
        pg.draw.rect(surf, pal.UI_BUBBLE, rect, border_radius=rad)
        pg.draw.rect(surf, pal.UI_BUBBLE_BORDER, rect, max(1, lay.px(2)), border_radius=rad)
        # tail pointing down toward the hero
        tail_x = max(bx + pad, min(bx + w - pad, ax))
        t = lay.px(4)
        pg.draw.polygon(
            surf, pal.UI_BUBBLE,
            [(tail_x - t, by + h - 1), (tail_x + t, by + h - 1), (tail_x, by + h + t + 1)],
        )
        for i, ln in enumerate(lines):
            surf.blit(_render(f.body, ln, pal.UI_BUBBLE_TEXT), (bx + pad, by + pad + i * lh))

    def draw_thought(self, surf, anchor: tuple[int, int], lay: Layout) -> None:
        pg = self.pg
        ax, ay = anchor
        w = lay.px(26)
        h = lay.px(16)
        bx, by = ax + lay.px(4), ay - lay.px(24)
        rect = pg.Rect(bx, by, w, h)
        rad = lay.px(6)
        pg.draw.rect(surf, pal.UI_BUBBLE, rect, border_radius=rad)
        pg.draw.rect(surf, pal.UI_BUBBLE_BORDER, rect, max(1, lay.px(2)), border_radius=rad)
        dots = _render(self.f.body, "…", pal.UI_BUBBLE_TEXT)
        surf.blit(dots, (bx + (w - dots.get_width()) // 2, by + (h - dots.get_height()) // 2))
        pg.draw.circle(surf, pal.UI_BUBBLE, (bx + lay.px(2), by + h + lay.px(2)), max(1, lay.px(2)))
        pg.draw.circle(surf, pal.UI_BUBBLE, (bx - lay.px(1), by + h + lay.px(6)), max(1, lay.px(1)))


class Overlays:
    """Modal panels: diary, help, craft menu, heaven's-voice input, result."""

    def __init__(self, pygame_module: "pygame", fonts: Fonts) -> None:
        self.pg = pygame_module
        self.f = fonts

    def _panel(self, surf, title: str, lay: Layout, margin_ref: int = 24):
        """Dim the whole window and draw a centred panel. Returns its rect."""
        pg = self.pg
        margin = lay.px(margin_ref)
        dim = pg.Surface((lay.win_w, lay.win_h), pg.SRCALPHA)
        dim.fill((0, 0, 0, 150))
        surf.blit(dim, (0, 0))
        rect = pg.Rect(margin, margin, lay.win_w - 2 * margin, lay.win_h - 2 * margin)
        rad = lay.px(4)
        pg.draw.rect(surf, pal.UI_PANEL, rect, border_radius=rad)
        pg.draw.rect(surf, pal.UI_BORDER, rect, max(1, lay.px(2)), border_radius=rad)
        surf.blit(_render(self.f.big, title, pal.UI_GOLD), (rect.x + lay.px(8), rect.y + lay.px(6)))
        return rect

    def draw_help(self, surf, lay: Layout) -> None:
        f = self.f
        rect = self._panel(surf, f.jp("あそびかた / ヘルプ", "How to play / Help"), lay)
        mouse_lines = [
            f.jp("◆ マウスだけで遊べます", "* Playable with the mouse alone"),
            f.jp("・タイルをクリック → 行動メニュー（移動・収穫・耕す など）",
                 "- Click a tile -> action menu (move, harvest, till, ...)"),
            f.jp("・タイルにカーソルを乗せると 名前・作物・距離 が出る",
                 "- Hover a tile to see its name, crop and distance"),
            f.jp("・下のボタン: 一時停止 / 速度 / 観戦⇔手動 / 日記 / 作る / 食べる / 天の声",
                 "- Buttons below: pause / speed / mode / diary / craft / eat / heaven"),
            f.jp("・「ここへ移動」を選ぶと自動で歩く（クリックで中断）",
                 "- 'Walk here' auto-walks there (click to interrupt)"),
        ]
        key_lines = [
            f.jp("◆ キーボードの近道（任意）", "* Keyboard accelerators (optional)"),
            "Space: " + f.jp("一時停止 / 閉じる", "pause / close") +
            "    Enter: " + f.jp("決定", "confirm") +
            "    Esc: " + f.jp("閉じる / 終了", "close / quit"),
            f.jp("移動: 矢印/WASD   E: その場の行動   O: 食べる",
                 "Move: arrows/WASD   E: context action   O: eat"),
            "X/V/Q/R/Z: " + f.jp("木/採掘/水/休/寝", "chop/mine/drink/rest/sleep") +
            "    C/D/T: " + f.jp("作る/日記/天の声", "craft/diary/heaven"),
            "1/2/3: " + f.jp("速度", "speed") + "    M: " +
            f.jp("観戦⇔手動", "watch<->manual") + "    H: " + f.jp("ヘルプ", "help"),
        ]
        lh = f.body.get_height() + lay.px(3)
        y = rect.y + f.big.get_height() + lay.px(10)
        x = rect.x + lay.px(10)
        for ln in mouse_lines:
            surf.blit(_render(f.body, ln, pal.UI_TEXT), (x, y))
            y += lh
        y += lay.px(6)
        for ln in key_lines:
            surf.blit(_render(f.body, ln, pal.UI_TEXT_DIM), (x, y))
            y += lh
        y += lay.px(6)
        surf.blit(_render(f.body,
                          f.jp("行灯のような小さな島。英雄が生き延びるのを見守る。",
                               "A small lamplit island. Watch the hero try to survive."),
                          pal.UI_GOLD), (x, y))
        foot = _render(f.label, f.jp("外側クリック / Space / H で閉じる",
                                     "click outside / Space / H to close"), pal.UI_TEXT_DIM)
        surf.blit(foot, (x, rect.bottom - foot.get_height() - lay.px(6)))

    def draw_diary(self, surf, sim, scroll: int, lay: Layout) -> None:
        pg = self.pg
        f = self.f
        margin = lay.px(24)
        dim = pg.Surface((lay.win_w, lay.win_h), pg.SRCALPHA)
        dim.fill((0, 0, 0, 150))
        surf.blit(dim, (0, 0))
        rect = pg.Rect(margin, margin, lay.win_w - 2 * margin, lay.win_h - 2 * margin)
        rad = lay.px(4)
        pg.draw.rect(surf, pal.PARCHMENT, rect, border_radius=rad)
        pg.draw.rect(surf, pal.PARCHMENT_LINE, rect, max(1, lay.px(2)), border_radius=rad)
        surf.blit(_render(f.body,
                          f.jp("日記 (ホイール/↑↓ でスクロール・外側クリックで閉じる)",
                               "Diary (wheel/Up-Down to scroll, click outside to close)"),
                          pal.PARCHMENT_TEXT), (rect.x + lay.px(8), rect.y + lay.px(6)))
        entries = sim.memory.diary
        text_lines: list[str] = []
        for entry in reversed(entries):
            for raw in entry.text.splitlines():
                text_lines.extend(_wrap(raw, 60, 4) or [""])
            text_lines.append("")
        if not text_lines:
            text_lines = [f.jp("まだ日記はない。", "(No diary entries yet.)")]
        start = max(0, min(scroll, max(0, len(text_lines) - 1)))
        lh = f.label.get_height() + lay.px(2)
        y = rect.y + f.body.get_height() + lay.px(8)
        for ln in text_lines[start:]:
            if y > rect.bottom - lh:
                break
            color = pal.PARCHMENT_TEXT if not ln.startswith("Day ") else pal.PARCHMENT_LINE
            surf.blit(_render(f.label, ln, color), (rect.x + lay.px(10), y))
            y += lh

    def craft_rows(self, sim) -> list[tuple[object, bool]]:
        """All recipes with an affordable flag (own materials + station, not
        already owned)."""
        hero = sim.hero
        rows = []
        for recipe in sim.recipe_book.all():
            if hero.has(recipe.key):
                affordable = False
            else:
                has_station = (not recipe.station) or hero.has(recipe.station)
                has_mats = all(hero.item_count(i) >= a for i, a in recipe.requires.items())
                affordable = has_station and has_mats
            rows.append((recipe, affordable))
        return rows

    def draw_craft(self, surf, sim, selected: int, lay: Layout, hover: int = -1) -> list:
        """Mouse-driven craft/build menu. Returns a list of (row_rect, recipe,
        affordable) in *window* coords so the app can hit-test clicks."""
        from . import uihelp as uh

        pg = self.pg
        f = self.f
        rect = self._panel(
            surf, f.jp("作る / 建てる  (行をクリック)", "Craft / Build  (click a row)"), lay
        )
        rows = self.craft_rows(sim)
        hero = sim.hero
        row_h = f.body.get_height() + lay.px(6)
        y = rect.y + f.big.get_height() + lay.px(10)
        hits = []
        for i, (recipe, affordable) in enumerate(rows):
            owned = hero.has(recipe.key)
            row_rect = pg.Rect(rect.x + lay.px(6), y - lay.px(1), rect.width - lay.px(12), row_h)
            if (i == hover or i == selected) and affordable:
                pg.draw.rect(surf, pal.UI_PANEL_LIGHT, row_rect, border_radius=lay.px(2))
            if owned:
                color = pal.STAT_COLORS["stamina"]
            elif affordable:
                color = pal.UI_TEXT
            else:
                color = pal.UI_TEXT_DIM
            name = uh.recipe_name(f, recipe)
            kind = f.jp("建てる", "build") if recipe.kind == "build" else f.jp("道具", "item")
            label = f"{name} ({kind})"
            surf.blit(_render(f.body, label, color), (row_rect.x + lay.px(6), y))
            # cost / status on the right
            if owned:
                tag = _render(f.label, f.jp("所持済", "owned"), pal.STAT_COLORS["stamina"])
                surf.blit(tag, (row_rect.right - tag.get_width() - lay.px(6), y))
            else:
                lx = row_rect.x + lay.px(220)
                parts = list(recipe.requires.items())
                if recipe.station and not hero.has(recipe.station):
                    sta = _render(f.label,
                                  f.jp(f"要:{uh.station_name(f, recipe.station)}",
                                       f"needs {recipe.station}"),
                                  pal.STAT_COLORS["hp"])
                    surf.blit(sta, (lx, y))
                    lx += sta.get_width() + lay.px(8)
                for item, amount in parts:
                    have = hero.item_count(item)
                    short = have < amount
                    col = pal.STAT_COLORS["hp"] if short else pal.UI_TEXT_DIM
                    txt = _render(f.label, f"{uh.mat_name(f, item)}x{amount}", col)
                    if lx + txt.get_width() > row_rect.right - lay.px(6):
                        break
                    surf.blit(txt, (lx, y))
                    lx += txt.get_width() + lay.px(8)
            hits.append((row_rect, recipe, affordable))
            y += row_h
        hint = _render(f.label,
                       f.jp("外側クリック か Space で閉じる", "click outside / Space to close"),
                       pal.UI_TEXT_DIM)
        surf.blit(hint, (rect.x + lay.px(10), rect.bottom - hint.get_height() - lay.px(6)))
        return hits

    def draw_eat(self, surf, sim, lay: Layout, hover: int = -1) -> list:
        """Eat popup: edible inventory as 'パン (満腹+45)' rows. Returns
        (row_rect, item) hit list in window coords. Centred panel."""
        from . import uihelp as uh
        from spl.core.crops import FOOD_VALUES

        pg = self.pg
        f = self.f
        foods = uh.edible_items(sim)
        title = f.jp("食べる  (行をクリック)", "Eat  (click a row)")
        row_h = f.body.get_height() + lay.px(7)
        w = lay.px(280)
        h = f.big.get_height() + lay.px(16) + max(1, len(foods)) * row_h
        h = min(h, lay.win_h - lay.px(48))
        rx = (lay.win_w - w) // 2
        ry = (lay.win_h - h) // 2
        dim = pg.Surface((lay.win_w, lay.win_h), pg.SRCALPHA)
        dim.fill((0, 0, 0, 150))
        surf.blit(dim, (0, 0))
        rect = pg.Rect(rx, ry, w, h)
        rad = lay.px(4)
        pg.draw.rect(surf, pal.UI_PANEL, rect, border_radius=rad)
        pg.draw.rect(surf, pal.UI_BORDER, rect, max(1, lay.px(2)), border_radius=rad)
        surf.blit(_render(f.big, title, pal.UI_GOLD), (rect.x + lay.px(8), rect.y + lay.px(6)))
        y = rect.y + f.big.get_height() + lay.px(10)
        hits = []
        if not foods:
            surf.blit(_render(f.body, f.jp("食べられる物がない", "(nothing edible)"),
                              pal.UI_TEXT_DIM), (rect.x + lay.px(10), y))
        for i, (item, _value) in enumerate(foods):
            row_rect = pg.Rect(rect.x + lay.px(6), y - lay.px(1), rect.width - lay.px(12), row_h)
            if i == hover:
                pg.draw.rect(surf, pal.UI_PANEL_LIGHT, row_rect, border_radius=lay.px(2))
            count = sim.hero.item_count(item)
            label = (f"{uh.food_name(f, item)} x{count}  "
                     f"({f.jp('満腹', 'Hun')}+{FOOD_VALUES[item]})")
            surf.blit(_render(f.body, label, pal.UI_TEXT), (row_rect.x + lay.px(6), y))
            hits.append((row_rect, item))
            y += row_h
        return hits

    def draw_heaven(self, surf, text: str, lay: Layout, send_hover: bool = False):
        """Heaven's-voice text entry. Returns the [送る] button rect (window
        coords) so the app can hit-test clicks (Enter still confirms)."""
        pg = self.pg
        f = self.f
        rect = self._panel(surf, f.jp("天の声を入力", "Heaven's Voice"), lay)
        x = rect.x + lay.px(10)
        surf.blit(
            _render(f.body,
                    f.jp("英雄に届く一言を授ける。Enter か [送る] で送信。",
                         "Whisper one line of guidance. Enter or [Send]."),
                    pal.UI_TEXT_DIM),
            (x, rect.y + f.big.get_height() + lay.px(10)),
        )
        box_y = rect.y + f.big.get_height() + lay.px(14) + f.body.get_height() + lay.px(8)
        box_h = f.body.get_height() + lay.px(8)
        box = pg.Rect(x, box_y, rect.width - lay.px(20), box_h)
        pg.draw.rect(surf, pal.UI_PANEL_LIGHT, box)
        pg.draw.rect(surf, pal.UI_BORDER, box, max(1, lay.px(2)))
        surf.blit(_render(f.body, text + "_", pal.UI_TEXT), (box.x + lay.px(4), box.y + lay.px(3)))
        # [送る] button
        send_w = lay.px(84)
        send_h = f.body.get_height() + lay.px(10)
        send = pg.Rect(x, box.bottom + lay.px(8), send_w, send_h)
        rad = lay.px(3)
        pg.draw.rect(surf, pal.UI_PANEL_LIGHT if send_hover else pal.UI_PANEL, send, border_radius=rad)
        pg.draw.rect(surf, pal.UI_GOLD if send_hover else pal.UI_BORDER, send,
                     max(1, lay.px(2)), border_radius=rad)
        slab = _render(f.body, f.jp("送る", "Send"), pal.UI_TEXT)
        surf.blit(slab, (send.centerx - slab.get_width() // 2, send.centery - slab.get_height() // 2))
        foot = _render(f.label, f.jp("Esc で閉じる", "Esc to close"), pal.UI_TEXT_DIM)
        surf.blit(foot, (x, rect.bottom - foot.get_height() - lay.px(6)))
        return send

    def draw_result(self, surf, sim, lay: Layout, hover: str = "",
                    motto: dict | None = None, motto_pending: bool = False):
        """End-of-run panel. Returns {"again": rect, "quit": rect} in window
        coords so the app can hit-test the [もう一度] / [終了] buttons."""
        pg = self.pg
        f = self.f
        rect = self._panel(surf, f.jp("結果", "Result"), lay, margin_ref=18)
        hero = sim.hero
        x = rect.x + lay.px(10)
        y = rect.y + f.big.get_height() + lay.px(10)
        # The crown of the run: the hermit's own 座右の銘, distilled from the
        # year's five best lines (LLM-written when a brain played).
        if motto_pending and not motto:
            slab = _render(f.body, f.jp("いま、座右の銘を綴っている……", "Writing the final motto..."), pal.UI_TEXT_DIM)
            surf.blit(slab, (x, y))
            y += f.body.get_height() + lay.px(8)
        elif motto:
            for wln in _wrap(f"座右の銘 「{motto.get('motto', '')}」", 44, 2):
                slab = _render(f.big, f.jp(wln, wln), pal.UI_GOLD)
                surf.blit(slab, (x, y))
                y += f.big.get_height() + lay.px(2)
            words = motto.get("words", "")
            if words:
                for wln in _wrap(words, 56, 2):
                    slab = _render(f.body, wln, pal.UI_TEXT)
                    surf.blit(slab, (x + lay.px(6), y))
                    y += f.body.get_height() + lay.px(2)
            y += lay.px(8)
        info = [
            sim.result_reason or (f.jp("生存中。", "Still alive.") if hero.alive
                                  else f.jp("英雄は倒れた。", "The hero fell.")),
            f"{f.jp('得点', 'Score')}: {sim.score()}",
            f"{f.jp('生存日数', 'Days')}: {hero.days_survived}",
            f"{f.jp('混乱', 'Confusions')}: {hero.confusion_count}",
            f"{f.jp('文明点', 'Civ pts')}: {hero.civilization_points()}",
            "",
            f.jp("銘言ベスト5:", "Best lines:"),
        ]
        lh = f.body.get_height() + lay.px(3)
        for ln in info:
            surf.blit(_render(f.body, ln, pal.UI_TEXT), (x, y))
            y += lh
        best = select_meigen(hero.spoken_lines, 5)
        if not best:
            best = [f.jp("(英雄は黙したまま)", "(the hero kept quiet)")]
        qlh = f.label.get_height() + lay.px(2)
        for line in best:
            for wln in _wrap(f'"{line}"', 60, 2):
                surf.blit(_render(f.label, wln, pal.UI_GOLD), (x + lay.px(6), y))
                y += qlh

        highlights = (motto or {}).get("highlights") or []
        if highlights:
            y += lay.px(8)
            surf.blit(_render(f.body, f.jp("天の声の記録:", "The watcher's record:"), pal.UI_TEXT), (x, y))
            y += f.body.get_height() + lay.px(3)
            for line in highlights[:5]:
                for wln in _wrap(str(line), 62, 2):
                    surf.blit(_render(f.label, wln, pal.UI_TEXT), (x + lay.px(6), y))
                    y += qlh

        # Buttons along the bottom.
        btn_h = f.body.get_height() + lay.px(10)
        again_w = lay.px(220)
        quit_w = lay.px(110)
        by = rect.bottom - btn_h - lay.px(8)
        again = pg.Rect(x, by, again_w, btn_h)
        quit_b = pg.Rect(again.right + lay.px(10), by, quit_w, btn_h)
        rad = lay.px(3)
        for key, b, label in (
            ("again", again, f.jp("もう一度（同じ島）", "Play again (same island)")),
            ("quit", quit_b, f.jp("終了", "Quit")),
        ):
            hot = hover == key
            pg.draw.rect(surf, pal.UI_PANEL_LIGHT if hot else pal.UI_PANEL, b, border_radius=rad)
            pg.draw.rect(surf, pal.UI_GOLD if hot else pal.UI_BORDER, b, max(1, lay.px(2)), border_radius=rad)
            lab = _render(f.body, label, pal.UI_TEXT)
            surf.blit(lab, (b.centerx - lab.get_width() // 2, b.centery - lab.get_height() // 2))
        return {"again": again, "quit": quit_b}

    # -- map hover: highlight diamond ----------------------------------------
    # These three draw onto the world layer (window coords now), as the top-face
    # diamond outline at the tile's screen centre. ``hw``/``hh`` are the iso
    # half-width/half-height at the current sprite scale.
    def draw_tile_highlight(self, surf, cx: int, cy: int, hw: int, hh: int,
                            color=None, width: int = 2) -> None:
        pg = self.pg
        col = color or pal.HILITE
        pts = [(cx, cy - hh), (cx + hw, cy), (cx, cy + hh), (cx - hw, cy)]
        pg.draw.polygon(surf, col, pts, max(2, width))

    def draw_walk_marker(self, surf, cx: int, cy: int, hw: int, hh: int,
                         phase: float, width: int = 2) -> None:
        pg = self.pg
        import math

        k = 0.5 + 0.5 * math.sin(phase * 2 * math.pi)
        w = int(hw * (0.6 + 0.4 * k))
        h = int(hh * (0.6 + 0.4 * k))
        pts = [(cx, cy - h), (cx + w, cy), (cx, cy + h), (cx - w, cy)]
        pg.draw.polygon(surf, (255, 240, 150), pts, max(2, width))
        c = max(2, width)
        pg.draw.polygon(surf, (255, 255, 220),
                        [(cx, cy - c), (cx + c, cy), (cx, cy + c), (cx - c, cy)])

    def draw_hero_locator(self, surf, cx: int, cy: int, hw: int, hh: int,
                          phase: float, width: int = 2) -> None:
        pg = self.pg
        import math

        k = 0.5 + 0.5 * math.sin(phase * 2 * math.pi)
        r = int(2 + 4 * k)
        pts = [(cx, cy - hh - r), (cx + hw + r, cy),
               (cx, cy + hh + r), (cx - hw - r, cy)]
        pg.draw.polygon(surf, (255, 255, 255), pts, max(2, width))

    # -- tooltip (LAYER 2: window coords, crisp text) ------------------------
    def draw_tooltip(self, surf, mx: int, my: int, lines: list, lay: Layout) -> None:
        """Small panel near the cursor (window coords) with up to a few short
        info lines."""
        pg = self.pg
        f = self.f
        lines = [ln for ln in lines if ln]
        if not lines:
            return
        widths = [f.tip.size(ln)[0] for ln in lines]
        pad = lay.px(6)
        w = max(widths) + pad * 2
        lh = f.tip.get_height() + lay.px(1)
        h = lh * len(lines) + pad * 2
        bx = mx + lay.px(14)
        by = my + lay.px(10)
        if bx + w > lay.win_w - 2:
            bx = mx - w - lay.px(8)
        if by + h > lay.win_h - 2:
            by = my - h - lay.px(8)
        bx = max(2, min(lay.win_w - w - 2, bx))
        by = max(2, min(lay.win_h - h - 2, by))
        panel = pg.Surface((w, h), pg.SRCALPHA)
        panel.fill((20, 18, 30, 224))
        surf.blit(panel, (bx, by))
        pg.draw.rect(surf, pal.UI_BORDER, (bx, by, w, h), max(1, lay.px(1)))
        for i, ln in enumerate(lines):
            color = pal.UI_GOLD if i == 0 else pal.UI_TEXT
            surf.blit(_render(f.tip, ln, color), (bx + pad, by + pad + i * lh))

    # -- click-to-act popup --------------------------------------------------
    def draw_popup(self, surf, header: str, items: list, anchor: tuple,
                   lay: Layout, hover: int = -1):
        """Context menu (window coords). ``items`` is a list of uihelp.MenuItem.
        ``anchor`` is the click point in window coords. Returns a list of row
        rects (parallel to ``items``) for hit-testing."""
        pg = self.pg
        f = self.f
        pad = lay.px(6)
        row_h = f.body.get_height() + lay.px(6)
        head_h = f.body.get_height() + lay.px(6)
        labels = [it.label + ("" if it.cost is None else f"  (AP{it.cost})") for it in items]
        widths = [f.body.size(header)[0]] + [f.body.size(s)[0] for s in labels]
        w = max(widths) + pad * 2 + lay.px(24)
        w = max(w, lay.px(150))
        h = head_h + row_h * len(items) + pad
        ax, ay = anchor
        bx = ax + lay.px(6)
        by = ay + lay.px(6)
        if bx + w > lay.win_w - 2:
            bx = ax - w - lay.px(6)
        if by + h > lay.win_h - 2:
            by = lay.win_h - h - 2
        bx = max(2, min(lay.win_w - w - 2, bx))
        by = max(2, min(lay.win_h - h - 2, by))
        rect = pg.Rect(bx, by, w, h)
        rad = lay.px(4)
        pg.draw.rect(surf, pal.UI_PANEL, rect, border_radius=rad)
        pg.draw.rect(surf, pal.UI_GOLD, rect, max(1, lay.px(2)), border_radius=rad)
        surf.blit(_render(f.body, header, pal.UI_GOLD), (bx + pad, by + lay.px(3)))
        pg.draw.line(surf, pal.UI_BORDER, (bx + lay.px(3), by + head_h - 1),
                     (bx + w - lay.px(3), by + head_h - 1), max(1, lay.px(1)))
        rects = []
        y = by + head_h
        for i, it in enumerate(items):
            row_rect = pg.Rect(bx + lay.px(3), y, w - lay.px(6), row_h)
            if i == hover and it.enabled:
                pg.draw.rect(surf, pal.UI_PANEL_LIGHT, row_rect, border_radius=lay.px(2))
            color = pal.UI_TEXT if it.enabled else pal.UI_TEXT_DIM
            surf.blit(_render(f.body, it.label, color), (row_rect.x + pad, y + lay.px(2)))
            if it.cost is not None:
                cost = _render(f.label, f"(AP{it.cost})", pal.UI_TEXT_DIM)
                surf.blit(cost, (row_rect.right - cost.get_width() - lay.px(6), y + lay.px(3)))
            rects.append(row_rect)
            y += row_h
        return rects

    # -- HUD button bar ------------------------------------------------------
    def draw_button_bar(self, surf, buttons: list, hover_key: str | None, lay: Layout):
        """Draw the clickable HUD buttons (window coords; ``.rect`` already laid
        out by the app). Hovered button gets a highlight + a tooltip above."""
        pg = self.pg
        f = self.f
        tip = None
        rad = lay.px(3)
        for b in buttons:
            r = b.rect
            hot = (b.key == hover_key) and b.enabled
            bg = pal.UI_PANEL_LIGHT if hot else pal.UI_PANEL
            border = pal.UI_GOLD if hot else pal.UI_BORDER
            pg.draw.rect(surf, bg, r, border_radius=rad)
            pg.draw.rect(surf, border, r, max(1, lay.px(2)), border_radius=rad)
            color = pal.UI_TEXT if b.enabled else pal.UI_TEXT_DIM
            lab = _render(f.label, b.label, color)
            surf.blit(lab, (r.centerx - lab.get_width() // 2, r.centery - lab.get_height() // 2))
            if hot and b.tooltip:
                tip = (b, b.tooltip)
        if tip is not None:
            b, text = tip
            ts = _render(f.label, text, pal.UI_TEXT)
            tw = ts.get_width() + lay.px(10)
            th = ts.get_height() + lay.px(4)
            tx = b.rect.centerx - tw // 2
            tx = max(2, min(lay.win_w - tw - 2, tx))
            ty = b.rect.y - th - lay.px(2)
            panel = pg.Surface((tw, th), pg.SRCALPHA)
            panel.fill((20, 18, 30, 230))
            surf.blit(panel, (tx, ty))
            pg.draw.rect(surf, pal.UI_BORDER, (tx, ty, tw, th), max(1, lay.px(1)))
            surf.blit(ts, (tx + lay.px(5), ty + lay.px(2)))

    # -- guide strip ---------------------------------------------------------
    def draw_guide(self, surf, text: str, lay: Layout) -> None:
        """Semi-transparent dark banner with JP guidance text, across the top of
        the (letterboxed) map band."""
        if not text:
            return
        pg = self.pg
        f = self.f
        ts = _render(f.label, text, pal.UI_TEXT)
        bar_h = ts.get_height() + lay.px(4)
        band = pg.Surface((lay.map_rect.width, bar_h), pg.SRCALPHA)
        band.fill((12, 10, 20, 150))
        surf.blit(band, (lay.map_rect.x, lay.map_rect.y))
        surf.blit(ts, (lay.map_rect.x + (lay.map_rect.width - ts.get_width()) // 2,
                       lay.map_rect.y + lay.px(2)))
