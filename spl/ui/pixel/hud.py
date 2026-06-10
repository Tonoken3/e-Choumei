from __future__ import annotations

"""HUD, speech/thought bubbles and overlay panels for the diorama.

All measurements are in *internal* pixels (the 640x460 surface, scaled x2 to
the window). A CJK font is located at construction; if none is found the JP
labels degrade to ASCII so rendering never crashes.
"""

import glob
from typing import TYPE_CHECKING

from spl.arena.leaderboard import select_meigen
from spl.core.crops import FOOD_VALUES
from spl.core.world import SEASON_NAMES, WEATHER_NAMES

from . import palette as pal

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pygame

VIEW_W = 640
MAP_H = 360
# The HUD got a taller bottom panel in stage 2 so the score line (得点/食料/混乱)
# is no longer clipped off the window edge, and so a clickable button bar fits
# along the top of the panel. The map size is unchanged (still 0..MAP_H).
HUD_H = 140
VIEW_H = MAP_H + HUD_H  # 500
# Height of the button-bar strip at the top of the HUD panel (mouse access).
BUTTON_BAR_H = 22

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


class Fonts:
    """Resolves a CJK-capable font (or falls back) and reports whether JP text
    is renderable so callers can swap to ASCII labels."""

    def __init__(self, pygame_module: "pygame") -> None:
        self.pg = pygame_module
        path = self._find_font_path()
        self.has_cjk = path is not None
        if path is not None:
            self.label = pygame_module.font.Font(path, 10)
            self.body = pygame_module.font.Font(path, 12)
            # ``num`` renders stat values + the score line large enough that an
            # "8" can never be mistaken for a "B" (art-review fix).
            self.num = pygame_module.font.Font(path, 13)
            self.big = pygame_module.font.Font(path, 18)
        else:
            self.label = pygame_module.font.Font(None, 12)
            self.body = pygame_module.font.Font(None, 14)
            self.num = pygame_module.font.Font(None, 16)
            self.big = pygame_module.font.Font(None, 20)

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


def _render(font, text: str, color) -> "pygame.Surface":
    return font.render(text, False, color)


def _wrap(text: str, max_chars: int, max_lines: int) -> list[str]:
    """Wrap on width *and* whitespace. CJK has no spaces, so we wrap on a hard
    character count as well; max_chars is a coarse internal-pixel proxy."""
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
    def draw(self, surf, sim, score: int) -> None:
        pg = self.pg
        hero = sim.hero
        world = sim.world
        top = MAP_H
        pg.draw.rect(surf, pal.UI_PANEL, (0, top, VIEW_W, HUD_H))
        pg.draw.line(surf, pal.UI_BORDER, (0, top), (VIEW_W, top), 1)

        # The button bar (drawn by the app) occupies the first BUTTON_BAR_H px;
        # all stat/text content starts below it so nothing overlaps.
        content_top = top + BUTTON_BAR_H + 2

        # stat bars (left column). Values use the larger ``num`` font so digits
        # are unambiguous (no "8"->"B").
        labels = self.f.stat_labels()
        order = ("hp", "hunger", "water", "stamina", "sanity")
        bar_x = 38
        bar_w = 150
        y = content_top
        for stat in order:
            value = getattr(hero, stat)
            lab = _render(self.f.label, labels[stat], pal.UI_TEXT_DIM)
            surf.blit(lab, (4, y + 1))
            pg.draw.rect(surf, pal.STAT_BG, (bar_x, y + 2, bar_w, 8))
            fill = int(bar_w * max(0, min(100, value)) / 100)
            pg.draw.rect(surf, pal.STAT_COLORS[stat], (bar_x, y + 2, fill, 8))
            pg.draw.rect(surf, pal.UI_BORDER, (bar_x, y + 2, bar_w, 8), 1)
            val = _render(self.f.num, str(value), pal.UI_TEXT)
            surf.blit(val, (bar_x + bar_w + 5, y - 1))
            y += 13

        # AP pips (12 small diamonds) below the bars
        ap_y = y + 2
        ap_lab = _render(self.f.label, self.f.jp("行動", "AP"), pal.UI_TEXT_DIM)
        surf.blit(ap_lab, (4, ap_y))
        px = bar_x
        for i in range(sim.ap_per_day):
            color = pal.AP_PIP if i < hero.ap_left else pal.AP_PIP_EMPTY
            cx, cy = px + 4, ap_y + 5
            pg.draw.polygon(surf, color, [(cx, cy - 3), (cx + 3, cy), (cx, cy + 3), (cx - 3, cy)])
            px += 9

        # right column: day / season / weather, inventory, log, score
        rx = 236
        ry = content_top
        season = SEASON_NAMES[world.season]
        weather = WEATHER_NAMES[world.weather]
        day_line = (
            f"{self.f.jp('日', 'Day')} {world.day}/{sim.max_days}  "
            f"{season} {world.day_in_season}  {weather}"
        )
        surf.blit(_render(self.f.body, day_line, pal.UI_GOLD), (rx, ry))
        ry += 15

        inv = hero.inventory_summary(limit=6)
        inv_text = "  ".join(f"{k}:{v}" for k, v in inv.items()) or self.f.jp("持ち物なし", "(empty)")
        surf.blit(_render(self.f.label, inv_text[:80], pal.UI_TEXT), (rx, ry))
        ry += 13

        for line in sim.full_log[-2:]:
            surf.blit(_render(self.f.label, line[:78], pal.UI_TEXT_DIM), (rx, ry))
            ry += 12

        # Score line: drawn at a fixed baseline well inside the panel (taller
        # HUD means it is no longer clipped) and in the legible ``num`` font.
        food = sum(FOOD_VALUES[i] * a for i, a in hero.inventory.items() if i in FOOD_VALUES)
        score_line = (
            f"{self.f.jp('得点', 'Score')} {score}   "
            f"{self.f.jp('食料', 'Food')} {food}   "
            f"{self.f.jp('混乱', 'Conf')} {hero.confusion_count}"
        )
        surf.blit(_render(self.f.num, score_line, pal.UI_GOLD), (rx, top + HUD_H - 18))

        if sim.current_offer is not None:
            mark = _render(self.f.label, self.f.jp("商人来訪", "Merchant"), pal.UI_GOLD)
            surf.blit(mark, (VIEW_W - mark.get_width() - 6, top + HUD_H - 14))

    # -- bubbles -------------------------------------------------------------
    def draw_speech(self, surf, text: str, anchor: tuple[int, int]) -> None:
        if not text:
            return
        pg = self.pg
        lines = _wrap(text, 26, 3)
        if not lines:
            return
        lh = self.f.body.get_height()
        widths = [self.f.body.size(ln)[0] for ln in lines]
        w = max(widths) + 10
        h = lh * len(lines) + 8
        ax, ay = anchor
        bx = max(2, min(VIEW_W - w - 2, ax - w // 2))
        by = max(2, ay - h - 8)
        rect = pg.Rect(bx, by, w, h)
        pg.draw.rect(surf, pal.UI_BUBBLE, rect, border_radius=5)
        pg.draw.rect(surf, pal.UI_BUBBLE_BORDER, rect, 1, border_radius=5)
        # tail pointing down toward the hero
        tail_x = max(bx + 6, min(bx + w - 6, ax))
        pg.draw.polygon(
            surf, pal.UI_BUBBLE,
            [(tail_x - 4, by + h - 1), (tail_x + 4, by + h - 1), (tail_x, by + h + 5)],
        )
        for i, ln in enumerate(lines):
            surf.blit(_render(self.f.body, ln, pal.UI_BUBBLE_TEXT), (bx + 5, by + 4 + i * lh))

    def draw_thought(self, surf, anchor: tuple[int, int]) -> None:
        pg = self.pg
        ax, ay = anchor
        bx, by = ax + 4, ay - 22
        rect = pg.Rect(bx, by, 22, 14)
        pg.draw.rect(surf, pal.UI_BUBBLE, rect, border_radius=6)
        pg.draw.rect(surf, pal.UI_BUBBLE_BORDER, rect, 1, border_radius=6)
        surf.blit(_render(self.f.body, "…", pal.UI_BUBBLE_TEXT), (bx + 5, by))
        pg.draw.circle(surf, pal.UI_BUBBLE, (bx + 2, by + 16), 2)
        pg.draw.circle(surf, pal.UI_BUBBLE, (bx - 1, by + 20), 1)


class Overlays:
    """Modal panels: diary, help, craft menu, heaven's-voice input, result."""

    def __init__(self, pygame_module: "pygame", fonts: Fonts) -> None:
        self.pg = pygame_module
        self.f = fonts

    def _panel(self, surf, title: str, margin: int = 24):
        pg = self.pg
        dim = pg.Surface((VIEW_W, VIEW_H), pg.SRCALPHA)
        dim.fill((0, 0, 0, 150))
        surf.blit(dim, (0, 0))
        rect = pg.Rect(margin, margin, VIEW_W - 2 * margin, VIEW_H - 2 * margin)
        pg.draw.rect(surf, pal.UI_PANEL, rect, border_radius=4)
        pg.draw.rect(surf, pal.UI_BORDER, rect, 1, border_radius=4)
        surf.blit(_render(self.f.body, title, pal.UI_GOLD), (rect.x + 8, rect.y + 6))
        return rect

    def draw_help(self, surf) -> None:
        rect = self._panel(surf, self.f.jp("あそびかた / ヘルプ", "How to play / Help"))
        # Mouse-first instructions at the top, keyboard accelerators below.
        mouse_lines = [
            self.f.jp("◆ マウスだけで遊べます", "* Playable with the mouse alone"),
            self.f.jp("・タイルをクリック → 行動メニュー（移動・収穫・耕す など）",
                      "- Click a tile -> action menu (move, harvest, till, ...)"),
            self.f.jp("・タイルにカーソルを乗せると 名前・作物・距離 が出る",
                      "- Hover a tile to see its name, crop and distance"),
            self.f.jp("・下のボタン: 一時停止 / 速度 / 観戦⇔手動 / 日記 / 作る / 食べる / 天の声",
                      "- Buttons below: pause / speed / mode / diary / craft / eat / heaven"),
            self.f.jp("・「ここへ移動」を選ぶと自動で歩く（クリックで中断）",
                      "- 'Walk here' auto-walks there (click to interrupt)"),
        ]
        key_lines = [
            self.f.jp("◆ キーボードの近道（任意）", "* Keyboard accelerators (optional)"),
            "Space: " + self.f.jp("一時停止 / 閉じる", "pause / close") +
            "    Enter: " + self.f.jp("決定", "confirm") +
            "    Esc: " + self.f.jp("閉じる / 終了", "close / quit"),
            self.f.jp("移動: 矢印/WASD   E: その場の行動   O: 食べる",
                      "Move: arrows/WASD   E: context action   O: eat"),
            "X/V/Q/R/Z: " + self.f.jp("木/採掘/水/休/寝", "chop/mine/drink/rest/sleep") +
            "    C/D/T: " + self.f.jp("作る/日記/天の声", "craft/diary/heaven"),
            "1/2/3: " + self.f.jp("速度", "speed") + "    M: " +
            self.f.jp("観戦⇔手動", "watch<->manual") + "    H: " + self.f.jp("ヘルプ", "help"),
        ]
        y = rect.y + 28
        for ln in mouse_lines:
            surf.blit(_render(self.f.label, ln, pal.UI_TEXT), (rect.x + 10, y))
            y += 14
        y += 6
        for ln in key_lines:
            surf.blit(_render(self.f.label, ln, pal.UI_TEXT_DIM), (rect.x + 10, y))
            y += 14
        y += 6
        surf.blit(_render(self.f.label,
                          self.f.jp("行灯のような小さな島。英雄が生き延びるのを見守る。",
                                    "A small lamplit island. Watch the hero try to survive."),
                          pal.UI_GOLD), (rect.x + 10, y))
        foot = _render(self.f.label, self.f.jp("外側クリック / Space / H で閉じる",
                                               "click outside / Space / H to close"), pal.UI_TEXT_DIM)
        surf.blit(foot, (rect.x + 10, rect.bottom - 14))

    def draw_diary(self, surf, sim, scroll: int) -> None:
        pg = self.pg
        margin = 24
        dim = pg.Surface((VIEW_W, VIEW_H), pg.SRCALPHA)
        dim.fill((0, 0, 0, 150))
        surf.blit(dim, (0, 0))
        rect = pg.Rect(margin, margin, VIEW_W - 2 * margin, VIEW_H - 2 * margin)
        pg.draw.rect(surf, pal.PARCHMENT, rect, border_radius=4)
        pg.draw.rect(surf, pal.PARCHMENT_LINE, rect, 2, border_radius=4)
        surf.blit(_render(self.f.body,
                          self.f.jp("日記 (ホイール/↑↓ でスクロール・外側クリックで閉じる)",
                                    "Diary (wheel/Up-Down to scroll, click outside to close)"),
                          pal.PARCHMENT_TEXT), (rect.x + 8, rect.y + 6))
        entries = sim.memory.diary
        # newest first; scroll skips from the top
        text_lines: list[str] = []
        for entry in reversed(entries):
            for raw in entry.text.splitlines():
                text_lines.extend(_wrap(raw, 60, 4) or [""])
            text_lines.append("")
        if not text_lines:
            text_lines = [self.f.jp("まだ日記はない。", "(No diary entries yet.)")]
        start = max(0, min(scroll, max(0, len(text_lines) - 1)))
        y = rect.y + 26
        for ln in text_lines[start:]:
            if y > rect.bottom - 14:
                break
            color = pal.PARCHMENT_TEXT if not ln.startswith("Day ") else pal.PARCHMENT_LINE
            surf.blit(_render(self.f.label, ln, color), (rect.x + 10, y))
            y += 12

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

    def draw_craft(self, surf, sim, selected: int, hover: int = -1) -> list:
        """Mouse-driven craft/build menu. Returns a list of (row_rect, recipe,
        affordable) so the app can hit-test clicks. Unaffordable rows are
        greyed with their missing materials in red."""
        from . import uihelp as uh

        pg = self.pg
        rect = self._panel(
            surf, self.f.jp("作る / 建てる  (行をクリック)", "Craft / Build  (click a row)")
        )
        rows = self.craft_rows(sim)
        hero = sim.hero
        row_h = 16
        y = rect.y + 26
        hits = []
        for i, (recipe, affordable) in enumerate(rows):
            owned = hero.has(recipe.key)
            row_rect = pg.Rect(rect.x + 6, y - 1, rect.width - 12, row_h)
            if (i == hover or i == selected) and affordable:
                pg.draw.rect(surf, pal.UI_PANEL_LIGHT, row_rect, border_radius=2)
            if owned:
                color = pal.STAT_COLORS["stamina"]
            elif affordable:
                color = pal.UI_TEXT
            else:
                color = pal.UI_TEXT_DIM
            name = uh.recipe_name(self.f, recipe)
            kind = self.f.jp("建てる", "build") if recipe.kind == "build" else self.f.jp("道具", "item")
            label = f"{name} ({kind})"
            surf.blit(_render(self.f.body, label, color), (row_rect.x + 6, y))
            # cost / status on the right
            if owned:
                tag = _render(self.f.label, self.f.jp("所持済", "owned"), pal.STAT_COLORS["stamina"])
                surf.blit(tag, (row_rect.right - tag.get_width() - 6, y + 1))
            else:
                # materials: those the hero lacks are drawn in red.
                lx = row_rect.x + 200
                parts = list(recipe.requires.items())
                if recipe.station and not hero.has(recipe.station):
                    sta = _render(self.f.label,
                                  self.f.jp(f"要:{uh.station_name(self.f, recipe.station)}",
                                            f"needs {recipe.station}"),
                                  pal.STAT_COLORS["hp"])
                    surf.blit(sta, (lx, y + 1))
                    lx += sta.get_width() + 8
                for item, amount in parts:
                    have = hero.item_count(item)
                    short = have < amount
                    col = pal.STAT_COLORS["hp"] if short else pal.UI_TEXT_DIM
                    txt = _render(self.f.label, f"{uh.mat_name(self.f, item)}x{amount}", col)
                    if lx + txt.get_width() > row_rect.right - 6:
                        break
                    surf.blit(txt, (lx, y + 1))
                    lx += txt.get_width() + 8
            hits.append((row_rect, recipe, affordable))
            y += row_h
        # close hint
        hint = _render(self.f.label,
                       self.f.jp("外側クリック か Space で閉じる", "click outside / Space to close"),
                       pal.UI_TEXT_DIM)
        surf.blit(hint, (rect.x + 10, rect.bottom - 14))
        return hits

    def draw_eat(self, surf, sim, hover: int = -1) -> list:
        """Eat popup: edible inventory as 'パン (満腹+45)' rows. Returns
        (row_rect, item) hit list. Centred panel."""
        from . import uihelp as uh
        from spl.core.crops import FOOD_VALUES

        pg = self.pg
        foods = uh.edible_items(sim)
        title = self.f.jp("食べる  (行をクリック)", "Eat  (click a row)")
        w = 240
        row_h = 18
        h = 40 + max(1, len(foods)) * row_h
        h = min(h, VIEW_H - 48)
        rx = (VIEW_W - w) // 2
        ry = (VIEW_H - h) // 2
        dim = pg.Surface((VIEW_W, VIEW_H), pg.SRCALPHA)
        dim.fill((0, 0, 0, 150))
        surf.blit(dim, (0, 0))
        rect = pg.Rect(rx, ry, w, h)
        pg.draw.rect(surf, pal.UI_PANEL, rect, border_radius=4)
        pg.draw.rect(surf, pal.UI_BORDER, rect, 1, border_radius=4)
        surf.blit(_render(self.f.body, title, pal.UI_GOLD), (rect.x + 8, rect.y + 6))
        y = rect.y + 26
        hits = []
        if not foods:
            surf.blit(_render(self.f.label, self.f.jp("食べられる物がない", "(nothing edible)"),
                              pal.UI_TEXT_DIM), (rect.x + 10, y))
        for i, (item, _value) in enumerate(foods):
            row_rect = pg.Rect(rect.x + 6, y - 1, rect.width - 12, row_h)
            if i == hover:
                pg.draw.rect(surf, pal.UI_PANEL_LIGHT, row_rect, border_radius=2)
            count = sim.hero.item_count(item)
            label = (f"{uh.food_name(self.f, item)} x{count}  "
                     f"({self.f.jp('満腹', 'Hun')}+{FOOD_VALUES[item]})")
            surf.blit(_render(self.f.body, label, pal.UI_TEXT), (row_rect.x + 6, y))
            hits.append((row_rect, item))
            y += row_h
        return hits

    def draw_heaven(self, surf, text: str, send_hover: bool = False):
        """Heaven's-voice text entry. Returns the [送る] button rect so the app
        can hit-test clicks (Enter still confirms)."""
        pg = self.pg
        rect = self._panel(surf, self.f.jp("天の声を入力", "Heaven's Voice"))
        surf.blit(
            _render(self.f.label,
                    self.f.jp("英雄に届く一言を授ける。Enter か [送る] で送信。",
                              "Whisper one line of guidance. Enter or [Send]."),
                    pal.UI_TEXT_DIM),
            (rect.x + 10, rect.y + 30),
        )
        box = pg.Rect(rect.x + 10, rect.y + 48, rect.width - 20, 20)
        pg.draw.rect(surf, pal.UI_PANEL_LIGHT, box)
        pg.draw.rect(surf, pal.UI_BORDER, box, 1)
        surf.blit(_render(self.f.body, text + "_", pal.UI_TEXT), (box.x + 4, box.y + 4))
        # [送る] button
        send = pg.Rect(rect.x + 10, box.bottom + 8, 72, 22)
        pg.draw.rect(surf, pal.UI_PANEL_LIGHT if send_hover else pal.UI_PANEL, send, border_radius=3)
        pg.draw.rect(surf, pal.UI_GOLD if send_hover else pal.UI_BORDER, send, 1, border_radius=3)
        slab = _render(self.f.body, self.f.jp("送る", "Send"), pal.UI_TEXT)
        surf.blit(slab, (send.centerx - slab.get_width() // 2, send.centery - slab.get_height() // 2))
        foot = _render(self.f.label, self.f.jp("Esc で閉じる", "Esc to close"), pal.UI_TEXT_DIM)
        surf.blit(foot, (rect.x + 10, rect.bottom - 14))
        return send

    def draw_result(self, surf, sim, hover: str = ""):
        """End-of-run panel. Returns {"again": rect, "quit": rect} so the app
        can hit-test the [もう一度（同じ島）] / [終了] buttons."""
        pg = self.pg
        rect = self._panel(surf, self.f.jp("結果", "Result"), margin=18)
        hero = sim.hero
        y = rect.y + 28
        info = [
            sim.result_reason or (self.f.jp("生存中。", "Still alive.") if hero.alive
                                  else self.f.jp("英雄は倒れた。", "The hero fell.")),
            f"{self.f.jp('得点', 'Score')}: {sim.score()}",
            f"{self.f.jp('生存日数', 'Days')}: {hero.days_survived}",
            f"{self.f.jp('混乱', 'Confusions')}: {hero.confusion_count}",
            f"{self.f.jp('文明点', 'Civ pts')}: {hero.civilization_points()}",
            "",
            self.f.jp("迷言ベスト5:", "Best lines:"),
        ]
        for ln in info:
            surf.blit(_render(self.f.body, ln, pal.UI_TEXT), (rect.x + 10, y))
            y += 15
        best = select_meigen(hero.spoken_lines, 5)
        if not best:
            best = [self.f.jp("(英雄は黙したまま)", "(the hero kept quiet)")]
        for line in best:
            for wln in _wrap(f'"{line}"', 60, 2):
                surf.blit(_render(self.f.label, wln, pal.UI_GOLD), (rect.x + 16, y))
                y += 12

        # Buttons along the bottom.
        again = pg.Rect(rect.x + 10, rect.bottom - 30, 168, 24)
        quit_b = pg.Rect(again.right + 10, rect.bottom - 30, 90, 24)
        for key, b, label in (
            ("again", again, self.f.jp("もう一度（同じ島）", "Play again (same island)")),
            ("quit", quit_b, self.f.jp("終了", "Quit")),
        ):
            hot = hover == key
            pg.draw.rect(surf, pal.UI_PANEL_LIGHT if hot else pal.UI_PANEL, b, border_radius=3)
            pg.draw.rect(surf, pal.UI_GOLD if hot else pal.UI_BORDER, b, 1, border_radius=3)
            lab = _render(self.f.body, label, pal.UI_TEXT)
            surf.blit(lab, (b.centerx - lab.get_width() // 2, b.centery - lab.get_height() // 2))
        return {"again": again, "quit": quit_b}

    # -- map hover: highlight diamond + tooltip ------------------------------
    def draw_tile_highlight(self, surf, cx: int, cy: int, color=None) -> None:
        """Bright diamond outline centred on a tile (cx, cy = tile centre)."""
        from .iso import HALF_H, HALF_W

        pg = self.pg
        col = color or pal.HILITE
        pts = [(cx, cy - HALF_H), (cx + HALF_W, cy), (cx, cy + HALF_H), (cx - HALF_W, cy)]
        pg.draw.polygon(surf, col, pts, 2)

    def draw_walk_marker(self, surf, cx: int, cy: int, phase: float) -> None:
        """Pulsing diamond marker on an auto-walk target. ``phase`` 0..1."""
        from .iso import HALF_H, HALF_W

        pg = self.pg
        import math

        k = 0.5 + 0.5 * math.sin(phase * 2 * math.pi)
        w = int(HALF_W * (0.6 + 0.4 * k))
        h = int(HALF_H * (0.6 + 0.4 * k))
        pts = [(cx, cy - h), (cx + w, cy), (cx, cy + h), (cx - w, cy)]
        col = (255, 240, 150)
        pg.draw.polygon(surf, col, pts, 2)
        pg.draw.polygon(surf, (255, 255, 220), [(cx, cy - 2), (cx + 2, cy), (cx, cy + 2), (cx - 2, cy)])

    def draw_hero_locator(self, surf, cx: int, cy: int, phase: float) -> None:
        """First-seconds pulsing white outline so the player finds the hero."""
        from .iso import HALF_H, HALF_W

        pg = self.pg
        import math

        k = 0.5 + 0.5 * math.sin(phase * 2 * math.pi)
        r = int(2 + 4 * k)
        pts = [(cx, cy - HALF_H - r), (cx + HALF_W + r, cy),
               (cx, cy + HALF_H + r), (cx - HALF_W - r, cy)]
        pg.draw.polygon(surf, (255, 255, 255), pts, 2)

    def draw_tooltip(self, surf, mx: int, my: int, lines: list) -> None:
        """Small panel near the cursor with up to a few short info lines."""
        pg = self.pg
        lines = [ln for ln in lines if ln]
        if not lines:
            return
        widths = [self.f.label.size(ln)[0] for ln in lines]
        w = max(widths) + 12
        lh = self.f.label.get_height() + 1
        h = lh * len(lines) + 8
        # place to the lower-right of the cursor, clamped to screen
        bx = mx + 14
        by = my + 10
        if bx + w > VIEW_W - 2:
            bx = mx - w - 8
        if by + h > VIEW_H - 2:
            by = my - h - 8
        bx = max(2, min(VIEW_W - w - 2, bx))
        by = max(2, min(VIEW_H - h - 2, by))
        panel = pg.Surface((w, h), pg.SRCALPHA)
        panel.fill((20, 18, 30, 224))
        surf.blit(panel, (bx, by))
        pg.draw.rect(surf, pal.UI_BORDER, (bx, by, w, h), 1)
        for i, ln in enumerate(lines):
            color = pal.UI_GOLD if i == 0 else pal.UI_TEXT
            surf.blit(_render(self.f.label, ln, color), (bx + 6, by + 4 + i * lh))

    # -- click-to-act popup --------------------------------------------------
    def draw_popup(self, surf, header: str, items: list, anchor: tuple,
                   hover: int = -1):
        """Pixel-art context menu. ``items`` is a list of uihelp.MenuItem.
        Returns a list of row rects (parallel to ``items``) for hit-testing.
        Anchored near ``anchor`` (the click point) and clamped to screen."""
        pg = self.pg
        row_h = 17
        pad = 6
        head_h = 18
        # width from the widest of header / rows
        labels = [it.label + ("" if it.cost is None else f"  (AP{it.cost})") for it in items]
        widths = [self.f.body.size(header)[0]] + [self.f.body.size(s)[0] for s in labels]
        w = max(widths) + pad * 2 + 10
        w = max(w, 132)
        h = head_h + row_h * len(items) + pad
        ax, ay = anchor
        bx = ax + 6
        by = ay + 6
        if bx + w > VIEW_W - 2:
            bx = ax - w - 6
        if by + h > VIEW_H - 2:
            by = VIEW_H - h - 2
        bx = max(2, min(VIEW_W - w - 2, bx))
        by = max(2, min(VIEW_H - h - 2, by))
        rect = pg.Rect(bx, by, w, h)
        pg.draw.rect(surf, pal.UI_PANEL, rect, border_radius=4)
        pg.draw.rect(surf, pal.UI_GOLD, rect, 1, border_radius=4)
        # header
        surf.blit(_render(self.f.body, header, pal.UI_GOLD), (bx + pad, by + 3))
        pg.draw.line(surf, pal.UI_BORDER, (bx + 3, by + head_h - 1),
                     (bx + w - 3, by + head_h - 1), 1)
        rects = []
        y = by + head_h
        for i, it in enumerate(items):
            row_rect = pg.Rect(bx + 3, y, w - 6, row_h)
            if i == hover and it.enabled:
                pg.draw.rect(surf, pal.UI_PANEL_LIGHT, row_rect, border_radius=2)
            color = pal.UI_TEXT if it.enabled else pal.UI_TEXT_DIM
            surf.blit(_render(self.f.body, it.label, color), (row_rect.x + pad, y + 2))
            if it.cost is not None:
                cost = _render(self.f.label, f"(AP{it.cost})", pal.UI_TEXT_DIM)
                surf.blit(cost, (row_rect.right - cost.get_width() - 6, y + 3))
            rects.append(row_rect)
            y += row_h
        return rects

    # -- HUD button bar ------------------------------------------------------
    def draw_button_bar(self, surf, buttons: list, hover_key: str | None):
        """Draw the clickable HUD buttons (a list of uihelp.Button with .rect
        already laid out). Hovered button gets a highlight + a tooltip above."""
        pg = self.pg
        tip = None
        for b in buttons:
            r = b.rect
            hot = (b.key == hover_key) and b.enabled
            bg = pal.UI_PANEL_LIGHT if hot else pal.UI_PANEL
            border = pal.UI_GOLD if hot else pal.UI_BORDER
            pg.draw.rect(surf, bg, r, border_radius=3)
            pg.draw.rect(surf, border, r, 1, border_radius=3)
            color = pal.UI_TEXT if b.enabled else pal.UI_TEXT_DIM
            lab = _render(self.f.label, b.label, color)
            surf.blit(lab, (r.centerx - lab.get_width() // 2, r.centery - lab.get_height() // 2))
            if hot and b.tooltip:
                tip = (b, b.tooltip)
        if tip is not None:
            b, text = tip
            tw = self.f.label.size(text)[0] + 10
            tx = b.rect.centerx - tw // 2
            tx = max(2, min(VIEW_W - tw - 2, tx))
            ty = b.rect.y - 16
            panel = pg.Surface((tw, 14), pg.SRCALPHA)
            panel.fill((20, 18, 30, 230))
            surf.blit(panel, (tx, ty))
            pg.draw.rect(surf, pal.UI_BORDER, (tx, ty, tw, 14), 1)
            surf.blit(_render(self.f.label, text, pal.UI_TEXT), (tx + 5, ty + 2))

    # -- guide strip ---------------------------------------------------------
    def draw_guide(self, surf, text: str) -> None:
        """Semi-transparent dark banner with small JP guidance text, top of the
        map area."""
        if not text:
            return
        pg = self.pg
        tw = self.f.label.size(text)[0]
        bar_h = 16
        band = pg.Surface((VIEW_W, bar_h), pg.SRCALPHA)
        band.fill((12, 10, 20, 150))
        surf.blit(band, (0, 0))
        surf.blit(_render(self.f.label, text, pal.UI_TEXT), ((VIEW_W - tw) // 2, 2))
