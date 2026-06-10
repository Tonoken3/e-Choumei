from __future__ import annotations

"""PixelApp: the 60fps game loop, map renderer, atmosphere, and controls.

The renderer reads the Simulation but only mutates it via ``sim.step(...)`` on
the main thread. An optional LLM brain runs in a worker thread so the render
loop never blocks; its chosen action is applied on the main thread when the
thread returns. Particles use the ``random`` module freely — they are pure
render decoration and never touch the deterministic sim RNG.
"""

import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from spl.agent.policy import LocalPolicyAgent
from spl.core.actions import GameAction
from spl.core.crops import FOOD_VALUES
from spl.core.hero import Position
from spl.core.sim import PROJECT_ROOT, Simulation

from . import iso
from . import palette as pal
from . import uihelp as uh
from .hud import BUTTON_BAR_H, HUD_H, MAP_H, VIEW_H, VIEW_W, Fonts, Hud, Overlays
from .sprites import SpriteFactory

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pygame

SCALE = 2
WIN_W = VIEW_W * SCALE
WIN_H = VIEW_H * SCALE

SPEED_DELAYS = {1: 0.6, 2: 0.18, 3: 0.03}

# Manual-mode key -> simple action (no args)
_SIMPLE_KEYS = {
    "x": "chop", "v": "mine", "q": "drink", "r": "rest", "z": "sleep",
}


@dataclass
class Particles:
    """Render-only weather particles, recycled. Uses the stdlib random module
    (never the sim RNG)."""

    rng: random.Random = field(default_factory=lambda: random.Random(1234))
    rain: list[list[float]] = field(default_factory=list)
    snow: list[list[float]] = field(default_factory=list)
    flash_timer: float = 0.0

    def ensure_rain(self, n: int) -> None:
        while len(self.rain) < n:
            self.rain.append([self.rng.uniform(0, VIEW_W), self.rng.uniform(0, MAP_H)])

    def ensure_snow(self, n: int) -> None:
        while len(self.snow) < n:
            self.snow.append([self.rng.uniform(0, VIEW_W), self.rng.uniform(0, MAP_H),
                              self.rng.uniform(-0.3, 0.3)])

    def update_rain(self, dt: float, speed: float) -> None:
        self.ensure_rain(80)
        for p in self.rain:
            p[1] += speed * dt
            p[0] += speed * 0.3 * dt
            if p[1] > MAP_H:
                p[0] = self.rng.uniform(0, VIEW_W)
                p[1] = -4

    def update_snow(self, dt: float) -> None:
        self.ensure_snow(60)
        for p in self.snow:
            p[1] += 22 * dt
            p[0] += p[2] * 14 * dt
            if p[1] > MAP_H:
                p[0] = self.rng.uniform(0, VIEW_W)
                p[1] = -4


class PixelApp:
    def __init__(self, args: object, headless: bool = False) -> None:
        import pygame

        self.pg = pygame
        # pygame.init() is idempotent; do it here so font/display are ready
        # before we build Fonts/Surfaces (run_window/run_shots call it again,
        # which is harmless).
        pygame.init()
        self.headless = headless
        self.args = args
        self.sim = Simulation(seed=getattr(args, "seed", 42), max_days=getattr(args, "days", 112))
        self.local_agent = LocalPolicyAgent()

        # optional fast-forward for verification of later seasons
        start_day = int(getattr(args, "start_day", 0) or 0)
        if start_day > 1:
            self._fast_forward(start_day)

        self.factory = SpriteFactory(pygame)
        self.fonts = Fonts(pygame)
        self.hud = Hud(pygame, self.fonts)
        self.overlays = Overlays(pygame, self.fonts)
        self.particles = Particles()

        self.offset_x, self.offset_y = iso.centering_offset(
            self.sim.world.width, self.sim.world.height, VIEW_W, MAP_H
        )

        # modes / state
        self.manual = bool(getattr(args, "manual", False))
        self.speed = int(getattr(args, "speed", 2) or 2)
        self.paused = False
        self.overlay: str | None = None  # diary | help | craft | heaven | eat
        self.diary_scroll = 0
        self.craft_sel = 0
        self.heaven_text = ""
        self.running = True

        # -- mouse / click-to-act state --------------------------------------
        self.mouse_pos = (0, 0)            # internal-surface coords
        self.hover_tile: Position | None = None
        # popup: dict(kind=tile|trade, tile, header, items, anchor, hover, rects)
        self.popup: dict | None = None
        # transient hit-rects produced by the last render of overlays/popups;
        # the next frame's click handler reads these (immediate-mode style).
        self._hits: dict[str, object] = {}
        self._overlay_hover = -1            # hovered craft/eat row index
        self.button_hover: str | None = None
        # auto-walk
        self.walk_target: Position | None = None
        self._last_walk_t = 0.0
        # first-seconds hero locator + guide
        self._session_start = 0.0
        self.buttons = self._build_buttons()

        # animation / timing
        self.anim_t = 0.0
        self.last_action_t = 0.0
        self.fade = 0.0  # 0..1 black fade for day change
        self._last_day = self.sim.world.day

        # speech bubble tracking
        self._spoken_index = len(self.sim.hero.spoken_lines)
        self._bubble_text = ""
        self._bubble_until = 0.0

        # LLM brain in a worker thread
        self.llm_enabled = bool(getattr(args, "llm", False))
        self.brain = self._make_brain() if self.llm_enabled else None
        if self.brain is not None:
            self.sim.set_diarist(self.brain)
        else:
            self.llm_enabled = False
        self._pending_action: GameAction | None = None
        self._thread_busy = False
        self._thread_lock = threading.Lock()

    # -- brain ---------------------------------------------------------------
    def _make_brain(self):
        from spl.agent.llm_client import OpenAICompatibleBrain, find_cassette

        try:
            cassette = find_cassette(
                PROJECT_ROOT / "config" / "models.toml", getattr(self.args, "cassette", None)
            )
        except Exception:  # noqa: BLE001
            return None
        if not cassette.base_url:
            return None
        return OpenAICompatibleBrain(cassette)

    def _fast_forward(self, target_day: int) -> None:
        agent = LocalPolicyAgent()
        guard = 0
        while not self.sim.done and self.sim.world.day < target_day and guard < target_day * 60:
            self.sim.step(agent.choose(self.sim), confuse_on_invalid=False)
            guard += 1

    # -- HUD button bar layout ----------------------------------------------
    def _build_buttons(self) -> list:
        """Lay out the HUD button bar once. Rects are in internal coords along
        the top BUTTON_BAR_H strip of the HUD panel. Labels with a ⇔/flip state
        are refreshed each frame in ``_refresh_buttons``."""
        f = self.fonts
        specs = [
            ("pause", f.jp("一時停止", "Pause"), f.jp("自動進行を止める/再開", "pause/resume")),
            ("speed", f.jp("速度:普", "Speed:Nrm"), f.jp("観戦の速さを変える", "cycle watch speed")),
            ("mode", f.jp("観戦⇔手動", "Watch/Manual"), f.jp("見守る/自分で操作", "watch vs play")),
            ("diary", f.jp("日記", "Diary"), f.jp("英雄の日記を読む", "read the diary")),
            ("craft", f.jp("作る", "Craft"), f.jp("道具や建物を作る", "craft & build")),
            ("eat", f.jp("食べる", "Eat"), f.jp("持ち物を食べる", "eat from inventory")),
            ("heaven", f.jp("天の声", "Heaven"), f.jp("英雄に助言する", "advise the hero")),
            ("help", f.jp("ヘルプ", "Help"), f.jp("操作の説明", "how to play")),
        ]
        x = 6
        y = MAP_H + 2
        h = BUTTON_BAR_H - 4
        gap = 4
        buttons = []
        for key, label, tip in specs:
            w = f.label.size(label)[0] + 14
            rect = self.pg.Rect(x, y, w, h)
            buttons.append(uh.Button(key, label, tip, rect=rect))
            x += w + gap
        return buttons

    def _refresh_buttons(self) -> None:
        """Update dynamic labels (pause/speed) and enabled state per mode."""
        f = self.fonts
        speed_jp = {1: "遅", 2: "普", 3: "速"}[self.speed]
        speed_en = {1: "Slow", 2: "Nrm", 3: "Fast"}[self.speed]
        for b in self.buttons:
            if b.key == "pause":
                b.label = f.jp("再開", "Resume") if self.paused else f.jp("一時停止", "Pause")
            elif b.key == "speed":
                b.label = f.jp(f"速度:{speed_jp}", f"Speed:{speed_en}")
            # In watch mode, eat/craft belong to manual play — grey them out.
            if b.key in {"craft", "eat"}:
                b.enabled = self.manual
        # recompute widths for the two relabelled buttons, keep bar left-aligned
        x = 6
        for b in self.buttons:
            w = f.label.size(b.label)[0] + 14
            b.rect.width = w
            b.rect.x = x
            x += w + 4

    # -- replay (result panel) ----------------------------------------------
    def _rebuild_sim(self) -> None:
        """Fresh Simulation on the same seed/args (the [もう一度] button)."""
        self.sim = Simulation(
            seed=getattr(self.args, "seed", 42),
            max_days=getattr(self.args, "days", 112),
        )
        if self.brain is not None:
            self.sim.set_diarist(self.brain)
        self.offset_x, self.offset_y = iso.centering_offset(
            self.sim.world.width, self.sim.world.height, VIEW_W, MAP_H
        )
        self.popup = None
        self.walk_target = None
        self.overlay = None
        self.fade = 0.0
        self._last_day = self.sim.world.day
        self._spoken_index = len(self.sim.hero.spoken_lines)
        self._bubble_text = ""
        self._bubble_until = 0.0
        self._session_start = self.anim_t  # re-trigger the hero locator

    # -- mouse helpers -------------------------------------------------------
    def _to_internal(self, wx: int, wy: int) -> tuple[int, int]:
        return wx // SCALE, wy // SCALE

    def _tile_under(self, ix: int, iy: int) -> Position | None:
        if iy >= MAP_H:
            return None
        tile = iso.screen_to_tile(ix, iy, self.offset_x, self.offset_y)
        if not self.sim.world.in_bounds(tile):
            return None
        return tile

    # -- mouse: motion -------------------------------------------------------
    def _handle_motion(self, wx: int, wy: int) -> None:
        ix, iy = self._to_internal(wx, wy)
        self.mouse_pos = (ix, iy)
        self.hover_tile = self._tile_under(ix, iy)
        self.button_hover = self._button_at(ix, iy)
        # popup row hover
        if self.popup is not None:
            self.popup["hover"] = -1
            for i, r in enumerate(self.popup.get("rects", [])):
                if r.collidepoint(ix, iy):
                    self.popup["hover"] = i
                    break
        # overlay row hover (craft / eat) stored as a simple index
        self._overlay_hover = -1
        rows = self._hits.get("rows")
        if rows:
            for i, entry in enumerate(rows):
                r = entry[0]
                if r.collidepoint(ix, iy):
                    self._overlay_hover = i
                    break

    def _button_at(self, ix: int, iy: int) -> str | None:
        for b in self.buttons:
            if b.enabled and b.rect.collidepoint(ix, iy):
                return b.key
        return None

    # -- mouse: click --------------------------------------------------------
    def _handle_click(self, wx: int, wy: int) -> None:
        ix, iy = self._to_internal(wx, wy)
        # 1) overlays consume clicks first (result panel always active when done)
        if self.sim.done:
            self._click_result(ix, iy)
            return
        if self.overlay is not None:
            self._click_overlay(ix, iy)
            return
        # 2) auto-walking: any click interrupts (spec: "クリックで中断")
        if self.walk_target is not None:
            self.walk_target = None
            return
        # 3) open popup? click on its row?
        if self.popup is not None:
            self._click_popup(ix, iy)
            return
        # 4) HUD button bar
        btn = self._button_at(ix, iy)
        if btn is not None:
            self._activate_button(btn)
            return
        # 5) a map tile -> open the context popup
        if iy < MAP_H:
            tile = self._tile_under(ix, iy)
            if tile is not None:
                self._open_tile_popup(tile, (ix, iy))

    def _open_tile_popup(self, tile: Position, anchor: tuple) -> None:
        f = self.fonts
        header = self._tile_header(tile)
        if not self.manual:
            # WATCH: info-only popup with a single "switch to manual" button.
            items = [uh.MenuItem(f.jp("手動モードに切り替えて操作", "Switch to manual to act"),
                                 "switch_mode")]
            self.popup = {"kind": "tile", "tile": tile, "header": header,
                          "items": items, "anchor": anchor, "hover": -1, "rects": [],
                          "info_only": True}
            return
        items = uh.build_tile_menu(f, self.sim, tile)
        items.append(uh.MenuItem(f.jp("閉じる", "Close"), "close"))
        self.popup = {"kind": "tile", "tile": tile, "header": header,
                      "items": items, "anchor": anchor, "hover": -1, "rects": [],
                      "info_only": False}

    def _tile_header(self, tile: Position) -> str:
        f = self.fonts
        name = uh.tile_name(f, self.sim.world.tile_at(tile))
        crop = uh.crop_tooltip(f, self.sim, tile)
        if crop:
            return f"{name} — {crop}"
        if tile == self.sim.hero.pos:
            return f"{name}（" + f.jp("現在地", "you") + "）"
        return name

    def _click_popup(self, ix: int, iy: int) -> None:
        rects = self.popup.get("rects", [])
        for i, r in enumerate(rects):
            if r.collidepoint(ix, iy):
                self._activate_menu_item(self.popup["items"][i])
                return
        # clicked outside the popup -> close
        self.popup = None

    def _activate_menu_item(self, item: "uh.MenuItem") -> None:
        if not item.enabled:
            return
        verb = item.action
        if verb == "close":
            self.popup = None
            return
        if verb == "switch_mode":
            self.manual = True
            self.popup = None
            return
        if verb == "submenu" and item.args.get("kind") == "trade":
            f = self.fonts
            items = uh.build_trade_menu(f, self.sim)
            items.append(uh.MenuItem(f.jp("閉じる", "Close"), "close"))
            offer = self.sim.current_offer
            header = (f.jp("商人", "Merchant") + ": "
                      + (offer.describe() if offer is not None else ""))
            self.popup = {"kind": "trade", "tile": self.sim.hero.pos, "header": header,
                          "items": items, "anchor": self.popup["anchor"],
                          "hover": -1, "rects": [], "info_only": False}
            return
        if verb == "move_to":
            # start an auto-walk to the chosen tile
            self.walk_target = Position(int(item.args["x"]), int(item.args["y"]))
            self._last_walk_t = self.anim_t
            self.popup = None
            return
        # a direct GameAction
        self.sim.step(GameAction.safe(verb, **item.args), confuse_on_invalid=False)
        self.popup = None

    def _activate_button(self, key: str) -> None:
        if key == "pause":
            self.paused = not self.paused
        elif key == "speed":
            self.speed = self.speed % 3 + 1
        elif key == "mode":
            self.manual = not self.manual
            self.popup = None
            self.walk_target = None
        elif key == "diary":
            self.overlay = "diary"
            self.diary_scroll = 0
        elif key == "craft" and self.manual:
            self.overlay = "craft"
            self.craft_sel = 0
        elif key == "eat" and self.manual:
            self.overlay = "eat"
        elif key == "heaven":
            self.overlay = "heaven"
            self.heaven_text = self.sim.advice_from_heaven or ""
        elif key == "help":
            self.overlay = "help"

    def _click_overlay(self, ix: int, iy: int) -> None:
        from spl.core.actions import GameAction as _GA

        if self.overlay == "craft":
            rows = self._hits.get("rows") or []
            for (r, recipe, affordable) in rows:
                if r.collidepoint(ix, iy):
                    if affordable:
                        action = "build" if recipe.kind == "build" else "craft"
                        self.sim.step(_GA.safe(action, recipe=recipe.key),
                                      confuse_on_invalid=False)
                    return
            self.overlay = None  # clicked outside any row -> close
            return
        if self.overlay == "eat":
            rows = self._hits.get("rows") or []
            for (r, item) in rows:
                if r.collidepoint(ix, iy):
                    self.sim.step(_GA.safe("eat", item=item), confuse_on_invalid=False)
                    self.overlay = None
                    return
            self.overlay = None
            return
        if self.overlay == "heaven":
            send = self._hits.get("heaven_send")
            if send is not None and send.collidepoint(ix, iy):
                self.sim.advice_from_heaven = self.heaven_text.strip() or None
                self.overlay = None
            return
        # diary / help: any click closes
        if self.overlay in {"diary", "help"}:
            self.overlay = None

    def _click_result(self, ix: int, iy: int) -> None:
        rects = self._hits.get("result") or {}
        again = rects.get("again")
        quit_b = rects.get("quit")
        if again is not None and again.collidepoint(ix, iy):
            self._rebuild_sim()
        elif quit_b is not None and quit_b.collidepoint(ix, iy):
            self.running = False

    # -- auto-walk -----------------------------------------------------------
    def _walk_step(self, now: float) -> None:
        """Issue one move-toward-target every ~0.12s while auto-walking; stop on
        arrival, a failed move (AP out), or sim done. A click/key clears the
        target elsewhere."""
        if self.walk_target is None or self.sim.done:
            return
        if self.sim.hero.pos == self.walk_target:
            self.walk_target = None
            return
        if now - self._last_walk_t < 0.12:
            return
        self._last_walk_t = now
        result = self.sim.step(
            GameAction.safe("move", x=self.walk_target.x, y=self.walk_target.y),
            confuse_on_invalid=False,
        )
        if not result.ok or self.sim.hero.pos == self.walk_target:
            self.walk_target = None

    # -- watch-mode stepping -------------------------------------------------
    def _llm_worker(self) -> None:
        try:
            action = self.brain.choose(self.sim)
        except Exception as exc:  # noqa: BLE001 - mirror cli.choose_action fallback
            self.sim.log(f"LLM unavailable, local policy takes over this turn: {exc}")
            action = self.local_agent.choose(self.sim)
        with self._thread_lock:
            self._pending_action = action
            self._thread_busy = False

    def _watch_step(self, now: float) -> None:
        if self.sim.done:
            return
        if self.llm_enabled and self.brain is not None:
            # apply a finished request, then maybe launch a new one on the timer
            with self._thread_lock:
                pending = self._pending_action
                self._pending_action = None
                busy = self._thread_busy
            if pending is not None:
                self.sim.step(pending, confuse_on_invalid=True)
                self.last_action_t = now
                return
            if not busy and now - self.last_action_t >= SPEED_DELAYS[self.speed]:
                with self._thread_lock:
                    self._thread_busy = True
                threading.Thread(target=self._llm_worker, daemon=True).start()
            return
        if now - self.last_action_t >= SPEED_DELAYS[self.speed]:
            self.sim.step(self.local_agent.choose(self.sim), confuse_on_invalid=False)
            self.last_action_t = now

    # -- manual-mode context action ------------------------------------------
    def _context_action(self) -> GameAction:
        sim = self.sim
        hero, world = sim.hero, sim.world
        pos = hero.pos
        if sim.current_offer is not None:
            return GameAction.safe("trade_accept", id=sim.current_offer.id)
        plot = world.plots.get(pos)
        if plot is not None and plot.ready:
            return GameAction.safe("harvest")
        if plot is not None and not plot.ready:
            crop = sim.crop_book.get(plot.crop)
            if crop.needs_water and world.weather != "rain":
                return GameAction.safe("water")
        tile = world.tile_at(pos)
        if tile == "field" and pos not in world.plots:
            seed_crop = self._best_seed_in_season()
            if seed_crop is not None:
                return GameAction.safe("plant", crop=seed_crop)
        if tile in {"grass", "beach"}:
            return GameAction.safe("till")
        if world.is_near(pos, "water"):
            return GameAction.safe("fish")
        if tile == "forest" or world.is_near(pos, "forest"):
            return GameAction.safe("forage")
        return GameAction.safe("rest")

    def _best_seed_in_season(self) -> str | None:
        sim = self.sim
        choices = [c for c in sim.crop_book.seasonal(sim.world.season) if sim.hero.has(c.seed)]
        return choices[0].key if choices else None

    def _best_food(self) -> str | None:
        hero = self.sim.hero
        foods = [(i, FOOD_VALUES[i]) for i, a in hero.inventory.items() if a > 0 and i in FOOD_VALUES]
        if not foods:
            return None
        foods.sort(key=lambda p: p[1], reverse=True)
        return foods[0][0]

    # -- events --------------------------------------------------------------
    def _handle_key(self, key) -> None:
        pg = self.pg

        # Result screen: Space/Enter -> replay same island, Esc/Q -> quit.
        if self.sim.done:
            if key in (pg.K_SPACE, pg.K_RETURN):
                self._rebuild_sim()
            elif key in (pg.K_ESCAPE, pg.K_q):
                self.running = False
            return

        # heaven's-voice text input captures most keys
        if self.overlay == "heaven":
            if key == pg.K_RETURN:
                self.sim.advice_from_heaven = self.heaven_text.strip() or None
                self.overlay = None
            elif key == pg.K_ESCAPE:
                self.overlay = None
            elif key == pg.K_BACKSPACE:
                self.heaven_text = self.heaven_text[:-1]
            else:
                ch = getattr(self, "_last_unicode", "")
                if ch and ch.isprintable() and len(self.heaven_text) < 60:
                    self.heaven_text += ch
            return

        # Click-to-act popup: Space/Esc close, Enter activates the first row,
        # Up/Down navigate.
        if self.popup is not None:
            items = self.popup["items"]
            if key in (pg.K_ESCAPE, pg.K_SPACE):
                self.popup = None
            elif key == pg.K_RETURN:
                idx = self.popup.get("hover", -1)
                if idx < 0:
                    idx = 0
                if items:
                    self._activate_menu_item(items[idx])
            elif key in (pg.K_DOWN, pg.K_s):
                self.popup["hover"] = (self.popup.get("hover", -1) + 1) % len(items)
            elif key in (pg.K_UP, pg.K_w):
                self.popup["hover"] = (self.popup.get("hover", 0) - 1) % len(items)
            return

        if key == pg.K_ESCAPE:
            if self.overlay is not None:
                self.overlay = None
            else:
                self.running = False
            return

        if self.overlay == "diary":
            if key in (pg.K_UP, pg.K_w):
                self.diary_scroll = max(0, self.diary_scroll - 1)
            elif key in (pg.K_DOWN, pg.K_s):
                self.diary_scroll += 1
            elif key in (pg.K_d, pg.K_h, pg.K_SPACE):
                self.overlay = None
            return
        if self.overlay == "craft":
            rows = self.overlays.craft_rows(self.sim)
            if key in (pg.K_UP, pg.K_w):
                self.craft_sel = (self.craft_sel - 1) % len(rows)
            elif key in (pg.K_DOWN, pg.K_s):
                self.craft_sel = (self.craft_sel + 1) % len(rows)
            elif key == pg.K_RETURN:
                self._do_craft_selected(rows)
            elif key in (pg.K_c, pg.K_h, pg.K_SPACE):
                self.overlay = None
            return
        if self.overlay == "eat":
            if key == pg.K_RETURN:
                food = self._best_food()
                if food:
                    self.sim.step(GameAction.safe("eat", item=food), confuse_on_invalid=False)
                self.overlay = None
            elif key in (pg.K_SPACE,):
                self.overlay = None
            return
        if self.overlay == "help":
            if key in (pg.K_h, pg.K_SPACE):
                self.overlay = None
            return

        # no overlay: global + mode keys
        if key == pg.K_h:
            self.overlay = "help"
            return
        if key == pg.K_d:
            self.overlay = "diary"
            self.diary_scroll = 0
            return
        if key == pg.K_c:
            self.overlay = "craft"
            self.craft_sel = 0
            return
        if key == pg.K_t:
            self.overlay = "heaven"
            self.heaven_text = self.sim.advice_from_heaven or ""
            return
        if key == pg.K_m:
            self.manual = not self.manual
            return
        if key in (pg.K_1, pg.K_2, pg.K_3):
            self.speed = {pg.K_1: 1, pg.K_2: 2, pg.K_3: 3}[key]
            return
        if key == pg.K_SPACE:
            # Space pauses the watch loop; in manual it is a no-op toggle that
            # still feels consistent (nothing auto-steps in manual).
            self.paused = not self.paused
            return
        if self.manual:
            self._handle_manual_key(key)

    def _handle_manual_key(self, key) -> None:
        if self.sim.done:
            return
        pg = self.pg
        # Arrows + WASD move one tile. (D/H/C/T/M etc. are intercepted earlier
        # in _handle_key, so they never reach here as movement.)
        move_map = {
            pg.K_UP: "north", pg.K_w: "north",
            pg.K_DOWN: "south", pg.K_s: "south",
            pg.K_LEFT: "west", pg.K_a: "west",
            pg.K_RIGHT: "east",
        }
        if key in move_map:
            self.sim.step(GameAction.safe("move", direction=move_map[key]), confuse_on_invalid=False)
            return
        if key == pg.K_e:
            self.sim.step(self._context_action(), confuse_on_invalid=False)
            return
        if key == pg.K_o:
            food = self._best_food()
            if food:
                self.sim.step(GameAction.safe("eat", item=food), confuse_on_invalid=False)
            return
        char_action = _SIMPLE_KEYS.get(self.pg.key.name(key))
        if char_action:
            self.sim.step(GameAction.safe(char_action), confuse_on_invalid=False)

    def _do_craft_selected(self, rows) -> None:
        recipe, affordable = rows[self.craft_sel]
        if not affordable:
            return
        action = "build" if recipe.kind == "build" else "craft"
        self.sim.step(GameAction.safe(action, recipe=recipe.key), confuse_on_invalid=False)

    # -- rendering -----------------------------------------------------------
    def _frame_index(self) -> int:
        return int(self.anim_t * 2) % 2  # ~2 fps idle bob / shimmer

    def render(self, surf) -> None:
        pg = self.pg
        world = self.sim.world
        season = world.season
        frame = self._frame_index()
        surf.fill(pal.UI_BG)
        # sky/sea backdrop for the map area
        pg.draw.rect(surf, (28, 40, 64) if season != "winter" else (40, 48, 64),
                     (0, 0, VIEW_W, MAP_H))

        hero = self.sim.hero
        # painter's order: draw ground, then objects/crops/hero per cell.
        for (x, y) in iso.painter_order(world.width, world.height):
            tile = world.tiles[y][x]
            base = "forest" if tile == "forest" else tile
            ground = self.factory.ground(base, season)
            sx, sy = iso.tile_to_screen(x, y, self.offset_x, self.offset_y)
            surf.blit(ground, (sx, sy))
            if tile == "water":
                surf.blit(self.factory.water_overlay(season, frame), (sx, sy))
            cx, cy = iso.tile_center(x, y, self.offset_x, self.offset_y)
            self._draw_cell_objects(surf, x, y, tile, season, frame, cx, cy)

        self._draw_atmosphere(surf)
        self._draw_map_cursor(surf)
        self._refresh_buttons()
        self.hud.draw(surf, self.sim, self.sim.score())
        self.overlays.draw_button_bar(surf, self.buttons, self.button_hover)
        self._draw_guide(surf)
        self._draw_bubbles(surf)
        self._draw_overlay(surf)

    def _draw_cell_objects(self, surf, x, y, tile, season, frame, cx, cy) -> None:
        pos = Position(x, y)
        world = self.sim.world
        # static tile objects anchored with their base at the tile centre
        if tile == "forest":
            spr = self.factory.tree(season)
            surf.blit(spr, (cx - spr.get_width() // 2, cy - spr.get_height() + 4))
        elif tile == "home":
            spr = self.factory.house()
            surf.blit(spr, (cx - spr.get_width() // 2, cy - spr.get_height() + 6))
        elif tile == "workshop":
            spr = self.factory.workshop()
            surf.blit(spr, (cx - spr.get_width() // 2, cy - spr.get_height() + 6))

        # crop overlay
        plot = world.plots.get(pos)
        if plot is not None:
            stage = self._crop_stage(plot)
            spr = self.factory.crop(plot.crop, stage, frame)
            surf.blit(spr, (cx - spr.get_width() // 2, cy - spr.get_height() + 4))

        # merchant marker near the home when an offer is active
        if self.sim.current_offer is not None and pos == Position(world.width // 2 + 1, world.height // 2 + 0):
            pass  # merchant drawn at hero-adjacent grass below if needed

        # hero (drawn in its own cell so painter order keeps it correct)
        if self.sim.hero.pos == pos:
            spr = self.factory.hero(frame)
            surf.blit(spr, (cx - spr.get_width() // 2, cy - spr.get_height() + 3))

    def _crop_stage(self, plot) -> int:
        if plot.ready:
            return 3
        crop = self.sim.crop_book.get(plot.crop)
        grow = max(1, crop.grow_days)
        done = grow - plot.days_left
        frac = done / grow
        if frac < 1 / 3:
            return 0
        if frac < 2 / 3:
            return 1
        return 2

    def _hero_screen_anchor(self) -> tuple[int, int]:
        hero = self.sim.hero
        cx, cy = iso.tile_center(hero.pos.x, hero.pos.y, self.offset_x, self.offset_y)
        return cx, cy - 16  # top of the hero sprite

    # -- map cursor / overlays-on-map ---------------------------------------
    def _draw_map_cursor(self, surf) -> None:
        """Hero locator (first seconds), auto-walk marker, hover highlight +
        tooltip. Skipped when the sim is over (result panel owns the screen)."""
        if self.sim.done:
            return
        # first ~10s: pulsing white outline on the hero's tile
        if self.anim_t - self._session_start < 10.0:
            hcx, hcy = iso.tile_center(self.sim.hero.pos.x, self.sim.hero.pos.y,
                                       self.offset_x, self.offset_y)
            self.overlays.draw_hero_locator(surf, hcx, hcy, self.anim_t * 1.4)
        # auto-walk target marker
        if self.walk_target is not None:
            wx, wy = iso.tile_center(self.walk_target.x, self.walk_target.y,
                                     self.offset_x, self.offset_y)
            self.overlays.draw_walk_marker(surf, wx, wy, (self.anim_t * 1.6) % 1.0)
        # hover highlight + tooltip (suppressed while an overlay/popup is open)
        if self.overlay is None and self.popup is None and self.hover_tile is not None:
            cx, cy = iso.tile_center(self.hover_tile.x, self.hover_tile.y,
                                     self.offset_x, self.offset_y)
            self.overlays.draw_tile_highlight(surf, cx, cy)
            self.overlays.draw_tooltip(surf, self.mouse_pos[0], self.mouse_pos[1],
                                       self._tooltip_lines(self.hover_tile))

    def _tooltip_lines(self, tile: Position) -> list:
        f = self.fonts
        lines = [uh.tile_name(f, self.sim.world.tile_at(tile))]
        crop = uh.crop_tooltip(f, self.sim, tile)
        if crop:
            lines.append(crop)
        dist = uh.manhattan(tile, self.sim.hero.pos)
        if dist > 1:
            lines.append(f.jp(f"距離 {dist} 歩", f"{dist} steps away"))
        elif tile == self.sim.hero.pos:
            lines.append(f.jp("クリックで行動", "click to act"))
        return lines

    def _draw_guide(self, surf) -> None:
        """Context guide strip across the top of the map area."""
        if self.sim.done:
            return
        f = self.fonts
        if self.popup is not None:
            text = f.jp("行動をクリック ／ 外側クリックか Space で閉じる",
                        "click an action / click outside or Space to close")
        elif self.walk_target is not None:
            text = f.jp("移動中… クリックで中断", "walking... click to interrupt")
        elif not self.manual:
            text = f.jp("観戦中 — 一時停止で止める、観戦⇔手動で操作、天の声で助言",
                        "watching - Pause to stop, Watch/Manual to play, Heaven to advise")
        elif self.overlay is None:
            text = f.jp("タイルをクリックして行動を選ぶ ／ 下のボタンで 日記・作る・食べる",
                        "click a tile to act / buttons below: diary, craft, eat")
        else:
            return
        self.overlays.draw_guide(surf, text)

    # -- atmosphere ----------------------------------------------------------
    def _draw_atmosphere(self, surf) -> None:
        pg = self.pg
        world = self.sim.world
        weather = world.weather
        season = world.season

        # weather particles + tint (clipped to the map area)
        clip = pg.Rect(0, 0, VIEW_W, MAP_H)
        if weather in pal.WEATHER_TINT:
            tint = pg.Surface((VIEW_W, MAP_H), pg.SRCALPHA)
            tint.fill(pal.WEATHER_TINT[weather])
            surf.blit(tint, (0, 0))
        if weather == "rain":
            self._draw_rain(surf, speed=420)
        elif weather == "storm":
            self._draw_rain(surf, speed=720)
            self._maybe_flash(surf)
        elif weather == "snow":
            self._draw_snow(surf)

        # time-of-day overlay (after weather so dusk darkens everything)
        tod = pal.time_of_day_tint(self.sim.hero.ap_left, self.sim.ap_per_day)
        if tod is not None:
            ov = pg.Surface((VIEW_W, MAP_H), pg.SRCALPHA)
            ov.fill(tod)
            surf.blit(ov, (0, 0))

        # day-change fade
        if self.fade > 0.001:
            fade = pg.Surface((VIEW_W, VIEW_H), pg.SRCALPHA)
            fade.fill((0, 0, 0, int(255 * min(1.0, self.fade))))
            surf.blit(fade, (0, 0))

    def _draw_rain(self, surf, speed: float) -> None:
        pg = self.pg
        for p in self.particles.rain:
            x, y = int(p[0]), int(p[1])
            pg.draw.line(surf, pal.RAIN_COLOR, (x, y), (x + 1, y + 4), 1)

    def _draw_snow(self, surf) -> None:
        for p in self.particles.snow:
            x, y = int(p[0]), int(p[1])
            if 0 <= y < MAP_H:
                surf.set_at((x % VIEW_W, y), pal.SNOW_COLOR)

    def _maybe_flash(self, surf) -> None:
        if self.particles.flash_timer > 0:
            flash = self.pg.Surface((VIEW_W, MAP_H), self.pg.SRCALPHA)
            flash.fill(pal.STORM_FLASH)
            surf.blit(flash, (0, 0))

    # -- bubbles / overlays --------------------------------------------------
    def _update_bubble(self, now: float) -> None:
        lines = self.sim.hero.spoken_lines
        if len(lines) > self._spoken_index:
            self._spoken_index = len(lines)
            self._bubble_text = lines[-1]
            self._bubble_until = now + 3.5

    def _draw_bubbles(self, surf) -> None:
        if self.sim.done:
            return
        anchor = self._hero_screen_anchor()
        now = self.anim_t
        if self.llm_enabled and self._thread_busy:
            self.hud.draw_thought(surf, anchor)
        if self._bubble_text and now < self._bubble_until:
            self.hud.draw_speech(surf, self._bubble_text, anchor)

    def _draw_overlay(self, surf) -> None:
        # The result panel owns the screen once the sim is over.
        if self.sim.done:
            hover = self._result_hover()
            self._hits["result"] = self.overlays.draw_result(surf, self.sim, hover)
            return
        # modal overlays (each stores its hit-list for the click handler)
        if self.overlay == "help":
            self.overlays.draw_help(surf)
        elif self.overlay == "diary":
            self.overlays.draw_diary(surf, self.sim, self.diary_scroll)
        elif self.overlay == "craft":
            self._hits["rows"] = self.overlays.draw_craft(
                surf, self.sim, self.craft_sel, getattr(self, "_overlay_hover", -1)
            )
        elif self.overlay == "eat":
            self._hits["rows"] = self.overlays.draw_eat(
                surf, self.sim, getattr(self, "_overlay_hover", -1)
            )
        elif self.overlay == "heaven":
            send_hover = self._heaven_send_hover()
            self._hits["heaven_send"] = self.overlays.draw_heaven(
                surf, self.heaven_text, send_hover
            )
        elif self.popup is not None:
            self._draw_popup(surf)

    def _draw_popup(self, surf) -> None:
        rects = self.overlays.draw_popup(
            surf, self.popup["header"], self.popup["items"],
            self.popup["anchor"], self.popup.get("hover", -1),
        )
        self.popup["rects"] = rects

    def _result_hover(self) -> str:
        rects = self._hits.get("result") or {}
        ix, iy = self.mouse_pos
        for key, r in rects.items():
            if r.collidepoint(ix, iy):
                return key
        return ""

    def _heaven_send_hover(self) -> bool:
        send = self._hits.get("heaven_send")
        if send is None:
            return False
        return send.collidepoint(*self.mouse_pos)

    # -- update --------------------------------------------------------------
    def _update(self, dt: float, now: float) -> None:
        self.anim_t += dt
        world = self.sim.world
        # particles
        if world.weather in {"rain"}:
            self.particles.update_rain(dt, 1.0)
        elif world.weather == "storm":
            self.particles.update_rain(dt, 1.6)
            self.particles.flash_timer -= dt
            if self.particles.flash_timer <= -3.0 and self.particles.rng.random() < 0.02:
                self.particles.flash_timer = 0.08
        elif world.weather == "snow":
            self.particles.update_snow(dt)

        # day-change fade trigger
        if world.day != self._last_day:
            self.fade = 1.0
            self._last_day = world.day
        if self.fade > 0:
            self.fade = max(0.0, self.fade - dt / 0.3)  # ~0.6s round trip handled by step()

        self._update_bubble(now)

        if self.sim.done:
            return
        # auto-walk runs in manual mode even with the guide showing, but pauses
        # while a menu/overlay is open.
        if self.manual and self.walk_target is not None and self.overlay is None:
            self._walk_step(now)
        if self.overlay in {"craft", "diary", "help", "heaven", "eat"}:
            return
        if not self.manual and not self.paused:
            self._watch_step(now)

    # -- main loop -----------------------------------------------------------
    def run_window(self) -> int:
        pg = self.pg
        pg.init()
        pg.display.set_caption("SPL — Island Diorama")
        window = pg.display.set_mode((WIN_W, WIN_H))
        internal = pg.Surface((VIEW_W, VIEW_H))
        clock = pg.time.Clock()
        deadline = time.time() + float(getattr(self.args, "_smoke_seconds", 0) or 0)
        smoke = getattr(self.args, "_smoke_seconds", 0)
        start = time.time()
        while self.running:
            dt = clock.tick(60) / 1000.0
            now = time.time() - start
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False
                elif event.type == pg.KEYDOWN:
                    self._last_unicode = getattr(event, "unicode", "")
                    self._handle_key(event.key)
                elif event.type == pg.MOUSEMOTION:
                    self._handle_motion(*event.pos)
                elif event.type == pg.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self._handle_click(*event.pos)
                    elif event.button in (4, 5) and self.overlay == "diary":
                        # mouse-wheel scroll the diary
                        self.diary_scroll = max(0, self.diary_scroll + (1 if event.button == 5 else -1))
            self._update(dt, now)
            self.render(internal)
            pg.transform.scale(internal, (WIN_W, WIN_H), window)
            pg.display.flip()
            if smoke and time.time() >= deadline:
                self.running = False
        pg.quit()
        return 0 if (self.sim.completed or not self.sim.done) else 1

    # -- headless screenshot runner ------------------------------------------
    def run_shots(self, n: int, out_dir: str) -> int:
        pg = self.pg
        pg.init()
        internal = pg.Surface((VIEW_W, VIEW_H))
        window = pg.Surface((WIN_W, WIN_H))
        os.makedirs(out_dir, exist_ok=True)
        saved = 0
        guard = 0
        # step the local brain every frame-batch; save the scaled frame after
        # each sim action.
        while saved < n and guard < n * 80:
            guard += 1
            self.anim_t += 1 / 30.0  # advance animation so frames differ
            # advance weather particles a little for variety in shots
            w = self.sim.world.weather
            if w == "rain":
                self.particles.update_rain(1 / 12.0, 1.0)
            elif w == "storm":
                self.particles.update_rain(1 / 12.0, 1.6)
                if self.particles.rng.random() < 0.15:
                    self.particles.flash_timer = 0.08
                else:
                    self.particles.flash_timer = -1
            elif w == "snow":
                self.particles.update_snow(1 / 8.0)
            if not self.sim.done:
                action = self.local_agent.choose(self.sim)
                self.sim.step(action, confuse_on_invalid=False)
                self._update_bubble(self.anim_t)
            self.render(internal)
            pg.transform.scale(internal, (WIN_W, WIN_H), window)
            path = os.path.join(out_dir, f"shot_{saved:03d}.png")
            pg.image.save(window, path)
            saved += 1
            if self.sim.done:
                break
        pg.quit()
        print(f"Saved {saved} screenshot(s) to {out_dir}")
        return 0


    # -- headless UI-state screenshots (debug: --shots-ui) -------------------
    def run_shots_ui(self, out_dir: str) -> int:
        """Capture the 7 stage-2 UI states headlessly by driving the same
        handler/draw code real input would. Each frame is rendered to the
        internal surface, scaled, and saved."""
        pg = self.pg
        pg.init()
        internal = pg.Surface((VIEW_W, VIEW_H))
        window = pg.Surface((WIN_W, WIN_H))
        os.makedirs(out_dir, exist_ok=True)

        # Fast-forward a dozen days so the hero has tools, food and crops — the
        # popups/HUD then show realistic content.
        self.manual = True
        self._fast_forward(14)
        # guarantee some edible items + seeds for the eat/plant popups, and a
        # little wood/stone/fiber so at least one craft recipe is affordable
        # (so the craft shot can show a hover-highlighted, buildable row).
        self.sim.hero.add_item("bread", 2)
        self.sim.hero.add_item("berries", 3)
        self.sim.hero.add_item("turnip_seed", 3)
        self.sim.hero.add_item("wood", 4)
        self.sim.hero.add_item("stone", 3)
        self.sim.hero.add_item("fiber", 3)
        self.anim_t = 20.0  # past the 10s hero-locator window for clean shots

        def save(name: str) -> None:
            self.render(internal)
            pg.transform.scale(internal, (WIN_W, WIN_H), window)
            pg.image.save(window, os.path.join(out_dir, name))

        world = self.sim.world
        hero = self.sim.hero

        def find(kind: str) -> "Position | None":
            return world.find_nearest(
                hero.pos,
                lambda p: world.tile_at(p) == kind and world.in_bounds(p),
            )

        def tile_center_px(p) -> tuple[int, int]:
            cx, cy = iso.tile_center(p.x, p.y, self.offset_x, self.offset_y)
            return cx, cy

        # (f) clean manual-idle guide strip + (a-setup) — start here so the
        # guide shot has no popup. Hover a tile right next to the hero.
        near = None
        for _, npos in world.neighbors(hero.pos):
            near = npos
            break
        if near is None:
            near = hero.pos
        self.hover_tile = near
        self.mouse_pos = tile_center_px(near)
        save("ui_f_guide.png")

        # (a) hover highlight + tooltip on a tile near the hero (prefer a crop
        # or a named tile that yields a tooltip with distance).
        target = None
        for kind in ("field", "forest", "water"):
            t = find(kind)
            if t is not None:
                target = t
                break
        if target is None:
            target = near
        self.hover_tile = target
        self.mouse_pos = tile_center_px(target)
        save("ui_a_hover.png")

        # (b) click popup OPEN with several rows. Put the hero onto a grass tile
        # that is adjacent to forest AND near water so move/till/chop/forage/
        # fish all appear. Fall back to the hero's own home tile.
        rich = world.find_nearest(
            hero.pos,
            lambda p: world.tile_at(p) in {"grass", "beach"}
            and (world.is_near(p, "forest") or world.is_near(p, "water")),
        )
        if rich is not None:
            hero.pos = rich
        popup_tile = hero.pos  # the hero's own tile -> the most actions
        # if a forest sits next to the hero, click the forest tile so chop/forage
        # show with a header naming the forest.
        forest_adj = None
        for _, npos in world.neighbors(hero.pos):
            if world.tile_at(npos) == "forest":
                forest_adj = npos
                break
        click_tile = forest_adj or popup_tile
        self.hover_tile = None
        self._open_tile_popup(click_tile, tile_center_px(click_tile))
        # hover the second row for the highlight
        if self.popup and len(self.popup["items"]) > 1:
            self.popup["hover"] = 1
        save("ui_b_popup.png")
        self.popup = None

        # (c) eat popup
        self.overlay = "eat"
        self._overlay_hover = 0
        save("ui_c_eat.png")
        self.overlay = None

        # (d) craft overlay with a row hovered (pick first affordable row)
        self.overlay = "craft"
        rows = self.overlays.craft_rows(self.sim)
        hov = next((i for i, (_, aff) in enumerate(rows) if aff), 0)
        self._overlay_hover = hov
        save("ui_d_craft.png")
        self.overlay = None

        # (e) HUD button bar with one button hovered
        self._overlay_hover = -1
        self.button_hover = "craft"
        save("ui_e_buttons.png")
        self.button_hover = None

        # (extra, for self-review) watch-mode info-only popup + auto-walk marker
        self.manual = False
        self._open_tile_popup(click_tile, tile_center_px(click_tile))
        save("ui_x_watch_popup.png")
        self.popup = None
        self.manual = True
        far = None
        for kind in ("rock", "water", "forest"):
            far = find(kind)
            if far is not None and uh.manhattan(far, hero.pos) >= 2:
                break
        if far is not None:
            self.walk_target = far
            self.hover_tile = None
            save("ui_x_walk.png")
            self.walk_target = None

        # (g) result panel with the two buttons (force completion)
        self.sim.completed = True
        self.sim.result_reason = self.fonts.jp("無事に冬を越えた。", "Survived the winter.")
        self._hits["result"] = {}
        self.mouse_pos = (0, 0)
        # render once to lay out the rects, then hover [もう一度]
        self.render(internal)
        again = self._hits.get("result", {}).get("again")
        if again is not None:
            self.mouse_pos = again.center
        save("ui_g_result.png")

        pg.quit()
        print(f"Saved the stage-2 UI screenshots (7 required states + extras) to {out_dir}")
        return 0


def run(args: object) -> int:
    shots = int(getattr(args, "shots", 0) or 0)
    shots_ui = bool(getattr(args, "shots_ui", False))
    if shots_ui:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        app = PixelApp(args, headless=True)
        return app.run_shots_ui(getattr(args, "shot_dir", "/tmp/spl_px"))
    if shots > 0:
        # Headless: set the dummy driver BEFORE pygame init.
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        app = PixelApp(args, headless=True)
        return app.run_shots(shots, getattr(args, "shot_dir", "/tmp/spl_px"))
    app = PixelApp(args, headless=False)
    return app.run_window()
