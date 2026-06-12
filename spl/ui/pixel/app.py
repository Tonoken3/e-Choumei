from __future__ import annotations

"""PixelApp: the 60fps game loop, voxel-world renderer, atmosphere, controls.

Stage-4 rendering is **two-layer, both native-resolution**:

* **Layer 1 (voxel world)** — the map (flat-shaded voxel blocks, recessed water,
  voxel trees/rocks/crops/buildings, hero, merchant, weather particles, tints,
  tile highlight, walk marker, hero locator) is drawn *directly into the
  full-width map band* at native resolution (no low-res upscale). The static
  terrain (all ground blocks, no objects) is composited once into a cached
  full-map surface per (season, water-frame, field-set signature) and only
  rebuilt when tiles/plots change; objects are blitted per frame on top in
  painter's order.
* **Layer 2 (crisp UI)** — all text and UI chrome (guide strip, HUD, button bar,
  tooltips, bubbles, every overlay/popup) is drawn on the same window surface at
  native resolution with antialiased fonts.

The window defaults to Full HD 1920x1080, is resizable, and reflows on resize.
``--scale`` is a window preset (0=auto / 1=small / 2=fhd / 3=large). The map is
drawn at a discrete *sprite scale* (0.75/1.0/1.25/1.5) chosen so the island
footprint fits the map band, keeping sprite caches bounded. The renderer reads
the Simulation but only mutates it via ``sim.step(...)`` on the main thread.
"""

import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from spl.agent.observer import ObservationBuilder
from spl.agent.policy import LocalPolicyAgent
from spl.core.actions import GameAction
from spl.core.crops import FOOD_VALUES
from spl.core.hero import Position
from spl.core.sim import PROJECT_ROOT, Simulation

from . import iso
from . import palette as pal
from . import uihelp as uh
from .hud import (
    DEFAULT_WINDOW, WINDOW_PRESETS, Fonts, Hud, Overlays,
    compute_layout, ui_scale_for,
)
from .sprites import SpriteFactory

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pygame

# The watch-speed ladder. 1=承認 (approval: run at the 普 timer but auto-pause at
# every day boundary for a pit-wall review), 2/3/4 = 遅/普/速 inter-action timers,
# 5=最速 (no artificial delay; the local brain may take several steps per frame so
# a whole year finishes in seconds). 承認 uses the 普 timer between actions.
SPEED_APPROVAL = 1
SPEED_MAX = 5
SPEED_DELAYS = {1: 0.18, 2: 0.6, 3: 0.18, 4: 0.03, 5: 0.0}
# 最速: how many local-brain sim steps to run per frame (LLM is inference-bound,
# so it still steps at most once per frame — just with zero delay).
_MAX_STEPS_PER_FRAME = 8

# Fraction of the desktop the window is allowed to occupy when auto-sizing.
_DESKTOP_FILL = 0.90
# Auto picks FHD if the desktop is at least this big, else fits 90% of desktop.
_FHD_MIN_DESKTOP = (2000, 1130)


@dataclass
class Particles:
    """Render-only weather particles, recycled. Uses the stdlib random module
    (never the sim RNG). Particle coords are in *map-band* (window) pixels and
    are sized for the bigger native-res view (~2x stage-3)."""

    band_w: int = 1920
    band_h: int = 900
    rng: random.Random = field(default_factory=lambda: random.Random(1234))
    rain: list[list[float]] = field(default_factory=list)
    snow: list[list[float]] = field(default_factory=list)
    flash_timer: float = 0.0

    def resize(self, band_w: int, band_h: int) -> None:
        self.band_w, self.band_h = max(1, band_w), max(1, band_h)
        self.rain.clear()
        self.snow.clear()

    def ensure_rain(self, n: int) -> None:
        while len(self.rain) < n:
            self.rain.append([self.rng.uniform(0, self.band_w),
                              self.rng.uniform(0, self.band_h)])

    def ensure_snow(self, n: int) -> None:
        while len(self.snow) < n:
            self.snow.append([self.rng.uniform(0, self.band_w),
                              self.rng.uniform(0, self.band_h),
                              self.rng.uniform(-0.3, 0.3)])

    def update_rain(self, dt: float, speed: float) -> None:
        self.ensure_rain(160)
        for p in self.rain:
            p[1] += speed * dt
            p[0] += speed * 0.3 * dt
            if p[1] > self.band_h:
                p[0] = self.rng.uniform(0, self.band_w)
                p[1] = -8

    def update_snow(self, dt: float) -> None:
        self.ensure_snow(120)
        for p in self.snow:
            p[1] += 30 * dt
            p[0] += p[2] * 20 * dt
            if p[1] > self.band_h:
                p[0] = self.rng.uniform(0, self.band_w)
                p[1] = -8


def choose_window(desktop_w: int, desktop_h: int, forced: int = 0) -> tuple[int, int]:
    """Window size from the --scale preset. 1/2/3 = small/fhd/large; 0 = auto:
    pick FHD if the desktop is big enough, else fit 90% of the desktop with a
    16:9-ish window."""
    if forced in WINDOW_PRESETS:
        return WINDOW_PRESETS[forced]
    if desktop_w >= _FHD_MIN_DESKTOP[0] and desktop_h >= _FHD_MIN_DESKTOP[1]:
        return DEFAULT_WINDOW
    w = int(desktop_w * _DESKTOP_FILL)
    h = int(desktop_h * _DESKTOP_FILL)
    return max(960, w), max(600, h)


class PixelApp:
    def __init__(self, args: object, headless: bool = False) -> None:
        import pygame

        self.pg = pygame
        pygame.init()
        self.headless = headless
        self.args = args
        self.sim = Simulation(
            seed=getattr(args, "seed", 42),
            max_days=getattr(args, "days", 112),
            difficulty=getattr(args, "difficulty", "修羅"),
        )
        # --strategy seeds the initial standing 作戦 before the run begins.
        strategy = getattr(args, "strategy", None)
        if strategy:
            self.sim.set_strategy(strategy)
        self.local_agent = LocalPolicyAgent()
        self.observer = ObservationBuilder()

        # optional fast-forward for verification of later seasons
        start_day = int(getattr(args, "start_day", 0) or 0)
        if start_day > 1:
            self._fast_forward(start_day)

        # -- window preset + layout -----------------------------------------
        self.forced_scale = int(getattr(args, "scale", 0) or 0)
        win_w, win_h = self._initial_window()
        ui = ui_scale_for(win_h)
        self.fonts = Fonts(pygame, scale=ui)
        # sprite scale is chosen against the map band -> need a provisional lay
        self.lay = compute_layout(pygame, win_w, win_h, 1.0)
        self.factory = SpriteFactory(pygame)
        self.hud = Hud(pygame, self.fonts)
        self.overlays = Overlays(pygame, self.fonts)
        self.particles = Particles()
        self._terrain: "pygame.Surface | None" = None
        self._terrain_sig: tuple | None = None
        self._terrain_baked: tuple[int, int] = (0, 0)
        # -- camera (offset, scale) -----------------------------------------
        # ``cam_scale`` is the live zoom; ``fit_scale_val`` is the whole-island
        # floor. ``follow`` glides the camera so the hermit stays centred; pan
        # (right/middle drag) suspends it. ``cam_target_*`` are the offsets the
        # follow/pan drive toward; ``offset_*`` lerp to them for a soft glide.
        self.cam_scale = 1.0
        self.fit_scale_val = 1.0
        self.follow = False
        self._follow_armed = False  # auto-enable follow the first zoom-in
        self.panning = False
        self._pan_last = (0, 0)
        self.cam_target_x = 0
        self.cam_target_y = 0
        self._apply_window(win_w, win_h)

        # modes / state
        self.manual = bool(getattr(args, "manual", False))
        self.speed = int(getattr(args, "speed", 2) or 2)
        # 八識熟考 (parallel deliberation) watch toggle: while ON, the LLM worker
        # fans out the eight識 instead of the serial choose(). Mirrored onto the
        # brain's deliberate_forced. Off by default; a body scream auto-bursts
        # regardless. Inert when the cassette has no parallel budget.
        self.deliberate_mode = bool(getattr(args, "deliberate", False))
        self.paused = False
        self.overlay: str | None = None  # diary | help | craft | heaven | eat | carve | miracle
        self.diary_scroll = 0
        self.craft_sel = 0
        self.heaven_text = ""
        # 神のレバー (共同モード): the 神託/夢 text box and the last status line shown
        # in the miracle palette (e.g. "神力が足りぬ"). Reuses the 作戦 input
        # machinery (TEXTINPUT + CTRL+V).
        self.miracle_text = ""
        self.miracle_status = ""
        # 刻む: the text being composed for the 句 carved into the stone (its own
        # overlay, mirroring the 作戦/heaven input box: IME + CTRL+V + Enter送る).
        self.carve_text = ""
        self._text_input_on = False
        # 承認制 (approval mode): when the day rolls over we auto-pause at the
        # boundary and show a pit-wall strip until the watcher resumes. This is
        # the day we have already paused on (so we pause once per boundary).
        self.approval_pause = False
        self._approval_acked_day = self.sim.world.day
        self.running = True

        # -- mouse / click-to-act state --------------------------------------
        # mouse_win: window coords. mouse_map: internal-map coords (or None when
        # the cursor is outside the map band).
        self.mouse_win = (0, 0)
        self.mouse_map: tuple[int, int] | None = (0, 0)
        self.hover_tile: Position | None = None
        # popup: dict(kind, tile, header, items, anchor(window coords), hover, rects)
        self.popup: dict | None = None
        self._hits: dict[str, object] = {}
        self._overlay_hover = -1
        self.button_hover: str | None = None
        # auto-walk
        self.walk_target: Position | None = None
        self._last_walk_t = 0.0
        self._session_start = 0.0
        self.buttons = self._build_buttons()

        # animation / timing
        self.anim_t = 0.0
        self.last_action_t = 0.0
        self.fade = 0.0
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
        # 八識熟考: a cassette with no parallel budget can't deliberate; force the
        # toggle off so the button reads inert. Mirror the toggle onto the brain.
        if self.brain is None or getattr(self.brain, "parallel", 0) <= 0:
            self.deliberate_mode = False
        self._sync_deliberate()
        self._motto: dict | None = None
        self._motto_busy = False
        self._pending_action: GameAction | None = None
        self._thread_busy = False
        self._thread_lock = threading.Lock()

        # ぼうけんのしょ: load this cassette's journal and inject the past-life
        # lessons into the brain's observer. The entry for THIS run is written
        # once, after the motto resolves at sim.done (guarded by _book_written).
        self.book = self._load_book()
        self._book_written = False
        self._book_entry: dict | None = None
        self._inject_book()

        # 石碑の記憶: load the stone (independent of --book) and let the past
        # hermits' carvings teach on day 1. Written once at run end.
        self._stone_written = False
        self._load_stone_into_sim()

    # -- window / scale / layout --------------------------------------------
    def _initial_window(self) -> tuple[int, int]:
        try:
            sizes = self.pg.display.get_desktop_sizes()
            dw, dh = sizes[0]
        except Exception:  # noqa: BLE001 - headless/dummy driver may not report
            dw, dh = 1920, 1080
        return choose_window(dw, dh, self.forced_scale)

    def _apply_window(self, win_w: int, win_h: int) -> None:
        """(Re)build fonts, layout, sprite factory and world offsets for a new
        window size. The *fit* scale (whole-island floor) is recomputed for the
        new band; the live camera zoom is kept (clamped into the new ladder) so
        a resize doesn't yank the view. Rebuilds the factory only when the scale
        changes, and invalidates the cached terrain."""
        ui = ui_scale_for(win_h)
        self.fonts.for_scale(ui)
        # provisional layout to know the map band, then pick the fit (floor) zoom
        lay0 = compute_layout(self.pg, win_w, win_h, 1.0)
        world = self.sim.world
        self.fit_scale_val = iso.fit_scale(world.width, world.height,
                                           lay0.map_rect.width, lay0.map_rect.height,
                                           headroom=int(120 * ui / 2))
        # keep the live camera zoom (clamped into [fit, max]); default to fit
        prev = getattr(self, "cam_scale", 0.0) or self.fit_scale_val
        self.cam_scale = iso.clamp_scale(prev, self.fit_scale_val)
        self._set_layout_for_scale(win_w, win_h, self.cam_scale)
        self._recompute_offsets()
        self.particles.resize(self.lay.map_rect.width, self.lay.map_rect.height)
        self._terrain = None
        self._terrain_sig = None
        self.buttons = self._build_buttons()

    def _set_layout_for_scale(self, win_w: int, win_h: int, scale: float) -> None:
        """Build the layout + sprite factory for a given camera zoom, rebuilding
        the factory only when the scale actually changes."""
        self.lay = compute_layout(self.pg, win_w, win_h, scale)
        if abs(self.factory.scale - scale) > 1e-6:
            self.factory = SpriteFactory(self.pg, scale=scale)

    def _centered_offsets(self, scale: float) -> tuple[int, int]:
        """The offsets that centre the whole island in the band at ``scale``
        (window coords). This is the resting view at fit scale and the clamp
        anchor for pan."""
        lay = self.lay
        world = self.sim.world
        overhang = int(round(48 * scale))  # headroom for tall sprites above ground
        ox, oy = iso.centering_offset(world.width, world.height,
                                      lay.map_rect.width, lay.map_rect.height,
                                      scale, overhang_top=overhang)
        return ox + lay.map_rect.x, oy + lay.map_rect.y

    def _recompute_offsets(self) -> None:
        """Reset the camera to the centred-island view (used on (re)build/resize
        and whenever we're at the fit-all floor)."""
        ox, oy = self._centered_offsets(self.lay.sprite_scale)
        self.offset_x = ox
        self.offset_y = oy
        self.cam_target_x = ox
        self.cam_target_y = oy

    # -- camera (zoom / follow / pan) ----------------------------------------
    def _at_fit_scale(self) -> bool:
        """True when zoomed all the way out to the whole-island floor — where
        follow/pan are inert and the view is the centred diorama."""
        return self.cam_scale <= self.fit_scale_val + 1e-6

    def _clamp_offset(self, ox: int, oy: int) -> tuple[int, int]:
        """Keep the island from sliding entirely out of the band: at least ~25%
        of its projected footprint must stay visible on every side. At the fit
        floor this collapses to the centred view."""
        lay = self.lay
        sc = self.lay.sprite_scale
        world = self.sim.world
        if self._at_fit_scale():
            return self._centered_offsets(sc)
        map_w, map_h = iso.map_pixel_size(world.width, world.height, sc)
        # leftmost projected screen-x (cell (0,h-1)) sits at min_sx + offset.
        min_sx = (0 - (world.height - 1)) * iso.half_w(sc)
        band = lay.map_rect
        keep_w = int(map_w * 0.25)
        keep_h = int(map_h * 0.25)
        # world bbox left/top/right/bottom in window coords for a given offset
        # left edge x = offset_x + min_sx ; span map_w. top edge y = offset_y.
        # Require keep_* px of the bbox to overlap the band on each axis.
        lo_x = band.left - (min_sx + map_w - keep_w)
        hi_x = band.right - (min_sx + keep_w)
        lo_y = band.top - (map_h - keep_h)
        hi_y = band.bottom - keep_h
        cx = max(lo_x, min(hi_x, ox))
        cy = max(lo_y, min(hi_y, oy))
        return cx, cy

    def _follow_offset(self) -> tuple[int, int]:
        """Offset that centres the hermit in the map band at the current zoom."""
        lay = self.lay
        sc = lay.sprite_scale
        hero = self.sim.hero.pos
        # screen position of the hero's tile centre at offset (0,0)
        cx0, cy0 = iso.tile_center(hero.x, hero.y, 0, 0, sc)
        cx0 -= self.factory.ground_top_y(self.sim.world.tile_at(hero))
        # raise the centre a touch so the speech bubble above the head has room
        band = lay.map_rect
        target_x = band.centerx
        target_y = band.centery + int(round(28 * sc))
        return target_x - cx0, target_y - cy0

    def _update_follow_target(self) -> None:
        """Set the camera target. Follow centres the hermit; otherwise the
        target is wherever pan left it (clamped). At the fit floor it's centred."""
        if self._at_fit_scale():
            self.cam_target_x, self.cam_target_y = self._centered_offsets(self.lay.sprite_scale)
            return
        if self.follow and not self.panning:
            ox, oy = self._follow_offset()
            self.cam_target_x, self.cam_target_y = self._clamp_offset(ox, oy)

    def _glide_camera(self) -> None:
        """Lerp the live offset toward the camera target (~0.12/frame) so motion
        glides instead of snapping."""
        if self._at_fit_scale():
            self.offset_x, self.offset_y = self.cam_target_x, self.cam_target_y
            return
        k = 0.12
        dx = self.cam_target_x - self.offset_x
        dy = self.cam_target_y - self.offset_y
        if abs(dx) < 1 and abs(dy) < 1:
            self.offset_x, self.offset_y = self.cam_target_x, self.cam_target_y
            return
        self.offset_x = int(round(self.offset_x + dx * k))
        self.offset_y = int(round(self.offset_y + dy * k))

    def _zoom_at(self, direction: int, anchor: tuple[int, int] | None) -> None:
        """Step the zoom ladder up (direction>0) or down. The world point under
        ``anchor`` (window px; defaults to band centre) stays fixed. Arms follow
        the first time we zoom past the fit floor."""
        new_scale = iso.step_scale(self.cam_scale, direction, self.fit_scale_val)
        if abs(new_scale - self.cam_scale) < 1e-6:
            return
        lay = self.lay
        if anchor is None or not lay.map_rect.collidepoint(*anchor):
            anchor = lay.map_rect.center
        ax, ay = anchor
        old_scale = self.cam_scale
        # world point (continuous tile coords) under the anchor at the old zoom
        ratio = new_scale / old_scale
        # new offset so that the same screen anchor maps to the same world point:
        #   screen = world*scale + offset  ->  keep (anchor-offset) scaled by ratio
        self.cam_scale = new_scale
        self._set_layout_for_scale(self.lay.win_w, self.lay.win_h, new_scale)
        # the projection is linear in scale about the offset, so:
        new_ox = ax - int(round((ax - self.offset_x) * ratio))
        new_oy = ay - int(round((ay - self.offset_y) * ratio))
        # auto-arm follow the first time the user closes in past the diorama
        if not self._at_fit_scale() and not self._follow_armed:
            self.follow = True
            self._follow_armed = True
        self.offset_x, self.offset_y = self._clamp_offset(new_ox, new_oy)
        self.cam_target_x, self.cam_target_y = self.offset_x, self.offset_y
        self._update_follow_target()
        # terrain is baked at origin per scale -> only the scale change matters
        self._terrain = None
        self._terrain_sig = None

    def _pan_by(self, dx: int, dy: int) -> None:
        """Right/middle-drag pan: move the camera and suspend follow."""
        if self._at_fit_scale():
            return
        if self.follow:
            self.follow = False
        ox, oy = self._clamp_offset(self.offset_x + dx, self.offset_y + dy)
        self.offset_x, self.offset_y = ox, oy
        self.cam_target_x, self.cam_target_y = ox, oy

    def _toggle_follow(self) -> None:
        self.follow = not self.follow
        self._follow_armed = True
        if self.follow:
            self._update_follow_target()

    def _camera_input_allowed(self) -> bool:
        """Camera input (wheel/pan) is live only over the map band with no modal
        overlay/popup grabbing the pointer, and not on the result screen."""
        return (self.overlay is None and self.popup is None and not self.sim.done)

    def _handle_wheel(self, direction: int, pos: tuple[int, int]) -> None:
        if not self._camera_input_allowed():
            return
        if not self.lay.map_rect.collidepoint(*pos):
            return
        self._zoom_at(direction, pos)

    def _begin_pan(self, pos: tuple[int, int]) -> None:
        if not self._camera_input_allowed() or self._at_fit_scale():
            return
        if not self.lay.map_rect.collidepoint(*pos):
            return
        self.panning = True
        self._pan_last = pos

    def _end_pan(self) -> None:
        self.panning = False

    def _resize(self, win_w: int, win_h: int) -> None:
        """On VIDEORESIZE: reflow to the new window size (UI scale follows the
        window height; the sprite zoom is re-chosen to fit the band)."""
        win_w = max(960, win_w)
        win_h = max(600, win_h)
        self._apply_window(win_w, win_h)

    # -- brain ---------------------------------------------------------------
    def _make_brain(self):
        from spl.agent.llm_client import OpenAICompatibleBrain, find_cassette

        name = getattr(self.args, "cassette", None)
        # --cassette MAGI builds the MAGI brain in code: "MAGI" → v2 pilot,
        # "MAGI-V1" → v1 committee.
        if name in ("MAGI", "MAGI-V1"):
            from spl.agent.magi import magi_mode_for_cassette, make_magi

            try:
                return make_magi(
                    PROJECT_ROOT / "config" / "models.toml",
                    mode=magi_mode_for_cassette(name),
                )
            except Exception:  # noqa: BLE001
                return None
        try:
            cassette = find_cassette(
                PROJECT_ROOT / "config" / "models.toml", name
            )
        except Exception:  # noqa: BLE001
            return None
        if not cassette.base_url:
            return None
        # --tps overrides the cassette's forced TPS (思考予算). 0 = auto-measure.
        tps = float(getattr(self.args, "tps", 0.0) or 0.0)
        if tps > 0:
            from dataclasses import replace

            cassette = replace(cassette, tps=tps)
        brain = OpenAICompatibleBrain(cassette)
        # 入植者の来歴: graft the player-written persona (--persona / --persona-replace).
        from spl.ui.cli import apply_persona

        apply_persona(brain, self.args)
        return brain

    # -- 八識熟考 toggle ------------------------------------------------------
    def _can_deliberate(self) -> bool:
        """True when an LLM brain with a parallel budget is wired (the [熟考]
        button is live only then)."""
        return self.brain is not None and getattr(self.brain, "parallel", 0) > 0

    def _sync_deliberate(self) -> None:
        """Mirror the UI toggle onto the brain's deliberate_forced flag."""
        if self.brain is not None:
            self.brain.deliberate_forced = bool(self.deliberate_mode) and self._can_deliberate()

    def _toggle_deliberate(self) -> None:
        if not self._can_deliberate():
            return
        self.deliberate_mode = not self.deliberate_mode
        self._sync_deliberate()

    # -- ぼうけんのしょ -------------------------------------------------------
    def _book_cassette_name(self) -> str:
        if self.brain is not None:
            name = getattr(getattr(self.brain, "cassette", None), "name", None)
            if name:
                return name
        return getattr(self.args, "cassette", None) or "Local仙人"

    def _load_book(self):
        """Load this cassette's adventure book when --book is on, else None."""
        if not bool(getattr(self.args, "book", False)):
            return None
        from spl.agent.bouken import BoukenNoSho

        return BoukenNoSho.for_cassette(self._book_cassette_name())

    def _inject_book(self) -> None:
        """Hand the past-life lessons to the brain's observer for THIS run, and
        carve the last 1-2 mottos onto the 石碑's back (古い石碑) before play."""
        if self.book is None:
            return
        # 古い石碑: the graves teach even for a local (non-LLM) run — set the
        # epitaphs on the sim regardless of whether a brain reads the lessons.
        mottos = [str(e.get("motto", "")).strip() for e in self.book.entries]
        self.sim.set_monument_epitaphs([m for m in mottos if m][-2:])
        if self.brain is None:
            return
        from spl.agent.bouken import inject_into_observer

        observer = getattr(self.brain, "observer", None)
        inject_into_observer(observer, self.book, getattr(self.args, "seed", 0))

    def _maybe_write_book(self) -> None:
        """Append this ended run to the book exactly once — after sim.done AND
        the motto has resolved (so its lessons are recorded). [もう一度] resets
        the guard via _rebuild_sim so each replayed life accumulates."""
        if self.book is None or self._book_written:
            return
        if not self.sim.done or self._motto is None or self._motto_busy:
            return
        from spl.agent.bouken import build_entry, fallback_compile

        self._book_entry = self.book.append(
            build_entry(
                self.sim, getattr(self.args, "seed", 0), self._motto,
                str(getattr(self.brain, "player_persona", "") or ""),
            )
        )
        # 家訓の編纂: revise the fixed 5-article canon from the new history.
        # LLM compiler when a brain is available; deterministic fallback on any
        # exception so the result screen never crashes.
        articles = None
        if self.brain is not None and hasattr(self.brain, "compile_canon"):
            try:
                articles = self.brain.compile_canon(self.book)
            except Exception:  # noqa: BLE001
                articles = None
        if not articles:
            articles = fallback_compile(self.book)
        self.book.set_canon(articles, self.book.canon_revision + 1)
        self._book_written = True

    # -- 石碑の記憶 (the carved stone, 刻む persistence) ----------------------
    # INDEPENDENT of --book: the stone always remembers. Same cassette+island
    # keying; the most-recent 3 same-seed verses teach the next hermit.
    def _stone_path(self):
        from spl.agent.bouken import stone_path_for

        return stone_path_for(self._book_cassette_name())

    def _load_stone_into_sim(self) -> None:
        """Hand the最新3句 of this seed to the sim BEFORE play, so a hermit reads
        what past hermits chose to carve here."""
        from spl.agent.bouken import load_stone

        seed = int(getattr(self.args, "seed", 0) or 0)
        entries = load_stone(self._stone_path())
        same = [e["text"] for e in entries if int(e.get("seed", 0)) == seed]
        self.sim.set_stone_carvings(same[-3:])

    def _maybe_write_stone(self) -> None:
        """Append this hermit's voluntary carvings (with days) to the stone once
        at run end. Guarded by _stone_written; [もう一度] resets it via
        _rebuild_sim so each life's carvings accumulate."""
        if self._stone_written or not self.sim.done:
            return
        from spl.agent.bouken import append_carvings

        life = self.book.lives if self.book is not None else 0
        append_carvings(self._stone_path(), int(getattr(self.args, "seed", 0) or 0),
                        list(getattr(self.sim, "carvings_made", []) or []), life=life)
        self._stone_written = True

    def _brain_status(self) -> str:
        """思考予算 tier+TPS line for the HUD, read from the brain safely."""
        if not self.llm_enabled or self.brain is None:
            return ""
        try:
            if getattr(self.brain, "calls", 0) <= 0 and getattr(self.brain, "forced_tps", 0) <= 0:
                return ""
            return self.brain.status_line()
        except Exception:  # noqa: BLE001
            return ""

    def _fast_forward(self, target_day: int) -> None:
        agent = LocalPolicyAgent()
        guard = 0
        while not self.sim.done and self.sim.world.day < target_day and guard < target_day * 60:
            self.sim.step(agent.choose(self.sim), confuse_on_invalid=False)
            guard += 1

    # -- HUD button bar layout ----------------------------------------------
    def _build_buttons(self) -> list:
        """Lay out the HUD button bar (window coords) along the top button-bar
        strip of the HUD panel. Dynamic labels are refreshed each frame."""
        f = self.fonts
        lay = self.lay
        specs = [
            ("pause", f.jp("一時停止", "Pause"), f.jp("自動進行を止める/再開", "pause/resume")),
            ("speed", f.jp("速度:普", "Speed:Nrm"), f.jp("承認/遅/普/速/最速 を切り替え", "cycle watch speed")),
            ("mode", f.jp("観戦⇔手動", "Watch/Manual"), f.jp("見守る/自分で操作", "watch vs play")),
            ("diary", f.jp("日記", "Diary"), f.jp("仙人の日記を読む", "read the diary")),
            ("craft", f.jp("作る", "Craft"), f.jp("道具や建物を作る", "craft & build")),
            ("eat", f.jp("食べる", "Eat"), f.jp("持ち物を食べる", "eat from inventory")),
            ("heaven", f.jp("作戦", "Strategy"), f.jp("仙人に作戦を授ける（天の声）", "set the standing order")),
            ("miracle", f.jp("奇跡", "Miracle"), f.jp("神のレバー：天候/マナ/商人/神託/夢（共同モード）", "divine levers (co-op)")),
            ("help", f.jp("ヘルプ", "Help"), f.jp("操作の説明", "how to play")),
            ("zoom_out", "－", f.jp("引く（ホイール下）", "zoom out (wheel)")),
            ("zoom_in", "＋", f.jp("寄る（ホイール上）", "zoom in (wheel)")),
            ("follow", f.jp("追従", "Follow"), f.jp("仙人を追って中央に映す", "keep the hermit centred")),
            ("deliberate", f.jp("熟考", "Deliberate"),
             f.jp("八識熟考：八つの識で同時に考える（キー G）", "八識: think with eight parallel minds (G)")),
        ]
        x = lay.map_rect.x + lay.px(6)
        y = lay.hud_top + lay.px(3)
        h = lay.button_bar_h - lay.px(6)
        gap = lay.px(4)
        pad = lay.px(14)
        buttons = []
        for key, label, tip in specs:
            w = f.label.size(label)[0] + pad
            rect = self.pg.Rect(x, y, w, h)
            buttons.append(uh.Button(key, label, tip, rect=rect))
            x += w + gap
        return buttons

    def _refresh_buttons(self) -> None:
        """Update dynamic labels (pause/speed) and enabled state per mode, then
        recompute widths/positions in window coords."""
        f = self.fonts
        lay = self.lay
        speed_jp = {1: "承認", 2: "遅", 3: "普", 4: "速", 5: "最速"}[self.speed]
        speed_en = {1: "Appr", 2: "Slow", 3: "Nrm", 4: "Fast", 5: "Max"}[self.speed]
        ladder = iso.zoom_ladder(self.fit_scale_val)
        at_max = self.cam_scale >= ladder[-1] - 1e-6
        at_min = self._at_fit_scale()
        for b in self.buttons:
            if b.key == "pause":
                b.label = f.jp("再開", "Resume") if self.paused else f.jp("一時停止", "Pause")
            elif b.key == "speed":
                b.label = f.jp(f"速度:{speed_jp}", f"Speed:{speed_en}")
            elif b.key == "follow":
                # ● marks ON so the active state reads at a glance
                b.label = f.jp("追従●", "Follow●") if self.follow else f.jp("追従", "Follow")
                b.active = self.follow
                b.enabled = not at_min  # inert at the whole-island floor
            elif b.key == "deliberate":
                # 八識熟考: ● marks the toggle ON. Inert without a parallel brain.
                on = bool(self.deliberate_mode)
                b.label = f.jp("熟考●", "Delib●") if on else f.jp("熟考", "Delib")
                b.active = on
                b.enabled = self._can_deliberate()
            if b.key in {"craft", "eat"}:
                b.enabled = self.manual
            if b.key == "zoom_in":
                b.enabled = not at_max
            if b.key == "zoom_out":
                b.enabled = not at_min
        x = lay.map_rect.x + lay.px(6)
        y = lay.hud_top + lay.px(3)
        h = lay.button_bar_h - lay.px(6)
        gap = lay.px(4)
        pad = lay.px(14)
        for b in self.buttons:
            w = f.label.size(b.label)[0] + pad
            b.rect.update(x, y, w, h)
            x += w + gap

    # -- replay (result panel) ----------------------------------------------
    def _rebuild_sim(self) -> None:
        """Fresh Simulation on the same seed/args (the [もう一度] button)."""
        self.sim = Simulation(
            seed=getattr(self.args, "seed", 42),
            max_days=getattr(self.args, "days", 112),
            difficulty=getattr(self.args, "difficulty", "修羅"),
        )
        strategy = getattr(self.args, "strategy", None)
        if strategy:
            self.sim.set_strategy(strategy)
        if self.brain is not None:
            self.sim.set_diarist(self.brain)
        self.approval_pause = False
        self._approval_acked_day = self.sim.world.day
        self._motto = None
        self._motto_busy = False
        # ぼうけんのしょ: [もう一度] is the NEXT life — reload the book so the
        # lessons just written are visible to this run, re-inject, reset the
        # write guard so this fresh life records exactly once.
        self.book = self._load_book()
        self._book_written = False
        self._book_entry = None
        self._inject_book()
        # 石碑の記憶: reload the stone so this fresh life inherits the carvings the
        # previous life just left, and reset the write guard.
        self._stone_written = False
        self._load_stone_into_sim()
        # reset the camera to the whole-island diorama for the fresh run
        self.cam_scale = self.fit_scale_val
        self.follow = False
        self._follow_armed = False
        self.panning = False
        self._set_layout_for_scale(self.lay.win_w, self.lay.win_h, self.cam_scale)
        self._recompute_offsets()
        self._terrain = None
        self._terrain_sig = None
        self.popup = None
        self.walk_target = None
        self.overlay = None
        self.fade = 0.0
        self._last_day = self.sim.world.day
        self._spoken_index = len(self.sim.hero.spoken_lines)
        self._bubble_text = ""
        self._bubble_until = 0.0
        self._session_start = self.anim_t

    # -- coordinate helpers --------------------------------------------------
    # The world is now drawn directly into the map band at native resolution, so
    # window coords == world-screen coords (within the band). These wrappers stay
    # for call-site clarity and so picking can clamp to the band.
    def _win_to_map(self, wx: int, wy: int) -> tuple[int, int] | None:
        """Window pixel within the map band, or None if outside it."""
        r = self.lay.map_rect
        if not r.collidepoint(wx, wy):
            return None
        return wx, wy

    def _map_to_win(self, mx: int, my: int) -> tuple[int, int]:
        """Identity now (world is drawn in window coords)."""
        return mx, my

    def _tile_under(self, mx: int, my: int) -> Position | None:
        if my >= self.lay.map_rect.bottom:
            return None
        tile = iso.screen_to_tile(mx, my, self.offset_x, self.offset_y,
                                  self.lay.sprite_scale)
        if not self.sim.world.in_bounds(tile):
            return None
        return tile

    # -- mouse: motion -------------------------------------------------------
    def _handle_motion(self, wx: int, wy: int) -> None:
        # right/middle-drag pan: move the camera by the cursor delta, suspend
        # follow. Done before hover so the pan stays glued to the cursor.
        if self.panning:
            dx = wx - self._pan_last[0]
            dy = wy - self._pan_last[1]
            self._pan_last = (wx, wy)
            self._pan_by(dx, dy)
        self.mouse_win = (wx, wy)
        self.mouse_map = self._win_to_map(wx, wy)
        if self.mouse_map is not None:
            self.hover_tile = self._tile_under(*self.mouse_map)
        else:
            self.hover_tile = None
        self.button_hover = self._button_at(wx, wy)
        # popup row hover (window coords)
        if self.popup is not None:
            self.popup["hover"] = -1
            for i, r in enumerate(self.popup.get("rects", [])):
                if r.collidepoint(wx, wy):
                    self.popup["hover"] = i
                    break
        # overlay row hover (craft / eat) — rects are window coords
        self._overlay_hover = -1
        rows = self._hits.get("rows")
        if rows:
            for i, entry in enumerate(rows):
                r = entry[0]
                if r.collidepoint(wx, wy):
                    self._overlay_hover = i
                    break

    def _button_at(self, wx: int, wy: int) -> str | None:
        for b in self.buttons:
            if b.enabled and b.rect.collidepoint(wx, wy):
                return b.key
        return None

    # -- mouse: click --------------------------------------------------------
    def _handle_click(self, wx: int, wy: int) -> None:
        # 1) overlays consume clicks first (result panel always active when done)
        if self.sim.done:
            self._click_result(wx, wy)
            return
        if self.overlay is not None:
            self._click_overlay(wx, wy)
            return
        # 1b) 承認制 pit wall takes first pick at the day boundary, but the HUD
        # button bar stays clickable (changing speed away from 承認 resumes).
        if self.approval_pause:
            if self._click_pitwall(wx, wy):
                return
            btn = self._button_at(wx, wy)
            if btn is not None:
                self._activate_button(btn)
            return
        # 2) auto-walking: any click interrupts (spec: "クリックで中断")
        if self.walk_target is not None:
            self.walk_target = None
            return
        # 3) open popup? click on its row?
        if self.popup is not None:
            self._click_popup(wx, wy)
            return
        # 4) HUD button bar
        btn = self._button_at(wx, wy)
        if btn is not None:
            self._activate_button(btn)
            return
        # 5) a map tile -> open the context popup
        mp = self._win_to_map(wx, wy)
        if mp is not None:
            tile = self._tile_under(*mp)
            if tile is not None:
                self._open_tile_popup(tile, (wx, wy))

    def _open_tile_popup(self, tile: Position, anchor: tuple) -> None:
        """``anchor`` is in window coords (where the popup is drawn)."""
        f = self.fonts
        header = self._tile_header(tile)
        if not self.manual:
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
        # 古い石碑: the stone's tile names the monument over the bare ground.
        if tile == self.sim.world.monument_pos:
            stele = f.jp("古い石碑", "Old Stone Monument")
            return f"{name} — {stele}" if tile != self.sim.hero.pos else stele
        if tile == self.sim.hero.pos:
            return f"{name}（" + f.jp("現在地", "you") + "）"
        return name

    def _click_popup(self, wx: int, wy: int) -> None:
        rects = self.popup.get("rects", [])
        for i, r in enumerate(rects):
            if r.collidepoint(wx, wy):
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
            self.walk_target = Position(int(item.args["x"]), int(item.args["y"]))
            self._last_walk_t = self.anim_t
            self.popup = None
            return
        # 刻む: opening the carve box rather than dispatching — the 句 is composed
        # in a text overlay (IME/paste), then 送る issues the carve action.
        if verb == "carve_open":
            self.overlay = "carve"
            self.carve_text = ""
            self.popup = None
            return
        self.sim.step(GameAction.safe(verb, **item.args), confuse_on_invalid=False)
        self.popup = None

    def _set_speed(self, speed: int) -> None:
        """Change the watch speed. Leaving 承認 lifts any pending boundary pause;
        entering it arms the next boundary (no stale pause on the current day)."""
        speed = max(1, min(SPEED_MAX, speed))
        if speed != SPEED_APPROVAL:
            self.approval_pause = False
        else:
            # arm at the next boundary, not the day we are already on
            self._approval_acked_day = self.sim.world.day
        self.speed = speed

    def _resume_next_day(self) -> None:
        """[次の日へ]: acknowledge the current boundary and let the run continue
        until the next one."""
        self._approval_acked_day = self.sim.world.day
        self.approval_pause = False
        self.last_action_t = self.anim_t

    def _activate_button(self, key: str) -> None:
        if key == "pause":
            self.paused = not self.paused
        elif key == "speed":
            self._set_speed(self.speed % SPEED_MAX + 1)
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
        elif key == "miracle":
            self.overlay = "miracle"
            self.miracle_text = ""
            self.miracle_status = ""
        elif key == "help":
            self.overlay = "help"
        elif key == "zoom_in":
            self._zoom_at(+1, None)  # buttons zoom about the band centre
        elif key == "zoom_out":
            self._zoom_at(-1, None)
        elif key == "follow":
            self._toggle_follow()
        elif key == "deliberate":
            self._toggle_deliberate()

    def _click_overlay(self, wx: int, wy: int) -> None:
        from spl.core.actions import GameAction as _GA

        if self.overlay == "craft":
            rows = self._hits.get("rows") or []
            for (r, recipe, affordable) in rows:
                if r.collidepoint(wx, wy):
                    if affordable:
                        action = "build" if recipe.kind == "build" else "craft"
                        self.sim.step(_GA.safe(action, recipe=recipe.key),
                                      confuse_on_invalid=False)
                    return
            self.overlay = None
            return
        if self.overlay == "eat":
            rows = self._hits.get("rows") or []
            for (r, item) in rows:
                if r.collidepoint(wx, wy):
                    self.sim.step(_GA.safe("eat", item=item), confuse_on_invalid=False)
                    self.overlay = None
                    return
            self.overlay = None
            return
        if self.overlay == "heaven":
            send = self._hits.get("heaven_send")
            if send is not None and send.collidepoint(wx, wy):
                self.sim.set_strategy(self.heaven_text)
                self.overlay = None
                return
            clear = self._hits.get("heaven_clear")
            if clear is not None and clear.collidepoint(wx, wy):
                self.sim.set_strategy(None)
                self.heaven_text = ""
                self.overlay = None
                return
            # one-click 作戦 suggestion rows (derived from live alerts)
            for (r, directive) in self._hits.get("heaven_suggestions", []):
                if r.collidepoint(wx, wy):
                    self.sim.set_strategy(directive)
                    self.overlay = None
                    return
            return
        if self.overlay == "carve":
            send = self._hits.get("carve_send")
            if send is not None and send.collidepoint(wx, wy):
                self._send_carve()
                return
            cancel = self._hits.get("carve_cancel")
            if cancel is not None and cancel.collidepoint(wx, wy):
                self.overlay = None
                return
            return
        if self.overlay == "miracle":
            # 神のレバー: a click on a lever/chip/send-button QUEUES that miracle
            # (it resolves at the night boundary). The box click is ignored (the
            # box is filled by typing); a click elsewhere stays in the palette so
            # several miracles can be cast before Esc closes it.
            for (r, kind, args) in self._hits.get("miracle_actions", []):
                if r.collidepoint(wx, wy):
                    self._queue_miracle_ui(kind, args)
                    return
            return
        if self.overlay in {"diary", "help"}:
            self.overlay = None

    def _click_pitwall(self, wx: int, wy: int) -> bool:
        hits = self._hits.get("pitwall") or {}
        nxt = hits.get("next")
        edit = hits.get("edit")
        miracle = hits.get("miracle")
        if nxt is not None and nxt.collidepoint(wx, wy):
            self._resume_next_day()
            return True
        if edit is not None and edit.collidepoint(wx, wy):
            self.overlay = "heaven"
            self.heaven_text = self.sim.advice_from_heaven or ""
            return True
        # 神のレバー: pit-wall quick access to the miracle palette.
        if miracle is not None and miracle.collidepoint(wx, wy):
            self.overlay = "miracle"
            self.miracle_text = ""
            self.miracle_status = ""
            return True
        return False

    def _click_result(self, wx: int, wy: int) -> None:
        rects = self._hits.get("result") or {}
        again = rects.get("again")
        quit_b = rects.get("quit")
        if again is not None and again.collidepoint(wx, wy):
            self._rebuild_sim()
        elif quit_b is not None and quit_b.collidepoint(wx, wy):
            self.running = False

    # -- auto-walk -----------------------------------------------------------
    def _walk_step(self, now: float) -> None:
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
            # 八識熟考: choose_or_deliberate fans out the eight識 when the toggle
            # is on OR the flesh is screaming (auto-burst); else the serial choose.
            action = self.brain.choose_or_deliberate(self.sim)
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
            with self._thread_lock:
                pending = self._pending_action
                self._pending_action = None
                busy = self._thread_busy
            if pending is not None:
                self.sim.step(pending, confuse_on_invalid=True)
                self.last_action_t = now
                self._maybe_approval_pause()
                return
            # 最速 zeroes the delay; the LLM is inference-bound so it still issues
            # at most one action per frame, just without an artificial wait.
            if not busy and now - self.last_action_t >= SPEED_DELAYS[self.speed]:
                with self._thread_lock:
                    self._thread_busy = True
                threading.Thread(target=self._llm_worker, daemon=True).start()
            return
        if now - self.last_action_t < SPEED_DELAYS[self.speed]:
            return
        # Local brain: 最速 runs several steps per frame so a whole year finishes
        # in seconds; other speeds take exactly one step per tick. Any day
        # boundary in approval mode stops the burst early to hold the pit wall.
        steps = _MAX_STEPS_PER_FRAME if self.speed == SPEED_MAX else 1
        for _ in range(steps):
            if self.sim.done:
                break
            self.sim.step(self.local_agent.choose(self.sim), confuse_on_invalid=False)
            if self._maybe_approval_pause():
                break
        self.last_action_t = now

    def _maybe_approval_pause(self) -> bool:
        """In 承認 mode, latch a pause the first time we cross a day boundary that
        the watcher has not yet acknowledged. Returns True when it just engaged so
        a 最速/loop caller can stop stepping immediately."""
        if self.speed != SPEED_APPROVAL or self.sim.done:
            return False
        if self.sim.world.day != self._approval_acked_day and not self.approval_pause:
            self.approval_pause = True
            return True
        return False

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

    def _paste_into_heaven(self) -> None:
        """CTRL+V into the 作戦 box. pygame has no IME, so Japanese is composed
        elsewhere and pasted. Best-effort across pygame's clipboard APIs; any
        failure is ignored silently."""
        text = ""
        pg = self.pg
        try:
            if hasattr(pg, "scrap") and getattr(pg.scrap, "get_init", lambda: False)():
                raw = pg.scrap.get(pg.SCRAP_TEXT)
                if raw:
                    text = raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else str(raw)
        except Exception:  # noqa: BLE001
            text = ""
        if not text:
            try:  # pygame-ce / SDL2 simple clipboard
                text = pg.scrap.get_text() or ""
            except Exception:  # noqa: BLE001
                text = ""
        if not text:
            return
        # single-line box: take the first line, respect the length cap
        text = text.replace("\r", "").split("\n", 1)[0]
        self.heaven_text = (self.heaven_text + text)[:60]

    def _paste_into_carve(self) -> None:
        """CTRL+V into the 刻む box. Mirrors _paste_into_heaven (pygame has no
        IME, so Japanese is composed elsewhere and pasted), capped at 60 chars."""
        text = ""
        pg = self.pg
        try:
            if hasattr(pg, "scrap") and getattr(pg.scrap, "get_init", lambda: False)():
                raw = pg.scrap.get(pg.SCRAP_TEXT)
                if raw:
                    text = raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else str(raw)
        except Exception:  # noqa: BLE001
            text = ""
        if not text:
            try:
                text = pg.scrap.get_text() or ""
            except Exception:  # noqa: BLE001
                text = ""
        if not text:
            return
        text = text.replace("\r", "").split("\n", 1)[0]
        self.carve_text = (self.carve_text + text)[:60]

    def _send_carve(self) -> None:
        """送る: dispatch the carve action with the composed 句, then close the
        box. A blank box (the hermit declined) just closes — carving is
        voluntary, so 送る on nothing is not an error."""
        text = self.carve_text.strip()
        self.overlay = None
        if not text:
            return
        self.sim.step(GameAction.safe("carve", text=text), confuse_on_invalid=False)

    # -- 神のレバー (共同モード) ---------------------------------------------
    def _queue_miracle_ui(self, kind: str, args: dict | None) -> None:
        """Queue a miracle from the palette. For the two 言葉 miracles (神託/夢)
        ``args`` is None and the text comes from the box. Updates the status line
        (success or the rejection reason) and clears the box on success so the
        palette stays open for the next lever (神は夜に働く — nothing happens now).
        A miracle NEVER dispatches a hand action; it only calls queue_miracle."""
        if args is None:  # text-send: 神託 / 夢
            text = self.miracle_text.strip()
            if not text:
                self.miracle_status = self.fonts.jp("言葉を入力せよ", "type the words first")
                return
            args = {"text": text}
        ok, reason = self.sim.queue_miracle(kind, args)
        from spl.core.divine import MIRACLE_LABELS

        label = MIRACLE_LABELS.get(kind, kind)
        if ok:
            self.miracle_status = self.fonts.jp(f"{label}を準備した（明日、夜が明けてから）",
                                                f"{label} queued (tomorrow, after the night)")
            if kind in ("oracle", "dream"):
                self.miracle_text = ""
        else:
            self.miracle_status = self.fonts.jp(f"✗ {reason}", f"x {reason}")

    def _paste_into_miracle(self) -> None:
        """CTRL+V into the 神託/夢 box. Mirrors _paste_into_heaven (pygame has no
        IME, so Japanese is composed elsewhere and pasted), capped at 60 chars."""
        text = ""
        pg = self.pg
        try:
            if hasattr(pg, "scrap") and getattr(pg.scrap, "get_init", lambda: False)():
                raw = pg.scrap.get(pg.SCRAP_TEXT)
                if raw:
                    text = raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else str(raw)
        except Exception:  # noqa: BLE001
            text = ""
        if not text:
            try:
                text = pg.scrap.get_text() or ""
            except Exception:  # noqa: BLE001
                text = ""
        if not text:
            return
        text = text.replace("\r", "").split("\n", 1)[0]
        self.miracle_text = (self.miracle_text + text)[:60]

    # -- events --------------------------------------------------------------
    def _handle_key(self, key) -> None:
        pg = self.pg

        if self.sim.done:
            if key in (pg.K_SPACE, pg.K_RETURN):
                self._rebuild_sim()
            elif key in (pg.K_ESCAPE, pg.K_q):
                self.running = False
            return

        if self.overlay == "heaven":
            mods = pg.key.get_mods()
            if key == pg.K_RETURN:
                self.sim.set_strategy(self.heaven_text)
                self.overlay = None
            elif key == pg.K_ESCAPE:
                self.overlay = None
            elif key == pg.K_BACKSPACE:
                self.heaven_text = self.heaven_text[:-1]
            elif key == pg.K_v and (mods & pg.KMOD_CTRL):
                self._paste_into_heaven()
            else:
                pass  # printable text arrives via TEXTINPUT (IME-composed too)
            return

        # 刻む: the carve box reuses the heaven box's keys (Enter送る / Esc取消 /
        # Backspace / CTRL+V paste). Enter dispatches the carve action.
        if self.overlay == "carve":
            mods = pg.key.get_mods()
            if key == pg.K_RETURN:
                self._send_carve()
            elif key == pg.K_ESCAPE:
                self.overlay = None
            elif key == pg.K_BACKSPACE:
                self.carve_text = self.carve_text[:-1]
            elif key == pg.K_v and (mods & pg.KMOD_CTRL):
                self._paste_into_carve()
            else:
                pass  # printable text arrives via TEXTINPUT (IME-composed too)
            return

        # 神のレバー: the palette's 神託/夢 box. Esc closes the palette; Backspace
        # edits; CTRL+V pastes. There are TWO sends (神託/夢) so Enter is reserved
        # for 神託 (the costlier 勅命) as a convenience. Levers/chips are clicked.
        if self.overlay == "miracle":
            mods = pg.key.get_mods()
            if key == pg.K_ESCAPE:
                self.overlay = None
            elif key == pg.K_BACKSPACE:
                self.miracle_text = self.miracle_text[:-1]
            elif key == pg.K_v and (mods & pg.KMOD_CTRL):
                self._paste_into_miracle()
            elif key == pg.K_RETURN and self.miracle_text.strip():
                self._queue_miracle_ui("oracle", None)
            else:
                pass  # printable text arrives via TEXTINPUT (IME-composed too)
            return

        # 承認制 pit wall (no editor/popup open): Enter = 次の日へ. Esc opens the
        # 作戦 editor rather than quitting, so a stray Esc never abandons the run.
        if self.approval_pause and self.overlay is None and self.popup is None:
            if key in (pg.K_RETURN, pg.K_SPACE):
                self._resume_next_day()
            elif key in (pg.K_t, pg.K_ESCAPE):
                self.overlay = "heaven"
                self.heaven_text = self.sim.advice_from_heaven or ""
            return

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
        if key == pg.K_k:
            # 神のレバー (共同モード) palette
            self.overlay = "miracle"
            self.miracle_text = ""
            self.miracle_status = ""
            return
        if key == pg.K_m:
            self.manual = not self.manual
            return
        # camera: +/- zoom (mouse-only parity), F toggles hermit-follow
        if key in (pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
            self._zoom_at(+1, self.mouse_win if self.mouse_map else None)
            return
        if key in (pg.K_MINUS, pg.K_KP_MINUS, pg.K_UNDERSCORE):
            self._zoom_at(-1, self.mouse_win if self.mouse_map else None)
            return
        if key == pg.K_f:
            self._toggle_follow()
            return
        if key == pg.K_g:
            self._toggle_deliberate()  # 八識熟考 watch toggle
            return
        if key in (pg.K_1, pg.K_2, pg.K_3, pg.K_4, pg.K_5):
            self._set_speed({pg.K_1: 1, pg.K_2: 2, pg.K_3: 3, pg.K_4: 4, pg.K_5: 5}[key])
            return
        if key == pg.K_SPACE:
            self.paused = not self.paused
            return
        if self.manual:
            self._handle_manual_key(key)

    def _handle_manual_key(self, key) -> None:
        if self.sim.done:
            return
        pg = self.pg
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

    def render(self, window) -> None:
        """Two-layer composite, both native-res: draw the voxel world directly
        into the map band (Layer 1), then all crisp UI on the window (Layer 2)."""
        lay = self.lay

        # -- Layer 1: the voxel world (drawn into the map-band subsurface) ---
        window.fill(pal.UI_BG)
        band = window.subsurface(lay.map_rect)
        self._render_world(band)

        # -- Layer 2: crisp UI ----------------------------------------------
        self._refresh_buttons()
        self.hud.draw(window, self.sim, self.sim.score(), lay, brain_status=self._brain_status())
        self.overlays.draw_button_bar(window, self.buttons, self.button_hover, lay)
        self._draw_window_cursor(window)
        self._draw_guide(window)
        self._draw_bubbles(window)
        self._draw_overlay(window)

    # -- static terrain cache ------------------------------------------------
    # The terrain slab is baked ONCE per (season, scale, tiles) at a canonical
    # *baked offset* into a surface sized to the full projected map. Panning and
    # following only change where that slab is blitted, so they never rebuild it
    # (a zoom *does* rebuild — different scale). ``_terrain_baked`` is the baked
    # offset (window-ish coords) the slab was composed at.
    def _terrain_baked_offset(self, sc: float) -> tuple[int, int]:
        """Offset the terrain slab is baked at: places the leftmost/​topmost
        projected pixel a small margin in from (0,0) so nothing clips the slab."""
        world = self.sim.world
        min_sx = (0 - (world.height - 1)) * iso.half_w(sc)
        margin = iso.half_w(sc)
        return -min_sx + margin, margin

    def _tiles_signature(self) -> tuple:
        world = self.sim.world
        rows = tuple("".join(row) for row in world.tiles)
        # NOTE: offsets are deliberately NOT in the signature — the slab is baked
        # at a fixed offset and re-blitted at the live camera offset.
        return (world.season, world.width, world.height,
                self.lay.sprite_scale, rows)

    def _ensure_terrain(self) -> "pygame.Surface":
        """Compose all ground blocks (no objects, no shimmer) into one cached
        full-map surface, baked at a fixed canonical offset. Rebuilt only when
        the tiles / season / scale change (NOT on pan/follow)."""
        sig = self._tiles_signature()
        if self._terrain is not None and self._terrain_sig == sig:
            return self._terrain
        pg = self.pg
        world = self.sim.world
        season = world.season
        sc = self.lay.sprite_scale
        w, h = world.width, world.height
        map_w, map_h = iso.map_pixel_size(w, h, sc)
        ox, oy = self._terrain_baked_offset(sc)
        surf = pg.Surface((ox + map_w, oy + map_h), pg.SRCALPHA)
        for (x, y) in iso.painter_order(w, h):
            tile = world.tiles[y][x]
            base = "forest" if tile == "forest" else tile
            edge = (x == 0 or y == 0 or x == w - 1 or y == h - 1)
            ground = self.factory.ground(base, season, edge=edge)
            sx, sy = iso.tile_to_screen(x, y, ox, oy, sc)
            recess = self.factory.ground_top_y(tile)
            surf.blit(ground, (sx, sy - recess))
        self._terrain = surf
        self._terrain_sig = sig
        self._terrain_baked = (ox, oy)
        return surf

    def _render_world(self, surf) -> None:
        """Draw the voxel world into the map band (band-local coords)."""
        pg = self.pg
        world = self.sim.world
        season = world.season
        frame = self._frame_index()
        lay = self.lay
        # sea/seabed backdrop, then the cached terrain slab on top. The slab is
        # baked at a fixed offset; blit it shifted by the live camera offset so
        # panning/following never rebuild it.
        surf.fill(pal.sea_backdrop(season))
        terrain = self._ensure_terrain()
        ox = self.offset_x - lay.map_rect.x
        oy = self.offset_y - lay.map_rect.y
        bake_x, bake_y = self._terrain_baked
        surf.blit(terrain, (ox - bake_x, oy - bake_y))

        sc = lay.sprite_scale
        # animated water shimmer + per-cell objects, in painter's order on top
        for (x, y) in iso.painter_order(world.width, world.height):
            tile = world.tiles[y][x]
            sx, sy = iso.tile_to_screen(x, y, ox, oy, sc)
            if tile == "water":
                recess = self.factory.ground_top_y(tile)
                surf.blit(self.factory.water_overlay(season, frame), (sx, sy))
            cx, cy = iso.tile_center(x, y, ox, oy, sc)
            cy -= self.factory.ground_top_y(tile)  # anchor to the (recessed) top
            self._draw_cell_objects(surf, x, y, tile, season, frame, cx, cy)

        self._draw_atmosphere(surf)
        self._draw_map_cursor(surf)

    def _draw_cell_objects(self, surf, x, y, tile, season, frame, cx, cy) -> None:
        pos = Position(x, y)
        world = self.sim.world
        s2 = self.factory._s(2)
        if tile == "forest":
            spr = self.factory.tree(season, variant=(x * 31 + y * 17) & 0xFF)
            surf.blit(spr, (cx - spr.get_width() // 2, cy - spr.get_height() + s2))
        elif tile == "rock":
            spr = self.factory.rock_object(season, variant=(x * 13 + y * 7) & 0xFF)
            surf.blit(spr, (cx - spr.get_width() // 2, cy - spr.get_height() + s2))
        elif tile == "home":
            spr = self.factory.house()
            surf.blit(spr, (cx - spr.get_width() // 2, cy - spr.get_height() + s2))
        elif tile == "workshop":
            spr = self.factory.workshop()
            surf.blit(spr, (cx - spr.get_width() // 2, cy - spr.get_height() + s2))

        # 古い石碑: the settlers' stone, on its fixed tile near home.
        if pos == self.sim.world.monument_pos:
            spr = self.factory.stele()
            surf.blit(spr, (cx - spr.get_width() // 2, cy - spr.get_height() + s2))

        plot = world.plots.get(pos)
        if plot is not None:
            stage = self._crop_stage(plot)
            spr = self.factory.crop(plot.crop, stage, frame)
            surf.blit(spr, (cx - spr.get_width() // 2, cy - spr.get_height() + s2))

        # Merchant stands one tile off the hero while an offer is live.
        if self.sim.current_offer is not None and pos == self._merchant_pos():
            spr = self.factory.merchant()
            surf.blit(spr, (cx - spr.get_width() // 2, cy - spr.get_height() + s2))

        if self.sim.hero.pos == pos:
            spr = self.factory.hero(frame)
            surf.blit(spr, (cx - spr.get_width() // 2, cy - spr.get_height() + s2))

    def _merchant_pos(self) -> Position:
        """A deterministic tile next to the hero to stand the merchant on."""
        world = self.sim.world
        for _, npos in world.neighbors(self.sim.hero.pos):
            if world.tile_at(npos) in {"grass", "beach", "field"}:
                return npos
        for _, npos in world.neighbors(self.sim.hero.pos):
            return npos
        return self.sim.hero.pos

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

    def _hero_win_anchor(self) -> tuple[int, int]:
        """Top of the hero sprite, in window coords (for Layer-2 bubbles)."""
        hero = self.sim.hero
        cx, cy = iso.tile_center(hero.pos.x, hero.pos.y, self.offset_x, self.offset_y,
                                 self.lay.sprite_scale)
        spr_h = self.factory.hero(0).get_height()
        return cx, cy - spr_h + self.factory._s(2)

    # -- world-layer cursor (top-face outlines, band-local coords) -----------
    def _draw_map_cursor(self, surf) -> None:
        """Hero locator, auto-walk marker, hover highlight — drawn as top-face
        diamond outlines on the world layer. The hover *tooltip* is Layer 2."""
        if self.sim.done:
            return
        lay = self.lay
        ox = self.offset_x - lay.map_rect.x
        oy = self.offset_y - lay.map_rect.y
        sc = lay.sprite_scale
        hw, hh = iso.half_w(sc), iso.half_h(sc)
        lw = max(2, int(round(2 * sc)))

        def top_center(p: Position, tile_kind: str) -> tuple[int, int]:
            cx, cy = iso.tile_center(p.x, p.y, ox, oy, sc)
            return cx, cy - self.factory.ground_top_y(tile_kind)

        world = self.sim.world
        if self.anim_t - self._session_start < 10.0:
            hp = self.sim.hero.pos
            cx, cy = top_center(hp, world.tile_at(hp))
            self.overlays.draw_hero_locator(surf, cx, cy, hw, hh, self.anim_t * 1.4, lw)
        if self.walk_target is not None:
            wt = self.walk_target
            cx, cy = top_center(wt, world.tile_at(wt))
            self.overlays.draw_walk_marker(surf, cx, cy, hw, hh, (self.anim_t * 1.6) % 1.0, lw)
        if self.overlay is None and self.popup is None and self.hover_tile is not None:
            ht = self.hover_tile
            cx, cy = top_center(ht, world.tile_at(ht))
            self.overlays.draw_tile_highlight(surf, cx, cy, hw, hh, width=lw)

    # -- Layer-2 cursor tooltip (crisp, on the window) -----------------------
    def _draw_window_cursor(self, window) -> None:
        if self.sim.done:
            return
        if (self.overlay is None and self.popup is None and self.hover_tile is not None
                and self.mouse_map is not None):
            self.overlays.draw_tooltip(window, self.mouse_win[0], self.mouse_win[1],
                                       self._tooltip_lines(self.hover_tile), self.lay)

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

    def _camera_hint(self) -> str:
        """Short, contextual camera hint appended to the guide strip. Kept brief
        so it always fits on one line beside the mode text."""
        f = self.fonts
        if self._at_fit_scale():
            # whole-island diorama: invite zooming in to read the 銘言
            return f.jp("ホイールで寄る／[追従]で仙人を追う",
                        "wheel to zoom in / [Follow] the hermit")
        if self.follow:
            return f.jp("右ドラッグで視点移動／ホイールで引く",
                        "right-drag to pan / wheel to zoom out")
        return f.jp("ホイールで引く／[追従]で仙人を中央に",
                    "wheel to zoom out / [Follow] re-centres")

    def _strategy_banner(self) -> str:
        """One-line, truncated 作戦 banner for the guide strip — always visible in
        watch mode so the standing order is never out of sight (DQの「さくせん」)."""
        f = self.fonts
        advice = getattr(self.sim, "advice_from_heaven", None)
        if not advice:
            return ""
        shown = advice if len(advice) <= 30 else advice[:29] + "…"
        return f.jp(f"作戦:「{shown}」", f"Order: \"{shown}\"")

    def _draw_guide(self, window) -> None:
        """Context guide strip across the top of the map band (Layer 2)."""
        if self.sim.done:
            return
        f = self.fonts
        sep = "  ／  " if f.has_cjk else "   "
        banner = self._strategy_banner()
        # 承認制 pit-wall hint takes over the strip at a day boundary.
        if self.approval_pause and self.overlay is None and self.popup is None:
            text = f.jp("［次の日へ］で進む ／ ［作戦］で指示",
                        "[Next day] to advance / [Strategy] to direct")
            if banner:
                text += sep + banner
            self.overlays.draw_guide(window, text, self.lay)
            return
        if self.popup is not None:
            text = f.jp("行動をクリック ／ 外側クリックか Space で閉じる",
                        "click an action / click outside or Space to close")
        elif self.walk_target is not None:
            text = f.jp("移動中… クリックで中断", "walking... click to interrupt")
        elif not self.manual:
            text = f.jp("観戦中 — 一時停止で止める、観戦⇔手動で操作",
                        "watching - Pause to stop, Watch/Manual to play")
            # the 作戦 banner rides the watch strip, ahead of the camera hint
            text += sep + (banner if banner else self._camera_hint())
        elif self.overlay is None:
            text = (f.jp("タイルをクリックして行動を選ぶ",
                         "click a tile to act")
                    + sep + self._camera_hint())
        else:
            return
        self.overlays.draw_guide(window, text, self.lay)

    # -- atmosphere (world layer: band-local coords) -------------------------
    def _band_size(self) -> tuple[int, int]:
        return self.lay.map_rect.width, self.lay.map_rect.height

    def _draw_atmosphere(self, surf) -> None:
        pg = self.pg
        world = self.sim.world
        weather = world.weather
        bw, bh = self._band_size()

        if weather in pal.WEATHER_TINT:
            tint = pg.Surface((bw, bh), pg.SRCALPHA)
            tint.fill(pal.WEATHER_TINT[weather])
            surf.blit(tint, (0, 0))
        if weather == "rain":
            self._draw_rain(surf, speed=560)
        elif weather == "storm":
            self._draw_rain(surf, speed=900)
            self._maybe_flash(surf)
        elif weather == "snow":
            self._draw_snow(surf)

        tod = pal.time_of_day_tint(self.sim.hero.ap_left, self.sim.ap_per_day)
        if tod is not None:
            ov = pg.Surface((bw, bh), pg.SRCALPHA)
            ov.fill(tod)
            surf.blit(ov, (0, 0))

        if self.fade > 0.001:
            fade = pg.Surface((bw, bh), pg.SRCALPHA)
            fade.fill((0, 0, 0, int(255 * min(1.0, self.fade))))
            surf.blit(fade, (0, 0))

    def _draw_rain(self, surf, speed: float) -> None:
        pg = self.pg
        sc = self.lay.sprite_scale
        dx = max(1, int(round(2 * sc)))
        dy = max(4, int(round(8 * sc)))
        wdt = max(1, int(round(sc)))
        for p in self.particles.rain:
            x, y = int(p[0]), int(p[1])
            pg.draw.line(surf, pal.RAIN_COLOR, (x, y), (x + dx, y + dy), wdt)

    def _draw_snow(self, surf) -> None:
        pg = self.pg
        _, bh = self._band_size()
        r = max(1, int(round(2 * self.lay.sprite_scale)))
        for p in self.particles.snow:
            x, y = int(p[0]), int(p[1])
            if 0 <= y < bh:
                pg.draw.circle(surf, pal.SNOW_COLOR, (x, y), r)

    def _maybe_flash(self, surf) -> None:
        if self.particles.flash_timer > 0:
            bw, bh = self._band_size()
            flash = self.pg.Surface((bw, bh), self.pg.SRCALPHA)
            flash.fill(pal.STORM_FLASH)
            surf.blit(flash, (0, 0))

    # -- bubbles / overlays (Layer 2) ----------------------------------------
    def _update_bubble(self, now: float) -> None:
        lines = self.sim.hero.spoken_lines
        if len(lines) > self._spoken_index:
            self._spoken_index = len(lines)
            self._bubble_text = lines[-1]
            self._bubble_until = now + 3.5

    def _draw_bubbles(self, window) -> None:
        if self.sim.done:
            return
        anchor = self._hero_win_anchor()
        now = self.anim_t
        if self.llm_enabled and self._thread_busy:
            self.hud.draw_thought(window, anchor, self.lay)
        if self._bubble_text and now < self._bubble_until:
            self.hud.draw_speech(window, self._bubble_text, anchor, self.lay)

    def _start_motto(self) -> None:
        """Fetch the final 座右の銘: the LLM hermit writes its own (off-thread,
        reading its five best lines); local runs get the deterministic one."""
        from spl.arena.leaderboard import fallback_motto

        if self.brain is None:
            self._motto = fallback_motto(self.sim)
            return
        self._motto_busy = True

        def work() -> None:
            from spl.arena.leaderboard import fallback_motto as fb

            try:
                motto = self.brain.write_motto(self.sim) or fb(self.sim)
            except Exception:  # noqa: BLE001 - the ending must never crash
                motto = fb(self.sim)
            self._motto = motto
            self._motto_busy = False

        import threading

        threading.Thread(target=work, daemon=True).start()

    def _draw_overlay(self, window) -> None:
        lay = self.lay
        if self.sim.done:
            if self._motto is None and not self._motto_busy:
                self._start_motto()
            # ぼうけんのしょ: once the motto (and its lessons) has resolved, record
            # this life exactly once.
            self._maybe_write_book()
            # 石碑の記憶: persist this hermit's voluntary carvings exactly once.
            self._maybe_write_stone()
            hover = self._result_hover()
            self._hits["result"] = self.overlays.draw_result(
                window, self.sim, lay, hover,
                motto=self._motto, motto_pending=self._motto_busy,
                book_lives=(self.book.lives if self.book is not None else 0),
                book_entry=self._book_entry,
                player_persona=str(getattr(self.brain, "player_persona", "") or ""),
            )
            return
        # 承認制 pit-wall strip at the day boundary (under any open editor).
        if self.approval_pause and self.overlay is None and self.popup is None:
            self._hits["pitwall"] = self.overlays.draw_pitwall(
                window, self.sim, lay, self.mouse_win
            )
        if self.overlay == "help":
            self.overlays.draw_help(window, lay)
        elif self.overlay == "diary":
            self.overlays.draw_diary(window, self.sim, self.diary_scroll, lay)
        elif self.overlay == "craft":
            self._hits["rows"] = self.overlays.draw_craft(
                window, self.sim, self.craft_sel, lay, getattr(self, "_overlay_hover", -1)
            )
        elif self.overlay == "eat":
            self._hits["rows"] = self.overlays.draw_eat(
                window, self.sim, lay, getattr(self, "_overlay_hover", -1)
            )
        elif self.overlay == "heaven":
            suggestions = uh.strategy_suggestions(self.fonts, self._obs_alerts())
            hits = self.overlays.draw_heaven(
                window, self.heaven_text, lay, self.sim.advice_from_heaven,
                suggestions, self.mouse_win,
            )
            self._hits["heaven_send"] = hits.get("send")
            self._hits["heaven_clear"] = hits.get("clear")
            self._hits["heaven_suggestions"] = hits.get("suggestions", [])
        elif self.overlay == "carve":
            hits = self.overlays.draw_carve(
                window, self.carve_text, lay, self.sim.stone_carvings,
                self.mouse_win,
            )
            self._hits["carve_send"] = hits.get("send")
            self._hits["carve_cancel"] = hits.get("cancel")
        elif self.overlay == "miracle":
            hits = self.overlays.draw_miracle(
                window, self.sim, lay, self.miracle_text, self.miracle_status,
                self.mouse_win,
            )
            self._hits["miracle_actions"] = hits.get("actions", [])
        elif self.popup is not None:
            self._draw_popup(window)

    def _draw_popup(self, window) -> None:
        rects = self.overlays.draw_popup(
            window, self.popup["header"], self.popup["items"],
            self.popup["anchor"], self.lay, self.popup.get("hover", -1),
        )
        self.popup["rects"] = rects

    def _result_hover(self) -> str:
        rects = self._hits.get("result") or {}
        wx, wy = self.mouse_win
        for key, r in rects.items():
            if r.collidepoint(wx, wy):
                return key
        return ""

    def _obs_alerts(self) -> list[str]:
        """Live observation alerts (reused by the 作戦 suggestion rows). Never
        crashes the frame."""
        try:
            return list(self.observer.build(self.sim).get("alerts", []))
        except Exception:  # noqa: BLE001
            return []

    # -- update --------------------------------------------------------------
    def _update(self, dt: float, now: float) -> None:
        self.anim_t += dt
        world = self.sim.world
        if world.weather in {"rain"}:
            self.particles.update_rain(dt, 1.0)
        elif world.weather == "storm":
            self.particles.update_rain(dt, 1.6)
            self.particles.flash_timer -= dt
            if self.particles.flash_timer <= -3.0 and self.particles.rng.random() < 0.02:
                self.particles.flash_timer = 0.08
        elif world.weather == "snow":
            self.particles.update_snow(dt)

        if world.day != self._last_day:
            # 最速: skip the day-change fade (it would only slow the flat-out run).
            if self.speed != SPEED_MAX:
                self.fade = 1.0
            self._last_day = world.day
        if self.fade > 0:
            self.fade = max(0.0, self.fade - dt / 0.3)

        self._update_bubble(now)

        # camera glide: refresh the follow/pan target, then ease the live offset
        # toward it. The hermit's bubble anchors via the live offset, so it rides
        # along automatically.
        self._update_follow_target()
        self._glide_camera()

        if self.sim.done:
            return
        if self.manual and self.walk_target is not None and self.overlay is None:
            self._walk_step(now)
        if self.overlay in {"craft", "diary", "help", "heaven", "eat", "carve", "miracle"}:
            return
        # 承認制: hold at the day boundary until the watcher taps [次の日へ].
        if self.approval_pause:
            return
        if not self.manual and not self.paused:
            self._watch_step(now)

    # -- main loop -----------------------------------------------------------
    def run_window(self) -> int:
        pg = self.pg
        pg.init()
        pg.display.set_caption("自給自足仙人 e:鴨長明 — SPL")
        window = pg.display.set_mode((self.lay.win_w, self.lay.win_h), pg.RESIZABLE)
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
                elif event.type == pg.VIDEORESIZE:
                    self._resize(event.w, event.h)
                    window = pg.display.set_mode((self.lay.win_w, self.lay.win_h), pg.RESIZABLE)
                elif event.type == pg.KEYDOWN:
                    self._last_unicode = getattr(event, "unicode", "")
                    self._handle_key(event.key)
                elif event.type == getattr(pg, "TEXTINPUT", -1):
                    if self.overlay == "heaven" and event.text:
                        self.heaven_text = (self.heaven_text + event.text)[:60]
                    elif self.overlay == "carve" and event.text:
                        self.carve_text = (self.carve_text + event.text)[:60]
                    elif self.overlay == "miracle" and event.text:
                        self.miracle_text = (self.miracle_text + event.text)[:60]
                elif event.type == pg.MOUSEMOTION:
                    self._handle_motion(*event.pos)
                elif event.type == pg.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self._handle_click(*event.pos)
                    elif event.button in (4, 5) and self.overlay == "diary":
                        self.diary_scroll = max(0, self.diary_scroll + (1 if event.button == 5 else -1))
                    elif event.button in (4, 5):
                        # wheel over the map (no overlay/popup) steps the zoom
                        # ladder, anchored at the cursor.
                        self._handle_wheel(1 if event.button == 4 else -1, event.pos)
                    elif event.button in (2, 3):
                        self._begin_pan(event.pos)
                elif event.type == pg.MOUSEBUTTONUP:
                    if event.button in (2, 3):
                        self._end_pan()
            want_ti = self.overlay in {"heaven", "carve", "miracle"}
            if want_ti != self._text_input_on:
                try:  # enables the OS IME (Japanese composition) over the input box
                    (pg.key.start_text_input if want_ti else pg.key.stop_text_input)()
                except Exception:  # noqa: BLE001 - dummy/headless drivers
                    pass
                self._text_input_on = want_ti
            self._update(dt, now)
            self.render(window)
            pg.display.flip()
            if smoke and time.time() >= deadline:
                self.running = False
        pg.quit()
        return 0 if (self.sim.completed or not self.sim.done) else 1

    # -- headless screenshot runner ------------------------------------------
    def run_shots(self, n: int, out_dir: str) -> int:
        pg = self.pg
        pg.init()
        window = pg.Surface((self.lay.win_w, self.lay.win_h))
        os.makedirs(out_dir, exist_ok=True)
        saved = 0
        guard = 0
        while saved < n and guard < n * 80:
            guard += 1
            self.anim_t += 1 / 30.0
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
            self.render(window)
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
        """Capture the UI states headlessly by driving the same handler/draw
        code real input would. Each frame is composited to the window surface
        and saved (so the PNGs are the real two-layer result)."""
        pg = self.pg
        pg.init()
        window = pg.Surface((self.lay.win_w, self.lay.win_h))
        os.makedirs(out_dir, exist_ok=True)

        self.manual = True
        self._fast_forward(14)
        self.sim.hero.add_item("bread", 2)
        self.sim.hero.add_item("berries", 3)
        self.sim.hero.add_item("turnip_seed", 3)
        self.sim.hero.add_item("wood", 4)
        self.sim.hero.add_item("stone", 3)
        self.sim.hero.add_item("fiber", 3)
        self.anim_t = 20.0  # past the 10s hero-locator window

        def save(name: str) -> None:
            self.render(window)
            pg.image.save(window, os.path.join(out_dir, name))

        world = self.sim.world
        hero = self.sim.hero

        def find(kind: str) -> "Position | None":
            return world.find_nearest(
                hero.pos,
                lambda p: world.tile_at(p) == kind and world.in_bounds(p),
            )

        def tile_center_win(p) -> tuple[int, int]:
            cx, cy = iso.tile_center(p.x, p.y, self.offset_x, self.offset_y,
                                     self.lay.sprite_scale)
            return cx, cy

        # (f) clean manual-idle guide strip. Hover a tile next to the hero.
        near = None
        for _, npos in world.neighbors(hero.pos):
            near = npos
            break
        if near is None:
            near = hero.pos
        self.hover_tile = near
        self.mouse_win = tile_center_win(near)
        self.mouse_map = tile_center_win(near)
        save("ui_f_guide.png")

        # (a) hover highlight + tooltip on a tile near the hero.
        target = None
        for kind in ("field", "forest", "water"):
            t = find(kind)
            if t is not None:
                target = t
                break
        if target is None:
            target = near
        self.hover_tile = target
        self.mouse_win = tile_center_win(target)
        self.mouse_map = tile_center_win(target)
        save("ui_a_hover.png")

        # (b) click popup OPEN with several rows.
        rich = world.find_nearest(
            hero.pos,
            lambda p: world.tile_at(p) in {"grass", "beach"}
            and (world.is_near(p, "forest") or world.is_near(p, "water")),
        )
        if rich is not None:
            hero.pos = rich
        popup_tile = hero.pos
        forest_adj = None
        for _, npos in world.neighbors(hero.pos):
            if world.tile_at(npos) == "forest":
                forest_adj = npos
                break
        click_tile = forest_adj or popup_tile
        self.hover_tile = None
        self._open_tile_popup(click_tile, tile_center_win(click_tile))
        if self.popup and len(self.popup["items"]) > 1:
            self.popup["hover"] = 1
        save("ui_b_popup.png")
        self.popup = None

        # (c) eat popup
        self.overlay = "eat"
        self._overlay_hover = 0
        save("ui_c_eat.png")
        self.overlay = None

        # (d) craft overlay with an affordable row hovered
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

        # (extra) watch-mode info-only popup + auto-walk marker
        self.manual = False
        self._open_tile_popup(click_tile, tile_center_win(click_tile))
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
        self.mouse_win = (0, 0)
        self.render(window)
        again = self._hits.get("result", {}).get("again")
        if again is not None:
            self.mouse_win = again.center
        save("ui_g_result.png")

        pg.quit()
        print(f"Saved the UI screenshots (required states + extras) to {out_dir}")
        return 0


def run(args: object) -> int:
    shots = int(getattr(args, "shots", 0) or 0)
    shots_ui = bool(getattr(args, "shots_ui", False))
    if shots_ui:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        app = PixelApp(args, headless=True)
        return app.run_shots_ui(getattr(args, "shot_dir", "/tmp/spl_px"))
    if shots > 0:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        app = PixelApp(args, headless=True)
        return app.run_shots(shots, getattr(args, "shot_dir", "/tmp/spl_px"))
    app = PixelApp(args, headless=False)
    return app.run_window()
