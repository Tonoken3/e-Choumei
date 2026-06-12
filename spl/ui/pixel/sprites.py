from __future__ import annotations

"""Procedural voxel-sprite factory (stage 4: neo-retro 3D look).

Every tile, object and character is drawn in code with ``pygame.draw.polygon``
onto alpha Surfaces and cached. There are no image assets. The look is
**flat-shaded isometric voxels** (Roblox / Minecraft vibe): each thing is one
or more axonometric cuboids with three visible faces — a lit *top*, a darker
*left* (~72%) and a darker *right* (~52%) — plus a thin edge line for
definition, under a single consistent sun.

Everything is rendered at native resolution (no low-res upscale). A ``scale``
float (0.75 / 1.0 / 1.25 / 1.5 — the *sprite scale*) multiplies the geometry so
the caches stay bounded while the island can be sized to fit the map band. The
factory is created for one scale; the app rebuilds it if the scale changes.

``random`` is used only at build time for deterministic speckle/furrow texture
and per-position size jitter; it never touches the simulation RNG.
"""

import random
from typing import TYPE_CHECKING

from . import iso
from . import palette as pal

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pygame


def _cuboid_faces(hw: int, hh: int, height: int):
    """Return (top, left, right) polygon point-lists for an axonometric cuboid
    whose top-face diamond has half-width ``hw``, half-height ``hh`` and whose
    sides drop ``height`` px. Coordinates are relative to the diamond's top
    corner at (hw, 0) inside a ``2*hw`` wide box."""
    tw = hw * 2
    top = [(hw, 0), (tw, hh), (hw, 2 * hh), (0, hh)]
    left = [(0, hh), (hw, 2 * hh), (hw, 2 * hh + height), (0, hh + height)]
    right = [(hw, 2 * hh), (tw, hh), (tw, hh + height), (hw, 2 * hh + height)]
    return top, left, right


class SpriteFactory:
    """Caches voxel surfaces for one sprite scale. Keys include the scale via a
    fresh factory per scale (the app swaps factories on zoom change)."""

    def __init__(self, pygame_module: "pygame", scale: float = 1.0) -> None:
        self.pg = pygame_module
        self.scale = float(scale)
        self.hw = iso.half_w(self.scale)
        self.hh = iso.half_h(self.scale)
        self.block_h = iso.block_h(self.scale)
        self.water_drop = iso.water_drop(self.scale)
        self.beach_drop = iso.beach_drop(self.scale)
        self.seabed_h = iso.seabed_h(self.scale)
        self._ground: dict[tuple, "pygame.Surface"] = {}
        self._objects: dict[tuple, "pygame.Surface"] = {}
        self._crops: dict[tuple, "pygame.Surface"] = {}
        self._hero: dict[int, "pygame.Surface"] = {}
        self._misc: dict[str, "pygame.Surface"] = {}

    def _s(self, v: float) -> int:
        return iso.s_round(v, self.scale)

    # -- ground blocks -------------------------------------------------------
    def ground_top_y(self, tile: str) -> int:
        """How far below a land block's top a given tile's *top face* sits.

        Land = 0; beach steps down ``beach_drop``; water is recessed
        ``water_drop``. Used by the app so coastlines step down correctly and
        sprites anchor to the right surface height."""
        if tile == "water":
            return self.water_drop
        if tile == "beach":
            return self.beach_drop
        return 0

    def ground(self, tile: str, season: str, edge: bool = False) -> "pygame.Surface":
        key = (tile, season, edge)
        cached = self._ground.get(key)
        if cached is not None:
            return cached
        surf = self._build_ground(tile, season, edge)
        self._ground[key] = surf
        return surf

    def _build_ground(self, tile: str, season: str, edge: bool) -> "pygame.Surface":
        pg = self.pg
        hw, hh = self.hw, self.hh
        palettes = pal.season_tile_palettes(season)
        p = palettes.get(tile, palettes["grass"])
        recess = self.ground_top_y(tile)
        # Edge blocks drop their outer sides all the way to the seabed slab.
        side_h = (self.block_h - recess) + (self.seabed_h if edge else 0)
        side_h = max(self._s(3), side_h)
        # Canvas: top face starts at y = recess; sides drop side_h below it.
        tw = hw * 2
        total_h = recess + 2 * hh + side_h + self._s(1)
        surf = pg.Surface((tw, total_h), pg.SRCALPHA)
        rng = random.Random(hash((tile, season, "g")) & 0xFFFFFFFF)

        top, left, right = _cuboid_faces(hw, hh, side_h)
        # shift everything down by `recess` so water/beach sit lower than grass
        off = recess
        top = [(x, y + off) for (x, y) in top]
        left = [(x, y + off) for (x, y) in left]
        right = [(x, y + off) for (x, y) in right]

        pg.draw.polygon(surf, p.left, left)
        pg.draw.polygon(surf, p.right, right)
        pg.draw.polygon(surf, p.top, top)
        # thin darker edge lines between faces for definition
        pg.draw.polygon(surf, p.line, top, 1)
        pg.draw.line(surf, p.line, left[0], left[1], 1)
        pg.draw.line(surf, p.line, right[1], right[2], 1)

        self._texture_top(surf, tile, season, p, rng, off)
        return surf

    def _texture_top(self, surf, tile, season, p, rng, off) -> None:
        pg = self.pg
        hw, hh = self.hw, self.hh

        def on_top(px: int, py: int) -> bool:
            return abs(px - hw) / hw + abs(py - (hh + off)) / hh <= 0.9

        if tile in {"grass", "home"}:
            for _ in range(int(14 * self.scale)):
                px = rng.randint(2, hw * 2 - 2)
                py = rng.randint(off + 1, off + 2 * hh - 1)
                if on_top(px, py):
                    surf.set_at((px, py), p.accent)
        elif tile == "beach":
            for _ in range(int(18 * self.scale)):
                px = rng.randint(2, hw * 2 - 2)
                py = rng.randint(off + 1, off + 2 * hh - 1)
                if on_top(px, py):
                    surf.set_at((px, py), p.accent if rng.random() < 0.5 else p.edge)
        elif tile == "field":
            # furrow grooves running along the iso grain
            dark = pal.shade(p.top, 0.78)
            for i in range(-2, 3):
                dy = off + hh + i * self._s(3)
                a = (hw - self._s(15), dy)
                b = (hw + self._s(15), dy)
                pg.draw.line(surf, dark, a, b, max(1, self._s(1)))
        elif tile == "water":
            if season == "winter":
                cr = p.accent
                pg.draw.line(surf, cr, (hw - self._s(12), off + hh - self._s(4)),
                             (hw + self._s(6), off + hh + self._s(6)), max(1, self._s(1)))
                pg.draw.line(surf, cr, (hw + self._s(4), off + hh - self._s(6)),
                             (hw + self._s(16), off + hh + self._s(4)), max(1, self._s(1)))
        elif tile == "rock":
            for _ in range(int(10 * self.scale)):
                px = rng.randint(self._s(4), hw * 2 - self._s(4))
                py = rng.randint(off + 2, off + 2 * hh - 2)
                if on_top(px, py):
                    surf.set_at((px, py), p.accent if rng.random() < 0.5 else p.edge)

    # -- water shimmer (2-frame object overlay, sits on the recessed top) ----
    def water_overlay(self, season: str, frame: int) -> "pygame.Surface":
        key = ("water_shimmer", season, frame)
        cached = self._objects.get(key)
        if cached is not None:
            return cached
        pg = self.pg
        hw, hh = self.hw, self.hh
        recess = self.water_drop
        surf = pg.Surface((hw * 2, recess + 2 * hh + self._s(2)), pg.SRCALPHA)
        if season == "winter":
            self._objects[key] = surf  # ice does not shimmer
            return surf
        p = pal.season_tile_palettes(season)["water"]
        shine = pal.lighten(p.top, 60)
        y = recess + hh - self._s(2) + frame * self._s(3)
        pg.draw.line(surf, shine, (hw - self._s(11), y), (hw - self._s(2), y), max(1, self._s(1)))
        pg.draw.line(surf, shine, (hw + self._s(4), y + self._s(2)),
                     (hw + self._s(13), y + self._s(2)), max(1, self._s(1)))
        self._objects[key] = surf
        return surf

    # -- generic voxel cuboid helper (for objects) ---------------------------
    def _draw_cuboid(self, surf, ox: int, oy: int, hw: int, hh: int, height: int,
                     top_color, line=True) -> tuple[int, int]:
        """Draw a flat-shaded cuboid with its top-diamond top corner at
        (ox, oy). Returns the centre of the *top face* (for stacking)."""
        pg = self.pg
        top, left, right = _cuboid_faces(hw, hh, height)
        top = [(x + ox, y + oy) for (x, y) in top]
        left = [(x + ox, y + oy) for (x, y) in left]
        right = [(x + ox, y + oy) for (x, y) in right]
        pg.draw.polygon(surf, pal.shade(top_color, pal.FACE_LEFT), left)
        pg.draw.polygon(surf, pal.shade(top_color, pal.FACE_RIGHT), right)
        pg.draw.polygon(surf, top_color, top)
        if line:
            ln = pal.shade(top_color, 0.42)
            pg.draw.polygon(surf, ln, top, 1)
            pg.draw.line(surf, ln, left[0], left[1], 1)
            pg.draw.line(surf, ln, right[1], right[2], 1)
        return ox + hw, oy + hh  # top-face centre

    # -- trees ---------------------------------------------------------------
    def tree(self, season: str, variant: int = 0) -> "pygame.Surface":
        key = ("tree", season, variant)
        cached = self._objects.get(key)
        if cached is not None:
            return cached
        pg = self.pg
        sc = self.scale
        # deterministic per-variant size jitter so forests look organic
        jr = random.Random((variant * 2654435761) & 0xFFFFFFFF)
        jitter = 0.82 + jr.random() * 0.36  # 0.82 .. 1.18
        leaf_hw = max(4, int(round(20 * sc * jitter)))
        leaf_hh = max(2, leaf_hw // 2)
        leaf_h = int(round(20 * sc * jitter))
        trunk_hw = max(2, int(round(5 * sc)))
        trunk_hh = max(1, trunk_hw // 2)
        trunk_h = int(round(12 * sc))
        w = leaf_hw * 2 + self._s(2)
        h = leaf_hh * 2 + leaf_h + trunk_h + self._s(4)
        surf = pg.Surface((w, h), pg.SRCALPHA)
        cx = w // 2

        # trunk cuboid sits at the bottom-centre
        trunk_top_y = h - (trunk_hh * 2 + trunk_h) - self._s(1)
        self._draw_cuboid(surf, cx - trunk_hw, trunk_top_y, trunk_hw, trunk_hh,
                          trunk_h, pal.TREE_TRUNK_TOP)

        if season == "winter":
            self._winter_pine(surf, cx, trunk_top_y, leaf_hw, leaf_hh)
        else:
            leaf = pal.tree_leaf_top(season)
            top_corner_y = trunk_top_y - (leaf_hh * 2 + leaf_h) + self._s(2)
            self._draw_cuboid(surf, cx - leaf_hw, top_corner_y, leaf_hw, leaf_hh,
                              leaf_h, leaf)
            # accent dabs on the lit top face (autumn = red-orange mix)
            acc = pal.tree_leaf_accent(season)
            ar = random.Random((variant * 40503) & 0xFFFFFFFF)
            for _ in range(int(8 * sc)):
                ax = cx + ar.randint(-leaf_hw + 2, leaf_hw - 2)
                ay = top_corner_y + leaf_hh + ar.randint(-leaf_hh + 1, 0)
                if 0 <= ax < w and 0 <= ay < h:
                    surf.set_at((ax, ay), acc)
        self._objects[key] = surf
        return surf

    def _winter_pine(self, surf, cx, base_y, hw, hh) -> None:
        """Winter tree: stacked shrinking green slabs + white snow slab on top."""
        green = pal.tree_leaf_top("winter")
        slabs = 3
        y = base_y + self._s(2)
        for i in range(slabs):
            frac = 1.0 - i * 0.26
            shw = max(3, int(hw * frac))
            shh = max(2, shw // 2)
            sh = self._s(7)
            self._draw_cuboid(surf, cx - shw, y - shh * 2 - sh, shw, shh, sh, green)
            y -= sh + self._s(2)
        # snow cap slab on top
        cap_hw = max(2, int(hw * (1.0 - slabs * 0.26) * 0.9))
        cap_hh = max(1, cap_hw // 2)
        self._draw_cuboid(surf, cx - cap_hw, y - cap_hh * 2 - self._s(4),
                          cap_hw, cap_hh, self._s(4), pal.TREE_SNOW)

    # -- rock ----------------------------------------------------------------
    def rock_object(self, season: str, variant: int = 0) -> "pygame.Surface":
        key = ("rock_obj", season, variant)
        cached = self._objects.get(key)
        if cached is not None:
            return cached
        pg = self.pg
        sc = self.scale
        jr = random.Random((variant * 0x9E3779B1) & 0xFFFFFFFF)
        hw1 = max(5, int(round(16 * sc * (0.85 + jr.random() * 0.3))))
        hh1 = max(2, hw1 // 2)
        h1 = int(round(12 * sc))
        hw2 = max(3, int(hw1 * 0.6))
        hh2 = max(2, hw2 // 2)
        h2 = int(round(9 * sc))
        w = hw1 * 2 + self._s(6)
        h = hh1 * 2 + h1 + hh2 * 2 + h2 + self._s(4)
        surf = pg.Surface((w, h), pg.SRCALPHA)
        cx = w // 2
        grey = pal.ROCK_TOP
        # lower (bigger) block
        base_top_y = h - (hh1 * 2 + h1) - self._s(1)
        tcx, tcy = self._draw_cuboid(surf, cx - hw1, base_top_y, hw1, hh1, h1, grey)
        # upper (smaller) block, offset for a chunky stacked look
        ox = self._s(2) if jr.random() < 0.5 else -self._s(2)
        up_top_y = base_top_y - (hh2 * 2 + h2) + self._s(2)
        self._draw_cuboid(surf, cx - hw2 + ox, up_top_y, hw2, hh2, h2, grey)
        # seasonal top dab
        top_cx, top_cy = cx + ox, up_top_y + hh2
        if season == "summer":
            for _ in range(int(6 * sc)):
                surf.set_at((top_cx + jr.randint(-hw2 + 2, hw2 - 2),
                             top_cy + jr.randint(-hh2 + 1, 0)), pal.ROCK_MOSS)
        elif season == "winter":
            pg.draw.polygon(surf, pal.ROCK_SNOW, [
                (top_cx, up_top_y), (top_cx + hw2, up_top_y + hh2),
                (top_cx, up_top_y + hh2 * 2), (top_cx - hw2, up_top_y + hh2),
            ])
        self._objects[key] = surf
        return surf

    # -- house / workshop ----------------------------------------------------
    def house(self) -> "pygame.Surface":
        cached = self._misc.get("house")
        if cached is not None:
            return cached
        pg = self.pg
        sc = self.scale
        hw = int(round(22 * sc))
        hh = max(2, hw // 2)
        wall_h = int(round(20 * sc))
        roof_h = int(round(18 * sc))
        w = hw * 2 + self._s(4)
        h = hh * 2 + wall_h + roof_h + self._s(6)
        surf = pg.Surface((w, h), pg.SRCALPHA)
        cx = w // 2
        wall_top_y = h - (hh * 2 + wall_h) - self._s(1)
        # wall cuboid
        self._draw_cuboid(surf, cx - hw, wall_top_y, hw, hh, wall_h, pal.HOUSE_WALL)
        # door on the right face, window on the left face
        door_w = max(3, int(6 * sc))
        door_h = int(11 * sc)
        dx = cx + hw // 2
        dy = wall_top_y + hh + int(hh * 0.6)
        pg.draw.rect(surf, pal.HOUSE_DOOR, (dx, dy, door_w, door_h))
        win = max(3, int(6 * sc))
        wx = cx - hw // 2 - win // 2
        wy = wall_top_y + hh + int(hh * 0.4)
        pg.draw.rect(surf, pal.HOUSE_WINDOW, (wx, wy, win, win))
        # gable roof = two sloped faces meeting at a ridge raised ``roof_h`` above
        # the wall-top diamond's N–S axis. West face is lit, east face darker.
        n = (cx, wall_top_y)               # north (back) eave corner
        e = (cx + hw, wall_top_y + hh)     # east  (right) eave corner
        s = (cx, wall_top_y + 2 * hh)      # south (front) eave corner
        wst = (cx - hw, wall_top_y + hh)   # west  (left) eave corner
        rn = (cx, wall_top_y - roof_h)             # ridge above north
        rs = (cx, wall_top_y + 2 * hh - roof_h)    # ridge above south
        roof = pal.HOUSE_ROOF
        pg.draw.polygon(surf, roof, [rn, n, wst, s, rs])          # west slope (lit)
        pg.draw.polygon(surf, pal.shade(roof, 0.66), [rn, n, e, s, rs])  # east slope
        ln = pal.HOUSE_ROOF_DARK
        pg.draw.line(surf, ln, rn, rs, max(1, self._s(1)))        # ridge line
        pg.draw.line(surf, ln, rn, n, max(1, self._s(1)))         # back hip
        pg.draw.line(surf, ln, rs, s, max(1, self._s(1)))         # front hip
        self._misc["house"] = surf
        return surf

    def workshop(self) -> "pygame.Surface":
        cached = self._misc.get("workshop")
        if cached is not None:
            return cached
        pg = self.pg
        sc = self.scale
        hw = int(round(20 * sc))
        hh = max(2, hw // 2)
        base_h = int(round(10 * sc))
        w = hw * 2 + self._s(4)
        h = hh * 2 + base_h + int(20 * sc)
        surf = pg.Surface((w, h), pg.SRCALPHA)
        cx = w // 2
        base_top_y = h - (hh * 2 + base_h) - self._s(1)
        self._draw_cuboid(surf, cx - hw, base_top_y, hw, hh, base_h, pal.WORKSHOP_BLOCK)
        # small anvil/bench cuboids on top
        bhw = max(3, int(8 * sc))
        bhh = max(2, bhw // 2)
        self._draw_cuboid(surf, cx - bhw - self._s(3), base_top_y - bhh * 2 - int(7 * sc),
                          bhw, bhh, int(7 * sc), pal.WORKSHOP_BENCH)
        ahw = max(2, int(5 * sc))
        ahh = max(1, ahw // 2)
        self._draw_cuboid(surf, cx + self._s(3), base_top_y - ahh * 2 - int(6 * sc),
                          ahw, ahh, int(6 * sc), pal.WORKSHOP_ANVIL)
        self._misc["workshop"] = surf
        return surf

    # -- 古い石碑 (the settlers' stone) --------------------------------------
    def stele(self) -> "pygame.Surface":
        """A small weathered stone stele: two stacked grey cuboids (a squat base
        and a taller upright slab) with a lighter inscribed front face and a
        moss dab on top. Built from the same flat-shaded cuboid helpers as the
        rocks/buildings so it sits in the diorama under the one sun."""
        cached = self._misc.get("stele")
        if cached is not None:
            return cached
        pg = self.pg
        sc = self.scale
        grey = pal.ROCK_TOP
        # squat base block
        base_hw = max(5, int(round(13 * sc)))
        base_hh = max(2, base_hw // 2)
        base_h = int(round(6 * sc))
        # upright slab (narrower, tall) — the inscribed stone itself
        slab_hw = max(4, int(round(9 * sc)))
        slab_hh = max(2, slab_hw // 2)
        slab_h = int(round(26 * sc))
        w = base_hw * 2 + self._s(4)
        h = base_hh * 2 + base_h + slab_hh * 2 + slab_h + self._s(4)
        surf = pg.Surface((w, h), pg.SRCALPHA)
        cx = w // 2
        base_top_y = h - (base_hh * 2 + base_h) - self._s(1)
        self._draw_cuboid(surf, cx - base_hw, base_top_y, base_hw, base_hh, base_h, grey)
        slab_top_y = base_top_y - (slab_hh * 2 + slab_h) + self._s(2)
        self._draw_cuboid(surf, cx - slab_hw, slab_top_y, slab_hw, slab_hh, slab_h, grey)
        # lighter inscribed front (right) face: a panel of carved text-lines so
        # the stone reads as a written monument, not just a rock.
        face = pal.lighten(grey, 26)
        skew = max(1, slab_hh - self._s(2))
        rx0 = cx + self._s(2)
        ry0 = slab_top_y + slab_hh + self._s(3)
        panel_w = max(self._s(3), slab_hw - self._s(3))
        panel_h = max(self._s(6), slab_h - self._s(6))
        pg.draw.polygon(surf, face, [
            (rx0, ry0),
            (rx0 + panel_w, ry0 + skew),
            (rx0 + panel_w, ry0 + skew + panel_h),
            (rx0, ry0 + panel_h),
        ])
        carve = pal.shade(grey, 0.5)
        line_w = max(1, self._s(1))
        rows = max(3, int(5 * sc))
        for i in range(rows):
            ly = ry0 + self._s(2) + i * max(2, self._s(3))
            if ly > ry0 + panel_h - self._s(1):
                break
            pg.draw.line(surf, carve, (rx0 + self._s(1), ly),
                         (rx0 + panel_w - self._s(1), ly + skew - self._s(1)), line_w)
        # moss dab on the slab top (deterministic, like the rocks)
        mr = random.Random(0x57E1E)
        top_cx, top_cy = cx, slab_top_y + slab_hh
        for _ in range(int(5 * sc) + 1):
            mx = top_cx + mr.randint(-slab_hw + 2, slab_hw - 2)
            my = top_cy + mr.randint(-slab_hh + 1, slab_hh - 1)
            if 0 <= mx < w and 0 <= my < h:
                surf.set_at((mx, my), pal.ROCK_MOSS)
        self._misc["stele"] = surf
        return surf

    # -- crops ---------------------------------------------------------------
    def crop(self, crop_key: str, stage: int, frame: int = 0) -> "pygame.Surface":
        f = frame if stage >= 3 else 0
        key = (crop_key, stage, f)
        cached = self._crops.get(key)
        if cached is not None:
            return cached
        pg = self.pg
        sc = self.scale
        w = int(round(28 * sc))
        h = int(round(30 * sc))
        surf = pg.Surface((w, h), pg.SRCALPHA)
        cx = w // 2
        base = h - self._s(4)
        color = pal.CROP_COLORS.get(crop_key, (210, 210, 210))
        leaf = pal.CROP_LEAF

        def sprout(height_px: int, leaves: int) -> None:
            pg.draw.line(surf, leaf, (cx, base), (cx, base - height_px), max(1, self._s(2)))
            for i in range(leaves):
                yy = base - int(height_px * (0.5 + 0.25 * i))
                dx = self._s(5) + i * self._s(2)
                pg.draw.line(surf, leaf, (cx, yy), (cx - dx, yy - self._s(3)), max(1, self._s(1)))
                pg.draw.line(surf, leaf, (cx, yy), (cx + dx, yy - self._s(3)), max(1, self._s(1)))

        if stage == 0:
            sprout(int(6 * sc), 1)
        elif stage == 1:
            sprout(int(11 * sc), 2)
        elif stage == 2:
            sprout(int(16 * sc), 3)
            # a tiny voxel bud cube
            self._tiny_cube(surf, cx, base - int(16 * sc), max(2, int(3 * sc)), color)
        else:  # ready: bright accent voxel cube + sparkle
            sprout(int(17 * sc), 3)
            cube = max(3, int(5 * sc))
            self._tiny_cube(surf, cx, base - int(17 * sc), cube, color)
            if f:
                star = (cx + int(8 * sc), base - int(24 * sc))
                r = max(2, int(3 * sc))
                pg.draw.line(surf, pal.CROP_SPARKLE, (star[0] - r, star[1]),
                             (star[0] + r, star[1]), max(1, self._s(1)))
                pg.draw.line(surf, pal.CROP_SPARKLE, (star[0], star[1] - r),
                             (star[0], star[1] + r), max(1, self._s(1)))
        self._crops[key] = surf
        return surf

    def _tiny_cube(self, surf, cx: int, cy: int, half: int, color) -> None:
        """A tiny flat-shaded cube centred (top face) at (cx, cy)."""
        hw = half
        hh = max(1, half // 2)
        self._draw_cuboid(surf, cx - hw, cy - hh, hw, hh, half, color, line=False)
        hi = pal.lighten(color, 30)
        surf.set_at((cx, cy + 1), hi)

    # -- hero (mini Roblox figure) -------------------------------------------
    def hero(self, frame: int = 0) -> "pygame.Surface":
        cached = self._hero.get(frame)
        if cached is not None:
            return cached
        self._hero[frame] = self._figure(frame, pal.HERO_TUNIC, pal.HERO_SKIN,
                                          pal.HERO_LEG, pack=False)
        return self._hero[frame]

    def merchant(self) -> "pygame.Surface":
        cached = self._misc.get("merchant")
        if cached is not None:
            return cached
        self._misc["merchant"] = self._figure(0, pal.MERCHANT_ROBE, pal.MERCHANT_SKIN,
                                               pal.shade(pal.MERCHANT_ROBE, 0.6), pack=True)
        return self._misc["merchant"]

    def _figure(self, frame: int, tunic, skin, leg, pack: bool) -> "pygame.Surface":
        """A mini voxel humanoid ~28px: head cube + torso cuboid + two leg
        stubs, with a 2-frame idle bob. Built from flat-shaded cuboids."""
        pg = self.pg
        sc = self.scale
        w = int(round(24 * sc))
        h = int(round(40 * sc))
        surf = pg.Surface((w, h), pg.SRCALPHA)
        cx = w // 2
        bob = self._s(2) if frame else 0
        # legs (two stubs)
        leg_hw = max(2, int(3 * sc))
        leg_hh = max(1, leg_hw // 2)
        leg_h = int(8 * sc)
        base_y = h - leg_hh * 2 - leg_h - self._s(1)
        self._draw_cuboid(surf, cx - leg_hw * 2, base_y, leg_hw, leg_hh, leg_h, leg, line=False)
        self._draw_cuboid(surf, cx, base_y, leg_hw, leg_hh, leg_h, leg, line=False)
        # torso cuboid
        t_hw = max(4, int(7 * sc))
        t_hh = max(2, t_hw // 2)
        t_h = int(12 * sc)
        torso_top_y = base_y - t_hh * 2 - t_h + self._s(2) - bob
        self._draw_cuboid(surf, cx - t_hw, torso_top_y, t_hw, t_hh, t_h, tunic)
        if pack:
            phw = max(2, int(4 * sc))
            phh = max(1, phw // 2)
            self._draw_cuboid(surf, cx + t_hw - phw, torso_top_y + t_hh, phw, phh,
                              int(8 * sc), pal.MERCHANT_PACK, line=False)
        # head cube
        hh_w = max(3, int(5 * sc))
        hh_h = max(2, hh_w // 2)
        head_top_y = torso_top_y - hh_h * 2 - int(8 * sc) + self._s(2)
        self._draw_cuboid(surf, cx - hh_w, head_top_y, hh_w, hh_h, int(8 * sc), skin)
        # hair dab on the lit top face
        for dx in range(-hh_w + 1, hh_w):
            surf.set_at((cx + dx, head_top_y + hh_h), pal.HERO_HAIR)
        return surf
