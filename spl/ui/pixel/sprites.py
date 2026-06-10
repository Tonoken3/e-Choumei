from __future__ import annotations

"""Procedural sprite factory.

Every tile, object and the hero are drawn in code onto pygame Surfaces and
cached. No image asset files exist. Surfaces are cached per (tile, season,
frame) so a season change simply rebuilds with a new palette via a new cache
key. ``random`` is used only at build time for speckle/furrow texture and never
touches the simulation's deterministic RNG; a fixed local Random keeps the art
stable across rebuilds of the same key.
"""

import random
from typing import TYPE_CHECKING

from . import palette as pal
from .iso import HALF_H, HALF_W, TILE_H, TILE_W

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pygame


class SpriteFactory:
    def __init__(self, pygame_module: "pygame") -> None:
        self.pg = pygame_module
        self._ground: dict[tuple[str, str], "pygame.Surface"] = {}
        self._objects: dict[tuple[str, str, int], "pygame.Surface"] = {}
        self._crops: dict[tuple[str, int, int], "pygame.Surface"] = {}
        self._hero: dict[int, "pygame.Surface"] = {}
        self._misc: dict[str, "pygame.Surface"] = {}

    # -- ground diamonds -----------------------------------------------------
    def _diamond_points(self, top_y: int = 0) -> list[tuple[int, int]]:
        # Diamond inside a TILE_W x TILE_H box, top corner at (HALF_W, top_y).
        return [
            (HALF_W, top_y),
            (TILE_W, top_y + HALF_H),
            (HALF_W, top_y + TILE_H),
            (0, top_y + HALF_H),
        ]

    def ground(self, tile: str, season: str) -> "pygame.Surface":
        key = (tile, season)
        cached = self._ground.get(key)
        if cached is not None:
            return cached
        surf = self._build_ground(tile, season)
        self._ground[key] = surf
        return surf

    def _build_ground(self, tile: str, season: str) -> "pygame.Surface":
        pg = self.pg
        palettes = pal.season_tile_palettes(season)
        p = palettes.get(tile, palettes["grass"])
        # Raised tiles (rock, home, workshop) get a thicker side wall.
        depth = 6
        if tile == "rock":
            depth = 9
        surf = pg.Surface((TILE_W, TILE_H + depth), pg.SRCALPHA)
        rng = random.Random(hash((tile, season)) & 0xFFFFFFFF)

        # side walls (left + right) for a little height
        left = [(0, HALF_H), (HALF_W, TILE_H), (HALF_W, TILE_H + depth), (0, HALF_H + depth)]
        right = [(HALF_W, TILE_H), (TILE_W, HALF_H), (TILE_W, HALF_H + depth), (HALF_W, TILE_H + depth)]
        side = p.side
        side_dark = tuple(max(0, c - 24) for c in side)
        pg.draw.polygon(surf, side, left)
        pg.draw.polygon(surf, side_dark, right)

        # top face
        pg.draw.polygon(surf, p.top, self._diamond_points())
        pg.draw.polygon(surf, p.edge, self._diamond_points(), 1)

        self._texture_top(surf, tile, season, p, rng)
        return surf

    def _texture_top(self, surf, tile, season, p, rng) -> None:
        pg = self.pg

        def on_top(px: int, py: int) -> bool:
            # inside the top diamond (Manhattan test in tile space)
            return abs(px - HALF_W) / HALF_W + abs(py - HALF_H) / HALF_H <= 0.92

        if tile in {"grass", "home"}:
            for _ in range(10):
                px, py = rng.randint(2, TILE_W - 2), rng.randint(1, TILE_H - 1)
                if on_top(px, py):
                    surf.set_at((px, py), p.accent)
        elif tile == "beach":
            for _ in range(14):
                px, py = rng.randint(2, TILE_W - 2), rng.randint(1, TILE_H - 1)
                if on_top(px, py):
                    surf.set_at((px, py), p.accent if rng.random() < 0.5 else p.edge)
        elif tile == "field":
            # darker furrow lines running along the iso grain
            for i in range(-1, 2):
                a = (HALF_W - 9, HALF_H + i * 3)
                b = (HALF_W + 9, HALF_H + i * 3)
                pg.draw.line(surf, p.side, a, b, 1)
        elif tile == "water":
            if season == "winter":
                # ice crack lines
                pg.draw.line(surf, p.accent, (HALF_W - 6, HALF_H - 2), (HALF_W + 3, HALF_H + 3), 1)
                pg.draw.line(surf, p.accent, (HALF_W + 2, HALF_H - 3), (HALF_W + 8, HALF_H + 2), 1)
            # shimmer handled by the 2-frame water object below
        elif tile == "rock":
            for _ in range(8):
                px, py = rng.randint(4, TILE_W - 4), rng.randint(2, TILE_H - 2)
                if on_top(px, py):
                    surf.set_at((px, py), p.accent if rng.random() < 0.5 else p.edge)

    # -- water shimmer (2-frame object overlay) ------------------------------
    def water_overlay(self, season: str, frame: int) -> "pygame.Surface":
        key = ("water_shimmer", season, frame)
        cached = self._objects.get(key)
        if cached is not None:
            return cached
        pg = self.pg
        surf = pg.Surface((TILE_W, TILE_H), pg.SRCALPHA)
        if season == "winter":
            self._objects[key] = surf  # ice does not shimmer
            return surf
        p = pal.season_tile_palettes(season)["water"]
        shine = p.accent
        # two horizontal glints, offset by frame for a slow shimmer
        y = HALF_H - 2 + (frame * 3)
        pg.draw.line(surf, shine, (HALF_W - 6, y), (HALF_W - 1, y), 1)
        pg.draw.line(surf, shine, (HALF_W + 2, y + 2), (HALF_W + 7, y + 2), 1)
        self._objects[key] = surf
        return surf

    # -- trees ---------------------------------------------------------------
    def tree(self, season: str, frame: int = 0) -> "pygame.Surface":
        key = ("tree", season, 0)
        cached = self._objects.get(key)
        if cached is not None:
            return cached
        pg = self.pg
        # 24px tall sprite, anchored so its base sits on the tile centre.
        w, h = 22, 26
        surf = pg.Surface((w, h), pg.SRCALPHA)
        cx = w // 2
        # trunk: 2px wide
        trunk_top = h - 9
        pg.draw.rect(surf, pal.TREE_TRUNK, (cx - 1, trunk_top, 2, 9))
        canopy = pal.tree_canopy(season)
        canopy_dark = tuple(max(0, c - 28) for c in canopy)
        if season == "winter":
            # dark pine triangle + snow cap
            pts = [(cx, 1), (cx - 9, trunk_top + 1), (cx + 9, trunk_top + 1)]
            pg.draw.polygon(surf, canopy, pts)
            pg.draw.polygon(surf, canopy_dark, pts, 1)
            pg.draw.polygon(surf, pal.TREE_SNOW, [(cx, 1), (cx - 4, 7), (cx + 4, 7)])
            pg.draw.line(surf, pal.TREE_SNOW, (cx - 6, trunk_top - 3), (cx + 6, trunk_top - 3), 1)
        else:
            # rounded blob canopy (overlapping circles)
            pg.draw.circle(surf, canopy_dark, (cx, 9), 9)
            pg.draw.circle(surf, canopy, (cx - 3, 8), 6)
            pg.draw.circle(surf, canopy, (cx + 3, 10), 6)
            pg.draw.circle(surf, canopy, (cx, 6), 6)
            if season == "autumn":
                hi = tuple(min(255, c + 30) for c in canopy)
                pg.draw.circle(surf, hi, (cx - 2, 6), 3)
        self._objects[key] = surf
        return surf

    # -- house / workshop ----------------------------------------------------
    def house(self) -> "pygame.Surface":
        cached = self._misc.get("house")
        if cached is not None:
            return cached
        pg = self.pg
        w, h = 28, 26
        surf = pg.Surface((w, h), pg.SRCALPHA)
        # walls
        wall = pg.Rect(5, 12, 18, 12)
        pg.draw.rect(surf, pal.HOUSE_WALL, wall)
        pg.draw.rect(surf, pal.HOUSE_ROOF_DARK, wall, 1)
        # gable roof
        roof = [(3, 12), (w // 2, 2), (25, 12)]
        pg.draw.polygon(surf, pal.HOUSE_ROOF, roof)
        pg.draw.polygon(surf, pal.HOUSE_ROOF_DARK, roof, 1)
        # door
        pg.draw.rect(surf, pal.HOUSE_DOOR, (w // 2 - 2, 16, 4, 8))
        # window
        pg.draw.rect(surf, (140, 196, 220), (8, 15, 3, 3))
        self._misc["house"] = surf
        return surf

    def workshop(self) -> "pygame.Surface":
        cached = self._misc.get("workshop")
        if cached is not None:
            return cached
        pg = self.pg
        w, h = 26, 20
        surf = pg.Surface((w, h), pg.SRCALPHA)
        # workbench
        pg.draw.rect(surf, pal.WORKSHOP_BENCH, (4, 9, 18, 5))
        pg.draw.rect(surf, pal.WORKSHOP_BENCH_DARK, (5, 14, 3, 5))
        pg.draw.rect(surf, pal.WORKSHOP_BENCH_DARK, (18, 14, 3, 5))
        # anvil
        pg.draw.rect(surf, pal.WORKSHOP_ANVIL, (10, 4, 7, 4))
        pg.draw.rect(surf, pal.WORKSHOP_ANVIL, (12, 8, 3, 2))
        self._misc["workshop"] = surf
        return surf

    # -- crops ---------------------------------------------------------------
    def crop(self, crop_key: str, stage: int, frame: int = 0) -> "pygame.Surface":
        # stage 0..3; sparkle blink only at stage 3 uses frame
        f = frame if stage >= 3 else 0
        key = (crop_key, stage, f)
        cached = self._crops.get(key)
        if cached is not None:
            return cached
        pg = self.pg
        w, h = 18, 18
        surf = pg.Surface((w, h), pg.SRCALPHA)
        cx = w // 2
        base = h - 3
        color = pal.CROP_COLORS.get(crop_key, (210, 210, 210))
        if stage == 0:  # sprout
            pg.draw.line(surf, pal.CROP_LEAF, (cx, base), (cx, base - 3), 1)
            surf.set_at((cx - 1, base - 3), pal.CROP_LEAF)
            surf.set_at((cx + 1, base - 3), pal.CROP_LEAF)
        elif stage == 1:  # mid
            pg.draw.line(surf, pal.CROP_LEAF, (cx, base), (cx, base - 6), 1)
            pg.draw.line(surf, pal.CROP_LEAF, (cx, base - 3), (cx - 3, base - 5), 1)
            pg.draw.line(surf, pal.CROP_LEAF, (cx, base - 3), (cx + 3, base - 5), 1)
        elif stage == 2:  # tall
            pg.draw.line(surf, pal.CROP_LEAF, (cx, base), (cx, base - 9), 1)
            pg.draw.line(surf, pal.CROP_LEAF, (cx, base - 6), (cx - 4, base - 9), 1)
            pg.draw.line(surf, pal.CROP_LEAF, (cx, base - 6), (cx + 4, base - 9), 1)
            surf.set_at((cx, base - 9), color)
        else:  # ready
            pg.draw.line(surf, pal.CROP_LEAF, (cx, base), (cx, base - 8), 1)
            pg.draw.circle(surf, color, (cx, base - 9), 4)
            hi = tuple(min(255, c + 28) for c in color)
            pg.draw.circle(surf, hi, (cx - 1, base - 10), 2)
            if f:  # blinking sparkle star
                star = (cx + 4, base - 13)
                pg.draw.line(surf, pal.CROP_SPARKLE, (star[0] - 2, star[1]), (star[0] + 2, star[1]), 1)
                pg.draw.line(surf, pal.CROP_SPARKLE, (star[0], star[1] - 2), (star[0], star[1] + 2), 1)
        self._crops[key] = surf
        return surf

    # -- hero ----------------------------------------------------------------
    def hero(self, frame: int = 0) -> "pygame.Surface":
        cached = self._hero.get(frame)
        if cached is not None:
            return cached
        pg = self.pg
        w, h = 12, 16
        surf = pg.Surface((w, h), pg.SRCALPHA)
        cx = w // 2
        bob = 1 if frame else 0  # 2-frame idle bob
        top = 2 + bob
        # head
        pg.draw.circle(surf, pal.HERO_SKIN, (cx, top + 2), 2)
        surf.set_at((cx - 1, top), pal.HERO_HAIR)
        surf.set_at((cx, top), pal.HERO_HAIR)
        surf.set_at((cx + 1, top), pal.HERO_HAIR)
        # tunic body
        body = pg.Rect(cx - 3, top + 4, 6, 6)
        pg.draw.rect(surf, pal.HERO_TUNIC, body)
        pg.draw.rect(surf, pal.HERO_TUNIC_DARK, body, 1)
        # legs
        pg.draw.line(surf, pal.HERO_OUTLINE, (cx - 1, top + 10), (cx - 1, top + 13), 1)
        pg.draw.line(surf, pal.HERO_OUTLINE, (cx + 1, top + 10), (cx + 1, top + 13), 1)
        self._hero[frame] = surf
        return surf

    # -- merchant marker -----------------------------------------------------
    def merchant(self) -> "pygame.Surface":
        cached = self._misc.get("merchant")
        if cached is not None:
            return cached
        pg = self.pg
        w, h = 12, 16
        surf = pg.Surface((w, h), pg.SRCALPHA)
        cx = w // 2
        pg.draw.circle(surf, (224, 204, 150), (cx, 4), 2)  # head
        body = pg.Rect(cx - 3, 6, 6, 7)
        pg.draw.rect(surf, (120, 96, 168), body)  # purple cloak
        pg.draw.rect(surf, (84, 64, 120), body, 1)
        # pack
        pg.draw.rect(surf, (150, 110, 70), (cx + 2, 7, 3, 4))
        self._misc["merchant"] = surf
        return surf
