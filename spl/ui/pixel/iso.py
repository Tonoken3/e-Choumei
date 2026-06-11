from __future__ import annotations

"""Isometric projection for the neo-retro voxel diorama (stage 4).

Every land cell is a **block**: a 2:1 isometric *top face* (a diamond) plus a
left and a right side face dropping down by ``BLOCK_H``. World cell ``(x, y)``
projects to the screen top corner of its diamond with::

    sx = (x - y) * HALF_W + offset_x
    sy = (x + y) * HALF_H + offset_y

The base tile is twice the stage-3 size (``TILE_W=64``, ``TILE_H=32``). The
world is rendered at *native* window resolution, so a ``scale`` float (the
sprite scale: 0.75 / 1.0 / 1.25 / 1.5) multiplies every geometric constant. All
functions take that ``scale`` so picking and centring stay exact at any zoom.

Heights (at scale 1.0, in screen px):

* ``BLOCK_H``    land block side height (top face sits this far above the base)
* ``WATER_DROP`` water surface recess below the land top
* ``BEACH_DROP`` beach top recess below grass top (shoreline step)
* ``SEABED_H``   how far the outer edge sides drop to the "seabed" slab

Painter's order draws tiles sorted by ``(x + y)`` (then ``y``) so nearer blocks
overwrite farther ones, and tall sprites layer correctly on top.
"""

import math

from spl.core.hero import Position

# -- base geometry (scale 1.0) -----------------------------------------------
TILE_W = 64
TILE_H = 32
HALF_W = TILE_W // 2  # 32
HALF_H = TILE_H // 2  # 16

# -- voxel heights (scale 1.0, screen px) ------------------------------------
BLOCK_H = 16      # land block side height
WATER_DROP = 8    # water top sits this far below a land top
BEACH_DROP = 4    # beach top sits this far below grass top
SEABED_H = 22     # outer edge sides drop this far to the seabed slab

# Discrete sprite scales the SpriteFactory caches at.
SPRITE_SCALES = (0.75, 1.0, 1.25, 1.5)


def s_round(v: float, scale: float) -> int:
    return int(round(v * scale))


def half_w(scale: float = 1.0) -> int:
    return s_round(HALF_W, scale)


def half_h(scale: float = 1.0) -> int:
    return s_round(HALF_H, scale)


def block_h(scale: float = 1.0) -> int:
    return s_round(BLOCK_H, scale)


def water_drop(scale: float = 1.0) -> int:
    return s_round(WATER_DROP, scale)


def beach_drop(scale: float = 1.0) -> int:
    return s_round(BEACH_DROP, scale)


def seabed_h(scale: float = 1.0) -> int:
    return s_round(SEABED_H, scale)


def tile_to_screen(x: int, y: int, offset_x: int, offset_y: int,
                   scale: float = 1.0) -> tuple[int, int]:
    """Top corner of the (top-face) diamond for world cell (x, y)."""
    sx = (x - y) * half_w(scale) + offset_x
    sy = (x + y) * half_h(scale) + offset_y
    return sx, sy


def tile_center(x: int, y: int, offset_x: int, offset_y: int,
                scale: float = 1.0) -> tuple[int, int]:
    """Centre of the diamond's top face — where objects/hero are anchored."""
    sx, sy = tile_to_screen(x, y, offset_x, offset_y, scale)
    return sx + half_w(scale), sy + half_h(scale)


def screen_to_tile(sx: int, sy: int, offset_x: int, offset_y: int,
                   scale: float = 1.0) -> Position:
    """Inverse projection for mouse picking. Returns the world cell whose
    *top-face diamond* contains the screen point. Exact inverse of
    :func:`tile_center` (round-trips). Not clamped to bounds."""
    hw = half_w(scale)
    hh = half_h(scale)
    rx = (sx - offset_x - hw) / hw  # = (x - y)
    ry = (sy - offset_y - hh) / hh  # = (x + y)
    fx = (rx + ry) / 2.0
    fy = (ry - rx) / 2.0
    return Position(int(math.floor(fx + 0.5)), int(math.floor(fy + 0.5)))


def painter_order(width: int, height: int) -> list[tuple[int, int]]:
    """All cells sorted back-to-front for the painter's algorithm."""
    cells = [(x, y) for y in range(height) for x in range(width)]
    cells.sort(key=lambda c: (c[0] + c[1], c[1]))
    return cells


def map_pixel_size(width: int, height: int, scale: float = 1.0) -> tuple[int, int]:
    """Bounding box of the projected block grid (ground blocks, incl. heights)."""
    w = (width + height) * half_w(scale)
    # top face span + the tallest drop (seabed at the far edge) below it
    h = (width + height) * half_h(scale) + tile_h_px(scale) + seabed_h(scale)
    return w, h


def tile_w_px(scale: float = 1.0) -> int:
    return half_w(scale) * 2


def tile_h_px(scale: float = 1.0) -> int:
    return half_h(scale) * 2


def centering_offset(width: int, height: int, viewport_w: int, viewport_h: int,
                     scale: float = 1.0, overhang_top: int = 0) -> tuple[int, int]:
    """Offset that centres the whole island block-slab in the viewport.
    ``overhang_top`` leaves headroom above the ground for tall sprites
    (trees, hero, house roof)."""
    map_w, map_h = map_pixel_size(width, height, scale)
    # Leftmost screen-x is produced by cell (0, height-1): (0-(h-1))*HALF_W.
    min_sx = (0 - (height - 1)) * half_w(scale)
    offset_x = (viewport_w - map_w) // 2 - min_sx
    offset_y = (viewport_h - map_h) // 2 + int(round(overhang_top))
    return offset_x, offset_y


def fit_scale(width: int, height: int, band_w: int, band_h: int,
              headroom: int = 96) -> float:
    """Pick the largest discrete sprite scale whose island footprint fits the
    given map band (with vertical headroom for tall sprites)."""
    best = SPRITE_SCALES[0]
    for sc in SPRITE_SCALES:
        mw, mh = map_pixel_size(width, height, sc)
        if mw <= band_w and mh + headroom <= band_h:
            best = sc
    return best
