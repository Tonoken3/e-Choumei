from __future__ import annotations

"""Isometric projection for the diorama.

Ground tiles are 32x16 diamonds (a 2:1 iso ratio). World tile (x, y) maps to
screen with::

    sx = (x - y) * 16 + offset_x
    sy = (x + y) * 8  + offset_y

where (sx, sy) is the *top* corner of the diamond. Painter's order draws tiles
sorted by (x + y) so nearer tiles overwrite farther ones.
"""

from spl.core.hero import Position

TILE_W = 32
TILE_H = 16
HALF_W = TILE_W // 2  # 16
HALF_H = TILE_H // 2  # 8


def tile_to_screen(x: int, y: int, offset_x: int, offset_y: int) -> tuple[int, int]:
    """Top corner of the diamond for world cell (x, y)."""
    sx = (x - y) * HALF_W + offset_x
    sy = (x + y) * HALF_H + offset_y
    return sx, sy


def tile_center(x: int, y: int, offset_x: int, offset_y: int) -> tuple[int, int]:
    """Centre of the diamond's top face — where objects/hero are anchored."""
    sx, sy = tile_to_screen(x, y, offset_x, offset_y)
    return sx + HALF_W, sy + HALF_H


def screen_to_tile(sx: int, sy: int, offset_x: int, offset_y: int) -> Position:
    """Inverse projection (for future mouse picking). Returns the world cell
    whose diamond contains the screen point. Not rounded to bounds."""
    rx = sx - offset_x - HALF_W
    ry = sy - offset_y - HALF_H
    # Invert the 2x2 iso matrix: x = rx/2W + ry/2H ; y = ry/2H - rx/2W
    fx = (rx / TILE_W) + (ry / TILE_H)
    fy = (ry / TILE_H) - (rx / TILE_W)
    import math

    return Position(int(math.floor(fx + 0.5)), int(math.floor(fy + 0.5)))


def painter_order(width: int, height: int) -> list[tuple[int, int]]:
    """All cells sorted back-to-front for the painter's algorithm."""
    cells = [(x, y) for y in range(height) for x in range(width)]
    cells.sort(key=lambda c: (c[0] + c[1], c[1]))
    return cells


def map_pixel_size(width: int, height: int) -> tuple[int, int]:
    """Bounding box of the projected diamond grid (ground only)."""
    w = (width + height) * HALF_W
    h = (width + height) * HALF_H + TILE_H
    return w, h


def centering_offset(width: int, height: int, viewport_w: int, viewport_h: int,
                     overhang_top: int = 28) -> tuple[int, int]:
    """Offset that centres the whole island in the viewport. ``overhang_top``
    leaves room above the ground for tall sprites (trees, hero)."""
    map_w, map_h = map_pixel_size(width, height)
    # The leftmost screen-x is produced by cell (0, height-1): (0-(h-1))*HALF_W.
    min_sx = (0 - (height - 1)) * HALF_W
    offset_x = (viewport_w - map_w) // 2 - min_sx
    offset_y = (viewport_h - map_h) // 2 + overhang_top
    return offset_x, offset_y
