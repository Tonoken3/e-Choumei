from __future__ import annotations

"""Island Diorama: a procedural isometric pixel-art viewer/player for SPL.

This package never touches the simulation's RNG or state mutators directly; it
only reads ``Simulation`` / ``World`` / ``Hero`` and calls ``sim.step(...)`` on
the main thread. All art is drawn procedurally onto pygame Surfaces at load and
cached. The simulation under ``spl/core`` is sacred and is not modified.
"""

__all__ = ["run_pixel"]


def run_pixel(args: object) -> int:
    """CLI entry point. Imports pygame lazily so the rest of the CLI works
    without it; prints a friendly message if pygame is missing."""
    try:
        import pygame  # noqa: F401
    except Exception:  # noqa: BLE001
        print(
            "The pixel frontend needs pygame-ce. Install it with:\n"
            "  pip install 'pygame-ce>=2.5'\n"
            "  (or: pip install '.[pixel]')"
        )
        return 2
    from .app import run as _run

    return _run(args)
