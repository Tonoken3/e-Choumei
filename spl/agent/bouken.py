from __future__ import annotations

"""ぼうけんのしょ — the cross-life lesson journal (DQ's adventure book).

When the watcher enables --book, each ended run appends an entry to a persistent
per-cassette journal: days survived, the ending, the score, and the THREE lessons
the hermit distilled for its next life (LLM-written via the epilogue call, or the
deterministic fallback). The NEXT run on the same cassette (book on) injects those
past-life lessons into the observation near the top — 前世たちが死をもって書き残
した教訓 — so the learning curve across lives becomes measurable.

The book lives entirely in the UI/agent layer (no sim core changes). Storage is
pure stdlib JSON, deterministic given the file content, written atomically
(tmp + rename) so a crash mid-write never corrupts the chronicle.
"""

import json
import os
import re
from pathlib import Path

from spl.core.sim import PROJECT_ROOT


def slugify(name: str) -> str:
    """Cassette name -> filesystem-safe slug (non-alnum -> '_')."""
    slug = re.sub(r"[^0-9A-Za-z぀-ヿ一-鿿]+", "_", (name or "").strip())
    slug = slug.strip("_")
    return slug or "default"


def book_dir() -> Path:
    """Where the journals live. SPL_BOOK_DIR overrides (used by tests); the
    default is ``savedata/`` under the project root, created lazily."""
    override = os.environ.get("SPL_BOOK_DIR")
    return Path(override) if override else (PROJECT_ROOT / "savedata")


def book_path_for(cassette_name: str) -> Path:
    """The ``bouken_<slug>.json`` path for a cassette."""
    return book_dir() / f"bouken_{slugify(cassette_name)}.json"


class BoukenNoSho:
    """A loaded adventure book. ``entries`` are dicts of
    {life, seed, days, score, ending, lessons:[str], motto}."""

    def __init__(self, path: Path, entries: list[dict] | None = None) -> None:
        self.path = Path(path)
        self.entries: list[dict] = list(entries or [])

    # -- loading -------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "BoukenNoSho":
        path = Path(path)
        entries: list[dict] = []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = None
        if isinstance(raw, dict):
            raw = raw.get("entries")
        if isinstance(raw, list):
            for row in raw:
                if isinstance(row, dict):
                    entries.append(row)
        return cls(path, entries)

    @classmethod
    def for_cassette(cls, cassette_name: str) -> "BoukenNoSho":
        return cls.load(book_path_for(cassette_name))

    # -- queries -------------------------------------------------------------
    @property
    def lives(self) -> int:
        return len(self.entries)

    def lessons_for(self, seed: int, limit: int = 6) -> list[str]:
        """The most recent lessons to hand the next life. Same-seed entries
        (they describe THIS island) come first, then the rest; within each
        group, most-recent first. Identical lesson strings are deduped."""
        same: list[str] = []
        other: list[str] = []
        for entry in reversed(self.entries):
            bucket = same if entry.get("seed") == seed else other
            for lesson in entry.get("lessons") or []:
                text = str(lesson).strip()
                if text:
                    bucket.append(text)
        out: list[str] = []
        seen: set[str] = set()
        for text in same + other:
            if text not in seen:
                seen.add(text)
                out.append(text)
            if len(out) >= limit:
                break
        return out

    # -- mutation ------------------------------------------------------------
    def append(self, entry: dict) -> dict:
        """Append one life's record and persist atomically (tmp + os.replace).
        ``life`` is filled in (1-based) when absent. Returns the stored entry."""
        record = dict(entry)
        record.setdefault("life", self.lives + 1)
        lessons = record.get("lessons") or []
        record["lessons"] = [str(s).strip() for s in lessons if str(s).strip()]
        self.entries.append(record)
        self._write()
        return record

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"entries": self.entries}, ensure_ascii=False, indent=2
        )
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self.path)


# ===========================================================================
# Wiring helpers shared by the CLI and the pixel UI.
# ===========================================================================
def inject_into_observer(observer: object, book: "BoukenNoSho", seed: int) -> None:
    """Hand a brain's ObservationBuilder the past-life lessons for this island,
    so the next run's observation carries bouken_no_sho near the top. A local
    brain has no observer attr — it simply ignores the book (it still RECORDS)."""
    if observer is None:
        return
    observer.book_lessons = book.lessons_for(seed)
    observer.book_lives = book.lives


def build_entry(sim: object, seed: int, motto: dict | None) -> dict:
    """Compose one life's record from the ended run and its final motto dict
    (which already carries 'lessons' from the LLM or the fallback)."""
    motto = motto or {}
    lessons = list(motto.get("lessons") or [])
    return {
        "seed": seed,
        "days": getattr(sim.hero, "days_survived", 0),
        "score": sim.score() if hasattr(sim, "score") else 0,
        "ending": getattr(sim, "result_reason", "") or "",
        "lessons": lessons,
        "motto": str(motto.get("motto", "")).strip(),
    }
