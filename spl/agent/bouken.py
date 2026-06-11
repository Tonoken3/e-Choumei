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
    {life, seed, days, score, ending, lessons:[str], motto}.

    家訓の編纂: alongside the per-life entries the book carries a fixed-size
    ``canon`` — up to 5 articles (条文) REVISED after each life rather than
    grown. When the canon is non-empty it is what the next life inherits, so
    near-duplicate lessons from many lives are merged into sharp standing rules
    under the compiler's selection pressure (long lives' articles earn their
    place; short lives' articles get rewritten)."""

    CANON_SIZE = 5

    def __init__(
        self,
        path: Path,
        entries: list[dict] | None = None,
        canon: list[str] | None = None,
        canon_revision: int = 0,
    ) -> None:
        self.path = Path(path)
        self.entries: list[dict] = list(entries or [])
        self.canon: list[str] = [str(s).strip() for s in (canon or []) if str(s).strip()]
        self.canon_revision: int = int(canon_revision or 0)

    # -- loading -------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "BoukenNoSho":
        path = Path(path)
        entries: list[dict] = []
        canon: list[str] = []
        canon_revision = 0
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = None
        rows = raw
        if isinstance(raw, dict):
            rows = raw.get("entries")
            # 家訓: absent in old files is fine -> empty canon, revision 0.
            canon_blob = raw.get("canon")
            if isinstance(canon_blob, dict):
                lessons = canon_blob.get("lessons")
                if isinstance(lessons, list):
                    canon = [str(s).strip() for s in lessons if str(s).strip()]
                canon_revision = int(canon_blob.get("revision") or 0)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    entries.append(row)
        return cls(path, entries, canon=canon, canon_revision=canon_revision)

    @classmethod
    def for_cassette(cls, cassette_name: str) -> "BoukenNoSho":
        return cls.load(book_path_for(cassette_name))

    # -- queries -------------------------------------------------------------
    @property
    def lives(self) -> int:
        return len(self.entries)

    def lessons_for(self, seed: int, limit: int = 6) -> list[str]:
        """What the next life inherits. 家訓: when the canon is non-empty it
        WINS — the next life carries the compiled 5 articles (no growth, no
        dilution). When there is no canon yet (old books / first lives) fall
        back to the recent/same-seed behaviour for full back-compat.

        Same-seed entries (they describe THIS island) come first, then the
        rest; within each group, most-recent first. Identical lessons deduped."""
        if self.canon:
            return list(self.canon[:limit])
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

    def set_canon(self, lessons: list[str], revision: int) -> list[str]:
        """Replace the 家訓 with a fresh (compiled) set of articles and persist
        atomically alongside the entries. The canon is capped at CANON_SIZE; the
        revision is stored as-is so callers track how many times it was編纂された."""
        self.canon = [str(s).strip() for s in (lessons or []) if str(s).strip()][: self.CANON_SIZE]
        self.canon_revision = int(revision)
        self._write()
        return list(self.canon)

    # -- compiler context ----------------------------------------------------
    def history_table(self) -> list[dict]:
        """The lineage's history for the 編纂者: each past life's number,
        lifespan, ending and the lessons it carried. The lifespan next to the
        lessons is the selection pressure — long lives' articles earned their
        place, short lives' articles failed their bearer."""
        table: list[dict] = []
        for entry in self.entries:
            table.append(
                {
                    "life": int(entry.get("life") or 0),
                    "days": int(entry.get("days") or 0),
                    "ending": str(entry.get("ending") or "").strip(),
                    "lessons": [str(s).strip() for s in (entry.get("lessons") or []) if str(s).strip()],
                }
            )
        return table

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "entries": self.entries,
                "canon": {"lessons": self.canon, "revision": self.canon_revision},
            },
            ensure_ascii=False,
            indent=2,
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


# ===========================================================================
# 家訓の編纂 — the deterministic compiler (LLM-free fallback).
# ===========================================================================
def _normalize_lesson(text: str) -> str:
    """Strip punctuation/whitespace for the dedupe comparison — so 「水を掘れ。」
    and 「水を掘れ」 collide. Keeps the letters (JP + ASCII alnum)."""
    return re.sub(r"[^0-9A-Za-z぀-ヿ一-鿿ぁ-ゟ゠-ヿ]+", "", text or "")


def _shares_long_substring(a: str, b: str, n: int = 6) -> bool:
    """True when ``a`` and ``b`` share any contiguous run of ``n`` characters —
    the near-duplicate test (e.g. four 'water' articles all share '水を')."""
    if not a or not b:
        return False
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) < n:
        return shorter in longer
    for i in range(len(shorter) - n + 1):
        if shorter[i : i + n] in longer:
            return True
    return False


def fallback_compile(book: "BoukenNoSho", cap: int = 5) -> list[str]:
    """Deterministic 編纂 when no brain is available: dedupe every life's
    lessons and keep the survivors, preferring the lessons carried by the
    LONGEST lives (newest first among ties), capped at ``cap``.

    Dedupe rule: a lesson is dropped if (after normalization) it equals OR
    shares a 6+ char substring with an already-kept (i.e. longer-lived / newer)
    lesson — so three near-identical water articles collapse to one."""
    # Rank every (lesson) by its bearer's lifespan desc, then life number desc
    # (newest first among ties), preserving the lesson order within a life.
    ranked: list[tuple[int, int, int, str]] = []
    for entry in book.entries:
        days = int(entry.get("days") or 0)
        life = int(entry.get("life") or 0)
        for pos, lesson in enumerate(entry.get("lessons") or []):
            text = str(lesson).strip()
            if text:
                ranked.append((days, life, -pos, text))
    ranked.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)

    kept: list[str] = []
    kept_norm: list[str] = []
    for _days, _life, _pos, text in ranked:
        norm = _normalize_lesson(text)
        if not norm:
            continue
        if any(_shares_long_substring(norm, prev) for prev in kept_norm):
            continue
        kept.append(text)
        kept_norm.append(norm)
        if len(kept) >= cap:
            break
    return kept
