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


# ===========================================================================
# 石碑の記憶 (the stone's memory) — the 刻む persistence layer.
#
# Unlike the 冒険の書 (which records every life's lessons automatically), the
# stone holds only the 句 a hermit CHOSE to carve. It persists per cassette+island
# and is INDEPENDENT of --book: the stone always remembers. Same stdlib-JSON,
# atomic-write discipline as the book.
# ===========================================================================
def stone_path_for(cassette_name: str) -> Path:
    """The ``sekihi_<slug>.json`` path for a cassette (the carved stone)."""
    return book_dir() / f"sekihi_{slugify(cassette_name)}.json"


def load_stone(path: str | Path) -> list[dict]:
    """Load the stone's carvings: a list of entries
    {seed, day, life, text}. A missing/corrupt file is an empty stone."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = raw.get("carvings") if isinstance(raw, dict) else raw
    out: list[dict] = []
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and str(row.get("text", "")).strip():
                out.append(
                    {
                        "seed": int(row.get("seed") or 0),
                        "day": int(row.get("day") or 0),
                        "life": int(row.get("life") or 0),
                        "text": str(row.get("text", "")).strip(),
                    }
                )
    return out


def append_carvings(path: str | Path, seed: int, day_texts: list[tuple[int, str]],
                    life: int = 0) -> list[dict]:
    """Append this life's carvings (each a (day, text) pair) to the stone and
    persist atomically (tmp + os.replace). Blank texts are dropped. An empty (or
    all-blank) ``day_texts`` is a true no-op — no file is written. Returns the
    full carving list after the append."""
    path = Path(path)
    entries = load_stone(path)
    added = False
    for day, text in day_texts or []:
        text = str(text).strip()
        if not text:
            continue
        entries.append({"seed": int(seed), "day": int(day), "life": int(life), "text": text})
        added = True
    if added:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"carvings": entries}, ensure_ascii=False, indent=2)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
    return entries


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
                    # 神のレバー: the共同 marker. The編纂者 reads this so an assisted
                    # life's lessons are weighed lightly (偽教訓汚染ガード) — a long
                    # life bought with miracles did not earn its articles unaided.
                    "miracles_used": int(entry.get("miracles_used") or 0),
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


def build_entry(sim: object, seed: int, motto: dict | None,
                player_persona: str = "") -> dict:
    """Compose one life's record from the ended run and its final motto dict
    (which already carries 'lessons' from the LLM or the fallback).

    入植者の来歴: when the watcher wrote a persona for this life, the entry records
    its first 120 chars so the chronicle remembers which soul lived this run (an
    empty string when no来歴 was given)."""
    motto = motto or {}
    lessons = list(motto.get("lessons") or [])
    # 神のレバー (共同): how many miracles this life leaned on. >0 marks the life
    # 共同 (co-op) — its lessons are 介助された生 and the編纂者 must discount them
    # (偽教訓汚染ガード), so the canon of an unassisted lineage is never polluted.
    divine = getattr(sim, "divine", None)
    miracles_used = int(getattr(divine, "miracles_used", 0) or 0)
    return {
        "seed": seed,
        "days": getattr(sim.hero, "days_survived", 0),
        "score": sim.score() if hasattr(sim, "score") else 0,
        "ending": getattr(sim, "result_reason", "") or "",
        "lessons": lessons,
        "motto": str(motto.get("motto", "")).strip(),
        "persona": (str(player_persona or "").strip())[:120],
        # きびしさ: which island this life was lived on (every record stays
        # 修羅-canonical, so the book reads which runs are comparable).
        "difficulty": getattr(sim, "difficulty", "修羅"),
        # 神のレバー: the共同 marker — miracle count for this life (0 = unassisted).
        "miracles_used": miracles_used,
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
    # Rank every (lesson) FIRST by whether its bearer was unassisted (神のレバー
    # 偽教訓汚染ガード: an unassisted life's lessons always outrank an assisted
    # life's), then by lifespan desc, then life number desc (newest first among
    # ties), preserving the lesson order within a life.
    ranked: list[tuple[int, int, int, int, str]] = []
    for entry in book.entries:
        days = int(entry.get("days") or 0)
        life = int(entry.get("life") or 0)
        unassisted = 0 if int(entry.get("miracles_used") or 0) > 0 else 1
        for pos, lesson in enumerate(entry.get("lessons") or []):
            text = str(lesson).strip()
            if text:
                ranked.append((unassisted, days, life, -pos, text))
    ranked.sort(key=lambda t: (t[0], t[1], t[2], t[3]), reverse=True)

    kept: list[str] = []
    kept_norm: list[str] = []
    for _unassisted, _days, _life, _pos, text in ranked:
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
