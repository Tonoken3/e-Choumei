from __future__ import annotations

import os
import shlex
import sys
import time

from spl.agent.bouken import (
    BoukenNoSho,
    append_carvings,
    build_entry,
    fallback_compile,
    inject_into_observer,
    load_stone,
    stone_path_for,
)
from spl.agent.llm_client import OpenAICompatibleBrain, find_cassette
from spl.agent.observer import ObservationBuilder
from spl.agent.policy import LocalPolicyAgent
from spl.arena.leaderboard import fallback_motto, select_meigen
from spl.core.actions import GameAction
from spl.core.crops import FOOD_VALUES
from spl.core.sim import PROJECT_ROOT, Simulation


HELP_TEXT = """Commands:
  move north|south|east|west    move one tile
  move home|forest|rock|water   move toward a target
  forage / fish / chop / mine
  till / plant turnip|wheat|tomato|pumpkin / water / harvest
  craft stone_axe|hoe|fishing_rod / build well|storage_barrel|stove|fence|house_upgrade
  cook fish|wheat|pumpkin|berries / store pumpkin|berries
  eat berries|turnip|bread|... / drink / rest / write_diary / sleep
  auto                         let the bundled local hero take one turn
  diary                        show recent diary
  help / quit
"""


def _book_cassette_name(args: object, brain: object | None) -> str:
    """The journal is keyed by cassette. LLM runs use the chosen cassette; a
    local (non-LLM) run keys on its cassette name if given, else 'Local仙人'."""
    name = getattr(args, "cassette", None)
    if getattr(args, "llm", False) and brain is not None:
        name = getattr(getattr(brain, "cassette", None), "name", None) or name
    return name or "Local仙人"


def _load_book(args: object, brain: object | None,
               sim: object | None = None) -> "BoukenNoSho | None":
    """When --book is on, load this cassette's adventure book and inject the
    past-life lessons into the (LLM) brain's observer. Returns the book so the
    caller can append the new life at the end. Local brains ignore the lessons
    but the book still records their runs via the fallback lessons.

    古い石碑: when a ``sim`` is given, carve the last 1-2 past lives' mottos onto
    the back of the stone BEFORE play, so the graves teach on day 1."""
    if not getattr(args, "book", False):
        return None
    book = BoukenNoSho.for_cassette(_book_cassette_name(args, brain))
    if brain is not None and getattr(brain, "observer", None) is not None:
        inject_into_observer(brain.observer, book, getattr(args, "seed", 0))
    if sim is not None:
        sim.set_monument_epitaphs(_book_epitaphs(book))
    return book


def _book_epitaphs(book: "BoukenNoSho") -> list[str]:
    """The last 1-2 past lives' mottos for the 石碑 back-face (most recent
    last). Skips blank mottos so an early life with no parting words leaves no
    empty carving."""
    mottos = [str(e.get("motto", "")).strip() for e in book.entries]
    return [m for m in mottos if m][-2:]


def _load_stone(args: object, brain: object | None, sim: object) -> None:
    """刻む: load this cassette+island's carved stone and hand the最新3句 of the
    SAME seed to the sim BEFORE play, so a hermit reads what past hermits chose
    to leave. INDEPENDENT of --book (the stone always remembers)."""
    seed = int(getattr(args, "seed", 0) or 0)
    entries = load_stone(stone_path_for(_book_cassette_name(args, brain)))
    same_seed = [e["text"] for e in entries if int(e.get("seed", 0)) == seed]
    sim.set_stone_carvings(same_seed[-3:])


def _persist_stone(args: object, brain: object | None, sim: object) -> None:
    """刻む: at run end, append any 句 this hermit voluntarily carved (with their
    days) to the cassette+island's stone, so the next hermit inherits them. The
    life number is the book's count when available, else 0."""
    carvings = list(getattr(sim, "carvings_made", []) or [])
    if not carvings:
        return
    life = 0
    if getattr(args, "book", False):
        try:
            life = BoukenNoSho.for_cassette(_book_cassette_name(args, brain)).lives
        except Exception:  # noqa: BLE001 - the stone must never crash a run
            life = 0
    append_carvings(stone_path_for(_book_cassette_name(args, brain)),
                    int(getattr(args, "seed", 0) or 0), carvings, life=life)


def _record_life(book: "BoukenNoSho | None", sim: object, seed: int,
                 motto: dict | None, brain: object | None = None) -> dict | None:
    """Append this ended run to the book (once), then 編纂: revise the fixed
    5-article 家訓 from the new history (LLM compiler if available, else the
    deterministic fallback). Returns the stored entry."""
    if book is None or not sim.done:
        return None
    entry = book.append(build_entry(sim, seed, motto, _player_persona(brain)))
    compile_canon(book, brain)
    return entry


def _player_persona(brain: object | None) -> str:
    """入植者の来歴 text active on the brain (empty when none / non-LLM run)."""
    return str(getattr(brain, "player_persona", "") or "")


def compile_canon(book: "BoukenNoSho", brain: object | None) -> list[str]:
    """家訓の編纂: revise the canon after a life and persist it at revision+1.
    Uses the LLM compiler when a brain is available; the deterministic
    fallback_compile otherwise, or on any compiler exception."""
    articles: list[str] | None = None
    if brain is not None and hasattr(brain, "compile_canon"):
        try:
            articles = brain.compile_canon(book)
        except Exception:  # noqa: BLE001 - the编纂 must never crash a run
            articles = None
    if not articles:
        articles = fallback_compile(book)
    return book.set_canon(articles, book.canon_revision + 1)


def run_play(args: object) -> int:
    sim = Simulation(seed=args.seed, max_days=args.days,
                     difficulty=getattr(args, "difficulty", "ふつう"))
    if getattr(args, "strategy", None):
        sim.set_strategy(args.strategy)
    local_agent = LocalPolicyAgent()
    brain = _make_brain(args)
    observer = ObservationBuilder()
    book = _load_book(args, brain if args.llm else None, sim)
    _load_stone(args, brain if args.llm else None, sim)
    last_day = sim.world.day
    if args.llm and brain is not None:
        sim.set_diarist(brain)
    turns = 0
    max_turns = args.days * 50

    while not sim.done and turns < max_turns:
        turns += 1
        if not args.no_clear:
            _clear()
        print_frame(sim, observer, radius=args.radius, brain=brain if args.llm else None)
        if args.manual:
            line = input("\nSPL> ").strip()
            special = line.lower()
            if special in {"quit", "q", "exit"}:
                print("Game saved in memory only; exiting this run.")
                return 0
            if special in {"help", "h", "?"}:
                print(HELP_TEXT)
                input("Press Enter...")
                continue
            if special == "diary":
                print_diary(sim)
                input("Press Enter...")
                continue
            if special == "auto":
                action = local_agent.choose(sim)
                sim.step(action)
                continue
            try:
                action = parse_manual_command(line)
            except ValueError as exc:
                print(f"Invalid command: {exc}")
                time.sleep(1.0)
                continue
            sim.step(action)
        else:
            if args.heaven and sys.stdin.isatty() and sim.world.day != last_day:
                current = sim.advice_from_heaven or "（なし）"
                advice = input(
                    f"\n天の声 — 現在の作戦「{current}」"
                    "（Enterで継承 / 文を入力で変更 / '-'で解除）: "
                ).strip()
                if advice in ("-", "clear"):
                    sim.set_strategy(None)
                elif advice:
                    sim.set_strategy(advice)
                # empty input keeps the standing 作戦 (it persists across days)
                last_day = sim.world.day
            action = choose_action(sim, brain, local_agent, llm_enabled=args.llm)
            sim.step(action, confuse_on_invalid=args.llm)
            if args.speed:
                time.sleep(args.speed)

    if turns >= max_turns and not sim.done:
        sim.result_reason = "Stopped by max turn guard."
    if not args.no_clear:
        _clear()
    motto = _final_motto(sim, brain if args.llm else None)
    entry = _record_life(book, sim, args.seed, motto, brain if args.llm else None)
    _persist_stone(args, brain if args.llm else None, sim)
    print_result(sim, motto=motto, brain=brain if args.llm else None,
                 book=book, book_entry=entry)
    return 0 if sim.completed else 1


def _final_motto(sim: Simulation, brain: object | None) -> dict[str, object]:
    motto = None
    if brain is not None and sim.done:
        try:
            motto = brain.write_motto(sim)
        except Exception:  # noqa: BLE001 - the ending must never crash
            motto = None
    if motto is None:
        motto = fallback_motto(sim)
    if not motto.get("highlights"):
        from spl.agent.chronicle import jp_chronicle

        motto["highlights"] = jp_chronicle(sim)
    return motto


def run_simulate(args: object) -> int:
    sim = Simulation(seed=args.seed, max_days=args.days,
                     difficulty=getattr(args, "difficulty", "ふつう"))
    if getattr(args, "strategy", None):
        sim.set_strategy(args.strategy)
    local_agent = LocalPolicyAgent()
    brain = _make_brain(args)
    book = _load_book(args, brain if args.llm else None, sim)
    _load_stone(args, brain if args.llm else None, sim)
    if args.llm and brain is not None:
        sim.set_diarist(brain)
    max_turns = args.days * 50
    turns = 0
    while not sim.done and turns < max_turns:
        action = choose_action(sim, brain, local_agent, llm_enabled=args.llm)
        sim.step(action, confuse_on_invalid=args.llm)
        turns += 1
    if turns >= max_turns and not sim.done:
        sim.result_reason = "Stopped by max turn guard."
    motto = _final_motto(sim, brain if args.llm else None)
    entry = _record_life(book, sim, args.seed, motto, brain if args.llm else None)
    _persist_stone(args, brain if args.llm else None, sim)
    print_result(sim, motto=motto, brain=brain if args.llm else None,
                 book=book, book_entry=entry)
    print(f"Turns: {turns}")
    if turns >= max_turns and not sim.done:
        print("Stopped by max turn guard.")
        return 2
    return 0 if sim.completed else 1


def choose_action(sim: Simulation, brain: object | None, local_agent: LocalPolicyAgent, llm_enabled: bool) -> GameAction:
    if llm_enabled and brain is not None:
        try:
            # 八識熟考: fan out the eight識 when --deliberate forced it OR the body
            # is screaming (auto-burst); else the serial choose. Brains without the
            # method (older fakes) fall back to choose().
            chooser = getattr(brain, "choose_or_deliberate", None) or brain.choose
            return chooser(sim)
        except Exception as exc:
            sim.log(f"LLM unavailable, local policy takes over this turn: {exc}")
    return local_agent.choose(sim)


def _make_brain(args: object) -> object | None:
    if not getattr(args, "llm", False):
        return None
    name = getattr(args, "cassette", None)
    # --cassette MAGI builds the MAGI brain in code: "MAGI" → v2 pilot (default),
    # "MAGI-V1" → v1 committee (評議と操縦の分離).
    if name in ("MAGI", "MAGI-V1"):
        from spl.agent.magi import magi_mode_for_cassette, make_magi

        return make_magi(
            PROJECT_ROOT / "config" / "models.toml",
            mode=magi_mode_for_cassette(name),
        )
    cassette = find_cassette(PROJECT_ROOT / "config" / "models.toml", name)
    if not cassette.base_url:
        return None
    cassette = _apply_tps_override(cassette, args)
    brain = OpenAICompatibleBrain(cassette)
    apply_persona(brain, args)
    # 八識熟考: --deliberate forces the fan-out every turn when the cassette has a
    # parallel budget (a body scream auto-bursts regardless of this flag).
    if getattr(args, "deliberate", False) and brain.parallel > 0:
        brain.deliberate_forced = True
    return brain


def _apply_tps_override(cassette: object, args: object) -> object:
    """--tps overrides the cassette's forced TPS (思考予算). 0 = auto-measure."""
    from dataclasses import replace

    tps = float(getattr(args, "tps", 0.0) or 0.0)
    if tps > 0:
        return replace(cassette, tps=tps)
    return cassette


def apply_persona(brain: object | None, args: object) -> None:
    """入植者の来歴: graft the player-written persona onto the brain. --persona
    appends to the cassette's own persona; --persona-replace makes the来歴 the
    whole persona. A no-op when the brain has no persona attributes (e.g. a MAGI
    council marker) or neither flag was given."""
    if brain is None or not hasattr(brain, "player_persona"):
        return
    replace_text = getattr(args, "persona_replace", None)
    append_text = getattr(args, "persona", None)
    preset = getattr(args, "persona_preset", None)
    if preset is not None:
        from spl.agent.prompts import PERSONA_PRESETS

        brain.player_persona = PERSONA_PRESETS[preset]
        brain.persona_mode = "append"
        return
    text = replace_text if replace_text is not None else append_text
    if text is not None and len(text) > 140:
        raise SystemExit(
            f"来歴は140字以内です（X投稿と同じ。いまは{len(text)}字）。魂は短く、深く。"
        )
    if replace_text is not None:
        brain.player_persona = replace_text
        brain.persona_mode = "replace"
    elif append_text is not None:
        brain.player_persona = append_text
        brain.persona_mode = "append"


def parse_manual_command(line: str) -> GameAction:
    if not line:
        raise ValueError("empty command")
    parts = shlex.split(line)
    action = parts[0].lower()
    rest = parts[1:]
    if action in {"n", "s", "e", "w", "north", "south", "east", "west"}:
        return GameAction(action="move", args={"direction": action})
    if action == "move":
        if not rest:
            raise ValueError("move needs direction or target")
        token = rest[0].lower()
        if token in {"north", "south", "east", "west", "n", "s", "e", "w"}:
            return GameAction(action="move", args={"direction": token})
        return GameAction(action="move", args={"target": token})
    if action in {"plant"}:
        return GameAction(action=action, args={"crop": rest[0].lower()} if rest else {})
    if action in {"craft"}:
        if not rest:
            raise ValueError("craft needs recipe")
        return GameAction(action=action, args={"recipe": rest[0].lower()})
    if action in {"build"}:
        if not rest:
            raise ValueError("build needs building")
        return GameAction(action=action, args={"recipe": rest[0].lower()})
    if action in {"cook", "eat", "store"}:
        return GameAction(action=action, args={"item": rest[0].lower()} if rest else {})
    if action == "trade_accept":
        return GameAction(action=action, args={"id": rest[0]} if rest else {})
    if action in {
        "till",
        "water",
        "harvest",
        "chop",
        "mine",
        "fish",
        "forage",
        "drink",
        "sleep",
        "trade_decline",
        "rest",
        "write_diary",
    }:
        return GameAction(action=action)
    raise ValueError(f"unknown action: {action}")


def print_frame(sim: Simulation, observer: ObservationBuilder, radius: int | None = 7,
                brain: object | None = None) -> None:
    hero = sim.hero
    print("SPL 『自給自足仙人 e:鴨長明』")
    print("=" * 72)
    print(sim.status_line())
    print(f"Score {sim.score()} | Confusion {hero.confusion_count} | Seed {sim.seed}")
    # 思考予算: show the serving stack's current tier + measured TPS.
    line = _brain_status(brain)
    if line:
        print(f"思考予算: {line}")
    print()
    print(sim.world.render_map(hero.pos, radius=radius))
    print()
    print("Legend: @ hero, H home, W workshop, F forest, ^ rock, ~ water, # field, * ready crop")
    print()
    inv = ", ".join(f"{item}:{amount}" for item, amount in sorted(hero.inventory.items()) if amount > 0)
    print("Inventory:", inv or "(empty)")
    food_value = sum(FOOD_VALUES[item] * amount for item, amount in hero.inventory.items() if item in FOOD_VALUES)
    print(f"Stored food value: {food_value}")
    if sim.current_offer:
        print("Merchant:", sim.current_offer.describe())
    obs = observer.build(sim)
    if obs["alerts"]:
        print("Alerts:", "; ".join(obs["alerts"]))
    if hero.spoken_lines:
        print("Hero:", hero.spoken_lines[-1])
    print()
    print("Recent log:")
    for line in sim.full_log[-8:]:
        print("  " + line)


def _brain_status(brain: object | None) -> str:
    """The brain's 思考予算 status line, or '' for a non-LLM run / before any call."""
    if brain is None or not hasattr(brain, "status_line"):
        return ""
    try:
        if getattr(brain, "calls", 0) <= 0 and getattr(brain, "forced_tps", 0) <= 0:
            return ""
        return brain.status_line()
    except Exception:  # noqa: BLE001
        return ""


def print_diary(sim: Simulation) -> None:
    print("\nDiary")
    print("=" * 72)
    if not sim.memory.diary:
        print("(No diary entries yet.)")
        return
    for entry in sim.memory.diary[-7:]:
        print(entry.text)
        print("-" * 40)


def print_result(sim: Simulation, motto: dict[str, object] | None = None,
                 brain: object | None = None, book: object | None = None,
                 book_entry: dict | None = None) -> None:
    print("SPL Result")
    print("=" * 72)
    # ぼうけんのしょ: this life's number and the three lessons it leaves the next.
    if book is not None and book_entry is not None:
        print(f"ぼうけんのしょ: {book_entry.get('life', book.lives)}回目の生")
        lessons = book_entry.get("lessons") or []
        if lessons:
            print("次の生への教訓:")
            for lesson in lessons:
                print(f"  ・{lesson}")
        print("-" * 72)
    motto = motto or fallback_motto(sim)
    print(f"座右の銘: 「{motto['motto']}」")
    if motto.get("words"):
        print(f"結びの言葉: {motto['words']}")
    # 入植者の来歴: when the watcher authored this hermit's soul, name it.
    persona = _player_persona(brain)
    if persona.strip():
        shown = persona.strip()
        shown = shown if len(shown) <= 40 else shown[:40] + "…"
        print(f"来歴: {shown}")
    print("-" * 72)
    # きびしさ: name the island only when it is NOT the canonical ふつう benchmark,
    # so a record run reads clean and a やさしい/修羅 run is honestly flagged.
    difficulty = getattr(sim, "difficulty", "ふつう")
    if difficulty != "ふつう":
        print(f"きびしさ: {difficulty}")
    print(sim.result_reason or ("Still alive." if sim.hero.alive else "The hero fell."))
    print(sim.status_line())
    print(f"Score: {sim.score()}")
    print(f"Days survived: {sim.hero.days_survived}")
    print(f"Confusions: {sim.hero.confusion_count}")
    # 作戦: how directed was the run? (0回 = unassisted)
    print(f"作戦変更: {getattr(sim, 'strategy_changes', 0)}回")
    if sim.advice_from_heaven:
        final = sim.advice_from_heaven
        shown = final if len(final) <= 40 else final[:39] + "…"
        print(f"最終作戦: 「{shown}」")
    # 思考予算 summary when an LLM brain played.
    if brain is not None and getattr(brain, "calls", 0) > 0:
        tier = brain.current_tier()
        print(
            f"思考予算: {tier.name} (avg {brain.avg_tps():.0f} t/s, "
            f"verify-corrections {brain.verify_corrections})"
        )
    print(f"Civilization points: {sim.hero.civilization_points()}")
    print()
    print("Inventory:")
    for item, amount in sorted(sim.hero.inventory.items()):
        if amount > 0:
            print(f"  {item}: {amount}")
    print()
    print("天の声の記録:")
    for line in (motto.get("highlights") or [])[:5]:
        print(f"  {line}")
    print()
    print("銘言ベスト5:")
    best = select_meigen(sim.hero.spoken_lines, 5)
    if best:
        for line in best:
            print(f"  \"{line}\"")
    else:
        print("  (the hero kept their thoughts to themselves)")
    print()
    print_diary(sim)


def _clear() -> None:
    if sys.stdout.isatty():
        os.system("cls" if os.name == "nt" else "clear")
