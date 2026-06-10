from __future__ import annotations

import os
import shlex
import sys
import time

from spl.agent.llm_client import OpenAICompatibleBrain, find_cassette
from spl.agent.observer import ObservationBuilder
from spl.agent.policy import LocalPolicyAgent
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


def run_play(args: object) -> int:
    sim = Simulation(seed=args.seed, max_days=args.days)
    local_agent = LocalPolicyAgent()
    brain = _make_brain(args)
    observer = ObservationBuilder()
    last_day = sim.world.day

    while not sim.done:
        if not args.no_clear:
            _clear()
        print_frame(sim, observer, radius=args.radius)
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
                advice = input("\nVoice from heaven for today (Enter to skip): ").strip()
                sim.advice_from_heaven = advice or None
                last_day = sim.world.day
            action = choose_action(sim, brain, local_agent, llm_enabled=args.llm)
            sim.step(action, confuse_on_invalid=args.llm)
            if args.speed:
                time.sleep(args.speed)

    if not args.no_clear:
        _clear()
    print_result(sim)
    return 0 if sim.completed else 1


def run_simulate(args: object) -> int:
    sim = Simulation(seed=args.seed, max_days=args.days)
    local_agent = LocalPolicyAgent()
    brain = _make_brain(args)
    max_turns = args.days * 50
    turns = 0
    while not sim.done and turns < max_turns:
        action = choose_action(sim, brain, local_agent, llm_enabled=args.llm)
        sim.step(action, confuse_on_invalid=args.llm)
        turns += 1
    if turns >= max_turns and not sim.done:
        sim.result_reason = "Stopped by max turn guard."
    print_result(sim)
    print(f"Turns: {turns}")
    if turns >= max_turns and not sim.done:
        print("Stopped by max turn guard.")
        return 2
    return 0 if sim.completed else 1


def choose_action(sim: Simulation, brain: object | None, local_agent: LocalPolicyAgent, llm_enabled: bool) -> GameAction:
    if llm_enabled and brain is not None:
        try:
            return brain.choose(sim)
        except Exception as exc:
            sim.log(f"LLM unavailable, local policy takes over this turn: {exc}")
    return local_agent.choose(sim)


def _make_brain(args: object) -> object | None:
    if not getattr(args, "llm", False):
        return None
    cassette = find_cassette(PROJECT_ROOT / "config" / "models.toml", getattr(args, "cassette", None))
    if not cassette.base_url:
        return None
    return OpenAICompatibleBrain(cassette)


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


def print_frame(sim: Simulation, observer: ObservationBuilder, radius: int | None = 7) -> None:
    hero = sim.hero
    print("SPL: Self-sufficient Hero")
    print("=" * 72)
    print(sim.status_line())
    print(f"Score {sim.score()} | Confusion {hero.confusion_count} | Seed {sim.seed}")
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


def print_diary(sim: Simulation) -> None:
    print("\nDiary")
    print("=" * 72)
    if not sim.memory.diary:
        print("(No diary entries yet.)")
        return
    for entry in sim.memory.diary[-7:]:
        print(entry.text)
        print("-" * 40)


def print_result(sim: Simulation) -> None:
    print("SPL Result")
    print("=" * 72)
    print(sim.result_reason or ("Still alive." if sim.hero.alive else "The hero fell."))
    print(sim.status_line())
    print(f"Score: {sim.score()}")
    print(f"Days survived: {sim.hero.days_survived}")
    print(f"Confusions: {sim.hero.confusion_count}")
    print(f"Civilization points: {sim.hero.civilization_points()}")
    print()
    print("Inventory:")
    for item, amount in sorted(sim.hero.inventory.items()):
        if amount > 0:
            print(f"  {item}: {amount}")
    print()
    print("Best lines:")
    for line in sim.hero.spoken_lines[-5:]:
        print(f"  \"{line}\"")
    print()
    print_diary(sim)


def _clear() -> None:
    if sys.stdout.isatty():
        os.system("cls" if os.name == "nt" else "clear")
