from __future__ import annotations

import argparse

from spl.arena.runner import run_cassette_arena, run_local_arena
from spl.ui.cli import run_play, run_simulate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spl", description="SPL: Self-sufficient Hero")
    sub = parser.add_subparsers(dest="command")

    play = sub.add_parser("play", help="watch or manually play the terminal game")
    _add_common(play)
    play.add_argument("--manual", action="store_true", help="control the hero manually")
    play.add_argument("--speed", type=float, default=0.12, help="delay between auto turns")
    play.add_argument("--radius", type=int, default=7, help="map view radius")
    play.add_argument("--no-clear", action="store_true", help="do not clear the terminal each frame")
    play.add_argument("--heaven", action="store_true", help="ask for one daily advice line in watch mode")
    play.set_defaults(func=run_play)

    simulate = sub.add_parser("simulate", help="run a fast non-interactive simulation")
    _add_common(simulate)
    simulate.set_defaults(func=run_simulate)

    arena = sub.add_parser("arena", help="compare cassettes on one seed, or one brain over many seeds")
    arena.add_argument("--seeds", default="42,43,44,45", help="comma-separated seeds (local sweep mode)")
    arena.add_argument("--seed", type=int, default=42, help="fixed seed when comparing cassettes")
    arena.add_argument("--days", type=int, default=112)
    arena.add_argument(
        "--cassettes",
        default=None,
        help="comma-separated cassette names or 'all' — pit them on the same seed (spec §6.3)",
    )
    arena.add_argument("--parallel", type=int, default=1, help="run cassettes concurrently")
    arena.set_defaults(func=run_arena)

    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=112)
    parser.add_argument("--llm", action="store_true", help="use OpenAI-compatible cassette when possible")
    parser.add_argument("--cassette", default="Qwen勇者", help="cassette name from config/models.toml")


def run_arena(args: object) -> int:
    if args.cassettes:
        from spl.agent.llm_client import load_cassettes
        from spl.core.sim import PROJECT_ROOT

        available = load_cassettes(PROJECT_ROOT / "config" / "models.toml")
        by_name = {c.name: c for c in available}
        if args.cassettes.strip().lower() == "all":
            chosen = available
        else:
            names = [n.strip() for n in args.cassettes.split(",") if n.strip()]
            chosen = [by_name[n] for n in names if n in by_name]
            missing = [n for n in names if n not in by_name]
            if missing:
                print(f"Unknown cassette(s): {', '.join(missing)}. Available: {', '.join(by_name)}")
            if not chosen:
                return 2
        print(
            f"Arena on seed {args.seed} for {args.days} days — "
            f"cassettes: {', '.join(c.name for c in chosen)}"
        )
        print(run_cassette_arena(args.seed, args.days, chosen, parallel=args.parallel))
        return 0
    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    print(run_local_arena(seeds, args.days))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args(["play"])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

