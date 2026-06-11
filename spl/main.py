from __future__ import annotations

import argparse

from spl.arena.runner import run_cassette_arena, run_local_arena
from spl.ui.cli import run_play, run_simulate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spl", description="SPL 『自給自足仙人 e:鴨長明』")
    sub = parser.add_subparsers(dest="command")

    play = sub.add_parser("play", help="watch or manually play the terminal game")
    _add_common(play)
    play.add_argument("--manual", action="store_true", help="control the hero manually")
    play.add_argument("--speed", type=float, default=0.12, help="delay between auto turns")
    play.add_argument("--radius", type=int, default=7, help="map view radius")
    play.add_argument("--no-clear", action="store_true", help="do not clear the terminal each frame")
    play.add_argument("--heaven", action="store_true", help="ask for one daily advice line in watch mode")
    play.add_argument("--strategy", default=None, help="set the initial standing 作戦 (天の声) before the run")
    play.set_defaults(func=run_play)

    simulate = sub.add_parser("simulate", help="run a fast non-interactive simulation")
    _add_common(simulate)
    simulate.add_argument("--strategy", default=None, help="set the initial standing 作戦 (天の声) before the run")
    simulate.set_defaults(func=run_simulate)

    pixel = sub.add_parser("pixel", help="play/watch in the isometric pixel-art window (needs pygame)")
    pixel.add_argument("--seed", type=int, default=42)
    pixel.add_argument("--days", type=int, default=112)
    pixel.add_argument("--llm", action="store_true", help="let an OpenAI-compatible cassette play")
    pixel.add_argument("--cassette", default="Qwen仙人", help="cassette name from config/models.toml")
    pixel.add_argument(
        "--tps",
        type=float,
        default=0.0,
        help="思考予算: force a constant tokens/sec (0=auto-measure)",
    )
    pixel.add_argument("--manual", action="store_true", help="control the hero manually")
    pixel.add_argument("--speed", type=int, default=3, choices=(1, 2, 3, 4, 5),
                       help="watch speed: 1 承認 / 2 遅 / 3 普 / 4 速 / 5 最速")
    pixel.add_argument("--strategy", default=None, help="set the initial standing 作戦 (天の声) before the run")
    pixel.add_argument(
        "--book",
        action="store_true",
        help="ぼうけんのしょ: 前世の教訓を次の生に届ける（cross-life lessons）",
    )
    pixel.add_argument("--scale", type=int, default=0, choices=(0, 1, 2, 3),
                       help="window preset: 0=auto (Full HD if desktop allows, else 90%% of desktop), "
                            "1=small 1280x720, 2=fhd 1920x1080, 3=large 2560x1440")
    pixel.add_argument("--start-day", type=int, default=0, help="debug: fast-forward (local brain) to this day before rendering")
    # hidden verification flags
    pixel.add_argument("--shots", type=int, default=0, help=argparse.SUPPRESS)
    pixel.add_argument("--shots-ui", action="store_true", help=argparse.SUPPRESS)
    pixel.add_argument("--shot-dir", default="/tmp/spl_px", help=argparse.SUPPRESS)
    pixel.set_defaults(func=run_pixel)

    evolve = sub.add_parser(
        "evolve",
        help="家訓の編纂: run N lives, revising the fixed 5-article canon after each",
    )
    evolve.add_argument("--lives", type=int, required=True, help="how many lives to run")
    evolve.add_argument("--seed", type=int, default=42)
    evolve.add_argument("--days", type=int, default=112)
    evolve.add_argument("--cassette", default="Qwen仙人vLLM", help="cassette name from config/models.toml")
    evolve.add_argument(
        "--llm",
        dest="llm",
        action="store_true",
        default=None,
        help="use the cassette's LLM 編纂者+brain (default: on when the cassette has a base_url)",
    )
    evolve.add_argument(
        "--no-llm",
        dest="llm",
        action="store_false",
        help="local brain + deterministic 編纂 (no server needed)",
    )
    evolve.add_argument(
        "--tps",
        type=float,
        default=0.0,
        help="思考予算: force a constant tokens/sec (0=auto-measure)",
    )
    evolve.add_argument("--book-dir", default=None, help="argparse passthrough; SPL_BOOK_DIR overrides the journal dir")
    evolve.set_defaults(func=run_evolve)

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
    parser.add_argument("--cassette", default="Qwen仙人", help="cassette name from config/models.toml")
    parser.add_argument(
        "--tps",
        type=float,
        default=0.0,
        help="思考予算: force a constant tokens/sec (0=auto-measure). "
        "e.g. --tps 10 (雲水) or --tps 1000 (仙界)",
    )
    parser.add_argument(
        "--book",
        action="store_true",
        help="ぼうけんのしょ: 前世の教訓を次の生に届ける（cross-life lessons）",
    )


def run_pixel(args: object) -> int:
    # Import lazily so the rest of the CLI works without pygame installed.
    from spl.ui.pixel import run_pixel as _run_pixel

    return _run_pixel(args)


def run_evolve(args: object) -> int:
    import os

    from spl.agent.llm_client import find_cassette
    from spl.arena.evolve import run_evolve as _run_evolve
    from spl.core.sim import PROJECT_ROOT
    from spl.ui.cli import _apply_tps_override

    # --book-dir is a convenience for SPL_BOOK_DIR (the journal directory).
    if getattr(args, "book_dir", None):
        os.environ["SPL_BOOK_DIR"] = args.book_dir

    name = getattr(args, "cassette", None)
    try:
        cassette = find_cassette(PROJECT_ROOT / "config" / "models.toml", name)
    except ValueError:
        # --no-llm runs need no real cassette; key the journal by the name only.
        from spl.agent.llm_client import Cassette

        if args.llm:
            raise
        cassette = Cassette(name=name or "Evolve仙人", base_url="")
    cassette = _apply_tps_override(cassette, args)
    # --llm defaults to True when the cassette has a base_url; --no-llm forces local.
    # The MAGI council marker has an empty base_url but IS an LLM brain.
    use_llm = args.llm
    if use_llm is None:
        use_llm = bool(getattr(cassette, "base_url", "")) or getattr(cassette, "name", None) == "MAGI"

    return _run_evolve(
        lives=args.lives,
        seed=args.seed,
        days=args.days,
        cassette=cassette,
        use_llm=use_llm,
        book_dir_cassette=getattr(args, "cassette", None),
    )


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

