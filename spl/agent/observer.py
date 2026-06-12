from __future__ import annotations

from spl.core.crops import FOOD_VALUES, monument_agronomy_line
from spl.core.world import SEASON_NAMES, TILE_NAMES, WEATHER_NAMES


class ObservationBuilder:
    # ぼうけんのしょ: lessons written by PAST SELVES who died on this island. Set
    # by the --book callers; empty (the default) means the book is off, and the
    # observation omits the bouken_no_sho block entirely.
    book_lessons: list[str] = []
    book_lives: int = 0

    def build(self, sim: object) -> dict[str, object]:
        hero = sim.hero
        world = sim.world
        nearby: list[str] = []
        for name, pos in world.neighbors(hero.pos):
            tile = TILE_NAMES.get(world.tile_at(pos), world.tile_at(pos))
            plot = world.plots.get(pos)
            if plot:
                crop = sim.crop_book.get(plot.crop)
                status = "ready" if plot.ready else f"{plot.days_left}d"
                tile = f"Field({crop.name}:{status})"
            nearby.append(f"{name}:{tile}")
        current = TILE_NAMES.get(world.tile_at(hero.pos), world.tile_at(hero.pos))
        if hero.pos in world.plots:
            plot = world.plots[hero.pos]
            crop = sim.crop_book.get(plot.crop)
            current = f"Field({crop.name}:{'ready' if plot.ready else str(plot.days_left) + 'd'})"
        alerts = self._alerts(sim)
        body = self._body_screams(sim)
        premonition = self._body_premonitions(sim)
        obs: dict[str, object] = {}
        # 体の声: the flesh interrupts everything — even the watcher's order.
        # A calm ledger line cannot compete with a long train of thought; a body
        # can. Present only when a stat turns critical.
        if body:
            obs["body"] = body
        # 体の予感: not pain yet — arithmetic. The whisper BEFORE the scream,
        # because procurement has lead time (誰だってわかること、を感覚器官に).
        if premonition:
            obs["premonition"] = premonition
        obs.update({
            # The watcher's standing order (作戦). Placed at the TOP on purpose:
            # the watcher SEES THE TRUE WORLD STATE, so this outranks the hermit's
            # own (possibly wrong) beliefs. Persists day after day until changed.
            "strategy_from_heaven": sim.advice_from_heaven,
        })
        # ぼうけんのしょ: when past lives left lessons, they sit right below the
        # watcher's order — 前世たちが死をもって書き残した教訓。weigh them like
        # scripture. Omitted entirely when the book is off / empty.
        lessons = list(getattr(self, "book_lessons", []) or [])
        if lessons:
            obs["bouken_no_sho"] = {
                "lives": int(getattr(self, "book_lives", 0) or 0),
                "lessons": lessons,
            }
        obs.update({
            "day": world.day,
            "season": SEASON_NAMES[world.season],
            "weather": WEATHER_NAMES[world.weather],
            "ap_left": hero.ap_left,
            "stats": {
                "hp": hero.hp,
                "hunger": hero.hunger,
                "water": hero.water,
                "stamina": hero.stamina,
                "sanity": hero.sanity,
            },
            "pos": current,
            "nearby": nearby,
            "landmarks": self._landmarks(sim),
            # The last few outcomes so the brain can learn within the day why an
            # action failed (e.g. "Planting needs a field.") — spec §4.1.
            "recent": list(getattr(sim, "day_log", [])[-3:]),
            "inventory": hero.inventory_summary(),
            "alerts": alerts,
            "trade_offer": sim.current_offer.describe() if sim.current_offer else None,
            "memory": sim.memory.recent_context(days=4),
            # 古い石碑: the settlers' agronomy as permanent background knowledge —
            # always present (the stone stands forever), placed AFTER inventory
            # and alerts so it reads as lore, not an alarm. ~120 chars, built
            # from the real crop data via the sim's loaded crop_book. 刻む: when a
            # previous hermit voluntarily cut verses into the stone, a compact
            # 先人の句 suffix is appended so the brain reads them as lore too.
            "monument": self._monument_text(sim),
        })
        return obs

    def _monument_text(self, sim: object) -> str:
        """The stone's permanent monument line: the settlers' agronomy, plus a
        compact 先人の句 suffix carrying any verses a PAST hermit voluntarily
        carved here (most recent last). The agronomy never drifts; the carvings
        are voluntary trans-generational messages."""
        base = monument_agronomy_line(sim.crop_book)
        carvings = [str(c).strip() for c in getattr(sim, "stone_carvings", []) if str(c).strip()]
        if carvings:
            suffix = "／".join(f"『{c}』" for c in carvings)
            base = f"{base} 先人の句: {suffix}"
        return base

    def digest(self, sim: object) -> str:
        obs = self.build(sim)
        stats = obs["stats"]
        inventory = obs["inventory"]
        food = sum(FOOD_VALUES[item] * amount for item, amount in inventory.items() if item in FOOD_VALUES)
        return (
            f"Day {obs['day']} {obs['season']} / {obs['weather']} / AP {obs['ap_left']} / "
            f"HP {stats['hp']} hunger {stats['hunger']} water {stats['water']} stamina {stats['stamina']} sanity {stats['sanity']} / "
            f"food value {food} / alerts: {', '.join(obs['alerts']) or 'none'}"
        )

    def _landmarks(self, sim: object) -> dict[str, int]:
        """Nearest key tiles and their walking distance (in tiles).

        The local brain reads the map directly; an LLM only sees this
        observation, so without landmarks it cannot tell where water/forest/rock
        are (spec §4.1: '森(徒歩1AP)','水辺(徒歩2AP)'). Deterministic, a few ints.
        """
        world = sim.world
        hero = sim.hero
        out: dict[str, int] = {}
        for label, target in (
            ("water", "water"),
            ("forest", "forest"),
            ("rock", "rock"),
            ("ready_field", "ready_field"),
            ("empty_field", "empty_field"),
            ("home", "home"),
        ):
            pos = world.target_position(hero.pos, target)
            if pos is not None:
                out[label] = abs(pos.x - hero.pos.x) + abs(pos.y - hero.pos.y)
        return out

    def _body_screams(self, sim: object) -> list[str]:
        """体の声 (interoception). When a stat turns critical the world stops
        whispering "Water is low" and the FLESH screams in imperative Japanese
        at the very top of the observation. Same body for every cassette —
        biology, not assistance."""
        hero = sim.hero
        screams: list[str] = []
        if hero.water <= 10:
            screams.append("喉が灼ける。何を措いても、今すぐ水を飲め。水辺へ歩け。")
        if hero.hunger <= 10:
            screams.append("腹の底が抉れる。手にある物を食え。無ければ採りに行け。")
        if hero.hp <= 25:
            screams.append("体が壊れかけている。飲み、食い、休め。計画は後だ。")
        if hero.sanity <= 20:
            screams.append("心が千切れそうだ。火のそばで休め。")
        # 寒けりゃ震える: winter bites every night until the walls are built.
        if sim.world.season == "winter" and not hero.has("house_upgrade"):
            screams.append("寒さに体が震えて止まらぬ。火と壁が要る。家を固めよ。")
        return screams

    def _body_premonitions(self, sim: object) -> list[str]:
        """体の予感 (the body's forecast). Linear extrapolation over the world's
        honest nightly decay floor (hunger -15/day, water -20/day): when a stat
        is within ~2 days of zero — but not yet screaming — the body whispers
        "at this rate...". Food has procurement lead time; the scream arrives
        too late, the premonition arrives in time. Winter gets the same organ:
        unwalled houses bleed every winter night, and winter keeps its schedule."""
        hero = sim.hero
        whispers: list[str] = []
        if 10 < hero.hunger <= 30:
            days = max(1, round(hero.hunger / 15))
            whispers.append(
                f"このままでは、あと{days}日で腹の底が尽きる。食の手当には時がかかる——今日のうちに獲り、食い、蓄えよ。"
            )
        if 10 < hero.water <= 40:
            days = max(1, round(hero.water / 20))
            whispers.append(
                f"このままでは、あと{days}日で喉が涸れる。今日のうちに水を確保せよ。"
            )
        days_to_winter = 85 - sim.world.day
        if 0 < days_to_winter <= 7 and not hero.has("house_upgrade"):
            whispers.append(
                f"冬まであと{days_to_winter}日。壁なき家は冬の夜ごとに体を削る。家を固め、蓄えを積むなら今だ。"
            )
        return whispers

    def _alerts(self, sim: object) -> list[str]:
        hero = sim.hero
        world = sim.world
        alerts: list[str] = []
        if hero.hunger < 35:
            alerts.append("Hunger is low")
        if hero.water < 35:
            alerts.append("Water is low")
        if hero.stamina < 22:
            alerts.append("Stamina is low")
        if hero.sanity < 30:
            alerts.append("Sanity is fraying")
        if world.weather in {"storm", "drought", "snow"}:
            alerts.append(f"Weather hazard: {world.weather}")
        ready = []
        dry = []
        needed = 2 if world.weather == "drought" else 1
        for plot in world.plots.values():
            crop = sim.crop_book.get(plot.crop)
            if plot.ready:
                ready.append(crop.name)
            elif crop.needs_water and world.weather != "rain" and plot.water_level < needed:
                dry.append(crop.name)
        if ready:
            alerts.append("Harvest ready: " + ", ".join(sorted(set(ready))))
        if dry:
            alerts.append("Crops need water twice today" if needed == 2 else "Crops need water")
        if world.season == "autumn" and not hero.has("storage_barrel"):
            alerts.append("Autumn: build storage for winter")
        if world.season == "winter":
            alerts.append("Winter: crops do not grow")
        if sim.current_offer:
            alerts.append("Merchant is here")
        return alerts[:8]

