from __future__ import annotations

from spl.core.crops import FOOD_VALUES
from spl.core.world import SEASON_NAMES, TILE_NAMES, WEATHER_NAMES


class ObservationBuilder:
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
        return {
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
            "inventory": hero.inventory_summary(),
            "alerts": alerts,
            "trade_offer": sim.current_offer.describe() if sim.current_offer else None,
            "advice_from_heaven": sim.advice_from_heaven,
            "memory": sim.memory.recent_context(days=4),
        }

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
        for plot in world.plots.values():
            crop = sim.crop_book.get(plot.crop)
            if plot.ready:
                ready.append(crop.name)
            elif crop.needs_water and world.weather != "rain" and plot.water_level == 0:
                dry.append(crop.name)
        if ready:
            alerts.append("Harvest ready: " + ", ".join(sorted(set(ready))))
        if dry:
            alerts.append("Crops need water")
        if world.season == "autumn" and not hero.has("storage_barrel"):
            alerts.append("Autumn: build storage for winter")
        if world.season == "winter":
            alerts.append("Winter: crops do not grow")
        if sim.current_offer:
            alerts.append("Merchant is here")
        return alerts[:8]

