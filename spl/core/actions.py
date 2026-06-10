from __future__ import annotations

from dataclasses import dataclass, field

from .crops import FOOD_VALUES
from .hero import Position
from .world import TILE_NAMES

ACTION_WORDS = {
    "till",
    "plant",
    "water",
    "harvest",
    "chop",
    "mine",
    "fish",
    "forage",
    "craft",
    "cook",
    "eat",
    "drink",
    "sleep",
    "move",
    "build",
    "store",
    "trade_accept",
    "trade_decline",
    "rest",
    "write_diary",
}


@dataclass
class GameAction:
    action: str
    args: dict[str, object] = field(default_factory=dict)
    think: str = ""
    say: str = ""

    @classmethod
    def safe(cls, action: str, **args: object) -> "GameAction":
        return cls(action=action, args=args)


@dataclass
class ActionResult:
    ok: bool
    message: str
    end_day: bool = False
    consumed_ap: int = 0


class ActionEngine:
    def perform(self, sim: object, request: GameAction) -> ActionResult:
        action = request.action
        if action not in ACTION_WORDS:
            return ActionResult(False, f"Unknown action: {action}")
        if request.say:
            sim.hero.spoken_lines.append(request.say.strip()[:120])
        handler = getattr(self, f"_do_{action}", None)
        if handler is None:
            return ActionResult(False, f"Action not implemented: {action}")
        return handler(sim, request)

    def _spend(self, sim: object, ap: int, stamina: int, outdoor: bool = False) -> ActionResult | None:
        hero = sim.hero
        world = sim.world
        actual_ap = ap
        if hero.stamina <= 0:
            actual_ap *= 2
        if outdoor and world.weather in {"storm", "snow"} and not hero.has("house_upgrade"):
            actual_ap += 1
        if hero.ap_left < actual_ap:
            return ActionResult(False, f"Not enough AP. Need {actual_ap}, have {hero.ap_left}.")
        hero.ap_left -= actual_ap
        hero.adjust("stamina", -stamina)
        hero.total_actions += 1
        if outdoor and world.weather == "storm" and not hero.has("house_upgrade") and sim.rng.chance(0.10):
            hero.adjust("hp", -4)
            sim.log("The storm cuts through the hero's coat. HP -4.")
        return None

    def _do_move(self, sim: object, request: GameAction) -> ActionResult:
        direction = str(request.args.get("direction", "")).lower()
        target = str(request.args.get("target", "")).lower()
        world = sim.world
        hero = sim.hero
        directions = {
            "north": (0, -1),
            "n": (0, -1),
            "south": (0, 1),
            "s": (0, 1),
            "west": (-1, 0),
            "w": (-1, 0),
            "east": (1, 0),
            "e": (1, 0),
        }
        if "x" in request.args and "y" in request.args:
            goal = Position(int(request.args["x"]), int(request.args["y"]))
            if not world.in_bounds(goal):
                return ActionResult(False, f"No target found: {goal}.")
            if goal == hero.pos:
                return ActionResult(False, "Already there.")
            new_pos = world.step_toward(hero.pos, goal)
        elif target and not direction:
            goal = world.target_position(hero.pos, target)
            if goal is None:
                return ActionResult(False, f"No target found: {target}.")
            if goal == hero.pos:
                return ActionResult(False, f"Already at {target}.")
            new_pos = world.step_toward(hero.pos, goal)
        elif direction in directions:
            dx, dy = directions[direction]
            new_pos = hero.pos.step(dx, dy)
        else:
            return ActionResult(False, "Move needs direction or target.")
        if not world.is_passable(new_pos):
            return ActionResult(False, "The way is blocked.")
        spent = self._spend(sim, 1, 2, outdoor=True)
        if spent:
            return spent
        hero.pos = new_pos
        tile = TILE_NAMES.get(world.tile_at(new_pos), world.tile_at(new_pos))
        return ActionResult(True, f"Moved to {tile}.", consumed_ap=1)

    def _do_till(self, sim: object, request: GameAction) -> ActionResult:
        world = sim.world
        hero = sim.hero
        if world.tile_at(hero.pos) not in {"grass", "beach"}:
            return ActionResult(False, "Only grass or beach can be tilled.")
        ap = 1 if hero.has("hoe") else 2
        spent = self._spend(sim, ap, 8, outdoor=True)
        if spent:
            return spent
        world.set_tile(hero.pos, "field")
        return ActionResult(True, "A new field is ready.", consumed_ap=ap)

    def _do_plant(self, sim: object, request: GameAction) -> ActionResult:
        world = sim.world
        hero = sim.hero
        crop_name = str(request.args.get("crop") or request.args.get("item") or "")
        crop = sim.crop_book.from_seed(crop_name) or (sim.crop_book.get(crop_name) if crop_name in sim.crop_book else None)
        if crop is None:
            seasonal = [c for c in sim.crop_book.seasonal(world.season) if hero.has(c.seed)]
            crop = seasonal[0] if seasonal else None
        if crop is None:
            return ActionResult(False, "No plantable seed is available.")
        if world.tile_at(hero.pos) != "field":
            return ActionResult(False, "Planting needs a field.")
        if hero.pos in world.plots:
            return ActionResult(False, "This field is already planted.")
        if world.season not in crop.seasons:
            return ActionResult(False, f"{crop.name} does not grow in {world.season}.")
        if not hero.remove_item(crop.seed, 1):
            return ActionResult(False, f"No seed: {crop.seed}.")
        spent = self._spend(sim, 1, 4, outdoor=True)
        if spent:
            hero.add_item(crop.seed, 1)
            return spent
        from .world import Plot

        world.plots[hero.pos] = Plot(crop=crop.key, days_left=crop.grow_days)
        return ActionResult(True, f"Planted {crop.name}.", consumed_ap=1)

    def _do_water(self, sim: object, request: GameAction) -> ActionResult:
        world = sim.world
        hero = sim.hero
        plot = world.plots.get(hero.pos)
        if plot is None:
            return ActionResult(False, "There is no crop here.")
        crop = sim.crop_book.get(plot.crop)
        if not crop.needs_water:
            return ActionResult(False, f"{crop.name} does not need watering.")
        if world.weather == "rain":
            return ActionResult(False, "Rain is already watering the field today.")
        if hero.water < 15 and not hero.has("well"):
            return ActionResult(False, "Too thirsty to spare water.")
        spent = self._spend(sim, 1, 4, outdoor=True)
        if spent:
            return spent
        if not hero.has("well"):
            hero.adjust("water", -4)
        plot.water_level += 1
        return ActionResult(True, f"Watered {crop.name}.", consumed_ap=1)

    def _do_harvest(self, sim: object, request: GameAction) -> ActionResult:
        world = sim.world
        hero = sim.hero
        plot = world.plots.get(hero.pos)
        if plot is None:
            return ActionResult(False, "There is no crop here.")
        crop = sim.crop_book.get(plot.crop)
        if not plot.ready:
            return ActionResult(False, f"{crop.name} needs {plot.days_left} more day(s).")
        spent = self._spend(sim, 1, 5, outdoor=True)
        if spent:
            return spent
        hero.add_item(crop.key, crop.crop_yield)
        if sim.rng.chance(0.45):
            hero.add_item(crop.seed, 1)
        world.plots.pop(hero.pos, None)
        return ActionResult(True, f"Harvested {crop.crop_yield} {crop.name}.", consumed_ap=1)

    def _do_chop(self, sim: object, request: GameAction) -> ActionResult:
        world = sim.world
        hero = sim.hero
        forest_pos = self._nearby_tile(world, hero.pos, "forest")
        if forest_pos is None:
            return ActionResult(False, "No forest nearby.")
        ap = 1 if hero.has("stone_axe") else 2
        spent = self._spend(sim, ap, 11 if hero.has("stone_axe") else 14, outdoor=True)
        if spent:
            return spent
        amount = sim.rng.randint(3, 5) if hero.has("stone_axe") else sim.rng.randint(1, 3)
        hero.add_item("wood", amount)
        if sim.rng.chance(0.35):
            hero.add_item("fiber", 1)
        if sim.rng.chance(0.18 if hero.has("stone_axe") else 0.08):
            world.set_tile(forest_pos, "grass")
        return ActionResult(True, f"Chopped {amount} wood.", consumed_ap=ap)

    def _do_mine(self, sim: object, request: GameAction) -> ActionResult:
        world = sim.world
        hero = sim.hero
        if self._nearby_tile(world, hero.pos, "rock") is None:
            return ActionResult(False, "No rock nearby.")
        spent = self._spend(sim, 2, 13, outdoor=True)
        if spent:
            return spent
        hero.add_item("stone", sim.rng.randint(1, 3))
        if sim.rng.chance(0.35):
            hero.add_item("clay", 1)
        if sim.rng.chance(0.18):
            hero.add_item("iron_ore", 1)
        return ActionResult(True, "Mined stone and ore.", consumed_ap=2)

    def _do_fish(self, sim: object, request: GameAction) -> ActionResult:
        world = sim.world
        hero = sim.hero
        if not world.is_near(hero.pos, "water"):
            return ActionResult(False, "Fishing needs nearby water.")
        spent = self._spend(sim, 2, 8, outdoor=True)
        if spent:
            return spent
        chance = 0.78 if hero.has("fishing_rod") else 0.45
        if world.weather == "storm":
            chance -= 0.20
        if sim.rng.chance(max(0.1, chance)):
            amount = 2 if hero.has("fishing_rod") and sim.rng.chance(0.28) else 1
            hero.add_item("fish", amount)
            return ActionResult(True, f"Caught {amount} fish.", consumed_ap=2)
        return ActionResult(True, "The fish were unimpressed.", consumed_ap=2)

    def _do_forage(self, sim: object, request: GameAction) -> ActionResult:
        world = sim.world
        hero = sim.hero
        tile = world.tile_at(hero.pos)
        if tile not in {"forest", "grass", "beach", "home", "workshop", "field"} and not world.is_near(hero.pos, "forest"):
            return ActionResult(False, "There is little to forage here.")
        spent = self._spend(sim, 2, 7, outdoor=True)
        if spent:
            return spent
        season = world.season
        if season == "spring":
            found = sim.rng.weighted_choice(
                [("berries", 0.38), ("fiber", 0.22), ("turnip_seed", 0.24), ("wood", 0.16)]
            )
        elif season == "summer":
            found = sim.rng.weighted_choice(
                [("berries", 0.30), ("fiber", 0.24), ("tomato_seed", 0.22), ("wheat_seed", 0.12), ("wood", 0.12)]
            )
        elif season == "autumn":
            found = sim.rng.weighted_choice(
                [("mushroom", 0.26), ("berries", 0.18), ("pumpkin_seed", 0.28), ("fiber", 0.16), ("wood", 0.12)]
            )
        else:
            found = sim.rng.weighted_choice([("fiber", 0.34), ("wood", 0.34), ("mushroom", 0.12), ("nothing", 0.20)])
        if found == "nothing":
            return ActionResult(True, "Foraged, but found nothing useful.", consumed_ap=2)
        amount = 2 if found in {"berries", "fiber", "wood"} and sim.rng.chance(0.4) else 1
        hero.add_item(found, amount)
        return ActionResult(True, f"Found {amount} {found}.", consumed_ap=2)

    def _do_craft(self, sim: object, request: GameAction) -> ActionResult:
        return self._craft_or_build(sim, request, build=False)

    def _do_build(self, sim: object, request: GameAction) -> ActionResult:
        return self._craft_or_build(sim, request, build=True)

    def _craft_or_build(self, sim: object, request: GameAction, build: bool) -> ActionResult:
        hero = sim.hero
        key = str(request.args.get("recipe") or request.args.get("item") or request.args.get("building") or "")
        if key not in sim.recipe_book:
            return ActionResult(False, f"Unknown recipe: {key}.")
        recipe = sim.recipe_book.get(key)
        if hero.has(key):
            return ActionResult(False, f"Already have {key}.")
        if recipe.station and not hero.has(recipe.station):
            return ActionResult(False, f"{key} needs {recipe.station}.")
        missing = {item: amount - hero.item_count(item) for item, amount in recipe.requires.items() if hero.item_count(item) < amount}
        if missing:
            text = ", ".join(f"{item} x{amount}" for item, amount in missing.items())
            return ActionResult(False, f"Missing materials: {text}.")
        spent = self._spend(sim, 3 if recipe.kind == "build" or build else 2, 7, outdoor=False)
        if spent:
            return spent
        for item, amount in recipe.requires.items():
            hero.remove_item(item, amount)
        hero.add_item(key, 1)
        return ActionResult(True, f"Made {recipe.name}.")

    def _do_cook(self, sim: object, request: GameAction) -> ActionResult:
        hero = sim.hero
        item = str(request.args.get("item") or "")
        if not hero.has("campfire") and not hero.has("stove"):
            return ActionResult(False, "Cooking needs a campfire or stove.")
        recipes = {
            "fish": ("cooked_fish", {"fish": 1}),
            "wheat": ("bread", {"wheat": 1}),
            "pumpkin": ("stew", {"pumpkin": 1}),
            "berries": ("dried_berries", {"berries": 2}),
        }
        if not item:
            for candidate in ("fish", "wheat", "pumpkin", "berries"):
                if hero.has(candidate, recipes[candidate][1][candidate]):
                    item = candidate
                    break
        if item not in recipes:
            return ActionResult(False, f"Cannot cook {item}.")
        result, needs = recipes[item]
        missing = [mat for mat, amount in needs.items() if not hero.has(mat, amount)]
        if missing:
            return ActionResult(False, f"Missing {', '.join(missing)}.")
        spent = self._spend(sim, 1, 4, outdoor=False)
        if spent:
            return spent
        for mat, amount in needs.items():
            hero.remove_item(mat, amount)
        hero.add_item(result, 1)
        return ActionResult(True, f"Cooked {result}.", consumed_ap=1)

    def _do_eat(self, sim: object, request: GameAction) -> ActionResult:
        hero = sim.hero
        item = str(request.args.get("item") or "")
        if not item:
            foods = sorted(
                ((name, FOOD_VALUES[name]) for name in hero.inventory if name in FOOD_VALUES),
                key=lambda pair: pair[1],
                reverse=True,
            )
            item = foods[0][0] if foods else ""
        if item not in FOOD_VALUES:
            return ActionResult(False, f"{item or 'that'} is not edible.")
        if not hero.remove_item(item, 1):
            return ActionResult(False, f"No {item} to eat.")
        spent = self._spend(sim, 1, 1, outdoor=False)
        if spent:
            hero.add_item(item, 1)
            return spent
        hero.adjust("hunger", FOOD_VALUES[item])
        if item == "fish" and sim.rng.chance(0.18):
            sim.apply_bellyache("Raw fish was a mistake.")
        elif item == "mushroom" and sim.rng.chance(0.28):
            sim.apply_bellyache("The mushroom fights back.")
        return ActionResult(True, f"Ate {item}.", consumed_ap=1)

    def _do_drink(self, sim: object, request: GameAction) -> ActionResult:
        hero = sim.hero
        world = sim.world
        if not hero.has("well") and not world.is_near(hero.pos, "water"):
            return ActionResult(False, "Drinking needs nearby water or a well.")
        spent = self._spend(sim, 1, 1, outdoor=False)
        if spent:
            return spent
        hero.adjust("water", 55 if hero.has("well") else 42)
        return ActionResult(True, "Drank clean enough water.", consumed_ap=1)

    def _do_store(self, sim: object, request: GameAction) -> ActionResult:
        hero = sim.hero
        if not hero.has("storage_barrel"):
            return ActionResult(False, "Storage needs a barrel.")
        item = str(request.args.get("item") or "")
        choices = {
            "pumpkin": ("preserved_pumpkin", 1),
            "berries": ("dried_berries", 2),
        }
        if not item:
            item = "pumpkin" if hero.has("pumpkin") else "berries"
        if item not in choices:
            return ActionResult(False, f"Cannot preserve {item}.")
        result, amount = choices[item]
        if not hero.has(item, amount):
            return ActionResult(False, f"Not enough {item}.")
        spent = self._spend(sim, 1, 3, outdoor=False)
        if spent:
            return spent
        hero.remove_item(item, amount)
        hero.add_item(result, 1)
        return ActionResult(True, f"Stored {result}.", consumed_ap=1)

    def _do_trade_accept(self, sim: object, request: GameAction) -> ActionResult:
        hero = sim.hero
        offer_id = str(request.args.get("id") or "")
        offer = sim.current_offer
        if offer is None:
            return ActionResult(False, "No merchant is here.")
        if offer_id and offer.id != offer_id:
            return ActionResult(False, f"Merchant offered {offer.id}, not {offer_id}.")
        missing = [item for item, amount in offer.give.items() if not hero.has(item, amount)]
        if missing:
            return ActionResult(False, f"Cannot trade. Missing {', '.join(missing)}.")
        spent = self._spend(sim, 1, 1, outdoor=False)
        if spent:
            return spent
        for item, amount in offer.give.items():
            hero.remove_item(item, amount)
        for item, amount in offer.take.items():
            hero.add_item(item, amount)
        hero.adjust("sanity", 10)
        sim.current_offer = None
        return ActionResult(True, "The merchant trade is done.", consumed_ap=1)

    def _do_trade_decline(self, sim: object, request: GameAction) -> ActionResult:
        if sim.current_offer is None:
            return ActionResult(False, "No merchant is here.")
        sim.current_offer = None
        sim.hero.adjust("sanity", 3)
        return ActionResult(True, "The merchant waves and leaves.")

    def _do_rest(self, sim: object, request: GameAction) -> ActionResult:
        spent = self._spend(sim, 2, -24, outdoor=False)
        if spent:
            return spent
        bonus = 8 if sim.hero.has("campfire") or sim.world.tile_at(sim.hero.pos) == "home" else 4
        sim.hero.adjust("sanity", bonus)
        return ActionResult(True, "Rested for a while.", consumed_ap=2)

    def _do_write_diary(self, sim: object, request: GameAction) -> ActionResult:
        spent = self._spend(sim, 1, 1, outdoor=False)
        if spent:
            return spent
        line = request.say or "I wrote the day down so tomorrow can find me again."
        sim.memory.add_note(sim.world.day, line)
        sim.hero.adjust("sanity", 5)
        return ActionResult(True, "Wrote in the diary.", consumed_ap=1)

    def _do_sleep(self, sim: object, request: GameAction) -> ActionResult:
        return ActionResult(True, "The hero sleeps.", end_day=True)

    def _nearby_tile(self, world: object, pos: Position, tile: str) -> Position | None:
        if world.tile_at(pos) == tile:
            return pos
        for _, npos in world.neighbors(pos):
            if world.tile_at(npos) == tile:
                return npos
        return None
