from __future__ import annotations

from spl.core.actions import GameAction
from spl.core.crops import FOOD_VALUES


class LocalPolicyAgent:
    """A deterministic, no-LLM survival brain used as the bundled cartridge."""

    def choose(self, sim: object) -> GameAction:
        hero = sim.hero
        world = sim.world
        self._sim = sim

        if sim.current_offer and self._can_pay(hero, sim.current_offer.give):
            return self._act("trade_accept", "A trade can turn scraps into winter plans.", id=sim.current_offer.id)

        if hero.water < 34:
            if hero.has("well") or world.is_near(hero.pos, "water"):
                return self._act("drink", "Water first. Heroism is mostly plumbing.")
            if hero.ap_left < self._outdoor_ap(sim, 1):
                return self._act("sleep", "No daylight remains to reach water safely.")
            return self._move("water", "Find water before any clever plan.")

        if hero.hunger < 38:
            ready = world.target_position(hero.pos, "ready_field")
            if ready is not None:
                if ready == hero.pos:
                    if hero.ap_left < self._outdoor_ap(sim, 1):
                        return self._act("sleep", "The crop can wait; the body cannot.")
                    return self._act("harvest", "Food is standing in the field. Take it.")
                if hero.ap_left >= self._outdoor_ap(sim, 1):
                    return self._move("ready_field", "Go harvest before fishing.")
            cook = self._cook_if_useful(hero)
            if cook:
                return cook
            food = self._best_food(hero)
            if food:
                return self._act("eat", f"Eat {food}; strategy needs calories.", item=food)
            if hero.ap_left <= 2:
                return self._act("sleep", "Too little daylight for a risky food run.")
            if hero.has("fishing_rod") or world.is_near(hero.pos, "water"):
                if world.is_near(hero.pos, "water"):
                    return self._outdoor_or_sleep(sim, "fish", "Fish for dinner before the stomach starts negotiating.", 2)
                return self._move_or_sleep(sim, "water", "Move to the shore and fish.")
            if world.is_near(hero.pos, "forest") or world.tile_at(hero.pos) in {"forest", "grass", "field", "home"}:
                return self._outdoor_or_sleep(sim, "forage", "Search for emergency food.", 2)
            return self._move_or_sleep(sim, "forest", "Move toward forage.")

        if hero.ap_left <= 1:
            return self._act("sleep", "The day is spent.")

        if hero.ap_left < self._outdoor_ap(sim, 1):
            food = self._best_food(hero)
            if hero.hunger < 76 and food:
                return self._act("eat", f"Use the last light for a ration: {food}.", item=food)
            if hero.sanity < 55:
                return self._act("write_diary", "No safe outdoor work remains; write instead.", say="The storm owns the path tonight.")
            return self._act("sleep", "No safe outdoor work remains today.")

        if hero.stamina < 18:
            if hero.ap_left >= 3:
                return self._act("rest", "Rest now so tomorrow has a body to use.")
            return self._act("sleep", "Sleep before exhaustion makes every step expensive.")

        if hero.sanity < 24 and hero.ap_left >= 3:
            return self._act("write_diary", "Write before the island gets too loud.", say="I put one line between me and the dark.")

        ready = world.target_position(hero.pos, "ready_field")
        if ready is not None:
            if ready == hero.pos:
                if hero.ap_left < self._outdoor_ap(sim, 1):
                    return self._act("sleep", "Harvest can wait until the storm clock resets.")
                return self._act("harvest", "Ready crops are future meals.")
            if hero.ap_left < self._outdoor_ap(sim, 1):
                return self._act("sleep", "No daylight remains to reach the ripe field.")
            return self._move("ready_field", "Go collect ripe food.")

        if world.weather not in {"rain"}:
            dry = self._dry_crop_position(sim)
            if dry is not None:
                if dry == hero.pos:
                    plot = world.plots[hero.pos]
                    crop = sim.crop_book.get(plot.crop)
                    needed = 2 if world.weather == "drought" else 1
                    if plot.water_level < needed:
                        if hero.ap_left < self._outdoor_ap(sim, 1):
                            return self._act("sleep", "Too little daylight to water safely.")
                        return self._act("water", f"{crop.name} needs water today.")
                if hero.ap_left < self._outdoor_ap(sim, 1):
                    return self._act("sleep", "No daylight remains to reach the dry field.")
                return self._move_to_pos(dry, "Move to the thirstiest field.")

        preserve = self._preserve_if_useful(hero, world.season)
        if preserve:
            return preserve

        cook = self._cook_if_useful(hero)
        if cook and (hero.hunger < 70 or world.season == "winter" or hero.has("storage_barrel")):
            return cook

        craft = self._next_craft(sim)
        if craft:
            return craft

        planting = self._planting_action(sim)
        if planting:
            return planting

        # Winter is survival, not prospecting. Building from existing stock is
        # still allowed above, but do not trek out to mine/chop for new
        # infrastructure while water or food is slipping — that is how a hero
        # freezes with a half-built house.
        winter_pressure = world.season == "winter" and (hero.water < 58 or hero.hunger < 62)
        if not winter_pressure:
            material = self._needed_material(sim)
            if material:
                return self._gather_material(sim, material)

        if world.season == "winter":
            if hero.water < 58 and (hero.has("well") or world.is_near(hero.pos, "water")):
                return self._act("drink", "Sip before the cold makes every trip costlier.")
            food = self._best_food(hero)
            if hero.hunger < 72 and food:
                return self._act("eat", f"Winter ration: {food}.", item=food)
            if hero.has("fishing_rod"):
                if world.is_near(hero.pos, "water"):
                    return self._outdoor_or_sleep(sim, "fish", "Fish through winter; the field is asleep.", 2)
                return self._move_or_sleep(sim, "water", "Winter food waits at the shore.")
            return self._outdoor_or_sleep(sim, "forage", "Winter forage is thin, but thin is not zero.", 2)

        if self._food_value(hero) < self._desired_food_value(world.day, world.season):
            if hero.has("fishing_rod") and world.is_near(hero.pos, "water"):
                return self._outdoor_or_sleep(sim, "fish", "Build the food buffer.", 2)
            if hero.has("fishing_rod"):
                return self._move_or_sleep(sim, "water", "Go fishing for reserves.")
            if world.tile_at(hero.pos) == "forest" or world.is_near(hero.pos, "forest"):
                return self._outdoor_or_sleep(sim, "forage", "Gather wild food and seeds.", 2)
            return self._move_or_sleep(sim, "forest", "Move toward forest forage.")

        if hero.ap_left >= 3 and hero.sanity < 75:
            return self._act("write_diary", "A calm mind is a winter tool.", say="Today did not defeat me.")
        return self._act("sleep", "Bank the remaining strength for tomorrow.")

    def _planting_action(self, sim: object) -> GameAction | None:
        hero = sim.hero
        world = sim.world
        if world.season == "winter":
            return None
        crop = self._best_crop_to_plant(sim)
        if crop is None:
            return None
        fields = [pos for pos in world.positions() if world.tile_at(pos) == "field"]
        planted = len(world.plots)
        desired = {"spring": 5, "summer": 5, "autumn": 7}.get(world.season, 3)
        empty = world.target_position(hero.pos, "empty_field")
        if empty is not None:
            if empty == hero.pos:
                if hero.ap_left < self._outdoor_ap(sim, 1):
                    return self._act("sleep", "Planting can wait until morning.")
                return self._act("plant", f"Plant {crop.key} while the season allows it.", crop=crop.key)
            if hero.ap_left < self._outdoor_ap(sim, 1):
                return self._act("sleep", "No daylight remains to reach the field.")
            return self._move("empty_field", "Use the prepared field.")
        if len(fields) < desired:
            if world.tile_at(hero.pos) in {"grass", "beach"}:
                base_ap = 1 if hero.has("hoe") else 2
                if hero.ap_left < self._outdoor_ap(sim, base_ap):
                    return self._act("sleep", "No daylight remains to open a field.")
                return self._act("till", "More soil means more winter chances.")
            if hero.ap_left < self._outdoor_ap(sim, 1):
                return self._act("sleep", "No daylight remains to reach new soil.")
            return self._move("grass", "Find grass to till.")
        return None

    def _best_crop_to_plant(self, sim: object) -> object | None:
        hero = sim.hero
        world = sim.world
        choices = [crop for crop in sim.crop_book.seasonal(world.season) if hero.has(crop.seed)]
        if not choices:
            return None
        if world.season == "spring":
            order = ["turnip", "wheat"]
        elif world.season == "summer":
            order = ["tomato", "wheat"]
        else:
            order = ["pumpkin", "turnip"]
        for key in order:
            for crop in choices:
                if crop.key == key:
                    return crop
        return choices[0]

    def _dry_crop_position(self, sim: object) -> object | None:
        world = sim.world
        needed = 2 if world.weather == "drought" else 1

        def needs_water(pos: object) -> bool:
            plot = world.plots.get(pos)
            if not plot or plot.ready:
                return False
            crop = sim.crop_book.get(plot.crop)
            return crop.needs_water and plot.water_level < needed and world.season in crop.seasons

        return world.find_nearest(sim.hero.pos, needs_water)

    def _craft_priorities(self, season: str) -> list[str]:
        # Get tools and a fire first so the hero can function. From summer on,
        # pull the winter-survival trio forward — a well (water), a barrel
        # (preserved food) and the house upgrade (no -4 HP/day in the snow) —
        # ahead of the comfort builds (stove, fence), so shelter is finished
        # before the cold arrives, not abandoned half-built in a blizzard.
        if season in {"summer", "autumn"}:
            return [
                "stone_axe",
                "hoe",
                "fishing_rod",
                "campfire",
                "well",
                "storage_barrel",
                "house_upgrade",
                "stove",
                "fence",
            ]
        return [
            "stone_axe",
            "hoe",
            "fishing_rod",
            "campfire",
            "storage_barrel",
            "well",
            "stove",
            "fence",
            "house_upgrade",
        ]

    def _next_craft(self, sim: object) -> GameAction | None:
        hero = sim.hero
        for key in self._craft_priorities(sim.world.season):
            if hero.has(key):
                continue
            recipe = sim.recipe_book.get(key)
            if recipe.station and not hero.has(recipe.station):
                continue
            if self._can_pay(hero, recipe.requires):
                needed_ap = 3 if recipe.kind == "build" else 2
                if hero.ap_left < needed_ap:
                    return None
                action = "build" if recipe.kind == "build" else "craft"
                return self._act(action, f"Make {key}; infrastructure is survival.", recipe=key)
        return None

    def _needed_material(self, sim: object) -> str | None:
        hero = sim.hero
        priorities = self._craft_priorities(sim.world.season)
        for key in priorities:
            if hero.has(key):
                continue
            recipe = sim.recipe_book.get(key)
            if recipe.station and not hero.has(recipe.station):
                continue
            for item, amount in recipe.requires.items():
                if hero.item_count(item) < amount:
                    return item
        if sim.world.season == "autumn" and hero.item_count("pumpkin_seed") < 2:
            return "seeds"
        if self._food_value(hero) < self._desired_food_value(sim.world.day, sim.world.season):
            return "food"
        return None

    def _gather_material(self, sim: object, material: str) -> GameAction:
        world = sim.world
        hero = sim.hero
        if material == "wood":
            if world.tile_at(hero.pos) == "forest" or world.is_near(hero.pos, "forest"):
                base = 1 if hero.has("stone_axe") else 2
                return self._outdoor_or_sleep(sim, "chop", "Wood is tomorrow's tool.", base)
            return self._move_or_sleep(sim, "forest", "Move to trees for wood.")
        if material in {"stone", "clay", "iron_ore"}:
            if world.tile_at(hero.pos) == "rock" or world.is_near(hero.pos, "rock"):
                return self._outdoor_or_sleep(sim, "mine", "Stone and clay unlock the camp.", 2)
            return self._move_or_sleep(sim, "rock", "Move to rocks for materials.")
        if material in {"fiber", "seeds", "food"}:
            if world.tile_at(hero.pos) == "forest" or world.is_near(hero.pos, "forest") or world.tile_at(hero.pos) in {"grass", "field"}:
                return self._outdoor_or_sleep(sim, "forage", "Forage for fiber, seeds, or small meals.", 2)
            return self._move_or_sleep(sim, "forest", "Move to forage.")
        return self._outdoor_or_sleep(sim, "forage", "Search for whatever the island offers.", 2)

    def _outdoor_or_sleep(self, sim: object, action: str, think: str, base: int) -> GameAction:
        if sim.hero.ap_left < self._outdoor_ap(sim, base):
            return self._act("sleep", "No daylight remains for outdoor work today.")
        return self._act(action, think)

    def _move_or_sleep(self, sim: object, target: str, think: str) -> GameAction:
        if sim.hero.ap_left < self._outdoor_ap(sim, 1):
            return self._act("sleep", "No daylight remains to travel safely.")
        return self._move(target, think)

    def _best_food(self, hero: object) -> str | None:
        foods = [(item, FOOD_VALUES[item]) for item, amount in hero.inventory.items() if amount > 0 and item in FOOD_VALUES]
        if not foods:
            return None
        foods.sort(key=lambda pair: pair[1], reverse=True)
        return foods[0][0]

    def _food_value(self, hero: object) -> int:
        return sum(FOOD_VALUES[item] * amount for item, amount in hero.inventory.items() if item in FOOD_VALUES)

    def _desired_food_value(self, day: int, season: str) -> int:
        if season == "spring":
            return 160 + day
        if season == "summer":
            return 260
        if season == "autumn":
            return 520
        return 120

    def _cook_if_useful(self, hero: object) -> GameAction | None:
        if not (hero.has("campfire") or hero.has("stove")):
            return None
        if hero.has("pumpkin"):
            return self._act("cook", "Turn pumpkin into real winter food.", item="pumpkin")
        if hero.has("wheat"):
            return self._act("cook", "Bake wheat into a better ration.", item="wheat")
        if hero.has("fish"):
            return self._act("cook", "Cook fish before eating it.")
        if hero.has("berries", 2) and hero.has("storage_barrel"):
            return self._act("cook", "Dry berries for the shelf.", item="berries")
        return None

    def _preserve_if_useful(self, hero: object, season: str) -> GameAction | None:
        if not hero.has("storage_barrel"):
            return None
        if hero.has("pumpkin") and season in {"autumn", "winter"}:
            return self._act("store", "Preserve pumpkin before winter takes the fields.", item="pumpkin")
        if hero.has("berries", 2) and season == "autumn":
            return self._act("store", "Dry berries for later.", item="berries")
        return None

    def _can_pay(self, hero: object, costs: dict[str, int]) -> bool:
        return all(hero.item_count(item) >= amount for item, amount in costs.items())

    def _move(self, target: str, think: str) -> GameAction:
        return self._act("move", think, target=target)

    def _move_to_pos(self, pos: object, think: str) -> GameAction:
        return self._act("move", think, x=pos.x, y=pos.y)

    def _outdoor_ap(self, sim: object, base: int) -> int:
        # Mirror ActionEngine._spend exactly so an AP guard never under-counts:
        # exhausted stamina doubles the base cost, then bad weather adds one.
        ap = base
        if sim.hero.stamina <= 0:
            ap *= 2
        if sim.world.weather in {"storm", "snow"} and not sim.hero.has("house_upgrade"):
            ap += 1
        return ap

    def _act(self, action: str, think: str, **args: object) -> GameAction:
        return GameAction(action=action, args=args, think=think, say=self._line_for(action))

    # A few lines per action, so the hero's voice — the 迷言 (memorable-quote)
    # supply the spec sells — does not collapse into one filler string. The
    # variant is picked from the day (pure, no RNG), keeping determinism intact.
    _LINES: dict[str, tuple[str, ...]] = {
        "till": ("The soil opens like a slow promise.", "Break the ground; ask it for a future.", "A field is just patience with edges."),
        "plant": ("Small seeds, long bets.", "I bury a little hope in the dirt.", "Grow slowly, but please grow."),
        "water": ("Grow, please. I am asking politely.", "A drink for you before one for me.", "Rain by hand is still rain."),
        "harvest": ("The soil answered.", "Payday, measured in leaves.", "What I planted, I now carry."),
        "chop": ("Wood now, warmth later.", "The forest lends; I will remember.", "Each swing is a wall I do not yet have."),
        "mine": ("Stone is just slow money.", "The rock gives grudgingly, but it gives.", "Dust on my hands, plans in my head."),
        "fish": ("If the sea has mercy, dinner has fins.", "Patience, baited.", "The water keeps its secrets and sometimes a meal."),
        "forage": ("The island hides snacks in strange places.", "Eyes down, hopes up.", "Whatever the island spares, I will take."),
        "craft": ("Hands, remember what the mind promised.", "A tool is a wish made of wood.", "Build the thing that builds the rest."),
        "build": ("One more piece of civilization.", "Today I out-stubborn the weather.", "Walls are arguments against winter."),
        "cook": ("Fire makes the difference between food and risk.", "A warm meal is a small victory.", "Better cooked than brave."),
        "eat": ("Fuel first, glory later.", "The body votes, and it votes for lunch.", "Quiet the stomach; free the mind."),
        "drink": ("A well-timed sip is a strategy.", "Water is the cheapest courage.", "Thirst loses to a steady hand."),
        "store": ("A full shelf is a calmer night.", "Save now, eat in the snow.", "Winter, I am getting ready for you."),
        "trade_accept": ("A fair swap is two people winning.", "Someone else's surplus, my survival.", "Trade turns scraps into plans."),
        "trade_decline": ("Not today, friend.", "I will keep what I have, thank you.", "A polite no is still a no."),
        "move": ("One foot, then the other.", "The island is wide; my legs are willing.", "Going to where the work is."),
        "rest": ("Even the stubborn must breathe.", "Rest is not surrender.", "I gather myself before the dark."),
        "write_diary": ("One line between me and the dark.", "If I write it down, the day was real.", "Ink is cheaper than forgetting."),
        "sleep": ("Tomorrow can carry the rest.", "I bank the day and close my eyes.", "Enough. The island will wait."),
    }

    def _line_for(self, action: str) -> str:
        variants = self._LINES.get(action)
        if not variants:
            return "One useful step."
        sim = getattr(self, "_sim", None)
        index = sim.world.day if sim is not None else 0
        return variants[index % len(variants)]
