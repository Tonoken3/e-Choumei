from __future__ import annotations

from spl.core.actions import GameAction
from spl.core.crops import FOOD_VALUES


class LocalPolicyAgent:
    """A deterministic, no-LLM survival brain used as the bundled cartridge."""

    def choose(self, sim: object) -> GameAction:
        hero = sim.hero
        world = sim.world

        if sim.current_offer and self._can_pay(hero, sim.current_offer.give):
            return self._act("trade_accept", "A trade can turn scraps into winter plans.", id=sim.current_offer.id)

        if hero.water < 34:
            if hero.has("well") or world.is_near(hero.pos, "water"):
                return self._act("drink", "Water first. Heroism is mostly plumbing.")
            move_ap = 2 if world.weather in {"storm", "snow"} and not hero.has("house_upgrade") else 1
            if hero.ap_left < move_ap:
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
                    return self._act("fish", "Fish for dinner before the stomach starts negotiating.")
                return self._move("water", "Move to the shore and fish.")
            if world.is_near(hero.pos, "forest") or world.tile_at(hero.pos) in {"forest", "grass", "field", "home"}:
                return self._act("forage", "Search for emergency food.")
            return self._move("forest", "Move toward forage.")

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

        material = self._needed_material(sim)
        if material:
            return self._gather_material(sim, material)

        if world.season == "winter":
            food = self._best_food(hero)
            if hero.hunger < 72 and food:
                return self._act("eat", f"Winter ration: {food}.", item=food)
            if hero.has("fishing_rod"):
                if world.is_near(hero.pos, "water"):
                    return self._act("fish", "Fish through winter; the field is asleep.")
                return self._move("water", "Winter food waits at the shore.")
            return self._act("forage", "Winter forage is thin, but thin is not zero.")

        if self._food_value(hero) < self._desired_food_value(world.day, world.season):
            if hero.has("fishing_rod") and world.is_near(hero.pos, "water"):
                return self._act("fish", "Build the food buffer.")
            if hero.has("fishing_rod"):
                return self._move("water", "Go fishing for reserves.")
            if world.tile_at(hero.pos) == "forest" or world.is_near(hero.pos, "forest"):
                return self._act("forage", "Gather wild food and seeds.")
            return self._move("forest", "Move toward forest forage.")

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

    def _next_craft(self, sim: object) -> GameAction | None:
        hero = sim.hero
        priorities = [
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
        if sim.world.season == "autumn":
            priorities = [
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
        for key in priorities:
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
        priorities = ["stone_axe", "hoe", "fishing_rod", "campfire", "storage_barrel", "well", "stove", "fence", "house_upgrade"]
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
        if material == "wood":
            if world.tile_at(sim.hero.pos) == "forest" or world.is_near(sim.hero.pos, "forest"):
                return self._act("chop", "Wood is tomorrow's tool.")
            return self._move("forest", "Move to trees for wood.")
        if material in {"stone", "clay", "iron_ore"}:
            if world.tile_at(sim.hero.pos) == "rock" or world.is_near(sim.hero.pos, "rock"):
                return self._act("mine", "Stone and clay unlock the camp.")
            return self._move("rock", "Move to rocks for materials.")
        if material in {"fiber", "seeds", "food"}:
            if world.tile_at(sim.hero.pos) == "forest" or world.is_near(sim.hero.pos, "forest") or world.tile_at(sim.hero.pos) in {"grass", "field"}:
                return self._act("forage", "Forage for fiber, seeds, or small meals.")
            return self._move("forest", "Move to forage.")
        return self._act("forage", "Search for whatever the island offers.")

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
        if sim.world.weather in {"storm", "snow"} and not sim.hero.has("house_upgrade"):
            return base + 1
        return base

    def _act(self, action: str, think: str, **args: object) -> GameAction:
        return GameAction(action=action, args=args, think=think, say=self._line_for(action))

    def _line_for(self, action: str) -> str:
        lines = {
            "eat": "Fuel first, glory later.",
            "drink": "A well-timed sip is a strategy.",
            "plant": "Small seeds, long bets.",
            "water": "Grow, please. I am asking politely.",
            "harvest": "The soil answered.",
            "craft": "Hands, remember what the mind promised.",
            "build": "One more piece of civilization.",
            "fish": "If the sea has mercy, dinner has fins.",
            "forage": "The island hides snacks in strange places.",
            "sleep": "Tomorrow can carry the rest.",
        }
        return lines.get(action, "One useful step.")
