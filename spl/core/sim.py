from __future__ import annotations

import tomllib
from pathlib import Path

from spl.agent.memory import Memory

from .actions import ACTION_WORDS, ActionEngine, ActionResult, GameAction
from .crops import FOOD_VALUES, CropBook
from .crafting import RecipeBook
from .events import EventBook, MerchantOffer
from .hero import Hero
from .rng import GameRng
from .world import SEASON_NAMES, WEATHER_NAMES, World

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The "混乱" voice (spec §4.3): a small pool of disoriented lines so a confused
# hero — the lovable, struggling small model — does not repeat one stock line.
CONFUSION_LINES = (
    "Where am I? The island has too many edges.",
    "Wait... which way was the sea?",
    "My hands forgot what they were holding.",
    "The map in my head went quiet.",
    "Was I going somewhere? The thought slipped away.",
    "Too many edges. I cannot find the next step.",
    "I blink, and the plan is gone.",
    "The wind said something. I did not catch it.",
)


class Simulation:
    def __init__(
        self,
        seed: int = 42,
        max_days: int | None = None,
        game_config: Path | None = None,
        data_dir: Path | None = None,
    ) -> None:
        self.seed = seed
        self.rng = GameRng(seed)
        self.data_dir = data_dir or PROJECT_ROOT / "data"
        self.config_path = game_config or PROJECT_ROOT / "config" / "game.toml"
        self.config = self._load_config(self.config_path)
        world_cfg = self.config.get("world", {})
        hero_cfg = self.config.get("hero", {})
        self.ap_per_day = int(world_cfg.get("ap_per_day", 12))
        self.max_days = max_days or int(world_cfg.get("days_per_year", 112))
        self.crop_book = CropBook.load(self.data_dir / "crops.toml")
        self.recipe_book = RecipeBook.load(self.data_dir / "recipes.toml")
        self.event_book = EventBook.load(self.data_dir / "events.toml")
        self.world = World.generate(
            width=int(world_cfg.get("width", 20)),
            height=int(world_cfg.get("height", 20)),
            season_length=int(world_cfg.get("season_length", 28)),
            rng=self.rng,
        )
        self.hero = Hero(
            pos=self.world.start_pos,
            hp=int(hero_cfg.get("hp", 100)),
            hunger=int(hero_cfg.get("hunger", 78)),
            water=int(hero_cfg.get("water", 82)),
            stamina=int(hero_cfg.get("stamina", 92)),
            sanity=int(hero_cfg.get("sanity", 85)),
            ap_left=self.ap_per_day,
            inventory=dict(self.config.get("start_inventory", {})),
        )
        self.engine = ActionEngine()
        self.memory = Memory()
        self.day_log: list[str] = []
        self.full_log: list[str] = []
        self.current_offer: MerchantOffer | None = None
        self.advice_from_heaven: str | None = None
        self.completed = False
        self.failed = False
        self.result_reason = ""
        # Optional LLM diarist: an object exposing write_diary(sim, season, weather)
        # -> str | None. When present, the hero's nightly diary is authored by the
        # model (spec §5); otherwise the deterministic template in Memory is used.
        self.diarist: object | None = None
        self.log(f"Day {self.world.day} begins: {SEASON_NAMES[self.world.season]}, {WEATHER_NAMES[self.world.weather]}.")
        self._start_day_events()

    @property
    def done(self) -> bool:
        return self.completed or self.failed or not self.hero.alive

    def step(self, request: GameAction, confuse_on_invalid: bool = False) -> ActionResult:
        if self.done:
            return ActionResult(False, "Simulation is already finished.")
        # An action word the world does not recognise can never become reality.
        # Make it confusion unconditionally so the world keeps turning no matter
        # how the brain is wired (spec §4.3: クラッシュは存在しない).
        if request.action not in ACTION_WORDS:
            return self.confuse(f"Unknown action: {request.action}")
        if self.hero.sanity <= 7 and self.rng.chance(0.20):
            return self.confuse("Sanity collapsed into static.")
        result = self.engine.perform(self, request)
        if not result.ok and confuse_on_invalid:
            # confuse() fully resolves the turn (safe action + its own end_day),
            # so return immediately: never let step() run end_day() a second time.
            return self.confuse(result.message)
        self.log(result.message)
        if result.end_day or self.hero.ap_left <= 0:
            self.end_day()
        if not self.hero.alive:
            self.failed = True
            self.result_reason = "The hero died."
        return result

    def confuse(self, reason: str) -> ActionResult:
        self.hero.confusion_count += 1
        self.hero.adjust("sanity", -3)
        self.log(f"Confusion: {reason}")
        self.hero.spoken_lines.append(self.rng.choice(CONFUSION_LINES))
        fallback = GameAction.safe("rest") if self.hero.ap_left >= 2 else GameAction.safe("sleep")
        result = self.engine.perform(self, fallback)
        self.log("[confused] " + result.message)
        if result.end_day or self.hero.ap_left <= 0:
            self.end_day()
        return ActionResult(True, "The hero became confused and took a safe action.", result.end_day, result.consumed_ap)

    def end_day(self) -> None:
        if self.done:
            return
        grow_messages = self.world.grow_crops(self.crop_book)
        for message in grow_messages:
            self.log(message)
        self._daily_decay()
        season = SEASON_NAMES[self.world.season]
        weather = WEATHER_NAMES[self.world.weather]
        llm_line = None
        if self.diarist is not None:
            try:
                llm_line = self.diarist.write_diary(self, season, weather)
            except Exception as exc:  # noqa: BLE001 - the diary must never crash the night
                self.log(f"Diary (LLM) unavailable; the hero writes by hand: {exc}")
        self.memory.nightly_entry(
            self.world.day, season, weather, self.day_log, self.hero.hp, llm_line=llm_line
        )
        self.hero.days_survived = self.world.day if self.hero.alive else max(0, self.world.day - 1)
        if not self.hero.alive:
            self.failed = True
            self.result_reason = f"Fell on day {self.world.day}."
            self.log(self.result_reason)
            return
        if self.world.day >= self.max_days:
            self.completed = True
            self.result_reason = f"Survived {self.max_days} days."
            self.log(self.result_reason)
            return
        self.world.day += 1
        self.world.weather = self.world.next_weather(self.rng, self.world.weather)
        self.day_log = []
        self.hero.ap_left = self.ap_per_day
        self.hero.adjust("stamina", 48 if self.hero.has("house_upgrade") else 42)
        if self.hero.has("well"):
            self.hero.adjust("water", 12)
        if self.hero.has("campfire"):
            self.hero.adjust("sanity", 2)
        self.log(f"Day {self.world.day} begins: {SEASON_NAMES[self.world.season]}, {WEATHER_NAMES[self.world.weather]}.")
        self._start_day_events()

    def apply_bellyache(self, message: str) -> None:
        self.hero.adjust("hp", -5)
        self.hero.adjust("sanity", -4)
        self.hero.ap_left = 0
        self.log(message + " Bellyache ends the day.")

    def log(self, message: str) -> None:
        text = f"D{self.world.day:03d} AP{self.hero.ap_left:02d}: {message}"
        self.day_log.append(message)
        self.full_log.append(text)

    def score(self) -> int:
        return self.hero.days_survived * 10 + self.hero.civilization_points() + self.hero.sanity

    def status_line(self) -> str:
        hero = self.hero
        world = self.world
        return (
            f"Day {world.day}/{self.max_days} {SEASON_NAMES[world.season]}-{world.day_in_season} "
            f"{WEATHER_NAMES[world.weather]} AP {hero.ap_left}/{self.ap_per_day} | "
            f"HP {hero.hp} Hu {hero.hunger} Wa {hero.water} St {hero.stamina} Sa {hero.sanity}"
        )

    def set_diarist(self, diarist: object | None) -> None:
        self.diarist = diarist

    def _start_day_events(self) -> None:
        if self.world.day > 1 and self.world.day % self.event_book.merchant_interval == 0:
            self.current_offer = self.rng.choice(self.event_book.offers)
            self.log("Merchant arrives: " + self.current_offer.describe())
        if self.world.day > 4 and not self.hero.has("fence") and self.rng.chance(self.event_book.dog_chance):
            stolen = self._steal_food()
            if stolen:
                self.hero.adjust("hp", -3)
                self.log(f"Wild dogs raided the food store and stole {stolen}. HP -3.")
        if self.world.weather == "storm" and not self.hero.has("house_upgrade") and self.rng.chance(0.18):
            self.hero.adjust("hp", -3)
            self.hero.adjust("sanity", -2)
            self.log("Storm damage rattles the house. HP -3.")

    def _steal_food(self) -> str:
        foods = [item for item in self.hero.inventory if item in FOOD_VALUES and self.hero.item_count(item) > 0]
        if not foods:
            return ""
        item = self.rng.choice(foods)
        amount = max(1, self.hero.item_count(item) // 4)
        self.hero.remove_item(item, amount)
        return f"{amount} {item}"

    def _daily_decay(self) -> None:
        hero = self.hero
        hero.adjust("hunger", -15)
        hero.adjust("water", -20)
        hero.adjust("sanity", -1 if hero.has("house_upgrade") else -2)
        if self.world.weather == "storm":
            hero.adjust("sanity", -3)
        if self.world.weather == "snow":
            hero.adjust("stamina", -4)
        if self.world.season == "winter" and not hero.has("house_upgrade"):
            hero.adjust("hp", -4)
            hero.adjust("sanity", -2)
        if hero.hunger <= 0:
            hero.adjust("hp", -10)
            self.log("Starvation bites. HP -10.")
        if hero.water <= 0:
            hero.adjust("hp", -15)
            self.log("Dehydration bites. HP -15.")

    def _load_config(self, path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        return tomllib.loads(path.read_text(encoding="utf-8"))

