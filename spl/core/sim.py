from __future__ import annotations

import tomllib
from pathlib import Path

from spl.agent.memory import Memory

from .actions import ACTION_WORDS, ActionEngine, ActionResult, GameAction
from .crops import (
    FOOD_VALUES,
    CropBook,
    merchant_planting_hint,
    monument_inscription,
)
from .crafting import RecipeBook
from .events import EventBook, MerchantOffer
from .hero import Hero
from .rng import GameRng
from .world import SEASON_NAMES, WEATHER_NAMES, World

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The "混乱" voice (spec §4.3): a small pool of disoriented lines so a confused
# hermit — the lovable, struggling small model — does not repeat one stock line.
CONFUSION_LINES = (
    "ここは……どこじゃ……？",
    "我は誰ぞ。庵はいずこ。",
    "思案が、風に攫われてしもうた。",
    "ゆく河の流れは絶えずして……はて、何の話であったか。",
    "月がふたつ見える。いや、ひとつか。",
    "道を忘れた。道のほうも、我を忘れたらしい。",
    "手に何を持っておったか、手に聞いても答えぬ。",
    "波の音が、右からも左からも聞こえる。",
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
        # The watcher's standing strategy (作戦 / DQの「さくせん」, F1のピットウォール指示).
        # Deliberately PERSISTENT: nothing in the day/night cycle clears it — once
        # set it is inherited every day until the watcher changes or lifts it.
        self.advice_from_heaven: str | None = None
        # How many times a *new, non-empty, different* directive was issued. Lets
        # the 戦績 tell an unassisted run (0回) from a directed one.
        self.strategy_changes: int = 0
        self.completed = False
        self.failed = False
        self.result_reason = ""
        # Optional LLM diarist: an object exposing write_diary(sim, season, weather)
        # -> str | None. When present, the hero's nightly diary is authored by the
        # model (spec §5); otherwise the deterministic template in Memory is used.
        self.diarist: object | None = None
        # 古い石碑: epitaphs carved on the BACK of the settlers' stone — the
        # mottos of the last 1-2 past lives. The UIs that load the 冒険の書 set
        # this BEFORE play starts (see cli.py / pixel app); empty means the stone
        # has only its agronomy face. The graves teach when the book is on.
        self.monument_epitaphs: list[str] = []
        # 刻む:句 the PREVIOUS hermits voluntarily cut into the FRONT of the stone
        # (per cassette+island). Unlike the epitaphs (auto-carved mottos) and the
        # agronomy face, these are a hermit's own free choice — voluntary
        # trans-generational communication. The UI layer (cli.py / pixel app)
        # loads the stone before play and calls set_stone_carvings(); empty means
        # no past hermit has carved here yet.
        self.stone_carvings: list[str] = []
        # 句 carved THIS life, with the day each was cut. Persisted at run end so
        # the next hermit on this island inherits them; also shown on the result
        # screen. Entries are (day:int, text:str).
        self.carvings_made: list[tuple[int, str]] = []
        # At most ONE carve per day; reset every morning in end_day().
        self.carved_today: bool = False
        self.log(f"Day {self.world.day} begins: {SEASON_NAMES[self.world.season]}, {WEATHER_NAMES[self.world.weather]}.")
        self._log_monument()
        self._start_day_events()

    @property
    def done(self) -> bool:
        return self.completed or self.failed or not self.hero.alive

    def step(self, request: GameAction, confuse_on_invalid: bool = False) -> ActionResult:
        # confuse_on_invalid is kept for API compatibility (callers still pass it),
        # but a VALID action word the world rejects is no longer routed to
        # confuse(): it "fumbles" instead — reality simply refuses, time passes,
        # one AP is lost, no sanity hit. 混乱 stays only for unknown action words
        # and the sanity<=7 collapse (思考予算 / scoring honesty).
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
        if not result.ok:
            # Fumble: a known action word that the world refused (wrong tile, not
            # adjacent, missing materials/seed, already planted, no AP...). The
            # hero loses one AP — reality refuses and time still passes — but
            # keeps their wits (no 混乱, no sanity loss). Faster rigs that "think"
            # more avoid these world-rejects; that is the 思考予算 race.
            self.hero.ap_left = max(0, self.hero.ap_left - 1)
            self.log(result.message + " [fumble -1AP]")
            if self.hero.ap_left <= 0:
                self.end_day()
            if not self.hero.alive:
                self.failed = True
                if not self.result_reason:
                    self.result_reason = "仙人は果てた。"
            return result
        self.log(result.message)
        if result.end_day or self.hero.ap_left <= 0:
            self.end_day()
        if not self.hero.alive:
            self.failed = True
            if not self.result_reason:
                self.result_reason = "仙人は果てた。"
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
            self.result_reason = f"{self.world.day}日目、仙人は果てた。"
            self.log(self.result_reason)
            return
        if self.world.day >= self.max_days:
            self.completed = True
            self.result_reason = f"{self.max_days}日を生き抜いた。"
            self.log(self.result_reason)
            return
        self.world.day += 1
        self.world.weather = self.world.next_weather(self.rng, self.world.weather)
        self.day_log = []
        self.carved_today = False
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

    def set_strategy(self, text: str | None) -> None:
        """Set (or clear) the standing 作戦. Strips ``text``; an empty/None value
        clears the directive. ``strategy_changes`` counts only a genuinely new
        directive — a non-empty string different from the one already standing —
        so re-sending the same line, or clearing, never inflates the count.

        The directive itself is never auto-cleared by the day/night cycle: it
        persists until the watcher changes it (DQの「さくせん」/F1の監督指示)."""
        cleaned = (text or "").strip() or None
        if cleaned is not None and cleaned != self.advice_from_heaven:
            self.strategy_changes += 1
            # 啓示: a NEW directive rings once in the world log so it lands in
            # the "recent" window the brain reads. Once — a hermit who hears
            # voices every turn is a different character (電波系).
            self.log(f"Heaven speaks: {cleaned}")
        elif cleaned is None and self.advice_from_heaven is not None:
            self.log("Heaven falls silent.")
        self.advice_from_heaven = cleaned

    def set_diarist(self, diarist: object | None) -> None:
        self.diarist = diarist

    def _log_monument(self) -> None:
        """Carve the settlers' agronomy onto the day-1 log (古い石碑). One line,
        built from REAL crop data so it can never drift from the world's rules.
        Lands in full_log; the day-1 recent window shows it to the brain.

        刻む: when a previous hermit voluntarily cut a verse into this stone, each
        such line is logged too — 先人の手で句が刻まれている. Normally empty on
        day 1 (the UI sets stone_carvings AFTER construction, so set_stone_carvings
        does the logging); this branch covers any caller that sets them first."""
        self.log(monument_inscription(self.crop_book))
        for verse in self.stone_carvings:
            self.log(f"石碑には先人の手で句が刻まれている: 『{verse}』")

    def set_stone_carvings(self, carvings: list[str]) -> None:
        """刻む: the句 PAST hermits voluntarily cut into the front of the stone
        (most-recent last). Called by the UI layer BEFORE play — independent of
        --book, since the stone always remembers. Logs each verse once so it
        lands in the day-1 'recent' window the brain reads, and makes the
        observer's monument value carry a compact 先人の句 suffix. Blank entries
        are dropped; a no-op when nothing is set. Mirrors set_monument_epitaphs,
        but these are the hermit's own free carvings, not auto-carved mottos."""
        cleaned = [str(c).strip() for c in (carvings or []) if str(c).strip()]
        self.stone_carvings = cleaned
        for verse in cleaned:
            self.log(f"石碑には先人の手で句が刻まれている: 『{verse}』")

    def set_monument_epitaphs(self, epitaphs: list[str]) -> None:
        """Carve up to 2 past-life mottos onto the BACK of the stone and ring
        them once in the log (古い石碑の裏). Called by the --book wiring BEFORE
        play, so on day 1 the graves teach. Empty/blank entries are dropped;
        a no-op when nothing is set (the stone keeps only its agronomy face)."""
        cleaned = [str(e).strip() for e in (epitaphs or []) if str(e).strip()][:2]
        self.monument_epitaphs = cleaned
        for motto in cleaned:
            self.log(f"石碑の裏に、後から刻まれた言葉がある: 『{motto}』")

    def _start_day_events(self) -> None:
        if self.world.day > 1 and self.world.day % self.event_book.merchant_interval == 0:
            self.current_offer = self.rng.choice(self.event_book.offers)
            self.log("Merchant arrives: " + self.current_offer.describe())
            # 行商人の世間話: one extra line of deterministic planting small-talk,
            # reckoned from the current day + crop data (no RNG). Lands in the
            # day_log -> recent window, so the brain reads the lead-time hint.
            hint = merchant_planting_hint(self.crop_book, self.world.day, self.world.season)
            self.log(f"行商人は世間話に言う: 『{hint}』")
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

