from __future__ import annotations

"""神のレバー — the divine miracle console (共同 / co-op layer).

This is the *world-or-mind* layer that sits above 観戦 (watching) and 作戦 (the
standing order). A human player who has earned 神力 (divine power) may QUEUE a
miracle at any time; nothing happens immediately — **神は夜に働く** (the god works
at night): every miracle resolves at the night boundary (``Simulation.end_day`` /
``_start_day_events``). This single rule dissolves the三つの厄介: it never fights
the pixel UI's ``_pending_action``, never confuses the 承認 pause, and keeps
replays deterministic.

Two laws guard the design:

1. **奇跡は世界か心に触れる。手には触れない。** A miracle never dispatches a
   ``GameAction`` for the hermit — it changes the weather, the inventory, the
   merchant's arrival, or whispers a 勅命 into the next observation. The hand
   stays the hermit's (or the brain's) own, so manual mode never collapses.
2. **sim.rng を汚さない。** A forced weather still *draws* ``next_weather`` and
   throws the result away, so the world RNG stream stays bit-identical to an
   un-helped run (a seed-comparison test guards this). The merchant lottery runs
   on a DEDICATED ``GameRng(seed ^ MERCHANT_RNG_SALT)`` channel, never the world
   RNG.

Everything here is deterministic given ``(seed, miracle_log)``: no miracle
internal ever consumes ``sim.rng``.
"""

from dataclasses import dataclass, field

# ===========================================================================
# 神力経済 = A-hybrid (ケンの決定, AskUserQuestion で確定)
# 開始3 / 7日ごとの朝に+1 / 保有上限5。RNG 不使用の決定論。年間総量 ≈ 18-19。
# ===========================================================================
STARTING_POWER = 3
POWER_CAP = 5
POWER_GRANT_INTERVAL = 7  # every 7 days, +1 at the morning boundary
POWER_GRANT_AMOUNT = 1

# The dedicated merchant-lottery RNG salt: 0x4D495241 == b"MIRA" (ミラクル).
# The 行商人召喚 offer is drawn on GameRng(seed ^ MERCHANT_RNG_SALT) so it never
# perturbs the world RNG stream (which still owns the natural merchant arrivals).
MERCHANT_RNG_SALT = 0x4D495241

# 行商人を呼ぶ: cooldown in days since the LAST merchant visit (natural OR summoned).
MERCHANT_COOLDOWN_DAYS = 3

# 恵みのマナ: the whitelist. ONLY raw, low-value consumables — never wood/stone/
# iron or crafted goods (those feed the civilization score directly, so a miracle
# must not hand them out). Each grant is capped per-cast.
MANNA_WHITELIST = {
    "berries": 3,
    "fish": 3,
    "mushroom": 3,
    "turnip_seed": 3,
    "wheat_seed": 3,
    "tomato_seed": 3,
    "pumpkin_seed": 3,
}

# A flavour line per manna item — the morning log reads as a natural gift of the
# island, not "the inventory changed" (浜に魚が打ち上げられていた式).
MANNA_FLAVOR = {
    "berries": "茂みが一夜で実をたわわに付けていた",
    "fish": "浜に魚が打ち上げられていた",
    "mushroom": "庵のそばに茸が生えそろっていた",
    "turnip_seed": "風が運んだか、カブの種が軒先に零れていた",
    "wheat_seed": "風が運んだか、小麦の種が軒先に零れていた",
    "tomato_seed": "風が運んだか、トマトの種が軒先に零れていた",
    "pumpkin_seed": "風が運んだか、カボチャの種が軒先に零れていた",
}

# 天候の奇跡: the natural weather PALETTE per season — a forced weather must stay
# inside what the island could roll on its own that season (so the奇跡 bends
# fortune, it does not break physics). Mirrors World.next_weather's season table.
SEASON_WEATHER_PALETTE = {
    "spring": ("sunny", "rain", "storm"),
    "summer": ("sunny", "rain", "storm", "drought"),
    "autumn": ("sunny", "rain", "storm"),
    "winter": ("sunny", "snow", "storm"),
}

# The five miracles and their costs (神力).
MIRACLE_COSTS = {
    "weather": 1,   # 🌦 天候の奇跡   — name tomorrow's weather
    "manna": 2,     # 🎁 恵みのマナ   — a consumable in tomorrow's stores
    "merchant": 2,  # 🛖 行商人を呼ぶ — a merchant arrives tomorrow
    "oracle": 3,    # 🗲 神託(勅命)  — a one-day command on tomorrow's top observation
    "dream": 1,     # 💭 夢のお告げ   — etched into tonight's memory, ~7d echo
}

# Stable display order (palette + result breakdown) and JP labels.
MIRACLE_ORDER = ("weather", "manna", "merchant", "oracle", "dream")
MIRACLE_LABELS = {
    "weather": "天候の奇跡",
    "manna": "恵みのマナ",
    "merchant": "行商人を呼ぶ",
    "oracle": "神託",
    "dream": "夢のお告げ",
}


@dataclass
class DivineState:
    """The god's ledger on a Simulation. Mutated only by ``queue_miracle`` (which
    validates + consumes 神力 + logs) and by the night-boundary resolvers in
    ``Simulation``. ``score()`` never reads any of this — 神力 is not a stat.

    Pending effects are buffered here and applied at the NEXT night boundary
    (神は夜に働く). ``miracle_log`` is the deterministic record: replaying the
    same (seed, miracle_log) reproduces the full_log exactly."""

    power: int = STARTING_POWER
    miracles_used: int = 0
    # (day, kind, args) for every successfully queued miracle — the共同 record.
    miracle_log: list[tuple] = field(default_factory=list)

    # -- pending effects, resolved at the night boundary --------------------
    forced_weather: str | None = None        # set tomorrow's weather to this
    pending_manna: dict[str, int] = field(default_factory=dict)  # item -> amount
    pending_merchant: bool = False           # summon a merchant tomorrow morning
    pending_oracle: str | None = None        # 勅命, promoted at the night boundary
    divine_command: str | None = None        # 勅命 live in tomorrow's top observation
    pending_dream: str | None = None         # an お告げ to etch into tonight's memory

    # -- constraint bookkeeping (last day each was used; 0 == never) --------
    last_forced_weather_day: int = 0  # 天候連日不可
    last_merchant_day: int = 0        # 商人3日CD (set by natural arrivals too)
    last_oracle_day: int = 0          # 神託1日1回


def power_after_grant(power: int, day: int) -> int:
    """The 神力 after the morning grant on ``day``: +1 on the morning of every
    POWER_GRANT_INTERVAL-th day (day 7, 14, 21 ...), capped at POWER_CAP.
    Deterministic — never touches any RNG. Called by end_day for the NEW day
    after the boundary increments world.day. Day 1 is the start (no grant), so
    the opening 神力 stays exactly STARTING_POWER (per A-hybrid: 開始3・7日目朝+1)."""
    if day >= POWER_GRANT_INTERVAL and day % POWER_GRANT_INTERVAL == 0:
        return min(POWER_CAP, power + POWER_GRANT_AMOUNT)
    return power


def validate_miracle(divine: DivineState, day: int, season: str,
                     kind: str, args: dict | None) -> tuple[bool, str]:
    """Pure validator: may this miracle be queued NOW? Returns (ok, reason).
    Checks 神力残, the per-miracle constraints (連日天候 / 商人CD / 神託1日1回 /
    マナ whitelist), but does NOT mutate anything. ``day``/``season`` are the
    CURRENT day/season (the miracle resolves the night that ends this day)."""
    args = args or {}
    cost = MIRACLE_COSTS.get(kind)
    if cost is None:
        return False, f"未知の奇跡: {kind}"

    # The per-miracle CONSTRAINT is checked BEFORE the 神力 balance, so a player
    # who has already used a once-per-day lever (or is on a cooldown) is told the
    # honest reason — 「一日に一度」/「まだ来られぬ」/「連日不可」 — instead of a
    # misleading 「神力が足りぬ」 (the constraint is what actually blocks them).
    if kind == "weather":
        target = str(args.get("weather", "")).strip()
        palette = SEASON_WEATHER_PALETTE.get(season, ())
        if target not in palette:
            return False, f"この季節に「{target}」は呼べぬ（候補: {', '.join(palette)}）"
        # 連日不可: a forced weather cannot follow yesterday's forced weather.
        if divine.last_forced_weather_day == day:
            return False, "天候の奇跡は連日は起こせぬ"
    elif kind == "manna":
        item = str(args.get("item", "")).strip()
        if item not in MANNA_WHITELIST:
            return False, f"「{item}」は恵みに含まれぬ（{', '.join(MANNA_WHITELIST)}）"
    elif kind == "merchant":
        # 3日CD since the last merchant (natural or summoned).
        if divine.last_merchant_day and day - divine.last_merchant_day < MERCHANT_COOLDOWN_DAYS:
            wait = MERCHANT_COOLDOWN_DAYS - (day - divine.last_merchant_day)
            return False, f"商人はまだ来られぬ（あと{wait}日）"
        if divine.pending_merchant:
            return False, "既に商人を呼んでいる"
    elif kind == "oracle":
        text = str(args.get("text", "")).strip()
        if not text:
            return False, "神託の言葉が空じゃ"
        # 1日1回: only one oracle may be queued per day.
        if divine.last_oracle_day == day:
            return False, "神託は一日に一度きり"
    elif kind == "dream":
        text = str(args.get("text", "")).strip()
        if not text:
            return False, "お告げの言葉が空じゃ"
    else:
        return False, f"未知の奇跡: {kind}"

    # Only after the constraint passes do we charge 神力.
    if divine.power < cost:
        return False, f"神力が足りぬ（要{cost}・残{divine.power}）"
    return True, ""
