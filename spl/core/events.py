from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MerchantOffer:
    id: str
    give: dict[str, int]
    take: dict[str, int]

    def describe(self) -> str:
        give = ", ".join(f"{amount} {item}" for item, amount in self.give.items())
        take = ", ".join(f"{amount} {item}" for item, amount in self.take.items())
        return f"{self.id}: give {give} -> receive {take}"


class EventBook:
    def __init__(self, merchant_interval: int, offers: list[MerchantOffer], dog_chance: float) -> None:
        self.merchant_interval = merchant_interval
        self.offers = offers
        self.dog_chance = dog_chance

    @classmethod
    def load(cls, path: Path) -> "EventBook":
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        merchant = data.get("merchant", {})
        offers = [
            MerchantOffer(
                id=row["id"],
                give=dict(row.get("give", {})),
                take=dict(row.get("take", {})),
            )
            for row in merchant.get("offer", [])
        ]
        dog = data.get("wild_dog", {})
        return cls(
            merchant_interval=int(merchant.get("interval", 7)),
            offers=offers,
            dog_chance=float(dog.get("base_chance", 0.04)),
        )

