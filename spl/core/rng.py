from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")


@dataclass
class GameRng:
    seed: int
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def random(self) -> float:
        return self._rng.random()

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def choice(self, items: Sequence[T]) -> T:
        if not items:
            raise ValueError("cannot choose from an empty sequence")
        return self._rng.choice(items)

    def chance(self, probability: float) -> bool:
        return self.random() < probability

    def weighted_choice(self, weighted: Iterable[tuple[T, float]]) -> T:
        choices = list(weighted)
        total = sum(max(weight, 0.0) for _, weight in choices)
        if total <= 0:
            raise ValueError("weighted choices must have a positive total")
        roll = self.random() * total
        upto = 0.0
        for item, weight in choices:
            upto += max(weight, 0.0)
            if roll <= upto:
                return item
        return choices[-1][0]

