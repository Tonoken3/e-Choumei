from __future__ import annotations

import unittest

from spl.agent.policy import LocalPolicyAgent
from spl.agent.schema import ActionParseError, parse_action_text
from spl.core.actions import GameAction
from spl.core.sim import Simulation


def run_local(seed: int, days: int) -> Simulation:
    sim = Simulation(seed=seed, max_days=days)
    agent = LocalPolicyAgent()
    for _ in range(days * 60):
        if sim.done:
            break
        sim.step(agent.choose(sim))
    return sim


class SplSimulationTests(unittest.TestCase):
    def test_seed_42_survives_one_year_with_local_policy(self) -> None:
        sim = run_local(seed=42, days=112)
        self.assertTrue(sim.completed, sim.result_reason)
        self.assertEqual(sim.hero.days_survived, 112)
        self.assertGreater(sim.score(), 1000)

    def test_local_policy_is_deterministic_for_same_seed(self) -> None:
        left = run_local(seed=45, days=30)
        right = run_local(seed=45, days=30)
        self.assertEqual(left.completed, right.completed)
        self.assertEqual(left.score(), right.score())
        self.assertEqual(left.hero.inventory, right.hero.inventory)
        self.assertEqual(left.full_log, right.full_log)

    def test_invalid_llm_action_becomes_confusion_without_crash(self) -> None:
        sim = Simulation(seed=7, max_days=3)
        result = sim.step(GameAction(action="invent_castle"), confuse_on_invalid=True)
        self.assertTrue(result.ok)
        self.assertEqual(sim.hero.confusion_count, 1)
        self.assertTrue(sim.hero.alive)

    def test_unknown_action_confuses_even_without_flag(self) -> None:
        # The world must keep turning no matter how the brain is wired: an
        # unknown action is confusion even when confuse_on_invalid is False.
        sim = Simulation(seed=7, max_days=3)
        result = sim.step(GameAction(action="cast_spell"))
        self.assertTrue(result.ok)
        self.assertEqual(sim.hero.confusion_count, 1)

    def test_confusion_at_low_ap_advances_exactly_one_day(self) -> None:
        # Regression for the double-end-day bug: a confused turn with a "sleep"
        # fallback must advance the day once, applying daily decay once.
        sim = Simulation(seed=42, max_days=10)
        sim.hero.ap_left = 1
        sim.hero.stamina = 50
        day_before = sim.world.day
        hunger_before = sim.hero.hunger
        sim.step(GameAction(action="mine"), confuse_on_invalid=True)
        self.assertEqual(sim.world.day, day_before + 1)
        self.assertEqual(hunger_before - sim.hero.hunger, 15)

    def test_local_policy_never_hangs(self) -> None:
        # Seeds that used to livelock the bundled brain must now resolve to a
        # terminal state (survive or fall) within the turn budget.
        for seed in (19, 27, 28, 37, 39, 40, 50, 58):
            sim = run_local(seed=seed, days=112)
            self.assertTrue(sim.done, f"seed {seed} did not terminate")


class SplSchemaTests(unittest.TestCase):
    def test_parse_action_json_with_fence_and_trailing_comma(self) -> None:
        parsed = parse_action_text(
            '```json\n{"think":"hungry","action":"eat","args":{"item":"berries",},"say":"Fuel.",}\n```'
        )
        self.assertEqual(parsed.action, "eat")
        self.assertEqual(parsed.args["item"], "berries")

    def test_reject_unknown_action(self) -> None:
        with self.assertRaises(ActionParseError):
            parse_action_text('{"think":"","action":"fly","args":{},"say":""}')

    def test_parse_skips_prose_braces(self) -> None:
        parsed = parse_action_text(
            'Sure! Here is my move {note}: {"action":"eat","args":{"item":"berries"},'
            '"think":"hungry","say":"fuel"} — done.'
        )
        self.assertEqual(parsed.action, "eat")
        self.assertEqual(parsed.args["item"], "berries")


if __name__ == "__main__":
    unittest.main()

