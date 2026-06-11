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


try:  # the pixel/voxel frontend needs pygame; skip its tests if it is absent
    import os as _os

    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame as _pygame  # noqa: F401

    _HAS_PYGAME = True
except Exception:  # noqa: BLE001
    _HAS_PYGAME = False


@unittest.skipUnless(_HAS_PYGAME, "pygame not installed")
class PixelVoxelTests(unittest.TestCase):
    """Stage-4 neo-retro voxel frontend: iso picking + synthetic UI clicks."""

    def test_iso_screen_to_tile_round_trips_at_all_sprite_scales(self) -> None:
        from spl.ui.pixel import iso

        for scale in iso.SPRITE_SCALES:
            for offset in ((0, 0), (500, 200), (-37, 91)):
                ox, oy = offset
                for x in range(0, 20):
                    for y in range(0, 20):
                        cx, cy = iso.tile_center(x, y, ox, oy, scale)
                        got = iso.screen_to_tile(cx, cy, ox, oy, scale)
                        self.assertEqual(
                            (got.x, got.y), (x, y),
                            f"round-trip failed: scale {scale} offset {offset} cell ({x},{y})",
                        )

    def _fhd_app(self):
        from types import SimpleNamespace

        from spl.ui.pixel.app import PixelApp

        args = SimpleNamespace(
            seed=42, days=112, llm=False, cassette="x", manual=True, speed=2,
            scale=2, start_day=0, shots=0, shots_ui=False, shot_dir="/tmp/spl_test",
        )
        return PixelApp(args, headless=True)

    def test_window_defaults_to_full_hd(self) -> None:
        from spl.ui.pixel import iso

        app = self._fhd_app()
        self.assertEqual((app.lay.win_w, app.lay.win_h), (1920, 1080))
        # map band is full-width and a tall band above the HUD
        self.assertEqual(app.lay.map_rect.width, 1920)
        self.assertGreater(app.lay.map_rect.height, 600)
        # a discrete sprite scale was chosen and the factory matches it
        self.assertIn(app.lay.sprite_scale, iso.SPRITE_SCALES)
        self.assertEqual(app.factory.scale, app.lay.sprite_scale)

    def test_tile_under_picks_hero_tile_from_its_screen_centre(self) -> None:
        from spl.ui.pixel import iso

        app = self._fhd_app()
        hero = app.sim.hero.pos
        cx, cy = iso.tile_center(hero.x, hero.y, app.offset_x, app.offset_y,
                                 app.lay.sprite_scale)
        picked = app._tile_under(cx, cy)
        self.assertIsNotNone(picked)
        self.assertEqual((picked.x, picked.y), (hero.x, hero.y))

    def test_synthetic_click_on_popup_row_dispatches_action(self) -> None:
        from spl.ui.pixel import iso

        app = self._fhd_app()
        app.manual = True
        world = app.sim.world
        spot = world.find_nearest(
            app.sim.hero.pos, lambda p: world.tile_at(p) in {"grass", "beach"}
        )
        self.assertIsNotNone(spot)
        app.sim.hero.pos = spot
        cx, cy = iso.tile_center(spot.x, spot.y, app.offset_x, app.offset_y,
                                 app.lay.sprite_scale)
        # Clicking the hero's own tile opens the context popup...
        app._handle_click(cx, cy)
        self.assertIsNotNone(app.popup)
        # ...render once so the popup row rects exist, then click the first row.
        win = app.pg.Surface((app.lay.win_w, app.lay.win_h))
        app.render(win)
        rects = app.popup.get("rects", [])
        self.assertTrue(rects, "popup produced no clickable row rects")
        before_ap = app.sim.hero.ap_left
        target = rects[0]
        app._handle_click(target.centerx, target.centery)
        # The action either ran (AP spent) or the popup closed — never crashed.
        acted = app.popup is None or app.sim.hero.ap_left != before_ap
        self.assertTrue(acted)

    def test_synthetic_click_on_hud_button_toggles_pause(self) -> None:
        app = self._fhd_app()
        app._refresh_buttons()  # lay buttons out against the live layout
        pause = next(b for b in app.buttons if b.key == "pause")
        before = app.paused
        app._handle_click(pause.rect.centerx, pause.rect.centery)
        self.assertNotEqual(app.paused, before)

    def test_render_runs_clean_across_all_seasons(self) -> None:
        # Smoke test: the whole voxel pipeline (terrain cache + objects +
        # atmosphere + UI) renders without error in every season/weather.
        app = self._fhd_app()
        win = app.pg.Surface((app.lay.win_w, app.lay.win_h))
        for weather in ("sunny", "rain", "storm", "snow", "drought"):
            for season_day in (2, 30, 58, 86):
                app.sim.world.day = season_day
                app.sim.world.weather = weather
                app.render(win)  # must not raise


if __name__ == "__main__":
    unittest.main()

