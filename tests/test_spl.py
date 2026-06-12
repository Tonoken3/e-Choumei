from __future__ import annotations

import threading
import time
import unittest
import unittest.mock
from collections import deque
from concurrent.futures import Future

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


class ThinkingBudgetTests(unittest.TestCase):
    """思考予算: inference speed → tier → survival depth (no LLM needed)."""

    def test_tier_for_tps_boundaries(self) -> None:
        from spl.agent.llm_client import tier_for_tps

        cases = {
            0.0: "雲水",
            29.9: "雲水",
            30.0: "行者",
            99.9: "行者",
            100.0: "羅漢",
            299.9: "羅漢",
            300.0: "仙界",
            5000.0: "仙界",
        }
        for tps, name in cases.items():
            self.assertEqual(tier_for_tps(tps).name, name, f"tps={tps}")

    def test_tier_budgets_escalate_with_speed(self) -> None:
        from spl.agent.llm_client import tier_for_tps

        cloud = tier_for_tps(10)     # 雲水
        sage = tier_for_tps(50)      # 行者
        arhat = tier_for_tps(150)    # 羅漢
        immortal = tier_for_tps(400)  # 仙界
        # token budgets grow monotonically with speed
        self.assertLess(cloud.max_tokens, sage.max_tokens)
        self.assertLess(sage.max_tokens, arhat.max_tokens)
        self.assertLess(arhat.max_tokens, immortal.max_tokens)
        # 雲水: no repair, no verify, single candidate
        self.assertFalse(cloud.repair)
        self.assertFalse(cloud.verify)
        self.assertEqual(cloud.candidates, 1)
        # verify only from 羅漢 up; only 仙界 proposes twice
        self.assertFalse(sage.verify)
        self.assertTrue(arhat.verify)
        self.assertTrue(immortal.verify)
        self.assertEqual(immortal.candidates, 2)

    def test_default_tps_before_samples_is_gyoja(self) -> None:
        from spl.agent.llm_client import Cassette, OpenAICompatibleBrain

        brain = OpenAICompatibleBrain(Cassette(name="x", base_url="http://localhost:1/v1"))
        # No samples yet: assume 80 TPS -> 行者.
        self.assertEqual(brain.avg_tps(), 80.0)
        self.assertEqual(brain.current_tier().name, "行者")

    def test_rolling_tps_average_over_window(self) -> None:
        from spl.agent.llm_client import Cassette, OpenAICompatibleBrain

        brain = OpenAICompatibleBrain(Cassette(name="x", base_url="http://localhost:1/v1"))
        # 100 tokens in 0.5s -> 200 TPS; two samples average to 200 -> 羅漢.
        brain._record_tps(100, 0.5)
        brain._record_tps(120, 0.6)  # also 200 TPS
        self.assertAlmostEqual(brain.avg_tps(), 200.0, places=3)
        self.assertEqual(brain.current_tier().name, "羅漢")
        # The window only keeps the last 8 samples.
        for _ in range(20):
            brain._record_tps(10, 1.0)  # 10 TPS each
        self.assertEqual(len(brain._tps_samples), brain._TPS_WINDOW)
        self.assertAlmostEqual(brain.avg_tps(), 10.0, places=3)
        self.assertEqual(brain.current_tier().name, "雲水")

    def test_forced_tps_overrides_measurement(self) -> None:
        from spl.agent.llm_client import Cassette, OpenAICompatibleBrain

        brain = OpenAICompatibleBrain(
            Cassette(name="x", base_url="http://localhost:1/v1", tps=1000.0)
        )
        # Even a slow measurement cannot pull the forced tier down.
        brain._record_tps(10, 1.0)
        self.assertEqual(brain.avg_tps(), 1000.0)
        self.assertEqual(brain.current_tier().name, "仙界")
        # status_line surfaces the tier + TPS + correction count.
        self.assertIn("仙界", brain.status_line())
        self.assertIn("1000", brain.status_line())

    def test_completion_token_fallback_when_usage_missing(self) -> None:
        from spl.agent.llm_client import Cassette, OpenAICompatibleBrain

        brain = OpenAICompatibleBrain(Cassette(name="x", base_url="http://localhost:1/v1"))
        # usage missing -> _record_tps is only fed by _chat's len//3 fallback;
        # here we feed it directly to confirm a sane sample is recorded.
        brain._record_tps(0, 0.5)  # zero tokens -> ignored
        self.assertEqual(len(brain._tps_samples), 0)
        brain._record_tps(60, 0.3)  # 200 TPS
        self.assertEqual(len(brain._tps_samples), 1)


class FumbleRuleTests(unittest.TestCase):
    """A valid action word the WORLD rejects fumbles (-1AP), never 混乱."""

    def test_valid_but_failing_action_fumbles_without_confusion(self) -> None:
        # At seed 42's start tile there is no rock nearby, so "mine" is a valid
        # action word that the world rejects -> a fumble, not confusion.
        sim = Simulation(seed=42, max_days=10)
        ap_before = sim.hero.ap_left
        sanity_before = sim.hero.sanity
        result = sim.step(GameAction(action="mine"), confuse_on_invalid=True)
        self.assertFalse(result.ok)
        self.assertEqual(sim.hero.ap_left, ap_before - 1)        # lost exactly 1 AP
        self.assertEqual(sim.hero.confusion_count, 0)            # no 混乱
        self.assertEqual(sim.hero.sanity, sanity_before)        # no sanity loss
        self.assertIn("[fumble -1AP]", sim.full_log[-1])

    def test_fumble_at_last_ap_ends_the_day(self) -> None:
        sim = Simulation(seed=42, max_days=10)
        sim.hero.ap_left = 1
        sim.hero.stamina = 50
        day_before = sim.world.day
        sim.step(GameAction(action="mine"), confuse_on_invalid=True)
        self.assertEqual(sim.hero.ap_left, sim.ap_per_day)       # day rolled over
        self.assertEqual(sim.world.day, day_before + 1)
        self.assertEqual(sim.hero.confusion_count, 0)

    def test_unknown_action_still_confuses(self) -> None:
        # The fumble rule must NOT swallow unknown words: those still confuse.
        sim = Simulation(seed=42, max_days=3)
        result = sim.step(GameAction(action="summon_dragon"))
        self.assertTrue(result.ok)
        self.assertEqual(sim.hero.confusion_count, 1)


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

    # -- camera: zoom / follow / pan -----------------------------------------
    def test_zoom_ladder_includes_close_read_rungs(self) -> None:
        from spl.ui.pixel import iso

        for rung in (2.0, 2.5, 3.0):
            self.assertIn(rung, iso.SPRITE_SCALES)
        # ladder above a fit floor never drops below the floor
        ladder = iso.zoom_ladder(1.0)
        self.assertEqual(ladder[0], 1.0)
        self.assertTrue(all(r >= 1.0 for r in ladder))

    def test_wheel_zoom_steps_ladder_and_tile_under_round_trips(self) -> None:
        from spl.ui.pixel import iso

        app = self._fhd_app()
        # zoom in past the fit floor with successive wheel-up steps at a cursor
        anchor = app.lay.map_rect.center
        before = app.lay.sprite_scale
        app._handle_wheel(+1, anchor)
        self.assertGreater(app.lay.sprite_scale, before)
        self.assertEqual(app.factory.scale, app.lay.sprite_scale)
        self.assertEqual(app.lay.sprite_scale, app.cam_scale)
        # picking is still exact at the new zoom: the hero tile round-trips from
        # its own screen centre using the LIVE camera offset/scale.
        hero = app.sim.hero.pos
        cx, cy = iso.tile_center(hero.x, hero.y, app.offset_x, app.offset_y,
                                 app.lay.sprite_scale)
        picked = app._tile_under(cx, cy)
        self.assertIsNotNone(picked)
        self.assertEqual((picked.x, picked.y), (hero.x, hero.y))

    def test_wheel_zoom_in_auto_arms_follow(self) -> None:
        app = self._fhd_app()
        self.assertFalse(app.follow)
        # only meaningful if there's headroom above the fit floor
        if app._at_fit_scale() and app.fit_scale_val >= max(__import__(
                "spl.ui.pixel.iso", fromlist=["SPRITE_SCALES"]).SPRITE_SCALES):
            self.skipTest("window already at max zoom")
        app._handle_wheel(+1, app.lay.map_rect.center)
        if not app._at_fit_scale():
            self.assertTrue(app.follow)

    def test_follow_centres_the_hermit_after_glide(self) -> None:
        app = self._fhd_app()
        # zoom in a couple of rungs and enable follow
        app._handle_wheel(+1, app.lay.map_rect.center)
        app._handle_wheel(+1, app.lay.map_rect.center)
        if app._at_fit_scale():
            self.skipTest("could not zoom past fit floor in this window")
        app.follow = True
        app._follow_armed = True
        # run enough update frames for the lerp to settle
        for _ in range(120):
            app._update(1 / 60.0, 0.0)
        # the hermit's screen centre should be within a few px of the band centre
        from spl.ui.pixel import iso

        hero = app.sim.hero.pos
        cx, cy = iso.tile_center(hero.x, hero.y, app.offset_x, app.offset_y,
                                 app.lay.sprite_scale)
        band = app.lay.map_rect
        self.assertLess(abs(cx - band.centerx), 8)
        # y target is intentionally below centre (bubble headroom); allow slack
        self.assertLess(abs(cy - (band.centery + int(round(28 * app.lay.sprite_scale)))), 8)

    def test_right_drag_pan_moves_offset_and_disables_follow(self) -> None:
        app = self._fhd_app()
        app._handle_wheel(+1, app.lay.map_rect.center)
        app._handle_wheel(+1, app.lay.map_rect.center)
        if app._at_fit_scale():
            self.skipTest("could not zoom past fit floor in this window")
        app.follow = True
        ox0, oy0 = app.offset_x, app.offset_y
        start = app.lay.map_rect.center
        app._begin_pan(start)
        self.assertTrue(app.panning)
        # drag right+down by a chunk
        app._handle_motion(start[0] + 80, start[1] + 40)
        app._end_pan()
        self.assertFalse(app.panning)
        self.assertFalse(app.follow, "pan must suspend follow")
        self.assertNotEqual((app.offset_x, app.offset_y), (ox0, oy0))

    def test_zoom_out_clamps_at_fit_scale(self) -> None:
        app = self._fhd_app()
        # spam zoom-out well past the floor; it must clamp at the fit scale and
        # reset to the centred-island view (follow/pan inert there).
        for _ in range(12):
            app._handle_wheel(-1, app.lay.map_rect.center)
        self.assertAlmostEqual(app.lay.sprite_scale, app.fit_scale_val, places=6)
        self.assertTrue(app._at_fit_scale())
        cx, cy = app._centered_offsets(app.lay.sprite_scale)
        self.assertEqual((app.offset_x, app.offset_y), (cx, cy))

    def test_pan_keeps_some_island_visible(self) -> None:
        app = self._fhd_app()
        for _ in range(4):
            app._handle_wheel(+1, app.lay.map_rect.center)
        if app._at_fit_scale():
            self.skipTest("could not zoom past fit floor in this window")
        app.follow = False
        # try to pan the island far off to one side; the clamp must keep ~25%
        # of the footprint in the band, so the hero tile stays projectable.
        for _ in range(60):
            app._begin_pan(app.lay.map_rect.center)
            app._handle_motion(app.lay.map_rect.right + 400, app.lay.map_rect.centery)
            app._end_pan()
        from spl.ui.pixel import iso

        world = app.sim.world
        map_w, _ = iso.map_pixel_size(world.width, world.height, app.lay.sprite_scale)
        min_sx = (0 - (world.height - 1)) * iso.half_w(app.lay.sprite_scale)
        left = app.offset_x + min_sx
        # at least a quarter of the island's width remains within the band
        visible = min(app.lay.map_rect.right, left + map_w) - max(app.lay.map_rect.left, left)
        self.assertGreaterEqual(visible, int(map_w * 0.2))

    def test_terrain_cache_survives_pan_without_rebuild(self) -> None:
        # Panning must NOT rebuild terrain (it's baked at origin, blitted at the
        # camera offset). The signature excludes the offset, so the cached slab
        # object is reused across a pan.
        app = self._fhd_app()
        for _ in range(3):
            app._handle_wheel(+1, app.lay.map_rect.center)
        if app._at_fit_scale():
            self.skipTest("could not zoom past fit floor in this window")
        slab = app._ensure_terrain()
        app._begin_pan(app.lay.map_rect.center)
        app._handle_motion(app.lay.map_rect.centerx + 50, app.lay.map_rect.centery + 30)
        app._end_pan()
        self.assertIs(app._ensure_terrain(), slab, "pan should not rebuild terrain")


class SakusenTests(unittest.TestCase):
    """作戦 (standing order) + 承認制/最速 speed ladder — the F1 pit-wall layer."""

    # -- core seam: set_strategy / strategy_changes / observation --------------
    def test_set_strategy_persists_and_counts_only_real_changes(self) -> None:
        sim = Simulation(seed=42, max_days=12)
        self.assertEqual(sim.strategy_changes, 0)
        self.assertIsNone(sim.advice_from_heaven)

        sim.set_strategy("水と食を最優先")
        self.assertEqual(sim.advice_from_heaven, "水と食を最優先")
        self.assertEqual(sim.strategy_changes, 1)

        # re-sending the same order does not inflate the counter
        sim.set_strategy("水と食を最優先")
        self.assertEqual(sim.strategy_changes, 1)

        # a genuinely different order counts
        sim.set_strategy("井戸を掘れ")
        self.assertEqual(sim.strategy_changes, 2)

        # clearing does not count, and empty/whitespace clears
        sim.set_strategy(None)
        self.assertIsNone(sim.advice_from_heaven)
        self.assertEqual(sim.strategy_changes, 2)
        sim.set_strategy("   ")
        self.assertIsNone(sim.advice_from_heaven)
        self.assertEqual(sim.strategy_changes, 2)

    def test_strategy_persists_across_the_day_night_cycle(self) -> None:
        # The order must survive a full day/night — nothing clears it.
        sim = Simulation(seed=42, max_days=12)
        sim.set_strategy("毎日まず水を確保せよ")
        agent = LocalPolicyAgent()
        start = sim.world.day
        for _ in range(400):
            if sim.world.day != start or sim.done:
                break
            sim.step(agent.choose(sim))
        self.assertNotEqual(sim.world.day, start, "did not cross a day boundary")
        self.assertEqual(sim.advice_from_heaven, "毎日まず水を確保せよ")

    def test_observation_carries_strategy_from_heaven_near_top(self) -> None:
        from spl.agent.observer import ObservationBuilder

        sim = Simulation(seed=42, max_days=12)
        sim.set_strategy("井戸と保存樽を最優先")
        obs = ObservationBuilder().build(sim)
        self.assertEqual(obs["strategy_from_heaven"], "井戸と保存樽を最優先")
        # the old key must be gone; the renamed key must be near the top
        self.assertNotIn("advice_from_heaven", obs)
        keys = list(obs.keys())
        self.assertLessEqual(keys.index("strategy_from_heaven"), 1,
                             f"strategy_from_heaven not near top: {keys}")

    def test_system_prompt_mentions_strategy_from_heaven(self) -> None:
        from spl.agent.prompts import SYSTEM_PROMPT

        self.assertIn("strategy_from_heaven", SYSTEM_PROMPT)
        self.assertIn("TRUE WORLD STATE", SYSTEM_PROMPT)

    def test_cli_strategy_flag_wires_into_the_sim(self) -> None:
        from types import SimpleNamespace

        from spl.ui import cli

        # run_simulate with --strategy must seed the standing order before play.
        args = SimpleNamespace(
            seed=42, days=8, llm=False, cassette=None, radius=7,
            strategy="水を切らすな", tps=0,
        )
        captured: dict[str, object] = {}
        orig = cli.print_result

        def _spy(sim, motto=None, **kw):  # noqa: ANN001
            captured["advice"] = sim.advice_from_heaven
            captured["changes"] = sim.strategy_changes

        cli.print_result = _spy
        try:
            cli.run_simulate(args)
        finally:
            cli.print_result = orig
        self.assertEqual(captured["advice"], "水を切らすな")
        self.assertEqual(captured["changes"], 1)

    # -- pixel: 承認制 / 最速 ----------------------------------------------------
    def _watch_app(self, speed: int = 3, strategy=None, days: int = 12):
        from types import SimpleNamespace

        from spl.ui.pixel.app import PixelApp

        args = SimpleNamespace(
            seed=42, days=days, llm=False, cassette="x", manual=False, speed=speed,
            scale=2, start_day=0, shots=0, shots_ui=False, shot_dir="/tmp/spl_test",
            strategy=strategy,
        )
        return PixelApp(args, headless=True)

    def test_strategy_flag_seeds_pixel_app(self) -> None:
        app = self._watch_app(strategy="井戸を最優先")
        self.assertEqual(app.sim.advice_from_heaven, "井戸を最優先")
        self.assertEqual(app.sim.strategy_changes, 1)

    def _run_watch(self, app, frames: int) -> bool:
        """Drive the watch loop with a steadily advancing fake clock (so the
        per-speed delay gate keeps opening). Stops when approval latches."""
        for i in range(frames):
            app._watch_step(app.anim_t + i * 0.5)
            if app.approval_pause or app.sim.done:
                return app.approval_pause
        return app.approval_pause

    def test_approval_mode_pauses_at_day_boundary_and_next_advances_one_day(self) -> None:
        app = self._watch_app(speed=1)  # 承認
        self.assertFalse(app.approval_pause)
        start_day = app.sim.world.day
        # step the watch loop until the boundary latches the pause
        self.assertTrue(self._run_watch(app, 3000), "承認 mode never paused at a boundary")
        paused_day = app.sim.world.day
        self.assertEqual(paused_day, start_day + 1)

        # while paused the sim must not advance — _update holds at the boundary
        for i in range(50):
            app._update(1 / 60.0, app.anim_t + (3000 + i) * 0.5)
        self.assertEqual(app.sim.world.day, paused_day)
        self.assertTrue(app.approval_pause)

        # [次の日へ] resumes; run until it pauses again — exactly one more day
        app._resume_next_day()
        self.assertFalse(app.approval_pause)
        self.assertTrue(self._run_watch(app, 3000))
        self.assertEqual(app.sim.world.day, paused_day + 1)

    def test_fastest_reaches_done_without_exceptions_and_faster_than_normal(self) -> None:
        # 最速 (speed 5): zero delay, multiple steps/frame -> reaches sim.done.
        # The fake clock barely advances, proving the burst loop (not the clock)
        # drives progress at 最速.
        fast = self._watch_app(speed=5, days=12)
        frames_fast = 0
        for _ in range(50000):
            fast._watch_step(fast.anim_t + frames_fast * 1e-6)
            frames_fast += 1
            if fast.sim.done:
                break
        self.assertTrue(fast.sim.done, "最速 did not reach sim.done")

        # 普 (speed 3) takes exactly one step per opened delay window, so it needs
        # far more frames to cover the same span — the fair, deterministic measure.
        norm = self._watch_app(speed=3, days=12)
        frames_norm = 0
        for _ in range(50000):
            norm._watch_step(norm.anim_t + frames_norm * 0.5)
            frames_norm += 1
            if norm.sim.done:
                break
        self.assertTrue(norm.sim.done)
        self.assertLess(frames_fast, frames_norm,
                        f"最速 should need fewer frames ({frames_fast}) than 普 ({frames_norm})")

    def test_result_panel_renders_with_strategy_stats(self) -> None:
        app = self._watch_app(speed=5, strategy="井戸と保存樽を最優先", days=12)
        for i in range(50000):
            app._watch_step(app.anim_t + i * 1e-6)
            if app.sim.done:
                break
        self.assertTrue(app.sim.done)
        # draw the result panel; strategy stats must render without raising
        win = app.pg.Surface((app.lay.win_w, app.lay.win_h))
        app.overlays.draw_result(win, app.sim, app.lay)
        self.assertGreaterEqual(app.sim.strategy_changes, 1)

    def test_strategy_overlay_clear_and_send_route_through_set_strategy(self) -> None:
        app = self._watch_app(speed=3)
        # open the 作戦 overlay and render it so the hit rects exist
        app.overlay = "heaven"
        app.heaven_text = "水を最優先"
        win = app.pg.Surface((app.lay.win_w, app.lay.win_h))
        app.render(win)
        send = app._hits.get("heaven_send")
        self.assertIsNotNone(send)
        app._click_overlay(send.centerx, send.centery)
        self.assertEqual(app.sim.advice_from_heaven, "水を最優先")
        self.assertEqual(app.sim.strategy_changes, 1)

        # now clear it via [作戦解除]
        app.overlay = "heaven"
        app.render(win)
        clear = app._hits.get("heaven_clear")
        self.assertIsNotNone(clear)
        app._click_overlay(clear.centerx, clear.centery)
        self.assertIsNone(app.sim.advice_from_heaven)
        # clearing does not inflate the change counter
        self.assertEqual(app.sim.strategy_changes, 1)

    def test_pitwall_strip_renders_and_next_button_resumes(self) -> None:
        app = self._watch_app(speed=1)
        self.assertTrue(self._run_watch(app, 3000))
        win = app.pg.Surface((app.lay.win_w, app.lay.win_h))
        app.render(win)  # draws the pit-wall strip, populating hit rects
        hits = app._hits.get("pitwall")
        self.assertTrue(hits and "next" in hits)
        nxt = hits["next"]
        app._click_pitwall(nxt.centerx, nxt.centery)
        self.assertFalse(app.approval_pause, "[次の日へ] did not resume")


@unittest.skipUnless(_HAS_PYGAME, "pygame not installed")
class PitwallRegressionTests(unittest.TestCase):
    """Bugs hit live by the player on 2026-06-11."""

    def _app(self):
        from types import SimpleNamespace

        from spl.ui.pixel.app import PixelApp

        args = SimpleNamespace(seed=42, days=112, llm=False, cassette="x", manual=False,
                               speed=1, scale=2, start_day=0, shots=0, shots_ui=False,
                               shot_dir="/tmp/spl_test", strategy=None, tps=0.0)
        return PixelApp(args, headless=True)

    def test_changing_speed_during_approval_pause_resumes(self) -> None:
        app = self._app()
        app.approval_pause = True
        app._refresh_buttons()
        speed_btn = next(b for b in app.buttons if b.key == "speed")
        app._handle_click(speed_btn.rect.centerx, speed_btn.rect.centery)
        self.assertNotEqual(app.speed, 1, "speed button must stay clickable at the pit wall")
        self.assertFalse(app.approval_pause, "leaving 承認 must lift the boundary pause")

    def test_textinput_appends_to_strategy_overlay(self) -> None:
        app = self._app()
        app.overlay = "heaven"
        app.heaven_text = ""
        # simulate what the TEXTINPUT branch does with IME-composed Japanese
        for piece in ("井戸を", "掘れ"):
            app.heaven_text = (app.heaven_text + piece)[:60]
        self.assertEqual(app.heaven_text, "井戸を掘れ")


class ConditionGateTests(unittest.TestCase):
    """内省は満腹の上に立つ: the body caps the effective thinking tier."""

    def _hero(self, **kw):
        from types import SimpleNamespace

        base = dict(hp=100, hunger=80, water=80, stamina=90, sanity=85)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_nourished_hermit_is_uncapped(self) -> None:
        from spl.agent.llm_client import _TIERS, condition_cap_index

        idx, reason = condition_cap_index(self._hero())
        self.assertEqual(idx, len(_TIERS) - 1)
        self.assertIsNone(reason)

    def test_starving_hermit_drops_to_reflex(self) -> None:
        from spl.agent.llm_client import condition_cap_index

        self.assertEqual(condition_cap_index(self._hero(hunger=0))[0], 0)
        self.assertEqual(condition_cap_index(self._hero(water=5))[0], 0)
        self.assertEqual(condition_cap_index(self._hero(sanity=15))[0], 0)

    def test_weary_hermit_loses_introspection(self) -> None:
        from spl.agent.llm_client import _TIERS, condition_cap_index

        idx, reason = condition_cap_index(self._hero(hunger=20))
        self.assertEqual(idx, 1)
        self.assertFalse(_TIERS[idx].verify, "a weary hermit cannot introspect")
        self.assertEqual(reason, "疲弊")

    def test_new_strategy_rings_once_in_the_log(self) -> None:
        from spl.core.sim import Simulation

        sim = Simulation(seed=11, max_days=10)
        sim.set_strategy("井戸を掘れ")
        sim.set_strategy("井戸を掘れ")  # same text: no second revelation
        rings = [ln for ln in sim.full_log if "Heaven speaks" in ln]
        self.assertEqual(len(rings), 1)


class BoukenNoShoTests(unittest.TestCase):
    """ぼうけんのしょ — the cross-life lesson journal."""

    def _book(self, tmpdir: str):
        from spl.agent.bouken import BoukenNoSho

        return BoukenNoSho.load(f"{tmpdir}/bouken_test.json")

    def test_round_trip_append_reload_lives_and_entries(self) -> None:
        import tempfile

        from spl.agent.bouken import BoukenNoSho

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/bouken_test.json"
            book = BoukenNoSho.load(path)
            self.assertEqual(book.lives, 0)
            book.append({"seed": 42, "days": 5, "score": 100,
                         "ending": "果てた", "lessons": ["水を掘れ"], "motto": "x"})
            book.append({"seed": 42, "days": 9, "score": 200,
                         "ending": "果てた", "lessons": ["火を建てよ"], "motto": "y"})
            reloaded = BoukenNoSho.load(path)
            self.assertEqual(reloaded.lives, 2)
            self.assertEqual(reloaded.entries[0]["life"], 1)
            self.assertEqual(reloaded.entries[1]["life"], 2)
            self.assertEqual(reloaded.entries[1]["days"], 9)

    def test_lessons_for_prefers_same_seed_and_dedupes(self) -> None:
        import tempfile

        from spl.agent.bouken import BoukenNoSho

        with tempfile.TemporaryDirectory() as tmp:
            book = BoukenNoSho.load(f"{tmp}/bouken_test.json")
            book.append({"seed": 99, "lessons": ["別の島の教訓"]})
            book.append({"seed": 42, "lessons": ["水を掘れ", "火を建てよ"]})
            book.append({"seed": 42, "lessons": ["火を建てよ", "魚を焼け"]})  # dup "火を建てよ"
            lessons = book.lessons_for(42, limit=6)
            # same-seed lessons come first, most-recent-first, deduped
            self.assertEqual(lessons[:3], ["火を建てよ", "魚を焼け", "水を掘れ"])
            # the other-island lesson appears only after the same-seed ones
            self.assertIn("別の島の教訓", lessons)
            self.assertGreater(lessons.index("別の島の教訓"), 0)
            self.assertEqual(len(lessons), len(set(lessons)))

    def test_fallback_motto_returns_three_lessons_on_each_branch(self) -> None:
        from spl.arena.leaderboard import fallback_motto

        for seed, days in ((42, 112), (0, 8), (3, 40)):
            sim = run_local(seed=seed, days=days)
            motto = fallback_motto(sim)
            self.assertIn("lessons", motto)
            self.assertEqual(len(motto["lessons"]), 3,
                             f"seed {seed} did not yield 3 lessons: {motto['lessons']}")
            for lesson in motto["lessons"]:
                self.assertTrue(lesson.strip())

    def test_observation_carries_bouken_no_sho_near_top_when_set(self) -> None:
        from spl.agent.observer import ObservationBuilder

        sim = Simulation(seed=42, max_days=12)
        builder = ObservationBuilder()
        builder.book_lessons = ["水を掘れ", "火を建てよ", "木の実を拾え"]
        builder.book_lives = 2
        obs = builder.build(sim)
        self.assertIn("bouken_no_sho", obs)
        self.assertEqual(obs["bouken_no_sho"]["lives"], 2)
        self.assertEqual(obs["bouken_no_sho"]["lessons"][0], "水を掘れ")
        keys = list(obs.keys())
        # right after strategy_from_heaven (index 0)
        self.assertEqual(keys.index("bouken_no_sho"), 1, f"not near top: {keys}")

    def test_observation_omits_bouken_no_sho_when_empty(self) -> None:
        from spl.agent.observer import ObservationBuilder

        sim = Simulation(seed=42, max_days=12)
        obs = ObservationBuilder().build(sim)
        self.assertNotIn("bouken_no_sho", obs)

    def test_motto_schema_requires_three_lessons(self) -> None:
        from spl.agent.llm_client import _motto_schema

        schema = _motto_schema()["schema"]
        self.assertIn("lessons", schema["required"])
        lessons = schema["properties"]["lessons"]
        self.assertEqual(lessons["minItems"], 3)
        self.assertEqual(lessons["maxItems"], 3)
        self.assertEqual(lessons["items"]["maxLength"], 80)

    def test_system_prompt_mentions_bouken_no_sho(self) -> None:
        from spl.agent.prompts import SYSTEM_PROMPT

        self.assertIn("bouken_no_sho", SYSTEM_PROMPT)
        self.assertIn("PAST SELVES", SYSTEM_PROMPT)

    def test_pixel_book_records_run_and_replay_reinjects(self) -> None:
        """The pixel app with --book records the ended run once, and [もう一度]
        (next life) reloads the book so the new run inherits the lessons."""
        import os
        import tempfile
        from types import SimpleNamespace

        from spl.ui.pixel.app import PixelApp

        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("SPL_BOOK_DIR")
            os.environ["SPL_BOOK_DIR"] = tmp
            try:
                args = SimpleNamespace(
                    seed=0, days=5, llm=False, cassette="PxTest", manual=False,
                    speed=5, scale=2, start_day=0, shots=0, shots_ui=False,
                    shot_dir="/tmp/spl_test", strategy=None, book=True,
                )
                app = PixelApp(args, headless=True)
                self.assertIsNotNone(app.book)
                self.assertEqual(app.book.lives, 0)
                # drive the local burst loop to completion
                for _ in range(50000):
                    app._watch_step(app.anim_t + 1e-6)
                    if app.sim.done:
                        break
                self.assertTrue(app.sim.done)
                # the result-draw path resolves the motto then writes the book once
                self.assertFalse(app._book_written)
                app._start_motto()       # local brain resolves the motto synchronously
                app._maybe_write_book()
                self.assertTrue(app._book_written)
                self.assertEqual(app.book.lives, 1)
                first_entry = app._book_entry
                self.assertEqual(len(first_entry["lessons"]), 3)

                # [もう一度] = next life: reload + re-inject + reset the guard
                app._rebuild_sim()
                self.assertFalse(app._book_written)
                self.assertEqual(app.book.lives, 1)  # the first life is on file
            finally:
                if old is None:
                    os.environ.pop("SPL_BOOK_DIR", None)
                else:
                    os.environ["SPL_BOOK_DIR"] = old

    def test_cli_book_flag_accumulates_lessons_across_lives(self) -> None:
        """A local --book simulate, run twice on the same seed+cassette, must see
        the second life inherit the first's lessons (lives=1, lessons injected)."""
        import os
        import tempfile
        from types import SimpleNamespace

        from spl.agent.bouken import book_path_for
        from spl.ui import cli

        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("SPL_BOOK_DIR")
            os.environ["SPL_BOOK_DIR"] = tmp
            captured: list[dict] = []
            orig = cli.print_result

            def _spy(sim, motto=None, brain=None, book=None, book_entry=None):
                captured.append({"book": book, "entry": book_entry})

            cli.print_result = _spy
            try:
                args = SimpleNamespace(seed=0, days=5, llm=False, cassette="TestCass",
                                       strategy=None, tps=0, book=True)
                # First life: writes life #1 with 3 fallback lessons.
                cli.run_simulate(args)
                self.assertEqual(captured[0]["entry"]["life"], 1)
                self.assertEqual(len(captured[0]["entry"]["lessons"]), 3)
                # The journal file now exists with one entry.
                path = book_path_for("TestCass")
                self.assertTrue(path.exists())
                # Second life: the book reloaded must report lives=1 going in.
                cli.run_simulate(args)
                self.assertEqual(captured[1]["entry"]["life"], 2)
                # The lessons_for the seed must surface the first life's lessons.
                from spl.agent.bouken import BoukenNoSho

                book = BoukenNoSho.load(path)
                self.assertEqual(book.lives, 2)
                self.assertTrue(book.lessons_for(0))
            finally:
                cli.print_result = orig
                if old is None:
                    os.environ.pop("SPL_BOOK_DIR", None)
                else:
                    os.environ["SPL_BOOK_DIR"] = old


class KakunTests(unittest.TestCase):
    """家訓の編纂 — the fixed 5-article canon, revised not grown."""

    def test_canon_round_trip_and_lessons_for_prefers_canon(self) -> None:
        import tempfile

        from spl.agent.bouken import BoukenNoSho

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/bouken_test.json"
            book = BoukenNoSho.load(path)
            self.assertEqual(book.canon, [])
            self.assertEqual(book.canon_revision, 0)
            book.append({"seed": 42, "days": 5, "lessons": ["古い教訓"]})
            # set a canon and reload — it round-trips with its revision
            book.set_canon(["第一条 水を掘れ", "第二条 火を建てよ"], 3)
            reloaded = BoukenNoSho.load(path)
            self.assertEqual(reloaded.canon, ["第一条 水を掘れ", "第二条 火を建てよ"])
            self.assertEqual(reloaded.canon_revision, 3)
            # canon WINS in lessons_for (the recent-lessons path is bypassed)
            self.assertEqual(reloaded.lessons_for(42), ["第一条 水を掘れ", "第二条 火を建てよ"])
            self.assertEqual(reloaded.lessons_for(999), ["第一条 水を掘れ", "第二条 火を建てよ"])

    def test_lessons_for_falls_back_to_recent_when_no_canon(self) -> None:
        # back-compat: no canon -> the original recent/same-seed behaviour.
        import tempfile

        from spl.agent.bouken import BoukenNoSho

        with tempfile.TemporaryDirectory() as tmp:
            book = BoukenNoSho.load(f"{tmp}/bouken_test.json")
            book.append({"seed": 42, "lessons": ["水を掘れ"]})
            self.assertEqual(book.canon, [])
            self.assertEqual(book.lessons_for(42), ["水を掘れ"])

    def test_canon_capped_at_five(self) -> None:
        import tempfile

        from spl.agent.bouken import BoukenNoSho

        with tempfile.TemporaryDirectory() as tmp:
            book = BoukenNoSho.load(f"{tmp}/bouken_test.json")
            book.set_canon([f"条文{i}" for i in range(8)], 1)
            self.assertEqual(len(book.canon), 5)

    def test_history_table_pairs_lifespan_with_lessons(self) -> None:
        import tempfile

        from spl.agent.bouken import BoukenNoSho

        with tempfile.TemporaryDirectory() as tmp:
            book = BoukenNoSho.load(f"{tmp}/bouken_test.json")
            book.append({"seed": 0, "days": 3, "ending": "渇き", "lessons": ["水"]})
            book.append({"seed": 0, "days": 40, "ending": "生存", "lessons": ["火", "実"]})
            table = book.history_table()
            self.assertEqual(len(table), 2)
            # 神のレバー: history_table now carries the共同 marker miracles_used
            # (0 for an unassisted life) so the 編纂者 can discount assisted lives.
            self.assertEqual(
                table[0],
                {"life": 1, "days": 3, "ending": "渇き", "lessons": ["水"], "miracles_used": 0},
            )
            self.assertEqual(table[1]["days"], 40)
            self.assertEqual(table[1]["lessons"], ["火", "実"])

    def test_fallback_compile_dedupes_water_and_prefers_long_lives(self) -> None:
        import tempfile

        from spl.agent.bouken import BoukenNoSho, fallback_compile

        with tempfile.TemporaryDirectory() as tmp:
            book = BoukenNoSho.load(f"{tmp}/bouken_test.json")
            # four water-themed near-duplicates across short lives + distinct
            # articles from a long life that must be kept.
            book.append({"seed": 1, "days": 4, "lessons": ["三日目までに水を確保せよ。"]})
            book.append({"seed": 1, "days": 5, "lessons": ["二日目までに水を確保せよ"]})
            book.append({"seed": 1, "days": 6, "lessons": ["まずは水を確保せよ動く前に"]})
            book.append({"seed": 1, "days": 7, "lessons": ["朝いちで水を確保せよ毎日"]})
            book.append({"seed": 1, "days": 80, "lessons": ["秋までに保存樽を建てよ", "焚き火を絶やすな"]})
            canon = fallback_compile(book)
            # near-duplicate water articles collapse to exactly one
            water = [a for a in canon if "水" in a]
            self.assertEqual(len(water), 1, f"water not merged: {canon}")
            # the long life's distinct articles survived
            self.assertTrue(any("保存樽" in a for a in canon))
            self.assertTrue(any("焚き火" in a for a in canon))
            # never more than 5
            self.assertLessEqual(len(canon), 5)

    def test_fallback_compile_caps_at_five(self) -> None:
        import tempfile

        from spl.agent.bouken import BoukenNoSho, fallback_compile

        with tempfile.TemporaryDirectory() as tmp:
            book = BoukenNoSho.load(f"{tmp}/bouken_test.json")
            distinct = [
                "水を切らすな掘削を急げ",
                "森で薪を集め火種とせよ",
                "畑を耕し蕪の種を蒔け",
                "岩場で鉱石を掘り斧を鍛えよ",
                "保存樽を建て冬に備えよ",
                "釣り竿を作り魚を獲れ",
                "住処を改修し寒気を防げ",
            ]
            for i, lesson in enumerate(distinct):
                book.append({"seed": 1, "days": i + 1, "lessons": [lesson]})
            self.assertEqual(len(fallback_compile(book)), 5)

    def test_compile_schema_requires_exactly_five(self) -> None:
        from spl.agent.llm_client import _compile_schema

        schema = _compile_schema()["schema"]
        self.assertIn("lessons", schema["required"])
        lessons = schema["properties"]["lessons"]
        self.assertEqual(lessons["minItems"], 5)
        self.assertEqual(lessons["maxItems"], 5)
        self.assertEqual(lessons["items"]["maxLength"], 80)

    def test_compile_prompt_exists_and_demands_five_jp_articles(self) -> None:
        from spl.agent.prompts import COMPILE_PROMPT

        self.assertIn("編纂者", COMPILE_PROMPT)
        self.assertIn("EXACTLY 5", COMPILE_PROMPT)
        self.assertIn("JAPANESE", COMPILE_PROMPT)

    def test_evolve_loop_two_lives_no_llm_writes_canon_rev_two(self) -> None:
        import io
        import os
        import tempfile
        from contextlib import redirect_stdout

        from spl.agent.bouken import BoukenNoSho, book_path_for
        from spl.arena.evolve import run_evolve

        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("SPL_BOOK_DIR")
            os.environ["SPL_BOOK_DIR"] = tmp
            try:
                from spl.agent.llm_client import Cassette

                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = run_evolve(
                        lives=2, seed=0, days=5,
                        cassette=Cassette(name="EvoTest", base_url=""),
                        use_llm=False, book_dir_cassette="EvoTest",
                    )
                self.assertEqual(rc, 0)
                out = buf.getvalue()
                self.assertIn("life 1", out)
                self.assertIn("life 2", out)
                book = BoukenNoSho.load(book_path_for("EvoTest"))
                self.assertEqual(book.lives, 2)
                self.assertEqual(book.canon_revision, 2)
                self.assertTrue(book.canon, "canon should be non-empty after evolve")
                self.assertLessEqual(len(book.canon), 5)
            finally:
                if old is None:
                    os.environ.pop("SPL_BOOK_DIR", None)
                else:
                    os.environ["SPL_BOOK_DIR"] = old


class _FakeSeat:
    """A stand-in for OpenAICompatibleBrain: scripted action proposals and a
    scripted think string, with the 思考予算 / health surface the council uses."""

    def __init__(self, action: GameAction, think: str = "計画", alive: bool = True,
                 persona: str = "p"):
        from spl.agent.llm_client import Cassette, tier_for_tps

        self._action = action
        self._think = think
        self._alive = alive
        self.cassette = Cassette(name="seat", base_url="http://x/v1", persona=persona)
        self._tier_for_tps = tier_for_tps
        self.propose_calls = 0
        self.choose_calls = 0
        self.think_calls = 0
        self.chat_calls = 0
        self.last_propose_extra = None       # the `extra` messages of the last action call
        self.last_choose_extra = None        # the `extra` messages of the last choose() call
        self.think_systems: list[str] = []   # system prompts seen by think_freetext

    def propose_action(self, obs, budget=None, extra=None):
        self.propose_calls += 1
        self.last_propose_extra = extra
        if not self._alive:
            raise RuntimeError("seat down")
        return self._action

    def choose(self, sim, extra=None):
        # v2.1 pilot path: MELCHIOR flies through its full choose() machinery
        # (here scripted), the day's 評定 carried in `extra`.
        self.choose_calls += 1
        self.last_choose_extra = extra
        if not self._alive:
            raise RuntimeError("seat down")
        return self._action

    def think_freetext(self, system, user, max_tokens=128):
        self.think_calls += 1
        self.think_systems.append(system)
        if not self._alive:
            raise RuntimeError("seat down")
        return self._think

    def _resolve_model(self):
        if not self._alive:
            raise RuntimeError("down")
        return "fake"

    def current_tier(self):
        return self._tier_for_tps(80.0)

    def avg_tps(self):
        return 80.0


class MagiCouncilTests(unittest.TestCase):
    def _sim(self):
        return Simulation(seed=42, max_days=112)

    def _council(self, mel, bal, casp=None):
        # These tests exercise the v1 per-turn deliberation → committee mode.
        from spl.agent.magi import MagiBrain

        return MagiBrain(mel, bal, casp, mode="committee")

    def test_majority_shortcut_picks_the_2_1_winner(self) -> None:
        gather = GameAction(action="gather", args={}, say="採取")
        rest = GameAction(action="rest", args={}, say="休む")
        # MELCHIOR=gather, BALTHASAR=gather, CASPER=rest → 2-1 majority gather.
        mel = _FakeSeat(gather)
        bal = _FakeSeat(gather)
        casp = _FakeSeat(rest)
        # MELCHIOR's FINAL action call also returns gather (confirms the winner).
        council = self._council(mel, bal, casp)
        action = council.choose(self._sim())
        self.assertEqual(action.action, "gather")
        rec = council.turn_records[-1]
        self.assertTrue(rec["agreed"])
        self.assertFalse(rec["moderator_used"])  # majority skips the 司会 THINK
        # The 司会 (BALTHASAR/Gemma) think_freetext must NOT have been used as moderator
        # (only CASPER's MELCHIOR-think may have run).
        self.assertEqual(bal.think_calls, 0)

    def test_moderator_path_on_three_way_split(self) -> None:
        a = GameAction(action="gather", args={}, say="a")
        b = GameAction(action="rest", args={}, say="b")
        c = GameAction(action="drink", args={}, say="c")
        mel = _FakeSeat(a)
        bal = _FakeSeat(b, think="bを採れ")
        casp = _FakeSeat(c)
        council = self._council(mel, bal, casp)
        council.choose(self._sim())
        rec = council.turn_records[-1]
        self.assertFalse(rec["agreed"])
        self.assertTrue(rec["moderator_used"])
        self.assertEqual(council.moderator_used, 1)
        # The 司会 ruling came from the Gemma (BALTHASAR) seat's think.
        self.assertGreaterEqual(bal.think_calls, 1)

    def test_unanimous_counts_and_status_line(self) -> None:
        g = GameAction(action="gather", args={}, say="g")
        council = self._council(_FakeSeat(g), _FakeSeat(g), _FakeSeat(g))
        council.choose(self._sim())
        rec = council.turn_records[-1]
        self.assertTrue(rec["unanimous"])
        self.assertEqual(council.unanimous, 1)
        self.assertIn("全会一致1", council.status_line())

    def test_degraded_mode_falls_back_to_melchior(self) -> None:
        g = GameAction(action="gather", args={}, say="g")
        mel = _FakeSeat(g)
        bal = _FakeSeat(g, alive=False)  # Gemma down
        sim = self._sim()
        council = self._council(mel, bal, _FakeSeat(g))
        action = council.choose(sim)
        self.assertEqual(action.action, "gather")
        rec = council.turn_records[-1]
        self.assertTrue(rec.get("degraded"))
        # BALTHASAR/CASPER were never asked for a proposal in degraded mode.
        self.assertEqual(bal.propose_calls, 0)

    def test_degraded_when_balthasar_is_none(self) -> None:
        g = GameAction(action="rest", args={}, say="g")
        council = self._council(_FakeSeat(g), None, None)
        action = council.choose(self._sim())
        self.assertEqual(action.action, "rest")
        self.assertTrue(council.turn_records[-1].get("degraded"))

    def test_compile_council_adopts_at_most_five_articles(self) -> None:
        # A fake book + seats whose _chat/think emit a 5-article canon.
        from spl.agent.magi import MagiBrain

        class _FakeBook:
            canon = ["古き条文"]

            def history_table(self):
                return [{"life": 1, "days": 90, "lessons": ["水を汲め"]}]

        class _CompileMel(_FakeSeat):
            def compile_canon(self, book):
                return ["第一", "第二", "第三", "第四", "第五"]

            def _chat(self, messages, schema=None, max_tokens=None):
                self.chat_calls += 1
                import json

                return json.dumps({"lessons": ["改一", "改二", "改三", "改四", "改五"]},
                                  ensure_ascii=False)

        mel = _CompileMel(GameAction(action="rest", args={}))
        bal = _FakeSeat(GameAction(action="rest", args={}), think="第三条は毒、棄却")
        council = MagiBrain(mel, bal, None)
        articles = council.compile_canon(_FakeBook())
        self.assertIsNotNone(articles)
        self.assertLessEqual(len(articles), 5)
        self.assertEqual(articles, ["改一", "改二", "改三", "改四", "改五"])
        # BALTHASAR reviewed (the mother's veto ran).
        self.assertGreaterEqual(bal.think_calls, 1)
        self.assertEqual(council.last_compile_review, "第三条は毒、棄却")

    def test_poison_article_detector(self) -> None:
        from spl.agent.magi import _is_poison_article

        self.assertTrue(_is_poison_article("無常の理を深く知る、空腹は忍ぶも、火の温もりを忘れず。"))
        self.assertTrue(_is_poison_article("飢えを忍べ、春の恵みを待て。"))
        # 凌ぐ / 癒す / 防ぐ are ACTIONS, not poison.
        self.assertFalse(_is_poison_article("七日目までに根菜を食え、飢えを凌げ。"))
        self.assertFalse(_is_poison_article("五日目までに井戸を深く掘れ、渇きを防げ。"))

    def test_compile_council_strips_poison_article_from_draft(self) -> None:
        from spl.agent.magi import MagiBrain

        poison = "無常の理を深く知る、空腹は忍ぶも、火の温もりを忘れず。"

        class _FakeBook:
            canon = [poison]

            def history_table(self):
                return []

        class _CompileMel(_FakeSeat):
            def compile_canon(self, book):
                # draft still carries the poison article
                return ["井戸を掘れ", "根菜を食え", "木の実を拾え", "種を蒔け", poison]

            def _chat(self, messages, schema=None, max_tokens=None):
                import json

                # the emitter echoes whatever 'canon' it was handed (the clean draft)
                canon = json.loads(messages[-1]["content"]).get("canon", [])
                # pad to 5 if short
                out = (canon + ["火を絶やすな"])[:5]
                return json.dumps({"lessons": out}, ensure_ascii=False)

        mel = _CompileMel(GameAction(action="rest", args={}))
        bal = _FakeSeat(GameAction(action="rest", args={}), think="空腹を忍ぶ条は毒、棄却")
        council = MagiBrain(mel, bal, None)
        articles = council.compile_canon(_FakeBook())
        self.assertLessEqual(len(articles), 5)
        self.assertNotIn(poison, articles)
        self.assertIn(poison, council.last_poison_stripped)

    def test_compile_council_degrades_to_draft_when_gemma_down(self) -> None:
        from spl.agent.magi import MagiBrain

        class _FakeBook:
            canon = []

            def history_table(self):
                return []

        class _CompileMel(_FakeSeat):
            def compile_canon(self, book):
                return ["素案一", "素案二", "素案三", "素案四", "素案五"]

        mel = _CompileMel(GameAction(action="rest", args={}))
        bal = _FakeSeat(GameAction(action="rest", args={}), alive=False)
        council = MagiBrain(mel, bal, None)
        articles = council.compile_canon(_FakeBook())
        self.assertEqual(articles, ["素案一", "素案二", "素案三", "素案四", "素案五"])


class MagiPilotModeTests(unittest.TestCase):
    """v2 操縦 (pilot) mode — 評議と操縦の分離."""

    def _sim(self):
        return Simulation(seed=42, max_days=112)

    def _pilot(self, mel, bal, casp=None):
        from spl.agent.magi import MagiBrain

        return MagiBrain(mel, bal, casp, mode="pilot")

    def test_default_mode_is_pilot(self) -> None:
        from spl.agent.magi import MagiBrain

        b = MagiBrain(_FakeSeat(GameAction(action="rest", args={})), None)
        self.assertEqual(b.mode, "pilot")

    def test_council_convenes_exactly_once_per_day(self) -> None:
        g = GameAction(action="gather", args={}, say="g")
        mel = _FakeSeat(g, think="計画")
        bal = _FakeSeat(g, think="評定")
        casp = _FakeSeat(g)
        council = self._pilot(mel, bal, casp)
        sim = self._sim()
        # Three turns on the SAME day → one morning council only.
        for _ in range(3):
            council.choose(sim)
        self.assertEqual(council.councils_held, 1)
        self.assertEqual(council.crisis_councils, 0)
        self.assertEqual(council.pilot_turns, 3)
        # Advance the in-game day → a second morning council convenes.
        sim.world.day += 1
        council.choose(sim)
        self.assertEqual(council.councils_held, 2)

    def test_crisis_reconvenes_once_per_day(self) -> None:
        g = GameAction(action="drink", args={}, say="g")
        council = self._pilot(_FakeSeat(g), _FakeSeat(g), _FakeSeat(g))
        sim = self._sim()
        council.choose(sim)                     # morning council (healthy)
        self.assertEqual(council.councils_held, 1)
        self.assertEqual(council.crisis_councils, 0)
        # Drive water through the crisis floor (≤20) → one re-council.
        sim.hero.water = 15
        council.choose(sim)
        self.assertEqual(council.crisis_councils, 1)
        # Still in crisis on the next turn, same day → NO second reconvene.
        council.choose(sim)
        self.assertEqual(council.crisis_councils, 1)

    def test_pilot_flies_through_melchior_choose_with_the_counsel(self) -> None:
        # v2.1: the pilot is MELCHIOR flying its OWN choose-path; the synthesized
        # 評定 must ride into that choose() call as an extra message every turn.
        g = GameAction(action="gather", args={}, say="g")
        mel = _FakeSeat(g, think="操縦の狙い")
        bal = _FakeSeat(g, think="その日の評定: 井戸を掘れ")
        council = self._pilot(mel, bal, _FakeSeat(g))
        council.choose(self._sim())
        self.assertTrue(council.counsel)
        # The MELCHIOR seat received the action call via choose() (not propose).
        self.assertEqual(mel.choose_calls, 1)
        self.assertEqual(mel.propose_calls, 0)
        # The day's 評定 rode in as an extra message on MELCHIOR's choose.
        extra = mel.last_choose_extra
        self.assertIsNotNone(extra)
        blob = " ".join(m["content"] for m in extra)
        self.assertIn(council.counsel, blob)
        self.assertIn("今日の評定", blob)

    def test_gemma_seat_gets_no_pilot_think_call(self) -> None:
        # The weakest brain no longer flies: the Gemma seat is only used for the
        # morning council counsel + synthesis (2 thinks/day), never to pilot.
        g = GameAction(action="rest", args={})
        mel = _FakeSeat(g)
        bal = _FakeSeat(g)
        council = self._pilot(mel, bal, _FakeSeat(g))
        # Three turns, same day → one council; the pilot adds NO think calls.
        for _ in range(3):
            council.choose(self._sim())
        # Gemma (bal) thinks exactly twice: its own counsel + the synthesis. No
        # per-turn pilot think exists any more, so it stays at 2 across 3 turns.
        self.assertEqual(bal.think_calls, 2)
        # MELCHIOR flew all three turns through choose().
        self.assertEqual(mel.choose_calls, 3)

    def test_council_seat_prompt_states_the_role_fact(self) -> None:
        from spl.agent.magi import COUNCIL_ROLE_FACT, _council_seat_prompt

        for seat in ("melchior", "balthasar", "casper"):
            self.assertIn(COUNCIL_ROLE_FACT, _council_seat_prompt(seat))
            self.assertIn("操縦はMELCHIORが行う", _council_seat_prompt(seat))

    def test_synthesis_prompt_is_the_survival_first_kata(self) -> None:
        # 評定 is a 型 (imperative routine) with the absolute 母優先 rule.
        from spl.agent.magi import COUNCIL_SYNTHESIS_PROMPT

        self.assertIn("母(BALTHASAR)の生存優先は絶対", COUNCIL_SYNTHESIS_PROMPT)
        self.assertIn("食と水より先に事業を語る評定を出して", COUNCIL_SYNTHESIS_PROMPT)
        self.assertIn("型", COUNCIL_SYNTHESIS_PROMPT)
        # The four 型 anchors.
        self.assertIn("朝まず水", COUNCIL_SYNTHESIS_PROMPT)
        self.assertIn("得たらすぐ食え", COUNCIL_SYNTHESIS_PROMPT)
        self.assertIn("腹と喉が満ちてから", COUNCIL_SYNTHESIS_PROMPT)
        self.assertIn("夕に明日", COUNCIL_SYNTHESIS_PROMPT)

    def test_role_fact_reaches_seat_think_calls(self) -> None:
        g = GameAction(action="rest", args={})
        mel = _FakeSeat(g)
        bal = _FakeSeat(g)
        council = self._pilot(mel, bal, _FakeSeat(g))
        council.choose(self._sim())
        # The role fact must appear in at least one seat's think system prompt.
        seen = mel.think_systems + bal.think_systems
        self.assertTrue(any("操縦はMELCHIORが行う" in s for s in seen))

    def test_pilot_status_line(self) -> None:
        g = GameAction(action="rest", args={})
        council = self._pilot(_FakeSeat(g), _FakeSeat(g), _FakeSeat(g))
        council.choose(self._sim())
        line = council.status_line()
        self.assertIn("MAGI v2.1 操縦MELCHIOR", line)
        self.assertIn("評定1", line)
        self.assertIn("手数1", line)

    def test_committee_mode_preserves_v1_behavior(self) -> None:
        from spl.agent.magi import MagiBrain

        a = GameAction(action="gather", args={}, say="a")
        b = GameAction(action="rest", args={}, say="b")
        c = GameAction(action="drink", args={}, say="c")
        mel = _FakeSeat(a)
        bal = _FakeSeat(b, think="bを採れ")
        casp = _FakeSeat(c)
        council = MagiBrain(mel, bal, casp, mode="committee")
        council.choose(Simulation(seed=42, max_days=112))
        rec = council.turn_records[-1]
        # v1 three-way split → moderator path, v1 status line, no pilot counters.
        self.assertFalse(rec["agreed"])
        self.assertTrue(rec["moderator_used"])
        self.assertIn("合議", council.status_line())
        self.assertEqual(council.pilot_turns, 0)
        self.assertEqual(council.councils_held, 0)

    def test_cassette_name_routes_mode(self) -> None:
        from spl.agent.magi import magi_mode_for_cassette

        self.assertEqual(magi_mode_for_cassette("MAGI"), "pilot")
        self.assertEqual(magi_mode_for_cassette(None), "pilot")
        self.assertEqual(magi_mode_for_cassette("MAGI-V1"), "committee")


class BodyScreamTests(unittest.TestCase):
    """体の声: critical stats scream at the TOP of the observation."""

    def test_dying_of_thirst_screams_first(self) -> None:
        from spl.agent.observer import ObservationBuilder
        from spl.core.sim import Simulation

        sim = Simulation(seed=42, max_days=112)
        sim.hero.water = 0
        obs = ObservationBuilder().build(sim)
        self.assertEqual(list(obs.keys())[0], "body")
        self.assertTrue(any("水" in s for s in obs["body"]))

    def test_healthy_hermit_has_no_body_key(self) -> None:
        from spl.agent.observer import ObservationBuilder
        from spl.core.sim import Simulation

        sim = Simulation(seed=42, max_days=112)
        obs = ObservationBuilder().build(sim)
        self.assertNotIn("body", obs)


class _StubBrain:
    """An OpenAICompatibleBrain whose ONLY network seam (_post_chat) is mocked, so
    the real _chat_timed / _propose_timed / ThreadPoolExecutor fan-out all run for
    real. A lens post (system prompt names one of the八識) returns a {counsel}
    JSON; the aggregator/choose post returns a valid action JSON. A set of lens
    names in ``dead_lenses`` raises inside the lens call to exercise the
    skip-on-failure path. ``posts`` records (is_lens, system_prompt) per call."""

    @staticmethod
    def make(parallel: int = 8, tps: float = 0.0, dead_lenses=None,
             lens_tokens: int = 7, agg_tokens: int = 40):
        from spl.agent.llm_client import Cassette, OpenAICompatibleBrain

        dead = set(dead_lenses or [])
        eight = __import__(
            "spl.agent.prompts", fromlist=["EIGHT_LENSES"]
        ).EIGHT_LENSES

        class _Brain(OpenAICompatibleBrain):
            def __init__(self):
                super().__init__(Cassette(
                    name="stub", base_url="http://stub/v1", parallel=parallel, tps=tps,
                ))
                self.posts = []  # (is_lens, system_prompt)

            def _resolve_model(self):
                return "stub-model"

            def _post_chat(self, payload):
                import json as _json

                system = payload["messages"][0]["content"]
                # a lens post is identified by its 識 marker 「<lens>」を司る識
                this_lens = next(
                    (lens for lens, _t in eight if f"「{lens}」を司る識" in system), None
                )
                is_lens = this_lens is not None
                self.posts.append((is_lens, system))
                if is_lens:
                    if this_lens in dead:
                        raise RuntimeError(f"lens {this_lens} is down")
                    body = _json.dumps(
                        {"counsel": f"{this_lens}の進言。水場へ向かえ。"},
                        ensure_ascii=False,
                    )
                    return body, lens_tokens
                # the 阿頼耶識 aggregator (or a normal choose proposal)
                body = _json.dumps(
                    {"think": "統合", "action": "rest", "args": {}, "say": "休む"},
                    ensure_ascii=False,
                )
                return body, agg_tokens

        return _Brain()


class HasshikiDeliberationTests(unittest.TestCase):
    """八識熟考 — parallel deliberation (mock transport; no live calls)."""

    def _sim(self, **stat_overrides):
        sim = Simulation(seed=42, max_days=112)
        for k, v in stat_overrides.items():
            setattr(sim.hero, k, v)
        return sim

    def test_cassette_parallel_flag_parses(self) -> None:
        from spl.agent.llm_client import find_cassette
        from spl.core.sim import PROJECT_ROOT

        path = PROJECT_ROOT / "config" / "models.toml"
        vllm = find_cassette(path, "Qwen仙人vLLM")
        self.assertEqual(vllm.parallel, 8)
        # a cassette without the key defaults to 0 (off): the bundled local policy.
        self.assertEqual(find_cassette(path, "Local仙人").parallel, 0)

    def test_lenses_for_uses_first_n_and_cycles_above_eight(self) -> None:
        from spl.agent.prompts import EIGHT_LENSES, lenses_for

        self.assertEqual([k for k, _ in lenses_for(3)], ["水", "食", "住"])
        self.assertEqual(len(lenses_for(8)), 8)
        twelve = [k for k, _ in lenses_for(12)]
        self.assertEqual(twelve[:8], [k for k, _ in EIGHT_LENSES])
        self.assertEqual(twelve[8:], ["水", "食", "住", "危険"])  # cycles
        self.assertEqual(lenses_for(0), [])

    def test_fanout_collects_n_counsels_and_one_aggregate_action(self) -> None:
        brain = _StubBrain.make(parallel=8)
        action = brain.deliberate(self._sim())
        self.assertEqual(action.action, "rest")  # the aggregator's synthesis
        # eight lens posts + one aggregate (non-lens) post.
        lens_posts = [p for p in brain.posts if p[0]]
        agg_posts = [p for p in brain.posts if not p[0]]
        self.assertEqual(len(lens_posts), 8)
        self.assertEqual(len(agg_posts), 1)
        self.assertEqual(len(brain.last_counsels), 8)
        self.assertEqual(brain.deliberations, 1)
        self.assertEqual(brain.calls, 1)

    def test_failed_lens_threads_are_skipped_without_crashing(self) -> None:
        brain = _StubBrain.make(parallel=8, dead_lenses={"水", "心", "長期"})
        action = brain.deliberate(self._sim())
        # the turn still produced an action; only the live lenses counselled.
        self.assertEqual(action.action, "rest")
        self.assertEqual(len(brain.last_counsels), 5)  # 8 - 3 dead
        kept = {lens for lens, _ in brain.last_counsels}
        self.assertNotIn("水", kept)
        self.assertNotIn("心", kept)
        # the aggregate action call still ran exactly once (the one non-lens post).
        self.assertEqual(len([p for p in brain.posts if not p[0]]), 1)

    def test_auto_burst_triggers_on_body_scream_and_not_otherwise(self) -> None:
        # Healthy hermit, toggle off → serial choose(), no deliberation.
        calm = _StubBrain.make(parallel=8)
        calm.deliberate_forced = False
        calm.choose_or_deliberate(self._sim())  # water/hunger healthy at seed 42
        self.assertEqual(calm.deliberations, 0)

        # Thirst screaming (water<=10) → auto-burst even with the toggle off.
        screaming = _StubBrain.make(parallel=8)
        screaming.deliberate_forced = False
        screaming.choose_or_deliberate(self._sim(water=8))
        self.assertEqual(screaming.deliberations, 1)

    def test_toggle_forces_deliberation_even_when_calm(self) -> None:
        brain = _StubBrain.make(parallel=8)
        brain.deliberate_forced = True
        brain.choose_or_deliberate(self._sim())  # calm, but forced
        self.assertEqual(brain.deliberations, 1)

    def test_parallel_zero_never_deliberates(self) -> None:
        # No parallel budget → choose_or_deliberate always falls to serial choose,
        # even when a body screams and the toggle is forced.
        brain = _StubBrain.make(parallel=0)
        brain.deliberate_forced = True
        brain.choose_or_deliberate(self._sim(water=2))
        self.assertEqual(brain.deliberations, 0)

    def test_aggregate_tps_recorded_once_as_sum_over_wall(self) -> None:
        # 8 lenses × 7 tokens + 40 aggregate = 96 tokens in ONE rolling sample.
        brain = _StubBrain.make(parallel=8, lens_tokens=7, agg_tokens=40)
        captured = {}
        orig = brain._record_tps_aggregate

        def _spy(tokens, seconds):
            captured["tokens"] = tokens
            captured["seconds"] = seconds
            return orig(tokens, seconds)

        brain._record_tps_aggregate = _spy
        self.assertEqual(len(brain._tps_samples), 0)
        brain.deliberate(self._sim())
        # exactly ONE aggregate sample (per-call recording suppressed).
        self.assertEqual(len(brain._tps_samples), 1)
        # the honesty contract: tokens = SUM of all N+1 completion tokens.
        self.assertEqual(captured["tokens"], 8 * 7 + 40)
        self.assertGreater(captured["seconds"], 0)
        # the recorded TPS is exactly sum / wall.
        self.assertAlmostEqual(
            brain._tps_samples[0], captured["tokens"] / captured["seconds"], places=6
        )
        self.assertEqual(brain.deliberations, 1)

    def test_status_line_gains_jukkou_marker_after_deliberation(self) -> None:
        brain = _StubBrain.make(parallel=8, tps=500.0)  # forced 仙界
        self.assertNotIn("熟考", brain.status_line())
        brain.deliberate(self._sim())
        self.assertIn("熟考1", brain.status_line())
        self.assertIn("仙界", brain.status_line())

    def test_condition_gate_caps_the_aggregator_tier(self) -> None:
        # A starving mind still fans out, but the aggregate call is condition-
        # capped to 雲水 (idx 0) exactly like choose().
        brain = _StubBrain.make(parallel=8, tps=500.0)  # would be 仙界 unconditionally
        brain.deliberate(self._sim(hunger=0))  # 飢渇 → cap to reflex
        self.assertEqual(brain.effective_tier_name, "雲水")
        self.assertEqual(brain.condition_note, "飢渇")

    def test_parse_counsel_reads_schema_field_and_falls_back(self) -> None:
        from spl.agent.llm_client import _parse_counsel

        # the schema path: {"counsel": "..."} → the field
        self.assertEqual(
            _parse_counsel('{"counsel": "水辺へ歩め。喉を潤せ。"}'), "水辺へ歩め。喉を潤せ。"
        )
        # schema dropped by an old backend → free-text fallback keeps clean JP
        self.assertEqual(_parse_counsel("水辺へ歩め。喉を潤せ。"), "水辺へ歩め。喉を潤せ。")
        self.assertIsNone(_parse_counsel(""))
        self.assertIsNone(_parse_counsel(None))

    def test_clean_counsel_strips_reasoning_leak(self) -> None:
        from spl.agent.llm_client import _clean_counsel

        # a clean Japanese reply passes through untouched
        self.assertEqual(_clean_counsel("水辺へ歩め。喉を潤せ。"), "水辺へ歩め。喉を潤せ。")
        # a <think> fence is stripped
        self.assertEqual(
            _clean_counsel("<think>reasoning…</think>水辺へ歩め。"), "水辺へ歩め。"
        )
        # an English 'thinking process' trace with NO JP conclusion → None
        leak = "Here's a thinking process:\n1. Analyze the body.\n2. Decide."
        self.assertIsNone(_clean_counsel(leak))

    def test_aggregate_call_carries_the_counsels(self) -> None:
        brain = _StubBrain.make(parallel=8)
        brain.deliberate(self._sim())
        # the schema (aggregate) post's last user message holds the 八識 block.
        # _post_chat only stores (had_schema, system); assert via last_counsels +
        # that the aggregate ran with all eight present.
        self.assertEqual(len(brain.last_counsels), 8)
        self.assertTrue(all(text for _, text in brain.last_counsels))


class PremonitionTests(unittest.TestCase):
    """体の予感: the whisper before the scream, on the honest decay arithmetic."""

    def _obs(self, **hero_kw):
        from spl.agent.observer import ObservationBuilder
        from spl.core.sim import Simulation

        sim = Simulation(seed=42, max_days=112)
        for k, v in hero_kw.items():
            setattr(sim.hero, k, v)
        return ObservationBuilder().build(sim)

    def test_low_hunger_whispers_before_screaming(self) -> None:
        obs = self._obs(hunger=28)
        self.assertIn("premonition", obs)
        self.assertTrue(any("腹の底が尽きる" in w for w in obs["premonition"]))
        self.assertNotIn("body", obs)  # not screaming yet

    def test_scream_supersedes_whisper(self) -> None:
        obs = self._obs(hunger=5)
        self.assertIn("body", obs)
        self.assertFalse(any("腹の底が尽きる" in w for w in obs.get("premonition", [])))

    def test_healthy_hermit_hears_nothing(self) -> None:
        obs = self._obs()
        self.assertNotIn("premonition", obs)


class SettlerBriefingTests(unittest.TestCase):
    """入植のしおり: the lethal arithmetic is disclosed at landing."""

    def test_system_prompt_discloses_the_laws(self) -> None:
        from spl.agent.prompts import SYSTEM_PROMPT

        for law in ("112 days", "HP reaches 0", "hunger -15", "water -20", "Winter"):
            self.assertIn(law, SYSTEM_PROMPT)


class MonumentTests(unittest.TestCase):
    """古い石碑 — the settlers' stone teaches REAL agronomy + past lives' mottos."""

    def test_inscription_logged_day_one_with_true_growth_days(self) -> None:
        # The stone's number for a crop must be exactly what the data says, read
        # from the SAME crop_book the sim loaded (no hard-coded magic number).
        sim = Simulation(seed=42, max_days=112)
        line = next((l for l in sim.full_log if "古い石碑が立つ" in l), None)
        self.assertIsNotNone(line, "monument inscription missing from day-1 log")
        turnip = sim.crop_book.get("turnip")
        wheat = sim.crop_book.get("wheat")
        self.assertIn(f"カブは{turnip.grow_days}日", line)
        self.assertIn(f"小麦は{wheat.grow_days}日", line)
        # the winter date and the settlers' warning are carved too
        self.assertIn("冬は85日目に来る", line)
        self.assertIn("実りより先に、種を数えよ", line)

    def test_obs_always_carries_monument_key(self) -> None:
        from spl.agent.observer import ObservationBuilder

        sim = Simulation(seed=42, max_days=112)
        obs = ObservationBuilder().build(sim)
        self.assertIn("monument", obs)
        mon = obs["monument"]
        self.assertIsInstance(mon, str)
        # a real crop's true growth days appears in the compact line
        self.assertIn(f"カブ{sim.crop_book.get('turnip').grow_days}日", mon)
        # background knowledge, not an alarm: it sits AFTER inventory/alerts
        keys = list(obs.keys())
        self.assertGreater(keys.index("monument"), keys.index("alerts"))
        self.assertGreater(keys.index("monument"), keys.index("inventory"))

    def test_epitaphs_logged_when_set_absent_otherwise(self) -> None:
        # No epitaphs by default -> no back-carving line on the stone.
        bare = Simulation(seed=42, max_days=20)
        self.assertEqual(bare.monument_epitaphs, [])
        self.assertFalse(any("石碑の裏" in l for l in bare.full_log))

        # Setting mottos carves them on the back: blanks dropped, capped at 2
        # (callers pass the last 1-2 entries; the sim just keeps the first two).
        sim = Simulation(seed=42, max_days=20)
        sim.set_monument_epitaphs(["ゆく河の流れは絶えずして", "", "水を絶やすな"])
        self.assertEqual(sim.monument_epitaphs, ["ゆく河の流れは絶えずして", "水を絶やすな"])
        carved = [l for l in sim.full_log if "石碑の裏" in l]
        self.assertEqual(len(carved), 2)
        self.assertIn("ゆく河の流れは絶えずして", carved[0])
        self.assertIn("水を絶やすな", carved[1])

    def test_merchant_arrival_appends_planting_hint(self) -> None:
        from spl.agent.policy import LocalPolicyAgent

        sim = Simulation(seed=42, max_days=20)
        agent = LocalPolicyAgent()
        for _ in range(20 * 60):
            if sim.done or sim.world.day > 7:
                break
            sim.step(agent.choose(sim))
        # the merchant arrives on day 7 (interval 7); the small-talk line follows.
        arrivals = [i for i, l in enumerate(sim.full_log) if "Merchant arrives" in l]
        self.assertTrue(arrivals, "no merchant arrival logged by day 7")
        hints = [l for l in sim.full_log if "行商人は世間話に言う" in l]
        self.assertTrue(hints, "merchant arrival did not append a planting hint")

    def test_merchant_hint_follows_the_deterministic_rule(self) -> None:
        from spl.core.crops import merchant_planting_hint

        book = Simulation(seed=42, max_days=112).crop_book
        # (a) a crop still fits -> recommend the most-nourishing such crop. On
        # day 70 pumpkin (12d, food 35) finishes by day 82 <= 84 -> pumpkin wins.
        self.assertEqual(
            merchant_planting_hint(book, 70, "autumn"),
            "今からカボチャを植えれば、冬の前に実りますよ",
        )
        # narrower window day 77: pumpkin no longer fits (77+12=89>84); the
        # best of what's left (tomato 7d food 20 vs turnip 4d food 18) is tomato.
        self.assertEqual(
            merchant_planting_hint(book, 77, "autumn"),
            "今からトマトを植えれば、冬の前に実りますよ",
        )
        # (b) the window has closed (nothing matures by winter) -> stores & walls.
        self.assertEqual(
            merchant_planting_hint(book, 84, "autumn"),
            "もう種時は過ぎましたなぁ。蓄えと壁の支度をなさい",
        )
        # (c) winter: nothing to plant -> keep seed for the 28-day spring.
        self.assertEqual(
            merchant_planting_hint(book, 90, "winter"),
            "春は二十八日続きます。種を残しておきなさい",
        )

    def test_cli_book_carves_last_mottos_as_epitaphs(self) -> None:
        # --book on: the last 1-2 past lives' mottos are carved on the stone's
        # back before play, so the graves teach on day 1.
        import os
        import tempfile
        from types import SimpleNamespace

        from spl.agent.bouken import BoukenNoSho, book_path_for
        from spl.ui import cli

        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("SPL_BOOK_DIR")
            os.environ["SPL_BOOK_DIR"] = tmp
            captured: dict = {}
            orig = cli.print_result

            def _spy(sim, motto=None, **kw):  # noqa: ANN001
                captured["epi"] = list(sim.monument_epitaphs)

            cli.print_result = _spy
            try:
                book = BoukenNoSho.load(book_path_for("EpiCass"))
                book.append({"seed": 0, "days": 5, "lessons": ["x"], "motto": "一生目の銘"})
                book.append({"seed": 0, "days": 9, "lessons": ["y"], "motto": "二生目の銘"})
                args = SimpleNamespace(seed=0, days=5, llm=False, cassette="EpiCass",
                                       strategy=None, tps=0, book=True)
                cli.run_simulate(args)
            finally:
                cli.print_result = orig
                if old is None:
                    os.environ.pop("SPL_BOOK_DIR", None)
                else:
                    os.environ["SPL_BOOK_DIR"] = old
            self.assertEqual(captured["epi"], ["一生目の銘", "二生目の銘"])

    def test_monument_draws_no_rng_so_runs_stay_deterministic(self) -> None:
        # Two same-seed local runs must still produce byte-identical full_log,
        # proving the stone + the merchant small-talk never touch sim.rng.
        left = run_local(seed=45, days=30)
        right = run_local(seed=45, days=30)
        self.assertEqual(left.full_log, right.full_log)
        # the new diegetic lines are actually present in that log
        self.assertTrue(any("古い石碑が立つ" in l for l in left.full_log))


@unittest.skipUnless(_HAS_PYGAME, "pygame not installed")
class MonumentPixelTests(unittest.TestCase):
    """古い石碑 in the voxel diorama: a stable, walkable tile + a rendered stele."""

    def _app(self):
        from types import SimpleNamespace

        from spl.ui.pixel.app import PixelApp

        args = SimpleNamespace(
            seed=42, days=112, llm=False, cassette="x", manual=True, speed=2,
            scale=2, start_day=0, shots=0, shots_ui=False, shot_dir="/tmp/spl_test",
            strategy=None, tps=0.0,
        )
        return PixelApp(args, headless=True)

    def test_monument_tile_is_deterministic_near_home_and_walkable(self) -> None:
        app = self._app()
        pos = app.sim.world.monument_pos
        # deterministic: the same world yields the same tile every read
        self.assertEqual((pos.x, pos.y), (app.sim.world.monument_pos.x, app.sim.world.monument_pos.y))
        # near home, never on water / home / workshop
        home = app.sim.world.start_pos
        self.assertLessEqual(abs(pos.x - home.x) + abs(pos.y - home.y), 4)
        self.assertNotIn(app.sim.world.tile_at(pos), {"water", "home", "workshop"})
        self.assertTrue(app.sim.world.in_bounds(pos))

    def test_tile_header_names_the_monument(self) -> None:
        app = self._app()
        header = app._tile_header(app.sim.world.monument_pos)
        self.assertIn("古い石碑", app.fonts.jp("古い石碑", "Old Stone Monument"))
        self.assertIn(app.fonts.jp("古い石碑", "Old Stone Monument"), header)

    def test_stele_sprite_builds_and_world_renders(self) -> None:
        app = self._app()
        spr = app.factory.stele()
        self.assertGreater(spr.get_width(), 0)
        self.assertGreater(spr.get_height(), 0)
        # the full voxel pipeline (which blits the stele on its tile) renders
        win = app.pg.Surface((app.lay.win_w, app.lay.win_h))
        app.render(win)  # must not raise


class KizamuTests(unittest.TestCase):
    """刻む (carve): the hermit's voluntary verse cut into the old stone, and its
    trans-generational persistence (per cassette+island)."""

    def _sim_at_stone(self, seed: int = 42, days: int = 12) -> Simulation:
        sim = Simulation(seed=seed, max_days=days)
        sim.hero.pos = sim.world.monument_pos
        return sim

    # -- the action ---------------------------------------------------------
    def test_carve_action_word_registered(self) -> None:
        from spl.core.actions import ACTION_WORDS

        self.assertIn("carve", ACTION_WORDS)

    def test_carve_success_adjacent_spends_ap_and_logs(self) -> None:
        sim = self._sim_at_stone()
        ap0 = sim.hero.ap_left
        result = sim.step(GameAction.safe("carve", text="ゆく河の流れは絶えずして"))
        self.assertTrue(result.ok, result.message)
        self.assertEqual(sim.hero.ap_left, ap0 - 1)
        self.assertEqual(sim.carvings_made, [(sim.world.day, "ゆく河の流れは絶えずして")])
        self.assertTrue(any("Carved into the stone" in ln for ln in sim.full_log))
        # 銘言 pool stays pure: a carve never appends to spoken_lines.
        self.assertEqual(sim.hero.spoken_lines, [])

    def test_carve_from_diagonal_neighbor_succeeds(self) -> None:
        sim = Simulation(seed=42, max_days=12)
        stone = sim.world.monument_pos
        from spl.core.hero import Position

        sim.hero.pos = Position(stone.x + 1, stone.y + 1)  # Chebyshev 1 (diagonal)
        result = sim.step(GameAction.safe("carve", text="月がふたつ"))
        self.assertTrue(result.ok, result.message)

    def test_carve_fails_when_far_from_stone(self) -> None:
        sim = Simulation(seed=42, max_days=12)
        stone = sim.world.monument_pos
        from spl.core.hero import Position

        # two tiles away on each axis -> Chebyshev 2, out of reach
        sim.hero.pos = Position(stone.x + 2, stone.y + 2)
        ap0 = sim.hero.ap_left
        result = sim.step(GameAction.safe("carve", text="遠い"))
        self.assertFalse(result.ok)
        self.assertIn("石碑のそばでなければ", result.message)
        # a world-reject fumbles -1 AP, but no carving is recorded
        self.assertEqual(sim.carvings_made, [])
        self.assertEqual(sim.hero.ap_left, ap0 - 1)

    def test_carve_rejects_over_60_chars(self) -> None:
        sim = self._sim_at_stone()
        result = sim.step(GameAction.safe("carve", text="あ" * 61))
        self.assertFalse(result.ok)
        self.assertEqual(sim.carvings_made, [])

    def test_carve_rejects_empty_text(self) -> None:
        sim = self._sim_at_stone()
        result = sim.step(GameAction.safe("carve", text="   "))
        self.assertFalse(result.ok)
        self.assertEqual(sim.carvings_made, [])

    def test_second_carve_same_day_fails_then_allowed_next_day(self) -> None:
        sim = self._sim_at_stone()
        first = sim.step(GameAction.safe("carve", text="一句目"))
        self.assertTrue(first.ok)
        second = sim.step(GameAction.safe("carve", text="二句目"))
        self.assertFalse(second.ok)
        self.assertIn("chisel needs rest", second.message)
        self.assertEqual(len(sim.carvings_made), 1)
        # roll to the next day; the chisel may carve again
        day0 = sim.world.day
        sim.end_day()
        self.assertEqual(sim.world.day, day0 + 1)
        self.assertFalse(sim.carved_today)
        sim.hero.pos = sim.world.monument_pos
        third = sim.step(GameAction.safe("carve", text="翌日の句"))
        self.assertTrue(third.ok, third.message)
        self.assertEqual(len(sim.carvings_made), 2)

    # -- persistence round-trip --------------------------------------------
    def test_stone_persistence_round_trip(self) -> None:
        import os
        import tempfile

        from spl.agent.bouken import append_carvings, load_stone, stone_path_for

        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("SPL_BOOK_DIR")
            os.environ["SPL_BOOK_DIR"] = tmp
            try:
                path = stone_path_for("StoneCass")
                self.assertEqual(load_stone(path), [])
                append_carvings(path, seed=42, day_texts=[(3, "水を掘れ"), (5, "火を建てよ")], life=1)
                rows = load_stone(path)
                self.assertEqual([r["text"] for r in rows], ["水を掘れ", "火を建てよ"])
                self.assertEqual(rows[0]["seed"], 42)
                self.assertEqual(rows[0]["day"], 3)
                self.assertEqual(rows[1]["life"], 1)
                # a second life appends, never overwrites
                append_carvings(path, seed=42, day_texts=[(2, "翌世の句")], life=2)
                self.assertEqual(len(load_stone(path)), 3)
                # empty day_texts is a true no-op
                append_carvings(path, seed=42, day_texts=[], life=3)
                self.assertEqual(len(load_stone(path)), 3)
            finally:
                if old is None:
                    os.environ.pop("SPL_BOOK_DIR", None)
                else:
                    os.environ["SPL_BOOK_DIR"] = old

    def test_set_stone_carvings_logs_and_observer_mentions_senjin(self) -> None:
        from spl.agent.observer import ObservationBuilder

        sim = Simulation(seed=42, max_days=12)
        sim.set_stone_carvings(["ゆく河の流れ", "実りより先に種を数えよ"])
        # day-1 log carries the 先人の手で句が line for each verse
        self.assertTrue(any("先人の手で句が刻まれている" in ln for ln in sim.full_log))
        obs = ObservationBuilder().build(sim)
        self.assertIn("先人の句", obs["monument"])
        self.assertIn("ゆく河の流れ", obs["monument"])

    def test_carve_then_persist_then_new_sim_inherits(self) -> None:
        """End-to-end: carve in life 1, persist, then a fresh sim fed the stone's
        carvings shows the day-1 先人の手で句が log and the observer monument
        mentions 先人の句."""
        import os
        import tempfile

        from spl.agent.bouken import append_carvings, load_stone, stone_path_for
        from spl.agent.observer import ObservationBuilder

        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("SPL_BOOK_DIR")
            os.environ["SPL_BOOK_DIR"] = tmp
            try:
                # life 1: carve a verse
                sim1 = self._sim_at_stone(seed=7)
                self.assertTrue(sim1.step(GameAction.safe("carve", text="先人の一句")).ok)
                path = stone_path_for("EndToEnd")
                append_carvings(path, seed=7, day_texts=sim1.carvings_made, life=0)
                # life 2 (same seed): inherit the most-recent 3 same-seed verses
                rows = load_stone(path)
                same = [r["text"] for r in rows if r["seed"] == 7][-3:]
                sim2 = Simulation(seed=7, max_days=12)
                sim2.set_stone_carvings(same)
                self.assertTrue(any("先人の手で句が刻まれている" in ln for ln in sim2.full_log))
                obs = ObservationBuilder().build(sim2)
                self.assertIn("先人の句", obs["monument"])
                self.assertIn("先人の一句", obs["monument"])
            finally:
                if old is None:
                    os.environ.pop("SPL_BOOK_DIR", None)
                else:
                    os.environ["SPL_BOOK_DIR"] = old

    # -- briefing / determinism --------------------------------------------
    def test_briefing_carries_the_carve_hint(self) -> None:
        from spl.agent.prompts import SYSTEM_PROMPT

        self.assertIn("carve", SYSTEM_PROMPT)
        self.assertIn("an old stone", SYSTEM_PROMPT)
        self.assertIn("One cut per day", SYSTEM_PROMPT)

    def test_local_policy_never_carves_and_stays_deterministic(self) -> None:
        # The local policy can't emit carve (policy.py untouched), so two same-seed
        # local runs stay bit-identical and neither leaves a carving.
        left = run_local(seed=45, days=30)
        right = run_local(seed=45, days=30)
        self.assertEqual(left.full_log, right.full_log)
        self.assertEqual(left.carvings_made, [])
        self.assertEqual(right.carvings_made, [])

    def test_local_policy_source_has_no_carve(self) -> None:
        import inspect

        from spl.agent import policy

        self.assertNotIn("carve", inspect.getsource(policy))


@unittest.skipUnless(_HAS_PYGAME, "pygame not installed")
class KizamuPixelTests(unittest.TestCase):
    """刻む in the voxel diorama: the popup item + the carve overlay dispatch."""

    def _app(self):
        from types import SimpleNamespace

        from spl.ui.pixel.app import PixelApp

        args = SimpleNamespace(
            seed=42, days=112, llm=False, cassette="x", manual=True, speed=2,
            scale=2, start_day=0, shots=0, shots_ui=False, shot_dir="/tmp/spl_test",
            strategy=None, tps=0.0,
        )
        return PixelApp(args, headless=True)

    def test_popup_on_monument_tile_when_adjacent_shows_carve_item(self) -> None:
        from spl.ui.pixel import iso

        app = self._app()
        app.manual = True
        stone = app.sim.world.monument_pos
        app.sim.hero.pos = stone  # standing on the stone (Chebyshev 0)
        cx, cy = iso.tile_center(stone.x, stone.y, app.offset_x, app.offset_y,
                                 app.lay.sprite_scale)
        app._handle_click(cx, cy)
        self.assertIsNotNone(app.popup)
        labels = [it.label for it in app.popup["items"]]
        carve_label = app.fonts.jp("句を刻む", "Carve a verse")
        self.assertIn(carve_label, labels)
        # the carve item opens the text overlay (does not dispatch directly)
        item = next(it for it in app.popup["items"] if it.label == carve_label)
        self.assertEqual(item.action, "carve_open")
        app._activate_menu_item(item)
        self.assertEqual(app.overlay, "carve")

    def test_carve_overlay_send_dispatches_the_action(self) -> None:
        app = self._app()
        app.manual = True
        app.sim.hero.pos = app.sim.world.monument_pos
        app.overlay = "carve"
        app.carve_text = "鴨長明の句"
        win = app.pg.Surface((app.lay.win_w, app.lay.win_h))
        app.render(win)  # populate the carve hit rects
        send = app._hits.get("carve_send")
        self.assertIsNotNone(send)
        app._click_overlay(send.centerx, send.centery)
        self.assertEqual(app.overlay, None)
        self.assertEqual([t for (_d, t) in app.sim.carvings_made], ["鴨長明の句"])

    def test_carve_overlay_esc_cancels_without_carving(self) -> None:
        app = self._app()
        app.sim.hero.pos = app.sim.world.monument_pos
        app.overlay = "carve"
        app.carve_text = "破棄される句"
        app._handle_key(app.pg.K_ESCAPE)
        self.assertIsNone(app.overlay)
        self.assertEqual(app.sim.carvings_made, [])


class ReasoningNoCountTests(unittest.TestCase):
    """推論ノーカウント: thought is taxed in wall-clock, not tokens."""

    def _brain(self, **kw):
        from spl.agent.llm_client import Cassette, OpenAICompatibleBrain

        kw.setdefault("max_tokens", 384)
        cas = Cassette(name="x", base_url="http://127.0.0.1:9", **kw)
        return OpenAICompatibleBrain(cas)

    def test_reasoning_model_gets_safety_ceiling(self) -> None:
        from spl.agent.llm_client import tier_for_tps

        brain = self._brain(reasoning=True)
        self.assertEqual(brain._completion_cap(tier_for_tps(50)), 4096)

    def test_plain_model_keeps_breathing_space_rule(self) -> None:
        from spl.agent.llm_client import tier_for_tps

        brain = self._brain(reasoning=False)
        self.assertEqual(brain._completion_cap(tier_for_tps(50)), 384)
        brain2 = self._brain(reasoning=False, max_tokens=1792)
        self.assertEqual(brain2._completion_cap(tier_for_tps(50)), 1792)


class _SerialExecutor:
    """A drop-in for ThreadPoolExecutor that runs submitted work IMMEDIATELY on the
    calling thread, in submission order, returning an already-resolved future. Used
    to make the concurrent-candidate reassembly DETERMINISTIC under test: each
    _propose runs in the exact order choose() submits it, so result index 0 is
    pinned to the first scripted value. (Concurrency itself is proven separately by
    the real-thread overlap test.)"""

    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def submit(self, fn, *args, **kwargs):
        fut: Future = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 - mirror executor semantics
            fut.set_exception(exc)
        return fut


def _serial_executor_patch():
    import spl.agent.llm_client as _mod

    return unittest.mock.patch.object(_mod, "ThreadPoolExecutor", _SerialExecutor)


class _CandidateStubBrain:
    """An OpenAICompatibleBrain whose ONLY network seam (_post_chat) is mocked, to
    exercise choose()'s candidate generation. Each proposal call returns a DISTINCT
    valid action keyed by the order the calls START (so the first-index successful
    proposal can be identified), and records the start order + thread id of every
    call so concurrency can be proven without any live request."""

    @staticmethod
    def make(parallel: int, tps: float = 500.0, actions=("forage", "rest", "drink")):
        from spl.agent.llm_client import Cassette, OpenAICompatibleBrain

        class _Brain(OpenAICompatibleBrain):
            def __init__(self):
                super().__init__(Cassette(
                    name="cand", base_url="http://stub/v1", parallel=parallel, tps=tps,
                ))
                self._lock = threading.Lock()
                self.start_order: list[int] = []   # call index in the order calls START
                self.threads: set[int] = set()     # distinct thread ids seen
                self.max_concurrent = 0
                self._inflight = 0

            def _resolve_model(self):
                return "stub-model"

            def _post_chat(self, payload):
                import json as _json

                from spl.agent.prompts import VERIFY_PROMPT

                system = payload["messages"][0]["content"]
                # The VERIFY pass (羅漢+) is a second thought, not a candidate. Echo
                # the FIRST proposal it was handed back UNCHANGED (a verify that
                # makes no correction), so it never perturbs the candidate
                # determinism/concurrency this stub is measuring.
                if system.startswith(VERIFY_PROMPT):
                    blob = _json.loads(payload["messages"][1]["content"])
                    first = blob["proposals"][0]
                    body = _json.dumps(
                        {"think": "verify", "action": first["action"],
                         "args": first.get("args", {}), "say": "案"},
                        ensure_ascii=False,
                    )
                    return body, 10

                with self._lock:
                    n = len(self.start_order)
                    self.start_order.append(n)
                    self.threads.add(threading.get_ident())
                    self._inflight += 1
                    self.max_concurrent = max(self.max_concurrent, self._inflight)
                # Earlier-starting calls take LONGER: if the calls were sequential
                # the slow first call would have to finish before the second began,
                # so observing 2 in-flight at once proves they did not wait.
                time.sleep(0.05 if n == 0 else 0.005)
                with self._lock:
                    self._inflight -= 1
                action = actions[n % len(actions)]
                body = _json.dumps(
                    {"think": f"c{n}", "action": action, "args": {}, "say": "案"},
                    ensure_ascii=False,
                )
                return body, 20

        return _Brain()


class ConcurrentCandidateTests(unittest.TestCase):
    """仙界 multi-candidate: parallel cassettes issue the proposals concurrently;
    a single-slot cassette stays strictly sequential. Mock transport, no live."""

    def _sim(self):
        return Simulation(seed=42, max_days=112)

    def test_candidates_issued_concurrently_when_parallel(self) -> None:
        # 仙界 (tps 500 → 2 candidates) + parallel=4 → both proposals in flight.
        # All candidates return the same action here; this test isolates the
        # CONCURRENCY claim (overlap), determinism is proven separately below.
        brain = _CandidateStubBrain.make(parallel=4, tps=500.0, actions=("forage",))
        self.assertEqual(brain.current_tier().candidates, 2)
        action = brain.choose(self._sim())
        # exactly two proposal calls went out, on two distinct threads, overlapping
        # — if they had been issued sequentially the slow first call would have had
        # to FINISH before the second began, so max_concurrent could never reach 2.
        self.assertEqual(len(brain.start_order), 2)
        self.assertEqual(len(brain.threads), 2)
        self.assertEqual(brain.max_concurrent, 2, "candidates did not overlap")
        self.assertEqual(action.action, "forage")

    def test_first_index_proposal_is_preferred_deterministically(self) -> None:
        # Determinism: choose() reassembles results IN SUBMISSION INDEX ORDER
        # (results[i] = fut.result()), so the FIRST-index proposal is the one
        # returned even when a later submission's value would differ. With the
        # executor patched to a serial, submission-order runner, each _propose call
        # consumes the next scripted result in submission order — index 0 ("forage")
        # then index 1 ("rest") — and choose() must prefer index 0.
        brain = _CandidateStubBrain.make(parallel=4, tps=500.0)
        self.assertEqual(brain.current_tier().candidates, 2)
        scripted = deque([("forage", None), ("rest", None)])

        def _scripted_propose(messages, budget):
            verb, _ = scripted.popleft()
            return GameAction(action=verb, args={}, think="s", say="案")

        brain._propose = _scripted_propose
        with _serial_executor_patch():
            action = brain.choose(self._sim())
        self.assertEqual(action.action, "forage")  # first submission index wins

    def test_concurrent_path_skips_failing_proposal_and_returns_valid(self) -> None:
        # A candidate that will not parse is skipped; as long as one valid proposal
        # exists the turn still acts (never invalid_llm_output).
        brain = _CandidateStubBrain.make(parallel=4, tps=500.0)
        scripted = deque([("__raise__", None), ("rest", None)])

        def _scripted_propose(messages, budget):
            verb, _ = scripted.popleft()
            if verb == "__raise__":
                raise ActionParseError("bad json")
            return GameAction(action=verb, args={}, think="s", say="案")

        brain._propose = _scripted_propose
        with _serial_executor_patch():
            action = brain.choose(self._sim())
        self.assertEqual(action.action, "rest")  # the one valid proposal survived
        self.assertNotEqual(action.action, "invalid_llm_output")

    def test_candidates_sequential_when_no_parallel_slots(self) -> None:
        # 仙界 wants 2 candidates but parallel<=1 → no executor, strictly serial.
        brain = _CandidateStubBrain.make(parallel=1, tps=500.0)
        self.assertEqual(brain.current_tier().candidates, 2)
        action = brain.choose(self._sim())
        self.assertEqual(len(brain.start_order), 2)
        # both calls ran on the SAME thread (the caller's), never overlapping.
        self.assertEqual(len(brain.threads), 1)
        self.assertEqual(brain.max_concurrent, 1, "sequential path must not overlap")
        self.assertEqual(action.action, "forage")  # first-index still preferred

    def test_single_candidate_tier_never_uses_executor(self) -> None:
        # 行者 (tps 80 → 1 candidate) with parallel=4 → one call, no fan-out.
        brain = _CandidateStubBrain.make(parallel=4, tps=80.0)
        self.assertEqual(brain.current_tier().candidates, 1)
        # 行者 has no verify pass, so choose() returns the lone proposal directly.
        action = brain.choose(self._sim())
        self.assertEqual(len(brain.start_order), 1)
        self.assertEqual(len(brain.threads), 1)
        self.assertEqual(action.action, "forage")

    def test_lenses_for_four_is_the_survival_core(self) -> None:
        # parallel=4 (LM Studio default) → the first four lenses: 水/食/住/危険.
        from spl.agent.prompts import lenses_for

        self.assertEqual([k for k, _ in lenses_for(4)], ["水", "食", "住", "危険"])

    def test_lms_cassettes_declare_parallel_four(self) -> None:
        from spl.agent.llm_client import find_cassette
        from spl.core.sim import PROJECT_ROOT

        path = PROJECT_ROOT / "config" / "models.toml"
        for name in ("ちび仙人", "Qwen仙人", "Step仙人"):
            self.assertEqual(find_cassette(path, name).parallel, 4, name)
        # vLLM stays at 8.
        self.assertEqual(find_cassette(path, "Qwen仙人vLLM").parallel, 8)
        self.assertEqual(find_cassette(path, "Gemma仙人MTP").parallel, 8)


class _OverflowStubBrain:
    """An OpenAICompatibleBrain whose ONLY network seam (_post_chat) is mocked to
    simulate a long-reasoning model whose completion is CUT OFF by max_tokens
    (finish_reason=length, empty/unparseable content). Each call sets
    self._last_finish_reason exactly as the real _post_chat would, and records the
    last user message of every call so the repair prompt used can be asserted.

    ``cut`` controls how many of the FIRST calls return a length-cut empty body;
    after that, calls return a valid action JSON (finish_reason=stop). With cut=1
    the first proposal overflows and the repair round succeeds; with cut=99 every
    call overflows so the turn falls through to invalid_llm_output."""

    @staticmethod
    def make(cut: int = 1, tps: float = 80.0, length_failure: bool = True):
        from spl.agent.llm_client import Cassette, OpenAICompatibleBrain

        class _Brain(OpenAICompatibleBrain):
            def __init__(self):
                super().__init__(Cassette(
                    name="of", base_url="http://stub/v1", tps=tps,
                ))
                self._calls = 0
                self.user_messages: list[str] = []  # last user content per call

            def _resolve_model(self):
                return "stub-model"

            def _post_chat(self, payload):
                import json as _json

                self._calls += 1
                self.user_messages.append(payload["messages"][-1]["content"])
                if self._calls <= cut:
                    # Cut off: empty (unparseable) content. The finish_reason is set
                    # exactly as the real _post_chat would on a max_tokens cut.
                    self._last_finish_reason = "length" if length_failure else "stop"
                    body = "" if length_failure else "garbled not json"
                    return body, 0
                # Recovery: a valid action, completion ran to a clean stop.
                self._last_finish_reason = "stop"
                body = _json.dumps(
                    {"think": "短く", "action": "rest", "args": {}, "say": "休む"},
                    ensure_ascii=False,
                )
                return body, 20

        return _Brain()


class RakuchouFeedbackTests(unittest.TestCase):
    """落丁フィードバック — when a completion is cut by max_tokens, the brain is
    told WHY (overflow-aware repair) and the cut is named in the world log.
    Mock transport only; no live calls to :1234 / :8011."""

    def _sim(self, **stat_overrides):
        sim = Simulation(seed=42, max_days=112)
        for k, v in stat_overrides.items():
            setattr(sim.hero, k, v)
        return sim

    def test_length_cut_repair_uses_overflow_prompt(self) -> None:
        from spl.agent.prompts import OVERFLOW_REPAIR_PROMPT

        # 行者 (tps 80) has a repair budget. First proposal is length-cut (empty),
        # so the repair round must use the OVERFLOW_REPAIR_PROMPT.
        brain = _OverflowStubBrain.make(cut=1, tps=80.0)
        action = brain.choose(self._sim())
        self.assertEqual(action.action, "rest")  # the repair recovered
        # Two calls: the cut proposal + the repair. The repair's user message is the
        # overflow-aware prompt, not the plain one.
        self.assertEqual(len(brain.user_messages), 2)
        self.assertIn(OVERFLOW_REPAIR_PROMPT, brain.user_messages[1])

    def test_repeated_length_cut_yields_overflow_cause(self) -> None:
        # Both the proposal AND the repair are length-cut → choose() returns
        # invalid_llm_output carrying args.cause == "overflow".
        brain = _OverflowStubBrain.make(cut=99, tps=80.0)
        action = brain.choose(self._sim())
        self.assertEqual(action.action, "invalid_llm_output")
        self.assertEqual(action.args.get("cause"), "overflow")

    def test_sim_step_logs_thought_overflow_confusion(self) -> None:
        # sim.step on an overflow invalid_llm_output logs the named confusion reason.
        sim = self._sim()
        action = GameAction(
            action="invalid_llm_output",
            args={"cause": "overflow"},
            think="cut off",
            say="",
        )
        sim.step(action)
        self.assertTrue(
            any("思考が長すぎて言葉にならなかった" in line for line in sim.full_log),
            sim.full_log,
        )
        # It is a confusion (a wasted turn), not a generic Unknown-action message.
        self.assertEqual(sim.hero.confusion_count, 1)
        self.assertFalse(
            any("Unknown action" in line for line in sim.full_log), sim.full_log
        )

    def test_non_length_parse_failure_uses_plain_repair(self) -> None:
        from spl.agent.prompts import OVERFLOW_REPAIR_PROMPT, REPAIR_PROMPT

        # A NON-length parse failure (finish_reason=stop, garbled body): the repair
        # round must use the plain REPAIR_PROMPT, never the overflow one.
        brain = _OverflowStubBrain.make(cut=1, tps=80.0, length_failure=False)
        action = brain.choose(self._sim())
        self.assertEqual(action.action, "rest")
        self.assertEqual(len(brain.user_messages), 2)
        self.assertIn(REPAIR_PROMPT, brain.user_messages[1])
        self.assertNotIn(OVERFLOW_REPAIR_PROMPT, brain.user_messages[1])

    def test_non_length_failure_has_no_overflow_cause(self) -> None:
        # Every call garbled but NOT length-cut → invalid_llm_output with NO cause,
        # and sim.step logs the generic Unknown-action confusion (not overflow).
        brain = _OverflowStubBrain.make(cut=99, tps=80.0, length_failure=False)
        action = brain.choose(self._sim())
        self.assertEqual(action.action, "invalid_llm_output")
        self.assertNotIn("cause", action.args)
        sim = self._sim()
        sim.step(action)
        self.assertTrue(
            any("Unknown action" in line for line in sim.full_log), sim.full_log
        )
        self.assertFalse(
            any("思考が長すぎて" in line for line in sim.full_log), sim.full_log
        )


class _PersonaStubBrain:
    """An OpenAICompatibleBrain whose ONLY network seam (_post_chat) is mocked so
    the real _chat_timed / message assembly runs. Every call records its system
    prompt in ``systems`` and returns a valid action or diary JSON depending on
    the prompt — so a test can assert the player persona reached the system
    message on both the action and the diary path. No live calls."""

    @staticmethod
    def make(persona: str = "庵の人格", tps: float = 80.0):
        from spl.agent.llm_client import Cassette, OpenAICompatibleBrain

        class _Brain(OpenAICompatibleBrain):
            def __init__(self):
                super().__init__(Cassette(
                    name="persona-stub", base_url="http://stub/v1",
                    persona=persona, tps=tps,
                ))
                self.systems: list[str] = []  # system prompt of every call

            def _resolve_model(self):
                return "stub-model"

            def _post_chat(self, payload):
                import json as _json

                from spl.agent.prompts import DIARY_PROMPT

                system = payload["messages"][0]["content"]
                self.systems.append(system)
                if DIARY_PROMPT in system:
                    return _json.dumps({"diary": "今日も生きた。"}, ensure_ascii=False), 20
                return _json.dumps(
                    {"think": "案", "action": "rest", "args": {}, "say": "休む"},
                    ensure_ascii=False,
                ), 20

        return _Brain()


class NyushokushaPersonaTests(unittest.TestCase):
    """入植者の来歴 — the player-written persona (mock transport; no live calls)."""

    def _brain(self, persona="庵の人格"):
        from spl.agent.llm_client import Cassette, OpenAICompatibleBrain

        return OpenAICompatibleBrain(
            Cassette(name="t", base_url="http://x/v1", persona=persona)
        )

    def _sim(self):
        return Simulation(seed=42, max_days=112)

    # -- effective_persona ---------------------------------------------------
    def test_no_player_persona_returns_cassette_persona_both_modes(self) -> None:
        b = self._brain("庵の人格")
        self.assertEqual(b.persona_mode, "append")  # default
        self.assertEqual(b.player_persona, "")
        self.assertEqual(b.effective_persona(), "庵の人格")
        b.persona_mode = "replace"
        self.assertEqual(b.effective_persona(), "庵の人格")

    def test_append_concatenates_with_bracket_header(self) -> None:
        b = self._brain("庵の人格")
        b.player_persona = "私は元・刀鍛冶。"
        eff = b.effective_persona()
        self.assertTrue(eff.startswith("庵の人格"))
        self.assertIn("入植者の来歴", eff)
        self.assertIn("It is who you are.", eff)
        self.assertIn("私は元・刀鍛冶。", eff)

    def test_replace_returns_player_text_only(self) -> None:
        b = self._brain("庵の人格")
        b.player_persona = "私は元・刀鍛冶。"
        b.persona_mode = "replace"
        self.assertEqual(b.effective_persona(), "私は元・刀鍛冶。")
        self.assertNotIn("庵の人格", b.effective_persona())

    def test_replace_empty_player_falls_back_to_cassette(self) -> None:
        b = self._brain("庵の人格")
        b.persona_mode = "replace"
        b.player_persona = "   "  # whitespace-only is empty
        self.assertEqual(b.effective_persona(), "庵の人格")

    # -- the persona reaches the system message (action + diary paths) -------
    def test_action_system_message_carries_player_persona(self) -> None:
        brain = _PersonaStubBrain.make(persona="庵の人格")
        brain.player_persona = "私は風を読む者。"
        brain.choose(self._sim())
        self.assertTrue(brain.systems, "no calls were made")
        self.assertTrue(
            all("私は風を読む者。" in s for s in brain.systems),
            brain.systems,
        )

    def test_diary_system_message_carries_player_persona(self) -> None:
        brain = _PersonaStubBrain.make(persona="庵の人格")
        brain.player_persona = "私は風を読む者。"
        out = brain.write_diary(self._sim(), season="春", weather="晴")
        self.assertEqual(out, "今日も生きた。")
        self.assertTrue(
            any("私は風を読む者。" in s for s in brain.systems), brain.systems
        )

    # -- CLI flags -----------------------------------------------------------
    def test_persona_flags_are_mutually_exclusive(self) -> None:
        from spl.main import build_parser

        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["play", "--persona", "A", "--persona-replace", "B"])

    def test_cli_persona_flag_wires_onto_the_brain(self) -> None:
        from types import SimpleNamespace

        from spl.ui.cli import apply_persona

        b = self._brain("庵の人格")
        apply_persona(b, SimpleNamespace(persona="私は旅人。", persona_replace=None))
        self.assertEqual(b.player_persona, "私は旅人。")
        self.assertEqual(b.persona_mode, "append")

    def test_cli_persona_replace_flag_wires_onto_the_brain(self) -> None:
        from types import SimpleNamespace

        from spl.ui.cli import apply_persona

        b = self._brain("庵の人格")
        apply_persona(b, SimpleNamespace(persona=None, persona_replace="私は鬼。"))
        self.assertEqual(b.player_persona, "私は鬼。")
        self.assertEqual(b.persona_mode, "replace")

    def test_apply_persona_noop_when_neither_flag(self) -> None:
        from types import SimpleNamespace

        from spl.ui.cli import apply_persona

        b = self._brain("庵の人格")
        apply_persona(b, SimpleNamespace(persona=None, persona_replace=None))
        self.assertEqual(b.player_persona, "")
        self.assertEqual(b.persona_mode, "append")
        # tolerant of a brain without the persona attributes (e.g. a MAGI marker)
        apply_persona(object(), SimpleNamespace(persona="x", persona_replace=None))

    # -- provenance: bouken entry persona (120-char cap) ---------------------
    def test_bouken_entry_records_persona_capped_at_120(self) -> None:
        from spl.agent.bouken import build_entry

        sim = self._sim()
        long_persona = "あ" * 200
        entry = build_entry(sim, seed=42, motto=None, player_persona=long_persona)
        self.assertEqual(len(entry["persona"]), 120)
        self.assertEqual(entry["persona"], "あ" * 120)

    def test_bouken_entry_persona_empty_when_none(self) -> None:
        from spl.agent.bouken import build_entry

        entry = build_entry(self._sim(), seed=42, motto=None)
        self.assertEqual(entry["persona"], "")


class PersonaPresetTests(unittest.TestCase):
    """来歴プリセット(DQ性格オマージュ) + 140字制限 + 冒頭注入。"""

    def test_presets_exist_and_fit_in_a_tweet(self) -> None:
        from spl.agent.prompts import PERSONA_PRESETS

        self.assertEqual(set(PERSONA_PRESETS), {"ごうけつ", "ひたむき", "こわがり", "まえむき"})
        for name, text in PERSONA_PRESETS.items():
            self.assertLessEqual(len(text), 140, name)

    def test_persona_is_injected_at_the_head(self) -> None:
        from spl.agent.llm_client import Cassette, OpenAICompatibleBrain
        from spl.agent.prompts import SYSTEM_PROMPT

        brain = OpenAICompatibleBrain(Cassette(name="x", base_url="http://stub/v1", persona="土台人格"))
        brain.player_persona = "汝は試験の仙人。"
        brain.persona_mode = "append"
        text = brain.system_for(SYSTEM_PROMPT)
        self.assertTrue(text.startswith("[入植者の来歴"))
        self.assertIn("汝は試験の仙人。", text.split("\n")[1])
        self.assertIn("土台人格", text)

    def test_over_140_is_rejected(self) -> None:
        from types import SimpleNamespace

        from spl.agent.llm_client import Cassette, OpenAICompatibleBrain
        from spl.ui.cli import apply_persona

        brain = OpenAICompatibleBrain(Cassette(name="x", base_url="http://stub/v1"))
        args = SimpleNamespace(persona="あ" * 141, persona_replace=None, persona_preset=None)
        with self.assertRaises(SystemExit):
            apply_persona(brain, args)

    def test_preset_flag_wires_the_text(self) -> None:
        from types import SimpleNamespace

        from spl.agent.llm_client import Cassette, OpenAICompatibleBrain
        from spl.agent.prompts import PERSONA_PRESETS
        from spl.ui.cli import apply_persona

        brain = OpenAICompatibleBrain(Cassette(name="x", base_url="http://stub/v1"))
        args = SimpleNamespace(persona=None, persona_replace=None, persona_preset="こわがり")
        apply_persona(brain, args)
        self.assertEqual(brain.player_persona, PERSONA_PRESETS["こわがり"])


class DifficultyTests(unittest.TestCase):
    """きびしさ (difficulty): two islands only — 楽園 (paradise) softens the
    nightly tax, 修羅 (carnage) is the canonical benchmark + every record's home
    ("the island was already 修羅")."""

    def test_shura_table_byte_matches_previous_constants(self) -> None:
        # Guard: 修羅 (the canonical benchmark, default) reproduces the exact
        # pre-difficulty numbers, so the record island never drifts. These are
        # byte-identical to the original constants.
        from spl.core.sim import DEFAULT_DIFFICULTY, DIFFICULTY

        self.assertEqual(DEFAULT_DIFFICULTY, "修羅")
        self.assertEqual(set(DIFFICULTY), {"楽園", "修羅"})  # exactly two islands
        f = DIFFICULTY["修羅"]
        self.assertEqual(f["hunger"], -15)
        self.assertEqual(f["water"], -20)
        self.assertEqual(f["sanity"], -2)
        self.assertEqual(f["sanity_house"], -1)
        self.assertEqual(f["starvation"], -10)
        self.assertEqual(f["dehydration"], -15)
        self.assertEqual(f["winter_hp"], -4)
        self.assertEqual(f["winter_sanity"], -2)
        self.assertEqual(f["inventory_bonus"], {})

    def test_paradise_difficulty_softens_one_end_day(self) -> None:
        # 楽園: a single nightly decay applies the gentler hunger/water deltas.
        sim = Simulation(seed=42, max_days=10, difficulty="楽園")
        self.assertEqual(sim.difficulty, "楽園")
        h0, w0 = sim.hero.hunger, sim.hero.water
        sim._daily_decay()
        self.assertEqual(sim.hero.hunger - h0, -10)
        self.assertEqual(sim.hero.water - w0, -14)

    def test_shura_difficulty_is_the_canonical_one_end_day(self) -> None:
        # 修羅: a single nightly decay applies the canonical hunger/water deltas.
        sim = Simulation(seed=42, max_days=10, difficulty="修羅")
        self.assertEqual(sim.difficulty, "修羅")
        h0, w0 = sim.hero.hunger, sim.hero.water
        sim._daily_decay()
        self.assertEqual(sim.hero.hunger - h0, -15)
        self.assertEqual(sim.hero.water - w0, -20)

    def test_paradise_difficulty_grants_start_inventory_bonus(self) -> None:
        easy = Simulation(seed=42, difficulty="楽園")
        norm = Simulation(seed=42, difficulty="修羅")
        self.assertEqual(easy.hero.inventory["berries"], norm.hero.inventory["berries"] + 3)
        self.assertEqual(easy.hero.inventory["wood"], norm.hero.inventory["wood"] + 2)

    def test_starvation_dehydration_bleed_scales_with_difficulty(self) -> None:
        # The HP bleed at hunger/water 0 follows the table (and logs the magnitude).
        for diff, starve, dehydrate in (("楽園", -7, -10), ("修羅", -10, -15)):
            sim = Simulation(seed=3, max_days=10, difficulty=diff)
            sim.hero.hunger = 0
            sim.hero.water = 0
            hp0 = sim.hero.hp
            sim._daily_decay()
            # both bleeds stack onto the same night
            self.assertEqual(sim.hero.hp - hp0, starve + dehydrate, diff)
            self.assertTrue(any(f"HP {starve}." in ln for ln in sim.full_log), diff)
            self.assertTrue(any(f"HP {dehydrate}." in ln for ln in sim.full_log), diff)

    def test_legacy_aliases_and_unknown_and_default_map_to_two_modes(self) -> None:
        # Default and any unknown value → 修羅 (canonical). Legacy three-mode names
        # still resolve: ふつう→修羅, やさしい→楽園. 楽園/修羅 pass through verbatim.
        self.assertEqual(Simulation(seed=1).difficulty, "修羅")
        self.assertEqual(Simulation(seed=1, difficulty="激辛").difficulty, "修羅")
        self.assertEqual(Simulation(seed=1, difficulty="ふつう").difficulty, "修羅")
        self.assertEqual(Simulation(seed=1, difficulty="やさしい").difficulty, "楽園")
        self.assertEqual(Simulation(seed=1, difficulty="楽園").difficulty, "楽園")
        self.assertEqual(Simulation(seed=1, difficulty="修羅").difficulty, "修羅")

    def test_default_run_matches_pre_change_behaviour_and_defaults_shura(self) -> None:
        # The existing determinism test proves same-seed reproduction; here we
        # assert explicitly that the DEFAULT difficulty is 修羅 and that an
        # explicit 修羅 reproduces the bare-default run byte-for-byte.
        bare = run_local(seed=45, days=30)
        explicit = Simulation(seed=45, max_days=30, difficulty="修羅")
        agent = LocalPolicyAgent()
        for _ in range(30 * 60):
            if explicit.done:
                break
            explicit.step(agent.choose(explicit))
        self.assertEqual(bare.difficulty, "修羅")
        self.assertEqual(bare.full_log, explicit.full_log)
        self.assertEqual(bare.score(), explicit.score())

    def test_difficulty_does_not_consume_rng(self) -> None:
        # きびしさ must scale fixed deltas only — never DRAW from the RNG. Drive two
        # sims (楽園 vs 修羅) with the SAME fixed action each turn so the hero
        # behaves identically; the RNG-driven world stream (weather transitions,
        # merchant arrivals, dog raids, storm damage) must then be byte-identical.
        # RNG-driven world events only: day/weather rolls, merchants, dog raids,
        # storm damage. NOT hero actions (identical here) and NOT the death line
        # (which legitimately fires sooner on the harsher tier).
        markers = ("begins:", "Merchant arrives", "Wild dogs", "Storm damage", "行商人")

        def world_stream(diff: str) -> tuple[list, list]:
            sim = Simulation(seed=7, max_days=14, difficulty=diff)
            weathers: list[str] = []
            for _ in range(14 * 12 + 20):
                if sim.done:
                    break
                sim.step(GameAction(action="rest"))  # same fixed action every turn
                weathers.append(sim.world.weather)
            events = [ln.split(": ", 1)[-1] for ln in sim.full_log
                      if any(m in ln for m in markers)]
            return weathers, events

        w_easy, e_easy = world_stream("楽園")
        w_shura, e_shura = world_stream("修羅")
        # 修羅 dies sooner, so its streams are a PREFIX of 楽園's — the RNG draw
        # order over the shared survival span must match exactly (difficulty never
        # draws; it only scales the tax that decides WHEN the prefix ends).
        n_w = min(len(w_easy), len(w_shura))
        n_e = min(len(e_easy), len(e_shura))
        self.assertGreater(n_e, 5, "streams too short to be a meaningful RNG check")
        self.assertEqual(w_easy[:n_w], w_shura[:n_w], "difficulty perturbed the weather RNG stream")
        self.assertEqual(e_easy[:n_e], e_shura[:n_e], "difficulty perturbed the RNG-driven world events")

    def test_settlers_briefing_tells_the_truth_per_difficulty(self) -> None:
        from spl.agent.prompts import settlers_briefing

        easy = settlers_briefing("楽園")
        self.assertIn("hunger -10, water -14", easy)
        self.assertIn("bleed 7 HP", easy)
        self.assertIn("bleed 10 HP", easy)
        self.assertIn("-2 HP and -1 sanity", easy)

        shura = settlers_briefing("修羅")
        self.assertIn("hunger -15, water -20", shura)
        self.assertIn("bleed 10 HP", shura)
        self.assertIn("bleed 15 HP", shura)
        self.assertIn("-4 HP and -2 sanity", shura)

    def test_system_prompt_shura_is_byte_identical_default(self) -> None:
        # The module default SYSTEM_PROMPT must carry the 修羅 briefing numbers
        # (the canonical benchmark; back-compat for the existing prompt tests).
        from spl.agent.prompts import SYSTEM_PROMPT, system_prompt_for_difficulty

        self.assertIn("hunger -15, water -20", SYSTEM_PROMPT)
        self.assertIn("bleed 10 HP", SYSTEM_PROMPT)
        self.assertIn("bleed 15 HP", SYSTEM_PROMPT)
        self.assertEqual(SYSTEM_PROMPT, system_prompt_for_difficulty("修羅"))

    def test_brain_action_system_message_carries_difficulty_numbers(self) -> None:
        # The brain's action system message (built from the sim) must contain the
        # difficulty-correct briefing — no server contact (system_prompt_for is a
        # pure string build over the stub cassette).
        from spl.agent.llm_client import Cassette, OpenAICompatibleBrain

        brain = OpenAICompatibleBrain(Cassette(name="x", base_url="http://stub/v1"))
        easy = Simulation(seed=42, max_days=10, difficulty="楽園")
        msg = brain.system_prompt_for(easy)
        self.assertIn("hunger -10, water -14", msg)
        self.assertIn("bleed 7 HP", msg)
        # a 修羅 sim yields the canonical numbers from the same call
        normal = Simulation(seed=42, max_days=10, difficulty="修羅")
        self.assertIn("hunger -15, water -20", brain.system_prompt_for(normal))

    def test_build_entry_records_difficulty(self) -> None:
        from spl.agent.bouken import build_entry

        sim = Simulation(seed=42, max_days=10, difficulty="楽園")
        entry = build_entry(sim, 42, {"motto": "x", "lessons": ["a", "b", "c"]})
        self.assertEqual(entry["difficulty"], "楽園")
        norm = build_entry(Simulation(seed=42, max_days=10), 42, {})
        self.assertEqual(norm["difficulty"], "修羅")

    def test_simulate_difficulty_flag_wires_into_the_sim(self) -> None:
        from types import SimpleNamespace

        from spl.ui import cli

        args = SimpleNamespace(
            seed=42, days=6, llm=False, cassette=None, strategy=None, tps=0,
            difficulty="楽園",
        )
        captured: dict[str, object] = {}
        orig = cli.print_result

        def _spy(sim, motto=None, **kw):  # noqa: ANN001
            captured["difficulty"] = sim.difficulty
            captured["berries"] = sim.hero.inventory.get("berries", 0)

        cli.print_result = _spy
        try:
            cli.run_simulate(args)
        finally:
            cli.print_result = orig
        self.assertEqual(captured["difficulty"], "楽園")

    def test_difficulty_parser_choices(self) -> None:
        from spl.main import build_parser

        parser = build_parser()
        for sub in ("play", "simulate", "pixel"):
            ns = parser.parse_args([sub, "--difficulty", "楽園"])
            self.assertEqual(ns.difficulty, "楽園")
        # default is 修羅 on simulate
        self.assertEqual(parser.parse_args(["simulate"]).difficulty, "修羅")
        # the legacy names are no longer argparse choices (they map via
        # normalize_difficulty, not the flag) — rejected at the parser
        for bad in ("激辛", "ふつう", "やさしい"):
            with self.assertRaises(SystemExit):
                parser.parse_args(["simulate", "--difficulty", bad])


def _advance_one_night(sim: Simulation, keep_alive: bool = True) -> None:
    """Cross exactly one day/night boundary deterministically: empty the AP and
    call end_day() once. Resolves any queued miracle at the boundary without
    relying on the local agent's choices, so the night-rule tests are exact.

    ``keep_alive`` tops up the hermit's vitals BEFORE the night so neglect alone
    never ends the run during a multi-night march (the death-night test passes
    keep_alive=False to let starvation/dehydration kill on purpose)."""
    if keep_alive:
        sim.hero.hp = 100
        sim.hero.hunger = 90
        sim.hero.water = 90
        sim.hero.sanity = 90
    sim.hero.ap_left = 0
    sim.end_day()


class DivineLeverTests(unittest.TestCase):
    """神のレバー (共度モード奇跡) — the full mock list from the plan. No network."""

    # -- 神力経済 (A-hybrid: 開始3・7日目朝+1・上限5・不足時拒否) -----------------
    def test_starting_power_is_three(self) -> None:
        sim = Simulation(seed=42, max_days=30)
        self.assertEqual(sim.divine.power, 3)
        self.assertEqual(sim.divine.miracles_used, 0)

    def test_power_grants_plus_one_every_seventh_morning(self) -> None:
        sim = Simulation(seed=42, max_days=30)
        # Spend nothing; walk to the morning of day 7 -> +1 (3 -> 4).
        while sim.world.day < 7 and not sim.done:
            _advance_one_night(sim)
        self.assertEqual(sim.world.day, 7)
        self.assertEqual(sim.divine.power, 4)
        # walk to day 14 -> +1 again (4 -> 5).
        while sim.world.day < 14 and not sim.done:
            _advance_one_night(sim)
        self.assertEqual(sim.divine.power, 5)

    def test_power_is_capped_at_five(self) -> None:
        sim = Simulation(seed=42, max_days=40)
        # by day 21 the un-spent power would be 6 without a cap; it must stay 5.
        while sim.world.day < 21 and not sim.done:
            _advance_one_night(sim)
        self.assertEqual(sim.divine.power, 5)

    def test_queue_rejected_when_power_insufficient(self) -> None:
        sim = Simulation(seed=42, max_days=10)
        sim.divine.power = 0
        ok, reason = sim.queue_miracle("weather", {"weather": "rain"})
        self.assertFalse(ok)
        self.assertIn("神力", reason)
        # nothing consumed, nothing logged, no pending effect
        self.assertEqual(sim.divine.power, 0)
        self.assertIsNone(sim.divine.forced_weather)
        self.assertEqual(sim.divine.miracles_used, 0)

    def test_score_never_counts_divine_power(self) -> None:
        a = Simulation(seed=42, max_days=10)
        before = a.score()
        a.divine.power = 5
        a.divine.miracles_used = 3
        self.assertEqual(a.score(), before)

    # -- 夜ルール (神は夜に働く) ------------------------------------------------
    def test_forced_weather_applies_next_morning(self) -> None:
        sim = Simulation(seed=1, max_days=10)
        # spring palette includes rain; force it for tomorrow.
        ok, _ = sim.queue_miracle("weather", {"weather": "rain"})
        self.assertTrue(ok)
        # nothing changes tonight — the EFFECT is at the boundary
        _advance_one_night(sim)
        self.assertEqual(sim.world.weather, "rain")
        self.assertIsNone(sim.divine.forced_weather)  # consumed

    def test_forced_weather_does_not_perturb_world_rng_stream(self) -> None:
        # The seed-comparison guard (plan §4): a forced weather still DRAWS
        # next_weather and DISCARDS it, and is fed the NATURAL weather shadow as
        # ``current`` — so the world RNG stream stays bit-identical to an un-helped
        # run. We ISOLATE the weather draw by walling the hermit off from every
        # weather-dependent RNG roll (house_upgrade => no storm-damage/winter roll;
        # fence => no dog raid) and forcing a CALM day so no new event fires; the
        # natural-weather shadow AND the live RNG state must then match exactly.
        def _walled(seed: int) -> Simulation:
            sim = Simulation(seed=seed, max_days=20)
            sim.hero.add_item("house_upgrade")
            sim.hero.add_item("fence")
            return sim

        plain = _walled(7)
        helped = _walled(7)
        helped.queue_miracle("weather", {"weather": "sunny"})  # calm forced day
        for _ in range(10):
            _advance_one_night(plain)
            _advance_one_night(helped)
            self.assertEqual(plain.last_natural_weather, helped.last_natural_weather,
                             f"natural-weather RNG diverged at day {plain.world.day}")
        self.assertEqual(plain.rng._rng.getstate(), helped.rng._rng.getstate(),
                         "world RNG state diverged after a forced weather")

    def test_manna_lands_in_the_morning_stores(self) -> None:
        sim = Simulation(seed=3, max_days=10)
        before = sim.hero.item_count("fish")
        ok, _ = sim.queue_miracle("manna", {"item": "fish"})
        self.assertTrue(ok)
        self.assertEqual(sim.hero.item_count("fish"), before)  # not yet
        _advance_one_night(sim)
        self.assertEqual(sim.hero.item_count("fish"), before + 3)
        self.assertEqual(sim.divine.pending_manna, {})  # consumed

    def test_manna_is_void_on_a_death_night_no_resurrection(self) -> None:
        # 神は夜に働く but does NOT resurrect: a hermit who dies tonight gets no
        # manna tomorrow (there is no tomorrow).
        sim = Simulation(seed=5, max_days=10)
        sim.queue_miracle("manna", {"item": "berries"})
        sim.hero.hp = 1
        sim.hero.hunger = 0
        sim.hero.water = 0
        before = sim.hero.item_count("berries")
        _advance_one_night(sim, keep_alive=False)  # starvation+dehydration kills
        self.assertFalse(sim.hero.alive)
        self.assertTrue(sim.failed)
        self.assertEqual(sim.hero.item_count("berries"), before)  # voided

    def test_dream_is_etched_into_tonights_memory(self) -> None:
        sim = Simulation(seed=9, max_days=10)
        ok, _ = sim.queue_miracle("dream", {"text": "西の岩場に水脈あり"})
        self.assertTrue(ok)
        day = sim.world.day
        _advance_one_night(sim)
        notes = sim.memory.notes.get(day, [])
        self.assertTrue(any("西の岩場に水脈あり" in n for n in notes),
                        f"dream not in tonight's notes: {notes}")
        # and it echoes through the recent window (memory diary)
        self.assertIn("西の岩場に水脈あり", sim.memory.recent_context(days=7))

    def test_summoned_merchant_arrives_next_morning_on_dedicated_rng(self) -> None:
        sim = Simulation(seed=11, max_days=10)
        self.assertIsNone(sim.current_offer)
        ok, _ = sim.queue_miracle("merchant", {})
        self.assertTrue(ok)
        _advance_one_night(sim)
        self.assertIsNotNone(sim.current_offer)
        self.assertFalse(sim.divine.pending_merchant)
        self.assertEqual(sim.divine.last_merchant_day, sim.world.day)

    def test_summoned_merchant_does_not_touch_world_rng(self) -> None:
        # The merchant lottery runs on GameRng(seed ^ SALT); a plain run's world
        # RNG stream must be untouched by the summon (weather sequence identical).
        plain = Simulation(seed=13, max_days=12)
        helped = Simulation(seed=13, max_days=12)
        helped.queue_miracle("merchant", {})
        for _ in range(6):
            _advance_one_night(plain)
            _advance_one_night(helped)
            self.assertEqual(plain.world.weather, helped.world.weather)

    # -- 制約 (constraints) ----------------------------------------------------
    def test_weather_cannot_be_forced_two_days_running(self) -> None:
        sim = Simulation(seed=1, max_days=10)
        ok, _ = sim.queue_miracle("weather", {"weather": "rain"})
        self.assertTrue(ok)
        # same day: a second weather cannot be queued (連日不可 keyed on the day).
        ok2, reason = sim.queue_miracle("weather", {"weather": "sunny"})
        self.assertFalse(ok2)
        self.assertIn("連日", reason)

    def test_weather_must_stay_in_season_palette(self) -> None:
        sim = Simulation(seed=1, max_days=10)  # day 1 = spring (no snow/drought)
        ok, reason = sim.queue_miracle("weather", {"weather": "snow"})
        self.assertFalse(ok)
        self.assertIn("季節", reason)

    def test_merchant_three_day_cooldown(self) -> None:
        sim = Simulation(seed=11, max_days=12)
        sim.queue_miracle("merchant", {})
        _advance_one_night(sim)  # merchant arrives day 2; last_merchant_day=2
        sim.current_offer = None  # clear so a new summon could land
        sim.divine.power = 5  # plenty of power, so the CD (not power) is tested
        # within 3 days of the last merchant, a summon is refused.
        ok, reason = sim.queue_miracle("merchant", {})
        self.assertFalse(ok)
        self.assertIn("商人", reason)

    def test_oracle_only_once_per_day(self) -> None:
        sim = Simulation(seed=2, max_days=10)
        # The 1日1回 constraint is checked BEFORE the 神力 balance, so it surfaces
        # even after the first oracle has spent the power (cost 3 from a start 3).
        ok, _ = sim.queue_miracle("oracle", {"text": "水を飲め"})
        self.assertTrue(ok)
        ok2, reason = sim.queue_miracle("oracle", {"text": "魚を獲れ"})
        self.assertFalse(ok2)
        self.assertIn("一日に一度", reason)

    def test_manna_rejects_items_off_the_whitelist(self) -> None:
        sim = Simulation(seed=2, max_days=10)
        for bad in ("wood", "stone", "iron_ore", "bread", "stew"):
            ok, reason = sim.queue_miracle("manna", {"item": bad})
            self.assertFalse(ok, f"{bad} should be off the manna whitelist")
            self.assertEqual(sim.divine.power, 3)  # nothing consumed

    # -- divine_command (神託 = 勅命) ------------------------------------------
    def test_divine_command_appears_below_body_above_strategy_and_consumes_once(self) -> None:
        from spl.agent.observer import ObservationBuilder

        sim = Simulation(seed=4, max_days=10)
        sim.set_strategy("井戸を最優先")
        sim.queue_miracle("oracle", {"text": "今すぐ水辺へ走れ"})
        sim.end_day()  # 神は夜に働く: the 勅命 is promoted at the night boundary
        # make the body scream so we can prove the ORDER of the keys
        sim.hero.water = 5
        obs = ObservationBuilder().build(sim)
        self.assertEqual(obs["divine_command"], "今すぐ水辺へ走れ")
        keys = list(obs.keys())
        # body (the flesh) outranks the 勅命; the 勅命 outranks the standing order
        self.assertLess(keys.index("body"), keys.index("divine_command"))
        self.assertLess(keys.index("divine_command"), keys.index("strategy_from_heaven"))
        # consumed on the first build — a second build no longer carries it
        obs2 = ObservationBuilder().build(sim)
        self.assertNotIn("divine_command", obs2)

    def test_divine_command_rides_the_stub_brains_system_and_observation(self) -> None:
        # The 勅命 must reach a real brain: it lands in the user-message obs JSON,
        # and the system prompt explains it. Uses the network-mocked _StubBrain.
        brain = _StubBrain.make(parallel=0)
        sim = Simulation(seed=4, max_days=10)
        sim.queue_miracle("oracle", {"text": "今すぐ水を飲め"})
        brain.choose(sim)
        # the LAST proposal/choose post carries the system prompt + the obs.
        # _StubBrain records (is_lens, system) per post; the system prompt must
        # name divine_command, and the obs JSON (built in choose) carried it.
        self.assertTrue(any("divine_command" in system for _is_lens, system in brain.posts))
        # the 勅命 was consumed exactly once by the brain's single build
        self.assertIsNone(sim.divine.divine_command)

    def test_system_prompt_explains_divine_command(self) -> None:
        from spl.agent.prompts import SYSTEM_PROMPT

        self.assertIn("divine_command", SYSTEM_PROMPT)
        self.assertIn("勅命", SYSTEM_PROMPT)

    # -- 共同 marking ----------------------------------------------------------
    def test_miracles_used_marks_the_bouken_entry_as_coop(self) -> None:
        from spl.agent.bouken import build_entry

        sim = Simulation(seed=6, max_days=10)
        sim.queue_miracle("dream", {"text": "備えよ"})
        sim.queue_miracle("manna", {"item": "fish"})
        entry = build_entry(sim, seed=6, motto={"motto": "x", "lessons": ["l"]})
        self.assertEqual(entry["miracles_used"], 2)
        # an unassisted life records 0
        clean = Simulation(seed=6, max_days=10)
        self.assertEqual(build_entry(clean, 6, {})["miracles_used"], 0)

    def test_compile_discounts_assisted_lives(self) -> None:
        import tempfile

        from spl.agent.bouken import BoukenNoSho, fallback_compile

        with tempfile.TemporaryDirectory() as tmp:
            book = BoukenNoSho.load(f"{tmp}/bouken_coop.json")
            # a LONG assisted life vs a SHORT unassisted life: the unassisted
            # life's lesson must win despite the shorter lifespan (偽教訓汚染ガード).
            book.append({"seed": 0, "days": 90, "lessons": ["奇跡で得た教え"],
                         "miracles_used": 4})
            book.append({"seed": 0, "days": 12, "lessons": ["己の足で得た教え"],
                         "miracles_used": 0})
            canon = fallback_compile(book)
            self.assertEqual(canon[0], "己の足で得た教え",
                             f"assisted life's lesson was not discounted: {canon}")
            # history_table exposes the marker to the LLM 編纂者 too
            self.assertEqual(book.history_table()[0]["miracles_used"], 4)

    def test_result_screen_marks_coop(self) -> None:
        import io
        from contextlib import redirect_stdout

        from spl.ui import cli

        sim = Simulation(seed=8, max_days=10)
        sim.queue_miracle("dream", {"text": "備えよ"})
        sim.completed = True
        sim.result_reason = "test"
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.print_result(sim)
        out = buf.getvalue()
        self.assertIn("共同", out)
        self.assertIn("夢のお告げ", out)

    # -- 決定論 ----------------------------------------------------------------
    def test_same_seed_and_miracle_log_reproduce_the_full_log(self) -> None:
        def run() -> Simulation:
            sim = Simulation(seed=21, max_days=12)
            scripted = {2: ("dream", {"text": "備えよ"}),
                        3: ("manna", {"item": "fish"}),
                        4: ("weather", {"weather": "rain"})}
            agent = LocalPolicyAgent()
            for _ in range(12 * 60):
                if sim.done:
                    break
                queued = scripted.pop(sim.world.day, None)
                if queued is not None:
                    sim.queue_miracle(*queued)
                sim.step(agent.choose(sim))
            return sim

        a, b = run(), run()
        self.assertEqual(a.full_log, b.full_log)
        self.assertEqual(a.divine.miracle_log, b.divine.miracle_log)
        self.assertEqual(a.score(), b.score())


if __name__ == "__main__":
    unittest.main()

