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
            self.assertEqual(table[0], {"life": 1, "days": 3, "ending": "渇き", "lessons": ["水"]})
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


if __name__ == "__main__":
    unittest.main()

