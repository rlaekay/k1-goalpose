#!/usr/bin/env python3
"""Dependency-free regressions for the M-cell causal plumbing.

The Mac authoring environment has no Isaac Gym/PyTorch/PyYAML, so these tests
guard the exact high-risk wiring statically; the server smoke supplies the
dynamic physics and PPO checks before training.
"""

import ast
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(relative):
    with open(os.path.join(ROOT, relative), encoding="utf-8") as f:
        return f.read()


class MCellStaticTest(unittest.TestCase):

    def test_python_sources_parse(self):
        for path in (
                "envs/K1/goal_pose_hbatch.py", "eval_goal_pose.py",
                "utils/runner_v3.py", "tools/make_mcell_configs.py",
                "tools/compare_mcells.py", "tools/smoke_v7.py"):
            ast.parse(read(path), filename=path)

    def test_mirror_ppo_uses_source_behavior_density(self):
        source = read("utils/runner_v3.py")
        self.assertNotIn("old_mirror_logprob", source)
        self.assertIn(
            "_bounded_surrogate_loss(\n"
            "                                old_logprob[idx][valid_pos]",
            source)

    def test_scenario_sampler_is_on_the_live_force_path(self):
        source = read("envs/K1/goal_pose_hbatch.py")
        self.assertIn("self._sample_scenario_events(fire)", source)
        self.assertIn("gymapi.ENV_SPACE", source)
        self.assertIn("dist_event_submitted_impulse_vec +=", source)
        self.assertIn("was_active & (self.dist_steps_left == 0)", source)

    def test_generator_uses_runner_schema_and_g1_mirror_ablation(self):
        source = read("tools/make_mcell_configs.py")
        self.assertIn('put(cfg, "runner.save_interval", SAVE_INTERVAL)', source)
        self.assertIn('put(cfg, "runner.load_optimizer_state", False)', source)
        self.assertNotIn('put(cfg, "basic.save_interval"', source)
        self.assertNotIn('put(cfg, "algorithm.load_optimizer_state"', source)
        self.assertIn('elif name == "M3_mirror_off-codex"', source)
        self.assertIn('put(cfg, "algorithm.symmetry_coef", 0.0)', source)

    def test_all_generated_config_and_report_names_are_codex_suffixed(self):
        generator = read("tools/make_mcell_configs.py")
        compare = read("tools/compare_mcells.py")
        self.assertIn('"M0_control-codex"', generator)
        self.assertIn('"mcell-report-codex.md"', compare)
        self.assertNotIn('"mcell-report.md"', compare)

    def test_eval_has_force_distribution_and_delivery_audit(self):
        source = read("eval_goal_pose.py")
        for token in (
                '"scenario_breakdown"', '"height_tier_breakdown"',
                '"direction_octants_robot_local"', '"delivery_audit"'):
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
