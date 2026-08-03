#!/usr/bin/env python3
"""Static unit tests for eval_goal_pose joint-DR probes.

The evaluator imports Isaac Gym before torch, which is unavailable on many
analysis machines.  Load only the pure configuration/provenance functions from
its AST so these regression tests remain CPU- and Isaac-Gym-independent.
"""

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import tempfile
import unittest


EVALUATOR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "eval_goal_pose.py"))
PURE_FUNCTIONS = {
    "_stable_protocol_sha",
    "_apply_hbatch_common_eval",
    "_validate_joint_dr_probe_value",
    "_nonnegative_finite_float",
    "_apply_joint_dr_probe",
    "prepare_cfg",
    "_artifact_sha256",
    "_file_provenance",
    "_attach_input_provenance",
}


def load_pure_functions():
    with open(EVALUATOR, "r", encoding="utf-8") as stream:
        tree = ast.parse(stream.read(), filename=EVALUATOR)
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in PURE_FUNCTIONS
    ]
    missing = PURE_FUNCTIONS.difference(node.name for node in selected)
    if missing:
        raise RuntimeError("missing evaluator functions: {}".format(sorted(missing)))
    namespace = {
        "argparse": argparse,
        "copy": copy,
        "hashlib": hashlib,
        "json": json,
        "math": math,
        "os": os,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), EVALUATOR, "exec"),
         namespace)
    return namespace


FN = load_pure_functions()


def hbatch_cfg():
    return {
        "basic": {},
        "env": {},
        "viewer": {},
        "commands": {},
        "noise": {"legacy": {"std": 1.0}},
        "randomization": {
            "kick_interval_s": 2.0,
            "push_interval_s": 3.0,
        },
        "evaluation": {
            "hbatch_common_eval": {
                "noise_overrides": {"common": {"std": 0.2}},
                "randomization_overrides": {
                    "joint_encoder_bias": {
                        "range": [-0.025, 0.025],
                        "operation": "additive",
                        "distribution": "uniform",
                    },
                    "joint_target_offset": {
                        "range": [-0.02, 0.02],
                        "operation": "additive",
                        "distribution": "uniform",
                    },
                    "init_dof_pos": {
                        "range": [0.0, 0.075],
                        "operation": "additive",
                        "distribution": "gaussian",
                    },
                },
                "disturbance": {"enabled": True},
            }
        },
    }


class JointProbeTests(unittest.TestCase):
    def prepare(self, cfg, **kwargs):
        return FN["prepare_cfg"](
            cfg, "K1/Goal_Pose_HBatch", 4, **kwargs)

    def test_unspecified_probe_preserves_common_profile(self):
        cfg = hbatch_cfg()
        self.prepare(cfg)
        randomization = cfg["randomization"]
        self.assertEqual(randomization["joint_encoder_bias"]["range"],
                         [-0.025, 0.025])
        self.assertEqual(randomization["joint_target_offset"]["range"],
                         [-0.02, 0.02])
        self.assertEqual(randomization["init_dof_pos"]["range"], [0.0, 0.075])
        self.assertEqual(cfg["evaluation"]["joint_dr_probe"], {
            "joint_encoder_bias_rad": None,
            "joint_target_offset_rad": None,
            "init_dof_std_rad": None,
            "active": False,
        })

    def test_probe_wins_after_common_override_and_survives_no_noise(self):
        cfg = hbatch_cfg()
        self.prepare(
            cfg, no_noise=True, joint_encoder_bias_rad=0.011,
            joint_target_offset_rad=0.012, init_dof_std_rad=0.013)
        randomization = cfg["randomization"]
        self.assertEqual(cfg["noise"], {})
        self.assertEqual(randomization["joint_encoder_bias"], {
            "range": [-0.011, 0.011], "operation": "additive",
            "distribution": "uniform",
        })
        self.assertEqual(randomization["joint_target_offset"]["range"],
                         [-0.012, 0.012])
        self.assertEqual(randomization["init_dof_pos"], {
            "range": [0.0, 0.013], "operation": "additive",
            "distribution": "gaussian",
        })
        self.assertTrue(cfg["evaluation"]["joint_dr_probe"]["active"])

    def test_explicit_zero_is_a_nominal_ablation(self):
        cfg = hbatch_cfg()
        self.prepare(cfg, joint_encoder_bias_rad=0.0,
                     joint_target_offset_rad=0.0, init_dof_std_rad=0.0)
        self.assertEqual(cfg["randomization"]["joint_encoder_bias"]["range"],
                         [-0.0, 0.0])
        self.assertEqual(cfg["randomization"]["joint_target_offset"]["range"],
                         [-0.0, 0.0])
        self.assertEqual(cfg["randomization"]["init_dof_pos"]["range"],
                         [0.0, 0.0])

    def test_invalid_magnitudes_are_rejected(self):
        validate = FN["_validate_joint_dr_probe_value"]
        for value in (-0.001, float("nan"), float("inf"), "bad"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate("probe", value)

    def test_probe_changes_effective_protocol_fingerprint(self):
        zero = hbatch_cfg()
        nonzero = hbatch_cfg()
        self.prepare(zero, joint_encoder_bias_rad=0.0)
        self.prepare(nonzero, joint_encoder_bias_rad=0.01)
        self.assertNotEqual(
            zero["evaluation"]["effective_eval_protocol_sha"],
            nonzero["evaluation"]["effective_eval_protocol_sha"])

    def test_input_provenance_hashes_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "model.pth")
            config = os.path.join(directory, "config.yaml")
            with open(checkpoint, "wb") as stream:
                stream.write(b"checkpoint-bytes")
            with open(config, "wb") as stream:
                stream.write(b"config-bytes")
            results = {}
            FN["_attach_input_provenance"](results, checkpoint, config)
            self.assertEqual(
                results["input_provenance"]["checkpoint"]["sha256"],
                hashlib.sha256(b"checkpoint-bytes").hexdigest())
            self.assertEqual(
                results["input_provenance"]["source_config"]["sha256"],
                hashlib.sha256(b"config-bytes").hexdigest())


if __name__ == "__main__":
    unittest.main(verbosity=2)
