#!/usr/bin/env python3
"""Validate RunnerV3's atomic post-update HBatch health attestation."""

import argparse
import json
import math
import os
import sys


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def verify(path, token, num_envs, min_iterations,
           horizon_length=24, mini_epochs=5, num_minibatches=4):
    require(os.path.isfile(path), "health marker is missing")
    with open(path, encoding="utf-8") as f:
        marker = json.load(f)
    require(marker.get("version") == 1 and marker.get("status") == "healthy",
            "health marker schema/status mismatch")
    require(marker.get("health_token") == token,
            "health marker belongs to a different launch")
    require(int(marker.get("num_envs", -1)) == num_envs,
            "health marker used {} envs, expected {}".format(
                marker.get("num_envs"), num_envs))
    require(int(marker.get("horizon_length", -1)) == horizon_length and
            int(marker.get("mini_epochs", -1)) == mini_epochs and
            int(marker.get("num_minibatches", -1)) == num_minibatches,
            "PPO shape differs from frozen HBatch production shape")
    completed = int(marker.get("completed_iterations", 0))
    steps = int(marker.get("optimizer_steps", 0))
    require(completed >= min_iterations,
            "only {} healthy iterations, need {}".format(
                completed, min_iterations))
    require(steps == completed * mini_epochs * num_minibatches,
            "optimizer-step attestation mismatch: {} for {} iterations".format(
                steps, completed))
    require(marker.get("finite_checks") is True and
            marker.get("post_update_forward_finite") is True,
            "finite/post-update-forward gate was not active")
    require(float(marker.get("parameter_delta_max", 0.0)) > 0.0,
            "optimizer did not change the actor parameters")

    configured_lr = float(marker.get("configured_learning_rate", float("nan")))
    initial_lr = float(marker.get(
        "initial_optimizer_learning_rate", float("nan")))
    current_lr = float(marker.get("current_learning_rate", float("nan")))
    grad_norm = float(marker.get("max_grad_norm_seen", float("nan")))
    mirror_share = float(marker.get("mirror_valid_share", float("nan")))
    require(all(math.isfinite(x) for x in
                (configured_lr, initial_lr, current_lr, grad_norm, mirror_share)),
            "health marker contains a nonfinite scalar")
    require(configured_lr == 5.0e-6 and initial_lr == configured_lr,
            "checkpoint optimizer LR leaked into H warm start: configured {}, initial {}"
            .format(configured_lr, initial_lr))
    require(1.0e-6 <= current_lr <= 1.0e-5,
            "adaptive LR left the frozen H bounds: {}".format(current_lr))
    require(grad_norm >= 0.0 and 0.0 < mirror_share <= 1.0,
            "invalid grad norm or mirror support share")
    return marker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", required=True)
    parser.add_argument("--health_token", required=True)
    parser.add_argument("--num_envs", type=int, required=True)
    parser.add_argument("--min_iterations", type=int, default=2)
    args = parser.parse_args()
    try:
        marker = verify(
            os.path.abspath(args.marker), args.health_token,
            args.num_envs, args.min_iterations)
    except Exception as exc:
        print("FAIL  HBatch training health: {}".format(exc), flush=True)
        return 1
    print("PASS  HBatch training health: {} envs, {} iterations, {} updates, "
          "lr {:.3g}, max grad {:.3g}, mirror support {:.1%}".format(
              marker["num_envs"], marker["completed_iterations"],
              marker["optimizer_steps"], marker["current_learning_rate"],
              marker["max_grad_norm_seen"], marker["mirror_valid_share"]),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
