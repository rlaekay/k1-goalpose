"""Apply a measured joint-zero offset (delta) at the deploy boundary.

WHERE delta COMES FROM
======================
    python3 tools/collect_joint_zero.py --out /tmp/zero_poses.json   # on robot
    python3 tools/estimate_joint_zero.py --solve /tmp/zero_poses.json \
            --emit-yaml deploy/configs/joint_zero.yaml

CONVENTION (identical to training -- envs/K1/goal_pose.py:420-437)
==================================================================
    q_meas = q_true + delta

The robot's onboard PD servo closes on q_meas, which we cannot change.  So
correcting delta needs BOTH halves, and applying only one is worse than
applying neither:

    read   q_true  = q_meas - delta      so the policy sees the real posture
    write  q_cmd   = q_want + delta      so the servo settles at q_want

Half of it alone just moves the bias somewhere else:
  * subtract on read only  -> the policy sees the truth but commands land at
                              q_want - delta, so the robot stands wrong.
  * add on write only      -> the robot stands right but the policy is told it
                              is somewhere it is not.

INTEGRATION -- exactly two call sites in deploy_goal_pose.py
============================================================
1. `_low_state_handler`, where LowState is unpacked (around line 738):

       for i, motor in enumerate(low_state_msg.motor_state_serial):
    -      self.dof_pos_latest[i] = motor.q
    +      self.dof_pos_latest[i] = motor.q - self.joint_zero.delta[i]

   and the same subtraction in the `self.dof_pos[i] = motor.q` loop below it.

2. The single publish point (around line 877):

    -  self.low_cmd.motor_cmd[i].q = float(q_target[i])
    +  self.low_cmd.motor_cmd[i].q = float(q_target[i] + self.joint_zero.delta[i])

Everything between those two points -- observations, foot offsets, the
prepare/RL blend ramps, the tilt watchdog, the output filter -- then runs
entirely in the TRUE joint frame, with no other change.  That is the reason to
correct at the boundary instead of inside the policy.

⛔ The delta must be re-measured whenever the joints are re-zeroed, a leg is
   reassembled, or the robot is dropped.  It is hardware state, not tuning.
   `max_age_days` makes a stale file fail loudly instead of quietly steering.

⚠️ IMPORT TRAP: there are two `utils` packages -- `htwk-gym/utils/` (which HAS
   an __init__.py) and `htwk-gym/deploy/utils/` (which does not).  A regular
   package beats a namespace package regardless of sys.path order, so
   `from utils.joint_zero import ...` fails with ModuleNotFoundError whenever
   htwk-gym/ is on the path ahead of nothing -- e.g. `python -c` run from
   htwk-gym/.  deploy_goal_pose.py runs from deploy/, where it resolves
   correctly; tools that import this from elsewhere must insert the deploy
   directory and must not have htwk-gym/ on the path.
"""

import os
import time

import numpy as np


class JointZero(object):
    """Holds delta and applies it.  Disabled -> exactly a no-op."""

    def __init__(self, delta=None, joint_cnt=22, leg_dof_start=10,
                 source="(none)", enabled=False):
        self.joint_cnt = int(joint_cnt)
        self.leg_dof_start = int(leg_dof_start)
        self.source = source
        self.enabled = bool(enabled)
        self.delta = np.zeros(self.joint_cnt, dtype=np.float32)
        if delta is not None and self.enabled:
            self.delta[:] = delta

    # -- the two halves ----------------------------------------------------
    def correct_measurement(self, q_meas, out=None):
        """q_true = q_meas - delta."""
        if out is None:
            return np.asarray(q_meas, dtype=np.float32) - self.delta
        np.subtract(q_meas, self.delta, out=out)
        return out

    def correct_command(self, q_want, out=None):
        """q_cmd = q_want + delta."""
        if out is None:
            return np.asarray(q_want, dtype=np.float32) + self.delta
        np.add(q_want, self.delta, out=out)
        return out

    def describe(self):
        if not self.enabled:
            return "joint_zero: DISABLED (no correction applied)"
        ls, n = self.leg_dof_start, 12
        legs = self.delta[ls:ls + n]
        return ("joint_zero: ENABLED from %s -- max |delta| %.3f deg, "
                "rms %.3f deg" % (self.source,
                                  float(np.max(np.abs(legs))) * 57.29578,
                                  float(np.sqrt(np.mean(legs ** 2))) * 57.29578))


def load(path, joint_cnt=22, max_abs_deg=15.0, max_age_days=None,
         required=False):
    """Load deploy/configs/joint_zero.yaml.  Missing file -> disabled no-op.

    max_abs_deg exists because a bad solve is far more dangerous than no
    correction: delta is added straight to the joint commands, so a wild value
    drives the robot into its own limits at CUSTOM entry, in the window where
    nothing closes the balance loop.  15 deg is already implausible for a robot
    that passes "stands up without falling", so anything beyond it is a bug in
    the estimate, not a very badly zeroed robot.
    """
    import yaml

    if not os.path.exists(path):
        if required:
            raise IOError("joint_zero required but %s is missing" % path)
        return JointZero(joint_cnt=joint_cnt, source=path, enabled=False)

    cfg = yaml.safe_load(open(path, encoding="utf-8")) or {}
    jz = cfg.get("joint_zero") or {}
    if not jz.get("enabled", False):
        return JointZero(joint_cnt=joint_cnt, source=path, enabled=False)

    ls = int(jz.get("leg_dof_start", 10))
    vals = jz.get("delta_rad")
    if vals is None:
        raise ValueError("%s: joint_zero.delta_rad is missing" % path)
    vals = [float(v) for v in vals]
    if len(vals) == 12:
        full = [0.0] * joint_cnt
        full[ls:ls + 12] = vals
    elif len(vals) == joint_cnt:
        full = vals
    else:
        raise ValueError(
            "%s: joint_zero.delta_rad has %d entries; expected 12 (legs only) "
            "or %d (all joints)" % (path, len(vals), joint_cnt))

    worst = max(abs(v) for v in full)
    if worst > max_abs_deg / 57.29578:
        raise ValueError(
            "%s: |delta| reaches %.2f deg, over the %.1f deg sanity limit. "
            "delta is added to the joint COMMANDS, so a bad estimate steers the "
            "robot into its limits during CUSTOM entry, where nothing closes "
            "the balance loop. Re-run the estimator and check its residual "
            "reduction before raising this limit."
            % (path, worst * 57.29578, max_abs_deg))

    if max_age_days is not None:
        age = (time.time() - os.path.getmtime(path)) / 86400.0
        if age > max_age_days:
            raise ValueError(
                "%s is %.1f days old (limit %.1f). Joint zeros drift; "
                "re-measure or pass max_age_days=None deliberately."
                % (path, age, max_age_days))

    return JointZero(full, joint_cnt=joint_cnt, leg_dof_start=ls,
                     source=path, enabled=True)
