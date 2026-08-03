#!/usr/bin/env python3
"""Enter CUSTOM while holding the current pose, and measure what changes.

This answers the two questions the safety policy hangs on, without running any
policy:

  - does rt/odometer_state keep publishing in CUSTOM?
  - what does the CUSTOM transition cost in wall clock and joint travel?

No policy, no gait, no motion command. The robot is commanded to hold the pose
it is already in, at the gains given in the config's `prepare` block, and that
command keeps streaming for the whole test. That is strictly less than what
deploy_goal_pose.py does today, which sends exactly one frame and then stops
publishing until the operator starts the gait.

A HUMAN MUST BE AT THE ROBOT. Ctrl-C requests DAMPING and exits.

    python3 probe_custom_hold.py --config <path to Goal_Pose_E0.yaml> --hold 6
"""

import argparse
import json
import signal
import sys
import threading
import time

import numpy as np
import yaml

from booster_robotics_sdk_python import (
    B1LocoClient, B1LowStateSubscriber, B1OdometerStateSubscriber,
    ChannelFactory, LowCmd, LowCmdType, MotorCmd, RobotMode, B1LowCmdPublisher,
)

# Modes reachable from the Python binding. kSoccer is absent from the Python
# RobotMode enum even though the C++ header has it, so a soccer transition can
# only be timed from the Brain side.
ALL_MODES = ["damping", "prepare", "walking", "custom"]


class Probe:
    def __init__(self, cfg, net):
        self.cfg = cfg
        self.running = True
        self.low_q = None
        self.low_stamps = []
        self.odom_stamps = []
        self._entered_custom = False
        self._lock = threading.Lock()

        ChannelFactory.Instance().Init(0, net)

        self.low_sub = B1LowStateSubscriber(self._on_low)
        self.low_sub.InitChannel()
        self.odom_sub = B1OdometerStateSubscriber(self._on_odom)
        self.odom_sub.InitChannel()
        self.pub = B1LowCmdPublisher()
        self.pub.InitChannel()
        self.client = B1LocoClient()
        self.client.Init()

        self.low_cmd = LowCmd()
        self.publisher_thread = None

    def _on_low(self, msg):
        # Copy the joint values out here. The message object is only valid for
        # the duration of the callback: holding the reference and indexing it
        # later raced the SDK reusing the buffer and threw IndexError mid-run.
        try:
            q = [float(j.q) for j in msg.motor_state_serial]
        except Exception:
            return
        if q:
            self.low_q = q
        self.low_stamps.append(time.monotonic())

    def _on_odom(self, _msg):
        self.odom_stamps.append(time.monotonic())

    def legs(self):
        q = self.low_q
        if not q:
            return None
        n = len(q)
        # Leg slice comes from the config, not from a hardcoded 11..23: this K1
        # has 22 joints with legs at 10..21.
        start = int(self.cfg["policy"].get("leg_dof_start", 10))
        count = int(self.cfg["policy"].get("num_actions", 12))
        if start + count > n:
            return None
        return q[start:start + count]

    def rate(self, stamps, t0, t1):
        n = sum(1 for s in stamps if t0 <= s <= t1)
        return n / max(1e-9, t1 - t0)

    def build_hold_cmd(self):
        """Hold exactly where the robot is now, at the config's prepare gains."""
        q = self.low_q
        if not q:
            raise RuntimeError("no low_state; robot stack not running")
        n = len(q)
        cfg_n = int(self.cfg["common"].get("joint_cnt",
                                           len(self.cfg["common"]["default_qpos"])))
        if n != cfg_n:
            raise RuntimeError(
                "joint count mismatch: robot reports %d, config says %d. "
                "Refusing to command joints with the wrong layout." % (n, cfg_n))
        self.low_cmd.cmd_type = LowCmdType.SERIAL
        self.low_cmd.motor_cmd = [MotorCmd() for _ in range(n)]
        stiff = self.cfg["prepare"]["stiffness"]
        damp = self.cfg["prepare"]["damping"]
        for i in range(n):
            self.low_cmd.motor_cmd[i].q = q[i]
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].kp = stiff[i]
            self.low_cmd.motor_cmd[i].kd = damp[i]
            self.low_cmd.motor_cmd[i].weight = 0.0
        return [self.low_cmd.motor_cmd[i].q for i in range(n)]

    def _publish_loop(self):
        dt = float(self.cfg["common"]["dt"])
        while self.running:
            self.pub.Write(self.low_cmd)
            time.sleep(dt)

    def start_publishing(self):
        self.publisher_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.publisher_thread.start()

    def cleanup(self):
        self.running = False
        if self.publisher_thread:
            self.publisher_thread.join(timeout=1.0)
        if self._entered_custom:
            print("[cleanup] requesting DAMPING")
            try:
                self.client.ChangeMode(RobotMode.kDamping)
            except Exception as exc:
                print("[cleanup] ChangeMode(kDamping) failed: %s" % exc)


MODE_ENUM = {
    "damping": RobotMode.kDamping,
    "prepare": RobotMode.kPrepare,
    "walking": RobotMode.kWalking,
    "custom": RobotMode.kCustom,
}


def watch_settle(probe, t0, timeout=6.0):
    """Seconds until the legs stop moving, and how far they travelled."""
    before = probe.legs()
    prev = before
    quiet = None
    settle = None
    peak = 0.0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.02)
        cur = probe.legs()
        if cur and prev:
            step = max(abs(a - b) for a, b in zip(cur, prev))
            if before:
                peak = max(peak, max(abs(a - b) for a, b in zip(cur, before)))
            if step < 5e-4:
                quiet = quiet or time.monotonic()
                if time.monotonic() - quiet > 0.3:
                    settle = time.monotonic() - t0 - 0.3
                    break
            else:
                quiet = None
        prev = cur
    return settle, peak


def run_sequence(probe, args, report):
    """Time every hop in a mode sequence, holding pose only while in CUSTOM."""
    seq = [m.strip() for m in args.sequence.split(",") if m.strip()]
    unknown = [m for m in seq if m not in MODE_ENUM]
    if unknown:
        print("unknown mode(s): %s (have %s)" % (unknown, list(MODE_ENUM)))
        return 2

    print("=== mode transition matrix ===")
    print("sequence: %s" % " -> ".join(seq))
    print("!! a human must be at the robot !!\n")
    print("%-22s %8s %9s %10s %10s %10s" %
          ("transition", "call[s]", "settle[s]", "move[rad]", "low[Hz]", "odom[Hz]"))
    print("-" * 74)

    rows = []
    publishing = False
    current = "as-found"

    for target in seq:
        # Only stream LowCmd for CUSTOM. In every other mode the SDK controller
        # owns the joints and our frames would fight it.
        if target == "custom" and not publishing:
            probe.build_hold_cmd()
            probe.start_publishing()
            publishing = True
            time.sleep(0.3)

        t0 = time.monotonic()
        rc = probe.client.ChangeMode(MODE_ENUM[target])
        t_call = time.monotonic() - t0
        if target == "custom":
            probe._entered_custom = True
        else:
            probe._entered_custom = False

        settle, moved = watch_settle(probe, t0)

        # Stop streaming once we have left CUSTOM.
        if publishing and target != "custom":
            probe.running = False
            if probe.publisher_thread:
                probe.publisher_thread.join(timeout=1.0)
            probe.running = True
            probe.publisher_thread = None
            publishing = False

        ta = time.monotonic(); time.sleep(args.hold); tb = time.monotonic()
        low_hz = probe.rate(probe.low_stamps, ta, tb)
        odom_hz = probe.rate(probe.odom_stamps, ta, tb)

        label = "%s -> %s" % (current, target)
        print("%-22s %8.3f %9s %10s %10.1f %10.1f" %
              (label, t_call,
               "%.2f" % settle if settle is not None else "TIMEOUT",
               "%.4f" % moved if moved is not None else "n/a",
               low_hz, odom_hz))
        rows.append({"from": current, "to": target, "rc": rc,
                     "call_sec": t_call, "settle_sec": settle,
                     "max_joint_move_rad": moved,
                     "low_hz": low_hz, "odom_hz": odom_hz})
        current = target

    report["transitions"] = rows
    with open("mode_matrix.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n=== VERDICT ===")
    for r in rows:
        if r["to"] == "custom":
            print("odometer_state in CUSTOM: %s (%.1f Hz)  -> imu_odom_mode should be %s"
                  % ("ALIVE" if r["odom_hz"] > 1.0 else "DEAD", r["odom_hz"],
                     '"off"/"auto"' if r["odom_hz"] > 1.0 else '"on"'))
    total = [r for r in rows if r["settle_sec"] is not None]
    if total:
        print("slowest hop: %.2fs settle (%s -> %s)"
              % max((r["settle_sec"], r["from"], r["to"]) for r in total))
    print("\nwrote mode_matrix.json")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--net", default="127.0.0.1")
    ap.add_argument("--hold", type=float, default=6.0,
                    help="seconds to observe in each mode")
    ap.add_argument("--return-mode", default="prepare",
                    choices=["prepare", "damping", "none"])
    ap.add_argument("--sequence", default="",
                    help="comma-separated modes to walk through, timing every "
                         "hop (e.g. prepare,walking,custom,prepare,damping). "
                         "Empty = the single custom round trip. LowCmd is only "
                         "published while in CUSTOM; in other modes the SDK owns "
                         "the joints and we just observe.")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    probe = Probe(cfg, args.net)
    signal.signal(signal.SIGINT, lambda *a: (probe.cleanup(), sys.exit(0)))

    report = {}
    try:
        # Wait for the channel to actually attach rather than assuming a fixed
        # delay is enough; discovery has taken over a second here.
        deadline = time.monotonic() + 10.0
        while probe.low_q is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if probe.low_q is None:
            print("!! no low_state after 10s -- robot stack is not running. aborting.")
            return 1
        print("low_state attached (%d samples buffered)" % len(probe.low_stamps))

        if args.sequence:
            return run_sequence(probe, args, report)

        # ---- baseline in the current (non-CUSTOM) mode ----
        t0 = time.monotonic(); time.sleep(3.0); t1 = time.monotonic()
        base = {"low_hz": probe.rate(probe.low_stamps, t0, t1),
                "odom_hz": probe.rate(probe.odom_stamps, t0, t1)}
        print("[before] low_state %.1f Hz   odometer_state %.1f Hz"
              % (base["low_hz"], base["odom_hz"]))

        # ---- hold current pose, then enter CUSTOM ----
        held = probe.build_hold_cmd()
        before_legs = probe.legs()
        probe.start_publishing()
        time.sleep(0.3)  # let a few frames land before the mode switch

        print("[custom] entering CUSTOM while holding current pose ...")
        t_call0 = time.monotonic()
        rc = probe.client.ChangeMode(RobotMode.kCustom)
        t_call = time.monotonic() - t_call0
        probe._entered_custom = True
        print("[custom] ChangeMode returned rc=%s in %.3f s" % (rc, t_call))

        # settle watch
        settle = None
        prev = probe.legs()
        quiet = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            time.sleep(0.05)
            cur = probe.legs()
            if cur and prev:
                d = max(abs(a - b) for a, b in zip(cur, prev))
                if d < 1e-3:
                    quiet = quiet or time.monotonic()
                    if time.monotonic() - quiet > 0.3:
                        settle = time.monotonic() - t_call0 - 0.3
                        break
                else:
                    quiet = None
            prev = cur
        after_legs = probe.legs()
        moved = (max(abs(a - b) for a, b in zip(after_legs, before_legs))
                 if after_legs and before_legs else None)
        print("[custom] legs settled in %s, max joint move %s"
              % ("%.2fs" % settle if settle is not None else "TIMEOUT",
                 "%.4f rad" % moved if moved is not None else "n/a"))

        # ---- the question: what still publishes in CUSTOM? ----
        t0 = time.monotonic(); time.sleep(args.hold); t1 = time.monotonic()
        incustom = {"low_hz": probe.rate(probe.low_stamps, t0, t1),
                    "odom_hz": probe.rate(probe.odom_stamps, t0, t1)}
        print("[in-custom] low_state %.1f Hz   odometer_state %.1f Hz"
              % (incustom["low_hz"], incustom["odom_hz"]))

        # ---- leave ----
        if args.return_mode != "none":
            mode = RobotMode.kPrepare if args.return_mode == "prepare" else RobotMode.kDamping
            print("[exit] returning to %s ..." % args.return_mode)
            t_back0 = time.monotonic()
            rc_back = probe.client.ChangeMode(mode)
            t_back = time.monotonic() - t_back0
            probe._entered_custom = False
            print("[exit] ChangeMode(%s) rc=%s in %.3f s" % (args.return_mode, rc_back, t_back))
            report["exit_call_sec"] = t_back
            time.sleep(1.5)
            t0 = time.monotonic(); time.sleep(2.0); t1 = time.monotonic()
            after = {"low_hz": probe.rate(probe.low_stamps, t0, t1),
                     "odom_hz": probe.rate(probe.odom_stamps, t0, t1)}
            print("[after]  low_state %.1f Hz   odometer_state %.1f Hz"
                  % (after["low_hz"], after["odom_hz"]))
            report["after"] = after

        report.update({"before": base, "in_custom": incustom,
                       "enter_call_sec": t_call, "settle_sec": settle,
                       "max_joint_move_rad": moved,
                       "held_pose_legs": held[11:23]})
    finally:
        probe.running = False
        if probe.publisher_thread:
            probe.publisher_thread.join(timeout=1.0)

    with open("custom_hold_probe.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n=== VERDICT ===")
    if "in_custom" in report:
        alive = report["in_custom"]["odom_hz"] > 1.0
        print("odometer_state in CUSTOM: %s (%.1f Hz)"
              % ("ALIVE" if alive else "DEAD", report["in_custom"]["odom_hz"]))
        print("  -> imu_odom_mode should be %s"
              % ("\"off\" or \"auto\"" if alive else "\"on\" (current default is correct)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
