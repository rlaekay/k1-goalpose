#!/usr/bin/env python3
"""Measure what the robot actually does across modes. READ-ONLY by default.

Four things we cannot answer from the headers alone, and that decide how the
safety policy has to be built:

  1. Does rt/fall_down keep publishing in CUSTOM mode? If the SDK's fall
     detector stops when we take the joints, we have to detect falls ourselves.
  2. Does rt/odometer_state keep publishing in CUSTOM? The whole imu_odom
     fallback exists because we assume it does not. Nobody has checked.
  3. What does a mode change actually cost in wall-clock time, and how far do
     the joints move during it?
  4. Which modes are even reachable from the Python binding? (kSoccer is absent
     from the Python RobotMode enum.)

Default run only *observes*: it subscribes and reports rates in whatever mode
the robot is already in. Mode changes happen only with --modes, and never
without a human next to the robot.

    # observe only, no mode change
    python3 probe_modes.py --seconds 5

    # observe, then step through modes (SOMEONE MUST BE HOLDING/WATCHING)
    python3 probe_modes.py --modes prepare,custom --seconds 5

Run this on the robot.
"""

import argparse
import json
import statistics
import sys
import time

import booster_robotics_sdk_python as sdk

# LocoApiId values that the Python binding does not name but does accept via
# B1LocoApiId(int). Verified: B1LocoApiId(2008) constructs.
API_GET_UP = 2008
API_GET_UP_WITH_MODE = 2025
API_GET_MODE = 2017

MODES = {
    "damping": sdk.RobotMode.kDamping,
    "prepare": sdk.RobotMode.kPrepare,
    "walking": sdk.RobotMode.kWalking,
    "custom": sdk.RobotMode.kCustom,
}


class RateProbe:
    """Counts messages and records arrival times for one subscriber."""

    def __init__(self, name):
        self.name = name
        self.stamps = []
        self.last_payload = None

    def on_msg(self, msg):
        self.stamps.append(time.monotonic())
        self.last_payload = msg

    def window(self, t0, t1):
        n = sum(1 for s in self.stamps if t0 <= s <= t1)
        dur = max(1e-9, t1 - t0)
        return {"count": n, "hz": n / dur}


def observe(low_probe, odom_probe, seconds, label):
    t0 = time.monotonic()
    time.sleep(seconds)
    t1 = time.monotonic()
    low = low_probe.window(t0, t1)
    odom = odom_probe.window(t0, t1)
    print("  [%-10s] low_state %6.1f Hz (n=%d)   odometer_state %6.1f Hz (n=%d)"
          % (label, low["hz"], low["count"], odom["hz"], odom["count"]))
    return {"label": label, "low_state": low, "odometer_state": odom}


def leg_positions(low_state):
    if low_state is None:
        return None
    try:
        return [low_state.motor_state_serial[i].q for i in range(11, 23)]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", default="127.0.0.1")
    ap.add_argument("--seconds", type=float, default=5.0,
                    help="observation window per mode")
    ap.add_argument("--modes", default="",
                    help="comma-separated modes to step through "
                         "(damping,prepare,walking,custom). EMPTY = observe only, "
                         "no mode change. Only use with a human at the robot.")
    ap.add_argument("--out", default="mode_probe.json")
    args = ap.parse_args()

    sdk.ChannelFactory.Instance().Init(0, args.net)

    low_probe = RateProbe("low_state")
    odom_probe = RateProbe("odometer_state")

    low_sub = sdk.B1LowStateSubscriber(low_probe.on_msg)
    low_sub.InitChannel()
    odom_sub = sdk.B1OdometerStateSubscriber(odom_probe.on_msg)
    odom_sub.InitChannel()

    client = sdk.B1LocoClient()
    client.Init()

    time.sleep(1.0)  # let channels attach

    report = {"observations": [], "transitions": [], "api_probe": {}}

    print("=== baseline (no mode change) ===")
    report["observations"].append(
        observe(low_probe, odom_probe, args.seconds, "as-found"))

    if low_probe.last_payload is None:
        print("\n!! No low_state at all. The robot's motion stack is not running;\n"
              "   power it on before trusting anything below.")

    # Which named API ids can we even reach from Python?
    for name, val in (("kGetUp", API_GET_UP),
                      ("kGetUpWithMode", API_GET_UP_WITH_MODE),
                      ("kGetMode", API_GET_MODE)):
        try:
            sdk.B1LocoApiId(val)
            report["api_probe"][name] = "constructible"
        except Exception as exc:
            report["api_probe"][name] = "FAILED: %s" % exc
    report["api_probe"]["RobotMode_names"] = [
        m for m in dir(sdk.RobotMode) if m.startswith("k")]
    print("\n=== python API reachability ===")
    for k, v in report["api_probe"].items():
        print("  %-20s %s" % (k, v))

    requested = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not requested:
        print("\n(no --modes given: observation only, robot mode untouched)")
    else:
        print("\n=== mode transitions ===")
        print("!! Changing modes now. A human must be watching the robot. !!")
        for name in requested:
            if name not in MODES:
                print("  skip unknown mode %r" % name)
                continue
            before = leg_positions(low_probe.last_payload)
            t0 = time.monotonic()
            rc = client.ChangeMode(MODES[name])
            t_call = time.monotonic() - t0

            # Watch the legs settle: sample until they stop moving or we time out.
            settle = None
            prev = before
            quiet_since = None
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                time.sleep(0.05)
                cur = leg_positions(low_probe.last_payload)
                if cur is None or prev is None:
                    prev = cur
                    continue
                delta = max(abs(a - b) for a, b in zip(cur, prev))
                if delta < 1e-3:
                    quiet_since = quiet_since or time.monotonic()
                    if time.monotonic() - quiet_since > 0.3:
                        settle = time.monotonic() - t0 - 0.3
                        break
                else:
                    quiet_since = None
                prev = cur

            after = leg_positions(low_probe.last_payload)
            moved = None
            if before and after:
                moved = max(abs(a - b) for a, b in zip(after, before))

            print("  -> %-8s rc=%s  ChangeMode returned in %.3fs  "
                  "legs settled in %s  max joint move %s"
                  % (name, rc, t_call,
                     ("%.2fs" % settle) if settle is not None else "timeout",
                     ("%.4f rad" % moved) if moved is not None else "n/a"))

            report["transitions"].append({
                "mode": name, "rc": rc, "change_mode_call_sec": t_call,
                "settle_sec": settle, "max_joint_move_rad": moved,
            })
            report["observations"].append(
                observe(low_probe, odom_probe, args.seconds, "in-" + name))

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("\nwrote %s" % args.out)

    # The two questions this exists to answer, called out explicitly.
    print("\n=== verdict ===")
    for obs in report["observations"]:
        tag = obs["label"]
        odom_alive = obs["odometer_state"]["hz"] > 1.0
        low_alive = obs["low_state"]["hz"] > 1.0
        print("  %-12s low_state=%-7s odometer_state=%s"
              % (tag, "ALIVE" if low_alive else "DEAD",
                 "ALIVE" if odom_alive else "DEAD"))
    print("\nNote: rt/fall_down has no Python binding in this SDK build, so its\n"
          "liveness must be read from ROS: ros2 topic hz /fall_down_recovery_state")
    return 0


if __name__ == "__main__":
    sys.exit(main())
