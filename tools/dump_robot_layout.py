#!/usr/bin/env python3
"""Dump the robot's actual joint layout as JSON. Run this ON the robot.

Exists because the SDK's own constants lie about this hardware: B1JointCnt is
23 and B1JointIndex places a waist at index 10 with legs at 11..22, but this K1
has no waist -- low_state carries 22 joints and the legs start at 10. A deploy
config written to the SDK map shifts every leg command by one joint, silently.

So the robot, not the SDK header and not the training config, is the authority
on the layout. check_policy_contract.py compares a deploy config against this
dump; install_policy.sh refuses to install when they disagree.

    python3 dump_robot_layout.py --out robot_layout.json

Read-only: subscribes to low_state, publishes nothing, changes no mode.
"""

import argparse
import json
import sys
import time

import booster_robotics_sdk_python as sdk


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", default="127.0.0.1")
    ap.add_argument("--out", default="robot_layout.json")
    ap.add_argument("--wait", type=float, default=3.0,
                    help="seconds to wait for a low_state sample")
    args = ap.parse_args()

    sdk.ChannelFactory.Instance().Init(0, args.net)

    # Copy the values out inside the callback and keep the newest sample. The
    # first message on the channel can arrive empty, and the message object is
    # only guaranteed valid for the duration of the callback.
    box = {}

    def on_low(m):
        try:
            ser = [float(j.q) for j in m.motor_state_serial]
            par = [float(j.q) for j in m.motor_state_parallel]
        except Exception:
            return
        if ser:
            box["serial"] = ser
            box["parallel"] = par

    sub = sdk.B1LowStateSubscriber(on_low)
    sub.InitChannel()

    deadline = time.time() + args.wait
    while "serial" not in box and time.time() < deadline:
        time.sleep(0.05)

    if "serial" not in box:
        print("no low_state within %.1fs -- is the robot powered and its motion "
              "stack running?" % args.wait, file=sys.stderr)
        return 1

    # Take a few more samples so a single noisy frame cannot decide the layout.
    time.sleep(0.3)
    serial = box["serial"]
    parallel = box["parallel"]
    n = len(serial)

    # The parallel-mechanism joints are the only ones where the serial and
    # parallel representations differ, which makes them a reliable fingerprint
    # of where the legs sit in the vector.
    parallel_idx = [
        i for i in range(min(n, len(parallel)))
        if abs(serial[i] - parallel[i]) > 1e-4
    ]

    layout = {
        "joint_cnt": n,
        "sdk_B1JointCnt": int(sdk.B1JointCnt),
        "parallel_mech_indexes": parallel_idx,
        "measured_q": [round(v, 5) for v in serial],
        "sdk_joint_index_map": {
            n_: int(getattr(sdk.B1JointIndex, n_).value)
            for n_ in dir(sdk.B1JointIndex)
            if not n_.startswith("_") and n_ not in ("name", "value")
        },
    }

    # Infer the leg start from the parallel indices: on this robot each leg ends
    # with two parallel (crank) joints, so the four of them are the last two of
    # each 6-joint leg block.
    if len(parallel_idx) == 4:
        left_crank_start = parallel_idx[0]
        layout["inferred_leg_dof_start"] = left_crank_start - 4
    else:
        layout["inferred_leg_dof_start"] = None

    with open(args.out, "w") as fh:
        json.dump(layout, fh, indent=2)

    print("joint_cnt              = %d   (SDK B1JointCnt = %d)"
          % (layout["joint_cnt"], layout["sdk_B1JointCnt"]))
    print("parallel_mech_indexes  = %s" % layout["parallel_mech_indexes"])
    print("inferred leg_dof_start = %s" % layout["inferred_leg_dof_start"])
    if layout["joint_cnt"] != layout["sdk_B1JointCnt"]:
        print("\nNOTE: the robot disagrees with the SDK constant. Trust the robot.")
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
