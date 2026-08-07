"""Collect the posture set for the joint-zero estimator.  RUNS ON THE ROBOT.

⛔ READ THE SAFETY SECTION BEFORE RUNNING.  This drives the robot in CUSTOM
mode, where nothing closes the balance loop -- the policy is not running and the
only thing holding the robot up is the joint position servo plus the fact that
every commanded posture is statically stable in double support.

    python3 tools/collect_joint_zero.py --dry-run      # no mode change, no motion
    python3 tools/collect_joint_zero.py --out /tmp/zero_poses.json

SAFETY
======
* Two people, or one person plus a gantry/strap.  Someone has a hand near the
  robot the whole time.  `touch /tmp/zero_abort` exits to DAMPING immediately,
  and the loop checks it every control tick.
* The robot must START standing in kPrepare, feet flat, on a FLAT HARD floor.
  Carpet defeats the whole method -- the sole plane is the measurement
  reference, and a compliant floor lets each foot sink by a different,
  posture-dependent amount.
* Every posture in the default set keeps both feet flat and the CoM inside the
  support polygon.  The largest excursion is a 1.0 rad knee bend (a squat), the
  lateral leans are +-0.12 rad and the yaw twists +-0.15 rad.
* Motions between postures are ramped at `--rate` rad/s (default 0.25).  This
  matters: on 2026-08-07 a 0.8 rad/s move in this same unbalanced window
  tripped the firmware joint protection and dropped the robot to DAMPING
  (HANDOFF_DEPLOY_ENTRY_20260807.md).
* `--dry-run` builds and validates every command frame WITHOUT changing mode.
  Run it first, every time.  On 2026-08-05 a probe called ChangeMode(kCustom)
  before constructing its command struct, and the robot went to the zero
  posture -- knees straight -- and juddered against the floor.

WHAT IT RECORDS
===============
For each posture, after the ramp and a settle wait, it averages `--hold-s`
seconds of LowState (about 500 samples/s):

    q12   the 12 leg joint angles from motor_state_serial
    rpy   IMU roll and pitch

and writes them as JSON for `estimate_joint_zero.py --solve`.

Averaging is the whole reason for the hold: the estimator's noise gain is about
7x, so per-sample noise of 0.2 deg would put ~1.4 deg into delta, while 1 s of
averaging brings it to ~0.06 deg.  See `--self-test` PART 2.

⚠️ Values are copied INSIDE the SDK callback.  Reading msg.motor_state_serial
outside it gives a dangling reference -- on 2026-08-05 that produced 7.7e28
garbage, a MemoryError and a segfault in that order.

QUALITY GATES (a refused sample is better than a wrong delta)
=============================================================
Each posture is rejected, and the run aborted, if:
  * the IMU is still moving          (gyro rms above --quiet-gyro)
  * the joints are still moving      (dof_vel rms above --quiet-dofvel)
  * joint tracking is poor           (|q - target| above --track-tol, meaning
                                      the posture was not reached -- usually a
                                      foot tipping or hitting a limit)
  * the trunk tilt exceeds --max-tilt (the robot is not standing the way the
                                      constraint assumes)
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ABORT = "/tmp/zero_abort"


def run_collection(args):
    import numpy as np
    import yaml
    from booster_robotics_sdk_python import (
        ChannelFactory, B1LocoClient, B1LowCmdPublisher, B1LowStateSubscriber,
        LowCmd, RobotMode)

    deploy_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy")
    sys.path.insert(0, deploy_dir)
    from utils.command import init_Cmd_T1              # noqa

    from estimate_joint_zero import DEFAULT_POSTURES
    from kinematics_k1 import LEG_DOF_START

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(deploy_dir, cfg_path)
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    n = int(cfg["common"]["joint_cnt"])
    dt = float(cfg["common"]["dt"])
    q_default = np.array(cfg["common"]["default_qpos"], dtype=np.float32)
    # Use the PREPARE gains: this is the same unbalanced window as `b`..`r`,
    # and the RL gains (100/2) are far too soft to hold a posture open-loop.
    kp = np.array(cfg["prepare"]["stiffness"], dtype=np.float32)
    kd = np.array(cfg["prepare"]["damping"], dtype=np.float32)

    state = {"q": None, "dq": None, "rpy": None, "gyro": None}

    def on_low_state(msg):
        try:
            ms = msg.motor_state_serial
            if len(ms) < n:
                return
            # copy inside the callback -- see module docstring
            state["q"] = [float(ms[i].q) for i in range(n)]
            state["dq"] = [float(ms[i].dq) for i in range(n)]
            state["rpy"] = [float(v) for v in msg.imu_state.rpy]
            state["gyro"] = [float(v) for v in msg.imu_state.gyro]
        except Exception:
            pass

    if os.path.exists(ABORT):
        os.remove(ABORT)
    ChannelFactory.Instance().Init(0, args.net)
    sub = B1LowStateSubscriber(on_low_state)
    sub.InitChannel()
    pub = B1LowCmdPublisher()
    pub.InitChannel()
    client = B1LocoClient()
    client.Init()

    t0 = time.time()
    while state["q"] is None and time.time() - t0 < 5:
        time.sleep(0.05)
    if state["q"] is None:
        print("no LowState -- is the robot up?")
        os._exit(1)
    if len(state["q"]) != n:
        print("joint count %d != config %d" % (len(state["q"]), n))
        os._exit(1)

    cmd = LowCmd()
    init_Cmd_T1(cmd, n)
    if len(cmd.motor_cmd) < n:
        print("motor_cmd length %d < %d -- cannot publish"
              % (len(cmd.motor_cmd), n))
        os._exit(4)

    def send(q_target):
        for i in range(n):
            cmd.motor_cmd[i].q = float(q_target[i])
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0
            cmd.motor_cmd[i].kp = float(kp[i])
            cmd.motor_cmd[i].kd = float(kd[i])
        pub.Write(cmd)

    def full_target(leg12):
        t = q_default.copy()
        t[LEG_DOF_START:LEG_DOF_START + 12] = leg12
        return t

    # ---- validate the publish path BEFORE touching the mode ---------------
    q_now = np.array(state["q"], dtype=np.float32)
    try:
        send(q_now)
    except Exception as e:
        print("first publish failed:", type(e).__name__, e)
        os._exit(5)
    print("publish path OK (%d joints)" % n)
    print("current leg pose (deg): %s"
          % np.round(np.degrees(q_now[LEG_DOF_START:LEG_DOF_START + 12]), 1))
    print("IMU roll %.2f deg  pitch %.2f deg"
          % (math.degrees(state["rpy"][0]), math.degrees(state["rpy"][1])))
    print("%d postures, ramp %.2f rad/s, settle %.1f s, hold %.1f s"
          % (len(DEFAULT_POSTURES), args.rate, args.settle_s, args.hold_s))
    est = sum(1.0 for _ in DEFAULT_POSTURES) * (args.settle_s + args.hold_s + 2.0)
    print("estimated duration: %.0f s" % est)
    if args.dry_run:
        print()
        print("DRY RUN -- mode unchanged, robot did not move.")
        for name, leg in DEFAULT_POSTURES:
            t = full_target(leg)
            bad = [i for i in range(LEG_DOF_START, LEG_DOF_START + 12)
                   if abs(t[i]) > 1.6]
            print("  %-16s max |q| = %.3f rad %s"
                  % (name, max(abs(v) for v in leg), "SUSPECT %s" % bad if bad else ""))
        sys.stdout.flush()
        os._exit(0)

    if not args.yes:
        print()
        print("This will move the robot in CUSTOM mode. Type 'go' to continue: ")
        if sys.stdin.readline().strip() != "go":
            print("aborted")
            os._exit(0)

    print("entering CUSTOM, holding the CURRENT pose (not ramping to default)")
    client.ChangeMode(RobotMode.kCustom)
    time.sleep(1.2)
    cur = np.array(state["q"], dtype=np.float32)
    t_hold = time.time()
    while time.time() - t_hold < 1.5:
        if os.path.exists(ABORT):
            break
        send(cur)
        time.sleep(dt)

    def bail(why):
        print("ABORT: %s" % why)
        try:
            client.ChangeMode(RobotMode.kDamping)
        except Exception:
            pass
        sys.stdout.flush()
        os._exit(3)

    def ramp_to(target, rate):
        start = np.array(state["q"], dtype=np.float32).copy()
        dmax = float(np.max(np.abs(target - start)))
        T = max(dmax / max(rate, 1e-6), 0.5)
        t_s = time.time()
        while True:
            if os.path.exists(ABORT):
                bail("/tmp/zero_abort")
            u = (time.time() - t_s) / T
            if u >= 1.0:
                break
            # smoothstep: zero velocity at both ends
            a = u * u * (3.0 - 2.0 * u)
            send(start + a * (target - start))
            time.sleep(dt)
        return T

    poses = []
    for name, leg in DEFAULT_POSTURES:
        target = full_target(leg)
        T = ramp_to(target, args.rate)
        t_s = time.time()
        while time.time() - t_s < args.settle_s:
            if os.path.exists(ABORT):
                bail("/tmp/zero_abort")
            send(target)
            time.sleep(dt)

        acc_q = [0.0] * 12
        acc_rp = [0.0, 0.0]
        gyro2 = 0.0
        dv2 = 0.0
        cnt = 0
        tilt_max = 0.0
        t_s = time.time()
        while time.time() - t_s < args.hold_s:
            if os.path.exists(ABORT):
                bail("/tmp/zero_abort")
            send(target)
            q = state["q"]
            rpy = state["rpy"]
            gy = state["gyro"]
            dq = state["dq"]
            if q and rpy:
                for k in range(12):
                    acc_q[k] += q[LEG_DOF_START + k]
                    dv2 += dq[LEG_DOF_START + k] ** 2
                acc_rp[0] += rpy[0]
                acc_rp[1] += rpy[1]
                gyro2 += sum(g * g for g in gy)
                tilt_max = max(tilt_max,
                               math.hypot(rpy[0], rpy[1]))
                cnt += 1
            time.sleep(dt)

        if cnt < 50:
            bail("posture %s: only %d samples" % (name, cnt))
        q12 = [v / cnt for v in acc_q]
        rp = [acc_rp[0] / cnt, acc_rp[1] / cnt]
        gyro_rms = math.sqrt(gyro2 / cnt / 3)
        dofvel_rms = math.sqrt(dv2 / cnt / 12)
        track = max(abs(q12[k] - float(target[LEG_DOF_START + k]))
                    for k in range(12))

        flags = []
        if gyro_rms > args.quiet_gyro:
            flags.append("gyro %.3f > %.3f" % (gyro_rms, args.quiet_gyro))
        if dofvel_rms > args.quiet_dofvel:
            flags.append("dof_vel %.3f > %.3f" % (dofvel_rms, args.quiet_dofvel))
        if track > args.track_tol:
            flags.append("tracking %.3f rad > %.3f" % (track, args.track_tol))
        if tilt_max > args.max_tilt:
            flags.append("tilt %.1f deg > %.1f"
                         % (math.degrees(tilt_max), math.degrees(args.max_tilt)))

        print("  %-16s n=%4d  ramp %.1fs  track %.4f rad  gyro %.4f  "
              "dofvel %.4f  rp (%.2f, %.2f) deg  %s"
              % (name, cnt, T, track, gyro_rms, dofvel_rms,
                 math.degrees(rp[0]), math.degrees(rp[1]),
                 "REJECT: " + "; ".join(flags) if flags else "ok"))
        if flags:
            bail("posture %s failed its quality gate" % name)
        poses.append(dict(name=name, q12=q12, rpy=rp,
                          n_samples=cnt, track_rad=track,
                          gyro_rms=gyro_rms, dofvel_rms=dofvel_rms))

    print("returning to the nominal pose, then DAMPING")
    ramp_to(full_target(DEFAULT_POSTURES[0][1]), args.rate)
    try:
        client.ChangeMode(RobotMode.kDamping)
    except Exception as e:
        print("ChangeMode failed:", e)

    with open(args.out, "w") as fh:
        json.dump(dict(
            robot="K1", collected=time.strftime("%Y-%m-%dT%H:%M:%S"),
            config=cfg_path, hold_s=args.hold_s, settle_s=args.settle_s,
            poses=poses), fh, indent=2)
    print("wrote %d postures -> %s" % (len(poses), args.out))
    print()
    print("now:  python3 tools/estimate_joint_zero.py --solve %s \\" % args.out)
    print("          --emit-yaml deploy/configs/joint_zero.yaml")
    sys.stdout.flush()
    os._exit(0)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/Goal_Pose_E0.yaml")
    ap.add_argument("--net", default="127.0.0.1")
    ap.add_argument("--out", default="/tmp/zero_poses.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation")
    ap.add_argument("--rate", type=float, default=0.25,
                    help="ramp rate rad/s between postures")
    ap.add_argument("--settle-s", type=float, default=1.0)
    ap.add_argument("--hold-s", type=float, default=1.5)
    ap.add_argument("--quiet-gyro", type=float, default=0.05,
                    help="rad/s rms; above this the robot is still moving")
    ap.add_argument("--quiet-dofvel", type=float, default=0.10,
                    help="rad/s rms over the 12 leg joints")
    ap.add_argument("--track-tol", type=float, default=0.12,
                    help="rad; max |q_meas - target|. NOTE this is measured in "
                         "the BIASED frame, so a real delta shows up here -- "
                         "set it above the delta you expect to find.")
    ap.add_argument("--max-tilt", type=float, default=0.26,
                    help="rad (15 deg) trunk tilt ceiling")
    args = ap.parse_args()
    return run_collection(args)


if __name__ == "__main__":
    sys.exit(main())
