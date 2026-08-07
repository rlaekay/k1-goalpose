"""K1 joint-zero (delta) estimator -- full form.

WHAT THIS SOLVES
================
K1's joint zeros are set by eye plus gravity, with no pins or jigs.  The only
acceptance test is "does not fall over in stand mode", which passes for a wide
band of physical zero offsets.  So delta drifts session to session, and the
policy -- Linear(54->256->128->128->12), no memory -- has no way to observe it.

Measure delta at startup and subtract it, and an RL robustness problem becomes a
calibration problem.

SIGN CONVENTION (matches training exactly -- envs/K1/goal_pose.py:420-437)
--------------------------------------------------------------------------
    q_meas = q_true + delta

That file derives, for encoder-zero drift b on joint j:

    measured        q_meas = q + b
    policy obs      q_meas - q_default
    onboard PD      tau = kp * (target_cmd - q_meas)
    equilibrium     q_eq = target_cmd - b

and states the exact sim equivalence is joint_encoder_bias = +b together with
joint_target_offset = -b.  So `delta` here IS `joint_encoder_bias`, and the two
deploy-side corrections are:

    observation:  q_corrected = q_meas - delta
    command:      publish       target  + delta

Apply BOTH or you have traded one bias for another.

THE MEASUREMENT
===============
Robot stands in double support on flat ground.  For each held posture k:

  * both soles are flat on the floor          -> each sole normal || gravity
  * the floor is one plane                    -> both soles at the same height
  * the feet do not slide                     -> |p_L - p_R| and the angle
                                                 between (p_L - p_R) and the
                                                 foot's forward axis are the
                                                 same at every posture

Gravity in the trunk frame comes from the IMU (the same vector deploy feeds the
policy as obs[0:3]).  No fixture, no external tracker, no torque model -- the
floor is the reference, which is Yamane's (US8805584) and iCub's idea and the
"double support constraint" of the whole-body self-calibration literature.

Residuals per posture (8 raw, rank 7):
    r1..3   n_L x g_hat                 left sole flat            (rank 2)
    r4..6   n_R x g_hat                 right sole flat           (rank 2)
    r7      (p_L - p_R) . g_hat         feet coplanar             (1)
    r8      |p_L - p_R| - D             stance width constant     (1)
    r9      angle(v, fwd_L) - Phi       feet not rotating         (1)

Unknowns: delta (12), D (1), Phi (1), and optionally the IMU roll/pitch mounting
bias (2).  D and Phi are nuisance parameters -- they absorb "where the feet
happen to be" so that no fixture is needed.

WHAT IS OBSERVABLE, AND FROM WHAT
=================================
Run `--observability` for the numbers on your own posture set.  Measured on the
default 12-posture set (2026-08-08), sigma_min / sigma_max of the delta block
with the nuisance parameters projected out:

    constraints used                          sigma_min   sigma_min/max
    feet flat only                            0.0         0.0        <- singular
    + feet coplanar                           3.28e-2     5.48e-3
    + stance width & angle held (full set)    6.62e-2     1.10e-2

Read that as a chain of causes:

  * Foot ORIENTATION alone can never split a leg's three pitch offsets, because
    hip_pitch, knee and ankle_pitch all rotate about +y and every joint origin
    rpy in this chain is exactly zero -- so the sole's pitch is exactly
    (q_hip_pitch + q_knee + q_ankle_pitch) and only that sum is visible.  With
    "flat" as the only constraint the weakest mode is precisely knee traded
    against ankle_pitch, at sigma = 0.  Likewise the roll-axis offsets
    (hip_roll, hip_yaw, ankle_roll) collapse onto one direction when hip_roll
    and hip_yaw are zero.
  * What splits them is HEIGHT: each of the three pitch joints moves the sole
    up by a different lever arm, so once the knee angle differs between
    postures, coplanarity separates them.  This is why the posture set must
    contain real squat depth, not just tilts.
  * The stance constraints (feet planted -> constant width and angle) roughly
    double sigma_min again.

Posture diversity is not a nicety here, it is the entire mechanism.  From
`--observability`, sigma_min against the number of postures:

    1-3 postures (pitch only)      0            singular
    4 postures  (+ deep squat)     1.8e-8       still singular in practice
    5-6         (+ lateral lean)   4-6e-5       barely
    7           (+ hip YAW twist)  3.1e-2       <- the jump is here
    12 (full set)                  7.1e-2       condition number 86

The one thing that is genuinely NOT observable:

  * IMU roll/pitch mounting bias is EXACTLY degenerate with a common-mode leg
    tilt on both legs -- rotating the measured gravity by beta and tilting both
    legs by -beta produce identical residuals at every posture.  So `--imu-bias`
    is refused.  Calibrate the IMU separately against a levelled trunk, or
    accept that delta is expressed in the IMU's frame (which, for a policy that
    also consumes that same IMU as obs[0:3], is arguably the frame you want).

ASSETS
======
See tools/kinematics_k1.py.  Short version: the robot's Hip_Pitch sits 15 mm
lower than every training asset says, and the sole plane is 2.9 mm lower than
the collision box every simulator stands on.  This tool refuses to run on the
training assets.

USAGE
=====
    # 1. prove the estimator works before trusting it (no robot needed)
    python3 tools/estimate_joint_zero.py --self-test

    # 2. design/score the posture set (no robot needed)
    python3 tools/estimate_joint_zero.py --observability

    # 3. on the robot: collect (needs booster SDK; see R7 in
    #    DEPLOY_REQUESTS_FROM_TRAINING.md for the safety procedure)
    python3 tools/estimate_joint_zero.py --collect --out /tmp/zero_poses.json

    # 4. solve, anywhere
    python3 tools/estimate_joint_zero.py --solve /tmp/zero_poses.json \
            --emit-yaml deploy/configs/joint_zero.yaml

    # explain the 2026-08-05 FK failure (-34 deg trunk pitch)
    python3 tools/estimate_joint_zero.py --diagnose-fk
"""

import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kinematics_k1 import (                                    # noqa: E402
    LEG_INDEX, LEG_JOINTS, LEG_DOF_START, SOLE_MESH_Z, SOLE_BOX_Z,
    cross, dot, find_urdf, gravity_in_body, jacobi_eigh, load_leg_chain,
    norm, solve_damped, solve_spd, PARALLEL_MECH_INDEXES,
)

REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

N_DELTA = 12
DEG = 180.0 / math.pi

# Parameter vector layout: [delta(12), D, Phi] (+ [imu_roll, imu_pitch])
IDX_D = 12
IDX_PHI = 13
IDX_IMU = 14


# ===========================================================================
# posture design
# ===========================================================================
# Each posture is the 12 leg joint TARGETS, ordered
#   [L hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll,
#    R hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]
#
# Design rules, in order of how much they matter:
#
# 1. Every posture must keep BOTH feet flat on the floor, or the constraint the
#    whole method rests on is false.  For a pitch change that means
#    hip_pitch + knee + ankle_pitch stays constant (the foot plate angle); for a
#    roll change it means hip_roll + ankle_roll stays constant.
# 2. The knee must actually move between postures.  Foot ORIENTATION only ever
#    sees the sum (hip_pitch + knee + ankle_pitch); what separates the three is
#    the sole HEIGHT, whose sensitivity to each is a different lever arm, and
#    those lever arms only differ once the knee angle differs.
# 3. Hip roll and hip yaw must be non-zero somewhere.  At hip_roll = hip_yaw = 0
#    the roll-axis offsets (hip_roll, hip_yaw, ankle_roll) collapse onto one
#    direction.
# 4. Asymmetric postures (left != right) are what separates left from right.
# 5. Stay inside the joint limits AND inside the support polygon.  Ankle roll
#    hard-stops at +-0.345 rad in the URDF; the audit of 2026-08-07 measured the
#    real robot outside that (-0.578..+0.537), which is unresolved -- so this
#    set stays well inside +-0.15 rad and does not probe it.
#
# The default set below is scored by --observability.  Do not edit it without
# re-running that.
DEFAULT_POSTURES = [
    # name,            L: hp    hr    hy    kn    ap    ar     R: hp    hr    hy    kn    ap    ar
    ("nominal",        [-0.20, 0.00, 0.00, 0.40, -0.25, 0.00, -0.20, 0.00, 0.00, 0.40, -0.25, 0.00]),
    ("squat_shallow",  [-0.10, 0.00, 0.00, 0.20, -0.10, 0.00, -0.10, 0.00, 0.00, 0.20, -0.10, 0.00]),
    ("squat_deep",     [-0.35, 0.00, 0.00, 0.70, -0.35, 0.00, -0.35, 0.00, 0.00, 0.70, -0.35, 0.00]),
    ("squat_deeper",   [-0.50, 0.00, 0.00, 1.00, -0.50, 0.00, -0.50, 0.00, 0.00, 1.00, -0.50, 0.00]),
    ("lean_left",      [-0.20, 0.12, 0.00, 0.40, -0.25, -0.12, -0.20, 0.12, 0.00, 0.40, -0.25, -0.12]),
    ("lean_right",     [-0.20, -0.12, 0.00, 0.40, -0.25, 0.12, -0.20, -0.12, 0.00, 0.40, -0.25, 0.12]),
    ("twist_out",      [-0.20, 0.00, 0.15, 0.40, -0.25, 0.00, -0.20, 0.00, -0.15, 0.40, -0.25, 0.00]),
    ("twist_in",       [-0.20, 0.00, -0.15, 0.40, -0.25, 0.00, -0.20, 0.00, 0.15, 0.40, -0.25, 0.00]),
    ("asym_knee",      [-0.35, 0.00, 0.00, 0.70, -0.35, 0.00, -0.12, 0.00, 0.00, 0.24, -0.12, 0.00]),
    ("asym_knee_rev",  [-0.12, 0.00, 0.00, 0.24, -0.12, 0.00, -0.35, 0.00, 0.00, 0.70, -0.35, 0.00]),
    ("asym_roll",      [-0.20, 0.10, 0.00, 0.40, -0.25, -0.10, -0.20, 0.00, 0.00, 0.40, -0.25, 0.00]),
    ("asym_yaw",       [-0.25, 0.00, 0.12, 0.50, -0.25, 0.00, -0.25, 0.06, -0.06, 0.50, -0.25, -0.06]),
]


# ===========================================================================
# forward model
# ===========================================================================
def leg_q(q22, side):
    return [q22[i] for i in LEG_INDEX[side]]


def pose_features(chains, q12_true, g_hat):
    """Everything a posture contributes, given TRUE joint angles."""
    RL, pL = chains["left"].fk(q12_true[:6])
    RR, pR = chains["right"].fk(q12_true[6:])
    nL = chains["left"].sole_normal(RL)
    nR = chains["right"].sole_normal(RR)
    fL = chains["left"].sole_forward(RL)
    v = [pL[i] - pR[i] for i in range(3)]
    return dict(nL=nL, nR=nR, fL=fL, v=v, pL=pL, pR=pR, g=g_hat)


def residuals(chains, poses, params, use_imu_bias=False):
    """Stacked residual vector over all postures.

    poses: list of dicts with keys q12 (measured, 12-vector) and rpy (roll,
    pitch) as measured by the IMU.
    """
    delta = params[:N_DELTA]
    D = params[IDX_D]
    Phi = params[IDX_PHI]
    br, bp = (params[IDX_IMU], params[IDX_IMU + 1]) if use_imu_bias else (0.0, 0.0)

    out = []
    for pose in poses:
        q_true = [pose["q12"][i] - delta[i] for i in range(N_DELTA)]
        g = gravity_in_body(pose["rpy"][0] - br, pose["rpy"][1] - bp)
        f = pose_features(chains, q_true, g)

        out.extend(cross(f["nL"], f["g"]))          # 3, rank 2
        out.extend(cross(f["nR"], f["g"]))          # 3, rank 2
        out.append(dot(f["v"], f["g"]))             # coplanar
        out.append(norm(f["v"]) - D)                # stance width

        # in-plane angle between the stance vector and the left toe direction
        vp = [f["v"][i] - dot(f["v"], f["g"]) * f["g"][i] for i in range(3)]
        fp = [f["fL"][i] - dot(f["fL"], f["g"]) * f["g"][i] for i in range(3)]
        nv, nf = norm(vp), norm(fp)
        if nv < 1e-9 or nf < 1e-9:
            out.append(0.0)
        else:
            c = max(-1.0, min(1.0, dot(vp, fp) / (nv * nf)))
            s = dot(cross(fp, vp), f["g"]) / (nv * nf)
            out.append(math.atan2(s, c) - Phi)
    return out


def jacobian(chains, poses, params, use_imu_bias=False, eps=1e-6):
    """Central-difference Jacobian.  16 columns max -- exact-enough and it can
    never disagree with the residual function, which an analytic Jacobian can."""
    n = len(params)
    cols = []
    for j in range(n):
        p1 = list(params)
        p2 = list(params)
        p1[j] += eps
        p2[j] -= eps
        r1 = residuals(chains, poses, p1, use_imu_bias)
        r2 = residuals(chains, poses, p2, use_imu_bias)
        cols.append([(a - b) / (2 * eps) for a, b in zip(r1, r2)])
    m = len(cols[0])
    return [[cols[j][i] for j in range(n)] for i in range(m)]


def init_params(chains, poses, use_imu_bias=False):
    """Seed delta = 0 and set D/Phi from the first posture so they start on the
    manifold instead of dragging the first few iterations."""
    p = [0.0] * (IDX_IMU + (2 if use_imu_bias else 0))
    pose = poses[0]
    g = gravity_in_body(pose["rpy"][0], pose["rpy"][1])
    f = pose_features(chains, pose["q12"], g)
    p[IDX_D] = norm(f["v"])
    vp = [f["v"][i] - dot(f["v"], g) * g[i] for i in range(3)]
    fp = [f["fL"][i] - dot(f["fL"], g) * g[i] for i in range(3)]
    nv, nf = norm(vp), norm(fp)
    if nv > 1e-9 and nf > 1e-9:
        c = max(-1.0, min(1.0, dot(vp, fp) / (nv * nf)))
        s = dot(cross(fp, vp), g) / (nv * nf)
        p[IDX_PHI] = math.atan2(s, c)
    return p


def solve(chains, poses, use_imu_bias=False, iters=60, verbose=False,
          prior_weight=0.0):
    """Levenberg-Marquardt.

    prior_weight > 0 adds a Tikhonov pull of delta toward 0.  That is not
    cosmetic: the nullspace is real, and without a prior the solver wanders
    along it to whatever the numerics prefer.  A prior makes the answer the
    minimum-norm delta consistent with the data, which is the honest choice --
    it puts zero into the directions the measurement genuinely cannot see.
    """
    params = init_params(chains, poses, use_imu_bias)
    lam = 1e-3
    r = residuals(chains, poses, params, use_imu_bias)
    cost = dot(r, r) + prior_weight * sum(x * x for x in params[:N_DELTA])
    for it in range(iters):
        J = jacobian(chains, poses, params, use_imu_bias)
        n = len(params)
        JtJ = [[sum(J[i][a] * J[i][b] for i in range(len(J)))
                for b in range(n)] for a in range(n)]
        Jtr = [sum(J[i][a] * r[i] for i in range(len(J))) for a in range(n)]
        if prior_weight > 0:
            for a in range(N_DELTA):
                JtJ[a][a] += prior_weight
                Jtr[a] += prior_weight * params[a]
        step = solve_damped(JtJ, Jtr, lam)
        cand = [params[i] + step[i] for i in range(n)]
        rc = residuals(chains, poses, cand, use_imu_bias)
        cc = dot(rc, rc) + prior_weight * sum(x * x for x in cand[:N_DELTA])
        if cc < cost:
            params, r, cost = cand, rc, cc
            lam = max(lam * 0.5, 1e-9)
        else:
            lam *= 4.0
            if lam > 1e8:
                break
        if verbose:
            print("  it%3d  cost=%.6e  lam=%.1e" % (it, cost, lam))
        if max(abs(s) for s in step) < 1e-12:
            break
    return params, cost, r


# ===========================================================================
# observability
# ===========================================================================
DELTA_NAMES = (["L_" + n for n in ("hip_pitch", "hip_roll", "hip_yaw",
                                   "knee", "ankle_pitch", "ankle_roll")] +
               ["R_" + n for n in ("hip_pitch", "hip_roll", "hip_yaw",
                                   "knee", "ankle_pitch", "ankle_roll")])


def observability(chains, poses, use_imu_bias=False):
    """Singular spectrum of the delta block of the Jacobian.

    The nuisance parameters (D, Phi, IMU bias) are projected OUT rather than
    just ignored -- otherwise a direction that is only "observable" because it
    can be traded against stance width would be counted as observable.
    """
    params = init_params(chains, poses, use_imu_bias)
    J = jacobian(chains, poses, params, use_imu_bias)
    m = len(J)
    n = len(params)
    nuis = list(range(N_DELTA, n))

    # project the nuisance columns out of the delta columns (Gram-Schmidt)
    Jn = [[J[i][c] for c in nuis] for i in range(m)]
    basis = []
    for c in range(len(nuis)):
        v = [Jn[i][c] for i in range(m)]
        for b in basis:
            d = dot(v, b)
            v = [v[i] - d * b[i] for i in range(m)]
        nv = norm(v)
        if nv > 1e-10:
            basis.append([x / nv for x in v])

    Jd = []
    for i in range(m):
        Jd.append([J[i][c] for c in range(N_DELTA)])
    for c in range(N_DELTA):
        col = [Jd[i][c] for i in range(m)]
        for b in basis:
            d = dot(col, b)
            col = [col[i] - d * b[i] for i in range(m)]
        for i in range(m):
            Jd[i][c] = col[i]

    JtJ = [[sum(Jd[i][a] * Jd[i][b] for i in range(m))
            for b in range(N_DELTA)] for a in range(N_DELTA)]
    vals, vecs = jacobi_eigh(JtJ)
    sing = [math.sqrt(max(v, 0.0)) for v in vals]
    return sing, vecs


def print_observability(sing, vecs, label=""):
    smax = sing[0] if sing else 0.0
    print("  singular values of d(residual)/d(delta), nuisances projected out%s"
          % (" -- " + label if label else ""))
    print("  %-4s %12s %10s   %s" % ("#", "sigma", "sigma/max", "dominant delta components"))
    for k in range(N_DELTA):
        ratio = sing[k] / smax if smax else 0.0
        comp = sorted(range(N_DELTA), key=lambda i: -abs(vecs[k][i]))[:4]
        desc = "  ".join("%s %+.2f" % (DELTA_NAMES[i], vecs[k][i])
                         for i in comp if abs(vecs[k][i]) > 0.15)
        flag = ""
        if ratio < 1e-6:
            flag = "  <- UNOBSERVABLE"
        elif ratio < 1e-3:
            flag = "  <- barely observable"
        print("  %-4d %12.4e %10.2e   %s%s" % (k, sing[k], ratio, desc, flag))
    print()
    print("  condition number (sigma_max / sigma_min) = %.3e"
          % (smax / sing[-1] if sing[-1] > 0 else float("inf")))
    return smax / sing[-1] if sing[-1] > 0 else float("inf")


# ===========================================================================
# synthetic data (self-test)
# ===========================================================================
def stance_features(chains, q12, roll, pitch):
    """(stance width D, in-plane angle Phi) for a posture -- the two quantities
    that are constant while the feet stay planted."""
    g = gravity_in_body(roll, pitch)
    f = pose_features(chains, q12, g)
    D = norm(f["v"])
    vp = [f["v"][i] - dot(f["v"], g) * g[i] for i in range(3)]
    fp = [f["fL"][i] - dot(f["fL"], g) * g[i] for i in range(3)]
    nv, nf = norm(vp), norm(fp)
    if nv < 1e-9 or nf < 1e-9:
        return D, 0.0
    c = max(-1.0, min(1.0, dot(vp, fp) / (nv * nf)))
    s = dot(cross(fp, vp), g) / (nv * nf)
    return D, math.atan2(s, c)


def settle_posture(chains, target12, delta, rng, max_iter=80,
                   stance=None):
    """Find TRUE joint angles near `target12` that put both feet flat on one
    plane -- i.e. simulate what the floor does to a commanded posture.

    The commanded posture is generally NOT loop-consistent (rotations about
    different axes do not commute, so hip_roll +x and ankle_roll -x do not
    cancel).  On the robot the floor resolves that: the feet stay flat, the
    trunk tilts, and the joints settle wherever the PD and the contact forces
    put them.  Here the same thing is done with task-priority IK --

        primary   : drive the double-support constraints to zero (exactly)
        secondary : among all solutions, stay as close to the command as
                    possible, in the nullspace of the primary task

    A soft penalty will NOT do.  The first version of this used one and it
    silently returned postures whose constraint residual was ~1e-2, i.e. feet
    that were not actually flat.  Eight of twelve postures were being dropped
    and the self-test was quietly running on four.  Exact constraint
    satisfaction is the whole point: the solver under test assumes r = 0.

    Returns (q12_true, roll, pitch) or None if it will not converge.
    """
    def resid(x):
        g = gravity_in_body(x[12], x[13])
        f = pose_features(chains, x[:12], g)
        out = list(cross(f["nL"], f["g"])) + list(cross(f["nR"], f["g"]))
        out.append(dot(f["v"], f["g"]))
        if stance is not None:
            # The feet are stuck to the floor by friction, so the stance width
            # and the foot-to-stance angle CANNOT change between postures.
            # Omitting these two was the first bug in this generator: the
            # synthetic robot slid its feet around, which violates exactly the
            # constraint the solver assumes, and the self-test then failed
            # identically for every planted delta -- including delta = 0.
            D, Phi = stance_features(chains, x[:12], x[12], x[13])
            out.append(D - stance[0])
            ang = Phi - stance[1]
            out.append(math.atan2(math.sin(ang), math.cos(ang)))
        return out

    n = 14
    x = list(target12) + [0.0, 0.0]
    for _ in range(max_iter):
        r = resid(x)
        if max(abs(v) for v in r) < 1e-13:
            break
        cols = []
        for j in range(n):
            xp = list(x)
            xp[j] += 1e-7
            r2 = resid(xp)
            cols.append([(a - b) / 1e-7 for a, b in zip(r2, r)])
        m = len(r)
        J = [[cols[j][i] for j in range(n)] for i in range(m)]

        # damped pseudo-inverse via the m x m normal matrix (m = 7, rank 5)
        JJt = [[sum(J[a][k] * J[b][k] for k in range(n)) for b in range(m)]
               for a in range(m)]
        for a in range(m):
            JJt[a][a] += 1e-10
        y = solve_spd(JJt, [-v for v in r])
        primary = [sum(J[a][j] * y[a] for a in range(m)) for j in range(n)]

        # secondary: pull the joints back toward the command, in the nullspace
        e = [(target12[j] - x[j]) if j < 12 else 0.0 for j in range(n)]
        Je = [sum(J[a][j] * e[j] for j in range(n)) for a in range(m)]
        z = solve_spd(JJt, Je)
        proj = [e[j] - sum(J[a][j] * z[a] for a in range(m)) for j in range(n)]

        x = [x[j] + primary[j] + 0.5 * proj[j] for j in range(n)]

    r = resid(x)
    if max(abs(v) for v in r) > 1e-9:
        return None
    return x[:12], x[12], x[13]


def make_synthetic(chains, postures, delta, noise_rad=0.0, imu_noise=0.0,
                   seed=0):
    """Synthesise what the robot would report for each commanded posture.

    Two stages, because the stance is set once and then held:
      1. settle the FIRST posture with feet flat + coplanar only.  Whatever
         stance width and foot angle come out of that is where the operator
         happened to put the feet.
      2. settle every posture (including the first) with that stance FROZEN --
         which is what friction does on a real floor.

    Then read q_meas = q_true + delta and the IMU, and add sensor noise.  The
    solver never sees q_true, and it reaches the answer by a different route
    (given q_meas, find delta), so recovering the planted delta is a real test.
    """
    rng = random.Random(seed)
    first = settle_posture(chains, postures[0][1], delta, rng)
    if first is None:
        raise RuntimeError("the reference posture does not settle")
    stance = stance_features(chains, first[0], first[1], first[2])

    poses = []
    for name, target in postures:
        got = settle_posture(chains, target, delta, rng, stance=stance)
        if got is None:
            continue
        q_true, roll, pitch = got
        q_meas = [q_true[i] + delta[i] + rng.gauss(0, noise_rad)
                  for i in range(12)]
        poses.append(dict(
            name=name,
            q12=q_meas,
            rpy=[roll + rng.gauss(0, imu_noise), pitch + rng.gauss(0, imu_noise)],
        ))
    return poses


# ===========================================================================
# subcommands
# ===========================================================================
def cmd_observability(args, chains):
    postures = DEFAULT_POSTURES
    print("=" * 78)
    print("OBSERVABILITY of the default posture set (%d postures)" % len(postures))
    print("=" * 78)
    poses = make_synthetic(chains, postures, [0.0] * 12, seed=1)
    print("postures that settle onto the double-support manifold: %d/%d"
          % (len(poses), len(postures)))
    print()
    sing, vecs = observability(chains, poses)
    cond = print_observability(sing, vecs)

    print()
    print("=" * 78)
    print("HOW THE SPECTRUM GROWS WITH THE POSTURE SET")
    print("=" * 78)
    print("  %-28s %6s %12s %12s %12s"
          % ("posture set", "n", "sigma_min", "sigma_max", "cond"))
    for k in range(1, len(postures) + 1):
        sub = poses[:k]
        if len(sub) < 1:
            continue
        s, _ = observability(chains, sub)
        c = s[0] / s[-1] if s[-1] > 0 else float("inf")
        print("  %-28s %6d %12.4e %12.4e %12.3e"
              % ("+" + postures[k - 1][0], len(sub), s[-1], s[0], c))
    print()
    print("  Read it this way: sigma_min is how strongly the WEAKEST direction")
    print("  of delta shows up in the measurement.  A direction with sigma_min")
    print("  near zero is one the floor cannot see no matter how long you")
    print("  measure, and the solver will return 0 there because of the prior.")
    return 0


def cmd_self_test(args, chains):
    print("=" * 78)
    print("SELF-TEST: plant a known delta, recover it")
    print("=" * 78)
    print("If this does not pass, nothing else in this file means anything.")
    print()
    rng = random.Random(args.seed)
    cases = [
        ("zero", [0.0] * 12),
        ("iid 1 deg", [rng.gauss(0, 1.0) / DEG for _ in range(12)]),
        ("iid 3 deg", [rng.gauss(0, 3.0) / DEG for _ in range(12)]),
        ("single joint: L_knee +4 deg",
         [0, 0, 0, 4.0 / DEG, 0, 0, 0, 0, 0, 0, 0, 0]),
        ("anti-mirror hip_roll +-5 deg",
         [0, 5.0 / DEG, 0, 0, 0, 0, 0, -5.0 / DEG, 0, 0, 0, 0]),
        ("leg_common: whole left leg +2 deg",
         [2.0 / DEG] * 6 + [0.0] * 6),
    ]
    print("-" * 78)
    print("PART 1 -- correctness: noiseless recovery")
    print("-" * 78)
    print("  %-34s %10s %10s %10s"
          % ("planted delta", "RMS err", "max err", "verdict"))
    worst_overall = 0.0
    for label, d in cases:
        poses = make_synthetic(chains, DEFAULT_POSTURES, d, seed=7)
        est, cost, r = solve(chains, poses, prior_weight=args.prior)
        err = [(est[i] - d[i]) * DEG for i in range(12)]
        rms = math.sqrt(sum(e * e for e in err) / 12)
        mx = max(abs(e) for e in err)
        worst_overall = max(worst_overall, mx)
        print("  %-34s %10.4f %10.4f %10s"
              % (label, rms, mx, "PASS" if mx < args.tol_deg else "FAIL"))
        if args.verbose:
            for i in range(12):
                print("       %-14s planted %+7.3f  est %+7.3f  err %+7.3f deg"
                      % (DELTA_NAMES[i], d[i] * DEG, est[i] * DEG, err[i]))
    print()

    print("-" * 78)
    print("PART 2 -- noise gain: how sensor noise propagates into delta")
    print("-" * 78)
    print("  The Jacobian is ill-conditioned by construction (see")
    print("  --observability), so this number, not the noiseless error, is what")
    print("  decides whether the measurement is worth taking.")
    print()
    print("  %-24s %12s %12s %10s"
          % ("per-sample noise (deg)", "delta RMS", "delta max", "gain"))
    d = cases[2][1]                       # iid 3 deg, a representative delta
    gains = []
    for s_deg in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3):
        s = s_deg / DEG
        errs = []
        for trial in range(3):
            poses = make_synthetic(chains, DEFAULT_POSTURES, d,
                                   noise_rad=s, imu_noise=s, seed=100 + trial)
            est, _, _ = solve(chains, poses, prior_weight=args.prior)
            errs.extend((est[i] - d[i]) * DEG for i in range(12))
        rms = math.sqrt(sum(e * e for e in errs) / len(errs))
        mx = max(abs(e) for e in errs)
        gains.append(rms / s_deg)
        print("  %-24.3f %12.4f %12.4f %10.1f" % (s_deg, rms, mx, rms / s_deg))
    gain = sum(gains) / len(gains)
    print()
    print("  mean noise gain ~ %.0fx.  So for delta good to %.2f deg you need"
          % (gain, args.tol_deg * 10))
    need = args.tol_deg * 10 / gain
    print("  effective per-sample noise below %.4f deg." % need)
    print()
    print("  That is reached by AVERAGING, not by better sensors.  LowState is")
    print("  published at 499.2 Hz (measured, ibatch 8-41), so holding a posture")
    print("  for T seconds gives 499*T samples and divides zero-mean noise by")
    print("  sqrt(499*T):")
    for T in (0.5, 1.0, 2.0, 3.0):
        n = 499 * T
        for raw in (0.05, 0.2):
            eff = raw / math.sqrt(n)
            if raw == 0.05:
                line = "    hold %.1f s (n=%4d):  raw 0.05 deg -> %.4f deg" % (
                    T, n, eff)
            else:
                line += "   |   raw 0.20 deg -> %.4f deg %s" % (
                    eff, "OK" if eff < need else "insufficient")
        print(line)
    print()
    print("  ⚠️  Averaging only kills ZERO-MEAN noise.  It does nothing for:")
    print("      - IMU roll/pitch mounting bias (degenerate with common tilt)")
    print("      - joint stiction/backlash, which makes the settled posture")
    print("        depend on the direction of approach.  Mitigation: approach")
    print("        every posture from the same direction, and re-run the whole")
    print("        sequence in reverse order as a repeatability check.  If the")
    print("        two runs disagree by more than the tolerance, hysteresis")
    print("        dominates and this method has hit its floor.")

    print()
    print("=" * 78)
    print("worst noiseless recovery error: %.4f deg (tolerance %.2f)  %s"
          % (worst_overall, args.tol_deg,
             "PASS" if worst_overall < args.tol_deg else "FAIL"))
    print("=" * 78)
    return 0 if worst_overall < args.tol_deg else 1


def cmd_diagnose_fk(args, chains):
    """Find what actually produced the 2026-08-05 FK failure.

    The repo records it as "parallel ankle, so a serial chain cannot solve it"
    (HANDOFF_TO_TRAINING.md section 7 item 6, and the docstring of
    tools/replay_real_in_mujoco.py).  The symptom was: trunk pitch -34 deg,
    foot 23 cm in front of the body.

    That explanation cannot be right, for three independent reasons:
      1. motor_state_serial is by definition the VIRTUAL SERIAL joint angle --
         the firmware has already inverted the parallel mechanism.  deploy's
         _verify_joint_layout proves the SDK distinguishes them and that they
         differ at exactly 4 indices.
      2. From the ankle joints upward the URDF chain is plain serial; the
         parallel linkage lives below the ankle-pitch/roll axes and does not
         change where the foot is relative to the shank.
      3. A parallel-linkage modelling error would be a few degrees at the
         ankle, not 34 degrees at the trunk.

    So this scans the error modes a hand-written FK actually makes and reports
    which one reproduces -34 deg / 23 cm.
    """
    print("=" * 78)
    print("DIAGNOSING the 2026-08-05 FK failure (trunk pitch -34 deg,")
    print("foot 23 cm in front).  Recorded cause: 'parallel ankle'.")
    print("=" * 78)
    print()

    q_nom = [-0.2, 0.0, 0.0, 0.4, -0.25, 0.0]
    print("posture used: the RL default, L/R identical")
    print("  hip_pitch %+.2f  hip_roll %+.2f  hip_yaw %+.2f  knee %+.2f"
          "  ankle_pitch %+.2f  ankle_roll %+.2f" % tuple(q_nom))
    print()
    print("URDF leg chain order (this matters and is NOT the usual one):")
    for j in chains["left"].joints:
        print("    %-18s axis %s  origin z %+.6f"
              % (j.name, tuple(int(a) for a in j.axis), j.xyz[2]))
    print()

    def trunk_pitch_for_flat_foot(q6):
        """FK the foot, then report the trunk pitch that makes the sole flat,
        and the resulting horizontal foot offset."""
        R, p = chains["left"].fk(q6)
        # sole normal in trunk frame; the trunk pitch needed to level it
        n = chains["left"].sole_normal(R)
        pitch = math.atan2(n[0], -n[2])
        # foot position after applying that trunk rotation
        c, s = math.cos(-pitch), math.sin(-pitch)
        x = c * p[0] + s * p[2]
        return pitch, x, p

    pitch0, x0, p0 = trunk_pitch_for_flat_foot(q_nom)
    print("CORRECT serial FK on this posture:")
    print("    trunk pitch to level the sole = %+.2f deg" % (pitch0 * DEG))
    print("    sole position in the trunk frame = (%+.4f, %+.4f, %+.4f) m"
          % tuple(p0))
    print("  -> the plain serial chain resolves this posture fine, using")
    print("     motor_state_serial and the robot-revision URDF.  Whatever went")
    print("     wrong in 2026-08-05, it was not that a serial chain cannot")
    print("     represent this robot.")
    print()

    print("THE STRUCTURAL FACT that makes a hand-written FK fragile here:")
    print("  hip_pitch, knee and ankle_pitch all rotate about +y, and every")
    print("  joint origin rpy in the leg is exactly 0.  So the sole's pitch is")
    print("  EXACTLY the sum q_hip_pitch + q_knee + q_ankle_pitch:")
    print("      %+.2f %+.2f %+.2f = %+.3f rad = %+.2f deg"
          % (q_nom[0], q_nom[3], q_nom[4], q_nom[0] + q_nom[3] + q_nom[4],
             (q_nom[0] + q_nom[3] + q_nom[4]) * DEG))
    print("  A sum of three nearly-cancelling terms is the worst possible")
    print("  numerical shape for a sign or index slip: flipping ONE sign moves")
    print("  the answer by twice that joint's angle, up to 46 deg for the knee.")
    print()

    # Exhaustive scan: every sign choice and every drop of the three pitch
    # joints, plus the plausible index permutations.  Hand-picking variants
    # invites fitting the answer; enumerate instead.
    target_pitch = -34.0
    print("EXHAUSTIVE SCAN over sign flips / dropped terms of the three pitch")
    print("joints (2^3 signs x 2^3 drops = 64), reporting everything that lands")
    print("within 3 deg of the reported -34 deg:")
    print()
    print("  %-46s %11s %11s" % ("error mode", "pitch(deg)", "|foot x| cm"))
    print("  " + "-" * 70)
    names = ("hip_pitch", "knee", "ankle_pitch")
    idxs = (0, 3, 4)
    hits = []
    for smask in range(8):
        for dmask in range(8):
            if smask & dmask:
                continue          # flipping a dropped term is the same variant
            q = list(q_nom)
            desc = []
            for b, (nm, ix) in enumerate(zip(names, idxs)):
                if dmask >> b & 1:
                    q[ix] = 0.0
                    desc.append("drop " + nm)
                elif smask >> b & 1:
                    q[ix] = -q[ix]
                    desc.append("flip " + nm)
            if not desc:
                continue
            pitch, x, p = trunk_pitch_for_flat_foot(q)
            if abs(pitch * DEG - target_pitch) <= 3.0:
                hits.append((", ".join(desc), pitch * DEG, abs(x) * 100,
                             abs(p[0]) * 100))
    for d, pd, xc, praw in sorted(hits, key=lambda h: abs(h[1] - target_pitch)):
        print("  %-46s %11.2f %11.2f" % (d, pd, praw))
    print()
    if hits:
        print("So -34 deg is reachable, and only by error modes that mis-sign or")
        print("drop one of the three collinear PITCH joints.  None of the")
        print("candidates involves the ankle ROLL axis or the parallel linkage.")
    else:
        print("No single-term sign/drop error reproduces -34 deg.")
    print()
    print("WHY THE RECORDED CAUSE CANNOT BE RIGHT (three independent reasons):")
    print("  1. motor_state_serial IS the virtual serial joint angle -- the")
    print("     firmware has already inverted the parallel mechanism.  deploy's")
    print("     _verify_joint_layout relies on serial != parallel at exactly")
    print("     indices %s to fingerprint the hardware layout."
          % (list(PARALLEL_MECH_INDEXES),))
    print("  2. The parallel linkage sits BELOW the ankle pitch/roll axes. It")
    print("     changes how the actuators reach an ankle angle, not where the")
    print("     foot is relative to the shank.  Above the ankle joint the chain")
    print("     is serial in the URDF and in the hardware alike.")
    print("  3. A parallel-linkage modelling error is a few degrees at the")
    print("     ankle.  It cannot become 34 degrees at the trunk.")
    print()
    print("CONSEQUENCE: 'CoM cannot be computed by FK' is not a true constraint")
    print("on this robot, and tools/replay_real_in_mujoco.py's stated reason for")
    print("existing is wrong (the tool itself is fine -- using MuJoCo's")
    print("kinematics is a perfectly good choice, just not a forced one).")
    return 0


def cmd_solve(args, chains):
    with open(args.solve, "r") as fh:
        data = json.load(fh)
    poses = data["poses"] if isinstance(data, dict) else data
    for p in poses:
        if len(p["q12"]) != 12:
            raise ValueError("pose %s: q12 must have 12 entries" % p.get("name"))

    print("=" * 78)
    print("SOLVE from %s -- %d postures" % (args.solve, len(poses)))
    print("=" * 78)
    sing, vecs = observability(chains, poses)
    print_observability(sing, vecs, "this posture set")
    print()

    params, cost, r = solve(chains, poses, use_imu_bias=args.imu_bias,
                            verbose=args.verbose, prior_weight=args.prior)
    delta = params[:N_DELTA]

    rms_before = math.sqrt(
        dot(residuals(chains, poses, [0.0] * len(params), args.imu_bias),
            residuals(chains, poses, [0.0] * len(params), args.imu_bias))
        / max(1, len(r)))
    rms_after = math.sqrt(dot(r, r) / max(1, len(r)))
    print("residual RMS: %.6e  ->  %.6e  (%.1fx reduction)"
          % (rms_before, rms_after,
             rms_before / rms_after if rms_after > 0 else float("inf")))
    print("stance width D = %.4f m, foot/stance angle Phi = %.2f deg"
          % (params[IDX_D], params[IDX_PHI] * DEG))
    print()
    print("  %-16s %12s" % ("joint", "delta(deg)"))
    for i in range(12):
        flag = "  <-- large" if abs(delta[i] * DEG) > 3.0 else ""
        print("  %-16s %12.3f%s" % (DELTA_NAMES[i], delta[i] * DEG, flag))
    print()
    print("  delta RMS %.3f deg, max |delta| %.3f deg"
          % (math.sqrt(sum((d * DEG) ** 2 for d in delta) / 12),
             max(abs(d * DEG) for d in delta)))

    if args.emit_yaml:
        write_yaml(args.emit_yaml, delta, poses, sing, rms_before, rms_after)
        print()
        print("wrote %s" % args.emit_yaml)
    return 0


def write_yaml(path, delta, poses, sing, rms_before, rms_after):
    q = ",\n    ".join(", ".join("%+.6f" % delta[r * 6 + c] for c in range(6))
                       for r in range(2))
    with open(path, "w") as fh:
        fh.write("""# Joint zero offsets for this robot, measured by
# tools/estimate_joint_zero.py.  Regenerate whenever the joints are
# re-zeroed -- these are hardware state, not tuning.
#
# CONVENTION (matches envs/K1/goal_pose.py:420-437 exactly):
#     q_meas = q_true + delta
# so deploy must apply BOTH of:
#     observation:  q_corrected = q_meas - delta
#     command:      publish       target  + delta
# Applying only one trades one bias for another.
#
# Order is the 12 leg DOFs starting at common joint index %d:
#   L hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
#   R hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
#
# provenance
#   postures            %d
#   residual RMS        %.4e -> %.4e
#   weakest observable  sigma_min/sigma_max = %.3e
#   NOTE: directions with sigma near zero are reported as 0 by the solver's
#   minimum-norm prior.  That is deliberate; see the module docstring.

joint_zero:
  enabled: true
  leg_dof_start: %d
  delta_rad: [
    %s
  ]
""" % (LEG_DOF_START, len(poses), rms_before, rms_after,
       sing[-1] / sing[0] if sing[0] else 0.0, LEG_DOF_START, q))


def cmd_collect(args, chains):
    """Collect postures on the robot.  Hardware only."""
    try:
        import booster_robotics_sdk_python as B   # noqa: F401
    except ImportError:
        print("booster_robotics_sdk_python not available -- this subcommand")
        print("only runs on the robot.  See R7 in")
        print("DEPLOY_REQUESTS_FROM_TRAINING.md for the procedure.")
        return 2
    from collect_joint_zero import run_collection   # noqa
    return run_collection(args)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", default=None,
                    help="robot-revision URDF (default: auto-detect)")
    ap.add_argument("--sole-z", type=float, default=SOLE_MESH_Z,
                    help="sole plane z in the foot link frame. Default %.6f "
                         "(measured from Left_Foot.STL). The collision box "
                         "every simulator uses is %.6f -- 2.9 mm higher."
                         % (SOLE_MESH_Z, SOLE_BOX_Z))
    ap.add_argument("--allow-asset-mismatch", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--observability", action="store_true")
    ap.add_argument("--diagnose-fk", action="store_true")
    ap.add_argument("--solve", metavar="POSES_JSON")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--emit-yaml", metavar="PATH")
    ap.add_argument("--imu-bias", action="store_true",
                    help="also estimate IMU roll/pitch mounting bias. REFUSED "
                         "by default: it is exactly degenerate with common-mode "
                         "leg tilt.")
    ap.add_argument("--prior", type=float, default=1e-4,
                    help="Tikhonov pull of delta toward zero (minimum-norm "
                         "solution in the unobservable directions)")
    ap.add_argument("--tol-deg", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="/tmp/zero_poses.json")
    ap.add_argument("--hold-s", type=float, default=2.0)
    ap.add_argument("--settle-s", type=float, default=1.0)
    ap.add_argument("--config", default="configs/Goal_Pose_E0.yaml")
    ap.add_argument("--net", default="127.0.0.1")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.imu_bias:
        print("--imu-bias refused: IMU roll/pitch mounting bias is EXACTLY")
        print("degenerate with a common-mode leg tilt on both legs.  Estimating")
        print("both makes the answer arbitrary along that 2-D subspace.")
        print("Calibrate the IMU separately against a levelled trunk first.")
        return 2

    urdf = args.urdf or find_urdf(REPO_ROOT)
    chains = load_leg_chain(urdf, sole_z=args.sole_z,
                            allow_mismatch=args.allow_asset_mismatch)
    print("asset: %s" % urdf)
    print("       Left_Hip_Pitch origin z = %+.6f m (robot revision)"
          % chains["left"].joints[0].xyz[2])
    print("       sole plane z = %+.6f m in the foot frame" % args.sole_z)
    print()

    if args.self_test:
        return cmd_self_test(args, chains)
    if args.observability:
        return cmd_observability(args, chains)
    if args.diagnose_fk:
        return cmd_diagnose_fk(args, chains)
    if args.solve:
        return cmd_solve(args, chains)
    if args.collect:
        return cmd_collect(args, chains)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
