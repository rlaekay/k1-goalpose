"""K1 leg forward kinematics + the linear algebra the zero estimator needs.

Why this file exists separately
-------------------------------
`estimate_joint_zero.py` has to run in two places that share almost no
dependencies:

  * the **robot** (deploy venv: booster SDK, numpy, no scipy)
  * a **laptop** doing offline solving / self-test (may have neither)

So everything here is stdlib-only.  12 unknowns and a few hundred residuals do
not need BLAS.  If numpy is present it is not used -- one code path only, so the
self-test exercises exactly the code the robot runs.

⛔ ASSET RULE (this is the load-bearing part -- read before changing anything)
------------------------------------------------------------------------------
The leg kinematics of the K1 assets in this repo are NOT all the same.  Audited
2026-08-08 by parsing all 12 URDFs:

    Left/Right_Hip_Pitch joint origin z
        -0.077 m   k1/K1_locomotion.urdf            (pulled off the robot)
                   resources/K1/K1_locomotion_robot.urdf
                   resources/K1/K1_robot_boxfoot.urdf
        -0.062 m   EVERY other asset, including resources/K1/K1_serial.xml
                   (the MJCF) and K1_locomotion_armsdown.urdf (what we trained on)

That is the ONLY leg-chain difference among all of them -- every other joint
origin, rpy and axis is byte-identical.  15 mm, both legs, purely vertical.

`load_leg_chain` therefore refuses any asset whose hip-pitch z is not -0.077
unless you pass allow_mismatch=True.  A zero estimator running on the wrong
chain does not fail loudly; it silently folds the geometry error into delta,
which is the exact failure mode this repo has hit four times.

Sole plane
----------
The foot's flat contact patch was measured by raycasting Left_Foot.STL on a
25 x 11 grid (2026-08-08): inside the 16 x 7 cm footprint the lowest surface is
planar to within 0.29 mm, centred on (x=+0.014, y=0).  So the real sole IS a
flat rectangle, not a rounded/rockered foot -- the rounding is confined to the
perimeter (toe above x=+94 mm, heel below x=-66 mm, edges beyond |y|=35 mm).

    mesh sole plane   z = -0.026896 m   <- the real robot
    collision box     z = -0.024    m   <- what Isaac and MuJoCo stand on

The 2.90 mm gap is a real modelling error but it is COMMON to both feet, so for
this estimator it lands in the unobservable base-height direction.  Default is
the mesh value because that is the hardware.
"""

import math
import os
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# joint layout on this robot (verified on hardware, see deploy_goal_pose.py
# _verify_joint_layout and configs/Goal_Pose_E0.yaml)
# ---------------------------------------------------------------------------
# 22 motors, no waist.  Legs start at 10.  serial != parallel only at the four
# ankle indices, which is how we know the ankles are the parallel mechanism and
# that motor_state_serial already carries virtual SERIAL joint angles.
LEG_DOF_START = 10
PARALLEL_MECH_INDEXES = (14, 15, 20, 21)

LEG_JOINTS = {
    "left": ["Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
             "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll"],
    "right": ["Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
              "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll"],
}
# index into the 22-vector, in the same order as LEG_JOINTS
LEG_INDEX = {"left": list(range(10, 16)), "right": list(range(16, 22))}

FOOT_LINK = {"left": "left_foot_link", "right": "right_foot_link"}

# Sole reference point in the foot link frame (metres).  x/y = centre of the
# measured flat patch; z = the measured sole plane of the mesh.
SOLE_MESH_Z = -0.026896
SOLE_BOX_Z = -0.024
SOLE_REF_XY = (0.014, 0.0)

EXPECTED_HIP_PITCH_Z = -0.077


# ---------------------------------------------------------------------------
# tiny linear algebra (stdlib only)
# ---------------------------------------------------------------------------
def mat_eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def mat_mul(A, B):
    n, k, m = len(A), len(B), len(B[0])
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        oi = out[i]
        for t in range(k):
            a = Ai[t]
            if a == 0.0:
                continue
            Bt = B[t]
            for j in range(m):
                oi[j] += a * Bt[j]
    return out


def mat_vec(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def transpose(A):
    return [list(col) for col in zip(*A)]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def unit(a):
    n = norm(a)
    if n < 1e-12:
        raise ValueError("cannot normalise a zero vector")
    return [x / n for x in a]


def rot_axis(axis, q):
    """Rodrigues rotation about a unit axis."""
    x, y, z = axis
    c, s = math.cos(q), math.sin(q)
    C = 1.0 - c
    return [
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ]


def rot_rpy(r, p, y):
    """ZYX Euler (yaw*pitch*roll) -- the convention the booster IMU reports and
    the one deploy_goal_pose.rotate_vector_inverse_rpy assumes."""
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def gravity_in_body(roll, pitch, yaw=0.0):
    """Gravity unit vector expressed in the trunk frame.

    Identical quantity to deploy's obs[0:3] (projected_gravity): R_wb^T @ [0,0,-1].
    Yaw cancels out for a vertical vector, so it is ignored.
    """
    R = rot_rpy(roll, pitch, yaw)
    return [-R[2][0], -R[2][1], -R[2][2]]


def solve_spd(A, b):
    """Solve A x = b for symmetric positive-definite A, by Cholesky.

    Falls back to Gaussian elimination with partial pivoting if A turns out not
    to be positive definite (which happens when Levenberg damping is small and
    the normal matrix is only semi-definite).  This is the hot path -- it runs
    once per Gauss-Newton iteration -- so it must not be an eigensolver.
    """
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    ok = True
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 1e-300:
                    ok = False
                    break
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
        if not ok:
            break
    if ok:
        y = [0.0] * n
        for i in range(n):
            y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
        return x
    return solve_lu(A, b)


def solve_lu(A, b):
    """Gaussian elimination with partial pivoting; returns a least-squares-ish
    answer with zeros in any exactly singular direction."""
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    piv = []
    row = 0
    for col in range(n):
        best = max(range(row, n), key=lambda r: abs(M[r][col]))
        if abs(M[best][col]) < 1e-14:
            continue
        M[row], M[best] = M[best], M[row]
        p = M[row][col]
        M[row] = [v / p for v in M[row]]
        for r in range(n):
            if r != row and M[r][col] != 0.0:
                f = M[r][col]
                M[r] = [M[r][c] - f * M[row][c] for c in range(n + 1)]
        piv.append((row, col))
        row += 1
        if row == n:
            break
    x = [0.0] * n
    for r, c in piv:
        x[c] = M[r][n]
    return x


def jacobi_eigh(A, iters=60, tol=1e-13):
    """Eigen-decomposition of a symmetric matrix by cyclic Jacobi.

    Returns (eigenvalues descending, eigenvectors as a list of columns).
    Only used for the observability spectrum, never in the solver's inner loop
    -- see solve_spd for that.
    """
    n = len(A)
    a = [row[:] for row in A]
    v = mat_eye(n)
    scale = max((abs(a[i][j]) for i in range(n) for j in range(n)), default=1.0)
    scale = scale if scale > 0 else 1.0
    for _ in range(iters):
        off = math.sqrt(sum(a[i][j] ** 2 for i in range(n)
                            for j in range(n) if i != j))
        if off < tol * scale:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = (1.0 if theta >= 0 else -1.0) / (
                    abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    vals = [a[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: -vals[i])
    vecs = [[v[r][i] for r in range(n)] for i in order]   # list of columns
    return [vals[i] for i in order], vecs


def solve_damped(JtJ, Jtr, lam):
    """(JtJ + lam*diag(JtJ) + floor) x = -Jtr.

    Levenberg damping is scaled by the diagonal so that badly-scaled parameters
    (radians vs metres) do not make lam mean different things per column.  An
    absolute floor is added as well, because a column that is exactly
    unobservable has a zero diagonal and relative damping alone leaves it
    singular -- that floor is what keeps the step out of the nullspace.
    """
    n = len(JtJ)
    d = [max(JtJ[i][i], 0.0) for i in range(n)]
    scale = max(d) if max(d) > 0 else 1.0
    A = [[JtJ[i][j] + ((lam * d[i] + 1e-12 * scale) if i == j else 0.0)
          for j in range(n)] for i in range(n)]
    return solve_spd(A, [-v for v in Jtr])


# ---------------------------------------------------------------------------
# URDF -> leg chain
# ---------------------------------------------------------------------------
class Joint(object):
    __slots__ = ("name", "xyz", "rpy", "axis", "lower", "upper")

    def __init__(self, name, xyz, rpy, axis, lower, upper):
        self.name = name
        self.xyz = xyz
        self.rpy = rpy
        self.axis = axis
        self.lower = lower
        self.upper = upper


class LegChain(object):
    """One leg: base(Trunk) -> foot link, plus the sole reference point."""

    def __init__(self, side, joints, sole_ref):
        self.side = side
        self.joints = joints            # 6 Joint, hip_pitch .. ankle_roll
        self.sole_ref = sole_ref        # 3-vector in the foot link frame

    def fk(self, q):
        """Return (R_trunk_foot, p_sole_in_trunk).

        q is the 6-vector of TRUE joint angles for this leg, in LEG_JOINTS order.
        Every joint origin rpy in the K1 leg chain is exactly zero (verified for
        all 12 assets), but the general term is kept so a future asset with a
        non-zero origin rotation does not silently break this.
        """
        R = mat_eye(3)
        p = [0.0, 0.0, 0.0]
        for j, qj in zip(self.joints, q):
            if j.rpy != (0.0, 0.0, 0.0):
                R = mat_mul(R, rot_rpy(*j.rpy))
            p = [p[i] + sum(R[i][k] * j.xyz[k] for k in range(3))
                 for i in range(3)]
            R = mat_mul(R, rot_axis(j.axis, qj))
        p_sole = [p[i] + sum(R[i][k] * self.sole_ref[k] for k in range(3))
                  for i in range(3)]
        return R, p_sole

    def sole_normal(self, R):
        """Downward sole normal in the trunk frame."""
        return [-R[0][2], -R[1][2], -R[2][2]]

    def sole_forward(self, R):
        """Foot +x (toe direction) in the trunk frame."""
        return [R[0][0], R[1][0], R[2][0]]


def _vec3(s, default=(0.0, 0.0, 0.0)):
    if s is None:
        return tuple(default)
    parts = [float(v) for v in s.split()]
    while len(parts) < 3:
        parts.append(0.0)
    return tuple(parts[:3])


def load_leg_chain(urdf_path, sole_z=SOLE_MESH_Z, allow_mismatch=False):
    """Parse a URDF into {'left': LegChain, 'right': LegChain}.

    Raises unless the asset is the robot revision (see ASSET RULE at the top).
    """
    root = ET.parse(urdf_path).getroot()
    jmap = {}
    for j in root.findall("joint"):
        name = j.get("name")
        o = j.find("origin")
        ax = j.find("axis")
        lim = j.find("limit")
        jmap[name] = Joint(
            name,
            _vec3(o.get("xyz") if o is not None else None),
            _vec3(o.get("rpy") if o is not None else None),
            _vec3(ax.get("xyz") if ax is not None else None, (0, 0, 1)),
            float(lim.get("lower")) if lim is not None else None,
            float(lim.get("upper")) if lim is not None else None,
        )

    missing = [n for side in LEG_JOINTS for n in LEG_JOINTS[side]
               if n not in jmap]
    if missing:
        raise ValueError("%s is missing leg joints: %s"
                         % (urdf_path, missing))

    hz = jmap["Left_Hip_Pitch"].xyz[2]
    if abs(hz - EXPECTED_HIP_PITCH_Z) > 1e-6 and not allow_mismatch:
        raise ValueError(
            "ASSET MISMATCH: %s has Left_Hip_Pitch origin z = %.4f m, the robot "
            "revision has %.4f m (a %.1f mm difference, both legs).\n"
            "Every training asset and the MJCF carry the -0.062 value; only\n"
            "  k1/K1_locomotion.urdf, resources/K1/K1_locomotion_robot.urdf,\n"
            "  resources/K1/K1_robot_boxfoot.urdf\n"
            "match the hardware.  Solving delta on the wrong chain folds the\n"
            "geometry error into the estimate instead of failing.  Pass\n"
            "allow_mismatch=True only if you mean to."
            % (urdf_path, hz, EXPECTED_HIP_PITCH_Z,
               abs(hz - EXPECTED_HIP_PITCH_Z) * 1000))

    sole_ref = (SOLE_REF_XY[0], SOLE_REF_XY[1], sole_z)
    chains = {}
    for side in ("left", "right"):
        joints = [jmap[n] for n in LEG_JOINTS[side]]
        ref = (sole_ref[0], sole_ref[1] * (1 if side == "left" else -1),
               sole_ref[2])
        chains[side] = LegChain(side, joints, ref)
    return chains


def default_urdf_candidates(repo_root):
    """Assets whose leg chain matches the hardware, most-primary first."""
    return [
        os.path.join(repo_root, "k1", "K1_locomotion.urdf"),
        os.path.join(repo_root, "htwk-gym", "resources", "K1",
                     "K1_locomotion_robot.urdf"),
        os.path.join(repo_root, "htwk-gym", "resources", "K1",
                     "K1_robot_boxfoot.urdf"),
    ]


def find_urdf(repo_root):
    for p in default_urdf_candidates(repo_root):
        if os.path.exists(p):
            return p
    raise IOError("no robot-revision K1 URDF found under %s" % repo_root)
