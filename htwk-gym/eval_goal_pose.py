"""Evaluation harness for the K1 GoalPose task (MASTERPLAN milestone 1).

Loads a trained checkpoint, rolls the policy in headless sim, and measures the
actual task metrics the reward curve cannot show:
  - final position error per goal segment (median / p90)
  - final heading error per goal segment
  - final base speed (did it stop?)
  - falls (non-timeout terminations)
  - setup/rollout wall-clock time and vectorized simulation throughput
then judges them against the MASTERPLAN gates and writes report.md / report.json
/ segments.csv into <run_dir>/eval/<timestamp>/.

A "goal segment" ends when the env's 4-8 s resample timer fires; the final
position is measured against the goal that was just replaced, so the numbers
are exact (not one control step stale).

Beyond the pass/fail gates it also answers "so what do I change?", because a
single median error number is compatible with several failure modes that need
opposite fixes:
  - the goal was unreachable in the time given   -> task definition, not reward
  - never got close                              -> far-field gradient (goal_progress)
  - got there, then wandered off                 -> holding (goal_reached/goal_stop)
  - got there, never stopped                     -> goal_stop
  - position fine, heading bad (or vice versa)   -> constellation_radius coupling
See summarize()/recommend() and the "실패 모드 분해" report section.

The rollout/summary/report functions are importable so tools/select_best_checkpoint.py
scores candidates with exactly this code rather than a reimplementation.

Usage (server):
    python eval_goal_pose.py --task K1/Goal_Pose --checkpoint -1 \
        --sim_device cuda:1 --rl_device cuda:1
"""

import os
import csv
import copy
import hashlib
import json
import glob
import time
import random
import argparse
import math

import isaacgym  # noqa: F401  (must be imported before torch)
from envs import *  # noqa: F401,F403  (registers task classes)

import torch
import numpy as np
import yaml

from isaacgym.torch_utils import get_euler_xyz, quat_rotate, quat_rotate_inverse
from utils.model import ActorCritic
from utils.runner import get_task_class


# goal_categories mixture in envs/K1/*.yaml, as recorded by GoalPose._resample_goals
CATEGORY_NAMES = {-1: "uniform", 0: "stand", 1: "straight", 2: "lateral", 3: "turn", 4: "combined",
                  5: "path", 6: "seq"}
# Mirrors envs/K1/goal_pose_v7.CATEGORY_PATH. Duplicated rather than imported so
# eval_goal_pose stays usable for tasks that never load the v7 env.
CATEGORY_PATH = 5

# start-distance bins [m] for the per-distance breakdown
DISTANCE_BINS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, float("inf")]

# A segment shorter than this cannot contain a cruise: at 1.3 m/s the robot
# needs ~1.4 m just to accelerate and brake, so anything under it reports a
# transient no matter how the policy behaves.  P2's sustained reading is only
# measurable above this distance, and 8-15 measured that only ~3% of segments
# clear it -- which is a fact about the task, not the robot.
LONG_SEGMENT_M = 2.0


def wrap(x):
    return (x + torch.pi) % (2 * torch.pi) - torch.pi


def synchronize_cuda_devices(*devices):
    """Synchronize every distinct CUDA device used by sim or policy."""
    synchronized = set()
    for value in devices:
        device = torch.device(value)
        if device.type != "cuda":
            continue
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index in synchronized:
            continue
        torch.cuda.synchronize(index)
        synchronized.add(index)


def find_latest_checkpoint(task_name):
    robot = task_name.split("/")[0]
    for pattern in [
        os.path.join("logs", robot, task_name, "**", "*.pth"),
        os.path.join("logs", "**", "*.pth"),
    ]:
        models = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)
        if models:
            return models[-1]
    return None


def _draw_disk(img, cx, cy, r, color):
    h, w = img.shape[:2]
    x0, x1 = max(0, int(cx - r)), min(w, int(cx + r) + 2)
    y0, y1 = max(0, int(cy - r)), min(h, int(cy + r) + 2)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    img[y0:y1, x0:x1][mask] = color


def _draw_line(img, x0, y0, x1, y1, color):
    n = int(max(abs(x1 - x0), abs(y1 - y0)) * 2) + 1
    h, w = img.shape[:2]
    for t in np.linspace(0.0, 1.0, n):
        x, y = int(x0 + (x1 - x0) * t), int(y0 + (y1 - y0) * t)
        if 0 <= y < h and 0 <= x < w:
            img[y, x] = color


# 5x7 bitmap font, one entry per glyph = 7 row bitmaps (bit 4 = leftmost column).
# Hand-rolled because the eval box has no cv2/PIL and the HUD must never be the
# reason a video fails to render.
_FONT = {
    "0": (0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E),
    "1": (0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "2": (0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F),
    "3": (0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E),
    "4": (0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02),
    "5": (0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E),
    "6": (0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E),
    "7": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08),
    "8": (0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E),
    "9": (0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C),
    "A": (0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "B": (0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E),
    "C": (0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E),
    "D": (0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C),
    "E": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F),
    "F": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10),
    "G": (0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F),
    "H": (0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "I": (0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "K": (0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11),
    "L": (0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F),
    "M": (0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11),
    "N": (0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11),
    "O": (0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "P": (0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10),
    "R": (0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11),
    "S": (0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E),
    "T": (0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    "U": (0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "V": (0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04),
    "W": (0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11),
    "X": (0x11, 0x0A, 0x04, 0x04, 0x04, 0x0A, 0x11),
    "Y": (0x11, 0x0A, 0x04, 0x04, 0x04, 0x04, 0x04),
    "Z": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F),
    ".": (0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C),
    ":": (0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00),
    "-": (0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00),
    "+": (0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00),
    "/": (0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10),
    "|": (0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    "*": (0x0C, 0x12, 0x12, 0x0C, 0x00, 0x00, 0x00),  # degree sign
    " ": (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00),
}


def _draw_text(img, x, y, text, color=(235, 235, 235), scale=2):
    """Blit `text` at (x, y) top-left, `scale` pixels per font pixel."""
    h, w = img.shape[:2]
    draw_color = np.asarray(color, dtype=img.dtype)
    if img.ndim == 3 and img.shape[2] == 4 and draw_color.shape[0] == 3:
        draw_color = np.asarray((color[0], color[1], color[2], 255), dtype=img.dtype)
    elif img.ndim == 3 and draw_color.shape[0] != img.shape[2]:
        draw_color = draw_color[:img.shape[2]]
    cx = x
    for ch in text.upper():
        glyph = _FONT.get(ch)
        if glyph is None:
            cx += 6 * scale
            continue
        for row, bits in enumerate(glyph):
            for col in range(5):
                if not (bits >> (4 - col)) & 1:
                    continue
                px, py = cx + col * scale, y + row * scale
                if 0 <= py < h and 0 <= px < w:
                    img[py:py + scale, px:px + scale] = draw_color
        cx += 6 * scale
    return cx


def draw_telemetry_hud(frame, st, size=240, scale=2):
    """Text HUD under the constellation inset: body velocity (the number the
    MASTERPLAN speed target is stated in), goal distance, position error and the
    external disturbance currently being applied. Without these on-screen there
    is no way to tell a slow policy from a policy given a near goal."""
    vx, vy, wz, dist, herr_deg, push_n, push_nm = st[4:11]
    speed = float(np.hypot(vx, vy))
    x0, y0 = 12, 8 + size + 8
    lines = [
        ("VEL  {:5.2f} M/S".format(speed), (120, 230, 255) if speed < 1.0 else (140, 255, 160)),
        ("  VX {:+5.2f}  VY {:+5.2f}".format(vx, vy), (190, 190, 190)),
        ("  WZ {:+5.2f} RAD/S".format(wz), (190, 190, 190)),
        ("DIST {:5.2f} M".format(dist), (235, 235, 235)),
        ("HEAD {:+6.1f}*".format(herr_deg), (235, 235, 235)),
    ]
    for i, (text, color) in enumerate(lines):
        _draw_text(frame, x0, y0 + i * 9 * scale, text, color, scale)
    y_push = y0 + len(lines) * 9 * scale
    if push_n > 1e-3 or push_nm > 1e-3:
        _draw_text(frame, x0, y_push, "PUSH {:4.1f}N {:4.1f}NM".format(push_n, push_nm), (255, 140, 120), scale)
    return frame


def draw_constellation_inset(frame, base_xy, base_yaw, goal_xy, goal_yaw, radius, size=240, span=3.0):
    """Top-down inset (upper-left corner): green ring = goal constellation, blue
    ring = robot's current constellation, big dot on each ring = its heading
    point. The visual gap between corresponding ring points IS the constellation
    error the reward penalizes; rings coinciding = pose reached."""
    channels = frame.shape[2]
    inset = np.zeros((size, size, channels), dtype=frame.dtype)
    inset[..., :3] = 25
    if channels == 4:
        inset[..., 3] = 255
    scale = size / (2.0 * span)

    def to_px(wx, wy):
        # robot-centered top-down view (world axes): +x up, +y left
        return size / 2.0 - (wy - base_xy[1]) * scale, size / 2.0 - (wx - base_xy[0]) * scale

    rgb = inset[..., :3]
    bx, by = to_px(base_xy[0], base_xy[1])
    gx, gy = to_px(goal_xy[0], goal_xy[1])
    _draw_line(rgb, bx, by, gx, gy, (110, 110, 110))
    for (cx_w, cy_w), yaw, color in (
        ((goal_xy[0], goal_xy[1]), goal_yaw, (70, 220, 120)),
        ((base_xy[0], base_xy[1]), base_yaw, (80, 160, 255)),
    ):
        cx, cy = to_px(cx_w, cy_w)
        for k in range(8):
            a = yaw + k * (2.0 * np.pi / 8.0)
            x, y = to_px(cx_w + radius * np.cos(a), cy_w + radius * np.sin(a))
            _draw_disk(rgb, x, y, 5 if k == 0 else 3, color)
        hx, hy = to_px(cx_w + 0.4 * np.cos(yaw), cy_w + 0.4 * np.sin(yaw))
        _draw_line(rgb, cx, cy, hx, hy, color)
        _draw_disk(rgb, cx, cy, 4, color)
    frame[8:8 + size, 8:8 + size] = inset
    return frame


def draw_goal_sequence(frame, base_xy, base_yaw, seq, k, size=240, span=3.0):
    """Overlay the upcoming goal sequence on the constellation inset (v8).

    Sequential navigation is impossible to judge from a single goal marker: the
    whole question is whether the robot sets up a turn BEFORE it arrives, which
    only reads if the next goals are visible in the same frame. Banked goals go
    grey, the active one amber, upcoming ones green and fading.

    Writes through `rgb = inset[..., :3]` so it is correct for both RGB and RGBA
    camera frames -- Isaac Gym's IMAGE_COLOR is RGBA, which is what broke the
    HUD once already (commit 5493840).
    """
    if seq is None or len(seq) == 0:
        return frame
    inset = frame[8:8 + size, 8:8 + size]
    rgb = inset[..., :3]
    scale = size / (2.0 * span)

    def to_px(wx, wy):
        return size / 2.0 - (wy - base_xy[1]) * scale, size / 2.0 - (wx - base_xy[0]) * scale

    prev = None
    for i, g in enumerate(seq):
        x, y = to_px(g[0], g[1])
        if i < k:
            color = (90, 90, 90)                       # banked
        elif i == k:
            color = (255, 190, 60)                     # active
        else:
            fade = max(0.35, 1.0 - 0.25 * (i - k))
            color = (int(70 * fade), int(220 * fade), int(120 * fade))
        _draw_disk(rgb, x, y, 6 if i == k else 4, color)
        hx, hy = to_px(g[0] + 0.3 * np.cos(g[2]), g[1] + 0.3 * np.sin(g[2]))
        _draw_line(rgb, x, y, hx, hy, color)
        if prev is not None:
            _draw_line(rgb, prev[0], prev[1], x, y, (60, 60, 60))
        prev = (x, y)
    return frame


def _project_world(point, camera_pose, horizontal_fov_deg, width, height):
    """Project a world point into the existing follow-camera RGBA image.

    This uses the camera pose that BaseTask passed to set_camera_location, so it
    avoids the row/column-major ambiguity that repeatedly caused bad overlays
    when view/projection matrices were interpreted across Isaac Gym versions.
    """
    pos, target = (np.asarray(camera_pose[0], dtype=float),
                   np.asarray(camera_pose[1], dtype=float))
    forward = target - pos
    forward /= max(np.linalg.norm(forward), 1.0e-9)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right /= max(np.linalg.norm(right), 1.0e-9)
    up = np.cross(right, forward)
    rel = np.asarray(point, dtype=float) - pos
    depth = float(np.dot(rel, forward))
    if not np.isfinite(depth) or depth <= 0.05:
        return None
    focal = 0.5 * width / np.tan(0.5 * np.radians(horizontal_fov_deg))
    u = 0.5 * width + focal * float(np.dot(rel, right)) / depth
    v = 0.5 * height - focal * float(np.dot(rel, up)) / depth
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return (u, v)


def _segment_intersects_image(p0, p1, width, height):
    """Return whether a 2-D segment intersects the rendered image rectangle."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0), (dx, width - 1 - x0),
                 (-dy, y0), (dy, height - 1 - y0)):
        if abs(p) <= 1.0e-12:
            if q < 0.0:
                return False
            continue
        r = q / p
        if p < 0.0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return False
    return True


def _force_arrow_points(st, camera_pose, horizontal_fov_deg, width, height):
    """Project the force that acted in a recorded frame, or return ``None``.

    This helper is shared by rendering and smoke telemetry.  Consequently a
    counted arrow frame means the red world-space shaft actually has drawable
    pixels in the simulator image, not merely that some env had a nonzero force.
    """
    if len(st) < 15:
        return None
    force = np.asarray(st[13], dtype=float)
    origin = np.asarray(st[14], dtype=float)
    mag = float(np.linalg.norm(force))
    if not np.isfinite(mag) or mag <= 1.0e-3:
        return None
    length = 0.15 + 0.85 * np.sqrt(min(mag, 150.0) / 150.0)
    end = origin + length * force / mag
    p0 = _project_world(origin, camera_pose, horizontal_fov_deg, width, height)
    p1 = _project_world(end, camera_pose, horizontal_fov_deg, width, height)
    if p0 is None or p1 is None:
        return None
    if not _segment_intersects_image(p0, p1, width, height):
        return None
    return p0, p1


def draw_perspective_scene(frame, st, camera_pose, horizontal_fov_deg, path_trace):
    """Draw path/carrot/waypoint and a force arrow on the simulator view."""
    rgb = frame[..., :3]
    h, w = rgb.shape[:2]

    def project_xy(xy, z=0.04):
        return _project_world((float(xy[0]), float(xy[1]), float(z)),
                              camera_pose, horizontal_fov_deg, w, h)

    # A path task has no hidden final goal: goal_pos_world is the moving
    # lookahead/carrot.  The recent carrot trace is the path actually demanded.
    if len(st) >= 16 and int(st[15]) == CATEGORY_PATH:
        last = None
        for xy in path_trace[-120:]:
            p = project_xy(xy)
            if p is not None and last is not None:
                _draw_line(rgb, last[0], last[1], p[0], p[1], (60, 210, 100))
            if p is not None:
                last = p
        goal_color = (255, 190, 50)
        label = "PATH CARROT"
    else:
        goal_color = (70, 230, 120)
        label = "WAYPOINT GOAL"

    g = project_xy(st[2], 0.06)
    gh = project_xy((st[2][0] + 0.35 * np.cos(st[3]),
                     st[2][1] + 0.35 * np.sin(st[3])), 0.06)
    if g is not None:
        _draw_disk(rgb, g[0], g[1], 7, goal_color)
        if gh is not None:
            _draw_line(rgb, g[0], g[1], gh[0], gh[1], goal_color)
        _draw_text(rgb, int(g[0]) + 10, int(g[1]) - 10, label, goal_color, 1)

    # HBatch stores force in ENV_SPACE.  The origin is the selected rigid body
    # COM and arrow length uses sqrt scaling so both 3 N support and 150 N hits
    # remain visible without saturating the screen.
    arrow = _force_arrow_points(st, camera_pose, horizontal_fov_deg, w, h)
    if arrow is not None:
        p0, p1 = arrow
        red = (255, 40, 40)
        _draw_line(rgb, p0[0], p0[1], p1[0], p1[1], red)
        ang = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
        for da in (2.55, -2.55):
            _draw_line(rgb, p1[0], p1[1],
                       p1[0] + 14 * np.cos(ang + da),
                       p1[1] + 14 * np.sin(ang + da), red)
    return frame


# --------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------

def _median(x):
    return float(np.median(x)) if len(x) else float("nan")


def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def _frac(mask):
    return float(np.mean(mask)) if len(mask) else float("nan")


def bootstrap_ci(x, q=50.0, n_boot=600, alpha=0.05, seed=0, max_n=20000):
    """Percentile-bootstrap CI for a quantile of x. Returned so two checkpoints'
    numbers can be compared honestly instead of chasing sampling noise."""
    x = np.asarray(x, dtype=float)
    if len(x) < 20:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    if len(x) > max_n:
        x = rng.choice(x, size=max_n, replace=False)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    stats = np.percentile(x[idx], q, axis=1)
    return (float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2))))


# --------------------------------------------------------------------------
# environment / policy setup
# --------------------------------------------------------------------------

def env_code_sha():
    """git SHA of the env/ tree, or None outside a repo.

    E1's re-evaluation was destroyed by exactly the gap this closes: the
    lookahead floor changed what path mode MEANS between training and scoring,
    so a checkpoint was graded against semantics it had never trained on (path
    error 24.8 -> 165.9 cm, falls 5 -> 346) and nothing in the output said so.
    Config drift is visible because config.yaml is saved next to the run; code
    drift is invisible. Recording the SHA makes it visible.
    """
    import subprocess
    try:
        root = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(["git", "-C", root, "log", "-1", "--format=%H", "--", "envs"],
                             capture_output=True, text=True, timeout=10)
        return (out.stdout or "").strip() or None
    except Exception:
        return None


def evaluation_protocol_sha():
    """Git SHA of the code that defines an evaluation protocol.

    ``env_code_sha`` catches task drift, while this additionally catches a
    changed evaluator/timeout or runner interface.  Cross-arm HBatch reports
    refuse to compare suites without the same nonempty protocol SHA.
    """
    import subprocess
    try:
        root = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(
            ["git", "-C", root, "log", "-1", "--format=%H", "--",
             "eval_goal_pose.py", "envs", "utils/runner.py", "utils/runner_v3.py"],
            capture_output=True, text=True, timeout=10)
        return (out.stdout or "").strip() or None
    except Exception:
        return None


def _stable_protocol_sha(value):
    """Hash an effective, JSON-like evaluation configuration deterministically."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _apply_hbatch_common_eval(cfg, task):
    """Replace arm-specific training randomization with one held-out test profile.

    H1/H2 deliberately train with wider joint offsets and H2 with a denser
    disturbance schedule.  Those are interventions, not permission to change
    the exam.  Every final HBatch report therefore uses the same observation,
    joint-calibration and force distribution; the effective post-override
    protocol is fingerprinted later in :func:`prepare_cfg`.
    """
    if task.split("/")[-1] != "Goal_Pose_HBatch":
        return
    evaluation = cfg.setdefault("evaluation", {})
    profile = evaluation.get("hbatch_common_eval")
    if not isinstance(profile, dict):
        raise ValueError(
            "HBatch evaluation requires evaluation.hbatch_common_eval; "
            "refusing an arm-specific test distribution")
    noise = cfg.setdefault("noise", {})
    randomization = cfg.setdefault("randomization", {})
    for name, value in (profile.get("noise_overrides") or {}).items():
        noise[name] = copy.deepcopy(value)
    for name, value in (profile.get("randomization_overrides") or {}).items():
        randomization[name] = copy.deepcopy(value)
    disturbance = profile.get("disturbance")
    if not isinstance(disturbance, dict):
        raise ValueError(
            "HBatch evaluation.hbatch_common_eval.disturbance is missing")
    randomization["disturbance"] = copy.deepcopy(disturbance)


def _validate_joint_dr_probe_value(name, value):
    """Return a finite non-negative probe magnitude, preserving ``None``.

    ``prepare_cfg`` is also imported by checkpoint-selection tools, so input
    validation cannot live only in argparse.  In particular, NaN would make a
    nominally reproducible protocol impossible to fingerprint and negative
    magnitudes would silently invert the configured range.
    """
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be a finite number >= 0".format(name))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("{} must be a finite number >= 0".format(name))
    return value


def _nonnegative_finite_float(text):
    """argparse adapter for held-out joint-DR probe magnitudes."""
    try:
        return _validate_joint_dr_probe_value("joint DR probe", text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))


def _apply_joint_dr_probe(cfg, joint_encoder_bias_rad=None,
                          joint_target_offset_rad=None,
                          init_dof_std_rad=None):
    """Apply explicit joint-DR severities after any common-eval override.

    Encoder/target magnitudes denote symmetric uniform half-widths.  Initial
    joint position denotes Gaussian standard deviation, matching the existing
    randomization schema.  ``None`` means leave that config entry untouched;
    explicit zero is therefore a useful nominal/ablation probe.
    """
    values = {
        "joint_encoder_bias_rad": _validate_joint_dr_probe_value(
            "joint_encoder_bias_rad", joint_encoder_bias_rad),
        "joint_target_offset_rad": _validate_joint_dr_probe_value(
            "joint_target_offset_rad", joint_target_offset_rad),
        "init_dof_std_rad": _validate_joint_dr_probe_value(
            "init_dof_std_rad", init_dof_std_rad),
    }
    randomization = cfg.setdefault("randomization", {})
    uniform_keys = (
        ("joint_encoder_bias_rad", "joint_encoder_bias"),
        ("joint_target_offset_rad", "joint_target_offset"),
    )
    for probe_key, config_key in uniform_keys:
        magnitude = values[probe_key]
        if magnitude is not None:
            randomization[config_key] = {
                "range": [-magnitude, magnitude],
                "operation": "additive",
                "distribution": "uniform",
            }
    if values["init_dof_std_rad"] is not None:
        randomization["init_dof_pos"] = {
            "range": [0.0, values["init_dof_std_rad"]],
            "operation": "additive",
            "distribution": "gaussian",
        }
    values["active"] = any(value is not None for value in values.values())
    cfg.setdefault("evaluation", {})["joint_dr_probe"] = copy.deepcopy(values)
    return values


# A single held-out disturbance profile, identical for every arm and harder
# than what any of them trained on. --keep_perturbations retains the ARM'S OWN
# disturbance config, which means an arm trained without disturbance is scored
# with none -- each policy graded on its own homework, which is exactly why
# E2, G2 and I1b all trained for robustness and produced no evidence of it.
# I1b trained on interval 8-14 s and 40-100 N; this is 4-8 s and 50-120 N, so
# passing it is generalisation rather than recall.
HELD_OUT_FORCE = {
    "enabled": True,
    "interval_s": [4.0, 8.0],
    "collision_share": 0.5,
    "ramp_steps": 1,
    "collision": {"force_n": [50.0, 120.0], "torque_nm": [3.0, 20.0],
                  "duration_s": [0.06, 0.12]},
    "support": {"force_n": [4.0, 10.0], "torque_nm": [0.2, 2.0],
                "duration_s": [0.5, 1.5]},
}


def prepare_cfg(cfg, task, num_envs, sim_device=None, rl_device=None,
                record_video=False, keep_perturbations=False, no_noise=False,
                stress=None, goal_pattern=None, force_visualization_probe=False,
                force_profile=None, terrain=None,
                joint_encoder_bias_rad=None, joint_target_offset_rad=None,
                init_dof_std_rad=None):
    """Apply the standard evaluation conditions to a task config, in place."""
    cfg["basic"]["task"] = task
    cfg["basic"]["headless"] = True
    cfg["env"]["num_envs"] = num_envs
    if sim_device:
        cfg["basic"]["sim_device"] = sim_device
    if rl_device:
        cfg["basic"]["rl_device"] = rl_device
    cfg["viewer"]["record_video"] = bool(record_video)
    cfg["viewer"]["record_env_idx"] = 0
    _apply_hbatch_common_eval(cfg, task)
    joint_dr_probe = _apply_joint_dr_probe(
        cfg, joint_encoder_bias_rad=joint_encoder_bias_rad,
        joint_target_offset_rad=joint_target_offset_rad,
        init_dof_std_rad=init_dof_std_rad)
    # The ground is part of the exam, not part of the policy. Every eval so far
    # took terrain from the ARM'S OWN config, so I2b_terrain was scored on 10 cm
    # trimesh rubble while I1b and I2a were scored on a flat plane -- and the
    # resulting 6.2 %p strict gap was read as "terrain costs accuracy" when it may
    # be nothing but a harder exam. Same mistake as scoring each policy under its
    # own disturbance config, which is what --force_profile exists to stop.
    if terrain:
        os.environ["EVAL_TERRAIN"] = terrain
        cfg["terrain"]["type"] = "plane" if terrain == "plane" else cfg["terrain"]["type"]
    if force_profile:
        os.environ["EVAL_FORCE_PROFILE"] = force_profile
    if force_profile == "heldout":
        # Overrides whatever the arm configured, in BOTH directions: an arm that
        # trained with disturbance gets this one instead of its own, and an arm
        # that trained without it gets it switched on. That is what makes the
        # comparison across arms mean anything.
        cfg["randomization"]["disturbance"] = json.loads(json.dumps(HELD_OUT_FORCE))
        cfg["randomization"]["kick_interval_s"] = 1.0e9
        cfg["randomization"]["push_interval_s"] = 1.0e9
        keep_perturbations = True
    if not keep_perturbations:
        cfg["randomization"]["kick_interval_s"] = 1.0e9
        cfg["randomization"]["push_interval_s"] = 1.0e9
        # v7's two-class disturbance runs off its OWN config key and per-env
        # timers, so the two interval knobs above do not touch it. Without this
        # a "clean" v7 eval would silently keep 150 N collisions switched on and
        # would not be comparable to the armA-D numbers.
        if isinstance(cfg["randomization"].get("disturbance"), dict):
            cfg["randomization"]["disturbance"]["enabled"] = False
    else:
        # Training ramps disturbance probability in gradually.  A fresh eval
        # process starts common_step_counter at zero, so retaining that ramp
        # would accidentally score an almost disturbance-free run.  Evaluation
        # instead exercises the shared held-out HBatch distribution installed
        # above (or the task's configured terminal distribution for non-H tasks).
        disturbance = cfg["randomization"].get("disturbance")
        if isinstance(disturbance, dict) and disturbance.get("enabled", False):
            disturbance["ramp_steps"] = 1
    if force_visualization_probe:
        disturbance = cfg["randomization"].get("disturbance")
        if not (keep_perturbations and isinstance(disturbance, dict)):
            raise ValueError(
                "--force_visualization_probe requires --keep_perturbations and "
                "the HBatch disturbance model")
        # Deterministic video-only protocol: long, gentle support forces keep
        # env0 alive, start early enough to overlap the recorded window, and
        # path-only commands guarantee the requested carrot/path visualization.
        disturbance.update({
            "enabled": True, "interval_s": [3.0, 4.0],
            "event_probability": 1.0, "ramp_steps": 1,
            "collision_share": 0.0,
        })
        cfg["commands"]["goal_mode_mixture"] = {
            "waypoint": 0.0, "path": 1.0}
        cfg.setdefault("evaluation", {})["force_visualization_probe"] = True
    if no_noise:
        cfg["noise"] = {}

    if stress == "jitter":
        # BT thrash / ball re-detection worst case: the TRUE goal is redrawn
        # uniformly in a +-3 m box on every control step (50 Hz). Position error
        # is undefined here -- the goal's expectation is the robot's own
        # neighbourhood -- so this run is scored on "does it stay upright and
        # non-divergent", not on the gates.
        c = cfg["commands"]
        c["resampling_time_s"] = [cfg["control"]["decimation"] * cfg["sim"]["dt"],
                                  2 * cfg["control"]["decimation"] * cfg["sim"]["dt"]]
        c["goal_dx"] = [-3.0, 3.0]
        c["goal_dy"] = [-3.0, 3.0]
        if isinstance(c.get("goal_categories"), dict):
            c["goal_categories"]["enabled"] = False   # else 30% of draws are zero-distance
        if isinstance(c.get("goal_mode_mixture"), dict):
            c["goal_mode_mixture"] = {"waypoint": 1.0, "path": 0.0}  # jitter is a waypoint stress
    if goal_pattern:
        cfg.setdefault("evaluation", {})["goal_pattern"] = goal_pattern
        c = cfg["commands"]
        if isinstance(c.get("goal_categories"), dict):
            c["goal_categories"]["enabled"] = False
        if isinstance(c.get("goal_mode_mixture"), dict):
            c["goal_mode_mixture"] = {"waypoint": 1.0, "path": 0.0}
        c["goal_dtheta"] = [0.0, 0.0]
        if goal_pattern == "lateral":
            c["goal_dx"], c["goal_dy"] = [0.0, 0.0], [-2.0, 2.0]
        elif goal_pattern == "reverse":
            c["goal_dx"], c["goal_dy"] = [-2.0, -1.0], [0.0, 0.0]
        elif goal_pattern == "forward_hold":
            # Steady-state speed.  On the normal task the robot covers 2.25 m and
            # stops, so acceleration and braking own most of the segment and no
            # cruise is ever held long enough to read (8-19: even the ideal run
            # holds 1.3 m/s for about a second).  Putting the goal back 2 m ahead
            # on a fixed cadence means it is never reached, never decelerates to
            # a stop, and simply walks -- so body_speed becomes the policy's
            # sustained speed rather than a transient.
            #
            # 2 m is the trained goal range (goal_dx +-2), so the observation
            # stays in distribution; a far goal like 20 m would not.
            # 1 s of cadence keeps the remaining distance between roughly 1.2
            # and 2.0 m -- always approaching, never arriving.
            c["goal_dx"], c["goal_dy"] = [2.0, 2.0], [0.0, 0.0]
            # Not a constant: torch.randint's high is exclusive, so equal
            # endpoints raise, and a constant interval would also resample all
            # envs in lockstep -- every goal jumping on the same tick would put
            # a synchronised dip in the speed histogram that is an artefact of
            # the probe, not the policy.  0.8-1.2 s spreads them out.
            c["resampling_time_s"] = [0.8, 1.2]
    # Fingerprint what the simulator will actually sample after every CLI and
    # common-profile override.  Cross-arm aggregation rejects a single missing
    # or unequal hash, preventing another comparison of unequal force/noise/DR
    # protocols while still calling them "the same evaluation".
    effective_protocol = {
        "task": task,
        "sim_dt": cfg.get("sim", {}).get("dt"),
        "control_decimation": cfg.get("control", {}).get("decimation"),
        "commands": cfg.get("commands", {}),
        "noise": cfg.get("noise", {}),
        "randomization": cfg.get("randomization", {}),
        "stress": stress,
        "goal_pattern": goal_pattern,
        "record_video": bool(record_video),
        "force_visualization_probe": bool(force_visualization_probe),
        "joint_dr_probe": copy.deepcopy(joint_dr_probe),
    }
    evaluation = cfg.setdefault("evaluation", {})
    evaluation["effective_eval_protocol_sha"] = _stable_protocol_sha(
        effective_protocol)
    evaluation["effective_disturbance_protocol"] = copy.deepcopy(
        cfg.get("randomization", {}).get("disturbance") or {"enabled": False})
    return cfg


def build_env(cfg, task):
    task_class = get_task_class(task.split("/")[-1])
    if task_class is None:
        raise ValueError("unknown task: {}".format(task))
    return task_class(cfg)


def load_policy(checkpoint, env, device, model=None, verbose=True,
                restore_task_state=False):
    """Load checkpoint weights, reusing `model` if given (so a sweep over many
    checkpoints does not rebuild the network each time).

    Evaluation normally keeps the config-defined task distribution fixed so
    paired checkpoints see the same protocol.  ``restore_task_state`` is an
    explicit native-resume diagnostic for inspecting the checkpoint's own
    curriculum distribution; it must not be silently enabled in a selector.
    """
    if model is None:
        model = ActorCritic(env.num_actions, env.num_obs, env.num_privileged_obs).to(device)
    model_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    load_result = model.load_state_dict(model_dict["model"], strict=False)
    env.eval_task_state_protocol = "config_fixed"
    if restore_task_state:
        if hasattr(env, "load_checkpoint_state") and "env_state" in model_dict:
            env.load_checkpoint_state(model_dict["env_state"])
            env.eval_task_state_protocol = "checkpoint_restored"
        elif verbose:
            print("NOTE: checkpoint has no restorable task curriculum state")
            env.eval_task_state_protocol = "requested_but_missing"
        else:
            env.eval_task_state_protocol = "requested_but_missing"
    if verbose and (load_result.missing_keys or load_result.unexpected_keys):
        print("WARNING: partial checkpoint load ({} missing, {} unexpected keys)".format(
            len(load_result.missing_keys), len(load_result.unexpected_keys)))
    model.eval()
    return model


# --------------------------------------------------------------------------
# rollout
# --------------------------------------------------------------------------

def rollout(env, model, total_steps, device, stochastic=False, record_video=False,
            record_video_s=8.0, progress_every=500, progress_prefix="  ", stress=None,
            cfg_speed_window_s=0.2):
    """Roll the policy and collect one record per completed goal segment.

    Per segment we keep not just the final error but the provenance needed to
    diagnose it: which goal category it was, how far away the goal was sampled,
    how long the policy had, the closest it ever got, and whether the residual
    error is along or across the approach direction.
    """
    instrumented = hasattr(env, "goal_start_pos") and hasattr(env, "goal_start_step")
    has_segment_id = hasattr(env, "goal_segment_id")
    has_path_speed = hasattr(env, "path_speed")
    has_path_lag = hasattr(env, "path_lag")

    keys = ("pos_err", "head_err", "speed", "category", "start_dist", "duration_s",
            "min_dist", "along", "cross", "peak_speed", "mean_speed", "cmd_speed",
            "time_above_1p0", "time_above_1p3", "cruise_1p3",
            "path_lag", "time_to_0p5_s", "time_to_0p8_s", "time_to_1p0_s",
            "direction_time_to_0p5_s", "direction_time_to_0p8_s",
            "direction_time_to_1p0_s",
            "min_speed_first_2s", "initial_speed_mps",
            "initial_goal_bearing_rad", "switch_gait_phase")
    seg = {k: [] for k in keys}
    # Whole-rollout body-speed histogram, over every env and every control step.
    # The per-segment "final_speed" only says how well it stops; this says how
    # fast it can actually go, which is the number the MASTERPLAN target is in.
    speed_hist = np.zeros(400, dtype=np.int64)  # 0..4 m/s in 1 cm/s bins
    speed_hist_max = 4.0
    # Body angular rate: under goal jitter the failure mode is not "wrong place"
    # but "shaking itself apart", and |omega| is what shows that.
    angvel_hist = np.zeros(400, dtype=np.int64)  # 0..8 rad/s
    angvel_hist_max = 8.0
    upright_steps = 0
    # Phase-conditioned stability metrics distinguish useful acceleration lean
    # from the steady high-speed wobble H2 is meant to remove.
    stability_hist = {
        "accel_pitch_deg": np.zeros(180, dtype=np.int64),
        "cruise_pitch_deg": np.zeros(180, dtype=np.int64),
        "cruise_roll_deg": np.zeros(180, dtype=np.int64),
        "cruise_ang_xy": np.zeros(400, dtype=np.int64),
        "cruise_z_vel": np.zeros(300, dtype=np.int64),
    }
    stability_counts = {"valid": 0, "accel": 0, "cruise": 0}
    mirror_error_hist = np.zeros(400, dtype=np.int64)  # RMS action error, 0..2
    mirror_error_hist_max = 2.0
    # First-contact gait evidence for H1 symmetry and the H3 heel-placement
    # ablation.  Fixed histograms keep memory bounded during long vectorized
    # rollouts while retaining the distribution, not just a final snapshot.
    touchdown_hist = {
        "heel_x_body": np.zeros(800, dtype=np.int64),
        "heel_x_left": np.zeros(800, dtype=np.int64),
        "heel_x_right": np.zeros(800, dtype=np.int64),
        "precontact_down_speed": np.zeros(400, dtype=np.int64),
        "contact_force": np.zeros(500, dtype=np.int64),
    }
    touchdown_bounds = {
        "heel_x_body": [-0.40, 0.40],
        "precontact_down_speed": [0.0, 4.0],
        "contact_force": [0.0, 1000.0],
    }
    touchdown_counts = {
        "samples": 0, "ahead": 0, "within_target_sigma": 0,
        "overstride": 0,
    }
    # Step-level path telemetry must stay bounded even for long evaluations.
    # Keep fixed-size streaming histograms rather than one value per env-step.
    # The distance range is derived from the configured hard leash cap, with
    # enough headroom to expose (rather than immediately clip) leash violations.
    path_cfg = env.cfg.get("commands", {}).get("path", {}) or {}
    path_ratio_hist_max = 4.0
    path_distance_hist_max = max(
        4.0, 2.0 * float(path_cfg.get("lookahead_max_m", 3.5)))
    path_step_hist = {
        "gap_over_lookahead": np.zeros(400, dtype=np.int64),
        "gap_m": np.zeros(800, dtype=np.int64),
        "lookahead_m": np.zeros(800, dtype=np.int64),
        "leash_m": np.zeros(800, dtype=np.int64),
        "floor_deficit_m": np.zeros(800, dtype=np.int64),
        "behind_lag_m": np.zeros(800, dtype=np.int64),
    }
    path_step_counts = {
        "samples": 0,
        "below_0p75": 0,
        "outside_leash": 0,
        "dwell_resume_recovery": 0,
        "steady_samples": 0,
        "steady_below_0p75": 0,
    }
    eval_accel_filtered = torch.zeros(env.num_envs, 3, device=env.device)
    force_events = force_active_steps = force_context_falls = 0
    force_cancelled_before_application = 0
    force_records = {
        key: [] for key in (
            "kind", "path", "duration_observed_s", "impulse_ns",
            "torque_impulse_nms", "max_tilt_deg", "speed_loss_mps",
            "baseline_goal_progress_speed_mps",
            "recovery_eligible", "recovery_censored_by_goal_protocol",
            "outcome_censored_by_episode_timeout",
            "outcome_censored_by_rollout_end", "recovery_90_s",
            "survived_2s", "survived_5s",
            # Scenario provenance and delivery audit.  IDs are accompanied by
            # immutable name tables in disturbance_eval so JSON stays compact.
            "scenario_id", "height_tier_id", "body_index",
            "direction_local_deg", "contact_offset_z_m",
            "expected_impulse_ns", "expected_torque_impulse_nms",
            "submitted_impulse_ns", "submitted_torque_impulse_nms")
    }
    fall_ctx = {k: [] for k in ("category", "goal_dist", "t_into_segment", "start_dist")}
    falls = 0
    censored = 0
    video_done = not record_video
    overlay_states = []

    # ---- speed is measured as d(pose)/dt, not as the trunk's instantaneous
    # linear velocity. root_states[:, 7:9] is the BASE LINK's velocity, which
    # carries the per-step sway of the gait: the trunk surges and yaws within
    # every stride even when the robot is travelling at a constant speed, so its
    # p99 and max report the sway, not the travel. The goal is an SE(2) pose, so
    # the honest speed is how fast that pose changes -- differenced over a short
    # window so one stride's oscillation averages out.
    pose_win = max(1, int(round(float(cfg_speed_window_s) / env.dt)))
    pose_hist = torch.zeros(pose_win + 1, env.num_envs, 3, device=env.device)
    pose_fill = 0

    # closest approach to the goal currently being pursued, per env
    min_dist = torch.full((env.num_envs,), float("inf"), device=env.device)
    # Swing apex height, per foot.  A foot only trips on what its LOWEST corner
    # fails to clear, and on a plane feet_swing stops paying once that corner is
    # 1 cm up -- so nothing in training asks for more and nothing in eval has
    # ever looked.  Terrain arms exist precisely to raise this, and without it
    # they get adopted or rejected on strict success, which is accuracy, not the
    # thing being bought.  Recorded at touchdown so each swing contributes once.
    # Support state and load sharing.  These exist because the FALL COUNT is not
    # measurable at the exposure this project can afford -- 8-12a retired a gate
    # for exactly that, and 4 falls in 13,922 segments cannot rank anything.  The
    # answer to a rare event is not more of it; it is a continuous quantity that
    # moves before the event does.  Single-support duration is balance margin
    # spent, and load asymmetry is that margin being spent unevenly.
    support_steps = np.zeros(3, dtype=np.int64)       # [비행, 단일지지, 양발]
    ss_run = None                                     # 진행 중인 단일지지 길이(스텝)
    ss_hist = np.zeros(150, dtype=np.int64)           # 0..3 s, 20 ms bins
    ss_hist_max_s = 3.0
    load_asym_hist = np.zeros(100, dtype=np.int64)    # 0..1
    swing_apex = None
    swing_apex_hist = np.zeros(200, dtype=np.int64)   # 0..20 cm, 1 mm bins
    swing_apex_max_m = 0.20
    if hasattr(env, "feet_clearance"):
        swing_apex = torch.zeros_like(env.feet_clearance)
    prev_single = None
    has_contact_forces = (hasattr(env, "contact_forces")
                          and hasattr(env, "feet_indices"))
    if hasattr(env, "feet_contact"):
        ss_run = torch.zeros(env.num_envs, device=env.device)
        prev_single = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # peak and accumulated body speed within the segment currently in progress
    peak_speed = torch.zeros(env.num_envs, device=env.device)
    sum_speed = torch.zeros(env.num_envs, device=env.device)
    n_speed = torch.zeros(env.num_envs, device=env.device)
    # P2 is a SUSTAINED speed, and a peak cannot answer it.  A 1.4 m peak on a
    # 0.7 m hop is an acceleration transient: the robot is speeding up and then
    # immediately braking, and it never cruised at anything.  Time above the
    # threshold answers "how long", and the longest CONTINUOUS stretch answers
    # "in one piece or in scraps" -- the same total is a cruise or a dozen
    # unrelated bursts, and only the second is what P2 asks for.
    # Counted in STEPS, not seconds.  Accumulating env.dt in float32 lands 50
    # steps at 0.99999958 s, which is below 1.0 as a raw float but rounds to
    # "1.0000" in segments.csv -- so the same segment counted as a 1 s cruise
    # from the CSV and not from report.json.  Step counts are exact in float32
    # (a 120 s run is 6000 of them), and the single multiply by dt happens once,
    # in float64, at extraction.
    n_above_1p0 = torch.zeros(env.num_envs, device=env.device)
    n_above_1p3 = torch.zeros(env.num_envs, device=env.device)
    run_above_1p3 = torch.zeros(env.num_envs, device=env.device)
    cruise_1p3 = torch.zeros(env.num_envs, device=env.device)

    obs, _ = env.reset()
    obs = obs.to(device)

    # Per-segment transient response.  These make abrupt lateral/reverse goals
    # and acceleration regressions measurable instead of relying on a final
    # speed snapshot after the transient has already ended.
    response_elapsed = torch.zeros(env.num_envs, device=env.device)
    response_min_speed_2s = torch.full(
        (env.num_envs,), float("inf"), device=env.device)
    response_t05 = torch.full((env.num_envs,), float("nan"), device=env.device)
    response_t08 = torch.full((env.num_envs,), float("nan"), device=env.device)
    response_t10 = torch.full((env.num_envs,), float("nan"), device=env.device)
    direction_t05 = torch.full((env.num_envs,), float("nan"), device=env.device)
    direction_t08 = torch.full((env.num_envs,), float("nan"), device=env.device)
    direction_t10 = torch.full((env.num_envs,), float("nan"), device=env.device)
    response_initial_speed = torch.norm(env.filtered_lin_vel[:, :2], dim=-1)
    response_goal_dir_world = env.goal_pos_world - env.base_pos[:, :2]
    response_goal_dir_world /= torch.norm(
        response_goal_dir_world, dim=-1, keepdim=True).clamp(min=1.0e-6)
    response_initial_bearing = torch.atan2(env.goal_rel_pos[:, 1], env.goal_rel_pos[:, 0])
    response_initial_phase = env.gait_process.clone()

    # One disturbance can be followed for five seconds because the H configs'
    # minimum inter-event interval is six seconds.  Keep the tracker generic;
    # overlapping events would simply replace an env's unfinished record and
    # are rejected by the event-count/record-count mismatch in the report.
    force_live = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    force_start_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    force_end_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    force_baseline_speed = torch.zeros(env.num_envs, device=env.device)
    force_min_speed = torch.zeros(env.num_envs, device=env.device)
    force_kind = torch.zeros(env.num_envs, dtype=torch.int8, device=env.device)
    force_path = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    force_impulse = torch.zeros(env.num_envs, device=env.device)
    force_torque_impulse = torch.zeros(env.num_envs, device=env.device)
    force_max_tilt = torch.zeros(env.num_envs, device=env.device)
    force_ended = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    force_recovery = torch.full((env.num_envs,), float("nan"), device=env.device)
    force_recovery_candidate = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device)
    force_recovery_valid = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device)
    force_recovery_censored = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device)
    force_outcome_censored = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device)
    force_rollout_censored = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device)
    force_goal_segment = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device)
    force_scenario_id = torch.zeros(
        env.num_envs, dtype=torch.int8, device=env.device)
    force_height_tier_id = torch.full(
        (env.num_envs,), -1, dtype=torch.int8, device=env.device)
    force_body_index = torch.full(
        (env.num_envs,), -1, dtype=torch.long, device=env.device)
    force_direction_local = torch.zeros(env.num_envs, 3, device=env.device)
    force_contact_offset_local = torch.zeros(env.num_envs, 3, device=env.device)
    force_expected_impulse = torch.full(
        (env.num_envs,), float("nan"), device=env.device)
    force_expected_torque_impulse = torch.full(
        (env.num_envs,), float("nan"), device=env.device)

    def close_force_records(mask, age_s):
        ids = mask.nonzero(as_tuple=False).flatten()
        if len(ids) == 0:
            return
        for idx in ids.tolist():
            age = float(age_s[idx].item())
            baseline = float(force_baseline_speed[idx].item())
            minimum = float(force_min_speed[idx].item())
            recovery_observed = bool(torch.isfinite(force_recovery[idx]).item())
            outcome_censored = bool(force_outcome_censored[idx].item())
            relevant_recovery = (bool(force_recovery_valid[idx].item())
                                 and (not outcome_censored or recovery_observed))
            recovery = float(force_recovery[idx].item()) if relevant_recovery else float("nan")
            force_records["kind"].append(int(force_kind[idx].item()))
            force_records["path"].append(bool(force_path[idx].item()))
            force_records["duration_observed_s"].append(min(age, 5.0))
            force_records["impulse_ns"].append(float(force_impulse[idx].item()))
            force_records["torque_impulse_nms"].append(float(force_torque_impulse[idx].item()))
            force_records["max_tilt_deg"].append(float(force_max_tilt[idx].item()))
            force_records["speed_loss_mps"].append(max(0.0, baseline - minimum))
            force_records["baseline_goal_progress_speed_mps"].append(baseline)
            force_records["recovery_eligible"].append(relevant_recovery)
            force_records["recovery_censored_by_goal_protocol"].append(
                bool(force_recovery_censored[idx].item()))
            force_records["outcome_censored_by_episode_timeout"].append(
                outcome_censored and not bool(force_rollout_censored[idx].item()))
            force_records["outcome_censored_by_rollout_end"].append(
                bool(force_rollout_censored[idx].item()))
            force_records["recovery_90_s"].append(recovery)
            force_records["survived_2s"].append(age >= 2.0)
            force_records["survived_5s"].append(age >= 5.0)
            force_records["scenario_id"].append(
                int(force_scenario_id[idx].item()))
            force_records["height_tier_id"].append(
                int(force_height_tier_id[idx].item()))
            force_records["body_index"].append(
                int(force_body_index[idx].item()))
            direction = force_direction_local[idx]
            force_records["direction_local_deg"].append(float(torch.rad2deg(
                torch.atan2(direction[1], direction[0])).item()))
            force_records["contact_offset_z_m"].append(float(
                force_contact_offset_local[idx, 2].item()))
            force_records["expected_impulse_ns"].append(float(
                force_expected_impulse[idx].item()))
            force_records["expected_torque_impulse_nms"].append(float(
                force_expected_torque_impulse[idx].item()))
            # The HBatch environment integrates the exact wrench tensor on
            # every physics substep.  Because the minimum inter-event interval
            # exceeds this five-second record window, this accumulator still
            # belongs to the record being closed (including fall/reset closes).
            submitted_i = float("nan")
            submitted_t = float("nan")
            if hasattr(env, "dist_event_submitted_impulse_vec"):
                submitted_i = float(torch.norm(
                    env.dist_event_submitted_impulse_vec[idx]).item())
                if (submitted_i <= 1.0e-12
                        and hasattr(env, "dist_last_submitted_impulse")):
                    submitted_i = float(
                        env.dist_last_submitted_impulse[idx].item())
            if hasattr(env, "dist_event_submitted_torque_impulse_vec"):
                submitted_t = float(torch.norm(
                    env.dist_event_submitted_torque_impulse_vec[idx]).item())
                if (submitted_t <= 1.0e-12
                        and hasattr(env, "dist_last_submitted_torque_impulse")):
                    submitted_t = float(
                        env.dist_last_submitted_torque_impulse[idx].item())
            force_records["submitted_impulse_ns"].append(submitted_i)
            force_records["submitted_torque_impulse_nms"].append(submitted_t)
        force_live[ids] = False
        force_ended[ids] = False
        force_recovery[ids] = float("nan")
        force_recovery_candidate[ids] = False
        force_recovery_valid[ids] = False
        force_recovery_censored[ids] = False
        force_outcome_censored[ids] = False
        force_rollout_censored[ids] = False

    # CUDA work is asynchronous. Synchronize only at the timing boundaries so
    # rollout_wall_s measures completed simulation/inference work without adding
    # a device-wide barrier to every control step.
    synchronize_cuda_devices(env.cfg["basic"]["sim_device"], device)
    started = time.perf_counter()

    for step_i in range(total_steps):
        # goal_dist still refers to the segment in progress; fold it in before the
        # step can replace the goal.
        torch.minimum(min_dist, env.goal_dist, out=min_dist)
        _, _, _yaw = get_euler_xyz(env.base_quat)
        pose_hist[pose_fill % (pose_win + 1), :, 0] = env.base_pos[:, 0]
        pose_hist[pose_fill % (pose_win + 1), :, 1] = env.base_pos[:, 1]
        pose_hist[pose_fill % (pose_win + 1), :, 2] = wrap(_yaw)
        pose_fill += 1
        # A window that STRADDLES A RESET differences the spawn pose against the
        # pre-reset pose, so the teleport is reported as travel. Measured on the
        # E/G batches: peak_speed reached 203 m/s and 15.7-17.6% of every run's
        # segments were contaminated -- a count that tracked
        # segments_censored_by_episode_end (761-767) almost exactly. That is the
        # metric this whole batch existed to move, so it silently invalidated the
        # headline result. episode_length_buf is zeroed by _reset_idx, so it
        # counts steps since spawn exactly; require a full window of them.
        speed_valid = env.episode_length_buf >= pose_win
        if pose_fill > pose_win:
            new = pose_hist[(pose_fill - 1) % (pose_win + 1)]
            old = pose_hist[(pose_fill - 1 - pose_win) % (pose_win + 1)]
            dxy = new[:, :2] - old[:, :2]
            cur_speed = torch.norm(dxy, dim=-1) / (pose_win * env.dt)
            cur_yawrate = wrap(new[:, 2] - old[:, 2]).abs() / (pose_win * env.dt)
            cur_speed = torch.where(speed_valid, cur_speed, torch.zeros_like(cur_speed))
            cur_yawrate = torch.where(speed_valid, cur_yawrate, torch.zeros_like(cur_yawrate))
        else:
            speed_valid = torch.zeros_like(speed_valid)
            cur_speed = torch.zeros(env.num_envs, device=env.device)
            cur_yawrate = torch.zeros(env.num_envs, device=env.device)
        torch.maximum(peak_speed, cur_speed, out=peak_speed)
        # Steps with an invalid speed estimate are not counted as slow -- they
        # are not counted at all, the same rule the mean below already follows.
        hot10 = (cur_speed >= 1.0) & speed_valid
        hot13 = (cur_speed >= 1.3) & speed_valid
        n_above_1p0 += hot10.float()
        n_above_1p3 += hot13.float()
        run_above_1p3 = torch.where(hot13, run_above_1p3 + 1.0,
                                    torch.zeros_like(run_above_1p3))
        torch.maximum(cruise_1p3, run_above_1p3, out=cruise_1p3)
        # Excluded from the mean as well, not just zeroed: a zeroed sample still
        # drags the average down by ~pose_win steps after every reset.
        sum_speed += cur_speed
        n_speed += speed_valid.float()
        # Histogram only the valid samples. Feeding the masked zeros in would put
        # a spike at 0 m/s worth ~pose_win steps per reset, which then moves the
        # median and the "fraction of time above 0.5/1.0 m/s" lines.
        valid_np = speed_valid.cpu().numpy()
        np.add.at(speed_hist,
                  np.clip((cur_speed.cpu().numpy()[valid_np] / speed_hist_max * len(speed_hist)).astype(int),
                          0, len(speed_hist) - 1), 1)
        # Stress stability needs roll/pitch motion too.  The historical harness
        # used pose-difference yaw rate and labelled it |omega|, silently hiding
        # exactly the high-speed torso wobble under investigation.
        cur_omega = torch.norm(env.base_ang_vel, dim=-1)
        np.add.at(angvel_hist,
                  np.clip((cur_omega.cpu().numpy()[valid_np] / angvel_hist_max * len(angvel_hist)).astype(int),
                          0, len(angvel_hist) - 1), 1)
        # projected_gravity z near -1 == upright; > -0.7 is ~45 deg off vertical
        upright_steps += int((env.projected_gravity[:, 2] < -0.7).sum().item())

        # Everything a terminated env needs must be snapshotted here: _reset_idx()
        # runs inside step() and zeroes base_pos/episode_length_buf for fallen envs,
        # so post-step reads would describe the fresh episode, not the failure.
        prev_goal_pos = env.goal_pos_world.clone()
        prev_goal_heading = env.goal_heading_world.clone()
        prev_segment_id = env.goal_segment_id.clone() if has_segment_id else None
        # path_speed is re-rolled at the segment boundary, so the commanded speed
        # that the finished segment was actually run under is the PREVIOUS value.
        prev_cmd_speed = env.path_speed.clone() if has_path_speed else None
        prev_path_lag = env.path_lag.clone() if has_path_lag else None
        prev_goal_dist = env.goal_dist.clone()
        prev_len = env.episode_length_buf.clone()
        prev_filtered_lin_vel = env.filtered_lin_vel.clone()
        prev_feet_contact = (env.feet_contact.clone()
                             if hasattr(env, "feet_contact") else None)
        prev_feet_world_vz = None
        if (hasattr(env, "body_states") and hasattr(env, "feet_indices")):
            prev_feet_world_vz = env.body_states[
                :, env.feet_indices, 9].clone()
        if hasattr(env, "pushing_forces"):
            # This is the wrench submitted during the physics ticks about to be
            # simulated.  The post-step tensor below is the NEXT control step's
            # wrench because HBatch schedules after physics.
            force_norm_before = torch.norm(
                env.pushing_forces, dim=-1).amax(dim=-1)
            torque_norm_before = torch.norm(
                env.pushing_torques, dim=-1).amax(dim=-1)
            force_before = force_norm_before > 1.0e-3
        else:
            force_norm_before = torch.zeros(env.num_envs, device=env.device)
            torque_norm_before = torch.zeros(env.num_envs, device=env.device)
            force_before = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device)
        prev_force_serial = (env.dist_event_serial.clone()
                             if hasattr(env, "dist_event_serial") else None)
        if instrumented:
            prev_category = env.goal_category.clone()
            prev_start_pos = env.goal_start_pos.clone()
            prev_start_step = env.goal_start_step.clone()

        # The camera is rendered before the post-physics goal update/resample.
        # Keep the command metadata from the action/frame being visualized; a
        # post-step read is one carrot tick ahead and, at a segment boundary,
        # would draw a newly re-anchored goal over the previous segment's frame.
        if not video_done:
            video_goal_pos = env.goal_pos_world[0].clone()
            video_goal_heading = env.goal_heading_world[0].clone()
            video_goal_category = int(
                env.goal_category[0].item()) if hasattr(env, "goal_category") else -1
            video_goal_segment = int(
                env.goal_segment_id[0].item()) if hasattr(env, "goal_segment_id") else -1
            video_seq_goals = (
                env.seq_goals[0].detach().cpu().numpy().copy()
                if hasattr(env, "seq_goals") and bool(env.is_seq_env[0]) else None)
            video_seq_idx = int(env.seq_idx[0].item()) if hasattr(env, "seq_idx") else 0

            # BaseTask captures the camera inside env.step(), after physics but
            # BEFORE _push_robots() installs the wrench for the next simulation
            # step.  Snapshot the already-installed wrench now so this overlay is
            # paired with the frame in which that force actually acts.  Reading it
            # after env.step() would draw every arrow one frame early and could let
            # a force created on the final recorded step produce a false video PASS.
            video_push_n = video_push_nm = 0.0
            video_force_vec = np.zeros(3, dtype=float)
            video_force_origin = env.base_pos[0].detach().cpu().numpy().copy()
            if hasattr(env, "pushing_forces"):
                body_norm = torch.norm(env.pushing_forces[0], dim=-1)
                body_idx = int(torch.argmax(body_norm).item())
                video_force_vec = env.pushing_forces[
                    0, body_idx].detach().cpu().numpy().copy()
                video_force_origin = env.body_states[
                    0, body_idx, :3].detach().cpu().numpy().copy()
                video_push_n = float(body_norm[body_idx].item())
                video_push_nm = float(torch.norm(
                    env.pushing_torques[0, body_idx, :]).item())

        with torch.no_grad():
            dist = model.act(obs)
            act = dist.sample() if stochastic else dist.loc
            if (step_i % 10 == 0 and hasattr(env, "mirror_obs")
                    and hasattr(env, "mirror_actions")):
                mirrored_mu = model.act(env.mirror_obs(obs)).loc
                equivariant_mu = env.mirror_actions(dist.loc)
                mirror_error = torch.sqrt(torch.square(
                    mirrored_mu - equivariant_mu).mean(dim=-1))
                err = mirror_error.detach().cpu().numpy()
                np.add.at(
                    mirror_error_hist,
                    np.clip((err / mirror_error_hist_max * len(mirror_error_hist)).astype(int),
                            0, len(mirror_error_hist) - 1),
                    1,
                )
        obs, _, done, infos = env.step(act.to(env.device))
        obs = obs.to(device)

        timeouts = infos["time_outs"].to(done.device)
        physical_failures = infos.get("physical_failures")
        if physical_failures is not None:
            fell = done & physical_failures.to(done.device)
        else:
            fell = done & ~timeouts
        episode_timeouts = infos.get("episode_time_outs")
        if episode_timeouts is not None:
            episode_timeouts = episode_timeouts.to(done.device)
        else:
            episode_timeouts = done & timeouts
        n_fell = int(fell.sum().item())
        falls += n_fell
        censored += int((done & episode_timeouts & ~fell).sum().item())

        # ---- acceleration / cruise separation -----------------------------
        reset_guard = float(env.cfg.get("evaluation", {}).get("reset_guard_s", 0.25))
        valid_phase = (~done) & (env.episode_length_buf.float() * env.dt >= reset_guard)
        # Match H2's phase detector: differentiate the existing low-pass body
        # velocity rather than one-step trunk velocity contaminated by gait sway.
        acc_instant = (env.filtered_lin_vel - prev_filtered_lin_vel) / env.dt
        eval_accel_filtered = 0.10 * acc_instant + 0.90 * eval_accel_filtered
        eval_accel_filtered[done] = 0.0
        acc_body = eval_accel_filtered
        acc_xy = torch.norm(acc_body[:, :2], dim=-1)
        speed_now = torch.norm(env.filtered_lin_vel[:, :2], dim=-1)
        # Direction-change probes must measure velocity TOWARD the new goal.
        # Raw speed alone falsely calls a robot that keeps running in its old
        # direction an instant success after a lateral/reverse switch.
        _, _, response_yaw = get_euler_xyz(env.base_quat)
        cy, sy = torch.cos(response_yaw), torch.sin(response_yaw)
        response_vel_world = torch.stack((
            cy * env.filtered_lin_vel[:, 0] - sy * env.filtered_lin_vel[:, 1],
            sy * env.filtered_lin_vel[:, 0] + cy * env.filtered_lin_vel[:, 1],
        ), dim=-1)
        response_closing_speed = torch.sum(
            response_vel_world * response_goal_dir_world, dim=-1)
        current_goal_direction = env.goal_pos_world - env.base_pos[:, :2]
        current_goal_direction /= torch.norm(
            current_goal_direction, dim=-1, keepdim=True).clamp(min=1.0e-6)
        current_goal_progress_speed = torch.sum(
            response_vel_world * current_goal_direction, dim=-1)
        accel_thr = float(env.cfg.get("evaluation", {}).get("steady_accel_threshold_mps2", 0.3))
        fast_thr = float(env.cfg.get("evaluation", {}).get("high_speed_threshold_mps", 0.8))
        accel_phase = valid_phase & (speed_now > 0.3) & (acc_body[:, 0] > accel_thr)
        cruise_phase = valid_phase & (speed_now >= fast_thr) & (acc_xy <= accel_thr)
        gx, gy, gz = env.projected_gravity.unbind(dim=-1)
        pitch_deg = torch.rad2deg(torch.atan2(-gx, -gz).abs())
        roll_deg = torch.rad2deg(torch.atan2(gy, -gz).abs())

        def add_hist(name, values, mask, maximum):
            arr = values[mask].detach().cpu().numpy()
            if len(arr):
                idx = np.clip((arr / maximum * len(stability_hist[name])).astype(int),
                              0, len(stability_hist[name]) - 1)
                np.add.at(stability_hist[name], idx, 1)

        add_hist("accel_pitch_deg", pitch_deg, accel_phase, 90.0)
        add_hist("cruise_pitch_deg", pitch_deg, cruise_phase, 90.0)
        add_hist("cruise_roll_deg", roll_deg, cruise_phase, 90.0)
        add_hist("cruise_ang_xy", torch.norm(env.base_ang_vel[:, :2], dim=-1), cruise_phase, 8.0)
        add_hist("cruise_z_vel", env.base_lin_vel[:, 2].abs(), cruise_phase, 3.0)
        stability_counts["valid"] += int(valid_phase.sum().item())
        stability_counts["accel"] += int(accel_phase.sum().item())
        stability_counts["cruise"] += int(cruise_phase.sum().item())

        # ---- support state / load share -----------------------------------
        if ss_run is not None:
            n_contact = env.feet_contact.sum(dim=1)
            alive = ~done
            for k in (0, 1, 2):
                support_steps[k] += int(((n_contact == k) & alive).sum().item())
            single = (n_contact == 1) & alive
            # 끝난 단일지지 구간만 기록한다. ss_run을 갱신하기 전에 읽어야
            # 그 구간의 길이가 남아 있다.
            ended = prev_single & ~single
            if bool(ended.any()):
                secs = (ss_run[ended] * env.dt).detach().cpu().numpy()
                np.add.at(ss_hist, np.clip(
                    (secs / ss_hist_max_s * len(ss_hist)).astype(int),
                    0, len(ss_hist) - 1), 1)
            ss_run = torch.where(single, ss_run + 1.0, torch.zeros_like(ss_run))
            prev_single = single
            if has_contact_forces:
                ff = torch.norm(env.contact_forces[:, env.feet_indices, :], dim=-1)
                both = (n_contact == 2) & alive
                if bool(both.any()):
                    lf, rf = ff[both, 0], ff[both, 1]
                    asym = ((lf - rf).abs() / (lf + rf).clamp(min=1e-6))
                    np.add.at(load_asym_hist, np.clip(
                        (asym.detach().cpu().numpy() * len(load_asym_hist)).astype(int),
                        0, len(load_asym_hist) - 1), 1)

        # ---- swing apex ---------------------------------------------------
        if swing_apex is not None and prev_feet_contact is not None:
            airborne = ~env.feet_contact
            swing_apex = torch.where(
                airborne, torch.maximum(swing_apex, env.feet_clearance), swing_apex)
            touchdown = env.feet_contact & ~prev_feet_contact & ~done.unsqueeze(-1)
            if bool(touchdown.any()):
                vals = swing_apex[touchdown].detach().cpu().numpy()
                idx = np.clip((vals / swing_apex_max_m * len(swing_apex_hist)).astype(int),
                              0, len(swing_apex_hist) - 1)
                np.add.at(swing_apex_hist, idx, 1)
                swing_apex = torch.where(touchdown, torch.zeros_like(swing_apex), swing_apex)

        # ---- first-contact foot placement / impact -----------------------
        if (prev_feet_contact is not None and prev_feet_world_vz is not None
                and hasattr(env, "feet_pos") and hasattr(env, "feet_quat")):
            first_contact = env.feet_contact & ~prev_feet_contact
            first_contact &= ~done.unsqueeze(-1)
            if hasattr(env, "is_path_env"):
                first_contact &= env.is_path_env.unsqueeze(-1)
            heel_cfg = env.cfg.get("rewards", {}).get("heel_strike", {}) or {}
            min_forward = float(heel_cfg.get("min_forward_speed_mps", 0.6))
            first_contact &= (env.base_lin_vel[:, 0] > min_forward).unsqueeze(-1)
            if bool(first_contact.any()):
                n_feet = len(env.feet_indices)
                heel_local = torch.tensor(
                    [-0.1015, 0.0, -0.03], device=env.device).view(1, 1, 3)
                heel_local = heel_local.expand(env.num_envs, n_feet, 3)
                heel_world = env.feet_pos + quat_rotate(
                    env.feet_quat.reshape(-1, 4),
                    heel_local.reshape(-1, 3)).reshape_as(env.feet_pos)
                heel_rel_world = heel_world - env.base_pos.unsqueeze(1)
                heel_rel_body = quat_rotate_inverse(
                    env.base_quat.unsqueeze(1).expand(
                        -1, n_feet, -1).reshape(-1, 4),
                    heel_rel_world.reshape(-1, 3)).reshape_as(heel_rel_world)
                heel_x = heel_rel_body[:, :, 0]
                target = (float(heel_cfg.get("velocity_gain_s", 0.08))
                          * env.base_lin_vel[:, 0]).clamp(
                              min=float(heel_cfg.get("target_min_m", 0.02)),
                              max=float(heel_cfg.get("target_max_m", 0.12)))
                sigma = float(heel_cfg.get("sigma_m", 0.04))
                down_speed = (-prev_feet_world_vz).clamp(min=0.0)
                contact_force = torch.norm(
                    env.contact_forces[:, env.feet_indices, :], dim=-1)

                hx = heel_x[first_contact].detach().cpu().numpy()
                ds = down_speed[first_contact].detach().cpu().numpy()
                cf = contact_force[first_contact].detach().cpu().numpy()
                target_error = torch.abs(heel_x - target.unsqueeze(-1))
                touchdown_counts["samples"] += int(first_contact.sum().item())
                touchdown_counts["ahead"] += int(
                    ((heel_x > 0.0) & first_contact).sum().item())
                touchdown_counts["within_target_sigma"] += int(
                    ((target_error <= sigma) & first_contact).sum().item())
                overstride_limit = float(
                    heel_cfg.get("target_max_m", 0.12)) + 2.0 * sigma
                touchdown_counts["overstride"] += int(
                    ((heel_x > overstride_limit) & first_contact).sum().item())

                def add_touchdown_hist(name, values, lo, hi):
                    hist = touchdown_hist[name]
                    idx = np.clip(
                        ((values - lo) / (hi - lo) * len(hist)).astype(int),
                        0, len(hist) - 1)
                    np.add.at(hist, idx, 1)

                hx_lo, hx_hi = touchdown_bounds["heel_x_body"]
                add_touchdown_hist("heel_x_body", hx, hx_lo, hx_hi)
                add_touchdown_hist(
                    "precontact_down_speed", ds,
                    *touchdown_bounds["precontact_down_speed"])
                add_touchdown_hist(
                    "contact_force", cf, *touchdown_bounds["contact_force"])
                for foot_i, name in enumerate(("heel_x_left", "heel_x_right")):
                    if foot_i >= n_feet:
                        break
                    foot_values = heel_x[:, foot_i][
                        first_contact[:, foot_i]].detach().cpu().numpy()
                    if len(foot_values):
                        add_touchdown_hist(name, foot_values, hx_lo, hx_hi)

        # ---- signed path floor/leash state, every running control step -----
        # Segment-end path_lag alone misses short floor collapses and leash
        # excursions.  Measure the post-step state against EACH env's sampled
        # lookahead and the exact leash construction used by GoalPoseV7.  Dwell
        # deliberately releases the floor, so it is explicitly out of scope.
        if hasattr(env, "is_path_env") and hasattr(env, "lookahead"):
            running_path = env.is_path_env & ~done & (env.lookahead > 1.0e-6)
            if hasattr(env, "path_dwell_left"):
                running_path &= env.path_dwell_left <= 0
            if bool(running_path.any()):
                gap = torch.norm(env.goal_pos_world - env.base_pos[:, :2], dim=-1)
                lookahead = env.lookahead
                leash = torch.clamp(
                    lookahead * float(path_cfg.get("leash_ratio", 1.6)),
                    max=float(path_cfg.get("lookahead_max_m", 3.5)),
                )
                leash = torch.maximum(leash, lookahead + 0.1)
                ratio = gap / lookahead.clamp(min=1.0e-6)
                floor_deficit = (lookahead - gap).clamp(min=0.0)
                behind_lag = (gap - lookahead).clamp(min=0.0)
                finite = (torch.isfinite(gap) & torch.isfinite(lookahead)
                          & torch.isfinite(leash) & torch.isfinite(ratio))
                sample_mask = running_path & finite
                if bool(sample_mask.any()):
                    recovering = getattr(
                        env, "path_floor_recovering",
                        torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
                    samples = torch.stack((
                        ratio,
                        gap,
                        lookahead,
                        leash,
                        floor_deficit,
                        behind_lag,
                        (ratio < 0.75).float(),
                        (gap > leash + 1.0e-4).float(),
                        recovering.float(),
                    ), dim=-1)[sample_mask].detach().cpu().numpy()
                    path_step_counts["samples"] += int(len(samples))
                    path_step_counts["below_0p75"] += int(samples[:, 6].sum())
                    path_step_counts["outside_leash"] += int(samples[:, 7].sum())
                    recovery = samples[:, 8] > 0.5
                    path_step_counts["dwell_resume_recovery"] += int(recovery.sum())
                    path_step_counts["steady_samples"] += int((~recovery).sum())
                    path_step_counts["steady_below_0p75"] += int(
                        ((samples[:, 6] > 0.5) & ~recovery).sum())
                    for name, column, maximum in (
                        ("gap_over_lookahead", 0, path_ratio_hist_max),
                        ("gap_m", 1, path_distance_hist_max),
                        ("lookahead_m", 2, path_distance_hist_max),
                        ("leash_m", 3, path_distance_hist_max),
                        ("floor_deficit_m", 4, path_distance_hist_max),
                        ("behind_lag_m", 5, path_distance_hist_max),
                    ):
                        hist = path_step_hist[name]
                        idx = np.clip(
                            (samples[:, column] / maximum * len(hist)).astype(int),
                            0, len(hist) - 1)
                        np.add.at(hist, idx, 1)

        # Disturbance exposure and fall counts are separate from clean gates.
        # A clean 0-fall report can never be cited as force robustness again.
        if hasattr(env, "pushing_forces"):
            force_norm = torch.norm(env.pushing_forces, dim=-1).amax(dim=-1)
            torque_norm = torch.norm(env.pushing_torques, dim=-1).amax(dim=-1)
            force_now = force_norm > 1.0e-3
            rising = force_now & ~force_before
            event_edge = ((env.dist_event_serial != prev_force_serial)
                          if prev_force_serial is not None else rising)
            # _push_robots installs a newly scheduled wrench after this step's
            # simulate() and before termination/reset.  If the env was already
            # done, _reset_idx clears it before it ever reaches physics.  Such
            # an edge is a cancelled schedule, not a collision outcome.
            applied_edge = event_edge & force_now & ~done
            force_cancelled_before_application += int(
                (event_edge & ~applied_edge).sum().item())
            force_events += int(applied_edge.sum().item())
            force_active_steps += int(force_before.sum().item())
            # Only the pre-step wrench acted in the physics that produced this
            # termination; force_now was merely installed for the next step.
            force_context_falls += int((fell & force_before).sum().item())

            if bool(applied_edge.any()):
                # H intervals are intentionally longer than the five-second
                # outcome window.  A rising edge while one is still live is a
                # config error; leave the older record live so the count
                # mismatch is visible rather than silently overwriting it.
                start = applied_edge & ~force_live
                force_live[start] = True
                force_start_step[start] = step_i + 1
                # The newly installed wrench acts on the NEXT simulate.  The
                # post-step state is therefore the actual pre-hit baseline and
                # already contains any goal reroll that happened on this step.
                force_baseline_speed[start] = current_goal_progress_speed[start].clamp(min=0.0)
                force_min_speed[start] = current_goal_progress_speed[start]
                force_path[start] = getattr(
                    env, "is_path_env",
                    torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))[start]
                dwelling = getattr(
                    env, "path_dwell_left",
                    torch.zeros(env.num_envs, dtype=torch.long, device=env.device)) > 0
                floor_recovering = getattr(
                    env, "path_floor_recovering",
                    torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
                force_recovery_candidate[start] = (
                    force_path[start]
                    & (force_baseline_speed[start] >= 0.5)
                    & ~dwelling[start]
                    & ~floor_recovering[start])
                force_recovery_valid[start] = force_recovery_candidate[start]
                force_recovery_censored[start] = False
                force_outcome_censored[start] = False
                force_rollout_censored[start] = False
                if has_segment_id:
                    force_goal_segment[start] = env.goal_segment_id[start]
                if hasattr(env, "dist_last_event_kind"):
                    force_kind[start] = env.dist_last_event_kind[start]
                elif hasattr(env, "dist_event_kind"):
                    force_kind[start] = env.dist_event_kind[start]
                else:
                    force_kind[start] = 0
                force_scenario_id[start] = 0
                force_height_tier_id[start] = -1
                force_body_index[start] = -1
                force_direction_local[start] = 0.0
                force_contact_offset_local[start] = 0.0
                force_expected_impulse[start] = float("nan")
                force_expected_torque_impulse[start] = float("nan")
                if hasattr(env, "dist_last_scenario_id"):
                    force_scenario_id[start] = env.dist_last_scenario_id[start]
                if hasattr(env, "dist_last_height_tier"):
                    force_height_tier_id[start] = env.dist_last_height_tier[start]
                if hasattr(env, "dist_active_body"):
                    force_body_index[start] = env.dist_active_body[start]
                if hasattr(env, "dist_last_direction_local"):
                    force_direction_local[start] = env.dist_last_direction_local[start]
                if hasattr(env, "dist_last_contact_offset_local"):
                    force_contact_offset_local[start] = (
                        env.dist_last_contact_offset_local[start])
                if hasattr(env, "dist_last_expected_impulse"):
                    force_expected_impulse[start] = (
                        env.dist_last_expected_impulse[start])
                if hasattr(env, "dist_last_expected_torque_impulse"):
                    force_expected_torque_impulse[start] = (
                        env.dist_last_expected_torque_impulse[start])
                force_impulse[start] = 0.0
                force_torque_impulse[start] = 0.0
                force_max_tilt[start] = 0.0
                force_ended[start] = False
                force_recovery[start] = float("nan")

            active_record = force_live
            # Integrate only the wrench that acted in the just-completed
            # physics interval.  Counting force_now here would add one phantom
            # control step at event creation and close the five-second outcome
            # window one step early.
            force_impulse[active_record] += (
                force_norm_before[active_record] * env.dt)
            force_torque_impulse[active_record] += (
                torque_norm_before[active_record] * env.dt)
            not_recovered = active_record & torch.isnan(force_recovery)
            force_min_speed[not_recovered] = torch.minimum(
                force_min_speed[not_recovered],
                current_goal_progress_speed[not_recovered])
            tilt_deg = torch.rad2deg(torch.acos(
                (-env.projected_gravity[:, 2]).clamp(-1.0, 1.0)))
            force_max_tilt[active_record] = torch.maximum(
                force_max_tilt[active_record], tilt_deg[active_record])

            # Recovery speed is meaningful only while the event's command
            # protocol remains the same.  A 4-8 s path reroll or a dwell can
            # otherwise make motion toward a new carrot look like recovery from
            # the old hit.  Keep survival tracking alive, but censor recovery.
            recovery_protocol_changed = torch.zeros_like(active_record)
            if has_segment_id:
                recovery_protocol_changed |= (
                    active_record
                    & (env.goal_segment_id != force_goal_segment))
            recovery_protocol_changed |= active_record & ~getattr(
                env, "is_path_env",
                torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
            if hasattr(env, "path_dwell_left"):
                recovery_protocol_changed |= active_record & (env.path_dwell_left > 0)
            if hasattr(env, "path_floor_recovering"):
                recovery_protocol_changed |= (
                    active_record & env.path_floor_recovering)
            # A physical fall is an observed recovery failure, not a command
            # censor.  Post-reset goal/path fields describe the new episode and
            # must not remove the failed hit from the recovery denominator.
            recovery_protocol_changed &= ~done
            newly_censored = (recovery_protocol_changed
                              & force_recovery_candidate
                              & force_recovery_valid
                              & torch.isnan(force_recovery))
            force_recovery_censored[newly_censored] = True
            force_recovery_valid[newly_censored] = False

            ended_now = active_record & force_before & ~force_now & ~force_ended
            force_ended[ended_now] = True
            force_end_step[ended_now] = step_i
            recovery_now = (active_record & ~done & force_ended & torch.isnan(force_recovery)
                            & force_recovery_valid
                            & (current_goal_progress_speed
                               >= 0.9 * force_baseline_speed)
                            & (tilt_deg <= 20.0))
            force_recovery[recovery_now] = (
                step_i - force_end_step[recovery_now]).float() * env.dt

            age_s = (step_i - force_start_step + 1).float() * env.dt
            timeout_censor = active_record & done & episode_timeouts & ~fell
            force_outcome_censored[timeout_censor] = True
            close_force_records(
                (active_record & (age_s >= 5.0))
                | (active_record & fell)
                | timeout_censor,
                age_s)
        if n_fell and instrumented:
            fids = fell.nonzero(as_tuple=False).flatten()
            elapsed = (prev_len[fids] + 1 - prev_start_step[fids]).clamp(min=0).float() * env.dt
            fall_ctx["category"].extend(prev_category[fids].cpu().tolist())
            fall_ctx["goal_dist"].extend(prev_goal_dist[fids].cpu().tolist())
            fall_ctx["t_into_segment"].extend(elapsed.cpu().tolist())
            fall_ctx["start_dist"].extend(
                torch.norm(prev_goal_pos[fids] - prev_start_pos[fids], dim=-1).cpu().tolist())

        if has_segment_id:
            # v7's path mode moves the goal every step, so "the goal moved" no
            # longer means "the segment ended"; the env tells us explicitly.
            changed = env.goal_segment_id != prev_segment_id
        else:
            changed = (env.goal_pos_world != prev_goal_pos).any(dim=1) | (env.goal_heading_world != prev_goal_heading)
        completed = changed & ~done
        if stress:
            # Goals are redrawn every control step, so a "segment" is one step
            # long and every per-segment statistic is meaningless. Skip them.
            completed = completed & False

        # Transient speed response belongs to the segment that was active when
        # this action was chosen.  Update it before closing a changed segment,
        # then reset from the newly sampled goal below.
        response_valid = ~done
        response_elapsed[response_valid] += env.dt
        within_2s = response_valid & (response_elapsed <= 2.0)
        response_min_speed_2s[within_2s] = torch.minimum(
            response_min_speed_2s[within_2s], speed_now[within_2s])
        for threshold, target in ((0.5, response_t05), (0.8, response_t08), (1.0, response_t10)):
            reached = response_valid & torch.isnan(target) & (speed_now >= threshold)
            target[reached] = response_elapsed[reached]
        for threshold, target in ((0.5, direction_t05), (0.8, direction_t08),
                                  (1.0, direction_t10)):
            reached = (response_valid & torch.isnan(target)
                       & (response_closing_speed >= threshold))
            target[reached] = response_elapsed[reached]

        if completed.any():
            ids = completed.nonzero(as_tuple=False).flatten()
            final_xy = env.base_pos[ids, :2]
            goal_xy = prev_goal_pos[ids]
            d = torch.norm(final_xy - goal_xy, dim=-1)
            _, _, yaw = get_euler_xyz(env.base_quat[ids])
            h = wrap(prev_goal_heading[ids] - yaw).abs()
            v = torch.norm(env.root_states[ids, 7:9], dim=-1)
            seg["pos_err"].extend(d.cpu().tolist())
            seg["head_err"].extend(h.cpu().tolist())
            seg["speed"].extend(v.cpu().tolist())
            seg["min_dist"].extend(torch.minimum(min_dist[ids], d).cpu().tolist())
            seg["peak_speed"].extend(peak_speed[ids].cpu().tolist())
            seg["mean_speed"].extend((sum_speed[ids] / n_speed[ids].clamp(min=1.0)).cpu().tolist())
            seg["time_above_1p0"].extend(
                [n * env.dt for n in n_above_1p0[ids].cpu().tolist()])
            seg["time_above_1p3"].extend(
                [n * env.dt for n in n_above_1p3[ids].cpu().tolist()])
            seg["cruise_1p3"].extend(
                [n * env.dt for n in cruise_1p3[ids].cpu().tolist()])
            if has_path_speed:
                seg["cmd_speed"].extend(prev_cmd_speed[ids].cpu().tolist())
            else:
                seg["cmd_speed"].extend([float("nan")] * len(ids))
            if has_path_lag and instrumented:
                nan = torch.full_like(prev_path_lag[ids], float("nan"))
                lag = torch.where(prev_category[ids] == CATEGORY_PATH, prev_path_lag[ids], nan)
                seg["path_lag"].extend(lag.cpu().tolist())
            else:
                seg["path_lag"].extend([float("nan")] * len(ids))
            seg["time_to_0p5_s"].extend(response_t05[ids].cpu().tolist())
            seg["time_to_0p8_s"].extend(response_t08[ids].cpu().tolist())
            seg["time_to_1p0_s"].extend(response_t10[ids].cpu().tolist())
            seg["direction_time_to_0p5_s"].extend(direction_t05[ids].cpu().tolist())
            seg["direction_time_to_0p8_s"].extend(direction_t08[ids].cpu().tolist())
            seg["direction_time_to_1p0_s"].extend(direction_t10[ids].cpu().tolist())
            seg["min_speed_first_2s"].extend(response_min_speed_2s[ids].cpu().tolist())
            seg["initial_speed_mps"].extend(response_initial_speed[ids].cpu().tolist())
            seg["initial_goal_bearing_rad"].extend(
                response_initial_bearing[ids].cpu().tolist())
            seg["switch_gait_phase"].extend(response_initial_phase[ids].cpu().tolist())

            if instrumented:
                approach = goal_xy - prev_start_pos[ids]
                length = torch.norm(approach, dim=-1)
                err = final_xy - goal_xy
                safe = length.clamp(min=1.0e-6)
                # + = stopped past the goal, - = stopped short of it
                along = (err[:, 0] * approach[:, 0] + err[:, 1] * approach[:, 1]) / safe
                cross = (err[:, 0] * approach[:, 1] - err[:, 1] * approach[:, 0]).abs() / safe
                degenerate = length < 1.0e-3  # stand goals have no approach direction
                nan = torch.full_like(along, float("nan"))
                seg["along"].extend(torch.where(degenerate, nan, along).cpu().tolist())
                seg["cross"].extend(torch.where(degenerate, nan, cross).cpu().tolist())
                seg["category"].extend(prev_category[ids].cpu().tolist())
                seg["start_dist"].extend(length.cpu().tolist())
                seg["duration_s"].extend(
                    ((prev_len[ids] + 1 - prev_start_step[ids]).clamp(min=1).float() * env.dt).cpu().tolist())
            else:
                nan_list = [float("nan")] * len(ids)
                seg["along"].extend(nan_list)
                seg["cross"].extend(nan_list)
                seg["category"].extend([-1] * len(ids))
                seg["start_dist"].extend(nan_list)
                seg["duration_s"].extend(nan_list)

        # reset the per-segment trackers for every env that got a new goal
        if changed.any() or done.any():
            stale = changed | done
            min_dist[stale] = float("inf")
            peak_speed[stale] = 0.0
            sum_speed[stale] = 0.0
            n_speed[stale] = 0.0
            n_above_1p0[stale] = 0.0
            n_above_1p3[stale] = 0.0
            run_above_1p3[stale] = 0.0
            cruise_1p3[stale] = 0.0
            response_elapsed[stale] = 0.0
            response_min_speed_2s[stale] = float("inf")
            response_t05[stale] = float("nan")
            response_t08[stale] = float("nan")
            response_t10[stale] = float("nan")
            direction_t05[stale] = float("nan")
            direction_t08[stale] = float("nan")
            direction_t10[stale] = float("nan")
            response_initial_speed[stale] = torch.norm(
                env.filtered_lin_vel[stale, :2], dim=-1)
            new_goal_delta = env.goal_pos_world[stale] - env.base_pos[stale, :2]
            response_goal_dir_world[stale] = new_goal_delta / torch.norm(
                new_goal_delta, dim=-1, keepdim=True).clamp(min=1.0e-6)
            response_initial_bearing[stale] = torch.atan2(
                env.goal_rel_pos[stale, 1], env.goal_rel_pos[stale, 0])
            response_initial_phase[stale] = env.gait_process[stale]

        if not video_done:
            _, _, yaw_all = get_euler_xyz(env.base_quat[0:1])
            video_goal_dist = float(torch.norm(
                video_goal_pos - env.base_pos[0, :2]).item())
            video_heading_error = float(torch.rad2deg(torch.abs(wrap(
                video_goal_heading - yaw_all[0]))).item())
            overlay_states.append((
                env.base_pos[0, :2].cpu().numpy().copy(),
                float(wrap(yaw_all)[0].item()),
                video_goal_pos.cpu().numpy().copy(),
                float(video_goal_heading.item()),
                float(env.base_lin_vel[0, 0].item()),
                float(env.base_lin_vel[0, 1].item()),
                float(env.base_ang_vel[0, 2].item()),
                video_goal_dist,
                video_heading_error,
                video_push_n,
                video_push_nm,
                video_seq_goals,
                video_seq_idx,
                video_force_vec,
                video_force_origin,
                video_goal_category,
                video_goal_segment,
            ))
            if (step_i + 1) * env.dt >= record_video_s:
                env.cfg["viewer"]["record_video"] = False  # stop accumulating frames (memory)
                video_done = True

        if progress_every and (step_i + 1) % progress_every == 0:
            elapsed = time.perf_counter() - started
            steps_per_s = (step_i + 1) / max(elapsed, 1.0e-9)
            eta_s = (total_steps - step_i - 1) / max(steps_per_s, 1.0e-9)
            print("{}step {}/{} — segments: {}, falls: {}, wall {:.1f}s, ETA {:.1f}s".format(
                progress_prefix, step_i + 1, total_steps, len(seg["pos_err"]), falls, elapsed, eta_s))

    synchronize_cuda_devices(env.cfg["basic"]["sim_device"], device)
    wall_s = time.perf_counter() - started

    # The end of a finite rollout is ordinary right-censoring, not an
    # overlapping/lost event.  Close every still-live record with exactly the
    # horizon it was observable for so 2 s and 5 s survival use honest,
    # separately eligible denominators.
    if bool(force_live.any()):
        rollout_age_s = (
            total_steps - force_start_step).clamp(min=0).float() * env.dt
        force_outcome_censored[force_live] = True
        force_rollout_censored[force_live] = True
        close_force_records(force_live.clone(), rollout_age_s)

    out = {k: np.asarray(v, dtype=float) for k, v in seg.items()}
    out["category"] = out["category"].astype(int)
    out["falls"] = falls
    out["censored"] = censored
    out["fall_ctx"] = {k: np.asarray(v, dtype=float) for k, v in fall_ctx.items()}
    out["rollout_wall_s"] = wall_s
    out["total_steps"] = total_steps
    out["instrumented"] = instrumented
    out["overlay_states"] = overlay_states
    out["support_steps"] = support_steps
    out["ss_hist"] = ss_hist
    out["ss_hist_max_s"] = ss_hist_max_s
    out["load_asym_hist"] = load_asym_hist
    out["swing_apex_hist"] = swing_apex_hist
    out["swing_apex_max_m"] = swing_apex_max_m
    out["speed_hist"] = speed_hist
    out["speed_hist_max"] = speed_hist_max
    out["angvel_hist"] = angvel_hist
    out["angvel_hist_max"] = angvel_hist_max
    out["v7_extras"] = dict(getattr(env, "extras", {}).get("v7", {}) or {})
    out["upright_share"] = upright_steps / float(max(total_steps * env.num_envs, 1))
    out["env_minutes"] = total_steps * env.dt * env.num_envs / 60.0
    out["stability_hist"] = stability_hist
    out["stability_counts"] = stability_counts
    out["mirror_error_hist"] = mirror_error_hist
    out["mirror_error_hist_max"] = mirror_error_hist_max
    out["touchdown_hist"] = touchdown_hist
    out["touchdown_bounds"] = touchdown_bounds
    out["touchdown_counts"] = touchdown_counts
    out["path_step_hist"] = path_step_hist
    out["path_step_hist_max"] = {
        "gap_over_lookahead": path_ratio_hist_max,
        "distance_m": path_distance_hist_max,
    }
    out["path_step_counts"] = path_step_counts

    # These are the frames that will actually be paired and written below
    # (`zip(camera_frames, overlay_states)`).  Counting overlay samples alone
    # would falsely claim visibility when headless graphics produced no images.
    captured_frames = (len(env.camera_frames)
                       if record_video and hasattr(env, "camera_frames") else 0)
    video_recorded_frames = min(captured_frames, len(overlay_states))
    video_force_active_frames = sum(
        1 for st in overlay_states[:video_recorded_frames]
        if len(st) >= 10 and float(st[9]) > 1.0e-3)
    video_force_arrow_drawn_frames = 0
    video_path_frames = sum(
        1 for st in overlay_states[:video_recorded_frames]
        if len(st) >= 16 and int(st[15]) == CATEGORY_PATH)
    video_path_carrot_drawn_frames = 0
    video_path_trace_drawn_frames = 0
    perspective = bool(env.cfg.get("evaluation", {}).get(
        "perspective_overlays", False))
    camera_poses = getattr(env, "camera_poses", [])
    fov = float(getattr(env, "camera_horizontal_fov", 75.0))
    if perspective:
        paired = min(video_recorded_frames, len(camera_poses))
        path_trace_world = []
        last_path_segment = None
        for i in range(paired):
            frame = env.camera_frames[i]
            h, w = frame.shape[:2]
            if _force_arrow_points(
                    overlay_states[i], camera_poses[i], fov, w, h) is not None:
                video_force_arrow_drawn_frames += 1
            st = overlay_states[i]
            if len(st) >= 16 and int(st[15]) == CATEGORY_PATH:
                segment = int(st[16]) if len(st) >= 17 else None
                if (last_path_segment is not None and segment is not None
                        and segment != last_path_segment):
                    path_trace_world = []
                path_trace_world.append(np.asarray(st[2], dtype=float).copy())
                point = _project_world(
                    (float(st[2][0]), float(st[2][1]), 0.06),
                    camera_poses[i], fov, w, h)
                if point is not None and 0 <= point[0] < w and 0 <= point[1] < h:
                    video_path_carrot_drawn_frames += 1
                trace_tail = path_trace_world[-120:]
                projected_trace = [
                    _project_world((float(xy[0]), float(xy[1]), 0.04),
                                   camera_poses[i], fov, w, h)
                    for xy in trace_tail
                ]
                if any(
                        a is not None and b is not None
                        and np.linalg.norm(xyb - xya) > 1.0e-6
                        and _segment_intersects_image(a, b, w, h)
                        for a, b, xya, xyb in zip(
                            projected_trace[:-1], projected_trace[1:],
                            trace_tail[:-1], trace_tail[1:])):
                    video_path_trace_drawn_frames += 1
                last_path_segment = segment
            else:
                path_trace_world = []
                last_path_segment = None

    fr = {k: np.asarray(v) for k, v in force_records.items()}

    def event_summary(kind=None, high_speed_only=False, scenario_id=None,
                      height_tier_id=None):
        count = len(fr["kind"])
        mask = np.ones(count, dtype=bool)
        if kind is not None:
            mask &= fr["kind"].astype(int) == kind
        if scenario_id is not None:
            mask &= fr["scenario_id"].astype(int) == scenario_id
        if height_tier_id is not None:
            mask &= fr["height_tier_id"].astype(int) == height_tier_id
        if high_speed_only:
            high_speed_threshold = float(env.cfg.get("evaluation", {}).get(
                "high_speed_threshold_mps", 0.8))
            mask &= (fr["baseline_goal_progress_speed_mps"].astype(float)
                     >= high_speed_threshold)
        eligible = mask & fr["recovery_eligible"].astype(bool)
        recovered = eligible & np.isfinite(fr["recovery_90_s"].astype(float))
        outcome_censored = fr[
            "outcome_censored_by_episode_timeout"].astype(bool)
        rollout_censored = fr["outcome_censored_by_rollout_end"].astype(bool)
        outcome_censored |= rollout_censored
        duration = fr["duration_observed_s"].astype(float)
        survival_2_eligible = mask & (~outcome_censored | (duration >= 2.0))
        survival_5_eligible = mask & (~outcome_censored | (duration >= 5.0))

        def masked_pct(key, percentile):
            values = fr[key].astype(float)[mask]
            return _pct(values, percentile) if len(values) else float("nan")

        return {
            "records": int(mask.sum()),
            "baseline_goal_progress_speed_mps_p50": masked_pct(
                "baseline_goal_progress_speed_mps", 50),
            "outcomes_censored_by_episode_timeout": int(
                (mask & fr["outcome_censored_by_episode_timeout"].astype(bool)).sum()),
            "outcomes_censored_by_rollout_end": int(
                (mask & rollout_censored).sum()),
            "survival_2s_eligible": int(survival_2_eligible.sum()),
            "survival_5s_eligible": int(survival_5_eligible.sum()),
            "survival_2s": (
                float(fr["survived_2s"].astype(bool)[survival_2_eligible].mean())
                if survival_2_eligible.any() else float("nan")),
            "survival_5s": (
                float(fr["survived_5s"].astype(bool)[survival_5_eligible].mean())
                if survival_5_eligible.any() else float("nan")),
            "impulse_ns_median": masked_pct("impulse_ns", 50),
            "impulse_ns_p90": masked_pct("impulse_ns", 90),
            "torque_impulse_nms_p90": masked_pct("torque_impulse_nms", 90),
            "max_tilt_deg_p90": masked_pct("max_tilt_deg", 90),
            "speed_loss_mps_p90": masked_pct("speed_loss_mps", 90),
            "recovery_eligible": int(eligible.sum()),
            "recovery_censored_by_goal_protocol": int(
                (mask & fr["recovery_censored_by_goal_protocol"].astype(bool)).sum()),
            "recovery_90_within_5s_share": (
                float(recovered.sum() / eligible.sum()) if eligible.any() else float("nan")),
            "recovery_90_s_p50": (_pct(fr["recovery_90_s"].astype(float)[recovered], 50)
                                  if recovered.any() else float("nan")),
            "recovery_90_s_p90": (_pct(fr["recovery_90_s"].astype(float)[recovered], 90)
                                  if recovered.any() else float("nan")),
        }

    scenario_names = list(getattr(env, "dist_scenario_names", ()))
    height_tier_names = list(getattr(env, "dist_height_tier_names", ()))
    scenario_breakdown = {
        name: event_summary(scenario_id=i + 1)
        for i, name in enumerate(scenario_names)
    }
    height_tier_breakdown = {
        name: event_summary(height_tier_id=i)
        for i, name in enumerate(height_tier_names)
    }

    # Eight robot-local horizontal octants make it obvious if a supposedly
    # omnidirectional treatment silently degenerates into front/back only.
    octant_labels = (
        "+x", "+x+y", "+y", "-x+y", "-x", "-x-y", "-y", "+x-y")
    direction_octants = {name: 0 for name in octant_labels}
    if len(fr["direction_local_deg"]):
        angles = fr["direction_local_deg"].astype(float)
        valid_direction = np.isfinite(angles)
        safe_angles = np.where(valid_direction, angles, 0.0)
        octants = np.floor(((safe_angles + 22.5) % 360.0) / 45.0).astype(int)
        for i, name in enumerate(octant_labels):
            direction_octants[name] = int((valid_direction & (octants == i)).sum())

    def delivery_error(expected_key, submitted_key):
        expected = fr[expected_key].astype(float)
        submitted = fr[submitted_key].astype(float)
        valid = np.isfinite(expected) & np.isfinite(submitted) & (expected > 0.0)
        relative = np.abs(submitted[valid] - expected[valid]) / expected[valid]
        return {
            "records": int(valid.sum()),
            "relative_error_median": (_pct(relative, 50) if valid.any()
                                      else float("nan")),
            "relative_error_p90": (_pct(relative, 90) if valid.any()
                                   else float("nan")),
            "relative_error_max": (float(relative.max()) if valid.any()
                                   else float("nan")),
        }

    out["disturbance_eval"] = {
        "recovery_definition": (
            "path-only: filtered world velocity projected toward the current "
            "carrot regains 90% of the post-schedule/pre-application goal-progress "
            "speed, with tilt <=20 deg; unresolved records are recovery-censored "
            "when segment/dwell/floor-recovery protocol changes"),
        "events": force_events,
        "scheduled_events_cancelled_before_application": (
            force_cancelled_before_application),
        "active_env_steps": force_active_steps,
        "active_share": force_active_steps / float(max(total_steps * env.num_envs, 1)),
        "falls_during_force": force_context_falls,
        "events_with_outcome_record": len(fr["kind"]),
        "events_censored_or_overlapping": max(0, force_events - len(fr["kind"])),
        "video_requested": bool(record_video),
        "video_recorded_frames": int(video_recorded_frames),
        "video_force_active_frames": int(video_force_active_frames),
        "video_force_active_share": (
            video_force_active_frames / float(video_recorded_frames)
            if video_recorded_frames else 0.0),
        "video_force_arrow_drawn_frames": int(video_force_arrow_drawn_frames),
        "video_force_arrow_drawn_share": (
            video_force_arrow_drawn_frames / float(video_recorded_frames)
            if video_recorded_frames else 0.0),
        "video_path_frames": int(video_path_frames),
        "video_path_carrot_drawn_frames": int(video_path_carrot_drawn_frames),
        "video_path_trace_drawn_frames": int(video_path_trace_drawn_frames),
        "overall": event_summary(),
        "high_speed": event_summary(high_speed_only=True),
        "collision": event_summary(1),
        "support": event_summary(2),
        "scenario_names_by_id": {
            str(i + 1): name for i, name in enumerate(scenario_names)},
        "height_tier_names_by_id": {
            str(i): name for i, name in enumerate(height_tier_names)},
        "body_names_by_index": {
            str(i): name for i, name in enumerate(getattr(env, "body_names", ()))},
        "scenario_breakdown": scenario_breakdown,
        "height_tier_breakdown": height_tier_breakdown,
        "direction_octants_robot_local": direction_octants,
        "delivery_audit": {
            "force": delivery_error(
                "expected_impulse_ns", "submitted_impulse_ns"),
            "torque": delivery_error(
                "expected_torque_impulse_nms",
                "submitted_torque_impulse_nms"),
        },
    }
    return out


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

def summarize(roll, cfg, num_envs, duration_s, dt, checkpoint, config_path, task,
              deterministic=True, perturbations=False, obs_noise=True,
              exploratory=False, setup_wall_s=0.0, seed=0,
              task_state_protocol="config_fixed"):
    """Turn a rollout into gate verdicts plus the breakdowns needed to act on them."""
    eval_cfg = cfg.get("evaluation", {})
    gates = eval_cfg.get("gates", {})
    g_pos_med = gates.get("pos_median_m", 0.05)
    g_pos_p90 = gates.get("pos_p90_m", 0.10)
    g_head_med = gates.get("heading_median_deg", 10.0)
    g_falls = gates.get("max_falls", 0)
    feasible_speed = eval_cfg.get("feasible_speed_mps", 0.6)
    stop_thr = cfg["rewards"].get("stop_speed_threshold", 0.1)

    pos_all = roll["pos_err"]
    head_all = np.degrees(roll["head_err"])
    speed_all = roll["speed"]
    cat_all = roll["category"]
    path_mask = cat_all == CATEGORY_PATH
    waypoint_mask = ~path_mask

    # The position gates are ONLY meaningful on waypoint segments.
    #
    # In path mode the goal deliberately sits lookahead_min ahead of the robot
    # and keeps moving, so "distance to the goal when the segment ended" is a
    # readout of the lookahead distance, not of tracking error -- it can never
    # approach zero by construction, and "arrived then left" is 87-100% by
    # definition because the carrot passes through the 5 cm radius and moves on.
    # In the 2026-07-27 v7 batch these segments were 44-49% of the sample and
    # dragged every headline number (E0: 6.5 cm on waypoints, 75 cm on path,
    # 13.2 cm reported). Path tracking has its own metric -- path_lag and the
    # commanded-vs-achieved table -- so it is scored there instead.
    gate_mask = waypoint_mask.copy()
    gate_scope = "waypoint_only"
    if not gate_mask.any():
        gate_mask = np.ones_like(cat_all, dtype=bool)
        gate_scope = "all_segments_no_waypoints"

    # Full arrays stay full: every per-category / per-distance / failure-mode
    # breakdown below indexes them by masks built from the same length, and the
    # path rows in those tables are still worth reading. Only the GATES are
    # restricted.
    pos, head_deg, speed = pos_all, head_all, speed_all
    n = len(pos)
    if n == 0:
        raise RuntimeError("no completed goal segments — duration too short or every env fell")
    falls = int(roll["falls"])
    attempts = n + falls

    gate_pos, gate_head = pos_all[gate_mask], head_all[gate_mask]
    n_gate, n_path = int(gate_mask.sum()), int((~gate_mask).sum())
    pos_med, pos_p90 = _median(gate_pos), _pct(gate_pos, 90)
    head_med = _median(gate_head)

    ok_pos_loose = pos <= g_pos_p90
    ok_head = head_deg <= g_head_med
    ok_stop = speed <= stop_thr

    results = {
        "checkpoint": checkpoint,
        "config": config_path,
        "task": task,
        "experiment_description": cfg.get("basic", {}).get("description", ""),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "num_envs": num_envs,
        "duration_s": duration_s,
        "deterministic": deterministic,
        "perturbations": perturbations,
        # Which force test this was scored under. Without it a force-ON report is
        # indistinguishable from a clean one six months later, and comparing two
        # arms scored under different profiles is silently wrong.
        "force_profile": os.environ.get("EVAL_FORCE_PROFILE") or None,
        "eval_terrain": os.environ.get("EVAL_TERRAIN") or None,
        "obs_noise": obs_noise,
        "authoritative_gate_evaluation": not exploratory,
        "task_state_protocol": task_state_protocol,
        "force_visualization_probe": bool(
            eval_cfg.get("force_visualization_probe", False)),
        "hbatch_protocol_version": eval_cfg.get("hbatch_protocol_version"),
        "effective_eval_protocol_sha": eval_cfg.get(
            "effective_eval_protocol_sha"),
        "effective_disturbance_protocol": copy.deepcopy(
            eval_cfg.get("effective_disturbance_protocol") or {}),
        "joint_dr_probe": copy.deepcopy(
            eval_cfg.get("joint_dr_probe") or {}),
        "hbatch_gates": dict(eval_cfg.get("hbatch_gates", {}) or {}),
        "env_code_sha": env_code_sha(),
        "evaluation_protocol_sha": evaluation_protocol_sha(),
        # Symmetry / joint-margin telemetry. These were being written into
        # env.extras["v7"] and read by nothing, so every "we added the metric"
        # claim was hollow -- the numbers never reached a report.
        "v7_extras": roll.get("v7_extras") or {},
        "segments_completed": n,
        "segments_waypoint": int(waypoint_mask.sum()),
        "segments_path": int(path_mask.sum()),
        "segments_scored_by_gates": n_gate,
        "gate_scope": gate_scope,
        "segments_path_excluded_from_gates": n_path,
        "segments_censored_by_episode_end": roll["censored"],
        "falls": falls,
        "fall_rate_per_attempt": falls / attempts if attempts else 0.0,
        "pos_err_m": {"median": pos_med, "p90": pos_p90,
                      "mean": float(np.mean(gate_pos)), "max": float(np.max(gate_pos))},
        "heading_err_deg": {"median": head_med, "p90": _pct(gate_head, 90),
                            "mean": float(np.mean(gate_head)), "max": float(np.max(gate_head))},
        "final_speed_mps": {"median": _median(speed), "p90": _pct(speed, 90), "mean": float(np.mean(speed))},
        "success_rate_strict": _frac((pos[gate_mask] <= g_pos_med) & ok_head[gate_mask] & ok_stop[gate_mask]),
        "success_rate_loose": _frac(ok_pos_loose[gate_mask] & ok_head[gate_mask]),
        "ci95": {
            "pos_median": bootstrap_ci(gate_pos, 50.0, seed=seed),
            "pos_p90": bootstrap_ci(gate_pos, 90.0, seed=seed),
            "heading_median": bootstrap_ci(gate_head, 50.0, seed=seed),
        },
        "timing": {
            "setup_wall_s": setup_wall_s,
            "rollout_wall_s": roll["rollout_wall_s"],
            "simulated_time_per_env_s": roll["total_steps"] * dt,
            "aggregate_env_simulated_time_s": roll["total_steps"] * dt * num_envs,
            "single_env_realtime_factor": (roll["total_steps"] * dt) / max(roll["rollout_wall_s"], 1.0e-9),
            "aggregate_env_seconds_per_wall_second":
                (roll["total_steps"] * dt * num_envs) / max(roll["rollout_wall_s"], 1.0e-9),
            "vectorized_control_iterations_per_wall_second":
                roll["total_steps"] / max(roll["rollout_wall_s"], 1.0e-9),
            "aggregate_env_transitions_per_wall_second":
                (roll["total_steps"] * num_envs) / max(roll["rollout_wall_s"], 1.0e-9),
        },
        "gates": {
            "pos_median": {"limit": g_pos_med, "value": pos_med, "pass": bool(pos_med <= g_pos_med)},
            "pos_p90": {"limit": g_pos_p90, "value": pos_p90, "pass": bool(pos_p90 <= g_pos_p90)},
            "heading_median": {"limit": g_head_med, "value": head_med, "pass": bool(head_med <= g_head_med)},
            "falls": {"limit": g_falls, "value": falls, "pass": bool(falls <= g_falls)},
        },
        "stop_speed_threshold": stop_thr,
        "instrumented": bool(roll["instrumented"]),
    }
    results["all_gates_pass"] = all(g["pass"] for g in results["gates"].values())

    # ---- failure-mode decomposition -------------------------------------
    # Mutually exclusive, in priority order. "arrived_then_left" vs "never_arrived"
    # is the important split: identical final error, opposite fixes.
    min_d = roll["min_dist"]
    ever_arrived = min_d <= g_pos_med
    mode = np.full(n, "excluded_path", dtype=object)
    mode[gate_mask] = "never_arrived"
    mode[gate_mask & ~ok_pos_loose & ever_arrived] = "arrived_then_left"
    mode[gate_mask & ok_pos_loose & ~ok_head] = "heading_only"
    mode[gate_mask & ok_pos_loose & ok_head & ~ok_stop] = "not_stopped"
    mode[gate_mask & ok_pos_loose & ok_head & ok_stop] = "ok"
    results["failure_modes"] = {
        name: {
            "count": int(np.sum(mode[gate_mask] == name)),
            "share": float(np.mean(mode[gate_mask] == name)) if gate_mask.any() else float("nan"),
        }
        for name in ("ok", "not_stopped", "heading_only", "arrived_then_left", "never_arrived")
    }
    results["failure_modes_scope"] = gate_scope
    results["closest_approach_m"] = {"median": _median(min_d[gate_mask]), "p90": _pct(min_d[gate_mask], 90)}

    # ---- along/cross-track split ----------------------------------------
    along, cross = roll["along"], roll["cross"]
    finite = np.isfinite(along) & gate_mask
    if finite.any():
        a = along[finite]
        results["approach_error_m"] = {
            "along_median": _median(a),
            "along_p10": _pct(a, 10),
            "along_p90": _pct(a, 90),
            "cross_median_abs": _median(np.abs(cross[finite])),
            "overshoot_share": _frac(a > 0),
            "n": int(finite.sum()),
        }
    else:
        results["approach_error_m"] = None

    # ---- time feasibility -----------------------------------------------
    # A 2.5 m goal with a 4 s deadline needs >0.6 m/s sustained AND a stop. If a
    # large share of segments are like that, the gate numbers are measuring the
    # task definition, not the policy.
    dur = roll["duration_s"]
    start_d = roll["start_dist"]
    feas_ok = np.isfinite(dur) & np.isfinite(start_d) & (dur > 0)
    feas_ok = feas_ok & gate_mask
    if feas_ok.any():
        required = np.full(n, np.nan)
        required[feas_ok] = start_d[feas_ok] / dur[feas_ok]
        feasible = feas_ok & (required <= feasible_speed)
        infeasible = feas_ok & (required > feasible_speed)
        # What this policy actually sustained, so feasible_speed_mps can be sanity
        # checked against measurement instead of being trusted as a constant.
        achieved = (start_d[feas_ok] - pos[feas_ok]) / dur[feas_ok]
        results["feasibility"] = {
            "feasible_speed_mps": feasible_speed,
            "achieved_closing_speed_p95": _pct(achieved, 95),
            "achieved_closing_speed_median": _median(achieved),
            "required_speed_median": _median(required[feas_ok]),
            "required_speed_p90": _pct(required[feas_ok], 90),
            "infeasible_share": float(np.sum(infeasible) / np.sum(feas_ok)),
            "infeasible_count": int(np.sum(infeasible)),
            "feasible_subset": {
                "n": int(np.sum(feasible)),
                "pos_median": _median(pos[feasible]),
                "pos_p90": _pct(pos[feasible], 90),
                "heading_median": _median(head_deg[feasible]),
                "success_rate_strict": _frac((pos[feasible] <= g_pos_med) & ok_head[feasible] & ok_stop[feasible]),
            },
            "infeasible_subset": {
                "n": int(np.sum(infeasible)),
                "pos_median": _median(pos[infeasible]),
                "pos_p90": _pct(pos[infeasible], 90),
            },
            "segment_duration_s": {"median": _median(dur[feas_ok]), "min": float(np.min(dur[feas_ok]))},
        }
    else:
        results["feasibility"] = None

    # ---- body speed -------------------------------------------------------
    # The MASTERPLAN speed target is stated as a body velocity, but every metric
    # above is an error in metres. Without this block a policy that is merely
    # never ASKED to walk fast is indistinguishable from one that CANNOT.
    hist = roll.get("speed_hist")
    if hist is not None and hist.sum() > 0:
        edges = np.arange(len(hist)) * (roll["speed_hist_max"] / len(hist))
        cdf = np.cumsum(hist) / hist.sum()

        def _hpct(p):
            return float(edges[int(np.searchsorted(cdf, p / 100.0))]) if p / 100.0 <= cdf[-1] else float(edges[-1])

        occupied = np.nonzero(hist)[0]
        peaks = roll.get("peak_speed", np.array([]))
        results["body_speed"] = {
            "median": _hpct(50), "p90": _hpct(90), "p99": _hpct(99),
            "max_instant": float(edges[occupied[-1]]) if len(occupied) else float("nan"),
            "share_above_0p5": float(hist[edges >= 0.5].sum() / hist.sum()),
            "share_above_1p0": float(hist[edges >= 1.0].sum() / hist.sum()),
            "segment_peak_median": _median(peaks) if len(peaks) else float("nan"),
            "segment_peak_p90": _pct(peaks, 90) if len(peaks) else float("nan"),
            "segment_peak_max": float(np.max(peaks)) if len(peaks) else float("nan"),
        }
        # P2 read as a SUSTAINED speed.  Neither of the two numbers above can
        # answer it.  A peak is an acceleration transient on a short hop, and a
        # run-wide "share above 1 m/s" is dominated by the goal distribution
        # rather than the robot: goal_categories puts 30% of draws at zero
        # distance (stand + turn) and collapses straight/lateral onto one axis,
        # so 44% of segments are under 0.5 m and only ~3% are over 2 m.  Pooling
        # those measures the task.  Condition on segments long enough to permit
        # a cruise at all, and report the longest CONTINUOUS stretch.
        sd = np.asarray(roll.get("start_dist", []), dtype=float)
        cru = np.asarray(roll.get("cruise_1p3", []), dtype=float)
        t13 = np.asarray(roll.get("time_above_1p3", []), dtype=float)
        long_m = ((sd >= LONG_SEGMENT_M) & np.isfinite(cru)
                  if len(sd) and len(sd) == len(cru) else np.zeros(len(sd), bool))
        n_long = int(long_m.sum())
        results["body_speed"]["sustained_1p3"] = {
            "long_segment_min_dist_m": LONG_SEGMENT_M,
            "n_long_segments": n_long,
            "cruise_median_s": _median(cru[long_m]) if n_long else float("nan"),
            "cruise_p90_s": _pct(cru[long_m], 90) if n_long else float("nan"),
            "cruise_max_s": float(np.max(cru[long_m])) if n_long else float("nan"),
            "share_cruise_ge_0p5s": float((cru[long_m] >= 0.5).mean()) if n_long else float("nan"),
            "share_cruise_ge_1p0s": float((cru[long_m] >= 1.0).mean()) if n_long else float("nan"),
            "time_above_1p3_median_s": _median(t13[long_m]) if n_long else float("nan"),
        }
    else:
        results["body_speed"] = None

    # ---- swing apex (foot clearance) -------------------------------------
    # The number terrain arms exist to move.  p10 matters more than the median:
    # tripping is a worst-case event, so what the LOW swings clear is the risk,
    # not what the average one does.
    sa_hist = roll.get("swing_apex_hist")
    sa_max = float(roll.get("swing_apex_max_m", 0.20))
    if sa_hist is not None and int(np.sum(sa_hist)) > 0:
        sa_edges = np.arange(len(sa_hist)) * (sa_max / len(sa_hist))
        sa_cdf = np.cumsum(sa_hist) / float(np.sum(sa_hist))

        def _sa_pct(p):
            i = int(np.searchsorted(sa_cdf, p / 100.0))
            return float(sa_edges[min(i, len(sa_edges) - 1)])

        results["swing_apex_m"] = {
            "n_swings": int(np.sum(sa_hist)),
            "p10": _sa_pct(10), "median": _sa_pct(50), "p90": _sa_pct(90),
            "share_below_0p02": float(sa_hist[sa_edges < 0.02].sum() / np.sum(sa_hist)),
            "share_below_0p03": float(sa_hist[sa_edges < 0.03].sum() / np.sum(sa_hist)),
            "contact_threshold_m": 0.01,
        }
    else:
        results["swing_apex_m"] = None

    # ---- support / load share --------------------------------------------
    sup = roll.get("support_steps")
    if sup is not None and int(np.sum(sup)) > 0:
        tot = float(np.sum(sup))
        ssh = roll.get("ss_hist")
        ss_max = float(roll.get("ss_hist_max_s", 3.0))
        ss_edges = np.arange(len(ssh)) * (ss_max / len(ssh))
        ss_cdf = np.cumsum(ssh) / max(float(np.sum(ssh)), 1.0)
        la = roll.get("load_asym_hist")
        la_edges = np.arange(len(la)) / float(len(la))
        la_cdf = np.cumsum(la) / max(float(np.sum(la)), 1.0)

        def _p(edges, cdf, p):
            i = int(np.searchsorted(cdf, p / 100.0))
            return float(edges[min(i, len(edges) - 1)])

        # NOTE: disturbance_eval["support"] is a DISTURBANCE CLASS (support-force
        # events), nothing to do with feet.  Different name on purpose.
        results["foot_support"] = {
            "flight_share": float(sup[0] / tot),
            "single_support_share": float(sup[1] / tot),
            "double_support_share": float(sup[2] / tot),
            "single_support_s": {
                "n": int(np.sum(ssh)),
                "median": _p(ss_edges, ss_cdf, 50),
                "p90": _p(ss_edges, ss_cdf, 90),
                "p99": _p(ss_edges, ss_cdf, 99),
            } if float(np.sum(ssh)) > 0 else None,
            # |L-R| / (L+R) during double support.  0 = even, 1 = one foot only.
            "load_asymmetry": {
                "n": int(np.sum(la)),
                "median": _p(la_edges, la_cdf, 50),
                "p90": _p(la_edges, la_cdf, 90),
            } if float(np.sum(la)) > 0 else None,
        }
    else:
        results["foot_support"] = None

    # ---- path floor/leash telemetry at control-step resolution ------------
    # This is intentionally separate from segment-end path_tracking.  A carrot
    # can violate the floor for a short interval and recover before resampling;
    # only a streaming step metric can expose that failure mode.
    path_hist = roll.get("path_step_hist") or {}
    path_counts = roll.get("path_step_counts") or {}
    path_hist_max = roll.get("path_step_hist_max") or {}
    path_samples = int(path_counts.get("samples", 0))
    if path_samples and path_hist:
        ratio_max = float(path_hist_max.get("gap_over_lookahead", 4.0))
        distance_max = float(path_hist_max.get("distance_m", 7.0))
        steady_samples = int(path_counts.get("steady_samples", 0))
        results["path_step_tracking"] = {
            "scope": "post-step is_path_env, excluding dwell and reset steps",
            "definition": (
                "signed error = gap - per-env lookahead; floor deficit is its "
                "negative magnitude, behind lag its positive magnitude; leash "
                "uses the per-env runtime bound; dwell-resume recovery is "
                "reported both included and excluded"),
            "samples": path_samples,
            "steady_samples_excluding_recovery": steady_samples,
            "gap_over_lookahead_p2": _hist_pct(
                path_hist["gap_over_lookahead"], ratio_max, 2),
            "gap_over_lookahead_p50": _hist_pct(
                path_hist["gap_over_lookahead"], ratio_max, 50),
            "below_0p75_share": (
                path_counts.get("below_0p75", 0) / float(path_samples)),
            "dwell_resume_recovery_share": (
                path_counts.get("dwell_resume_recovery", 0) / float(path_samples)),
            "below_0p75_share_excluding_recovery": (
                path_counts.get("steady_below_0p75", 0)
                / float(steady_samples) if steady_samples > 0 else float("nan")),
            "gap_m_p2": _hist_pct(path_hist["gap_m"], distance_max, 2),
            "gap_m_p50": _hist_pct(path_hist["gap_m"], distance_max, 50),
            "lookahead_m_p2": _hist_pct(
                path_hist["lookahead_m"], distance_max, 2),
            "lookahead_m_p50": _hist_pct(
                path_hist["lookahead_m"], distance_max, 50),
            "leash_m_p2": _hist_pct(path_hist["leash_m"], distance_max, 2),
            "leash_m_p50": _hist_pct(path_hist["leash_m"], distance_max, 50),
            "floor_deficit_m_p50": _hist_pct(
                path_hist["floor_deficit_m"], distance_max, 50),
            "floor_deficit_m_p90": _hist_pct(
                path_hist["floor_deficit_m"], distance_max, 90),
            "behind_lag_m_p50": _hist_pct(
                path_hist["behind_lag_m"], distance_max, 50),
            "behind_lag_m_p90": _hist_pct(
                path_hist["behind_lag_m"], distance_max, 90),
            "outside_leash_share": (
                path_counts.get("outside_leash", 0) / float(path_samples)),
        }
    else:
        results["path_step_tracking"] = None

    # ---- phase-conditioned high-speed stability --------------------------
    sh = roll.get("stability_hist") or {}
    sc = roll.get("stability_counts") or {}
    if sh:
        results["high_speed_stability"] = {
            "phase_definition": {
                "acceleration": "v>0.3 m/s and forward acceleration>threshold",
                "cruise": "v>=high_speed_threshold and |a_xy|<=steady threshold",
                "high_speed_threshold_mps": float(eval_cfg.get("high_speed_threshold_mps", 0.8)),
                "steady_accel_threshold_mps2": float(eval_cfg.get("steady_accel_threshold_mps2", 0.3)),
            },
            "samples": {k: int(v) for k, v in sc.items()},
            "accel_pitch_abs_p50_deg": _hist_pct(sh["accel_pitch_deg"], 90.0, 50),
            "accel_pitch_abs_p90_deg": _hist_pct(sh["accel_pitch_deg"], 90.0, 90),
            "cruise_pitch_abs_p90_deg": _hist_pct(sh["cruise_pitch_deg"], 90.0, 90),
            "cruise_roll_abs_p90_deg": _hist_pct(sh["cruise_roll_deg"], 90.0, 90),
            "cruise_ang_xy_p90_radps": _hist_pct(sh["cruise_ang_xy"], 8.0, 90),
            "cruise_abs_z_vel_p90_mps": _hist_pct(sh["cruise_z_vel"], 3.0, 90),
            "cruise_share_of_valid": sc.get("cruise", 0) / float(max(sc.get("valid", 0), 1)),
        }
    else:
        results["high_speed_stability"] = None

    th = roll.get("touchdown_hist") or {}
    tc = roll.get("touchdown_counts") or {}
    tb = roll.get("touchdown_bounds") or {}
    touchdown_samples = int(tc.get("samples", 0))
    if th and touchdown_samples > 0:
        heel_lo, heel_hi = tb.get("heel_x_body", [-0.40, 0.40])
        speed_lo, speed_hi = tb.get("precontact_down_speed", [0.0, 4.0])
        force_lo, force_hi = tb.get("contact_force", [0.0, 1000.0])
        left_med = _hist_pct_range(
            th["heel_x_left"], heel_lo, heel_hi, 50)
        right_med = _hist_pct_range(
            th["heel_x_right"], heel_lo, heel_hi, 50)
        results["gait_touchdown"] = {
            "scope": (
                "first foot-contact transitions during forward path motion "
                "(body vx above heel_strike.min_forward_speed_mps), excluding resets"),
            "samples": touchdown_samples,
            "heel_x_body_m": {
                "p10": _hist_pct_range(
                    th["heel_x_body"], heel_lo, heel_hi, 10),
                "median": _hist_pct_range(
                    th["heel_x_body"], heel_lo, heel_hi, 50),
                "p90": _hist_pct_range(
                    th["heel_x_body"], heel_lo, heel_hi, 90),
            },
            "ahead_of_trunk_share": tc.get("ahead", 0) / float(touchdown_samples),
            "within_dynamic_target_one_sigma_share": (
                tc.get("within_target_sigma", 0) / float(touchdown_samples)),
            "overstride_share": tc.get("overstride", 0) / float(touchdown_samples),
            "left_samples": int(th["heel_x_left"].sum()),
            "right_samples": int(th["heel_x_right"].sum()),
            "left_heel_x_median_m": left_med,
            "right_heel_x_median_m": right_med,
            "left_right_heel_x_median_abs_diff_m": (
                abs(left_med - right_med)
                if np.isfinite(left_med) and np.isfinite(right_med)
                else float("nan")),
            "precontact_down_speed_p90_mps": _hist_pct_range(
                th["precontact_down_speed"], speed_lo, speed_hi, 90),
            "contact_force_p90_n": _hist_pct_range(
                th["contact_force"], force_lo, force_hi, 90),
        }
    else:
        results["gait_touchdown"] = None
    results["disturbance_eval"] = roll.get("disturbance_eval") or {}
    mirror_hist = roll.get("mirror_error_hist")
    if mirror_hist is not None and mirror_hist.sum() > 0:
        mirror_max = float(roll.get("mirror_error_hist_max", 2.0))
        results["symmetry_eval"] = {
            "definition": "RMS(pi(Ms) - M pi(s)) over actions; sampled every 10 control steps",
            "samples": int(mirror_hist.sum()),
            "median": _hist_pct(mirror_hist, mirror_max, 50),
            "p90": _hist_pct(mirror_hist, mirror_max, 90),
            "p99": _hist_pct(mirror_hist, mirror_max, 99),
        }
    else:
        results["symmetry_eval"] = None

    # ---- segment transient response -------------------------------------
    def response_stats(values, mask):
        values = np.asarray(values, dtype=float)
        reached = mask & np.isfinite(values)
        return {
            "eligible": int(mask.sum()),
            "reached_share": float(reached.sum() / mask.sum()) if mask.any() else float("nan"),
            "p50_s": _pct(values[reached], 50) if reached.any() else float("nan"),
            "p90_s": _pct(values[reached], 90) if reached.any() else float("nan"),
        }

    if path_mask.any():
        initial_speed = np.asarray(roll["initial_speed_mps"], dtype=float)
        commanded_speed = np.asarray(roll["cmd_speed"], dtype=float)
        from_rest = path_mask & np.isfinite(initial_speed) & (initial_speed <= 0.30)
        results["path_acceleration_response"] = {
            "definition": (
                "from-rest segments only (initial speed <=0.30 m/s), and only "
                "when commanded path speed is at least the reported threshold"),
            "time_to_0p5": response_stats(
                roll["time_to_0p5_s"], from_rest & (commanded_speed >= 0.5)),
            "time_to_0p8": response_stats(
                roll["time_to_0p8_s"], from_rest & (commanded_speed >= 0.8)),
            "time_to_1p0": response_stats(
                roll["time_to_1p0_s"], from_rest & (commanded_speed >= 1.0)),
        }
    else:
        results["path_acceleration_response"] = None

    goal_pattern = eval_cfg.get("goal_pattern")
    if goal_pattern:
        response_mask = np.ones(n, dtype=bool)
        bearing = np.degrees(np.abs(roll["initial_goal_bearing_rad"]))
        min2 = np.asarray(roll["min_speed_first_2s"], dtype=float)
        phase = np.mod(np.asarray(roll["switch_gait_phase"], dtype=float), 1.0)
        phase_bin = np.minimum((phase * 4.0).astype(int), 3)
        phase_rows = []
        for b in range(4):
            pm = response_mask & (phase_bin == b)
            if not pm.any():
                continue
            phase_rows.append({
                "phase_lo": b / 4.0,
                "phase_hi": (b + 1) / 4.0,
                "n": int(pm.sum()),
                "time_to_0p5": response_stats(
                    roll["direction_time_to_0p5_s"], pm),
                "time_to_1p0": response_stats(
                    roll["direction_time_to_1p0_s"], pm),
            })
        results["directional_response"] = {
            "pattern": goal_pattern,
            "segments": n,
            "initial_goal_bearing_abs_deg": {
                "median": _median(bearing), "p10": _pct(bearing, 10), "p90": _pct(bearing, 90)},
            "min_speed_first_2s_mps": {
                "median": _median(min2[np.isfinite(min2)]),
                "p10": _pct(min2[np.isfinite(min2)], 10)},
            "speed_definition": (
                "filtered world velocity projected onto the initial new-goal "
                "direction; continuing quickly in the old direction does not count"),
            "time_to_0p5": response_stats(
                roll["direction_time_to_0p5_s"], response_mask),
            "time_to_0p8": response_stats(
                roll["direction_time_to_0p8_s"], response_mask),
            "time_to_1p0": response_stats(
                roll["direction_time_to_1p0_s"], response_mask),
            "by_switch_gait_quarter": phase_rows,
        }
    else:
        results["directional_response"] = None

    # ---- commanded vs achieved speed (path mode) --------------------------
    # The point of a WIDE FIXED commanded-speed distribution (no curriculum) is
    # that this curve saturates: below the robot's physical limit achieved
    # tracks commanded, above it the curve flattens. The knee is the limit --
    # measured, not scheduled. A curriculum can never show this, because it
    # stops raising the demand once the policy stops keeping up and so cannot
    # distinguish "not yet" from "never".
    cmd = roll.get("cmd_speed")
    results["speed_tracking"] = None
    if cmd is not None and np.isfinite(cmd).any():
        # path_speed retains its last sampled value on waypoint envs.  Without
        # this category gate, the old report mixed every waypoint into the
        # commanded-speed curve and reported a fictitious saturation knee.
        ok = path_mask & np.isfinite(cmd) & (cmd > 0) & np.isfinite(roll["mean_speed"])
        if ok.sum() >= 20:
            bins = [0.0, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, float("inf")]
            rows = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = ok & (cmd >= lo) & (cmd < hi)
                if m.sum() < 5:
                    continue
                rows.append({
                    "cmd_lo": lo, "cmd_hi": hi, "n": int(m.sum()),
                    "cmd_median": _median(cmd[m]),
                    "achieved_mean_median": _median(roll["mean_speed"][m]),
                    "achieved_peak_median": _median(roll["peak_speed"][m]),
                    "tracking_ratio": float(_median(roll["mean_speed"][m]) / max(_median(cmd[m]), 1e-9)),
                })
            results["speed_tracking"] = {"bins": rows, "n": int(ok.sum())}

    # ---- path tracking ------------------------------------------------------
    # Path mode is not a waypoint task: the carrot is supposed to stay ahead of
    # the robot. Raw distance-to-goal is therefore a lookahead/gap readout, not a
    # "did we arrive?" error. Score it with lag behind the lookahead floor and
    # commanded-vs-achieved speed instead.
    lag = roll.get("path_lag")
    results["path_tracking"] = None
    if lag is not None and path_mask.any():
        p_cfg = cfg.get("commands", {}).get("path", {}) or {}
        keep = float(p_cfg.get("keepup_gap_m", 2.0))
        p_lag = lag[path_mask]
        finite_lag = np.isfinite(p_lag)
        if finite_lag.any():
            p_lag = p_lag[finite_lag]
            p_mean_speed = roll["mean_speed"][path_mask][finite_lag]
            p_peak_speed = roll["peak_speed"][path_mask][finite_lag]
            p_cmd = roll["cmd_speed"][path_mask][finite_lag]
            cmd_ok = np.isfinite(p_cmd) & (p_cmd > 0)
            results["path_tracking"] = {
                "n": int(len(p_lag)),
                "lag_median": _median(p_lag),
                "lag_p90": _pct(p_lag, 90),
                "lag_mean": float(np.mean(p_lag)),
                "lag_max": float(np.max(p_lag)),
                "keepup_threshold_m": keep,
                "keepup_share": _frac(p_lag < keep),
                "mean_speed_median": _median(p_mean_speed),
                "peak_speed_p90": _pct(p_peak_speed, 90),
                "cmd_speed_median": _median(p_cmd[cmd_ok]),
                "tracking_ratio_median": (
                    _median(p_mean_speed[cmd_ok] / np.maximum(p_cmd[cmd_ok], 1e-9))
                    if cmd_ok.any() else float("nan")
                ),
                "raw_goal_dist_median": _median(pos[path_mask]),
                "raw_goal_dist_p90": _pct(pos[path_mask], 90),
            }

    # ---- per goal category ----------------------------------------------
    cats = roll["category"]
    per_cat = {}
    for c in sorted(set(cats.tolist())):
        m = cats == c
        name = CATEGORY_NAMES.get(c, str(c))
        entry = {
            "n": int(np.sum(m)),
            "share": float(np.mean(m)),
            "pos_median": _median(pos[m]),
            "pos_p90": _pct(pos[m], 90),
            "heading_median": _median(head_deg[m]),
            "speed_median": _median(speed[m]),
            "metric_kind": "path_tracking" if int(c) == CATEGORY_PATH else "waypoint_pose",
        }
        if int(c) == CATEGORY_PATH:
            lag_c = roll["path_lag"][m]
            finite_lag_c = np.isfinite(lag_c)
            entry.update({
                "path_lag_median": _median(lag_c[finite_lag_c]),
                "path_lag_p90": _pct(lag_c[finite_lag_c], 90),
                "success_rate_strict": None,
                "arrived_then_left_share": None,
                "never_arrived_share": None,
            })
        else:
            entry.update({
                "success_rate_strict": _frac((pos[m] <= g_pos_med) & ok_head[m] & ok_stop[m]),
                "arrived_then_left_share": _frac(mode[m] == "arrived_then_left"),
                "never_arrived_share": _frac(mode[m] == "never_arrived"),
            })
        per_cat[name] = entry
    results["per_category"] = per_cat

    # ---- per start distance ---------------------------------------------
    per_dist = []
    if np.isfinite(start_d).any():
        for lo, hi in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:]):
            # Distance-to-final-goal is not a path metric.  Keep this table on
            # the exact same waypoint scope as failure modes and pose gates.
            m = gate_mask & np.isfinite(start_d) & (start_d >= lo) & (start_d < hi)
            if not m.any():
                continue
            per_dist.append({
                "lo": lo, "hi": hi, "n": int(np.sum(m)),
                "pos_median": _median(pos[m]),
                "pos_p90": _pct(pos[m], 90),
                "heading_median": _median(head_deg[m]),
                "arrived_then_left_share": _frac(mode[m] == "arrived_then_left"),
                "never_arrived_share": _frac(mode[m] == "never_arrived"),
            })
    results["per_start_distance"] = per_dist

    # ---- falls ------------------------------------------------------------
    fc = roll["fall_ctx"]
    if falls and len(fc["category"]):
        fcat = fc["category"].astype(int)
        results["fall_analysis"] = {
            "per_category": {CATEGORY_NAMES.get(int(c), str(int(c))): int(np.sum(fcat == c))
                             for c in sorted(set(fcat.tolist()))},
            "t_into_segment_median_s": _median(fc["t_into_segment"]),
            "start_dist_median_m": _median(fc["start_dist"]),
        }
    else:
        results["fall_analysis"] = None

    classified_falls = int(len(fc["category"]))
    fall_classification_complete = classified_falls == falls
    path_falls = (results["fall_analysis"]["per_category"].get("path", 0)
                  if results["fall_analysis"] else 0)
    waypoint_falls = max(0, classified_falls - int(path_falls))
    path_attempts = int(path_mask.sum()) + int(path_falls)
    waypoint_attempts = int(waypoint_mask.sum()) + int(waypoint_falls)
    overall_attempts = n + falls
    results["path_safety"] = {
        "falls": int(path_falls),
        "attempts_completed_plus_falls": path_attempts,
        "fall_classification_complete": fall_classification_complete,
        "falls_per_1000_attempts": (
            1000.0 * path_falls / path_attempts
            if fall_classification_complete and path_attempts else float("nan")),
    }
    results["waypoint_safety"] = {
        "falls": int(waypoint_falls),
        "attempts_completed_plus_falls": waypoint_attempts,
        "fall_classification_complete": fall_classification_complete,
        "falls_per_1000_attempts": (
            1000.0 * waypoint_falls / waypoint_attempts
            if fall_classification_complete and waypoint_attempts else float("nan")),
    }
    results["overall_safety"] = {
        "falls": falls,
        "attempts_completed_plus_falls": overall_attempts,
        "falls_per_1000_attempts": (
            1000.0 * falls / overall_attempts
            if overall_attempts else float("nan")),
    }

    results["recommendations"] = recommend(results, cfg)
    return results


def gate_ratios(results):
    """Normalized distance-to-passing per gate (<=1 means passing). Used to rank
    checkpoints: the worst ratio is 'how far from passing everything' this is."""
    def upper(value, limit):
        try:
            value, limit = float(value), float(limit)
        except (TypeError, ValueError):
            return float("inf")
        return (value / max(limit, 1.0e-9)
                if np.isfinite(value) and np.isfinite(limit) else float("inf"))

    def lower(value, limit):
        try:
            value, limit = float(value), float(limit)
        except (TypeError, ValueError):
            return float("inf")
        return (limit / max(value, 1.0e-9)
                if np.isfinite(value) and np.isfinite(limit) else float("inf"))

    g = results["gates"]
    hg = results.get("hbatch_gates") or {}
    if hg:
        path = results.get("path_tracking") or {}
        safety = results.get("path_safety") or {}
        waypoint_safety = results.get("waypoint_safety") or {}
        overall_safety = results.get("overall_safety") or {}
        failure_modes = results.get("failure_modes") or {}
        step = results.get("path_step_tracking") or {}
        accel = (results.get("path_acceleration_response") or {}).get(
            "time_to_1p0") or {}
        ratios = {
            # H has a documented G1 preservation exam.  Do not also inject the
            # legacy 5 cm / zero-fall MASTERPLAN gate: that makes the H0
            # acceptance criteria internally contradictory.
            "h_waypoint_pos_median": upper(
                results.get("pos_err_m", {}).get("median", float("nan")),
                hg.get("waypoint_pos_median_max_m", 0.0552)),
            "h_waypoint_pos_p90": upper(
                results.get("pos_err_m", {}).get("p90", float("nan")),
                hg.get("waypoint_pos_p90_max_m", 0.0742)),
            "h_waypoint_heading_median": upper(
                results.get("heading_err_deg", {}).get(
                    "median", float("nan")),
                hg.get("waypoint_heading_median_max_deg", 2.54)),
            "h_waypoint_never_arrived": upper(
                (failure_modes.get("never_arrived") or {}).get(
                    "share", float("nan")),
                hg.get("waypoint_never_arrived_share_max", 0.015)),
            "h_overall_falls": upper(
                overall_safety.get("falls_per_1000_attempts", float("nan")),
                hg.get("overall_falls_per_1000_max", 5.0)),
            "h_waypoint_falls": upper(
                waypoint_safety.get("falls_per_1000_attempts", float("nan")),
                hg.get("waypoint_falls_per_1000_max", 2.0)),
            "h_path_speed": lower(
                path.get("mean_speed_median", float("nan")),
                hg.get("path_speed_median_min", 0.95)),
            "h_path_falls": upper(
                safety.get("falls_per_1000_attempts", float("nan")),
                hg.get("path_falls_per_1000_max", 5.0)),
            "h_path_steady_samples": (
                0.0 if step.get("steady_samples_excluding_recovery", 0) > 0
                else float("inf")),
            "h_path_floor": upper(
                step.get("below_0p75_share_excluding_recovery", float("nan")),
                hg.get("path_floor_below_0p75_max", 0.10)),
            "h_path_recovery_share": upper(
                step.get("dwell_resume_recovery_share", float("nan")),
                hg.get("path_dwell_resume_recovery_share_max", 0.15)),
            "h_path_leash": upper(
                step.get("outside_leash_share", float("nan")),
                hg.get("path_outside_leash_max", 0.01)),
            "h_from_rest_t1_reached": lower(
                accel.get("reached_share", float("nan")),
                hg.get("time_to_1mps_reached_share_min", 0.80)),
            "h_from_rest_t1_p90": upper(
                accel.get("p90_s", float("nan")),
                hg.get("time_to_1mps_p90_s_max", 3.0)),
        }
        desc = str(results.get("experiment_description", ""))
        if desc.startswith("H1_") or desc.startswith("H2_"):
            mirror = results.get("symmetry_eval") or {}
            stable = results.get("high_speed_stability") or {}
            gait = results.get("gait_touchdown") or {}
            ratios.update({
                "h_mirror_p90": upper(
                    mirror.get("p90", float("nan")),
                    hg.get("mirror_error_p90_max", 0.10)),
                # H1 is H2's parent.  If checkpoint selection lets H1 win with
                # no valid cruise samples, the cross-arm comparator cannot tell
                # whether H2 preserved or improved high-speed behaviour.
                "h_cruise_coverage": lower(
                    stable.get("cruise_share_of_valid", float("nan")),
                    hg.get("cruise_share_of_valid_min", 0.05)),
                "h_touchdown_samples": lower(
                    gait.get("samples", float("nan")),
                    hg.get("touchdown_samples_min", 100)),
                "h_touchdown_lr_bias": upper(
                    gait.get("left_right_heel_x_median_abs_diff_m", float("nan")),
                    hg.get("touchdown_lr_bias_max_m", 0.02)),
            })
        if desc.startswith("H2_"):
            stable = results.get("high_speed_stability") or {}
            ratios.update({
                "h_cruise_pitch": upper(
                    stable.get("cruise_pitch_abs_p90_deg", float("nan")),
                    hg.get("cruise_pitch_p90_max_deg", 20.0)),
                "h_cruise_roll": upper(
                    stable.get("cruise_roll_abs_p90_deg", float("nan")),
                    hg.get("cruise_roll_p90_max_deg", 15.0)),
                "h_cruise_ang_xy": upper(
                    stable.get("cruise_ang_xy_p90_radps", float("nan")),
                    hg.get("cruise_ang_xy_p90_max_radps", 3.0)),
            })
        if desc.startswith("H3_"):
            gait = results.get("gait_touchdown") or {}
            ratios.update({
                # Selection must optimize the lever H3 actually changes;
                # cross-arm H0 non-regression remains a later full-suite gate.
                "h_touchdown_samples": lower(
                    gait.get("samples", float("nan")),
                    hg.get("touchdown_samples_min", 100)),
                "h_touchdown_target": lower(
                    gait.get("within_dynamic_target_one_sigma_share", float("nan")),
                    hg.get("touchdown_target_share_min", 0.40)),
                "h_touchdown_overstride": upper(
                    gait.get("overstride_share", float("nan")),
                    hg.get("touchdown_overstride_share_max", 0.10)),
            })
    else:
        fall_ref = max(results["segments_completed"], 1) * 0.002
        ratios = {
            "pos_median": upper(
                g["pos_median"]["value"], g["pos_median"]["limit"]),
            "pos_p90": upper(
                g["pos_p90"]["value"], g["pos_p90"]["limit"]),
            "heading_median": upper(
                g["heading_median"]["value"], g["heading_median"]["limit"]),
            "falls": upper(results["falls"], fall_ref),
        }
    return ratios


# --------------------------------------------------------------------------
# recommendations
# --------------------------------------------------------------------------

def recommend(r, cfg):
    """Map the measured failure signature onto specific yaml keys.

    Deliberately opinionated: each branch names the knob and the direction, with
    the reason, so the report can be acted on without re-deriving the analysis.
    """
    out = []
    rw = cfg["rewards"]
    scales = rw.get("scales", {})
    c_weight = rw.get("constellation_weight", 0.2)
    c_radius = rw.get("constellation_radius", 1.0)
    reach_radius = rw.get("goal_reach_radius", 0.1)
    fm = r["failure_modes"]
    g = r["gates"]

    feas = r.get("feasibility")
    if (feas and feas["infeasible_share"] > 0.10
            and np.isfinite(feas["infeasible_subset"]["pos_median"])
            and np.isfinite(feas["feasible_subset"]["pos_median"])):
        out.append(
            "**과제 실현가능성 먼저 확인**: 전체 구간의 {:.0f}%가 주어진 시간 안에 도달 불가"
            "(필요속도 > {:.2f} m/s, 필요속도 p90 = {:.2f} m/s). 이 구간들의 위치오차 median은 "
            "{:.1f} cm로 실현가능 구간({:.1f} cm)보다 크며, 게이트 수치를 정책 성능이 아니라 "
            "과제 정의가 깎고 있다. → `commands.resampling_time_s`를 [4,8]에서 [6,10]으로 늘리거나 "
            "`commands.goal_dx/goal_dy` 범위를 줄일 것. V3라면 `commands.goal_curriculum`을 켜서 "
            "성공률에 맞춰 범위가 자동으로 자라게 하는 쪽이 낫다.".format(
                feas["infeasible_share"] * 100, feas["feasible_speed_mps"], feas["required_speed_p90"],
                feas["infeasible_subset"]["pos_median"] * 100, feas["feasible_subset"]["pos_median"] * 100))

    failing = {k: v["share"] for k, v in fm.items()
               if k != "ok" and np.isfinite(v.get("share", float("nan")))}
    dominant = max(failing, key=failing.get) if failing else None

    if dominant == "never_arrived" and failing["never_arrived"] > 0.15:
        out.append(
            "**주 실패 모드: 애초에 도달 못함** ({:.0f}%가 목표 {:.0f}cm 안에 한 번도 못 들어옴). "
            "constellation은 exp(-w·d²) 형태라 멀리서는 기울기가 거의 사라진다 "
            "(w={}, d=2m → exp(-{:.1f})). → `rewards.scales.goal_progress`를 0.5~1.0으로 켜서 "
            "원거리 유인을 주거나, `rewards.constellation_weight`를 {}에서 0.1로 낮춰 basin을 넓힐 것. "
            "둘 다 하면 과하니 하나씩.".format(
                failing["never_arrived"] * 100, g["pos_median"]["limit"] * 100,
                c_weight, c_weight * 4.0, c_weight))

    if dominant == "arrived_then_left" and failing["arrived_then_left"] > 0.10:
        out.append(
            "**주 실패 모드: 도달했다가 이탈** ({:.0f}%가 목표 {:.0f}cm 안에 들어갔다가 {:.0f}cm 밖에서 끝남). "
            "이 경우 `goal_progress`는 오히려 해롭다(계속 움직일 이유를 준다). "
            "→ `rewards.scales.goal_reached`를 +1~2로 켜서 '도착해서 멈춰 있는 상태' 자체를 보상하고, "
            "`goal_stop`을 {}에서 -3으로, `stand_posture`를 -0.5로 줘서 도착 자세를 고정할 것. "
            "V3의 `rewards.final_window_s` timed gate가 정확히 이 실패를 겨냥한 장치다.".format(
                failing["arrived_then_left"] * 100, g["pos_median"]["limit"] * 100,
                g["pos_p90"]["limit"] * 100, scales.get("goal_stop", 0.0)))

    if fm["not_stopped"]["share"] > 0.10:
        out.append(
            "**도착 후 안 멈춤** ({:.0f}%가 위치·방향은 맞았는데 속도 > {:.2f} m/s). "
            "→ `rewards.scales.goal_stop`을 {}에서 -3으로 강화하거나 `goal_reached`를 켤 것 "
            "(`goal_reach_radius` = {} m 안에서만 작동하므로 이 반경도 함께 확인).".format(
                fm["not_stopped"]["share"] * 100, r["stop_speed_threshold"],
                scales.get("goal_stop", 0.0), reach_radius))

    pos_bad = not (g["pos_median"]["pass"] and g["pos_p90"]["pass"])
    head_bad = not g["heading_median"]["pass"]
    if pos_bad and not head_bad:
        out.append(
            "**위치만 미달, heading은 통과**: constellation의 d_con = d² + 2r²(1-cosθ)에서 "
            "r={} m라 heading 항의 가중이 상대적으로 크다. → `rewards.constellation_radius`를 0.6~0.7로 "
            "낮춰 위치 항 비중을 올리거나, `rewards.goal_position_sigma`를 {}에서 절반으로 줄여 "
            "근거리 정밀도를 세울 것.".format(c_radius, rw.get("goal_position_sigma", 1.0)))
    elif head_bad and not pos_bad:
        out.append(
            "**heading만 미달, 위치는 통과**: → `rewards.constellation_radius`를 {}에서 1.4~1.6으로 올려 "
            "heading 커플링(2r²)을 키우거나, `rewards.scales.heading_near_goal`을 1.0으로 켜서 "
            "근거리에서만 heading을 요구할 것.".format(c_radius))
    elif pos_bad and head_bad:
        out.append("**위치·heading 동시 미달**: 아직 과제 자체를 못 푸는 단계다. 보상 미세조정보다 "
                   "학습량/커리큘럼(V3 `goal_curriculum`)을 먼저 의심할 것.")

    ap = r.get("approach_error_m")
    if ap and np.isfinite(ap["along_median"]):
        if ap["overshoot_share"] > 0.65:
            out.append("**오버슈트 경향**: 접근 방향 기준 잔차 median {:+.2f} m ({:.0f}%가 목표를 지나침). "
                       "감속이 늦다 → `goal_stop` 강화 또는 `action_rate`/`dof_vel` 페널티 재확인.".format(
                           ap["along_median"], ap["overshoot_share"] * 100))
        elif ap["overshoot_share"] < 0.35:
            out.append("**언더슈트 경향**: 접근 방향 기준 잔차 median {:+.2f} m ({:.0f}%가 목표 앞에서 멈춤). "
                       "마지막 몇 cm를 좁힐 유인이 없다 → `goal_position_sigma` 축소 또는 "
                       "`goal_reached`(+)로 최종 도달을 명시적으로 보상.".format(
                           ap["along_median"], (1 - ap["overshoot_share"]) * 100))

    pc = r.get("per_category") or {}
    ranked = [(k, v) for k, v in pc.items()
              if v.get("metric_kind") == "waypoint_pose"
              and v["n"] >= 30 and np.isfinite(v["pos_median"])]
    if len(ranked) >= 2:
        ranked.sort(key=lambda kv: kv[1]["pos_median"])
        best, worst = ranked[0], ranked[-1]
        if worst[1]["pos_median"] > 2.0 * max(best[1]["pos_median"], 1e-4):
            out.append(
                "**유형 편차 큼**: `{}` 구간의 위치오차 median {:.1f} cm vs `{}` {:.1f} cm. "
                "전체 median만 보면 가려지는 차이다 → `commands.goal_categories`에서 `{}` 비중을 "
                "{:.2f}에서 올려 해당 유형을 더 학습시킬 것.".format(
                    worst[0], worst[1]["pos_median"] * 100, best[0], best[1]["pos_median"] * 100,
                    worst[0], cfg["commands"].get("goal_categories", {}).get(worst[0], 0.0)))

    fa = r.get("fall_analysis")
    if r["falls"] > 0:
        detail = ""
        if fa:
            top = max(fa["per_category"], key=fa["per_category"].get)
            detail = " 낙상의 최다 유형은 `{}`({}회), 구간 시작 후 median {:.1f}s 시점.".format(
                top, fa["per_category"][top], fa["t_into_segment_median_s"])
        out.append("**낙상 {}회** (기준 {}회).{} → 영상에서 해당 시점 확인(`--record_video`), "
                   "`rewards.terminate_height`({})와 `orientation`/`base_height` 페널티 균형 점검.".format(
                       r["falls"], r["gates"]["falls"]["limit"], detail, rw.get("terminate_height", "?")))

    if r["all_gates_pass"]:
        if r.get("hbatch_gates"):
            out.append("legacy MASTERPLAN 게이트는 통과했다. H arm 채택은 아직 별개이므로 "
                       "7-report suite와 H0–H3 cross-arm 비교까지 완료할 것.")
        else:
            out.append("게이트 전부 통과 → 영상 확인 후 milestone 4(export + MuJoCo 검증)로 진행.")

    out.append("눈으로 확인: `python eval_goal_pose.py ... --record_video` (env 0을 mp4로 저장) 또는 로컬에서 `play.py`.")
    return out


# --------------------------------------------------------------------------
# report rendering
# --------------------------------------------------------------------------

def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return lines


def _pf(ok):
    return "✅ PASS" if ok else "❌ FAIL"


def render_report(r):
    g = r["gates"]
    ci = r["ci95"]
    md = []
    md.append("# GoalPose 평가 리포트 — {}".format(r["date"]))
    md.append("")
    md.append("- checkpoint: `{}`".format(r["checkpoint"]))
    md.append("- config: `{}`".format(r["config"]))
    md.append("- task-state protocol: `{}`".format(
        r.get("task_state_protocol", "config_fixed")))
    if r.get("effective_eval_protocol_sha"):
        md.append("- effective eval protocol: `{}`".format(
            r["effective_eval_protocol_sha"]))
    if r.get("joint_dr_probe", {}).get("active"):
        md.append("- joint-DR probe (rad): encoder ±{}, target ±{}, init σ{}".format(
            r["joint_dr_probe"].get("joint_encoder_bias_rad"),
            r["joint_dr_probe"].get("joint_target_offset_rad"),
            r["joint_dr_probe"].get("init_dof_std_rad")))
    md.append("- 조건: {} envs × {:.0f}s, {} 정책, 외란 {}, 관측노이즈 {}, seed {}".format(
        r["num_envs"], r["duration_s"],
        "결정론적" if r["deterministic"] else "확률적",
        "ON" if r["perturbations"] else "OFF",
        "ON" if r["obs_noise"] else "OFF", r["seed"]))
    t = r["timing"]
    md.append("- 벽시계: setup {:.1f}s + rollout {:.1f}s; env당 {:.1f}× real-time, 총 {:.0f} env·s/wall-s".format(
        t["setup_wall_s"], t["rollout_wall_s"], t["single_env_realtime_factor"],
        t["aggregate_env_seconds_per_wall_second"]))
    md.append("- 완료 구간 {}개 (waypoint {}개, path {}개; 게이트 채점 {}개) / 낙상 {}회 / 에피소드경계 절단 {}개".format(
        r["segments_completed"], r.get("segments_waypoint", r["segments_completed"]),
        r.get("segments_path", 0), r.get("segments_scored_by_gates", r["segments_completed"]),
        r["falls"], r["segments_censored_by_episode_end"]))
    md.append("")

    md.append("## 게이트 참고치 (탐색용 preview — 공식 판정 아님)" if not r["authoritative_gate_evaluation"]
              else "## 게이트 판정 (MASTERPLAN §성공 기준)")
    md.append("")
    md += _table(
        ["게이트", "기준", "측정값", "95% CI", "판정"],
        [
            ["위치 오차 median", "≤ {:.0f} cm".format(g["pos_median"]["limit"] * 100),
             "{:.1f} cm".format(g["pos_median"]["value"] * 100),
             "[{:.1f}, {:.1f}]".format(ci["pos_median"][0] * 100, ci["pos_median"][1] * 100),
             _pf(g["pos_median"]["pass"])],
            ["위치 오차 p90", "≤ {:.0f} cm".format(g["pos_p90"]["limit"] * 100),
             "{:.1f} cm".format(g["pos_p90"]["value"] * 100),
             "[{:.1f}, {:.1f}]".format(ci["pos_p90"][0] * 100, ci["pos_p90"][1] * 100),
             _pf(g["pos_p90"]["pass"])],
            ["heading 오차 median", "≤ {:.0f}°".format(g["heading_median"]["limit"]),
             "{:.1f}°".format(g["heading_median"]["value"]),
             "[{:.1f}, {:.1f}]".format(ci["heading_median"][0], ci["heading_median"][1]),
             _pf(g["heading_median"]["pass"])],
            ["낙상", "≤ {}".format(g["falls"]["limit"]), "{}".format(g["falls"]["value"]), "—",
             _pf(g["falls"]["pass"])],
        ])
    md.append("")
    if r["authoritative_gate_evaluation"]:
        if r.get("hbatch_gates"):
            md.append("**legacy MASTERPLAN 참고 판정: {} (H 채택은 full-suite cross-arm gate에서 별도 판정)**"
                      .format("✅ 통과" if r["all_gates_pass"] else "❌ 미통과"))
        else:
            md.append("**종합: {}**".format(
                "✅ 전체 게이트 통과" if r["all_gates_pass"] else "❌ 미통과 게이트 있음"))
    else:
        md.append("**탐색 결과: {} (표본/조건이 표준 프로토콜이 아니므로 공식 판정 아님)**".format(
            "모든 수치 기준 충족" if r["all_gates_pass"] else "미충족 수치 있음"))
    if r.get("segments_path", 0):
        md.append("")
        md.append("> 참고: 위 게이트는 **waypoint 구간만** 채점한다. path 구간은 목표가 계속 앞서가는 "
                  "moving-carrot 과제라 final position error가 도착 오차가 아니며, 아래 `path 추종` "
                  "섹션의 lag/speed 지표로 따로 본다.")
    md.append("")

    overall_safety = r.get("overall_safety") or {}
    waypoint_safety = r.get("waypoint_safety") or {}
    path_safety = r.get("path_safety") or {}
    if overall_safety:
        md.append("## 낙상 안전성 — survivor bias 보정")
        md.append("")
        md.append("완료 구간만 분모로 쓰면 낙상한 시도가 사라지므로 `완료+낙상`을 "
                  "attempt로 센다. fall context 분류 완전성: **{}**.".format(
                      "PASS" if path_safety.get(
                          "fall_classification_complete", False) else "FAIL"))
        md.append("")
        md += _table(
            ["범위", "falls / attempts", "falls / 1000 attempts"],
            [["전체", "{} / {}".format(
                overall_safety.get("falls", 0),
                overall_safety.get("attempts_completed_plus_falls", 0)),
              "{:.3f}".format(overall_safety.get(
                  "falls_per_1000_attempts", float("nan")))],
             ["waypoint", "{} / {}".format(
                waypoint_safety.get("falls", 0),
                waypoint_safety.get("attempts_completed_plus_falls", 0)),
              "{:.3f}".format(waypoint_safety.get(
                  "falls_per_1000_attempts", float("nan")))],
             ["path", "{} / {}".format(
                path_safety.get("falls", 0),
                path_safety.get("attempts_completed_plus_falls", 0)),
              "{:.3f}".format(path_safety.get(
                  "falls_per_1000_attempts", float("nan")))]])
        md.append("")

    hs = r.get("high_speed_stability")
    if hs:
        md.append("## 고속 안정성 — 가속과 순항 분리")
        md.append("")
        md.append("가속 중 전방 lean은 빠른 가속에 유용할 수 있으므로 실패로 세지 않는다. "
                  "`v≥{:.1f} m/s`이면서 `|a_xy|≤{:.1f} m/s²`인 순항만 별도로 채점한다.".format(
                      hs["phase_definition"]["high_speed_threshold_mps"],
                      hs["phase_definition"]["steady_accel_threshold_mps2"]))
        md.append("")
        md += _table(
            ["phase 지표", "값"],
            [["가속 pitch |.| median / p90", "{:.1f}° / {:.1f}°".format(
                hs["accel_pitch_abs_p50_deg"], hs["accel_pitch_abs_p90_deg"])],
             ["순항 pitch |.| p90", "{:.1f}°".format(hs["cruise_pitch_abs_p90_deg"])],
             ["순항 roll |.| p90", "{:.1f}°".format(hs["cruise_roll_abs_p90_deg"])],
             ["순항 |ωxy| p90", "{:.2f} rad/s".format(hs["cruise_ang_xy_p90_radps"])],
             ["순항 |vz| p90", "{:.2f} m/s".format(hs["cruise_abs_z_vel_p90_mps"])],
             ["유효 sample 중 순항", "{:.1f}%".format(hs["cruise_share_of_valid"] * 100)]])
        md.append("")

    pa = r.get("path_acceleration_response")
    if pa:
        md.append("## path 가속 응답")
        md.append("")
        md.append(pa.get("definition", ""))
        md.append("")
        md += _table(
            ["임계", "도달 비율", "도달한 구간 p50 / p90"],
            [[label, "{:.1f}% (n={})".format(
                100 * pa[key]["reached_share"], pa[key]["eligible"]),
              "{:.2f} / {:.2f} s".format(pa[key]["p50_s"], pa[key]["p90_s"])]
             for label, key in (("0.5 m/s", "time_to_0p5"),
                                ("0.8 m/s", "time_to_0p8"),
                                ("1.0 m/s", "time_to_1p0"))])
        md.append("")

    dr = r.get("directional_response")
    if dr:
        md.append("## 급격한 {} goal 응답".format(dr["pattern"]))
        md.append("")
        md.append("- 시작 goal bearing |.| median {:.1f}°, p10–p90 {:.1f}–{:.1f}°".format(
            dr["initial_goal_bearing_abs_deg"]["median"],
            dr["initial_goal_bearing_abs_deg"]["p10"],
            dr["initial_goal_bearing_abs_deg"]["p90"]))
        md.append("- 시작 후 2 s 내 최저속도 median/p10 {:.2f}/{:.2f} m/s".format(
            dr["min_speed_first_2s_mps"]["median"], dr["min_speed_first_2s_mps"]["p10"]))
        md.append("- 응답 속도 정의: {}".format(dr.get("speed_definition", "goal-direction closing speed")))
        md += _table(
            ["임계", "도달 비율", "p50 / p90"],
            [[label, "{:.1f}%".format(100 * dr[key]["reached_share"]),
              "{:.2f} / {:.2f} s".format(dr[key]["p50_s"], dr[key]["p90_s"])]
             for label, key in (("0.5 m/s", "time_to_0p5"),
                                ("0.8 m/s", "time_to_0p8"),
                                ("1.0 m/s", "time_to_1p0"))])
        md.append("- gait phase quarter별 같은 지표는 JSON에 저장된다. 특정 발 접지 시점에만 "
                  "side/back 응답이 느려지는지 분리할 수 있다.")
        md.append("")

    sy = r.get("symmetry_eval")
    if sy:
        md.append("## 좌우 policy equivariance")
        md.append("")
        md.append("`RMS(π(Ms) − Mπ(s))` action error: median {:.3f}, p90 {:.3f}, p99 {:.3f} "
                  "({} samples).".format(
                      sy["median"], sy["p90"], sy["p99"], sy["samples"]))
        md.append("")

    td = r.get("gait_touchdown")
    if td:
        md.append("## 첫 접지 heel / impact")
        md.append("")
        md.append(td.get("scope", ""))
        md.append("")
        md += _table(
            ["지표", "값"],
            [["표본", td["samples"]],
             ["heel이 trunk보다 앞", "{:.1f}%".format(
                 100 * td["ahead_of_trunk_share"])],
             ["dynamic target ±1σ", "{:.1f}%".format(
                 100 * td["within_dynamic_target_one_sigma_share"])],
             ["heel x body p10 / med / p90", "{:.3f} / {:.3f} / {:.3f} m".format(
                 td["heel_x_body_m"]["p10"], td["heel_x_body_m"]["median"],
                 td["heel_x_body_m"]["p90"])],
             ["left/right heel median 차이", "{:.3f} m".format(
                 td["left_right_heel_x_median_abs_diff_m"])],
             ["overstride", "{:.1f}%".format(100 * td["overstride_share"])],
             ["접지 직전 하강속도 p90", "{:.2f} m/s".format(
                 td["precontact_down_speed_p90_mps"])],
             ["접지 force p90", "{:.1f} N".format(
                 td["contact_force_p90_n"])]])
        md.append("")

    de = r.get("disturbance_eval") or {}
    if de:
        md.append("## 외력 노출 감사")
        md.append("")
        md.append("- 감지된 force event: {}회; 외력 active env-step 비율 {:.2f}%".format(
            de.get("events", 0), 100 * de.get("active_share", 0.0)))
        md.append("- physics 적용 전 reset으로 취소된 schedule: {}회 (event/outcome에서 제외)".format(
            de.get("scheduled_events_cancelled_before_application", 0)))
        md.append("- 외력 인가 중 낙상: {}회".format(de.get("falls_during_force", 0)))
        md.append("- 5 s outcome record {} / censored·overlap {}".format(
            de.get("events_with_outcome_record", 0), de.get("events_censored_or_overlapping", 0)))
        rows = []
        for label, key in (("전체", "overall"), ("고속", "high_speed"),
                           ("충돌", "collision"), ("support", "support")):
            e = de.get(key) or {}
            if e.get("records", 0):
                rows.append([
                    label, e["records"], "{:.1f}/{:.1f}% ({}/{})".format(
                        100 * e["survival_2s"], 100 * e["survival_5s"],
                        e.get("survival_2s_eligible", 0),
                        e.get("survival_5s_eligible", 0)),
                    "{:.1f}".format(e["impulse_ns_p90"]),
                    "{:.1f}°".format(e["max_tilt_deg_p90"]),
                    "{:.2f}".format(e["speed_loss_mps_p90"]),
                    "{:.1f}% / {:.2f}s".format(
                        100 * e["recovery_90_within_5s_share"], e["recovery_90_s_p90"]),
                ])
        if rows:
            md += _table(
                ["유형", "n", "2s/5s 생존 (denom)", "impulse p90 N·s", "max tilt p90",
                 "speed loss p90", "90% 회복≤5s / p90"], rows)
        if de.get("events", 0) == 0:
            md.append("- ⚠️ 외력 event가 0회이므로 이 report는 충돌 강건성 근거로 사용할 수 없다.")
        if de.get("video_requested", False):
            recorded = int(de.get("video_recorded_frames", 0))
            active = int(de.get("video_force_active_frames", 0))
            drawn = int(de.get("video_force_arrow_drawn_frames", 0))
            if recorded:
                md.append("- 영상 env0 외력 물리 frame: **{} / {}** ({:.1f}%); "
                          "renderer가 실제 화면 안에 그린 빨간 화살표: **{} frame**."
                          .format(active, recorded,
                                  100 * de.get("video_force_active_share", 0.0), drawn))
            else:
                md.append("- ❌ 영상을 요청했지만 실제 기록 frame이 0개다. 외력 화살표 가시성을 검증할 수 없다.")
        md.append("")

    bs = r.get("body_speed")
    if bs:
        md.append("## 몸통 속도 (body velocity)")
        md.append("")
        md.append("위의 오차 지표는 전부 '거리'다. 속도를 따로 보지 않으면 **느린 정책**과 "
                  "**빠르게 갈 이유가 없었던 정책**을 구분할 수 없다.")
        md.append("")
        md.append("**정의**: 몸통 링크의 순간 선속도가 아니라 **pose(x, y, θ)의 시간미분**이다. "
                  "목표가 SE(2) pose이므로 그 pose가 얼마나 빨리 변하는지가 정직한 속도이고, "
                  "0.2 s 창으로 차분해 **한 걸음 안의 몸통 흔들림이 평균화**된다. "
                  "순간 선속도로 재면 p99·최대가 이동속도가 아니라 보행 중 흔들림을 보고한다.")
        md.append("")
        md += _table(
            ["지표", "값"],
            [["median", "{:.2f} m/s".format(bs["median"])],
             ["p90", "{:.2f} m/s".format(bs["p90"])],
             ["p99", "{:.2f} m/s".format(bs["p99"])],
             ["순간 최대", "{:.2f} m/s".format(bs["max_instant"])],
             ["구간 최고속도 median", "{:.2f} m/s".format(bs["segment_peak_median"])],
             ["구간 최고속도 p90", "{:.2f} m/s".format(bs["segment_peak_p90"])],
             ["구간 최고속도 최대", "{:.2f} m/s".format(bs["segment_peak_max"])],
             ["0.5 m/s 초과 시간 비율", "{:.1f}%".format(bs["share_above_0p5"] * 100)],
             ["1.0 m/s 초과 시간 비율", "{:.1f}%".format(bs["share_above_1p0"] * 100)]])
        md.append("")
        if bs["share_above_1p0"] < 0.01:
            md.append("> ⚠️ 1.0 m/s를 넘긴 시간이 {:.1f}%다. 목표 속도(1.3~1.5 m/s)를 학습·측정하려면 "
                      "**목표 샘플링이 그 속도를 요구해야 한다** — 아래 실현가능성 표의 '필요속도'를 "
                      "함께 볼 것. 필요속도 p90이 0.3 m/s 수준이면 이 수치는 정책의 한계가 아니라 "
                      "과제의 한계다.".format(bs["share_above_1p0"] * 100))
            md.append("")

    st = r.get("speed_tracking")
    if st and st["bins"]:
        md.append("## 명령속도 vs 실제속도 (path mode)")
        md.append("")
        md.append("추종비 = 실제/명령. 1.0 근처면 따라가는 것이고, 명령을 올려도 실제가 "
                  "안 오르면 **거기가 이 로봇의 물리적 한계**다.")
        md.append("")
        md += _table(
            ["명령속도 구간", "n", "명령 median", "실제 평균속도", "실제 최고속도", "추종비"],
            [["{:.1f}–{:.1f} m/s".format(b["cmd_lo"], b["cmd_hi"]) if b["cmd_hi"] != float("inf")
              else "{:.1f}+ m/s".format(b["cmd_lo"]),
              b["n"], "{:.2f}".format(b["cmd_median"]),
              "{:.2f} m/s".format(b["achieved_mean_median"]),
              "{:.2f} m/s".format(b["achieved_peak_median"]),
              "{:.2f}".format(b["tracking_ratio"])] for b in st["bins"]])
        md.append("")
        knee = [b for b in st["bins"] if b["tracking_ratio"] < 0.75]
        if knee:
            md.append("> 📌 추종비가 0.75 아래로 떨어지는 첫 구간: **{:.1f}–{:.1f} m/s** "
                      "(실제 평균 {:.2f} m/s). 이 부근이 현재 정책의 지속 가능 상한이다."
                      .format(knee[0]["cmd_lo"], knee[0]["cmd_hi"], knee[0]["achieved_mean_median"]))
        else:
            md.append("> 📌 모든 구간에서 추종비 0.75 이상 — **아직 한계에 도달하지 않았다.** "
                      "`commands.path.speed_range_mps` 상단을 더 올려 한계를 찾을 것.")
        md.append("")

    pst = r.get("path_step_tracking")
    if pst:
        md.append("## path step floor/leash 감사")
        md.append("")
        md.append("각 control step의 running path만 집계한다(dwell/reset 제외). 전역 최소값이 아니라 "
                  "**각 env가 실제로 샘플한 lookahead와 leash**를 사용한다. "
                  "signed error `gap-lookahead`가 음수면 floor deficit, 양수면 behind lag다.")
        md.append("")
        md += _table(
            ["step 지표", "값"],
            [["유효 running path sample", pst["samples"]],
             ["gap/lookahead p2 / p50", "{:.3f} / {:.3f}".format(
                 pst["gap_over_lookahead_p2"], pst["gap_over_lookahead_p50"])],
             ["gap/lookahead < 0.75", "{:.2f}%".format(
                 100 * pst["below_0p75_share"])],
             ["dwell-resume floor 복구 중", "{:.2f}%".format(
                 100 * pst["dwell_resume_recovery_share"])],
             ["복구 transition 제외 gap/lookahead < 0.75", "{:.2f}%".format(
                 100 * pst["below_0p75_share_excluding_recovery"])],
             ["gap p2 / p50", "{:.3f} / {:.3f} m".format(
                 pst["gap_m_p2"], pst["gap_m_p50"])],
             ["per-env lookahead p2 / p50", "{:.3f} / {:.3f} m".format(
                 pst["lookahead_m_p2"], pst["lookahead_m_p50"])],
             ["per-env leash p2 / p50", "{:.3f} / {:.3f} m".format(
                 pst["leash_m_p2"], pst["leash_m_p50"])],
             ["floor deficit p50 / p90", "{:.1f} / {:.1f} cm".format(
                 100 * pst["floor_deficit_m_p50"],
                 100 * pst["floor_deficit_m_p90"])],
             ["behind lag p50 / p90", "{:.1f} / {:.1f} cm".format(
                 100 * pst["behind_lag_m_p50"],
                 100 * pst["behind_lag_m_p90"])],
             ["per-env leash 밖", "{:.2f}%".format(
                 100 * pst["outside_leash_share"])]])
        md.append("")

    pt = r.get("path_tracking")
    if pt:
        md.append("## path 추종")
        md.append("")
        md.append("path mode에서는 목표가 lookahead로 앞서가는 것이 정상이다. 따라서 raw goal distance는 "
                  "도착 오차가 아니라 carrot과의 간격이고, 여기서는 `path_lag = max(gap - lookahead_min, 0)`를 본다.")
        md.append("")
        md += _table(
            ["지표", "값"],
            [["path 구간 수", pt["n"]],
             ["path_lag median", "{:.1f} cm".format(pt["lag_median"] * 100)],
             ["path_lag p90", "{:.1f} cm".format(pt["lag_p90"] * 100)],
             ["path_lag max", "{:.1f} cm".format(pt["lag_max"] * 100)],
             ["keepup 기준", "{:.1f} m".format(pt["keepup_threshold_m"])],
             ["keepup 비율", "{:.1f}%".format(pt["keepup_share"] * 100)],
             ["명령속도 median", "{:.2f} m/s".format(pt["cmd_speed_median"])],
             ["실제 평균속도 median", "{:.2f} m/s".format(pt["mean_speed_median"])],
             ["구간 최고속도 p90", "{:.2f} m/s".format(pt["peak_speed_p90"])],
             ["추종비 median", "{:.2f}".format(pt["tracking_ratio_median"])],
             ["raw goal distance median", "{:.1f} cm (참고용)".format(pt["raw_goal_dist_median"] * 100)]])
        md.append("")

    feas = r.get("feasibility")
    if feas:
        md.append("## 과제 실현가능성 점검")
        md.append("")
        md.append("목표는 4~8s마다 재샘플되므로, 먼 목표에는 애초에 도달할 시간이 없을 수 있다. "
                  "필요속도 = 초기거리 / 구간시간.")
        md.append("")
        md.append("- 필요속도 median {:.2f} m/s, p90 {:.2f} m/s (실현가능 기준 {:.2f} m/s)".format(
            feas["required_speed_median"], feas["required_speed_p90"], feas["feasible_speed_mps"]))
        md.append("- 이 정책이 실제로 낸 접근속도: median {:.2f} m/s, p95 {:.2f} m/s "
                  "— 기준값 {:.2f}와 크게 다르면 `evaluation.feasible_speed_mps`를 실측에 맞게 조정할 것".format(
                      feas["achieved_closing_speed_median"], feas["achieved_closing_speed_p95"],
                      feas["feasible_speed_mps"]))
        md.append("- **시간 내 도달 불가 구간: {:.1f}%** ({}개)".format(
            feas["infeasible_share"] * 100, feas["infeasible_count"]))
        md += _table(
            ["부분집합", "n", "위치 median", "위치 p90", "heading median"],
            [["실현가능", feas["feasible_subset"]["n"],
              "{:.1f} cm".format(feas["feasible_subset"]["pos_median"] * 100),
              "{:.1f} cm".format(feas["feasible_subset"]["pos_p90"] * 100),
              "{:.1f}°".format(feas["feasible_subset"]["heading_median"])],
             ["시간부족", feas["infeasible_subset"]["n"],
              "{:.1f} cm".format(feas["infeasible_subset"]["pos_median"] * 100),
              "{:.1f} cm".format(feas["infeasible_subset"]["pos_p90"] * 100), "—"]])
        md.append("")
        if feas["infeasible_share"] > 0.10:
            md.append("> ⚠️ 실현 불가 구간 비중이 높다. 위 게이트 수치는 정책 성능과 과제 정의가 "
                      "섞인 값이므로, 보상을 만지기 전에 목표 범위/시간부터 조정할 것.")
            md.append("")

    fm = r["failure_modes"]
    md.append("## 실패 모드 분해")
    md.append("")
    md.append("최종 오차가 같아도 원인이 다르면 처방이 반대다. 이 표는 **게이트와 같은 scope**"
              "(`{}`)에서 계산한다. path 구간은 도착/이탈 개념이 맞지 않아 제외된다.".format(
                  r.get("failure_modes_scope", "waypoint_only")))
    md.append("")
    labels = {
        "ok": "성공 (위치·heading·정지 모두 충족)",
        "not_stopped": "도착했으나 안 멈춤",
        "heading_only": "위치는 맞고 heading만 미달",
        "arrived_then_left": "도달했다가 이탈",
        "never_arrived": "한 번도 도달 못함",
    }
    md += _table(["모드", "구간 수", "비율"],
                 [[labels[k], fm[k]["count"], "{:.1f}%".format(fm[k]["share"] * 100)]
                  for k in ("ok", "not_stopped", "heading_only", "arrived_then_left", "never_arrived")])
    md.append("")
    md.append("- 최근접 거리 median {:.1f} cm / p90 {:.1f} cm (구간 중 목표에 가장 가까웠던 순간)".format(
        r["closest_approach_m"]["median"] * 100, r["closest_approach_m"]["p90"] * 100))
    ap = r.get("approach_error_m")
    if ap:
        md.append("- 접근방향 잔차 median {:+.2f} m (+ = 목표를 지나침), 횡방향 |오차| median {:.2f} m, "
                  "오버슈트 비율 {:.0f}%".format(
                      ap["along_median"], ap["cross_median_abs"], ap["overshoot_share"] * 100))
    md.append("")

    pc = r.get("per_category") or {}
    if len(pc) > 1:
        md.append("## 목표 유형별")
        md.append("")
        md += _table(
            ["유형", "n", "비중", "주 지표", "p90", "heading median", "성공률(엄격)", "도달후이탈"],
            [[name, v["n"], "{:.0f}%".format(v["share"] * 100),
              ("path_lag {:.1f} cm".format(v["path_lag_median"] * 100)
               if v.get("metric_kind") == "path_tracking"
               else "{:.1f} cm".format(v["pos_median"] * 100)),
              ("{:.1f} cm".format(v["path_lag_p90"] * 100)
               if v.get("metric_kind") == "path_tracking"
               else "{:.1f} cm".format(v["pos_p90"] * 100)),
              "{:.1f}°".format(v["heading_median"]),
              ("—" if v.get("success_rate_strict") is None
               else "{:.0f}%".format(v["success_rate_strict"] * 100)),
              ("—" if v.get("arrived_then_left_share") is None
               else "{:.0f}%".format(v["arrived_then_left_share"] * 100))]
             for name, v in sorted(pc.items(), key=lambda kv: -kv[1]["pos_median"])])
        md.append("")

    pd_rows = r.get("per_start_distance") or []
    if pd_rows:
        md.append("## 초기 목표거리별")
        md.append("")
        md += _table(
            ["거리 구간", "n", "위치 median", "위치 p90", "heading median", "미도달", "도달후이탈"],
            [["{:.2f}–{} m".format(b["lo"], "∞" if b["hi"] == float("inf") else "{:.2f}".format(b["hi"])),
              b["n"], "{:.1f} cm".format(b["pos_median"] * 100), "{:.1f} cm".format(b["pos_p90"] * 100),
              "{:.1f}°".format(b["heading_median"]),
              "{:.0f}%".format(b["never_arrived_share"] * 100),
              "{:.0f}%".format(b["arrived_then_left_share"] * 100)]
             for b in pd_rows])
        md.append("")

    fa = r.get("fall_analysis")
    if fa:
        md.append("## 낙상 분석")
        md.append("")
        md.append("- 유형별: {}".format(", ".join("{} {}회".format(k, v) for k, v in fa["per_category"].items())))
        md.append("- 구간 시작 후 median {:.1f}s 시점, 해당 구간 초기 목표거리 median {:.2f} m".format(
            fa["t_into_segment_median_s"], fa["start_dist_median_m"]))
        md.append("")

    md.append("## 부가 지표")
    md.append("")
    md.append("- 도착 시 속도 median {:.2f} m/s (정지 기준 {:.2f} m/s)".format(
        r["final_speed_mps"]["median"], r["stop_speed_threshold"]))
    md.append("- 성공률(엄격: {:.0f}cm+{:.0f}°+정지): {:.1f}%".format(
        g["pos_median"]["limit"] * 100, g["heading_median"]["limit"], r["success_rate_strict"] * 100))
    md.append("- 성공률(완화: {:.0f}cm+{:.0f}°): {:.1f}%".format(
        g["pos_p90"]["limit"] * 100, g["heading_median"]["limit"], r["success_rate_loose"] * 100))
    md.append("")
    md.append("## 다음에 확인/시도할 것")
    md.append("")
    for line in r["recommendations"]:
        md.append("- {}".format(line))
    if not r["instrumented"]:
        md.append("")
        md.append("> 참고: 이 env는 구간 메타데이터(goal_category/goal_start_pos)를 노출하지 않아 "
                  "유형별·거리별·실현가능성 분석이 생략되었다.")
    return "\n".join(md)


def _hist_pct(hist, hist_max, p):
    if hist.sum() == 0:
        return float("nan")
    edges = np.arange(len(hist)) * (hist_max / len(hist))
    cdf = np.cumsum(hist) / hist.sum()
    idx = int(np.searchsorted(cdf, p / 100.0))
    return float(edges[min(idx, len(edges) - 1)])


def _hist_pct_range(hist, lo, hi, p):
    if hist.sum() == 0:
        return float("nan")
    edges = lo + np.arange(len(hist)) * ((hi - lo) / len(hist))
    cdf = np.cumsum(hist) / hist.sum()
    idx = int(np.searchsorted(cdf, p / 100.0))
    return float(edges[min(idx, len(edges) - 1)])


def summarize_stress(roll, cfg, num_envs, duration_s, dt, checkpoint, config_path,
                     task, mode, perturbations, setup_wall_s, seed,
                     task_state_protocol="config_fixed"):
    """Stress runs are scored on survival and oscillation, not on the gates.

    Under 50 Hz goal jitter the goal's expectation IS the robot's neighbourhood,
    so "position error" measures the sampler, not the policy. What still carries
    information is whether the robot stays upright and whether it shakes itself
    apart trying to chase a target that keeps moving.
    """
    sh, sm = roll["speed_hist"], roll["speed_hist_max"]
    ah, am = roll["angvel_hist"], roll["angvel_hist_max"]
    env_min = roll["env_minutes"]
    falls = roll["falls"]
    results = {
        "task": task, "config": config_path, "checkpoint": checkpoint,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "stress:" + mode,
        "authoritative_gate_evaluation": False,
        "task_state_protocol": task_state_protocol,
        "env_code_sha": env_code_sha(),
        "evaluation_protocol_sha": evaluation_protocol_sha(),
        "force_visualization_probe": bool(
            cfg.get("evaluation", {}).get("force_visualization_probe", False)),
        "hbatch_protocol_version": cfg.get("evaluation", {}).get(
            "hbatch_protocol_version"),
        "effective_eval_protocol_sha": cfg.get("evaluation", {}).get(
            "effective_eval_protocol_sha"),
        "effective_disturbance_protocol": copy.deepcopy(
            cfg.get("evaluation", {}).get(
                "effective_disturbance_protocol") or {}),
        "joint_dr_probe": copy.deepcopy(
            cfg.get("evaluation", {}).get("joint_dr_probe") or {}),
        "num_envs": num_envs, "duration_s": duration_s, "seed": seed,
        "perturbations": bool(perturbations),
        "env_minutes": env_min,
        "falls": falls,
        "falls_per_env_minute": falls / max(env_min, 1e-9),
        "upright_share": roll["upright_share"],
        "body_speed": {"median": _hist_pct(sh, sm, 50), "p90": _hist_pct(sh, sm, 90),
                       "p99": _hist_pct(sh, sm, 99)},
        "body_angvel": {"median": _hist_pct(ah, am, 50), "p90": _hist_pct(ah, am, 90),
                        "p99": _hist_pct(ah, am, 99)},
        "disturbance_eval": roll.get("disturbance_eval") or {},
        "timing": {"setup_wall_s": setup_wall_s, "rollout_wall_s": roll["rollout_wall_s"]},
    }

    md = ["# GoalPose 강건성 스트레스 리포트 ({}) — {}".format(mode, results["date"]), ""]
    md.append("- checkpoint: `{}`".format(checkpoint))
    md.append("- config: `{}`".format(config_path))
    md.append("- task-state protocol: `{}`".format(task_state_protocol))
    if results.get("effective_eval_protocol_sha"):
        md.append("- effective eval protocol: `{}`".format(
            results["effective_eval_protocol_sha"]))
    if results.get("joint_dr_probe", {}).get("active"):
        md.append("- joint-DR probe (rad): encoder ±{}, target ±{}, init σ{}".format(
            results["joint_dr_probe"].get("joint_encoder_bias_rad"),
            results["joint_dr_probe"].get("joint_target_offset_rad"),
            results["joint_dr_probe"].get("init_dof_std_rad")))
    md.append("- 조건: {} envs × {:.0f}s, 목표를 매 제어스텝(50 Hz) ±3 m 균일 재추첨, 외란 {}".format(
        num_envs, duration_s, "ON" if perturbations else "OFF"))
    md.append("- 누적 관측 시간: {:.0f} env·분".format(env_min))
    md.append("")
    de = results["disturbance_eval"]
    if perturbations:
        md.append("- 외력 event {}회, active env-step {:.2f}%, 외력 중 낙상 {}회".format(
            de.get("events", 0), 100 * de.get("active_share", 0.0),
            de.get("falls_during_force", 0)))
        overall = de.get("overall") or {}
        if overall.get("records", 0):
            md.append("- outcome {}건: 2 s/5 s 생존 {:.1f}/{:.1f}%, max tilt p90 {:.1f}°, "
                      "90% 속도회복≤5 s {:.1f}%".format(
                          overall["records"], 100 * overall["survival_2s"],
                          100 * overall["survival_5s"], overall["max_tilt_deg_p90"],
                          100 * overall["recovery_90_within_5s_share"]))
        if de.get("events", 0) == 0:
            md.append("- ❌ 외력을 켰지만 event가 0회다. 이 결과를 combined robustness로 채택하지 않는다.")
        if de.get("video_requested", False):
            recorded = int(de.get("video_recorded_frames", 0))
            active = int(de.get("video_force_active_frames", 0))
            drawn = int(de.get("video_force_arrow_drawn_frames", 0))
            md.append("- 영상 env0 외력 물리 frame {} / {} ({:.1f}%); "
                      "renderer-confirmed 빨간 화살표 {} frame".format(
                          active, recorded,
                          100 * de.get("video_force_active_share", 0.0), drawn))
        md.append("")
    md.append("> ⚠️ **이 리포트에는 위치오차 게이트가 없다.** 목표가 50 Hz로 무작위 재추첨되면 "
              "참값 목표의 기댓값이 로봇 주변이 되어 위치오차는 정책이 아니라 샘플러를 측정한다. "
              "여기서 의미가 있는 것은 **넘어지지 않는가**와 **발산하지 않는가**뿐이다.")
    md.append("")
    md.append("## 생존")
    md.append("")
    md += _table(["지표", "값"], [
        ["낙상", "{}회".format(falls)],
        ["낙상률", "{:.2f} 회/env·분".format(results["falls_per_env_minute"])],
        ["직립 유지 시간 비율", "{:.1f}%".format(results["upright_share"] * 100)],
    ])
    md.append("")
    md.append("## 발산 여부 (몸통 각속도)")
    md.append("")
    md.append("목표를 쫓아 몸통이 경련하면 |ω|가 커진다. 정상 보행의 |ω|는 대략 1~2 rad/s다.")
    md.append("")
    md += _table(["지표", "각속도 |ω|", "선속도 |v|"], [
        ["median", "{:.2f} rad/s".format(results["body_angvel"]["median"]),
         "{:.2f} m/s".format(results["body_speed"]["median"])],
        ["p90", "{:.2f} rad/s".format(results["body_angvel"]["p90"]),
         "{:.2f} m/s".format(results["body_speed"]["p90"])],
        ["p99", "{:.2f} rad/s".format(results["body_angvel"]["p99"]),
         "{:.2f} m/s".format(results["body_speed"]["p99"])],
    ])
    md.append("")
    md.append("## 판정")
    md.append("")
    verdict = []
    if results["falls_per_env_minute"] > 0.5:
        verdict.append("❌ **낙상률 {:.2f}/env·분** — 목표 흔들림에 무너진다. 학습 쪽 "
                       "`noise.goal_bt_flicker.prob_per_step`을 올려 더 많이 노출시킬 것."
                       .format(results["falls_per_env_minute"]))
    else:
        verdict.append("✅ 낙상률 {:.2f}/env·분 — 목표 흔들림에서도 서 있다."
                       .format(results["falls_per_env_minute"]))
    if results["body_angvel"]["p90"] > 3.0:
        verdict.append("❌ **각속도 p90 {:.2f} rad/s** — 흔들리는 목표를 쫓고 있다(경련). "
                       "정책이 필터링을 배우지 못했다는 뜻이므로, 보상이 참값 목표를 읽고 "
                       "있는지 확인하고 `action_rate` 페널티를 키울 것."
                       .format(results["body_angvel"]["p90"]))
    else:
        verdict.append("✅ 각속도 p90 {:.2f} rad/s — 목표 흔들림을 쫓지 않고 걸러낸다."
                       .format(results["body_angvel"]["p90"]))
    md += ["- " + v for v in verdict]
    md.append("")
    return results, "\n".join(md)


def write_outputs(out_dir, results, roll, report_md, env=None, cfg=None):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    with open(os.path.join(out_dir, "segments.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pos_err_m", "heading_err_deg", "final_speed_mps", "category",
                    "start_dist_m", "duration_s", "min_dist_m", "along_err_m", "cross_err_m",
                    "peak_speed_mps", "mean_speed_mps",
                    "time_above_1p0_s", "time_above_1p3_s", "cruise_1p3_s",
                    "cmd_speed_mps", "path_lag_m",
                    "time_to_0p5_s", "time_to_0p8_s", "time_to_1p0_s",
                    "direction_time_to_0p5_s", "direction_time_to_0p8_s",
                    "direction_time_to_1p0_s", "min_speed_first_2s",
                    "initial_speed_mps", "initial_goal_bearing_rad", "switch_gait_phase"])
        rows = zip(roll["pos_err"], np.degrees(roll["head_err"]), roll["speed"],
                   roll["category"], roll["start_dist"], roll["duration_s"],
                   roll["min_dist"], roll["along"], roll["cross"],
                   roll["peak_speed"], roll["mean_speed"],
                   roll["time_above_1p0"], roll["time_above_1p3"], roll["cruise_1p3"],
                   roll["cmd_speed"],
                   roll["path_lag"], roll["time_to_0p5_s"], roll["time_to_0p8_s"],
                   roll["time_to_1p0_s"], roll["direction_time_to_0p5_s"],
                   roll["direction_time_to_0p8_s"], roll["direction_time_to_1p0_s"],
                   roll["min_speed_first_2s"], roll["initial_speed_mps"],
                   roll["initial_goal_bearing_rad"], roll["switch_gait_phase"])
        for row in rows:
            w.writerow([CATEGORY_NAMES.get(int(c), c) if i == 3 else "{:.4f}".format(c)
                        for i, c in enumerate(row)])
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report_md + "\n")

    if env is not None and cfg is not None and not (hasattr(env, "camera_frames") and len(env.camera_frames) > 0):
        # The 2026-07-27 v7 batch produced no mp4 for any of its four runs and
        # nothing said why. Pin down which layer dropped it: Isaac Gym silently
        # disables graphics when headless unless viewer.record_video was already
        # true at sim-creation time, and create_camera_sensor can also fail on a
        # headless box with several processes contending for the GPU.
        print("!!! VIDEO: no frames captured.")
        print("    graphics_device_id = {} (-1 means Isaac Gym created the sim with NO "
              "graphics; viewer.record_video must be true BEFORE build_env)".format(
                  getattr(env, "graphics_device_id", "?")))
        print("    viewer.record_video (now) = {}".format(cfg.get("viewer", {}).get("record_video")))
        print("    env.camera = {}".format(getattr(env, "camera", "<unset>")))
        print("    env.camera_frames = {}".format(
            len(env.camera_frames) if hasattr(env, "camera_frames") else "<attribute never created>"))

    if env is not None and cfg is not None and hasattr(env, "camera_frames") and len(env.camera_frames) > 0:
        import imageio
        radius = cfg["rewards"].get("constellation_radius", 1.0)
        video_path = os.path.join(out_dir, "rollout_env0.mp4")
        perspective = bool(cfg.get("evaluation", {}).get("perspective_overlays", False))
        path_trace = []
        last_path_segment = None
        camera_poses = getattr(env, "camera_poses", [])
        fov = float(getattr(env, "camera_horizontal_fov", 75.0))
        with imageio.get_writer(video_path, fps=int(1.0 / env.dt)) as writer:
            for i, (frame, st) in enumerate(zip(env.camera_frames, roll["overlay_states"])):
                if len(st) >= 16 and int(st[15]) == CATEGORY_PATH:
                    segment = int(st[16]) if len(st) >= 17 else None
                    if (last_path_segment is not None and segment is not None
                            and segment != last_path_segment):
                        path_trace = []
                    path_trace.append(np.asarray(st[2]).copy())
                    last_path_segment = segment
                elif path_trace:
                    path_trace = []
                    last_path_segment = None
                if perspective and i < len(camera_poses):
                    f = draw_perspective_scene(frame.copy(), st, camera_poses[i], fov, path_trace)
                else:
                    f = draw_constellation_inset(frame.copy(), st[0], st[1], st[2], st[3], radius)
                if len(st) >= 11:
                    f = draw_telemetry_hud(f, st)
                if not perspective and len(st) >= 13 and st[11] is not None:
                    f = draw_goal_sequence(f, st[0], st[1], st[11], st[12], radius and 240 or 240)
                if f.ndim == 3 and f.shape[2] == 4:
                    f = f[..., :3]
                writer.append_data(f)
        print("video written ({} + velocity/disturbance HUD): {}".format(
            "perspective path/goal/force overlays" if perspective else "constellation inset", video_path))


def _artifact_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_provenance(path):
    """Best-effort immutable identity for an evaluation input file."""
    if not path:
        return {"path": path, "sha256": None}
    resolved = os.path.abspath(os.path.expanduser(os.fspath(path)))
    try:
        digest = _artifact_sha256(resolved) if os.path.isfile(resolved) else None
    except OSError:
        digest = None
    return {"path": resolved, "sha256": digest}


def _attach_input_provenance(results, checkpoint, source_config):
    """Add optional fields without changing or removing the legacy schema."""
    results["input_provenance"] = {
        "checkpoint": _file_provenance(checkpoint),
        "source_config": _file_provenance(source_config),
    }
    return results


def write_eval_completion_marker(out_dir, completion_token=None, include_video=False):
    """Atomically attest that every generated artifact was closed completely.

    Isaac Gym's native camera teardown can signal after Python has finished the
    evaluation.  A caller may distinguish that teardown-only failure from an
    interrupted write only when this marker, hashes and a decodable video all
    agree.  The marker is deliberately the final filesystem mutation in main.
    """
    artifacts = {}
    names = ["report.json", "report.md", "segments.csv"]
    if include_video:
        names.append("rollout_env0.mp4")
    for name in names:
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            artifacts[name] = {
                "bytes": os.path.getsize(path),
                "sha256": _artifact_sha256(path),
            }
    if "report.json" not in artifacts:
        raise RuntimeError("cannot mark incomplete eval without report.json")
    marker = {
        "version": 1,
        "status": "complete",
        "completion_token": completion_token,
        "artifacts": artifacts,
    }
    target = os.path.join(out_dir, "eval-complete-codex.json")
    temporary = target + ".tmp-{}".format(os.getpid())
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(marker, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, target)


# --------------------------------------------------------------------------

def main():
    eval_started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Evaluate a GoalPose checkpoint against the MASTERPLAN gates.")
    parser.add_argument("--task", default="K1/Goal_Pose")
    parser.add_argument("--config", help="evaluation yaml (default: envs/<task>.yaml); use a run's config.yaml for native-dynamics preview")
    parser.add_argument("--checkpoint", default="-1", help=".pth path, or -1 for the latest under logs/")
    parser.add_argument(
        "--restore_task_state", action="store_true",
        help="restore checkpoint curriculum/grid state for a native-resume diagnostic; "
             "default evaluation keeps the config-defined protocol fixed")
    parser.add_argument("--num_envs", type=int, help="override evaluation.num_envs from the yaml")
    parser.add_argument("--duration_s", type=float, help="override evaluation.duration_s from the yaml")
    parser.add_argument("--sim_device", help="override yaml sim_device")
    parser.add_argument("--rl_device", help="override yaml rl_device")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of the deterministic mean")
    parser.add_argument("--keep_perturbations", action="store_true", help="keep random kicks/pushes on during eval")
    parser.add_argument("--force_profile", choices=["heldout"], default=None,
                        help="replace the arm's disturbance with ONE held-out profile "
                             "(interval 4-8 s, collision 50-120 N, support 4-10 N), the "
                             "same for every arm, so robustness is compared on a common "
                             "test rather than on each policy's own training distribution")
    parser.add_argument("--terrain", choices=["as_trained", "plane"], default=None,
                        help="ground the eval runs on. Default: whatever the arm "
                             "trained with -- which means a terrain arm is scored on "
                             "rough ground and a flat arm on a plane, and their "
                             "numbers are NOT comparable. Use plane to put every arm "
                             "on the same ground.")
    parser.add_argument(
        "--no_noise", action="store_true",
        help="disable observation noise only; joint encoder/target/init DR, "
             "including --joint_* probes, remains active")
    parser.add_argument(
        "--joint_encoder_bias_rad", type=_nonnegative_finite_float,
        help="held-out encoder-bias half-width in rad: uniform [-v,+v], "
             "applied after the HBatch common eval override; >=0")
    parser.add_argument(
        "--joint_target_offset_rad", type=_nonnegative_finite_float,
        help="held-out motor-target offset half-width in rad: uniform [-v,+v], "
             "applied after the HBatch common eval override; >=0")
    parser.add_argument(
        "--init_dof_std_rad", type=_nonnegative_finite_float,
        help="held-out initial joint-position Gaussian stddev in rad, applied "
             "after the HBatch common eval override; >=0")
    parser.add_argument("--record_video", action="store_true", help="also record an mp4 of env 0 (first --record_video_s seconds)")
    parser.add_argument("--record_video_s", type=float, default=8.0)
    parser.add_argument(
        "--force_visualization_probe", action="store_true",
        help="video-only HBatch protocol: path-only goals plus a guaranteed early support force")
    parser.add_argument("--feasible_speed", type=float, help="override evaluation.feasible_speed_mps (m/s) used by the feasibility check")
    parser.add_argument("--stress", choices=["jitter"], help="robustness stress mode instead of a gate evaluation. 'jitter': the true goal is redrawn uniformly in a +-3 m box every control step (50 Hz), modelling BT thrash / ball re-detection. Scored on falls and body oscillation -- position error is undefined in this mode.")
    parser.add_argument("--goal_pattern", choices=["lateral", "reverse", "forward_hold"],
                        help="force abrupt robot-local side or rear waypoint goals")
    parser.add_argument("--exploratory", action="store_true", help="label this run as a non-authoritative preview rather than an official gate evaluation")
    parser.add_argument("--out", help="output dir (default: <run_dir>/eval/<timestamp>)")
    parser.add_argument(
        "--completion_token",
        help="caller-generated nonce copied into the atomic completion marker")
    args = parser.parse_args()

    # A reused --out directory must never retain a completion attestation from
    # an older run if this process fails before producing fresh artifacts.
    if args.out:
        try:
            os.unlink(os.path.join(args.out, "eval-complete-codex.json"))
        except FileNotFoundError:
            pass

    config_path = args.config or os.path.join("envs", "{}.yaml".format(args.task))
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    eval_cfg = cfg.setdefault("evaluation", {})
    if args.feasible_speed is not None:
        eval_cfg["feasible_speed_mps"] = args.feasible_speed

    num_envs = args.num_envs or eval_cfg.get("num_envs", 256)
    duration_s = args.duration_s or eval_cfg.get("duration_s", 120.0)

    prepare_cfg(cfg, args.task, num_envs, args.sim_device, args.rl_device,
                record_video=args.record_video, keep_perturbations=args.keep_perturbations,
                force_profile=args.force_profile, terrain=args.terrain,
                no_noise=args.no_noise, stress=args.stress, goal_pattern=args.goal_pattern,
                force_visualization_probe=args.force_visualization_probe,
                joint_encoder_bias_rad=args.joint_encoder_bias_rad,
                joint_target_offset_rad=args.joint_target_offset_rad,
                init_dof_std_rad=args.init_dof_std_rad)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint = args.checkpoint
    if checkpoint in ("-1", -1, None, ""):
        checkpoint = find_latest_checkpoint(args.task)
        if checkpoint is None:
            raise FileNotFoundError("no .pth checkpoint found under logs/")
    print("Evaluating checkpoint: {}".format(checkpoint))

    # Loud warning if the env code moved since this checkpoint was trained.
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint)))
    stamp = os.path.join(run_dir, "ENV_CODE_SHA")
    if os.path.exists(stamp):
        trained_sha = open(stamp, encoding="utf-8").read().strip()
        now_sha = env_code_sha()
        if trained_sha and now_sha and trained_sha != now_sha:
            print("!" * 70)
            print("WARNING: envs/ has changed since this checkpoint was trained.")
            print("  trained under: {}".format(trained_sha[:12]))
            print("  evaluating as: {}".format(now_sha[:12]))
            print("  If the change altered what the TASK means (goal semantics,")
            print("  reward structure, reaching conditions), these numbers score the")
            print("  policy against a task it never saw. That is what invalidated")
            print("  E1's re-evaluation on 2026-07-27: path 24.8 -> 165.9 cm.")
            print("!" * 70)

    env = build_env(cfg, args.task)
    device = cfg["basic"]["rl_device"]
    model = load_policy(
        checkpoint, env, device, restore_task_state=args.restore_task_state)

    setup_wall_s = time.perf_counter() - eval_started
    roll = rollout(env, model, int(duration_s / env.dt), device,
                   stochastic=args.stochastic, record_video=args.record_video,
                   record_video_s=args.record_video_s, stress=args.stress,
                   cfg_speed_window_s=cfg.get("evaluation", {}).get("speed_window_s", 0.2))

    if args.stress:
        results, report_md = summarize_stress(
            roll, cfg, num_envs, duration_s, env.dt, checkpoint, config_path,
            args.task, args.stress, args.keep_perturbations, setup_wall_s, args.seed,
            task_state_protocol=getattr(
                env, "eval_task_state_protocol", "config_fixed"))
        _attach_input_provenance(results, checkpoint, config_path)
        out_dir = args.out
        if not out_dir:
            run_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint)))
            out_dir = os.path.join(run_dir, "eval",
                                   time.strftime("%Y-%m-%d-%H-%M-%S") + "_stress_" + args.stress)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=float)
        with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write(report_md + "\n")
        video_ready = (args.record_video and hasattr(env, "camera_frames")
                       and len(env.camera_frames) > 0)
        if video_ready:
            write_outputs(out_dir, results, roll, report_md, env=env, cfg=cfg)
        write_eval_completion_marker(
            out_dir, args.completion_token, include_video=video_ready)
        print(report_md)
        print("\nwritten: {}".format(out_dir))
        return

    results = summarize(
        roll, cfg, num_envs, duration_s, env.dt, checkpoint, config_path, args.task,
        deterministic=not args.stochastic, perturbations=bool(args.keep_perturbations),
        obs_noise=not args.no_noise,
        exploratory=(args.exploratory or args.force_visualization_probe),
        setup_wall_s=setup_wall_s, seed=args.seed,
        task_state_protocol=getattr(
            env, "eval_task_state_protocol", "config_fixed"))
    _attach_input_provenance(results, checkpoint, config_path)
    report_md = render_report(results)

    out_dir = args.out
    if not out_dir:
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint)))
        out_dir = os.path.join(run_dir, "eval", time.strftime("%Y-%m-%d-%H-%M-%S"))
    write_outputs(out_dir, results, roll, report_md, env=env if args.record_video else None, cfg=cfg)
    video_ready = (args.record_video and hasattr(env, "camera_frames")
                   and len(env.camera_frames) > 0)
    write_eval_completion_marker(
        out_dir, args.completion_token, include_video=video_ready)

    print("")
    print(report_md)
    print("")
    print("report saved to: {}".format(out_dir))


if __name__ == "__main__":
    main()
