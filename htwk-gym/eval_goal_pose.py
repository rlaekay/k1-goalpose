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
import json
import glob
import time
import random
import argparse

import isaacgym  # noqa: F401  (must be imported before torch)
from envs import *  # noqa: F401,F403  (registers task classes)

import torch
import numpy as np
import yaml

from isaacgym.torch_utils import get_euler_xyz
from utils.model import ActorCritic
from utils.runner import get_task_class


# goal_categories mixture in envs/K1/*.yaml, as recorded by GoalPose._resample_goals
CATEGORY_NAMES = {-1: "uniform", 0: "stand", 1: "straight", 2: "lateral", 3: "turn", 4: "combined",
                  5: "path"}

# start-distance bins [m] for the per-distance breakdown
DISTANCE_BINS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, float("inf")]


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
                    img[py:py + scale, px:px + scale] = color
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

def prepare_cfg(cfg, task, num_envs, sim_device=None, rl_device=None,
                record_video=False, keep_perturbations=False, no_noise=False):
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
    if not keep_perturbations:
        cfg["randomization"]["kick_interval_s"] = 1.0e9
        cfg["randomization"]["push_interval_s"] = 1.0e9
    if no_noise:
        cfg["noise"] = {}
    return cfg


def build_env(cfg, task):
    task_class = get_task_class(task.split("/")[-1])
    if task_class is None:
        raise ValueError("unknown task: {}".format(task))
    return task_class(cfg)


def load_policy(checkpoint, env, device, model=None, verbose=True):
    """Load checkpoint weights, reusing `model` if given (so a sweep over many
    checkpoints does not rebuild the network each time)."""
    if model is None:
        model = ActorCritic(env.num_actions, env.num_obs, env.num_privileged_obs).to(device)
    model_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    load_result = model.load_state_dict(model_dict["model"], strict=False)
    if verbose and (load_result.missing_keys or load_result.unexpected_keys):
        print("WARNING: partial checkpoint load ({} missing, {} unexpected keys)".format(
            len(load_result.missing_keys), len(load_result.unexpected_keys)))
    model.eval()
    return model


# --------------------------------------------------------------------------
# rollout
# --------------------------------------------------------------------------

def rollout(env, model, total_steps, device, stochastic=False, record_video=False,
            record_video_s=8.0, progress_every=500, progress_prefix="  "):
    """Roll the policy and collect one record per completed goal segment.

    Per segment we keep not just the final error but the provenance needed to
    diagnose it: which goal category it was, how far away the goal was sampled,
    how long the policy had, the closest it ever got, and whether the residual
    error is along or across the approach direction.
    """
    instrumented = hasattr(env, "goal_start_pos") and hasattr(env, "goal_start_step")

    keys = ("pos_err", "head_err", "speed", "category", "start_dist", "duration_s",
            "min_dist", "along", "cross", "peak_speed", "mean_speed")
    seg = {k: [] for k in keys}
    # Whole-rollout body-speed histogram, over every env and every control step.
    # The per-segment "final_speed" only says how well it stops; this says how
    # fast it can actually go, which is the number the MASTERPLAN target is in.
    speed_hist = np.zeros(400, dtype=np.int64)  # 0..4 m/s in 1 cm/s bins
    speed_hist_max = 4.0
    fall_ctx = {k: [] for k in ("category", "goal_dist", "t_into_segment", "start_dist")}
    falls = 0
    censored = 0
    video_done = not record_video
    overlay_states = []

    # closest approach to the goal currently being pursued, per env
    min_dist = torch.full((env.num_envs,), float("inf"), device=env.device)
    # peak and accumulated body speed within the segment currently in progress
    peak_speed = torch.zeros(env.num_envs, device=env.device)
    sum_speed = torch.zeros(env.num_envs, device=env.device)
    n_speed = torch.zeros(env.num_envs, device=env.device)

    obs, _ = env.reset()
    obs = obs.to(device)

    # CUDA work is asynchronous. Synchronize only at the timing boundaries so
    # rollout_wall_s measures completed simulation/inference work without adding
    # a device-wide barrier to every control step.
    synchronize_cuda_devices(env.cfg["basic"]["sim_device"], device)
    started = time.perf_counter()

    for step_i in range(total_steps):
        # goal_dist still refers to the segment in progress; fold it in before the
        # step can replace the goal.
        torch.minimum(min_dist, env.goal_dist, out=min_dist)
        cur_speed = torch.norm(env.root_states[:, 7:9], dim=-1)
        torch.maximum(peak_speed, cur_speed, out=peak_speed)
        sum_speed += cur_speed
        n_speed += 1.0
        np.add.at(speed_hist,
                  np.clip((cur_speed.cpu().numpy() / speed_hist_max * len(speed_hist)).astype(int),
                          0, len(speed_hist) - 1), 1)

        # Everything a terminated env needs must be snapshotted here: _reset_idx()
        # runs inside step() and zeroes base_pos/episode_length_buf for fallen envs,
        # so post-step reads would describe the fresh episode, not the failure.
        prev_goal_pos = env.goal_pos_world.clone()
        prev_goal_heading = env.goal_heading_world.clone()
        prev_goal_dist = env.goal_dist.clone()
        prev_len = env.episode_length_buf.clone()
        if instrumented:
            prev_category = env.goal_category.clone()
            prev_start_pos = env.goal_start_pos.clone()
            prev_start_step = env.goal_start_step.clone()

        with torch.no_grad():
            dist = model.act(obs)
            act = dist.sample() if stochastic else dist.loc
        obs, _, done, infos = env.step(act.to(env.device))
        obs = obs.to(device)

        timeouts = infos["time_outs"].to(done.device)
        fell = done & ~timeouts
        n_fell = int(fell.sum().item())
        falls += n_fell
        censored += int((done & timeouts).sum().item())

        if n_fell and instrumented:
            fids = fell.nonzero(as_tuple=False).flatten()
            elapsed = (prev_len[fids] + 1 - prev_start_step[fids]).clamp(min=0).float() * env.dt
            fall_ctx["category"].extend(prev_category[fids].cpu().tolist())
            fall_ctx["goal_dist"].extend(prev_goal_dist[fids].cpu().tolist())
            fall_ctx["t_into_segment"].extend(elapsed.cpu().tolist())
            fall_ctx["start_dist"].extend(
                torch.norm(prev_goal_pos[fids] - prev_start_pos[fids], dim=-1).cpu().tolist())

        changed = (env.goal_pos_world != prev_goal_pos).any(dim=1) | (env.goal_heading_world != prev_goal_heading)
        completed = changed & ~done
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

        if not video_done:
            _, _, yaw_all = get_euler_xyz(env.base_quat[0:1])
            push_n = push_nm = 0.0
            if hasattr(env, "pushing_forces") and hasattr(env, "base_indice"):
                push_n = float(torch.norm(env.pushing_forces[0, env.base_indice, :]).item())
                push_nm = float(torch.norm(env.pushing_torques[0, env.base_indice, :]).item())
            overlay_states.append((
                env.base_pos[0, :2].cpu().numpy().copy(),
                float(wrap(yaw_all)[0].item()),
                env.goal_pos_world[0].cpu().numpy().copy(),
                float(env.goal_heading_world[0].item()),
                float(env.base_lin_vel[0, 0].item()),
                float(env.base_lin_vel[0, 1].item()),
                float(env.base_ang_vel[0, 2].item()),
                float(env.goal_dist[0].item()),
                float(np.degrees(env.heading_error[0].item())),
                push_n,
                push_nm,
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

    out = {k: np.asarray(v, dtype=float) for k, v in seg.items()}
    out["category"] = out["category"].astype(int)
    out["falls"] = falls
    out["censored"] = censored
    out["fall_ctx"] = {k: np.asarray(v, dtype=float) for k, v in fall_ctx.items()}
    out["rollout_wall_s"] = wall_s
    out["total_steps"] = total_steps
    out["instrumented"] = instrumented
    out["overlay_states"] = overlay_states
    out["speed_hist"] = speed_hist
    out["speed_hist_max"] = speed_hist_max
    return out


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

def summarize(roll, cfg, num_envs, duration_s, dt, checkpoint, config_path, task,
              deterministic=True, perturbations=False, obs_noise=True,
              exploratory=False, setup_wall_s=0.0, seed=0):
    """Turn a rollout into gate verdicts plus the breakdowns needed to act on them."""
    eval_cfg = cfg.get("evaluation", {})
    gates = eval_cfg.get("gates", {})
    g_pos_med = gates.get("pos_median_m", 0.05)
    g_pos_p90 = gates.get("pos_p90_m", 0.10)
    g_head_med = gates.get("heading_median_deg", 10.0)
    g_falls = gates.get("max_falls", 0)
    feasible_speed = eval_cfg.get("feasible_speed_mps", 0.6)
    stop_thr = cfg["rewards"].get("stop_speed_threshold", 0.1)

    pos = roll["pos_err"]
    head_deg = np.degrees(roll["head_err"])
    speed = roll["speed"]
    n = len(pos)
    if n == 0:
        raise RuntimeError("no completed goal segments — duration too short or every env fell")
    falls = int(roll["falls"])
    attempts = n + falls

    pos_med, pos_p90 = _median(pos), _pct(pos, 90)
    head_med = _median(head_deg)

    ok_pos_loose = pos <= g_pos_p90
    ok_head = head_deg <= g_head_med
    ok_stop = speed <= stop_thr

    results = {
        "checkpoint": checkpoint,
        "config": config_path,
        "task": task,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "num_envs": num_envs,
        "duration_s": duration_s,
        "deterministic": deterministic,
        "perturbations": perturbations,
        "obs_noise": obs_noise,
        "authoritative_gate_evaluation": not exploratory,
        "segments_completed": n,
        "segments_censored_by_episode_end": roll["censored"],
        "falls": falls,
        "fall_rate_per_attempt": falls / attempts if attempts else 0.0,
        "pos_err_m": {"median": pos_med, "p90": pos_p90, "mean": float(np.mean(pos)), "max": float(np.max(pos))},
        "heading_err_deg": {"median": head_med, "p90": _pct(head_deg, 90),
                            "mean": float(np.mean(head_deg)), "max": float(np.max(head_deg))},
        "final_speed_mps": {"median": _median(speed), "p90": _pct(speed, 90), "mean": float(np.mean(speed))},
        "success_rate_strict": _frac((pos <= g_pos_med) & ok_head & ok_stop),
        "success_rate_loose": _frac(ok_pos_loose & ok_head),
        "ci95": {
            "pos_median": bootstrap_ci(pos, 50.0, seed=seed),
            "pos_p90": bootstrap_ci(pos, 90.0, seed=seed),
            "heading_median": bootstrap_ci(head_deg, 50.0, seed=seed),
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
    mode = np.full(n, "", dtype=object)
    mode[:] = "never_arrived"
    mode[~ok_pos_loose & ever_arrived] = "arrived_then_left"
    mode[ok_pos_loose & ~ok_head] = "heading_only"
    mode[ok_pos_loose & ok_head & ~ok_stop] = "not_stopped"
    mode[ok_pos_loose & ok_head & ok_stop] = "ok"
    results["failure_modes"] = {
        name: {"count": int(np.sum(mode == name)), "share": float(np.mean(mode == name))}
        for name in ("ok", "not_stopped", "heading_only", "arrived_then_left", "never_arrived")
    }
    results["closest_approach_m"] = {"median": _median(min_d), "p90": _pct(min_d, 90)}

    # ---- along/cross-track split ----------------------------------------
    along, cross = roll["along"], roll["cross"]
    finite = np.isfinite(along)
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
    else:
        results["body_speed"] = None

    # ---- per goal category ----------------------------------------------
    cats = roll["category"]
    per_cat = {}
    for c in sorted(set(cats.tolist())):
        m = cats == c
        per_cat[CATEGORY_NAMES.get(c, str(c))] = {
            "n": int(np.sum(m)),
            "share": float(np.mean(m)),
            "pos_median": _median(pos[m]),
            "pos_p90": _pct(pos[m], 90),
            "heading_median": _median(head_deg[m]),
            "speed_median": _median(speed[m]),
            "success_rate_strict": _frac((pos[m] <= g_pos_med) & ok_head[m] & ok_stop[m]),
            "arrived_then_left_share": _frac(mode[m] == "arrived_then_left"),
            "never_arrived_share": _frac(mode[m] == "never_arrived"),
        }
    results["per_category"] = per_cat

    # ---- per start distance ---------------------------------------------
    per_dist = []
    if np.isfinite(start_d).any():
        for lo, hi in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:]):
            m = np.isfinite(start_d) & (start_d >= lo) & (start_d < hi)
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

    results["recommendations"] = recommend(results, cfg)
    return results


def gate_ratios(results):
    """Normalized distance-to-passing per gate (<=1 means passing). Used to rank
    checkpoints: the worst ratio is 'how far from passing everything' this is."""
    g = results["gates"]
    fall_ref = max(results["segments_completed"], 1) * 0.002  # 0.2% of attempts as the soft reference
    return {
        "pos_median": g["pos_median"]["value"] / max(g["pos_median"]["limit"], 1e-9),
        "pos_p90": g["pos_p90"]["value"] / max(g["pos_p90"]["limit"], 1e-9),
        "heading_median": g["heading_median"]["value"] / max(g["heading_median"]["limit"], 1e-9),
        "falls": results["falls"] / max(fall_ref, 1e-9),
    }


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

    failing = {k: v["share"] for k, v in fm.items() if k != "ok"}
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
    ranked = [(k, v) for k, v in pc.items() if v["n"] >= 30 and np.isfinite(v["pos_median"])]
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
    md.append("- 조건: {} envs × {:.0f}s, {} 정책, 외란 {}, 관측노이즈 {}, seed {}".format(
        r["num_envs"], r["duration_s"],
        "결정론적" if r["deterministic"] else "확률적",
        "ON" if r["perturbations"] else "OFF",
        "ON" if r["obs_noise"] else "OFF", r["seed"]))
    t = r["timing"]
    md.append("- 벽시계: setup {:.1f}s + rollout {:.1f}s; env당 {:.1f}× real-time, 총 {:.0f} env·s/wall-s".format(
        t["setup_wall_s"], t["rollout_wall_s"], t["single_env_realtime_factor"],
        t["aggregate_env_seconds_per_wall_second"]))
    md.append("- 완료 구간 {}개 / 낙상 {}회 / 에피소드경계 절단 {}개".format(
        r["segments_completed"], r["falls"], r["segments_censored_by_episode_end"]))
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
        md.append("**종합: {}**".format("✅ 전체 게이트 통과" if r["all_gates_pass"] else "❌ 미통과 게이트 있음"))
    else:
        md.append("**탐색 결과: {} (표본/조건이 표준 프로토콜이 아니므로 공식 판정 아님)**".format(
            "모든 수치 기준 충족" if r["all_gates_pass"] else "미충족 수치 있음"))
    md.append("")

    bs = r.get("body_speed")
    if bs:
        md.append("## 몸통 속도 (body velocity)")
        md.append("")
        md.append("위의 오차 지표는 전부 '거리'다. 속도를 따로 보지 않으면 **느린 정책**과 "
                  "**빠르게 갈 이유가 없었던 정책**을 구분할 수 없다. 아래는 전 env·전 스텝의 "
                  "|v_xy| 분포다.")
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
    md.append("최종 오차가 같아도 원인이 다르면 처방이 반대다. `도달후이탈`은 멈추게 만들어야 하고, "
              "`미도달`은 더 가게 만들어야 한다.")
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
            ["유형", "n", "비중", "위치 median", "위치 p90", "heading median", "성공률(엄격)", "도달후이탈"],
            [[name, v["n"], "{:.0f}%".format(v["share"] * 100),
              "{:.1f} cm".format(v["pos_median"] * 100), "{:.1f} cm".format(v["pos_p90"] * 100),
              "{:.1f}°".format(v["heading_median"]), "{:.0f}%".format(v["success_rate_strict"] * 100),
              "{:.0f}%".format(v["arrived_then_left_share"] * 100)]
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


def write_outputs(out_dir, results, roll, report_md, env=None, cfg=None):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    with open(os.path.join(out_dir, "segments.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pos_err_m", "heading_err_deg", "final_speed_mps", "category",
                    "start_dist_m", "duration_s", "min_dist_m", "along_err_m", "cross_err_m",
                    "peak_speed_mps", "mean_speed_mps"])
        rows = zip(roll["pos_err"], np.degrees(roll["head_err"]), roll["speed"],
                   roll["category"], roll["start_dist"], roll["duration_s"],
                   roll["min_dist"], roll["along"], roll["cross"],
                   roll["peak_speed"], roll["mean_speed"])
        for row in rows:
            w.writerow([CATEGORY_NAMES.get(int(c), c) if i == 3 else "{:.4f}".format(c)
                        for i, c in enumerate(row)])
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report_md + "\n")

    if env is not None and cfg is not None and hasattr(env, "camera_frames") and len(env.camera_frames) > 0:
        import imageio
        radius = cfg["rewards"].get("constellation_radius", 1.0)
        video_path = os.path.join(out_dir, "rollout_env0.mp4")
        with imageio.get_writer(video_path, fps=int(1.0 / env.dt)) as writer:
            for frame, st in zip(env.camera_frames, roll["overlay_states"]):
                f = draw_constellation_inset(frame.copy(), st[0], st[1], st[2], st[3], radius)
                if len(st) >= 11:
                    f = draw_telemetry_hud(f, st)
                writer.append_data(f)
        print("video written (constellation inset + velocity/disturbance HUD): {}".format(video_path))


# --------------------------------------------------------------------------

def main():
    eval_started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Evaluate a GoalPose checkpoint against the MASTERPLAN gates.")
    parser.add_argument("--task", default="K1/Goal_Pose")
    parser.add_argument("--config", help="evaluation yaml (default: envs/<task>.yaml); use a run's config.yaml for native-dynamics preview")
    parser.add_argument("--checkpoint", default="-1", help=".pth path, or -1 for the latest under logs/")
    parser.add_argument("--num_envs", type=int, help="override evaluation.num_envs from the yaml")
    parser.add_argument("--duration_s", type=float, help="override evaluation.duration_s from the yaml")
    parser.add_argument("--sim_device", help="override yaml sim_device")
    parser.add_argument("--rl_device", help="override yaml rl_device")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of the deterministic mean")
    parser.add_argument("--keep_perturbations", action="store_true", help="keep random kicks/pushes on during eval")
    parser.add_argument("--no_noise", action="store_true", help="disable observation noise")
    parser.add_argument("--record_video", action="store_true", help="also record an mp4 of env 0 (first --record_video_s seconds)")
    parser.add_argument("--record_video_s", type=float, default=8.0)
    parser.add_argument("--feasible_speed", type=float, help="override evaluation.feasible_speed_mps (m/s) used by the feasibility check")
    parser.add_argument("--exploratory", action="store_true", help="label this run as a non-authoritative preview rather than an official gate evaluation")
    parser.add_argument("--out", help="output dir (default: <run_dir>/eval/<timestamp>)")
    args = parser.parse_args()

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
                no_noise=args.no_noise)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint = args.checkpoint
    if checkpoint in ("-1", -1, None, ""):
        checkpoint = find_latest_checkpoint(args.task)
        if checkpoint is None:
            raise FileNotFoundError("no .pth checkpoint found under logs/")
    print("Evaluating checkpoint: {}".format(checkpoint))

    env = build_env(cfg, args.task)
    device = cfg["basic"]["rl_device"]
    model = load_policy(checkpoint, env, device)

    setup_wall_s = time.perf_counter() - eval_started
    roll = rollout(env, model, int(duration_s / env.dt), device,
                   stochastic=args.stochastic, record_video=args.record_video,
                   record_video_s=args.record_video_s)

    results = summarize(
        roll, cfg, num_envs, duration_s, env.dt, checkpoint, config_path, args.task,
        deterministic=not args.stochastic, perturbations=bool(args.keep_perturbations),
        obs_noise=not args.no_noise, exploratory=args.exploratory,
        setup_wall_s=setup_wall_s, seed=args.seed)
    report_md = render_report(results)

    out_dir = args.out
    if not out_dir:
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint)))
        out_dir = os.path.join(run_dir, "eval", time.strftime("%Y-%m-%d-%H-%M-%S"))
    write_outputs(out_dir, results, roll, report_md, env=env if args.record_video else None, cfg=cfg)

    print("")
    print(report_md)
    print("")
    print("report saved to: {}".format(out_dir))


if __name__ == "__main__":
    main()
