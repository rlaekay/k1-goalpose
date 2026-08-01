"""HBatch: conservative G1-derived goal-pose experiments.

The class keeps GoalPoseV7's 54-observation/12-action interface and isolates the
new mechanisms requested after the E/G audit: episode-constant joint offsets,
multi-body disturbances, acceleration-preserving high-speed stabilization and
an H3-only touchdown placement hypothesis.  Every mechanism is config gated.
"""

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import (
    get_euler_xyz,
    quat_rotate,
    quat_rotate_inverse,
    torch_rand_float,
)

assert gymtorch

import math
import numpy as np
import torch

from envs.K1.goal_pose_v7 import GoalPoseV7


# These defaults are deliberately behind ``disturbance.scenario_aware.enabled``.
# The released H0--H3 configs do not contain that switch, so their sampling and
# random-number consumption stay unchanged.  They describe player-contact
# *wrench proxies*, not a simulated second robot collision.  In particular the
# old short 40--100 N frontal-collision class is not part of this distribution.
SCENARIO_AWARE_DEFAULTS = (
    {
        "name": "omni_shove",
        "weight": 0.50,
        "force_n": (15.0, 40.0),
        "duration_s": (0.25, 0.45),
        "twist_nm": (0.0, 0.0),
        "direction_mode": "uniform",
    },
    {
        # A source behind the robot pushes approximately along +robot-x.  The
        # force is still frozen in world coordinates once contact begins.
        "name": "rear_push",
        "weight": 0.30,
        "force_n": (15.0, 40.0),
        "duration_s": (0.25, 0.45),
        "twist_nm": (0.0, 0.0),
        "direction_mode": "rear_cone",
        "half_angle_deg": 22.5,
        "height_tiers": ("chest", "arm_proxy"),
    },
    {
        "name": "arm_entanglement",
        "weight": 0.20,
        "force_n": (6.0, 18.0),
        "duration_s": (0.30, 0.80),
        "twist_nm": (1.0, 4.0),
        "direction_mode": "uniform",
        "height_tiers": ("arm_proxy",),
    },
)


# Fixed arm links are collapsed into Trunk by the current asset import.  The
# upper two tiers therefore use Trunk plus an off-COM contact point as an arm /
# chest proxy.  Hip and shank tiers retain genuine loaded rigid bodies.  The
# offset is expressed in robot axes from the selected body's COM; ``mirror_y``
# samples the same arm envelope on both sides.
HEIGHT_TIER_DEFAULTS = (
    {
        "name": "shank",
        "weight": 0.05,
        "body_weights": {"Left_Shank": 0.5, "Right_Shank": 0.5},
        "offset_x_m": (-0.02, 0.02),
        "offset_y_m": (-0.015, 0.015),
        "offset_z_m": (-0.08, 0.08),
    },
    {
        "name": "hip",
        "weight": 0.05,
        "body_weights": {"Left_Hip_Roll": 0.5, "Right_Hip_Roll": 0.5},
        "offset_x_m": (-0.03, 0.03),
        "offset_y_m": (-0.03, 0.03),
        "offset_z_m": (-0.04, 0.04),
    },
    {
        "name": "chest",
        "weight": 0.30,
        "body_weights": {"Trunk": 1.0},
        "offset_x_m": (-0.04, 0.06),
        "offset_y_m": (-0.08, 0.08),
        "offset_z_m": (0.04, 0.14),
    },
    {
        "name": "arm_proxy",
        "weight": 0.60,
        "body_weights": {"Trunk": 1.0},
        "offset_x_m": (-0.04, 0.06),
        "offset_y_m": (0.10, 0.25),
        "offset_z_m": (0.14, 0.25),
        "mirror_y": True,
    },
)


def _merge_named_specs(defaults, overrides, label):
    """Return validated named weighted specs without touching any RNG.

    ``overrides`` may be a mapping keyed by name or a list of dictionaries.
    Existing named defaults are updated; a new name must provide a complete
    spec.  This helper intentionally uses only Python builtins so the schema can
    be unit-tested on a machine without Isaac Gym or PyTorch.
    """
    merged = {item["name"]: dict(item) for item in defaults}
    order = [item["name"] for item in defaults]
    if overrides:
        if isinstance(overrides, dict):
            items = []
            for name, value in overrides.items():
                if not isinstance(value, dict):
                    raise ValueError("{}.{} must be a mapping".format(label, name))
                item = dict(value)
                item.setdefault("name", name)
                items.append(item)
        elif isinstance(overrides, (list, tuple)):
            items = list(overrides)
        else:
            raise ValueError("{} must be a mapping or list".format(label))
        for item in items:
            if not isinstance(item, dict) or not item.get("name"):
                raise ValueError("each {} entry needs a name".format(label))
            name = item["name"]
            if name not in merged:
                merged[name] = {}
                order.append(name)
            merged[name].update(item)

    specs = [merged[name] for name in order if float(merged[name].get("weight", 0.0)) > 0.0]
    if not specs:
        raise ValueError("{} has no positive-weight entries".format(label))
    total = sum(float(item["weight"]) for item in specs)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("{} weights must have a finite positive sum".format(label))
    for item in specs:
        item["weight"] = float(item["weight"]) / total
    return specs


def _finite_range(value, label, nonnegative=False):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("{} must be [low, high]".format(label))
    lo, hi = float(value[0]), float(value[1])
    if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
        raise ValueError("{} must be a finite ordered range".format(label))
    if nonnegative and lo < 0.0:
        raise ValueError("{} must be nonnegative".format(label))
    return lo, hi


class GoalPoseHBatch(GoalPoseV7):

    def _init_buffers(self):
        super()._init_buffers()
        self.last_feet_contact = torch.zeros_like(self.feet_contact)
        self.last_stability_vel = torch.zeros_like(self.filtered_lin_vel)
        self.stability_accel_filtered = torch.zeros_like(self.filtered_lin_vel)

    def __init__(self, cfg):
        super().__init__(cfg)
        names = (self.cfg["randomization"].get("disturbance") or {}).get(
            "body_names", [self.cfg["asset"]["base_name"]])
        missing = [name for name in names if name not in self.body_names]
        if missing:
            raise ValueError(
                "disturbance body names were removed or misspelled: {} (loaded: {})".format(
                    missing, self.body_names))
        indices = [self.body_names.index(name) for name in names]
        if not indices:
            indices = [int(self.base_indice)]
        self.dist_body_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        self.dist_active_body = torch.full(
            (self.num_envs,), int(self.base_indice), dtype=torch.long, device=self.device)
        # 0=none, 1=short collision, 2=long support.  Eval uses this to avoid
        # pooling two physically different recovery problems into one number.
        self.dist_event_kind = torch.zeros(
            self.num_envs, dtype=torch.int8, device=self.device)
        self.dist_last_event_kind = torch.zeros(
            self.num_envs, dtype=torch.int8, device=self.device)
        self.dist_event_serial = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        self.dist_last_expected_impulse = torch.zeros(
            self.num_envs, device=self.device)
        self.dist_last_expected_torque_impulse = torch.zeros(
            self.num_envs, device=self.device)
        # Vector event provenance.  ``expected`` is the analytic held-wrench
        # integral, while ``submitted`` is accumulated from the tensor actually
        # passed on every physics substep.  The latter verifies the scheduler and
        # decimation plumbing; it is not a measurement of contact-free m*delta-v
        # because feet, gravity and joint actuation also exchange momentum.
        self.dist_last_expected_impulse_vec = torch.zeros(
            self.num_envs, 3, device=self.device)
        self.dist_last_expected_torque_impulse_vec = torch.zeros(
            self.num_envs, 3, device=self.device)
        self.dist_event_submitted_impulse_vec = torch.zeros(
            self.num_envs, 3, device=self.device)
        self.dist_event_submitted_torque_impulse_vec = torch.zeros(
            self.num_envs, 3, device=self.device)
        self.dist_last_submitted_impulse_vec = torch.zeros(
            self.num_envs, 3, device=self.device)
        self.dist_last_submitted_torque_impulse_vec = torch.zeros(
            self.num_envs, 3, device=self.device)
        self.dist_last_submitted_impulse = torch.zeros(
            self.num_envs, device=self.device)
        self.dist_last_submitted_torque_impulse = torch.zeros(
            self.num_envs, device=self.device)
        self.dist_last_direction_local = torch.zeros(
            self.num_envs, 3, device=self.device)
        self.dist_last_contact_offset_local = torch.zeros(
            self.num_envs, 3, device=self.device)
        self.dist_last_scenario_id = torch.zeros(
            self.num_envs, dtype=torch.int8, device=self.device)
        self.dist_last_height_tier = torch.full(
            (self.num_envs,), -1, dtype=torch.int8, device=self.device)
        self._configure_scenario_disturbance(
            (self.cfg["randomization"].get("disturbance") or {}).get(
                "scenario_aware") or {})
        self.dist_wrench_apply_calls = 0

    def _configure_scenario_disturbance(self, scenario_cfg):
        """Build immutable lookup tables for the optional player-contact model."""
        self._dist_scenario_enabled = bool(scenario_cfg.get("enabled", False))
        self.dist_scenario_names = ()
        self.dist_height_tier_names = ()
        if not self._dist_scenario_enabled:
            return

        scenarios = _merge_named_specs(
            SCENARIO_AWARE_DEFAULTS, scenario_cfg.get("scenarios"),
            "disturbance.scenario_aware.scenarios")
        tiers = _merge_named_specs(
            HEIGHT_TIER_DEFAULTS, scenario_cfg.get("height_tiers"),
            "disturbance.scenario_aware.height_tiers")

        tier_by_name = {item["name"]: i for i, item in enumerate(tiers)}
        for spec in scenarios:
            for key in ("force_n", "duration_s", "twist_nm"):
                spec[key] = _finite_range(
                    spec.get(key), "scenario {}.{}".format(spec["name"], key),
                    nonnegative=True)
            if spec["duration_s"][0] <= 0.0:
                raise ValueError("scenario {} duration must be positive".format(
                    spec["name"]))
            mode = spec.get("direction_mode", "uniform")
            if mode not in ("uniform", "rear_cone"):
                raise ValueError("scenario {} has unknown direction_mode {}".format(
                    spec["name"], mode))
            # There is intentionally no frontal/high-speed collision mode.
            if mode == "rear_cone":
                half = float(spec.get("half_angle_deg", 22.5))
                if not math.isfinite(half) or not 0.0 <= half <= 90.0:
                    raise ValueError("scenario {} rear cone must be 0..90 deg".format(
                        spec["name"]))
            allowed = spec.get("height_tiers")
            if allowed is None:
                spec["height_tier_indices"] = tuple(range(len(tiers)))
            else:
                unknown = [name for name in allowed if name not in tier_by_name]
                if unknown:
                    raise ValueError("scenario {} has unknown height tiers {}".format(
                        spec["name"], unknown))
                spec["height_tier_indices"] = tuple(
                    tier_by_name[name] for name in allowed)

        for spec in tiers:
            for key in ("offset_x_m", "offset_y_m", "offset_z_m"):
                spec[key] = _finite_range(
                    spec.get(key), "height tier {}.{}".format(spec["name"], key))
            body_weights = spec.get("body_weights")
            if not isinstance(body_weights, dict) or not body_weights:
                raise ValueError("height tier {} needs body_weights".format(spec["name"]))
            unknown = [name for name in body_weights if name not in self.body_names]
            if unknown:
                raise ValueError(
                    "height tier {} references unloaded bodies {} (loaded: {})".format(
                        spec["name"], unknown, self.body_names))
            weights = [float(body_weights[name]) for name in body_weights]
            if any((not math.isfinite(value) or value < 0.0) for value in weights):
                raise ValueError("height tier {} body weights must be nonnegative".format(
                    spec["name"]))
            total = sum(weights)
            if total <= 0.0:
                raise ValueError("height tier {} body weights sum to zero".format(
                    spec["name"]))
            spec["body_indices"] = torch.tensor(
                [self.body_names.index(name) for name in body_weights],
                dtype=torch.long, device=self.device)
            spec["body_probability"] = torch.tensor(
                [value / total for value in weights],
                dtype=torch.float, device=self.device)

        self._dist_scenario_specs = scenarios
        self._dist_height_tier_specs = tiers
        self._dist_scenario_probability = torch.tensor(
            [item["weight"] for item in scenarios],
            dtype=torch.float, device=self.device)
        self._dist_height_tier_probability = torch.tensor(
            [item["weight"] for item in tiers],
            dtype=torch.float, device=self.device)
        self.dist_scenario_names = tuple(item["name"] for item in scenarios)
        self.dist_height_tier_names = tuple(item["name"] for item in tiers)

    def _finalize_submitted_impulse(self, env_mask):
        """Snapshot the substep-integrated wrench for events that just ended."""
        if not bool(env_mask.any()):
            return
        self.dist_last_submitted_impulse_vec[env_mask] = (
            self.dist_event_submitted_impulse_vec[env_mask])
        self.dist_last_submitted_torque_impulse_vec[env_mask] = (
            self.dist_event_submitted_torque_impulse_vec[env_mask])
        self.dist_last_submitted_impulse[env_mask] = torch.norm(
            self.dist_event_submitted_impulse_vec[env_mask], dim=-1)
        self.dist_last_submitted_torque_impulse[env_mask] = torch.norm(
            self.dist_event_submitted_torque_impulse_vec[env_mask], dim=-1)

    # H1/H2 reflect the robot about its local y=0 plane.  Policy observation
    # and actions are handled by GoalPoseV3; the asymmetric critic channels
    # also have to be reflected for transition-level PPO augmentation.
    def mirror_privileged_obs(self, obs):
        out = obs.clone()
        # base_mass_scaled[0:4] stores the raw U[0,1] latent returned by
        # apply_randomization(..., return_noise=True), not the physical offset.
        # For symmetric uniform COM-y bounds, physical y -> -y is therefore
        # latent u -> 1-u.  The old u -> -u sent the mirrored critic OOD values
        # in [-1,0].
        out[..., 1] = 1.0 - out[..., 1]
        # base linear vy; applied force y. Torque is an axial vector, hence Tx
        # and Tz change sign under a y reflection while Ty does not.
        out[..., [5, 9, 11, 13]] *= -1.0
        return out

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        # A force active at a fall must never leak into the freshly spawned
        # episode.  V7 omitted this reset and could continue pushing a reset robot.
        self.pushing_forces[env_ids] = 0.0
        self.pushing_torques[env_ids] = 0.0
        self.dist_steps_left[env_ids] = 0
        self.dist_event_kind[env_ids] = 0
        self.last_feet_contact[env_ids] = False
        self.last_stability_vel[env_ids] = 0.0
        self.stability_accel_filtered[env_ids] = 0.0
        if hasattr(self, "dist_event_submitted_impulse_vec"):
            reset_mask = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device)
            reset_mask[env_ids] = True
            self._finalize_submitted_impulse(reset_mask)
            self.dist_event_submitted_impulse_vec[env_ids] = 0.0
            self.dist_event_submitted_torque_impulse_vec[env_ids] = 0.0
        d = self.cfg["randomization"].get("disturbance") or {}
        if d.get("enabled", False):
            lo, hi = d.get("interval_s", [8.0, 14.0])
            self.dist_next[env_ids] = torch.randint(
                max(1, int(lo / self.dt)), max(2, int(hi / self.dt)),
                (len(env_ids),), device=self.device)

    def step(self, actions):
        out = super().step(actions)
        done = out[2]
        self.last_feet_contact[:] = self.feet_contact & ~done.unsqueeze(-1)
        self.last_stability_vel[:] = self.filtered_lin_vel
        return out

    def _sample_scenario_events(self, fire):
        """Install non-collision player-contact wrench proxies for ``fire``.

        Isaac Gym's tensor API applies force at a rigid-body COM.  Because this
        asset collapses fixed arms into Trunk, an arm/chest contact is represented
        by an exactly equivalent rigid-body wrench: ``F`` at COM plus
        ``r x F`` and an independent twist torque.  It reproduces the resultant
        motion of the merged rigid body, but cannot reproduce arm-joint compliance,
        contact geometry or a second robot's coupled dynamics.

        Directions are sampled in robot coordinates at event onset, rotated to
        ENV_SPACE once, and then held there.  Using LOCAL_SPACE would make a
        person/contact direction rotate with the recovering robot.
        """
        k = len(fire)
        scenario_id = torch.multinomial(
            self._dist_scenario_probability, k, replacement=True)
        tier_id = torch.empty(k, dtype=torch.long, device=self.device)

        # Tier sampling is scenario-conditional (rear pushes target chest/arms;
        # entanglement targets the folded-arm proxy), while omni shove exposes
        # every height to every horizontal direction.
        for sid, scenario in enumerate(self._dist_scenario_specs):
            mask = scenario_id == sid
            n = int(mask.sum().item())
            if n == 0:
                continue
            allowed = torch.tensor(
                scenario["height_tier_indices"], dtype=torch.long,
                device=self.device)
            weights = self._dist_height_tier_probability[allowed]
            weights = weights / weights.sum()
            sampled = torch.multinomial(weights, n, replacement=True)
            tier_id[mask] = allowed[sampled]

        body = torch.empty(k, dtype=torch.long, device=self.device)
        contact_offset_local = torch.zeros(k, 3, device=self.device)
        for tid, tier in enumerate(self._dist_height_tier_specs):
            mask = tier_id == tid
            n = int(mask.sum().item())
            if n == 0:
                continue
            pick = torch.multinomial(
                tier["body_probability"], n, replacement=True)
            body[mask] = tier["body_indices"][pick]
            for axis, key in enumerate(
                    ("offset_x_m", "offset_y_m", "offset_z_m")):
                lo, hi = tier[key]
                value = torch_rand_float(
                    lo, hi, (n, 1), device=self.device).squeeze(1)
                if axis == 1 and bool(tier.get("mirror_y", False)):
                    side = torch.where(
                        torch.rand(n, device=self.device) < 0.5,
                        -torch.ones(n, device=self.device),
                        torch.ones(n, device=self.device))
                    value = value.abs() * side
                contact_offset_local[mask, axis] = value

        force_magnitude = torch.zeros(k, device=self.device)
        twist_magnitude = torch.zeros(k, device=self.device)
        duration = torch.zeros(k, device=self.device)
        angle_local = torch.zeros(k, device=self.device)
        for sid, scenario in enumerate(self._dist_scenario_specs):
            mask = scenario_id == sid
            n = int(mask.sum().item())
            if n == 0:
                continue
            flo, fhi = scenario["force_n"]
            dlo, dhi = scenario["duration_s"]
            tlo, thi = scenario["twist_nm"]
            force_magnitude[mask] = torch_rand_float(
                flo, fhi, (n, 1), device=self.device).squeeze(1)
            duration[mask] = torch_rand_float(
                dlo, dhi, (n, 1), device=self.device).squeeze(1)
            twist_magnitude[mask] = torch_rand_float(
                tlo, thi, (n, 1), device=self.device).squeeze(1)
            if scenario.get("direction_mode", "uniform") == "uniform":
                angle_local[mask] = torch_rand_float(
                    -np.pi, np.pi, (n, 1), device=self.device).squeeze(1)
            else:
                half = math.radians(float(scenario.get("half_angle_deg", 22.5)))
                angle_local[mask] = torch_rand_float(
                    -half, half, (n, 1), device=self.device).squeeze(1)

        direction_local = torch.stack((
            torch.cos(angle_local), torch.sin(angle_local),
            torch.zeros_like(angle_local)), dim=-1)
        force_local = direction_local * force_magnitude.unsqueeze(-1)
        force_world = quat_rotate(self.base_quat[fire], force_local)

        # Gaussian direction normalized to the sphere is isotropic.  The legacy
        # cube-normalized torque sampler remains untouched in the legacy branch.
        twist_axis_local = torch.randn(k, 3, device=self.device)
        twist_axis_local /= twist_axis_local.norm(
            dim=-1, keepdim=True).clamp(min=1.0e-6)
        twist_local = twist_axis_local * twist_magnitude.unsqueeze(-1)
        twist_world = quat_rotate(self.base_quat[fire], twist_local)
        offset_world = quat_rotate(
            self.base_quat[fire], contact_offset_local)
        moment_world = torch.cross(offset_world, force_world, dim=-1)
        torque_world = moment_world + twist_world

        duration_steps = torch.ceil(duration / self.dt).long().clamp(min=1)
        applied_duration = duration_steps.float() * self.dt

        self.pushing_forces[fire, body] = force_world
        self.pushing_torques[fire, body] = torque_world
        self.dist_active_body[fire] = body
        # These are sustained support/entanglement events, not the excluded
        # short full-speed collision. Keep the old evaluator's class as support;
        # scenario identity is carried independently below.
        self.dist_event_kind[fire] = 2
        self.dist_last_event_kind[fire] = 2
        self.dist_event_serial[fire] += 1
        self.dist_steps_left[fire] = duration_steps

        self.dist_last_expected_impulse[fire] = (
            force_magnitude * applied_duration)
        self.dist_last_expected_torque_impulse[fire] = (
            torch.norm(torque_world, dim=-1) * applied_duration)
        self.dist_last_expected_impulse_vec[fire] = (
            force_world * applied_duration.unsqueeze(-1))
        self.dist_last_expected_torque_impulse_vec[fire] = (
            torque_world * applied_duration.unsqueeze(-1))
        self.dist_last_direction_local[fire] = direction_local
        self.dist_last_contact_offset_local[fire] = contact_offset_local
        self.dist_last_scenario_id[fire] = (scenario_id + 1).to(torch.int8)
        self.dist_last_height_tier[fire] = tier_id.to(torch.int8)
        self.dist_event_submitted_impulse_vec[fire] = 0.0
        self.dist_event_submitted_torque_impulse_vec[fire] = 0.0

    def _record_legacy_event_telemetry(self, fire, body, applied_duration):
        """Populate new provenance buffers without changing legacy sampling."""
        force_world = self.pushing_forces[fire, body]
        torque_world = self.pushing_torques[fire, body]
        force_magnitude = torch.norm(force_world, dim=-1)
        direction_world = force_world / force_magnitude.unsqueeze(-1).clamp(
            min=1.0e-6)
        self.dist_last_expected_impulse_vec[fire] = (
            force_world * applied_duration.unsqueeze(-1))
        self.dist_last_expected_torque_impulse_vec[fire] = (
            torque_world * applied_duration.unsqueeze(-1))
        self.dist_last_direction_local[fire] = quat_rotate_inverse(
            self.base_quat[fire], direction_world)
        self.dist_last_contact_offset_local[fire] = 0.0
        self.dist_last_scenario_id[fire] = 0
        self.dist_last_height_tier[fire] = -1
        self.dist_event_submitted_impulse_vec[fire] = 0.0
        self.dist_event_submitted_torque_impulse_vec[fire] = 0.0

    def _push_robots(self):
        d = self.cfg["randomization"].get("disturbance") or {}
        if not d.get("enabled", False):
            return super()._push_robots()

        was_active = self.dist_steps_left > 0
        self.dist_steps_left = (self.dist_steps_left - 1).clamp(min=0)
        expired = was_active & (self.dist_steps_left == 0)
        # Snapshot exactly the wrench submitted on the preceding physics
        # ticks before clearing it.  Treating every already-inactive env as an
        # expiration would overwrite the last completed event with zeros on
        # every control step.
        self._finalize_submitted_impulse(expired)
        self.pushing_forces[expired] = 0.0
        self.pushing_torques[expired] = 0.0
        self.dist_event_kind[expired] = 0

        self.dist_next -= 1
        due = (self.dist_next <= 0).nonzero(as_tuple=False).flatten()
        if len(due) > 0:
            lo, hi = d.get("interval_s", [8.0, 14.0])
            self.dist_next[due] = torch.randint(
                max(1, int(lo / self.dt)), max(2, int(hi / self.dt)),
                (len(due),), device=self.device)

            ramp_steps = max(1, int(d.get("ramp_steps", 1)))
            ramp = min(1.0, self.common_step_counter / float(ramp_steps))
            probability = float(d.get("event_probability", 1.0)) * ramp
            event_prob = torch.full((len(due),), probability, device=self.device)
            boost = float(d.get("high_speed_probability_boost", 1.0))
            if boost > 1.0:
                fast_due = (self.is_path_env[due]
                            & (torch.norm(self.filtered_lin_vel[due, :2], dim=-1)
                               >= float(d.get("high_speed_threshold_mps", 0.8))))
                event_prob[fast_due] *= boost
            fire = due[torch.rand(len(due), device=self.device) < event_prob.clamp(max=1.0)]
            if len(fire) > 0:
                k = len(fire)
                # One event owns one body.  Clear any still-active wrench before
                # replacing it so an accidentally short interval cannot leave a
                # force on the previous body and silently accumulate multi-body
                # loads.  Normal H configs already space events beyond their
                # maximum duration; this makes the invariant explicit in code.
                self.pushing_forces[fire] = 0.0
                self.pushing_torques[fire] = 0.0
                if self._dist_scenario_enabled:
                    self._sample_scenario_events(fire)
                    return
                body = self.dist_body_indices[
                    torch.randint(0, len(self.dist_body_indices), (k,), device=self.device)]
                self.dist_active_body[fire] = body
                is_collision = torch.rand(k, device=self.device) < float(d.get("collision_share", 0.5))
                self.dist_event_kind[fire] = torch.where(
                    is_collision,
                    torch.ones(k, dtype=torch.int8, device=self.device),
                    torch.full((k,), 2, dtype=torch.int8, device=self.device),
                )
                self.dist_last_event_kind[fire] = self.dist_event_kind[fire]
                self.dist_event_serial[fire] += 1
                collision, support = d.get("collision", {}), d.get("support", {})

                def sample_pair(section, key, default):
                    bounds = section.get(key, default)
                    return torch_rand_float(bounds[0], bounds[1], (k, 1), device=self.device).squeeze(1)

                fmag = torch.where(is_collision,
                                   sample_pair(collision, "force_n", [40.0, 100.0]),
                                   sample_pair(support, "force_n", [3.0, 8.0]))
                tmag = torch.where(is_collision,
                                   sample_pair(collision, "torque_nm", [3.0, 12.0]),
                                   sample_pair(support, "torque_nm", [0.2, 1.0]))
                duration = torch.where(is_collision,
                                       sample_pair(collision, "duration_s", [0.05, 0.10]),
                                       sample_pair(support, "duration_s", [0.5, 1.5]))
                duration_steps = torch.ceil(duration / self.dt).long().clamp(min=1)
                applied_duration = duration_steps.float() * self.dt
                self.dist_last_expected_impulse[fire] = fmag * applied_duration
                self.dist_last_expected_torque_impulse[fire] = (
                    tmag * applied_duration)

                angle = torch_rand_float(-np.pi, np.pi, (k, 1), device=self.device).squeeze(1)
                self.pushing_forces[fire, body, 0] = fmag * torch.cos(angle)
                self.pushing_forces[fire, body, 1] = fmag * torch.sin(angle)
                axis = torch_rand_float(-1.0, 1.0, (k, 3), device=self.device)
                axis /= axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                self.pushing_torques[fire, body] = axis * tmag.unsqueeze(-1)
                self.dist_steps_left[fire] = duration_steps
                self._record_legacy_event_telemetry(
                    fire, body, applied_duration)

    def _apply_external_wrenches_substep(self):
        """Apply a held control-step wrench on every decimated physics tick.

        ``apply_rigid_body_force_tensors`` is an immediate-timestep API.  The
        event schedule remains at the 50 Hz control rate, while this hook makes
        its configured duration/impulse real at the 500 Hz physics rate.
        """
        d = self.cfg["randomization"].get("disturbance") or {}
        if not d.get("enabled", False):
            return
        self.dist_wrench_apply_calls += 1
        # Integrate the exact tensors submitted on each 500 Hz physics tick.
        # This proves scheduler/decimation delivery and is compared with the
        # analytic event impulse.  It is not an observed robot momentum change,
        # because feet, gravity and actuation exchange momentum concurrently.
        sim_dt = float(self.cfg["sim"]["dt"])
        self.dist_event_submitted_impulse_vec += (
            self.pushing_forces.sum(dim=1) * sim_dt)
        self.dist_event_submitted_torque_impulse_vec += (
            self.pushing_torques.sum(dim=1) * sim_dt)
        # ENV_SPACE keeps a long support push fixed in the world instead of
        # rotating the force vector with the robot.  It is still a wrench proxy,
        # not a second simulated robot collision.
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.pushing_forces),
            gymtorch.unwrap_tensor(self.pushing_torques),
            gymapi.ENV_SPACE,
        )

    def _reward_high_speed_stability(self):
        """Stabilize only steady high-speed motion; do not punish acceleration lean."""
        c = self.cfg["rewards"].get("high_speed_stability", {}) or {}
        # Use the existing ~0.2 s low-pass velocity, not one-step trunk-link
        # velocity.  The latter contains stride sway and would classify nearly
        # every high-speed step as "accelerating", making this term inert.
        speed = torch.norm(self.filtered_lin_vel[:, :2], dim=-1)
        acc_instant = (self.filtered_lin_vel - self.last_stability_vel) / self.dt
        alpha = float(c.get("accel_filter_alpha", 0.10))
        self.stability_accel_filtered[:] = (
            alpha * acc_instant + (1.0 - alpha) * self.stability_accel_filtered)
        acc_body = self.stability_accel_filtered
        acc_xy = torch.norm(acc_body[:, :2], dim=-1)
        speed_gate = torch.sigmoid((speed - float(c.get("min_speed_mps", 0.8))) /
                                   float(c.get("speed_width_mps", 0.10)))
        # At |a| above the threshold this gate tends to zero, explicitly
        # preserving the useful forward lean used to accelerate.
        steady_gate = torch.sigmoid((float(c.get("max_accel_mps2", 0.3)) - acc_xy) /
                                    float(c.get("accel_width_mps2", 0.08)))
        gx, gy, gz = self.projected_gravity.unbind(dim=-1)
        pitch = torch.atan2(-gx, -gz)
        roll = torch.atan2(gy, -gz)
        angular = torch.square(self.base_ang_vel[:, :2]).sum(dim=-1)
        vertical = torch.square(self.base_lin_vel[:, 2])
        penalty = (torch.square(pitch) + torch.square(roll)
                   + float(c.get("angular_rate_weight", 0.10)) * angular
                   + float(c.get("vertical_velocity_weight", 0.02)) * vertical)
        return speed_gate * steady_gate * penalty

    def _reward_heel_strike_ahead(self):
        """H3-only kinematic touchdown proxy, gated off outside forward walking.

        Isaac Gym exposes a net force per foot body, not the true sole contact
        point.  We therefore use the first-contact transition plus the known
        heel corner.  A smooth capture-point-like target avoids the brittle
        binary rule "heel must be ahead of trunk" and its over-striding failure.
        """
        c = self.cfg["rewards"].get("heel_strike", {}) or {}
        first = self.feet_contact & ~self.last_feet_contact
        forward = self.base_lin_vel[:, 0] > float(c.get("min_forward_speed_mps", 0.6))
        active = first & forward.unsqueeze(-1) & self.is_path_env.unsqueeze(-1)

        heel_local = torch.tensor([-0.1015, 0.0, -0.03], device=self.device)
        heel_local = heel_local.view(1, 1, 3).expand(self.num_envs, len(self.feet_indices), 3)
        heel_world = self.feet_pos + quat_rotate(
            self.feet_quat.reshape(-1, 4), heel_local.reshape(-1, 3)).reshape_as(self.feet_pos)
        rel_world = heel_world - self.base_pos.unsqueeze(1)
        rel_body = quat_rotate_inverse(
            self.base_quat.unsqueeze(1).expand(-1, len(self.feet_indices), -1).reshape(-1, 4),
            rel_world.reshape(-1, 3),
        ).reshape_as(rel_world)
        target = (float(c.get("velocity_gain_s", 0.08)) * self.base_lin_vel[:, 0]).clip(
            min=float(c.get("target_min_m", 0.02)), max=float(c.get("target_max_m", 0.12)))
        sigma = float(c.get("sigma_m", 0.04))
        score = torch.exp(-torch.square(rel_body[:, :, 0] - target.unsqueeze(-1)) /
                          max(sigma * sigma, 1e-8))
        return (score * active.float()).sum(dim=-1)

    def _compute_observations(self):
        super()._compute_observations()
        # Base GoalPose publishes only body index 0's wrench.  HBatch can push
        # five different bodies and stores those wrenches in ENV_SPACE, so give
        # the asymmetric critic the resultant expressed in the robot frame.
        # This makes the privileged mirror map well-defined and prevents an
        # arm/hip hit from becoming invisible to the critic.
        force_body = quat_rotate_inverse(self.base_quat, self.pushing_forces.sum(dim=1))
        torque_body = quat_rotate_inverse(self.base_quat, self.pushing_torques.sum(dim=1))
        self.privileged_obs_buf[:, 8:11] = force_body * self.cfg["normalization"]["push_force"]
        self.privileged_obs_buf[:, 11:14] = torque_body * self.cfg["normalization"]["push_torque"]
        self.extras["privileged_obs"] = self.privileged_obs_buf
        e = self.extras.setdefault("hbatch", {})
        speed = torch.norm(self.base_lin_vel[:, :2], dim=-1)
        fast = speed >= float((self.cfg["rewards"].get("high_speed_stability") or {}).get(
            "min_speed_mps", 0.8))
        if bool(fast.any()):
            gx, gy, gz = self.projected_gravity[fast].unbind(dim=-1)
            e["high_speed_pitch_abs_mean"] = float(torch.atan2(-gx, -gz).abs().mean().item())
            e["high_speed_roll_abs_mean"] = float(torch.atan2(gy, -gz).abs().mean().item())
            e["high_speed_ang_xy_mean"] = float(torch.norm(self.base_ang_vel[fast, :2], dim=-1).mean().item())
        e["disturbance_active_share"] = float((self.dist_steps_left > 0).float().mean().item())
        e["joint_encoder_bias_abs_mean"] = float(self.joint_encoder_bias.abs().mean().item())
        e["joint_target_offset_abs_mean"] = float(self.joint_target_offset.abs().mean().item())
