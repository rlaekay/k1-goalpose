"""RunnerV3 -- PPO with minibatching + optional mirror-symmetry loss.

utils/runner.py stays frozen; this subclass only replaces train(). Differences:

  - standard minibatched PPO: mini_epochs x num_minibatches gradient updates on
    shuffled slices of the rollout, instead of mini_epochs full-batch updates.
    (rsl_rl/legged_gym convention: 5 epochs x 4 minibatches. Same number of
    gradient steps as the old full-batch-x20 at 1/4 the per-step compute, with
    better-conditioned updates.)
  - optional symmetry loss L_sym = MSE(pi(mirror(s)), mirror(pi(s))) using the
    env's mirror maps (Abdolhosseini et al. 2019). Enabled by
    algorithm.symmetry_coef > 0 AND the env providing mirror_obs/mirror_actions.
    NOTE: the long-dead algorithm.symmetric_coef key in older configs was never
    consumed by any code; v3 deliberately uses a new name to avoid silently
    activating a stale value.
  - logs curriculum goal_level / success EMA when the env provides them.

Same STOP-file graceful early stop and checkpoint format as the parent, so
tools/auto_stop.py and eval_goal_pose.py work unchanged.
"""

import json
import os

import torch
import torch.nn.functional as F

from utils.runner import Runner
from utils.utils import discount_values
from utils.recorder import Recorder


def _bounded_surrogate_loss(old_logprob, new_logprob, advantages,
                            max_abs_log_ratio):
    """PPO surrogate whose exponent cannot overflow in float32.

    The ordinary PPO clip happens after exp(log-ratio), which is too late for
    mirrored synthetic actions in the tail of a narrow Gaussian.  Saturating
    the log-ratio is a numerical guard; the usual epsilon clip still defines
    the PPO objective inside that finite envelope.
    """
    log_ratio = torch.clamp(
        new_logprob - old_logprob,
        min=-float(max_abs_log_ratio), max=float(max_abs_log_ratio))
    ratio = torch.exp(log_ratio)
    surrogate = -advantages * ratio
    surrogate_clipped = -advantages * torch.clamp(ratio, 0.8, 1.2)
    return torch.max(surrogate, surrogate_clipped).mean()


def _normal_kl(old_mu, old_sigma, new_mu, new_sigma):
    return torch.sum(
        torch.log(new_sigma / old_sigma)
        + 0.5 * (torch.square(old_sigma) + torch.square(new_mu - old_mu))
        / torch.square(new_sigma)
        - 0.5,
        dim=-1,
    )


class RunnerV3(Runner):

    @staticmethod
    def _require_finite(name, *tensors):
        for tensor in tensors:
            if tensor is None:
                continue
            if not bool(torch.isfinite(tensor).all().item()):
                bad = int((~torch.isfinite(tensor)).sum().item())
                raise FloatingPointError(
                    "nonfinite {}: {} / {} values".format(
                        name, bad, tensor.numel()))

    @staticmethod
    def _write_health_marker(path, payload):
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        temporary = path + ".tmp-{}".format(os.getpid())
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)

    def train(self):
        self.recorder = Recorder(self.cfg)
        # Stamp the env-code SHA beside the run. Config drift is already visible
        # (config.yaml sits next to the checkpoints); code drift is not, and it
        # silently invalidated E1's re-evaluation on 2026-07-27.
        try:
            import subprocess
            _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            _sha = subprocess.run(["git", "-C", _root, "log", "-1", "--format=%H", "--", "envs"],
                                  capture_output=True, text=True, timeout=10).stdout.strip()
            if _sha:
                with open(os.path.join(self.recorder.dir, "ENV_CODE_SHA"), "w") as _f:
                    _f.write(_sha + "\n")
        except Exception:
            pass
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        privileged_obs = infos["privileged_obs"].to(self.device)

        horizon = self.cfg["runner"]["horizon_length"]
        mini_epochs = self.cfg["runner"]["mini_epochs"]
        num_minibatches = self.cfg["runner"].get("num_minibatches", 4)
        sym_coef = self.cfg["algorithm"].get("symmetry_coef", 0.0)
        mirror_aug_coef = self.cfg["algorithm"].get("mirror_augmentation_coef", 0.0)
        use_symmetry = sym_coef > 0.0 and hasattr(self.env, "mirror_obs")
        use_mirror_aug = mirror_aug_coef > 0.0 and hasattr(self.env, "mirror_obs")
        batch_size = horizon * self.env.num_envs
        mb_size = batch_size // num_minibatches
        finite_checks = bool(self.cfg["algorithm"].get("finite_checks", False))
        min_learning_rate = float(self.cfg["algorithm"].get(
            "min_learning_rate",
            min(float(self.cfg["algorithm"]["learning_rate"]), 1.0e-5)))
        max_learning_rate = float(self.cfg["algorithm"].get(
            "max_learning_rate", 1.0e-2))
        if not (0.0 < min_learning_rate <= max_learning_rate):
            raise ValueError("invalid adaptive learning-rate bounds: [{}, {}]".format(
                min_learning_rate, max_learning_rate))
        self.learning_rate = min(max_learning_rate,
                                 max(min_learning_rate, self.learning_rate))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate
        initial_optimizer_lr = self.learning_rate
        max_abs_log_ratio = float(self.cfg["algorithm"].get(
            "max_abs_log_ratio", 20.0))
        mirror_max_std = float(self.cfg["algorithm"].get(
            "mirror_augmentation_max_std", 5.0))
        mirror_min_valid_share = float(self.cfg["algorithm"].get(
            "mirror_augmentation_min_valid_share", 0.0))
        if max_abs_log_ratio <= 0.0 or mirror_max_std <= 0.0:
            raise ValueError("log-ratio and mirror-support bounds must be positive")

        health_path = os.environ.get("HBATCH_HEALTH_MARKER", "")
        health_token = os.environ.get("HBATCH_HEALTH_TOKEN", "")
        health_iterations = max(
            1, int(os.environ.get("HBATCH_HEALTH_ITERATIONS", "2")))
        optimizer_steps = 0
        max_grad_norm_seen = 0.0
        initial_actor_parameters = [
            parameter.detach().clone() for parameter in self.model.actor.parameters()]
        last_mirror_valid_share = 1.0
        if finite_checks:
            self._require_finite("initial observations", obs, privileged_obs)
            self._require_finite(
                "initial model parameters", *list(self.model.parameters()))

        for it in range(self.cfg["basic"]["max_iterations"]):
            # ---- rollout (identical to parent) ------------------------------
            for n in range(horizon):
                self.buffer.update_data("obses", n, obs)
                self.buffer.update_data("privileged_obses", n, privileged_obs)
                with torch.no_grad():
                    dist = self.model.act(obs)
                    act = dist.sample()
                if finite_checks:
                    self._require_finite(
                        "rollout policy/action at iteration {} step {}".format(
                            it + 1, n + 1),
                        dist.loc, dist.scale, act)
                obs, rew, done, infos = self.env.step(act)
                obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
                privileged_obs = infos["privileged_obs"].to(self.device)
                self.buffer.update_data("actions", n, act)
                self.buffer.update_data("rewards", n, rew)
                self.buffer.update_data("dones", n, done)
                self.buffer.update_data("time_outs", n, infos["time_outs"].to(self.device))
                ep_info = {"reward": rew}
                ep_info.update(infos["rew_terms"])
                self.recorder.record_episode_statistics(done, ep_info, it, n == (horizon - 1))

            if finite_checks:
                self._require_finite(
                    "rollout tensors at iteration {}".format(it + 1),
                    self.buffer["obses"], self.buffer["privileged_obses"],
                    self.buffer["actions"], self.buffer["rewards"], obs,
                    privileged_obs)

            # ---- fixed reference policy for the whole update phase ----------
            with torch.no_grad():
                old_dist = self.model.act(self.buffer["obses"])
                old_mu = old_dist.loc.reshape(batch_size, -1)
                old_sigma = old_dist.scale.reshape(batch_size, -1)
                old_logprob = old_dist.log_prob(self.buffer["actions"]).sum(dim=-1).reshape(batch_size)
                if use_mirror_aug:
                    old_mirror_dist = self.model.act(self.env.mirror_obs(self.buffer["obses"]))
                    old_mirror_mu = old_mirror_dist.loc.reshape(batch_size, -1)
                    old_mirror_sigma = old_mirror_dist.scale.reshape(batch_size, -1)
                    old_mirror_logprob = old_mirror_dist.log_prob(
                        self.env.mirror_actions(self.buffer["actions"])
                    ).sum(dim=-1).reshape(batch_size)

            obs_b = self.buffer["obses"].reshape(batch_size, -1)
            priv_b = self.buffer["privileged_obses"].reshape(batch_size, -1)
            act_b = self.buffer["actions"].reshape(batch_size, -1)
            mirrored_act_b = self.env.mirror_actions(act_b) if use_mirror_aug else None
            if use_mirror_aug:
                mirror_z = torch.abs(
                    (mirrored_act_b - old_mirror_mu) / old_mirror_sigma)
                mirror_valid = mirror_z.amax(dim=-1) <= mirror_max_std
                last_mirror_valid_share = float(mirror_valid.float().mean().item())
                if last_mirror_valid_share < mirror_min_valid_share:
                    raise RuntimeError(
                        "mirror augmentation support {:.3f} is below configured minimum "
                        "{:.3f} at iteration {}".format(
                            last_mirror_valid_share, mirror_min_valid_share, it + 1))
            if finite_checks:
                finite_refs = [old_mu, old_sigma, old_logprob]
                if use_mirror_aug:
                    finite_refs.extend([
                        old_mirror_mu, old_mirror_sigma,
                        old_mirror_logprob, mirror_z])
                self._require_finite(
                    "frozen PPO reference at iteration {}".format(it + 1),
                    *finite_refs)

            stats = {"value_loss": 0.0, "actor_loss": 0.0,
                     "mirror_actor_loss": 0.0, "bound_loss": 0.0,
                     "entropy": 0.0, "symmetry_loss": 0.0,
                     "mirror_valid_share": 0.0, "grad_norm": 0.0,
                     "mirror_kl_mean": 0.0}
            num_updates = 0
            kl_mean = torch.tensor(0.0, device=self.device)

            for _ in range(mini_epochs):
                # refresh advantages/returns once per epoch (matches the
                # parent's per-epoch recompute, but only once per 4 updates)
                with torch.no_grad():
                    values_seq = self.model.est_value(self.buffer["obses"], self.buffer["privileged_obses"])
                    last_values = self.model.est_value(obs, privileged_obs)
                    self.buffer["rewards"][self.buffer["time_outs"]] = values_seq[self.buffer["time_outs"]]
                    advantages = discount_values(
                        self.buffer["rewards"],
                        self.buffer["dones"] | self.buffer["time_outs"],
                        values_seq,
                        last_values,
                        self.cfg["algorithm"]["gamma"],
                        self.cfg["algorithm"]["lam"],
                    )
                    returns_b = (values_seq + advantages).reshape(batch_size)
                    adv_b = advantages.reshape(batch_size)
                    adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
                    if finite_checks:
                        self._require_finite(
                            "returns/advantages at iteration {}".format(it + 1),
                            values_seq, last_values, returns_b, adv_b)

                perm = torch.randperm(batch_size, device=self.device)
                for k in range(num_minibatches):
                    idx = perm[k * mb_size:(k + 1) * mb_size]
                    mb_obs = obs_b[idx]

                    values = self.model.est_value(mb_obs, priv_b[idx])
                    value_loss = F.mse_loss(values, returns_b[idx])
                    if use_mirror_aug and hasattr(self.env, "mirror_privileged_obs"):
                        mirror_values = self.model.est_value(
                            self.env.mirror_obs(mb_obs),
                            self.env.mirror_privileged_obs(priv_b[idx]),
                        )
                        value_loss = 0.5 * (
                            value_loss + F.mse_loss(mirror_values, returns_b[idx]))

                    dist = self.model.act(mb_obs)
                    logprob = dist.log_prob(act_b[idx]).sum(dim=-1)
                    actor_loss = _bounded_surrogate_loss(
                        old_logprob[idx], logprob, adv_b[idx],
                        max_abs_log_ratio)

                    mirror_actor_loss = torch.tensor(0.0, device=self.device)
                    mirrored_dist = None
                    mirror_logprob = None
                    if use_mirror_aug:
                        mirrored_dist = self.model.act(self.env.mirror_obs(mb_obs))
                        valid_local = mirror_valid[idx]
                        valid_pos = torch.nonzero(
                            valid_local, as_tuple=False).squeeze(-1)
                        if valid_pos.numel() > 0:
                            valid_actions = mirrored_act_b[idx][valid_pos]
                            valid_mirror_dist = torch.distributions.Normal(
                                mirrored_dist.loc[valid_pos],
                                mirrored_dist.scale[valid_pos])
                            mirror_logprob = valid_mirror_dist.log_prob(
                                valid_actions).sum(dim=-1)
                            # The reflected transition is a pseudo-on-policy
                            # augmentation referenced to frozen pi_old(Ms).
                            # Restrict it to transformed actions within the old
                            # mirrored policy's configured Gaussian support;
                            # the explicit mean-equivariance loss below covers
                            # the remaining asymmetric samples without a tail
                            # likelihood-ratio explosion.
                            mirror_actor_loss = _bounded_surrogate_loss(
                                old_mirror_logprob[idx][valid_pos],
                                mirror_logprob, adv_b[idx][valid_pos],
                                max_abs_log_ratio)

                    bound_loss = (
                        torch.clip(dist.loc - 1.0, min=0.0).square().mean()
                        + torch.clip(dist.loc + 1.0, max=0.0).square().mean()
                    )
                    entropy = dist.entropy().sum(dim=-1)

                    loss = (
                        value_loss
                        + actor_loss
                        + mirror_aug_coef * mirror_actor_loss
                        + self.cfg["algorithm"]["bound_coef"] * bound_loss
                        + self.cfg["algorithm"]["entropy_coef"] * entropy.mean()
                    )

                    sym_loss = torch.tensor(0.0, device=self.device)
                    if use_symmetry:
                        if mirrored_dist is None:
                            mirrored_dist = self.model.act(self.env.mirror_obs(mb_obs))
                        sym_loss = F.mse_loss(mirrored_dist.loc, self.env.mirror_actions(dist.loc))
                        loss = loss + sym_coef * sym_loss

                    if finite_checks:
                        self._require_finite(
                            "PPO losses at iteration {} minibatch {}".format(
                                it + 1, num_updates + 1),
                            values, value_loss, dist.loc, dist.scale, logprob,
                            actor_loss, mirror_actor_loss, mirror_logprob,
                            bound_loss, entropy, sym_loss, loss)
                    self.optimizer.zero_grad()
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 1.0,
                        error_if_nonfinite=finite_checks)
                    self.optimizer.step()
                    optimizer_steps += 1
                    grad_norm_value = float(grad_norm.item())
                    max_grad_norm_seen = max(max_grad_norm_seen, grad_norm_value)
                    if finite_checks:
                        self._require_finite(
                            "model parameters after optimizer step {}".format(
                                optimizer_steps),
                            *list(self.model.parameters()))

                    # Re-forward AFTER optimizer.step.  The old implementation
                    # measured the stale pre-step distribution, so its first KL
                    # was exactly zero and increased LR without evidence.  For
                    # mirrored arms, control against the worse of original and
                    # mirrored KL.
                    with torch.no_grad():
                        post_dist = self.model.act(mb_obs)
                        original_kl_mean = _normal_kl(
                            old_mu[idx], old_sigma[idx],
                            post_dist.loc, post_dist.scale).mean()
                        mirror_kl_mean = torch.tensor(0.0, device=self.device)
                        post_mirror_dist = None
                        if use_mirror_aug:
                            post_mirror_dist = self.model.act(
                                self.env.mirror_obs(mb_obs))
                            mirror_kl_mean = _normal_kl(
                                old_mirror_mu[idx], old_mirror_sigma[idx],
                                post_mirror_dist.loc,
                                post_mirror_dist.scale).mean()
                        kl_mean = torch.maximum(
                            original_kl_mean, mirror_kl_mean)
                        if finite_checks:
                            self._require_finite(
                                "post-update policy/KL at optimizer step {}".format(
                                    optimizer_steps),
                                post_dist.loc, post_dist.scale,
                                post_mirror_dist.loc if post_mirror_dist is not None else None,
                                post_mirror_dist.scale if post_mirror_dist is not None else None,
                                original_kl_mean, mirror_kl_mean, kl_mean)
                        if kl_mean > self.cfg["algorithm"]["desired_kl"] * 2:
                            self.learning_rate = max(
                                min_learning_rate, self.learning_rate / 1.5)
                        elif (kl_mean > 0 and
                              kl_mean < self.cfg["algorithm"]["desired_kl"] / 2):
                            self.learning_rate = min(
                                max_learning_rate, self.learning_rate * 1.5)
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                    stats["value_loss"] += value_loss.item()
                    stats["actor_loss"] += actor_loss.item()
                    stats["mirror_actor_loss"] += mirror_actor_loss.item()
                    stats["bound_loss"] += bound_loss.item()
                    stats["entropy"] += entropy.mean().item()
                    stats["symmetry_loss"] += sym_loss.item()
                    stats["mirror_valid_share"] += last_mirror_valid_share
                    stats["grad_norm"] += grad_norm_value
                    stats["mirror_kl_mean"] += mirror_kl_mean.item()
                    num_updates += 1

            with torch.no_grad():
                post_update_probe = self.model.act(obs[:min(64, obs.shape[0])])
            optimizer_state_tensors = []
            for state in self.optimizer.state.values():
                optimizer_state_tensors.extend(
                    value for value in state.values()
                    if isinstance(value, torch.Tensor))
            if finite_checks:
                self._require_finite(
                    "post-iteration policy at iteration {}".format(it + 1),
                    post_update_probe.loc, post_update_probe.scale)
                self._require_finite(
                    "optimizer state at iteration {}".format(it + 1),
                    *optimizer_state_tensors)

            parameter_delta = max(
                float((parameter.detach() - initial).abs().max().item())
                for parameter, initial in zip(
                    self.model.actor.parameters(), initial_actor_parameters))
            health_payload = {
                "version": 1,
                "status": "healthy",
                "health_token": health_token,
                "completed_iterations": it + 1,
                "optimizer_steps": optimizer_steps,
                "num_envs": self.env.num_envs,
                "horizon_length": horizon,
                "mini_epochs": mini_epochs,
                "num_minibatches": num_minibatches,
                "finite_checks": finite_checks,
                "post_update_forward_finite": True,
                "parameter_delta_max": parameter_delta,
                "max_grad_norm_seen": max_grad_norm_seen,
                "mirror_valid_share": last_mirror_valid_share,
                "configured_learning_rate": float(
                    self.cfg["algorithm"]["learning_rate"]),
                "initial_optimizer_learning_rate": initial_optimizer_lr,
                "current_learning_rate": self.learning_rate,
                "checkpoint": self.cfg["basic"].get("checkpoint"),
            }

            record = {name: value / num_updates for name, value in stats.items()}
            record.update(
                {
                    "kl_mean": kl_mean,
                    "lr": self.learning_rate,
                    "curriculum/goal_level": getattr(self.env, "goal_level", 1.0),
                    "curriculum/goal_success_ema": getattr(self.env, "goal_success_ema", 0.0),
                    "curriculum/mean_lin_vel_level": self.env.mean_lin_vel_level,
                    "curriculum/mean_ang_vel_level": self.env.mean_ang_vel_level,
                    "curriculum/max_lin_vel_level": self.env.max_lin_vel_level,
                    "curriculum/max_ang_vel_level": self.env.max_ang_vel_level,
                }
            )
            self.recorder.record_statistics(record, it)

            if (it + 1) % self.cfg["runner"]["save_interval"] == 0:
                self.recorder.save(self._checkpoint_payload(), it + 1)
            print("epoch: {}/{}".format(it + 1, self.cfg["basic"]["max_iterations"]))
            if (it + 1 <= health_iterations or
                    (it + 1) % self.cfg["runner"]["save_interval"] == 0 or
                    it + 1 == self.cfg["basic"]["max_iterations"]):
                self._write_health_marker(health_path, health_payload)

            # graceful early stop: tools/auto_stop.py (or a human) drops a STOP
            # file into the run dir when the reward curve plateaus
            if os.path.exists(os.path.join(self.recorder.dir, "STOP")):
                self.recorder.save(self._checkpoint_payload(), it + 1)
                print("STOP file found in {}; saved checkpoint and stopped at iteration {}.".format(self.recorder.dir, it + 1))
                break
