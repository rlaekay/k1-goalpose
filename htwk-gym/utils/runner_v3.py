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

import os

import torch
import torch.nn.functional as F

from utils.runner import Runner
from utils.utils import discount_values, surrogate_loss
from utils.recorder import Recorder


class RunnerV3(Runner):

    def train(self):
        self.recorder = Recorder(self.cfg)
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        privileged_obs = infos["privileged_obs"].to(self.device)

        horizon = self.cfg["runner"]["horizon_length"]
        mini_epochs = self.cfg["runner"]["mini_epochs"]
        num_minibatches = self.cfg["runner"].get("num_minibatches", 4)
        sym_coef = self.cfg["algorithm"].get("symmetry_coef", 0.0)
        use_symmetry = sym_coef > 0.0 and hasattr(self.env, "mirror_obs")
        batch_size = horizon * self.env.num_envs
        mb_size = batch_size // num_minibatches

        for it in range(self.cfg["basic"]["max_iterations"]):
            # ---- rollout (identical to parent) ------------------------------
            for n in range(horizon):
                self.buffer.update_data("obses", n, obs)
                self.buffer.update_data("privileged_obses", n, privileged_obs)
                with torch.no_grad():
                    dist = self.model.act(obs)
                    act = dist.sample()
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

            # ---- fixed reference policy for the whole update phase ----------
            with torch.no_grad():
                old_dist = self.model.act(self.buffer["obses"])
                old_mu = old_dist.loc.reshape(batch_size, -1)
                old_sigma = old_dist.scale.reshape(batch_size, -1)
                old_logprob = old_dist.log_prob(self.buffer["actions"]).sum(dim=-1).reshape(batch_size)

            obs_b = self.buffer["obses"].reshape(batch_size, -1)
            priv_b = self.buffer["privileged_obses"].reshape(batch_size, -1)
            act_b = self.buffer["actions"].reshape(batch_size, -1)

            stats = {"value_loss": 0.0, "actor_loss": 0.0, "bound_loss": 0.0, "entropy": 0.0, "symmetry_loss": 0.0}
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

                perm = torch.randperm(batch_size, device=self.device)
                for k in range(num_minibatches):
                    idx = perm[k * mb_size:(k + 1) * mb_size]
                    mb_obs = obs_b[idx]

                    values = self.model.est_value(mb_obs, priv_b[idx])
                    value_loss = F.mse_loss(values, returns_b[idx])

                    dist = self.model.act(mb_obs)
                    logprob = dist.log_prob(act_b[idx]).sum(dim=-1)
                    actor_loss = surrogate_loss(old_logprob[idx], logprob, adv_b[idx])

                    bound_loss = (
                        torch.clip(dist.loc - 1.0, min=0.0).square().mean()
                        + torch.clip(dist.loc + 1.0, max=0.0).square().mean()
                    )
                    entropy = dist.entropy().sum(dim=-1)

                    loss = (
                        value_loss
                        + actor_loss
                        + self.cfg["algorithm"]["bound_coef"] * bound_loss
                        + self.cfg["algorithm"]["entropy_coef"] * entropy.mean()
                    )

                    sym_loss = torch.tensor(0.0, device=self.device)
                    if use_symmetry:
                        mirrored_dist = self.model.act(self.env.mirror_obs(mb_obs))
                        sym_loss = F.mse_loss(mirrored_dist.loc, self.env.mirror_actions(dist.loc))
                        loss = loss + sym_coef * sym_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    # adaptive-KL learning rate, same rule as the parent but on
                    # the minibatch against the fixed pre-update reference
                    with torch.no_grad():
                        kl = torch.sum(
                            torch.log(dist.scale / old_sigma[idx])
                            + 0.5 * (torch.square(old_sigma[idx]) + torch.square(dist.loc - old_mu[idx])) / torch.square(dist.scale)
                            - 0.5,
                            axis=-1,
                        )
                        kl_mean = torch.mean(kl)
                        if kl_mean > self.cfg["algorithm"]["desired_kl"] * 2:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.cfg["algorithm"]["desired_kl"] / 2:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                    stats["value_loss"] += value_loss.item()
                    stats["actor_loss"] += actor_loss.item()
                    stats["bound_loss"] += bound_loss.item()
                    stats["entropy"] += entropy.mean().item()
                    stats["symmetry_loss"] += sym_loss.item()
                    num_updates += 1

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
                self.recorder.save(
                    {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "curriculum": self.env.curriculum_prob,
                    },
                    it + 1,
                )
            print("epoch: {}/{}".format(it + 1, self.cfg["basic"]["max_iterations"]))

            # graceful early stop: tools/auto_stop.py (or a human) drops a STOP
            # file into the run dir when the reward curve plateaus
            if os.path.exists(os.path.join(self.recorder.dir, "STOP")):
                self.recorder.save(
                    {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "curriculum": self.env.curriculum_prob,
                    },
                    it + 1,
                )
                print("STOP file found in {}; saved checkpoint and stopped at iteration {}.".format(self.recorder.dir, it + 1))
                break
