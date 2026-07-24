"""RunnerCrossQ -- off-policy CrossQ training loop for the v4 GetUp task.

Reuses the base Runner's CLI / config / seed helpers and get_task_class, but
replaces the PPO machinery (ActorCritic, ExperienceBuffer, on-policy train())
with CrossQ actor/critic + a GPU replay buffer.

An "iteration" is crossq.steps_per_log vector env-steps, so --max_iterations,
runner.save_interval, the STOP-file graceful stop, and the Recorder tensorboard
cadence all keep the same meaning as in the PPO runner. tools/auto_stop.py works
unchanged (it watches the "reward" scalar).

Checkpoints are {"actor","critic","log_alpha","env_steps"} -- a distinct format
from the PPO {"model",...} dict, so the two never collide.
"""

import os

import torch
import torch.nn.functional as F

from utils.runner import Runner, get_task_class
from utils.crossq import CrossQActor, CrossQCritic, ReplayBuffer
from utils.recorder import Recorder


class RunnerCrossQ(Runner):

    def __init__(self, test=False):
        self.test = test
        # reuse the base helpers (identical CLI) without its PPO model/buffer setup
        self._get_args()
        self._update_cfg_from_args()
        self._set_seed()

        task_name = self.cfg["basic"]["task"]
        if "/" in task_name:
            task_name = task_name.split("/")[-1]
        task_class = get_task_class(task_name)
        if task_class is None:
            raise ValueError(f"Unknown task: {task_name}")
        self.env = task_class(self.cfg)
        self.device = self.cfg["basic"]["rl_device"]

        cq = self.cfg["crossq"]
        self.actor = CrossQActor(
            self.env.num_obs, self.env.num_actions,
            hidden=tuple(cq["actor_hidden"]), bn_momentum=cq["bn_momentum"], bn_warmup=cq["bn_warmup_updates"],
        ).to(self.device)
        self.critic = CrossQCritic(
            self.env.num_obs, self.env.num_privileged_obs, self.env.num_actions,
            hidden=tuple(cq["critic_hidden"]), bn_momentum=cq["bn_momentum"], bn_warmup=cq["bn_warmup_updates"],
        ).to(self.device)

        betas = tuple(cq.get("adam_betas", [0.5, 0.999]))
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cq["lr"], betas=betas)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cq["lr"], betas=betas)
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cq["lr"], betas=betas)
        self.target_entropy = cq["target_entropy"]

        self.buffer = ReplayBuffer(
            cq["buffer_size"], self.env.num_obs, self.env.num_privileged_obs, self.env.num_actions, self.device
        )
        self._load()

    def _load(self):
        ckpt = self.cfg["basic"].get("checkpoint")
        if not ckpt or ckpt in ("-1", -1):
            return
        print("Loading CrossQ checkpoint from {}".format(ckpt))
        d = torch.load(ckpt, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        with torch.no_grad():
            self.log_alpha.copy_(d["log_alpha"].to(self.device))

    def _update(self, update_idx):
        cq = self.cfg["crossq"]
        obs, priv, act, rew, next_obs, next_priv, done = self.buffer.sample(cq["batch_size"])
        alpha = self.log_alpha.exp()

        # --- critic update: the CrossQ joint (s,a)+(s',a') forward pass ---
        self.actor.eval()
        with torch.no_grad():
            next_act, next_logp = self.actor.sample(next_obs)
        self.critic.train()
        cat_obs = torch.cat([obs, next_obs], dim=0)
        cat_priv = torch.cat([priv, next_priv], dim=0)
        cat_act = torch.cat([act, next_act], dim=0)
        q_all = self.critic(cat_obs, cat_priv, cat_act)
        b = obs.shape[0]
        q_pred, q_next = q_all[:b], q_all[b:].detach()
        with torch.no_grad():
            min_q_next = q_next.min(dim=1).values
            target = rew + cq["gamma"] * (1.0 - done) * (min_q_next - alpha.detach() * next_logp)
        critic_loss = F.mse_loss(q_pred[:, 0], target) + F.mse_loss(q_pred[:, 1], target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = torch.tensor(0.0, device=self.device)
        if update_idx % cq["policy_delay"] == 0:
            # --- actor + temperature update ---
            self.actor.train()
            new_act, logp = self.actor.sample(obs)
            self.critic.eval()  # no BN-stat update / no critic step; grad flows to actions
            q_new = self.critic(obs, priv, new_act).min(dim=1).values
            actor_loss = (alpha.detach() * logp - q_new).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        return critic_loss.item(), actor_loss.item(), alpha.item()

    def train(self):
        self.recorder = Recorder(self.cfg)
        cq = self.cfg["crossq"]
        steps_per_log = cq["steps_per_log"]
        learning_starts = cq["learning_starts"]
        updates_per_step = cq["updates_per_step"]

        obs, extras = self.env.reset()
        obs = obs.to(self.device)
        priv = extras["privileged_obs"].to(self.device)
        env_steps = 0
        update_idx = 0

        for it in range(self.cfg["basic"]["max_iterations"]):
            c_loss = a_loss = alpha_val = 0.0
            n_updates = 0
            for n in range(steps_per_log):
                if self.buffer.count < learning_starts:
                    action = torch.empty(self.env.num_envs, self.env.num_actions, device=self.device).uniform_(-1.0, 1.0)
                else:
                    self.actor.eval()
                    with torch.no_grad():
                        action, _ = self.actor.sample(obs)

                next_obs, rew, done, extras = self.env.step(action.to(self.env.device))
                rew = rew.to(self.device)
                done = done.to(self.device)
                s_prime = extras["terminal_obs"].to(self.device)
                s_prime_priv = extras["terminal_privileged_obs"].to(self.device)
                time_outs = extras["time_outs"].to(self.device)
                settling = extras["settling"].to(self.device)

                # true termination bootstraps to 0; timeouts keep bootstrapping.
                bootstrap_done = done & ~time_outs
                keep = ~settling
                self.buffer.insert(
                    obs[keep], priv[keep], action[keep], rew[keep], s_prime[keep], s_prime_priv[keep], bootstrap_done[keep]
                )

                ep_info = {"reward": rew}
                ep_info.update({k: v.to(self.device) for k, v in extras["rew_terms"].items()})
                self.recorder.record_episode_statistics(done, ep_info, it, n == (steps_per_log - 1))

                obs = next_obs.to(self.device)
                priv = extras["privileged_obs"].to(self.device)
                env_steps += self.env.num_envs

                if self.buffer.count >= learning_starts:
                    for k in range(updates_per_step):
                        cl, al, av = self._update(update_idx)
                        update_idx += 1
                        c_loss += cl
                        a_loss += al
                        alpha_val = av
                        n_updates += 1

            denom = max(n_updates, 1)
            self.recorder.record_statistics(
                {
                    "critic_loss": c_loss / denom,
                    "actor_loss": a_loss / denom,
                    "alpha": alpha_val,
                    "buffer_fill": self.buffer.count / self.buffer.capacity,
                    "env_steps": env_steps,
                },
                it,
            )

            if (it + 1) % self.cfg["runner"]["save_interval"] == 0:
                self.recorder.save(self._checkpoint(env_steps), it + 1)
            print("epoch: {}/{}".format(it + 1, self.cfg["basic"]["max_iterations"]))

            if os.path.exists(os.path.join(self.recorder.dir, "STOP")):
                self.recorder.save(self._checkpoint(env_steps), it + 1)
                print("STOP file found in {}; saved checkpoint and stopped at iteration {}.".format(self.recorder.dir, it + 1))
                break

    def _checkpoint(self, env_steps):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "env_steps": env_steps,
        }

    def play(self):
        obs, extras = self.env.reset()
        obs = obs.to(self.device)
        self.actor.eval()
        while True:
            with torch.no_grad():
                action = self.actor.act_deterministic(obs)
            obs, _, _, _ = self.env.step(action.to(self.env.device))
            obs = obs.to(self.device)
