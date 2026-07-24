"""CrossQ building blocks (Bhatt et al., ICLR 2024) for the v4 GetUp task.

CrossQ = SAC without target networks. The two essentials implemented here:
  1. BatchRenorm in the critics (and actor trunk): running-statistic
     normalization that behaves like plain BatchNorm during a warmup period and
     then clamps the train/running statistic ratio (r, d) for stability.
  2. The joint (s,a)+(s',a') forward pass: current and next transitions are
     concatenated into ONE batch before the critic so BN statistics cover both
     distributions -- this is what replaces the target network. Never run s'
     through the critic separately in train mode.

torch 2.0 has no BatchRenorm; implemented from the paper (Ioffe 2017) with the
CrossQ-recommended clamps r in [1/3, 3], d in [-5, 5].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchRenorm1d(nn.Module):

    def __init__(self, num_features, momentum=0.01, eps=1e-5, warmup_updates=100000, r_max=3.0, d_max=5.0):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.warmup_updates = warmup_updates
        self.r_max = r_max
        self.d_max = d_max
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_updates", torch.zeros(1, dtype=torch.long))

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            batch_std = torch.sqrt(batch_var + self.eps)
            running_std = torch.sqrt(self.running_var + self.eps)

            if self.num_updates.item() < self.warmup_updates:
                # plain BatchNorm behavior while running stats are unreliable
                x_hat = (x - batch_mean) / batch_std
            else:
                r = (batch_std / running_std).detach().clamp(1.0 / self.r_max, self.r_max)
                d = ((batch_mean - self.running_mean) / running_std).detach().clamp(-self.d_max, self.d_max)
                x_hat = (x - batch_mean) / batch_std * r + d

            with torch.no_grad():
                self.running_mean += self.momentum * (batch_mean - self.running_mean)
                self.running_var += self.momentum * (batch_var - self.running_var)
                self.num_updates += 1
        else:
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        return x_hat * self.weight + self.bias


class CrossQActor(nn.Module):
    """Tanh-squashed Gaussian policy, MLP [384, 256] with BatchRenorm (FRASA/CrossQ)."""

    LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0

    def __init__(self, num_obs, num_actions, hidden=(384, 256), bn_momentum=0.01, bn_warmup=100000):
        super().__init__()
        layers = [BatchRenorm1d(num_obs, momentum=bn_momentum, warmup_updates=bn_warmup)]
        in_dim = num_obs
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), BatchRenorm1d(h, momentum=bn_momentum, warmup_updates=bn_warmup)]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.mu_head = nn.Linear(in_dim, num_actions)
        self.log_std_head = nn.Linear(in_dim, num_actions)

    def forward(self, obs):
        z = self.trunk(obs)
        mu = self.mu_head(z)
        log_std = self.log_std_head(z).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs):
        """Returns (action in [-1,1], log_prob) with the tanh correction."""
        mu, log_std = self(obs)
        std = log_std.exp()
        noise = torch.randn_like(mu)
        pre_tanh = mu + std * noise
        action = torch.tanh(pre_tanh)
        # log prob of tanh-squashed gaussian (numerically stable form)
        log_prob = (-0.5 * noise.square() - log_std - 0.5 * torch.log(torch.tensor(2.0 * torch.pi))).sum(dim=-1)
        log_prob -= (2.0 * (torch.log(torch.tensor(2.0)) - pre_tanh - F.softplus(-2.0 * pre_tanh))).sum(dim=-1)
        return action, log_prob

    def act_deterministic(self, obs):
        mu, _ = self(obs)
        return torch.tanh(mu)


class CrossQCritic(nn.Module):
    """Ensemble of 2 Q-networks over (obs, privileged_obs, action).

    Asymmetric off-policy critic: the privileged channels (base velocity, mass,
    height) are sim-only, exactly like the PPO stack's critic. No target copy
    exists -- callers must use the joint (s,s') forward pass in train mode.
    """

    def __init__(self, num_obs, num_priv, num_actions, hidden=(1024, 1024), bn_momentum=0.01, bn_warmup=100000):
        super().__init__()
        in_dim = num_obs + num_priv + num_actions
        self.nets = nn.ModuleList()
        for _ in range(2):
            layers = [BatchRenorm1d(in_dim, momentum=bn_momentum, warmup_updates=bn_warmup)]
            d = in_dim
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ReLU(), BatchRenorm1d(h, momentum=bn_momentum, warmup_updates=bn_warmup)]
                d = h
            layers += [nn.Linear(d, 1)]
            self.nets.append(nn.Sequential(*layers))

    def forward(self, obs, priv, action):
        x = torch.cat((obs, priv, action), dim=-1)
        return torch.cat([net(x) for net in self.nets], dim=-1)  # (B, 2)


class ReplayBuffer:
    """Preallocated GPU circular buffer; vectorized insert of num_envs rows/step."""

    def __init__(self, capacity, num_obs, num_priv, num_actions, device):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros(capacity, num_obs, device=device)
        self.priv = torch.zeros(capacity, num_priv, device=device)
        self.actions = torch.zeros(capacity, num_actions, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.next_obs = torch.zeros(capacity, num_obs, device=device)
        self.next_priv = torch.zeros(capacity, num_priv, device=device)
        # done for bootstrap masking = true termination only (timeouts bootstrap)
        self.dones = torch.zeros(capacity, device=device)
        self.pos = 0
        self.count = 0

    def insert(self, obs, priv, actions, rewards, next_obs, next_priv, dones):
        n = obs.shape[0]
        if n == 0:
            return
        idx = (self.pos + torch.arange(n, device=self.device)) % self.capacity
        self.obs[idx] = obs
        self.priv[idx] = priv
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.next_obs[idx] = next_obs
        self.next_priv[idx] = next_priv
        self.dones[idx] = dones.float()
        self.pos = int((self.pos + n) % self.capacity)
        self.count = min(self.count + n, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.count, (batch_size,), device=self.device)
        return (
            self.obs[idx],
            self.priv[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.next_priv[idx],
            self.dones[idx],
        )
