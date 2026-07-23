# htwk-gym upstream provenance

`htwk-gym/` in this repo started as a plain copy of
[NaoHTWK/htwk-gym](https://github.com/NaoHTWK/htwk-gym), forked at:

- commit: `cbb4f51942d3e2b1f0be0127ebf1030146b5455d`
- upstream branch: `main`
- date pulled into this repo: 2026-07-23

It is tracked here as regular files (not a git submodule) — see the
"변경 이력" section in [MASTERPLAN.md](MASTERPLAN.md) for why.

## Diffing against upstream

To compare our changes against the fork point, clone upstream separately
and diff by path, e.g.:

```bash
git clone https://github.com/NaoHTWK/htwk-gym.git /tmp/htwk-gym-upstream
git -C /tmp/htwk-gym-upstream checkout cbb4f51942d3e2b1f0be0127ebf1030146b5455d
diff -ru /tmp/htwk-gym-upstream/envs htwk-gym/envs
```

## Known local changes vs. upstream

- `envs/K1/goal_pose.py` + `envs/K1/Goal_Pose.yaml` — new GoalPose task (not upstream).
- `envs/__init__.py` — registers `GoalPose`.
- `envs/K1/parameter_walk.py` — **not yet cleaned up**: still contains the
  debug artifacts documented in MASTERPLAN.md's 변경 이력 (per-step `print`,
  CSV logging, hardcoded command override). GoalPose was built from a
  cleaned copy of this logic, but the original ParameterWalk file itself
  is untouched pending milestone 0 (baseline reproduction).
