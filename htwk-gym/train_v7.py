import isaacgym  # noqa: F401  (must be imported before torch)
from utils.runner_v3 import RunnerV3

if __name__ == "__main__":
    # v7 reuses the v3 runner: minibatched PPO + L/R symmetry loss.
    runner = RunnerV3(test=False)
    runner.train()
