import isaacgym  # noqa: F401  (must be imported before torch)

from utils.runner_v3 import RunnerV3


if __name__ == "__main__":
    RunnerV3(test=False).train()
