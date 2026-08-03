import isaacgym  # noqa: F401  (must be imported before torch)
from utils.runner_crossq import RunnerCrossQ

if __name__ == "__main__":
    runner = RunnerCrossQ(test=False)
    runner.train()
