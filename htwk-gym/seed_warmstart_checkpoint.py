import os
import argparse
import yaml
import torch

from utils.model import ActorCritic

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seed a runner-loadable checkpoint (.pth) for a new task by warm-starting "
        "its actor from a deploy-exported, torch.jit-scripted actor-only model (e.g. "
        "deploy/models/parameter_walk.pt, produced by export_model.py). The critic is left "
        "randomly initialized since deploy exports don't include it, and no optimizer state "
        "is written -- utils/runner.py's _load() already tolerates a missing 'optimizer' key."
    )
    parser.add_argument("--task", required=True, help="Target task, matches envs/<task>.yaml (e.g. K1/Goal_Pose)")
    parser.add_argument("--source", required=True, help="Path to the torch.jit-scripted actor-only .pt")
    parser.add_argument("--out", required=True, help="Output checkpoint path (.pth)")
    args = parser.parse_args()

    cfg_file = os.path.join("envs", "{}.yaml".format(args.task))
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    model = ActorCritic(cfg["env"]["num_actions"], cfg["env"]["num_observations"], cfg["env"]["num_privileged_obs"])

    scripted_actor = torch.jit.load(args.source, map_location="cpu")
    # strict=True on purpose: a shape/name mismatch here means the source model's
    # architecture doesn't match utils/model.py's ActorCritic.actor, and silently
    # ignoring that would produce a checkpoint that looks fine but isn't warm-started.
    model.actor.load_state_dict(scripted_actor.state_dict())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model": model.state_dict()}, args.out)
    print(f"Seeded checkpoint written to {args.out}")
    print(f"  actor warm-started from: {args.source}")
    print("  critic: randomly initialized (not present in the deploy export)")
