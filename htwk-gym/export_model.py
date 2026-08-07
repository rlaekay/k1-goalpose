import os
import glob
import yaml
import argparse
import torch
from utils.model import *  # ActorCritic lives here; utils.model_thomas does not exist

def get_robot_type(task_name):
    """Determine robot type from task name."""
    # Check if task name starts with K1 or T1
    if task_name.startswith("K1"):
        return "K1"
    elif task_name.startswith("T1"):
        return "T1"
    else:
        # Default fallback - could be extended for other robot types
        return "Unknown"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
    parser.add_argument("--checkpoint", type=str, help="Path of model checkpoint to load. Overrides config file if provided.")
    args = parser.parse_args()
    cfg_file = os.path.join("envs", "{}.yaml".format(args.task))
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    if args.checkpoint is not None:
        cfg["basic"]["checkpoint"] = args.checkpoint

    if not cfg["basic"]["checkpoint"] or (cfg["basic"]["checkpoint"] == "-1") or (cfg["basic"]["checkpoint"] == -1):
        # Look for models in hierarchical structure: logs/robot_type/task_name/**/*.pth
        task_name = args.task
        robot_type = get_robot_type(task_name)
        
        # First try: exact task in robot-specific folder
        task_log_pattern = os.path.join("logs", robot_type, task_name, "**/*.pth")
        task_models = sorted(glob.glob(task_log_pattern, recursive=True), key=os.path.getmtime)
        
        if task_models:
            cfg["basic"]["checkpoint"] = task_models[-1]
        else:
            # Second try: any task in robot-specific folder
            robot_log_pattern = os.path.join("logs", robot_type, "**/*.pth")
            robot_models = sorted(glob.glob(robot_log_pattern, recursive=True), key=os.path.getmtime)
            
            if robot_models:
                cfg["basic"]["checkpoint"] = robot_models[-1]
            else:
                # Fallback: all logs if no robot-specific models found
                cfg["basic"]["checkpoint"] = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
    print("Loading model from {}".format(cfg["basic"]["checkpoint"]))
    model_dict = torch.load(cfg["basic"]["checkpoint"], map_location="cpu", weights_only=True)

    # ⛔ 폭을 **체크포인트에서 읽는다.** 예전에는 `envs/{task}.yaml`(베이스)의
    # 54/14 로 ActorCritic 을 만들고 strict 로드해서, 관측을 넓힌 arm 은
    # export 자체가 불가능했다 -- NA_histzero(270) N4_hist(270) N5_tau(66)
    # N6_foot(56) N7_critic(priv 29) **다섯 arm 의 출구가 동시에 막혀 있었다.**
    # 배포 런타임(policy_goal_pose.py)은 이미 넓은 관측을 지원하므로 끊긴 곳은
    # 여기 하나였다.
    #
    # 체크포인트가 폭의 유일한 진실원이다 -- config 는 arm 마다 다른데 이 스크립트는
    # 베이스만 읽는다. 가중치 모양에서 역산하면 어느 arm 이든 맞는다.
    sd = model_dict["model"]
    num_act = int(sd["actor.6.weight"].shape[0])
    num_obs = int(sd["actor.0.weight"].shape[1])
    num_priv = int(sd["critic.0.weight"].shape[1]) - num_obs
    base = (cfg["env"]["num_observations"], cfg["env"]["num_privileged_obs"])
    if (num_obs, num_priv) != base:
        print("  체크포인트 폭이 베이스 config 와 다르다: obs {}->{} priv {}->{} "
              "(관측 확장 arm). 체크포인트 쪽을 따른다.".format(
                  base[0], num_obs, base[1], num_priv))
    model = ActorCritic(num_act, num_obs, num_priv)
    model.load_state_dict(sd)
    # 배포가 실제로 넘길 폭으로 한 번 굴려 본다. 관측 순서가 어긋난 정책은 예외를
    # 내지 않고 그냥 이상하게 걷기 때문에, 여기서 잡을 수 있는 것은 폭뿐이다.
    with torch.no_grad():
        probe = model.actor(torch.zeros((1, num_obs), dtype=torch.float32))
    if tuple(probe.shape) != (1, num_act):
        raise SystemExit("export 스모크 실패: actor 출력 {} (기대 (1,{}))".format(
            tuple(probe.shape), num_act))
    print("  export 폭: obs {} / priv {} / act {}".format(num_obs, num_priv, num_act))

    model.eval()
    script_module = torch.jit.script(model.actor)
    save_path = os.path.splitext(cfg["basic"]["checkpoint"])[0] + ".pt"
    script_module.save(save_path)
    print(f"Saved model to {save_path}")
