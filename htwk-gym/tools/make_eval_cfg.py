"""공통 평가 프로토콜에 그 arm 의 **관측 인터페이스만** 이식한 config 를 만든다.

    python tools/make_eval_cfg.py --common sweeps/N0_ctrl.yaml \
        --run logs/K1/K1/Goal_Pose_V7/2026-...-NA_histzero --out /tmp/eval_NA.yaml

⛔ 왜 필요한가 (2026-08-07 실제 사고): `eval_round.sh` 는 모든 arm 을 같은
`sweeps/N0_ctrl.yaml` 로 채점한다 -- arm 마다 자기 config 로 채점하면 비교가 아니기
때문이다. 그런데 그 config 는 과제만 정하는 게 아니라 `num_observations` 도 정하고,
모델은 config 로부터 만들어진다. 그래서 관측을 넓힌 arm(NA_histzero, obs 270)을
채점하려 하면 체크포인트가 아예 안 올라간다:

    size mismatch for actor.0.weight:
      checkpoint [256, 270] vs current model [256, 54]

**구분이 필요하다.** config 안에는 두 종류가 섞여 있다:

  * 프로토콜 — 목표 분포, 노이즈, 외란, 지형, 자산. arm 끼리 비교하려면 **같아야** 한다.
  * 정책 인터페이스 — `observation.*` 와 `env.num_observations/num_privileged_obs`.
    이건 체크포인트의 성질이라 **정책을 따라가야** 한다. 여기를 공통값으로 고정하는
    것은 "같은 자로 재는 것"이 아니라 그냥 못 재는 것이다.

그래서 이 스크립트는 공통 config 를 바탕으로 **정확히 그 두 그룹의 키만** 옮긴다.
`--config` 를 통째로 갈아끼우는 것과 정반대다(그건 24개 키가 같이 바뀌어 35.6분을
버린 전례가 있다).

⚠️ `noise.obs_delay_steps` 는 옮기지 **않는다.** 그건 정책 인터페이스가 아니라
환경 조건이고, 평가는 모든 arm 을 같은 조건에서 재야 한다. NA 가 지연 속에서
학습했다는 사실은 그 정책의 성질이지 그 정책만 다른 시험지를 받을 이유가 아니다.
"""

import argparse
import copy
import os
import sys

import yaml

# 정책 인터페이스로 취급해 이식하는 키. 이 목록이 곧 "무엇이 체크포인트의 성질인가"에
# 대한 답이고, 늘어나면 여기에만 추가한다.
TRANSPLANT = [
    ("observation",),
    ("env", "num_observations"),
    ("env", "num_privileged_obs"),
]


def get_in(cfg, path):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None, False
        cur = cur[k]
    return cur, True


def set_in(cfg, path, value):
    cur = cfg
    for k in path[:-1]:
        cur = cur.setdefault(k, {})
    cur[path[-1]] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--common", required=True, help="공통 프로토콜 config")
    ap.add_argument("--run", required=True, help="채점할 run 디렉터리(그 안의 config.yaml 을 읽는다)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run_cfg_path = os.path.join(args.run, "config.yaml")
    if not os.path.exists(run_cfg_path):
        raise SystemExit("run config 가 없다: {}".format(run_cfg_path))

    with open(args.common, encoding="utf-8") as f:
        common = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(run_cfg_path, encoding="utf-8") as f:
        run = yaml.load(f.read(), Loader=yaml.FullLoader)

    moved = []
    for path in TRANSPLANT:
        val, ok = get_in(run, path)
        if not ok:
            continue      # 옛 run 은 observation 블록이 없다 -- 공통값(=기본값)이 맞다
        old, _ = get_in(common, path)
        if old != val:
            moved.append("{} : {} -> {}".format(".".join(path), old, val))
        set_in(common, path, copy.deepcopy(val))

    common.setdefault("basic", {})["description"] = "eval:{}".format(os.path.basename(args.run))
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.dump(common, f, sort_keys=False, allow_unicode=True)

    print("공통 config: {}".format(args.common))
    print("run:         {}".format(os.path.basename(args.run)))
    if moved:
        print("이식한 관측 인터페이스:")
        for m in moved:
            print("  " + m)
    else:
        print("이식할 것 없음 -- 공통 config 그대로다.")
    print("-> {}".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
