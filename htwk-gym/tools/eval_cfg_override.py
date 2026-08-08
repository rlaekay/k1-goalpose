#!/usr/bin/env python3
"""공통 평가 config 에 **지정한 키만** 이식한다. 다른 것이 하나라도 바뀌면 죽는다.

    python tools/eval_cfg_override.py \
        --cfg /tmp/base_eval.yaml --from sweeps/NJ_armasset.yaml \
        --keys asset.armature_by_joint --out /tmp/own_physics.yaml

---- 왜 이게 필요한가 ---------------------------------------------------------

이 저장소가 **반복해서** 낸 결함이 하나 있다: **arm 의 레버가 공통 평가 config 에
없어서 레버를 끈 채로 채점한다.** 확인된 것만:

  * `NC_actfilter`  -- `control.action_filter_tau` 가 공통 config 에 없다 -> 필터 OFF 로 채점
  * `NZ_zeroiid` / `N9_zerostruct` / `NB_zerocritic` -- `joint_zero.enabled: false` 로 채점
  * `NG_armature`   -- `asset.armature` 0.02 로 학습하고 **0.0 으로 채점**(§8-52)

앞의 둘은 채점이 끝난 뒤에야 알았다. 세 번째는 미리 알아서 2x2 로 갈랐고, 그때
쓴 방법(공통 config 에서 한 키만 sed 로 바꾸고 diff 가 두 줄인지 센다)이 통했다.
그 방법을 **도구로 만든다** -- 다음 arm 에서 또 손으로 하면 또 틀린다.

`make_eval_cfg.py` 와의 역할 분담:
  * `make_eval_cfg.py` -- **정책 인터페이스**(observation 폭)를 옮긴다. 안 옮기면 로드가 안 된다.
  * 이 도구            -- **물리/레버**를 옮긴다. 안 옮기면 로드는 되는데 **딴 것을 잰다.**

⛔ 두 채점을 **둘 다** 해야 한다. 레버를 켠 채점만 하면 arm 끼리 비교가 안 되고
(각자 다른 물리에서 잰 값이 된다), 끈 채점만 하면 레버의 효과를 못 본다.
§8-52 의 2x2 가 그 이유를 숫자로 보여 준다 -- 같은 체크포인트가 채점 물리에 따라
낙상간격 1.5 s 와 3,740 s 사이를 오간다.
"""
import argparse
import copy
import sys

try:
    import yaml
except ImportError:
    sys.exit("PyYAML 이 필요하다: pip install pyyaml")


def walk(node, prefix=""):
    """dict 를 점 표기 경로 -> 값 으로 편다. list 는 잎으로 둔다."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{prefix}.{k}" if prefix else str(k))
    else:
        yield prefix, node


def get_dotted(cfg, dotted):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(dotted)
        cur = cur[part]
    return cur


def set_dotted(cfg, dotted, value):
    parts = dotted.split(".")
    cur = cfg
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = copy.deepcopy(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="바탕이 되는 평가 config (보통 make_eval_cfg 출력)")
    ap.add_argument("--from", dest="src", required=True, help="레버를 가져올 arm config")
    ap.add_argument("--keys", required=True, nargs="+", help="이식할 점 표기 키(여러 개 가능)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = yaml.safe_load(open(args.cfg))
    src = yaml.safe_load(open(args.src))
    out = copy.deepcopy(base)

    for key in args.keys:
        try:
            val = get_dotted(src, key)
        except KeyError:
            sys.exit(f"⛔ arm config 에 '{key}' 가 없다: {args.src}")
        try:
            old = get_dotted(base, key)
        except KeyError:
            old = "(없음)"
        set_dotted(out, key, val)
        print(f"이식: {key}\n   {old}  ->  {val}")

    # ---- 검증: 요청한 키 말고 **아무것도** 안 바뀌었는지 직접 확인한다 --------
    # 이식이 의도한 것만 바꿨다는 것을 텍스트 diff 가 아니라 **파싱된 값**으로 본다.
    # yaml 왕복은 서식을 통째로 바꾸므로 텍스트 diff 는 여기서 쓸 수 없다.
    before = dict(walk(base))
    after = dict(walk(out))
    changed = sorted(
        set(before) ^ set(after)
        | {k for k in set(before) & set(after) if before[k] != after[k]}
    )
    allowed = [k for k in changed if any(k == key or k.startswith(key + ".") for key in args.keys)]
    stray = [k for k in changed if k not in allowed]
    if stray:
        print("⛔ 요청하지 않은 키가 바뀌었다:", file=sys.stderr)
        for k in stray[:20]:
            print(f"   {k}: {before.get(k, '(없음)')} -> {after.get(k, '(없음)')}", file=sys.stderr)
        sys.exit(1)
    if not allowed:
        print("⛔ 아무것도 안 바뀌었다 -- 레버가 이미 같은 값이거나 키가 틀렸다.", file=sys.stderr)
        sys.exit(1)

    with open(args.out, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)
    print(f"바뀐 경로 {len(allowed)}개, 그 밖의 변화 0개 -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
