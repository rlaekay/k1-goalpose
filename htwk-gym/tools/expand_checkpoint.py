"""관측을 넓힌 정책으로 **기능적으로 동일하게** warm start 하기 위한 체크포인트 수술.

문제: `utils/runner.py::_load` 는 `load_state_dict(..., strict=False)` 로 읽는데,
strict=False 는 *없는 키/남는 키*만 봐준다. **모양이 다른 키는 그대로 예외**다.
그래서 num_observations 를 54 -> 그 이상으로 바꾸면 warm start 자체가 실패한다.
처음부터 다시 학습하면 M3 기준 5.8시간이고, 그건 관측 하나 시험하는 값이 아니다.

해법: 첫 층 가중치만 넓히고 새 열을 0으로 채운다. 새 채널의 기여가 정확히 0이므로
수술 직후 정책은 **옛 정책과 출력이 완전히 같고**, 학습이 진행되며 새 채널을 쓰기
시작한다. 이건 근사가 아니라 항등이다 -- 아래 --verify 가 그걸 실제로 검사한다.

모델 구조(utils/model.py):
    actor.0  : Linear(num_obs, 256)
    critic.0 : Linear(num_obs + num_privileged_obs, 256)

⚠️ critic 이 까다롭다. 입력이 concat(obs, privileged) 라 obs 가 넓어지면 privileged
열이 **오른쪽으로 밀린다.** 그래서 단순히 뒤에 0을 붙이면 안 되고, 옛 privileged
가중치를 새 위치로 옮겨야 한다. 이 파일이 존재하는 가장 큰 이유가 그 한 가지다.

관측 레이아웃 규약(envs/K1/goal_pose.py::_obs_extra_channels):
    한 프레임 = [legacy 54][foot_offset 2][dof_tau 12]
    전체      = [frame_t][frame_{t-1}] ... [frame_{t-k+1}]
옛 54 채널이 항상 새 벡터의 [0:54] 에 그대로 있으므로 옛 가중치는 [:, :54] 로 간다.

사용:
    python tools/expand_checkpoint.py \
        --src logs/.../nn/model_1500.pth --dst logs/.../nn/model_1500_expanded.pth \
        --old-obs 54 --new-obs 270 --old-priv 14 --new-priv 17 --verify
"""

import argparse
import sys

import torch


def expand_linear_in(weight, new_in, blocks):
    """[out, old_in] -> [out, new_in]. blocks = [(옛 시작, 새 시작, 길이), ...].

    명시된 구간만 복사하고 나머지는 0이다. '뒤에 0을 붙인다'가 아니라 '어디서
    어디로 옮기는지'를 호출자가 적게 만든 것은, critic 의 privileged 구간이
    가운데에서 밀리기 때문이다.
    """
    out = torch.zeros(weight.shape[0], new_in, dtype=weight.dtype, device=weight.device)
    for src0, dst0, n in blocks:
        out[:, dst0:dst0 + n] = weight[:, src0:src0 + n]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--old-obs", type=int, default=54)
    ap.add_argument("--new-obs", type=int, required=True)
    ap.add_argument("--old-priv", type=int, default=14)
    ap.add_argument("--new-priv", type=int, default=None,
                    help="생략하면 old-priv 와 같다(비평자 추가 없음)")
    ap.add_argument("--verify", action="store_true",
                    help="옛/새 actor 가 같은 관측에 대해 같은 행동을 내는지 검사")
    args = ap.parse_args()
    new_priv = args.new_priv if args.new_priv is not None else args.old_priv

    if args.new_obs < args.old_obs or new_priv < args.old_priv:
        raise SystemExit("이 도구는 넓히기만 한다(줄이기는 항등을 보장할 수 없다)")

    ck = torch.load(args.src, map_location="cpu", weights_only=True)
    # 원본을 따로 붙들어 둔다. 아래에서 model 을 제자리 수정하므로, 검증에 쓸
    # "옛 정책"을 같은 객체에서 다시 읽으면 이미 수술된 것을 읽게 된다.
    orig = {k: v.clone() if torch.is_tensor(v) else v for k, v in ck["model"].items()}
    model = ck["model"]

    aw = model["actor.0.weight"]
    cw = model["critic.0.weight"]
    if aw.shape[1] != args.old_obs:
        raise SystemExit("actor.0.weight 입력이 {} 인데 --old-obs 는 {}".format(
            aw.shape[1], args.old_obs))
    if cw.shape[1] != args.old_obs + args.old_priv:
        raise SystemExit("critic.0.weight 입력이 {} 인데 old-obs+old-priv 는 {}".format(
            cw.shape[1], args.old_obs + args.old_priv))

    # actor: 옛 54 -> 새 벡터의 앞 54. 나머지는 0.
    model["actor.0.weight"] = expand_linear_in(
        aw, args.new_obs, [(0, 0, args.old_obs)])

    # critic: obs 구간은 제자리, privileged 구간은 new_obs 만큼 오른쪽으로 이동.
    model["critic.0.weight"] = expand_linear_in(
        cw, args.new_obs + new_priv,
        [(0, 0, args.old_obs),                                   # obs
         (args.old_obs, args.new_obs, args.old_priv)])           # privileged (이동)

    # ---- 옵티마이저 상태도 같이 넓힌다 -----------------------------------
    #
    # ⛔ 이걸 빠뜨려서 NA_histzero 가 죽었다(2026-08-07). runner.py:204 가
    # `optimizer.load_state_dict(model_dict["optimizer"])` 를 하는데, Adam 의
    # state 는 **위치 기반**이고 load 시점에 모양을 검사하지 않는다. 그래서
    # 학습이 정상적으로 시작하고 첫 `optimizer.step()` 에서야 터진다:
    #     RuntimeError: tensor a (68) must match tensor b (284)
    # (68 = 옛 critic 입력 54+14, 284 = 새 입력 270+14)
    #
    # 순전파만 검증하면 절대 못 잡는다 -- 그게 내가 낸 실수다. 아래 --verify 는
    # 이제 실제로 optimizer.step() 을 돌린다.
    #
    # 매칭은 인덱스가 아니라 **모양**으로 한다. optimizer.state 의 키는
    # param_groups 의 위치 인덱스인데 그 순서가 named_parameters() 순서와 다르다
    # (실측: state[0] 이 logstd (1,12), state[1] 이 critic.0.weight (256,68)).
    # (256, old_obs) 와 (256, old_obs+old_priv) 는 이 모델에서 유일하므로 모양이
    # 인덱스보다 안전한 식별자다.
    if "optimizer" in ck and isinstance(ck["optimizer"], dict):
        ost = ck["optimizer"].get("state") or {}
        want_actor = (aw.shape[0], args.old_obs)
        want_critic = (cw.shape[0], args.old_obs + args.old_priv)
        hits = {"actor": 0, "critic": 0}
        for _, entry in ost.items():
            for key in ("exp_avg", "exp_avg_sq"):
                t = entry.get(key)
                if not torch.is_tensor(t):
                    continue
                if tuple(t.shape) == want_actor:
                    entry[key] = expand_linear_in(t, args.new_obs,
                                                  [(0, 0, args.old_obs)])
                    hits["actor"] += 1
                elif tuple(t.shape) == want_critic:
                    entry[key] = expand_linear_in(
                        t, args.new_obs + new_priv,
                        [(0, 0, args.old_obs),
                         (args.old_obs, args.new_obs, args.old_priv)])
                    hits["critic"] += 1
        # 0 으로 채우는 것이 Adam 에서 옳다: 한 번도 갱신된 적 없는 파라미터의
        # 1차/2차 모멘트가 정확히 0 이다. 새 열은 실제로 그런 파라미터다.
        print("옵티마이저 상태 확장: actor {}개, critic {}개 텐서".format(
            hits["actor"], hits["critic"]))
        if args.new_obs != args.old_obs and hits["actor"] != 2:
            raise SystemExit("⛔ actor 옵티마이저 상태를 못 찾았다(기대 2개, 발견 {})".format(
                hits["actor"]))
        if hits["critic"] != 2:
            raise SystemExit("⛔ critic 옵티마이저 상태를 못 찾았다(기대 2개, 발견 {})".format(
                hits["critic"]))

    if args.verify:
        sys.path.insert(0, ".")
        from utils.model import ActorCritic
        num_act = int(orig["actor.6.weight"].shape[0])
        old = ActorCritic(num_act, args.old_obs, args.old_priv)
        new = ActorCritic(num_act, args.new_obs, new_priv)
        old.load_state_dict(orig, strict=False)
        new.load_state_dict(model, strict=False)
        old.eval(); new.eval()
        # float64 로 검사한다. 수술은 **대수적 항등**이므로 정확히 0 이 나와야
        # 하는데, float32 에서는 0 이 안 나올 수 있다: critic 의 0 이 아닌 가중치가
        # 입력 벡터의 양 끝(obs 앞부분과 밀려난 privileged)으로 갈라져서 GEMM 의
        # 누산 순서가 달라지고, 4개 층을 지나며 그 반올림이 커진다. actor 는
        # 가중치가 앞쪽에 연속이라 float32 에서도 정확히 0 이 나온다.
        # 즉 float32 의 잔차는 로직이 아니라 결합법칙 문제이고, float64 로 재면
        # 그 둘이 갈린다 -- 이건 추측이 아니라 검사다.
        old = old.double(); new = new.double()
        n = 64
        o_old = torch.randn(n, args.old_obs, dtype=torch.float64)
        # 새 채널에는 아무 값이나 넣는다. 0 을 넣으면 '0 열이라 같다'를 확인하는 게
        # 아니라 '입력이 0이라 같다'를 확인하게 된다 -- 검사가 아무것도 안 한다.
        o_new = torch.cat(
            [o_old, torch.randn(n, args.new_obs - args.old_obs, dtype=torch.float64)], dim=-1)
        with torch.no_grad():
            a_old = old.actor(o_old)
            a_new = new.actor(o_new)
            p_old = torch.randn(n, args.old_priv, dtype=torch.float64)
            p_new = torch.cat(
                [p_old, torch.randn(n, new_priv - args.old_priv, dtype=torch.float64)], dim=-1)
            v_old = old.critic(torch.cat([o_old, p_old], dim=-1))
            v_new = new.critic(torch.cat([o_new, p_new], dim=-1))
        da = (a_old - a_new).abs().max().item()
        dv = (v_old - v_new).abs().max().item()
        # float64 에서 대수적 항등의 잔차는 반올림 단위(~1e-16)에 층수와 폭을 곱한
        # 정도다. 1e-10 은 그보다 한참 위이고 실제 로직 오류(가중치 한 열이라도
        # 잘못 놓이면 O(1) 이 튄다)보다는 한참 아래라, 둘을 확실히 가른다.
        print("검증(float64): actor 최대차 {:.3e}   critic 최대차 {:.3e}".format(da, dv))
        if da > 1e-10 or dv > 1e-10:
            raise SystemExit("⛔ 항등이 깨졌다. 저장하지 않는다.")
        print("검증 통과 -- 새 채널에 난수를 넣어도 출력이 옛 정책과 같다.")

        # ⛔ 순전파 항등만으로는 부족하다. NA_histzero 는 위 검증을 통과하고도
        # 첫 optimizer.step() 에서 죽었다 -- Adam 의 exp_avg 가 옛 모양이었기 때문이다.
        # 그래서 학습이 실제로 하는 것(load -> backward -> step)을 그대로 돌려 본다.
        if "optimizer" in ck:
            m = ActorCritic(num_act, args.new_obs, new_priv)
            m.load_state_dict(model, strict=False)
            opt = torch.optim.Adam(
                [m.logstd] + list(m.critic.parameters()) + list(m.actor.parameters()),
                lr=1e-4)
            try:
                opt.load_state_dict(ck["optimizer"])
            except Exception as exc:
                raise SystemExit("⛔ 옵티마이저 상태를 못 읽는다: {}".format(exc))
            for _ in range(2):
                o = torch.randn(8, args.new_obs)
                p = torch.randn(8, new_priv)
                loss = m.actor(o).square().mean() + m.est_value(o, p).square().mean()
                opt.zero_grad(); loss.backward(); opt.step()
            print("검증 통과 -- optimizer.step() 2회가 실제로 돈다.")

    ck["model"] = model
    torch.save(ck, args.dst)
    print("저장: {}  (obs {} -> {}, priv {} -> {})".format(
        args.dst, args.old_obs, args.new_obs, args.old_priv, new_priv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
