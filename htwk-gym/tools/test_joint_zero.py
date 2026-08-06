"""_resample_joint_zero 가 뽑는 것이 의도한 분포인지 직접 검사한다.

"학습이 돌았다"는 "의도가 코드에 닿았다"가 아니다. 이 저장소에서 그 둘을 섞어서
잃은 시간이 크다 -- swing_apex 를 done 에서 리셋하지 않은 것도, 존재하지 않는
results["commands"] 키를 읽어 가드가 한 번도 발동하지 않은 것도 같은 부류다.
그래서 통계를 실제로 뽑아 본다.

Isaac Gym 없이 돈다: 메서드만 떼어 스텁 객체에 붙인다.

    python tools/test_joint_zero.py
"""

import math
import sys
import types

import torch


# goal_pose.py 를 import 하면 isaacgym 이 딸려 온다. 이 검사는 순수 텐서 연산만
# 쓰므로, 파일에서 메서드 소스만 떼어 실행한다 -- 검사 대상이 실제로 학습이 쓰는
# 그 코드여야 하므로 로직을 복사해 오지는 않는다.
def _load_method(path, name):
    src = open(path, encoding="utf-8").read()
    head = src.index("    def {}(self".format(name))
    tail = src.index("\n    def ", head + 10)
    body = src[head:tail]
    body = "\n".join(line[4:] if line.startswith("    ") else line
                     for line in body.split("\n"))
    ns = {"torch": torch, "math": math}
    exec(compile(body, path, "exec"), ns)
    return ns[name]


NAMES = ["Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
         "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
         "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
         "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll"]
WEIGHT = [1.0, 0.5, 1.0, 1.0, 0.7, 0.7, 1.0, 0.5, 1.0, 1.0, 0.7, 0.7]


def make_stub(modes, n=4096, level=1.0, curriculum=False):
    s = types.SimpleNamespace()
    s.device = "cpu"
    s.num_envs = n
    s.num_dofs = 12
    s.dof_names = NAMES
    s._ZERO_MODES = ("iid", "single", "leg_common", "anti_mirror", "mirror")
    s.joint_encoder_bias = torch.zeros(n, 12)
    s.joint_target_offset = torch.zeros(n, 12)
    s.cfg = {"randomization": {"joint_zero": {
        "enabled": True, "max_deg": 10.0, "curriculum": curriculum,
        "init_level": level, "min_level": 0.0, "step": 0.05,
        "modes": modes, "joint_weight": WEIGHT}}}
    s._resample_joint_zero = types.MethodType(
        _load_method("envs/K1/goal_pose.py", "_resample_joint_zero"), s)
    return s


def check(label, ok, detail=""):
    print("  {}  {}{}".format("✅" if ok else "⛔", label,
                              ("  -- " + detail) if detail else ""))
    return ok


def main():
    allok = True
    ids = torch.arange(4096)
    max_rad = math.radians(10.0)

    print("\n== 1. 상관 수정: target_offset == -encoder_bias (이 설계의 핵심) ==")
    s = make_stub({"iid": 0.2, "single": 0.3, "leg_common": 0.2,
                   "anti_mirror": 0.2, "mirror": 0.1})
    s._resample_joint_zero(ids)
    b, o = s.joint_encoder_bias, s.joint_target_offset
    allok &= check("정확히 부호 반대", torch.equal(o, -b),
                   "최대차 {:.3e}".format((o + b).abs().max().item()))
    allok &= check("실제로 0 이 아니다", b.abs().max().item() > 1e-4,
                   "최대 {:.2f}도".format(math.degrees(b.abs().max().item())))

    print("\n== 2. 모드별 구조 ==")
    s = make_stub({"single": 1.0}); s._resample_joint_zero(ids)
    nz = (s.joint_encoder_bias.abs() > 1e-9).sum(dim=1)
    allok &= check("single: env 마다 정확히 관절 1개", bool((nz == 1).all()),
                   "비영 관절수 분포 {}".format(torch.bincount(nz).tolist()))

    s = make_stub({"leg_common": 1.0}); s._resample_joint_zero(ids)
    b = s.joint_encoder_bias
    L, R = b[:, :6], b[:, 6:]
    one_side = ((L.abs().sum(1) > 1e-9) ^ (R.abs().sum(1) > 1e-9))
    same_sign = torch.where(L.abs().sum(1) > 1e-9,
                            (torch.sign(L) * torch.sign(L[:, :1])).min(1).values,
                            (torch.sign(R) * torch.sign(R[:, :1])).min(1).values)
    allok &= check("leg_common: 한 다리만", bool(one_side.all()))
    allok &= check("leg_common: 그 다리 안에서 부호 동일", bool((same_sign >= 0).all()))

    s = make_stub({"anti_mirror": 1.0}); s._resample_joint_zero(ids)
    b = s.joint_encoder_bias
    # 진폭에 관절 가중치가 곱해지므로 좌우 짝은 가중치까지 나눠야 정확히 반대다.
    w = torch.tensor(WEIGHT)
    nl, nr = b[:, :6] / w[:6], b[:, 6:] / w[6:]
    allok &= check("anti_mirror: 좌우 정확히 반대(가중치 보정 후)",
                   torch.allclose(nl, -nr, atol=1e-6),
                   "최대차 {:.3e}".format((nl + nr).abs().max().item()))

    s = make_stub({"mirror": 1.0}); s._resample_joint_zero(ids)
    b = s.joint_encoder_bias
    nl, nr = b[:, :6] / w[:6], b[:, 6:] / w[6:]
    allok &= check("mirror: 좌우 정확히 같음(가중치 보정 후)",
                   torch.allclose(nl, nr, atol=1e-6))

    print("\n== 3. 진폭이 max_deg 와 관절 가중치를 지키는가 ==")
    s = make_stub({"iid": 1.0}, level=1.0); s._resample_joint_zero(ids)
    b = s.joint_encoder_bias
    lim = max_rad * torch.tensor(WEIGHT)
    allok &= check("모든 관절이 max_deg*weight 이내",
                   bool((b.abs() <= lim + 1e-6).all()),
                   "Hip_Roll 최대 {:.2f}도 (상한 {:.1f}도)".format(
                       math.degrees(b[:, 1].abs().max().item()), 10.0 * 0.5))
    # 가중치가 실제로 작동하는지: Hip_Roll(0.5) 이 Hip_Pitch(1.0) 의 절반이어야 한다
    r = b[:, 1].abs().max().item() / max(b[:, 0].abs().max().item(), 1e-9)
    allok &= check("가중치가 실제로 반영됨 (Hip_Roll/Hip_Pitch ~ 0.5)",
                   0.45 < r < 0.55, "비 {:.3f}".format(r))

    print("\n== 4. per-env 커리큘럼: 살아남으면 올리고 넘어지면 내린다 ==")
    s = make_stub({"iid": 1.0}, n=8, level=0.5, curriculum=True)
    s._terminated_by_fall = torch.tensor([True] * 4 + [False] * 4)
    ids8 = torch.arange(8)
    s._resample_joint_zero(ids8)
    lv = s._zero_level.clone()
    allok &= check("넘어진 env 는 강등", bool((lv[:4] < 0.5).all()),
                   "{:.2f}".format(lv[0].item()))
    allok &= check("버틴 env 는 승급", bool((lv[4:] > 0.5).all()),
                   "{:.2f}".format(lv[4].item()))
    # 상한/하한
    for _ in range(60):
        s._resample_joint_zero(ids8)
    allok &= check("[0,1] 밖으로 안 나간다",
                   bool((s._zero_level >= 0).all() and (s._zero_level <= 1).all()),
                   "범위 [{:.2f}, {:.2f}]".format(
                       s._zero_level.min().item(), s._zero_level.max().item()))

    print("\n== 5. 모드 혼합 비율이 config 를 따르는가 ==")
    s = make_stub({"iid": 0.20, "single": 0.30, "leg_common": 0.20,
                   "anti_mirror": 0.20, "mirror": 0.10}, n=40000)
    s._resample_joint_zero(torch.arange(40000))
    b = s.joint_encoder_bias
    nz = (b.abs() > 1e-9).sum(dim=1)
    share_single = float((nz == 1).float().mean())
    allok &= check("single 비율이 ~0.30", 0.27 < share_single < 0.33,
                   "실측 {:.3f}".format(share_single))

    print("\n" + ("전부 통과" if allok else "⛔ 실패한 검사가 있다"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
