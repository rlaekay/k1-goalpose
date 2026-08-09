# 2026-08-09 후속 정책 짧은 실기 시험

목표는 장기 안정성 인증이 아니라, 교수님 도착 전 **현재 I3b보다 나은 1~3초 보행
후보를 찾는 것**이다. 30초 보행은 요구하지 않는다. 첫 실행에서는 정책만 바꾸고
`Kp/Kd`, action filter, torque mode는 바꾸지 않는다.

## 후보

1. `NP_clockarm/model_6000` — 1순위
   - 현재 로봇 정책 `I3b_stance10/model_200`에서 직접 warm-start.
   - `NH_zeroclock`과 동일한 box-foot 자산, path 0.35, joint-zero DR,
     dwell 중 gait pause에 관절별 armature를 추가.
   - export SHA256:
     `8f0e4cc5ff7bdd0dacba48a9a3a97f7373a1278fdab2c48ab2361d915773377b`
2. `NH_zeroclock/model_6000` — 비교/후퇴 후보
   - NP에서 armature만 뺀 모델. 5-seed sim 보행 결과가 존재한다.
   - export SHA256:
     `862eeada102375f940ee0597d6a0423312885a4d57c2151e55f502fdd9d19e86`
3. 기존 `goal_pose_i3b.pt`는 덮어쓰지 않는다. 언제든 즉시 되돌린다.

## 1. 로봇 연결 후 설치 — Mac

```bash
cd /Users/dmdrb/RoboCup/k1-goalpose

./tools/install_policy.sh --skip-export \
  --checkpoint logs/K1/K1/Goal_Pose_V7/2026-08-08-23-35-12_NP_clockarm/nn/model_6000.pth \
  --name goal_pose_np --config Goal_Pose_E0.yaml

./tools/install_policy.sh --skip-export \
  --checkpoint logs/K1/K1/Goal_Pose_V7/2026-08-08-11-56-11_NH_zeroclock/nn/model_6000.pth \
  --name goal_pose_nh --config Goal_Pose_E0.yaml
```

설치기가 server/robot SHA, actor `(1,54)->(1,12)`, frozen training config와 deploy
config, 실제 22-joint layout을 모두 통과해야 한다. 하나라도 실패하면 지면 시험을 하지
않는다.

## 2. 공통 안전 세팅

- 평평한 바닥, 물리 리모컨/E-stop 담당자 한 명, 로봇을 받을 사람 한 명.
- 사람 손은 로봇에 힘을 주지 않고 바로 옆에서 받을 준비만 한다. 손이 닿으면 그 실행은
  `supported`로 기록하되 안전을 위해 주저하지 않는다.
- 별도 터미널에 아래 명령을 준비한다.

```bash
touch /tmp/e0_abort
```

- 첫 실행에서는 `--rate-fixed-filter`, `--parallel-torque`, Kp/Kd 변경을 모두 사용하지
  않는다. 정책 비교와 구동기 비교를 섞지 않는다.
- `--test-tilt-abort-deg 12`와 자동 시간 제한으로 붕괴 전체를 기다리지 않는다.
  이 test gate는 12도에서 get-up을 시작하지 않고 정상 cleanup의 DAMPING으로 종료한다.

## 3. NP 단계별 시험 — 로봇

각 명령에서 `b`로 CUSTOM 진입 후 `--hold-diag` 결과를 본다. gait 시작 전 tilt가
대부분 3도 안이고 5도를 넘는 표본이 없을 때만 `r`을 누른다.

### A. goal 0, 정책 stand 1초

```bash
cd ~/Workspace/deploy
PYTHONUNBUFFERED=1 ./run_e0.sh fixed 0,0,0 \
  --policy-path ./models/goal_pose_np.pt \
  --hold-prepare --hold-diag 1 \
  --max-policy-seconds 1.0 --test-tilt-abort-deg 12 \
  --log-timing /tmp/np_stand.csv
```

정책이 켜지는 순간 발을 급히 모으거나 한쪽 hip이 계속 이동하면 NP 지면 보행을 중단한다.

### B. 0.2 m, 최대 1.5초

```bash
PYTHONUNBUFFERED=1 ./run_e0.sh fixed 0.2,0,0 \
  --policy-path ./models/goal_pose_np.pt \
  --hold-prepare --hold-diag 1 \
  --max-policy-seconds 1.5 --test-tilt-abort-deg 12 \
  --log-timing /tmp/np_walk_02.csv
```

목표는 1~2번의 지지발 전환이다. 발 교차, 발끝 걸림, 한쪽 hip의 단조 드리프트,
recovery 진입 중 하나라도 보이면 C로 올라가지 않는다.

### C. B가 깨끗할 때만 0.4 m, 최대 2초

```bash
PYTHONUNBUFFERED=1 ./run_e0.sh fixed 0.4,0,0 \
  --policy-path ./models/goal_pose_np.pt \
  --hold-prepare --hold-diag 1 \
  --max-policy-seconds 2.0 --test-tilt-abort-deg 12 \
  --log-timing /tmp/np_walk_04.csv
```

0.6 m는 A~C가 모두 깨끗할 때만 같은 방식으로 최대 2.5초 실행한다. 30초 시험은 하지
않는다.

## 4. NH 비교와 I3b 복귀

NP가 A 또는 B에서 실패하면 같은 단계 하나만 NH로 반복한다. `policy-path`와 로그 이름만
바꾼다.

```bash
PYTHONUNBUFFERED=1 ./run_e0.sh fixed 0.2,0,0 \
  --policy-path ./models/goal_pose_nh.pt \
  --hold-prepare --hold-diag 1 \
  --max-policy-seconds 1.5 --test-tilt-abort-deg 12 \
  --log-timing /tmp/nh_walk_02.csv
```

기존 모델 복귀는 override를 빼면 된다.

```bash
PYTHONUNBUFFERED=1 ./run_e0.sh fixed 0.2,0,0 \
  --hold-prepare --hold-diag 1 \
  --max-policy-seconds 1.5 --test-tilt-abort-deg 12 \
  --log-timing /tmp/i3b_walk_02.csv
```

## 5. 실행 직후 판정

로그를 Mac으로 가져온 뒤 동일 도구로 읽는다.

```bash
python3 htwk-gym/tools/read_walk_log.py <가져온.csv>
```

1차 판정은 다음 네 가지뿐이다.

1. 실제 무릎/hip 진폭이 있어 지지발 전환이 발생했는가.
2. tilt가 12도 watchdog에 닿지 않았는가.
3. 좌우 hip/발 간격이 한 방향으로 누적되지 않았는가.
4. 사람 손이 닿기 전에 얻은 구간이 몇 초·몇 step인가.

`walking=1`이나 gait phase 누적값은 걸음 수가 아니므로 성공 판정에 쓰지 않는다.
