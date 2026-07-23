# K1 Goal-Pose RL Learning Masterplan

## 📋 목표
평지에서 로봇 로컬 프레임 기준 목표 자세 (Δx, Δy, Δθ)를 받아, 
넘어지지 않고 그 자세에 도달해서 정지하는 K1용 관절 위치 정책.

## 🎯 범위 고정
- 목표 샘플링: Δx ∈ [-2, 2] m, Δy ∈ [-1.5, 1.5] m, Δθ ∈ [-π, π]
- 평지, 장애물 없음, 단일 목표
- 에피소드: 4~8초마다 목표 재샘플링

## ✅ 성공 기준 (초기값)
| 지표 | 게이트 |
|---|---|
| 최종 위치 오차 (median / p90) | ≤ 5 cm / ≤ 10 cm |
| 최종 heading 오차 | ≤ 10° |
| 넘어짐률 | 0% |

## 🔧 학습 환경
- Framework: htwk-gym (Isaac Gym Preview 4)
- Robot: K1
- Hardware: A6000 x2
- Server: user-ESC4000A-E12 (git clone at /mnt/DATA/workspace/ws_eungkyu/k1-goalpose,
  htwk-gym이 그 안의 서브디렉토리 — 2026-07-23 변경 이력 참고)

## 🧠 학습 아키텍쳐
- Approach: Warm-start (ParameterWalk) → End-to-end GoalPose
- Network: MLP [512,256,128] + history
- Action: 관절 위치 목표
- Reward: constellation + style/regularization 상속

## 📍 마일스톤
| # | 할 일 | 조건 |
|---|---|---|
| -1 | 서버 환경 구축 (PyTorch/IsaacGym/deps) | `train.py --task=K1/ParameterWalk` 크래시 없이 1 iter |
| 0 | 베이스라인 (ParameterWalk 재현) | 영상 확인 |
| 1 | 평가 하네스 | 오차 분포 숫자 |
| 2 | GoalPose 태스크 골격 | 크래시 없음 |
| 3 | constellation 학습 | 게이트 통과 |
| 4 | export + MuJoCo 검증 | 배포 준비 |

## 📝 변경 이력 (원래 계획 대비 조정)
> 이 섹션은 위 §목표/범위/성공 기준을 바꾸지 않는다. 실행 중 발견한 현실적 제약으로
> 생긴 **전술적 변경**만 기록한다. 새 세션에서 작업을 이어받을 때 이 로그부터 읽을 것.

### 2026-07-23 — 태스크 위치: `tasks/` 아님, `envs/` 사용
- **발견**: htwk-gym은 `tasks/<robot>/<Task>/`가 아니라 `envs/<robot>/<task>.py` +
  `envs/<robot>/<Task>.yaml` 쌍으로 태스크를 정의한다(`utils/runner.py`의
  `get_task_class`가 `envs` 패키지를 스캔). `tasks/K1/GoalPose/`는 빈 디렉토리로 미사용.
- **변경**: GoalPose를 [envs/K1/goal_pose.py](htwk-gym/envs/K1/goal_pose.py) +
  [envs/K1/Goal_Pose.yaml](htwk-gym/envs/K1/Goal_Pose.yaml)로 구현. 클래스를
  [envs/__init__.py](htwk-gym/envs/__init__.py)에 등록.
- (2026-07-23 커밋 `b15eb13`에서 README.md "폴더 구조" 다이어그램을 `envs/` 기준으로 수정 완료.)

### 2026-07-23 — K1 ParameterWalk 소스에 디버그 코드 발견 (학습 불가 상태였음)
- **발견**: `envs/K1/parameter_walk.py`에만 있고 `envs/T1/parameter_walk.py`엔 없는 코드:
  - `_compute_observations()`가 매 스텝 `commands`를 0으로 덮어쓰고 `lin_vel_x=0.5`,
    `gait_frequency=1.9`를 하드코딩 → 리샘플된 command 완전 무시 (전이 학습/RL 자체 불가).
  - `step()`/`_compute_observations()`에 매 스텝 `print(actions)`, `print(obs_buf)`.
  - env0 전용 CSV 로깅(`_init_csv_logging`)이 매 스텝 파일 flush.
- **변경**: GoalPose는 이 디버그 코드를 전부 제거한 클린 베이스에서 시작 (T1 버전
  로직 + K1 로봇 설정으로 재구성). ParameterWalk 원본 파일 자체는 손대지 않음
  (baseline 재현이 milestone 0 과제이므로 별도 정리 필요 — 아직 미착수).
- **부수 발견**: `envs/__init__.py`가 `from envs.T1.parameter_walk import ParameterWalk`를
  K1 import 뒤에 재선언 → `task: "K1/ParameterWalk"`로 설정해도 실제로는 **T1 클래스가 로드됨**.
  GoalPose는 클래스명이 겹치지 않아 무관하지만, milestone 0(K1 베이스라인 재현) 진행 시
  반드시 먼저 고쳐야 함.

### 2026-07-23 — GoalPose 구현 전략: 웜스타트를 위한 command 슬롯 재사용
- **결정**: 관찰 벡터(54차원)·네트워크(`utils/model.py`의 고정 MLP `[256,128,128]`,
  history 없음)를 그대로 두고, 기존 10개 command 슬롯의 **의미만 재정의**해서
  `runner.py`의 `load_state_dict(..., strict=False)`로 ParameterWalk 체크포인트를
  그대로 웜스타트 로드.
  - `cmd[0,1]`: `lin_vel_x,y` → 로봇 로컬 프레임 기준 목표까지 상대 위치 `(Δx,Δy)` (매 스텝 갱신)
  - `cmd[2]`: `ang_vel_yaw` → heading error (목표 yaw − 현재 yaw, wrap, 매 스텝 갱신)
  - `cmd[3..9]`: gait_frequency + 스타일(foot_yaw/pitch/roll/offset) 그대로 유지,
    단 기본은 yaml에서 min=max로 고정(중립 자세) — 리샘플 코드는 원본 그대로 두어
    milestone 3에서 다시 다양화 가능.
- **MASTERPLAN 원안과의 차이**: `MASTERPLAN.md`의 네트워크 항목("MLP [512,256,128] +
  history")은 실제 `model.py`와 다름. 아키텍처를 지금 바꾸면 웜스타트가 깨지므로,
  **milestone 2/3은 현재의 소형 MLP·no-history로 진행**한다. `[512,256,128]+history`로
  바꾸려면 처음부터 재학습이 필요한 별도 작업(추후 결정).
- heading error는 단일 wrap 각도로 시작(웜스타트 차원 유지 우선). ±π 근처 학습이
  잘 안 되면 sin/cos 2채널로 업그레이드(관찰 차원 55로 변경, 이 경우 웜스타트 1층은
  재초기화 필요) — 아직 미착수, 필요시에만 적용.
- `feet_offset_x/y` 보상이 원래 `commands[:,0]/[:,1]`(속도)로 스케일링했는데, 이제 그
  자리엔 위치 목표가 들어감. 실제 속도(`filtered_lin_vel`) 기반으로 스케일링하도록 수정,
  기본 scale은 0(비활성)으로 시작 — milestone 3에서 필요시 켠다.

### 2026-07-23 — 목표 재샘플링: 4~8초 주기 (§범위 고정과 일치)
- ParameterWalk의 `cmd_resample_time`/`resampling_time_s` 메커니즘을 그대로 재사용해
  4~8초마다 새 목표를 로봇 **현재 위치·자세 기준**으로 재샘플링. 별도 구현 불필요.

### 2026-07-23 — 코드 동기화 방식: rsync(SYNC.sh) → git
- **변경 이유**: 애초 계획은 scp/rsync로 로컬↔서버를 직접 동기화하는 것이었으나,
  GitHub 저장소(`rlaekay/k1-goalpose`)를 만들어 쓰는 쪽으로 방향 전환.
- **변경**: `SYNC.sh`(push+pull 겸용) 삭제. 코드는 로컬에서 `git push` → 서버에서
  `git pull`. 체크포인트/로그/영상처럼 git에 안 맞는 큰 바이너리만 [PULL.sh](PULL.sh)
  (구 SYNC.sh의 pull 부분만 남긴 버전)로 rsync 회수. 자세한 명령은 README.md
  "코드 동기화 (git)" 절 참고.
- **주의 (미착수)**: 서버의 기존 경로(`/mnt/DATA/workspace/ws_eungkyu/htwk-gym/`, 사용자가
  공유한 진단 결과 기준)는 예전 rsync push로 만들어진 구조(`htwk-gym/htwk-gym/` 이중 중첩
  + 빈 `tasks/`)라서 새 git 워크플로우와 안 맞는다. 서버에서 새로 `git clone`해서
  `/mnt/DATA/workspace/ws_eungkyu/k1-goalpose/`로 옮기는 걸 권장(README 절차 참고).
  기존 디렉토리를 그대로 재사용하려면 먼저 정리 필요 — milestone -1에 포함.
