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

## 🤝 공유 서버 운영 원칙 (2026-07-23부터, 항상 적용)
서버(`user-ESC4000A-E12`)는 여러 사람이 같이 쓰는 장비다. 아래는 매번 학습을 돌릴 때마다
지켜야 하는 상시 규칙 — 아래 §변경 이력의 "왜 이렇게 됐는지"와 달리, 이건 **매번 반복 적용**한다.
- **GPU**: 학습 시작 전 항상 `nvidia-smi`로 어느 GPU가 비어있는지 확인하고, 그 번호를
  `--sim_device`/`--rl_device`에 명시적으로 지정한다. 두 GPU를 동시에 잡지 않는다.
- **중간에 다른 프로세스가 나타나면 우리가 양보한다**: 실행 중 다른 사용자(또는 root/공용
  서비스)의 프로세스가 같은 GPU에 새로 붙으면, 누가 "먼저"인지 `ps -p <PID> -o user,lstart,cmd`로
  확인하고, 우리가 나중이면 우리 학습을 `kill -SIGINT <PID>`(체크포인트 로직 안 깨지는 안전 종료)로
  멈춘 뒤 비어있는 다른 GPU로 옮긴다. (2026-07-23: root 소유 Isaac Lab 프로세스와 GPU 0에서
  충돌 → 우리 쪽을 GPU 1로 이동해서 해결한 실제 사례 있음.)
- **디스크**: `$HOME`(루트 파티션, 서버 전체가 공유)은 절대 안 씀. conda·pip 캐시·저장소·
  IsaacGym·체크포인트 전부 `/mnt/DATA/workspace/ws_eungkyu/` 안에만 둔다.
- **삭제는 항상 본인 소유 파일에 한정**: 다른 사용자의 `ws_*` 디렉토리는 목록만 확인(`du -sh`
  등)하고 내용을 열거나 지우지 않는다.

## 🧠 학습 아키텍쳐
- Approach: Warm-start (ParameterWalk) → End-to-end GoalPose
- Network: MLP [512,256,128] + history
- Action: 관절 위치 목표
- Reward: constellation + style/regularization 상속

## 📍 마일스톤
| # | 할 일 | 조건 |
|---|---|---|
| -1 | ✅ 서버 환경 구축 (PyTorch/IsaacGym/deps) | 2026-07-23 완료 (아래 변경 이력) |
| 0 | ⬜ 베이스라인 (ParameterWalk 재현) | 영상 확인 — **미착수**, 아래 참고 |
| 1 | ✅ 평가 하네스 | 코드 완성 2026-07-23 (`eval_goal_pose.py` + `tools/auto_stop.py`) — 실측은 학습 종료 후 |
| 2 | ✅ GoalPose 태스크 골격 | 2026-07-23 완료: 크래시 없음 + 512env/200iter 스케일 검증 통과 |
| 3 | 🔶 constellation 학습 | 게이트 통과 — **진행 중** (아래 참고) |
| 4 | ⬜ export + MuJoCo 검증 | 배포 준비 — 미착수 |

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

### 2026-07-23 — milestone -1 완료: 서버 환경 구축 실기(實記)
- **서버가 다인 공유 환경**이라는 게 확인됨 (`/mnt/DATA/workspace/`에 `ws_hojun`(422G),
  `ws_minho`(1.2T), `ws_wonhyuk`(3.3T) 등 여러 사용자 워크스페이스가 quota 없이 공존,
  `/mnt/DATA` 자체가 7.0T 중 95% 사용 중). 이후 모든 설치를 아래 원칙으로 진행:
  - `$HOME`(`/dev/nvme0n1p2`, `/` 마운트, 879G 중 88% 사용, 전 사용자 공유)은 건드리지 않음
    — 여기가 차면 서버 전체(로그인/시스템 서비스)가 죽어 전원에게 피해가 감.
  - 대신 **`/mnt/DATA/workspace/ws_eungkyu/`에만** conda(Miniconda3)·pip 캐시·저장소·IsaacGym을
    전부 설치 (`ws_eungkyu`가 quota 없는 공용 풀에서 본인 몫으로 이미 관행적으로 쓰이던 영역).
  - GPU도 두 장 중 `cuda:0` 한 장만 `--sim_device`/`--rl_device`로 명시 지정, 다른 사용자가
    쓸 수도 있는 GPU 1은 건드리지 않음.
- **실제 설치 순서** (재현 시 참고):
  1. `/mnt/DATA/workspace/ws_eungkyu/miniconda3`에 Miniconda 설치 (`-b -p`로 무인 설치).
     `defaults` 채널은 Anaconda 이용약관 동의가 필요해서 걸리므로,
     `conda create -c conda-forge --override-channels python=3.8`로 conda-forge만 사용.
     (IsaacGym Preview4의 prebuilt 바인딩이 Python 3.8까지만 있어서 3.8 고정 필요 — 시스템
     python은 3.10이라 그대로는 못 씀.)
  2. `pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118`
     (PyTorch 공식 cu118 인덱스). `torch.cuda.is_available() == True` 확인.
  3. IsaacGym Preview4는 NVIDIA 개발자 계정 로그인 필요라 자동화 불가 — 로컬에서 다운로드 후
     `scp`로 `/mnt/DATA/workspace/ws_eungkyu/`에 업로드, `tar -xvf`(압축 안 됐으면 `-z` 빼도 됨)
     후 `isaacgym/python`에서 `pip install -e .`.
  4. **알려진 이슈 2개**:
     - `ImportError: libpython3.8.so.1.0: cannot open shared object file` — conda 환경의
       lib 경로가 `LD_LIBRARY_PATH`에 없어서 발생. 해결: env 전용 활성화 훅
       (`$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh`)에
       `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` 추가 (전역 `.bashrc`
       대신 env 스코프로 — 다른 conda env에 영향 안 주려고).
     - `AttributeError: module 'numpy' has no attribute 'float'` — IsaacGym 코드
       (`torch_utils.py`)가 numpy 1.24에서 제거된 `np.float` 별칭을 씀. `pip install "numpy<1.24"`
       (1.23.5로 고정)로 해결. **주의**: `pip install -r requirements.txt`를 numpy 고정 이후에
       돌리면 다시 numpy가 최신으로 끌려 올라갈 수 있으니, requirements 설치 후엔 항상
       `numpy` 버전을 재확인할 것.
  5. `train.py`의 `--task` 인자는 **클래스 이름이 아니라 yaml 파일명**을 기준으로
     `envs/{task}.yaml` 경로를 만든다 (`utils/runner.py`). GoalPose는 파일명이
     `Goal_Pose.yaml`이므로 `--task=K1/Goal_Pose`로 호출해야 함 (`K1/GoalPose`는
     `FileNotFoundError`). `--headless`는 `argparse type=bool`이라 `--headless True`처럼
     값을 반드시 명시해야 함 (플래그만 주면 에러).
  6. 검증: `python train.py --task=K1/Goal_Pose --headless True --num_envs 4 --sim_device cuda:0
     --rl_device cuda:0 --max_iterations 5` — 5 iteration 크래시 없이 완료 (milestone 2 통과).
- **정정**: `basic.task`는 `train.py --task` 인자 값으로 그대로 덮어써지므로(런타임에
  `K1/Goal_Pose`), 실제 로그 경로는 `logs/K1/K1/Goal_Pose/<timestamp>/summaries` (yaml
  안에 적어둔 `task: "K1/GoalPose"` 값이 아니라 CLI에서 준 언더스코어 버전으로 남음).
- 512 env, 200 iteration까지 확장 실행 완료 — `reward` 0.01→0.27로 꾸준히 상승,
  `value_loss`/`actor_loss` 안정, `kl_mean`이 `desired_kl` 근처 유지, `entropy` 완만히
  감소. 발산/NaN 없음 (milestone 2 스케일 검증 통과).

### 2026-07-23 — 웜스타트 체크포인트 확보: deploy 모델은 actor만 있음
- **발견**: `deploy/models/parameter_walk.pt`는 학습 체크포인트가 아니라
  `export_model.py`가 `torch.jit.script(model.actor)`로 만든 **actor(정책)만 담긴
  배포용 스크립트 모듈**. `utils/runner.py`가 `--checkpoint`로 기대하는
  `{model, optimizer, curriculum}` state_dict 딕셔너리 포맷이 아니고, critic(가치망)
  가중치도 없음. `torch.load(...)["model"]`로 바로 못 씀.
- **대응**: [seed_warmstart_checkpoint.py](htwk-gym/seed_warmstart_checkpoint.py) 추가 —
  스크립트된 actor의 state_dict를 새 `ActorCritic.actor`에 그대로 로드(레이어 이름/shape가
  `utils/model.py`와 다르면 `strict=True`라 바로 에러로 드러남), critic은 랜덤 초기화,
  `{"model": ...}`만 있는 체크포인트로 저장. `optimizer`/`curriculum` 키가 없어도
  `runner.py._load()`가 이미 try/except로 넘어가게 되어 있어 문제 없음.
  ```bash
  python seed_warmstart_checkpoint.py --task K1/Goal_Pose \
    --source deploy/models/parameter_walk.pt \
    --out logs/warmstart/parameter_walk_actor_seed.pth
  ```
  이후 `train.py --checkpoint logs/warmstart/parameter_walk_actor_seed.pth`로 웜스타트.
- **한계**: critic은 진짜 웜스타트가 아니라 랜덤 초기화라, 학습 초반 value 추정이 부정확할
  수 있음 — actor만 웜스타트해도 처음부터 학습하는 것보다는 유리할 것으로 기대하지만,
  진짜 풀 체크포인트(critic 포함)를 나중에 서버 학습 로그에서 구하면 그걸로 교체 권장.

### 2026-07-23 — GPU 0에서 다른 사용자와 충돌 → GPU 1로 이동, 원칙화
- **발생**: `--num_envs 2048`로 GPU 0에서 실제 장기 학습 시작 직후, GPU 0 사용률이 96%까지
  치솟음. 확인해보니 `root` 소유의 Isaac Lab/Sim 프로세스(`/workspace/isaaclab/...`,
  우리보다 먼저 시작)가 같은 GPU에 붙어있었음 — 우리 세션이 설치/실행한 것과는 무관한
  별개 작업(경로·소유자·제품 자체가 다름: Isaac Lab/Sim vs 우리의 Isaac Gym Preview4).
- **대응**: 우리 학습 프로세스를 `kill -SIGINT`로 안전 종료 후, 완전히 비어있던 GPU 1로
  재시작(`--sim_device cuda:1 --rl_device cuda:1`). 이후 GPU 0(root 프로세스만)/GPU 1
  (우리 프로세스만)로 깔끔히 분리됨.
- **원칙화**: 위 "🤝 공유 서버 운영 원칙" 섹션에 상시 규칙으로 등록 (GPU 점유는 매번 재확인,
  충돌 시 우리가 양보).
- **현재 상태**: GPU 1에서 `--checkpoint logs/warmstart/parameter_walk_actor_seed.pth
  --num_envs 2048 --max_iterations 20000` 장기 학습 진행 중 (tmux 세션, 시작 직후 확인 시점).
  milestone 3("constellation 학습 — 게이트 통과")에 해당하는 실제 학습 단계.
- **milestone 순서 이탈 기록**: 원래 순서는 0(베이스라인 재현)→1(평가 하네스)→2→3이었으나,
  실제로는 milestone 0/1을 건너뛰고 2→3으로 바로 진행 중. 이유: milestone 0의 목적("웜스타트용
  ParameterWalk 정책 확보")을 직접 재학습하는 대신 기존 배포 모델(`deploy/models/parameter_walk.pt`)의
  actor 가중치로 대체했기 때문에 별도 재현이 당장 필수는 아니었음. **단, milestone 1(평가
  하네스)은 아직 없어서, 지금 진행 중인 milestone 3 학습이 실제로 §성공 기준 게이트(위치오차
  5cm/10cm, heading 10°, 낙상률 0%)를 통과했는지 숫자로 확인할 방법이 없음** — 장기 학습이
  끝나기 전에(또는 병행해서) milestone 1을 만들어야 함.

### 2026-07-23 — 코드/보상 출처 명확화 (Q&A 기록)
새 세션·협업자가 오해하지 않도록 출처를 명시한다:
- **베이스 코드**: htwk-gym의 K1 ParameterWalk(속도+파라미터 명령 걷기)가 유일한 코드 출처.
  htwk-gym 자체는 Booster Gym 계열이다(T1 = Booster T1 로봇, `deploy/README.md`가 Booster
  Robotics SDK 설치를 안내) — 즉 Booster의 학습 프레임워크는 이미 사용 중.
- **GoalPose 보상(goal_position/goal_heading/goal_stop 및 모듈 대안들)**: 특정 논문("GoTo" 등)에서
  가져온 것이 **아님**. 이 세션에서 ParameterWalk의 기존 추종 보상과 같은 지수 커널 형태로 직접
  작성. (형태 자체는 목표 지향 보행 RL의 표준 패턴이지만 어떤 논문도 참조·복제하지 않음.)
  논문 기반 보상 세트 도입은 별도 조사 작업으로 미착수.
- **`max_iterations: 20000`**: 설계된 실험 길이가 아니라 ParameterWalk 설정에서 물려받은 상한값.
  실제 종료 시점은 수렴 판정(아래 auto_stop)으로 결정한다.

### 2026-07-23 — milestone 1 완성: 평가 하네스 + 수렴 자동 정지 (모듈형)
목표("게이트 통과 여부를 숫자로")를 위해 추가한 것들 — 기존 동작은 전부 보존, 전부 끼우고 뺄 수 있음:
- **[eval_goal_pose.py](htwk-gym/eval_goal_pose.py)**: 체크포인트를 로드해 headless로 굴리면서
  목표 구간(4~8초 타이머)마다 **교체 직전 목표 기준의 정확한 최종 위치/heading 오차·정지 속도**를
  실측. 낙상(타임아웃 아닌 종료)을 별도 집계. 게이트(§성공 기준) PASS/FAIL 표 + 미달 시 "다음에
  시도할 것" 제안까지 담긴 `report.md`/`report.json`/`segments.csv`를 run 디렉토리 `eval/` 아래 저장.
  기본은 결정론적 정책(dist.loc)·외란(kick/push) OFF, `--stochastic`/`--keep_perturbations`/
  `--no_noise`/`--record_video`(env0 mp4)로 조건 전환.
- **[tools/auto_stop.py](htwk-gym/tools/auto_stop.py)**: 학습 옆에서(tmux 별창) TensorBoard 스칼라를
  주기 폴링 → 최근 window 평균이 직전 window 대비 상대 개선 `rel_eps`(기본 2%) 미만이 `patience`회
  연속이면 **run 디렉토리에 STOP 파일 생성** → 학습이 체크포인트 저장 후 스스로 종료 → 디렉토리가
  잠잠해지면 eval을 자동 실행해 리포트 생성. (학습이 이미 죽었/끝났으면 `--stale_min` 후 바로 eval.)
- **[utils/runner.py](htwk-gym/utils/runner.py)**: train 루프에 STOP 파일 체크 추가(매 iteration 끝).
  파일 없으면 기존과 100% 동일 동작.
- **[envs/K1/goal_pose.py](htwk-gym/envs/K1/goal_pose.py) 모듈 보상 3종 추가 (기본 scale 0 = 비활성)**:
  | 보상 | 무엇 | 언제 켜나 |
  |---|---|---|
  | `goal_progress` | 거리 줄인 속도 [m/s], potential-based | 원거리에서 exp 보상이 평평해 유인이 약할 때 |
  | `goal_reached` | 목표 반경 안 + 정지 시 +1/step (희소, 진짜 성공 조건) | 도착 후 "머무르기" 강화 필요할 때 |
  | `heading_near_goal` | 목표 근처에서만 heading 요구 (거리 게이트) | 걷는 중 heading 강제가 보행을 방해할 때 (`goal_heading` 0으로 끄고 교체) |
  스위치는 전부 [Goal_Pose.yaml](htwk-gym/envs/K1/Goal_Pose.yaml) `rewards.scales`에서만 조정 —
  코드 수정 없이 조합 실험 가능. 평가 게이트 기준치도 yaml `evaluation:` 섹션으로 이동.
- **[seed_warmstart_checkpoint.py](htwk-gym/seed_warmstart_checkpoint.py) 확장**: 소스가 jit actor
  export든 풀 학습 체크포인트(critic 포함)든 자동 감지 — critic 포함 웜스타트 경로 마련.
- **[envs/base_task.py](htwk-gym/envs/base_task.py) 버그 수정**: record_video가 K1처럼 단일 액터
  (2-D root_states) 태스크에서 크래시하던 인덱싱을 차원 감지로 수정 → headless 영상 녹화 가능.
- **주의**: 지금 서버에서 도는 학습은 STOP 훅이 없는 이전 코드다. auto_stop을 쓰려면
  `git pull` 후 `--checkpoint -1`(최신 체크포인트 자동 탐색)로 재시작해야 함. 재시작 없이
  두려면: 그대로 두고 나중에 수동으로 `eval_goal_pose.py --checkpoint -1`만 돌려도 된다.

### 2026-07-23 — 원 논문 확인: "No More Marching" (arXiv:2508.14098) 대조 결과
- **확인된 사실**: §범위 고정의 목표 범위(Δx∈[-2,2], Δy∈[-1.5,1.5], Δθ∈[-π,π])와 §학습
  아키텍쳐의 "Reward: constellation + style/regularization 상속"은 논문
  *"No More Marching: Learning Humanoid Locomotion for Short-Range SE(2) Targets"*
  (Dugar et al., arXiv:2508.14098)의 설정과 일치 — 마스터플랜의 원 출처로 판단.
  htwk-gym이 Booster Gym(arXiv:2506.15132) 기반이라는 것도 htwk-gym GitHub 설명에서 공식 확인.
- **논문의 constellation reward (정확한 정의)**: 베이스 프레임에 반지름 r=1m 원형으로 고정한
  점들 vs 목표 자세의 같은 점들의 평균제곱거리 `d_con = ‖Δc‖² + I_c·θ²` (I_c=r²),
  보상은 **단일 커널** `r_con = exp(-0.2·d_con)`. 총보상 = r_con + style + regularization.
  gait clock **없음**(행진 제거가 논문의 핵심), 에피소드 8초, 목표 유형 혼합
  (stand 0.1 / straight 0.2 / lateral 0.2 / turn 0.2 / combined 0.3), 4096 envs, curriculum 없음.
- **우리 v0(현재 학습 중)과의 차이 = 미구현이었음을 확인**:
  1. v0은 위치/방향을 **덧셈 분리**(goal_position + goal_heading) — 논문은 곱 결합 단일 커널.
     덧셈은 "멀리서 방향만 맞추고 보상 파밍"이 가능, 논문 방식은 둘 다 좋아야 보상이 큼.
  2. v0은 **gait clock + feet_swing(+3.0)이 항상 활성** (gait_frequency ∈ [1.8,2.0]로 0이 될 일
     없음) → 목표 도착 후에도 제자리 행진이 보상됨. goal_stop(-1)과 정면 충돌하는 구조적 결함.
  3. v0은 에피소드 30초/목표 유형 혼합 없음 — 논문은 8초/혼합 샘플링.
- **조치 (커밋 참조)**: 전부 스위치로 추가, 기본은 v0 유지(돌던 학습과 호환):
  - `_reward_constellation` — 논문 충실 구현. 원형 constellation의 정확한 기하로
    `d_con = d² + 2r²(1-cosθ)` 사용 (논문의 I_c·θ²는 소각 근사; 1-cos 형태가 ±π에서 매끄러움).
    yaml에서 `constellation: ~3.5` + `goal_position/goal_heading: 0`으로 교체 장착.
  - `goal_categories` 샘플링 — yaml `commands.goal_categories.enabled: true`로 논문 혼합 활성화.
    stand 목표는 gait_frequency=0으로 feet_swing 행진 유인 제거 (ParameterWalk still env 방식 재사용).
  - gait clock 완전 제거 실험은 코드 수정 없이 `gait_frequency: [0., 0.]`로 가능 (관찰 차원 유지;
    단 웜스타트한 ParameterWalk actor가 clock 의존이라 보행 자체가 무너질 수 있음 — 실험으로 판단).
  - 에피소드 8초는 `episode_length_s: 8.`로 가능.
- **권장 v1 설정** (사용자 취사선택 대기): constellation 3.5 / goal_position·goal_heading·goal_stop 0 /
  goal_categories enabled / episode_length_s 8 / 스타일·정규화 보상은 현행 유지(논문의 r_sty+r_reg 대응).

### 2026-07-23 — 전체 로드맵 현황 (최종 목표까지)
**최종 목표**: K1이 (Δx,Δy,Δθ) 목표에 게이트(5cm/10cm/10°/낙상0) 수준으로 도달·정지하는
정책을 export해 MuJoCo 검증(→실기 배포 준비)까지.
1. ✅ 걷기 확보 — Booster/htwk ParameterWalk의 배포 actor를 웜스타트 시드로 사용 중
2. 🔶 v0 학습 (지금 GPU 1에서 진행 중) — 논문과 다른 단순 덧셈 보상 + gait clock 상시 활성.
   **역할 재정의: 게이트 통과용이 아니라 베이스라인 수치 확보용.** 곧 중단하고 평가 예정.
3. ⬜ v0 평가 — eval_goal_pose.py로 위치/heading 오차·정지속도·낙상 실측 → v1과 비교할 기준점
4. ⬜ v1 학습 — constellation + goal_categories로 전환(위 권장 설정), auto_stop으로 수렴 시 자동
   정지+평가. 게이트 미달 시 리포트의 제안(스위치 조정)대로 반복
5. ⬜ 게이트 통과 → milestone 4: export_model.py로 actor export + MuJoCo에서 sim-to-sim 검증
6. (선택) critic 포함 풀 웜스타트, sin/cos heading, 네트워크 확장 — 필요 시에만
