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
  `--sim_device`/`--rl_device`에 명시적으로 지정한다. 기본은 두 GPU를 동시에 잡지 않는다.
  단, 사용자가 그날의 점유 상황을 확인해 명시적으로 허용한 경우에만 빈 GPU를 평가/독립 실험에
  추가 사용한다(2026-07-24에는 GPU 0 사용 허용; 아래 기록 참고).
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
- Network: actor MLP [256,128,128], feed-forward(no history); critic MLP [256,256,128]
- Action: 관절 위치 목표
- Reward: constellation + style/regularization 상속

## 📍 마일스톤
| # | 할 일 | 조건 |
|---|---|---|
| -1 | ✅ 서버 환경 구축 (PyTorch/IsaacGym/deps) | 2026-07-23 완료 (아래 변경 이력) |
| 0 | ⬜ 베이스라인 (ParameterWalk 재현) | 영상 확인 — **미착수**, 아래 참고 |
| 1 | ✅ 평가 하네스 | 코드+v0/v1 실측 완료; wall-clock/3-arm preview 지원 |
| 2 | ✅ GoalPose 태스크 골격 | 2026-07-23 완료: 크래시 없음 + 512env/200iter 스케일 검증 통과 |
| 3 | 🔶 constellation 학습 | v1@20000 후 3-arm sweep 진행 중; 게이트 미통과 |
| 4 | ⬜ export + MuJoCo 검증 | 배포 준비 — 미착수 |
| 5 | ⬜ 상태추정 + BT/RLkick 통합 | Δpose estimator, goal conditioner, 실기 handoff 검증 |

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

### 2026-07-23 — 외부 리뷰 5건 판정 (코드·설정 사실 대조)
- **① 웜스타트 입력 스케일 불일치 → 반박 (이미 해결돼 있음)**: Goal_Pose.yaml
  `normalization.goal_pos: 0.5`, `goal_heading: 1/π`가 처음부터 들어가 있어 관찰값 기준
  Δx ±2m→±1.0, Δy ±1.5m→±0.75, heading ±π→±1.0. 원래 슬롯의 lin_vel ±1.0(scale 1.0),
  ang_vel_yaw ±1.6(scale 1.0)과 같은 범위다. "속도 0.5 vs 위치 2.0의 4배 차이"는 관찰
  벡터에 존재하지 않음. 남는 것은 시간 분포 차이(구간 상수 명령 vs 연속 감소 오차)뿐이고
  이는 슬롯 재사용 웜스타트에 내재된 것 — fine-tuning이 흡수할 몫. 조치 불필요.
- **② parameter_walk.pt가 K1용인지 미확인 → 수용 + 원지적보다 심각**: 사실 확인 결과
  (a) T1/K1 ParameterWalk의 네트워크 차원이 **완전히 동일**(54/14/12) → strict 로드 성공은
  로봇 구분의 증거가 **전혀 아님**; (b) deploy 폴더는 T1 지향 정황: `deploy_parameter_walk.py`가
  Booster SDK `B1JointCnt`(23관절) 기준이고 deploy config의 stiffness/qpos 배열도 23개(팔 포함),
  `policy_path`는 `thomas_walk_4.pt`를 가리킴; (c) `export_model.py`는 존재하지 않는
  `utils/model_thomas.py`를 import → 현 코드로는 export가 깨져 있어 `parameter_walk.pt`는
  이전/다른 코드 상태에서 만들어진 산출물(출처 불명). **검증 방법(코드 추가 불필요)**:
  시드 체크포인트를 zero-shot 평가 —
  `python eval_goal_pose.py --task K1/Goal_Pose --checkpoint logs/warmstart/parameter_walk_actor_seed.pth --sim_device cuda:1 --rl_device cuda:1`
  → 낙상률이 낮고 목표 방향으로 이동하면 K1 보행 정책 맞음; 즉시 넘어지면 T1산(웜스타트 무효,
  from-scratch가 기준선). v0 평가 직후 바로 실행할 것. 부기: `__init__.py` K1→T1 shadowing
  버그는 우리가 ParameterWalk 학습을 한 번도 안 돌렸으므로 **어떤 실행에도 영향 준 적 없음**.
- **③ 낙상 0% 게이트의 통계적 정의 → 수용**: 게이트를 프로토콜 상대 정의로 확정 —
  "표준 평가 프로토콜(결정론적 정책, 외란 OFF, 256 env × 120 s ≈ 4~5천 구간)에서 낙상 0회".
  0/5000이면 rule of three로 실제 낙상률 95% 상한 ≈ 0.06%. 외란 ON(`--keep_perturbations`)
  낙상률은 게이트가 아니라 별도 강건성 지표로 본다. (eval 기본값이 이미 이 프로토콜.)
- **④ heading sin/cos를 v1에서 선제 전환 → 근거 기반 반박, 단일 각도 유지**:
  참조 논문(No More Marching) 자체가 전 범위 Δθ∈[-π,π]를 **원시 각도 그대로** 관찰에 넣고
  실기 전이까지 보고함 — 같은 태스크에서 단일 wrap 각도가 검증된 설계. ±π에서 좌/우회전이
  동등한 모호성은 인코딩과 무관한 태스크 본질. 또한 "나중 전환 = 차원 파괴(웜스타트 손실)"
  전제가 성립 안 함: v1에서 상시 0인 스타일 슬롯(4 또는 5)에 (cosθ−1)을, 슬롯 2에 sinθ를
  넣으면 **54차원 유지한 채** sin/cos 전환 가능(θ=0일 때 두 값 모두 0 = 기존 중립값과 일치).
  v1은 논문대로 단일 각도로 가고, 회전 방향 진동(dithering)이 관측되면 위 방법으로 전환.
- **⑤ 도메인 랜덤화 문서 부재 → 문서 공백은 수용, 실체 부재는 반박**: Goal_Pose.yaml이
  ParameterWalk의 DR 전체를 이미 상속 중 — 마찰 [0.1,2.0], restitution/compliance,
  base 질량 ×[0.8,1.2], base CoM ±10cm, 타 링크 질량/CoM, PD 게인 ×[0.95,1.05],
  관절 마찰 [0,2], 관측 노이즈(gravity/ang_vel/dof_pos/dof_vel), 액션 지연 0~10 sim step
  (delay_steps), kick 4s/push 5s 주기(정지 자세에도 가해짐). **milestone 4 사전 체크리스트에
  추가**: MuJoCo 검증 전 (i) DR 범위가 K1 실기 스펙과 맞는지 대조, (ii) 목표 도달 정지 상태에서
  push 강건성 별도 평가(외란 ON eval), (iii) export 경로 수리(`utils/model_thomas.py` 부재로
  현재 export_model.py 깨짐 — 어차피 milestone 4에서 필수 수리).

### 2026-07-23 — v1 설정 확정 (사용자 취사선택)
- 겹치는 모듈 4개 결정에 대한 사용자 선택:
  1. **목표 보상: constellation 단독** (3.5) — goal_position/goal_heading/goal_stop은 0으로 보존
     (스위치만 내리면 v0 복원 가능). 보조 보상(goal_progress/goal_reached/heading_near_goal)은
     v1 결과 보고 필요 시에만.
  2. **목표 샘플링: 논문 혼합** — goal_categories.enabled: true (stand 10% 포함)
  3. **gait clock: 유지 + stand 목표만 0** — 웜스타트 보행 보존 절충안
  4. **에피소드: 30초 유지** (논문의 8초 대신 — 연속 목표 전환 단련 우선)
- v0 평가 중간 확인: 4,669 구간 / 낙상 33회(≈0.7%, 외란 OFF) → **낙상 게이트 FAIL 확정**.
  위치/heading 수치는 report.json (서버 logs/.../2026-07-23-18-36-53/eval/) 참조.
- v1 웜스타트 소스: v0의 model_3400.pth 권장 (이미 K1+GoalPose에 적응된 정책 —
  출처 불명 시드보다 확실). 시드 zero-shot 검증(리뷰 ② 판정)은 별도로 계속 유효.

### 2026-07-23 — v0 기준선 + 시드 검증 결과 (리뷰 ② 종결)
표준 평가 프로토콜(결정론적, 외란 OFF, 256env×120s) 실측:
| 지표 | 시드 zero-shot | v0 (model_3400) | 게이트 |
|---|---|---|---|
| 위치 median / p90 | 30.2 / 45.3 cm | 12.8 / 17.9 cm | 5/10cm ❌ |
| heading median | 12.1° | 1.7° | 10° ✅ |
| 낙상 | 1/4633 (0.02%) | 33/4702 (0.7%) | 0 ❌ |
| 도착 시 속도 median | 0.12 m/s | 0.117 m/s | (정지 기준 0.1) |
- **리뷰 ② 종결**: 시드가 K1에서 낙상 0.02%로 안정 보행 → parameter_walk.pt는 K1산 확정,
  웜스타트 전제 유효. 슬롯 재사용 덕에 zero-shot으로도 30cm/12°까지 접근(비례 제어 창발).
- **v0 진단**: heading은 해결(1.7°). 병목은 (a) 위치 ~13cm 정체 + 도착 속도 0.117 m/s
  = 목표 근처에서 완전 정지 못 하고 잔걸음 유지 (gait clock+feet_swing 구조 문제 실증),
  (b) 낙상이 시드 대비 30배 악화 = v0 보상이 접근을 공격적으로 만들어 안정성 희생.
- **v1 기대/관찰 포인트**: constellation+stand 혼합이 (a)를 겨냥. v1 평가에서
  final_speed_mps가 0.1 아래로 내려오는지가 행진 해소의 직접 지표. 비-stand 목표
  도착 후 행진이 남으면(사용자 선택 절충안의 알려진 한계) 다음 레버는 goal_reached
  보너스 또는 gait clock 완전 제거.

### 2026-07-24 — v1 첫 결과 (2165 iter, auto_stop 조기정지) vs v0
| 지표 | v0 (3400 iter) | v1 (2165 iter) | 게이트 |
|---|---|---|---|
| 위치 median/p90 | 12.8/17.9 cm | 12.7/20.1 cm | 5/10cm ❌ |
| heading median | 1.7° | 1.9° | 10° ✅ |
| 낙상 | 33/4702 (0.7%) | 6/4649 (0.13%) | 0 ❌ |
| 도착 속도 median | 0.117 m/s | 0.123 m/s | (기준 0.1) |
- **constellation 효과 실증**: 낙상 5배 감소(0.7%→0.13%) — 위치·방향 곱 결합 커널이 안정성
  개선에 유효. 단, 위치 정확도·정지 행동은 무변화 — 예측한 gait-clock 절충안의 한계가 실측됨.
- **auto_stop 조기정지 의심**: min_iters(2000) 직후 165 iter 만에 정체 판정. 낙상이 그 시점에도
  개선 중이었을 가능성 → 진짜 수렴이 아니라 `rel_eps`(2%)가 너무 민감해 성급히 멈췄을 수 있음.
  20000 iter까지 다 돈 별도 실행(logs/K1/K1/Goal_Pose/2026-07-23-21-54-01/)의 결과로 검증 예정:
  더 학습해도 위치/정지가 그대로면 clock 구조 문제 확정, 개선되면 auto_stop 파라미터 튜닝 필요.

### 2026-07-24 — v1@20000iter: auto_stop 조기정지 가설 확정, 계속 개선 중
| 지표 | v0(3400) | v1@2165 | v1@20000 | 게이트 |
|---|---|---|---|---|
| 위치 median/p90 | 12.8/17.9cm | 12.7/20.1cm | **8.5/16.4cm** | 5/10cm ❌(근접) |
| heading median | 1.7° | 1.9° | 3.6° | 10° ✅ |
| 낙상 | 33(0.7%) | 6(0.13%) | **4(0.09%)** | 0 ❌(근접) |
| 도착 속도 median | 0.117 | 0.123 | **0.10(경계)** | ≤0.10 |
| 성공률 엄격/완화 | 0.2/1.6% | — | **12.4/60.2%** | — |
- **확정**: 2165 iter에서의 auto_stop 정지는 성급했음 — 20000까지 이어가니 전 지표 개선
  (낙상 8배, 위치 -35%, 성공률(완화) 1.6%→60%). constellation 보상은 유효하게 작동 중,
  단지 더 많은 iteration이 필요했던 것.
- **조치 필요**: `tools/auto_stop.py` 기본값(`rel_eps=0.02`, `patience=3`) 완화 필요 —
  다음 실행부터 `--rel_eps 0.005 --patience 6 --min_iters 8000` 정도로 보수적으로 재설정
  (2165에서 멈췄던 실행 기준 20000까지도 계속 개선세였으므로 여유를 훨씬 더 줘야 함).
- **남은 격차**: 위치 median 8.5cm(게이트 5cm), p90 16.4cm(게이트 10cm), 낙상 4(게이트 0),
  도착속도가 정확히 정지 임계값(0.10)에 걸쳐있어 아직 완전히 안 멈추는 case 존재.
  다음 시도(사용자 확인 후): v1 model_20000에서 이어서 더 길게 학습(느슨한 auto_stop으로),
  그래도 정체되면 `goal_reached`(도착+정지 보너스) 또는 `constellation_weight` 상향(0.2→더 크게,
  근거리 정밀도 강화) 검토.
- 눈으로 확인용 영상: `logs/K1/K1/Goal_Pose/2026-07-23-21-54-01/eval/2026-07-24-15-26-52/rollout_env0.mp4`
  (scp로 로컬 확보, PULL.sh의 `videos/`가 아니라 `logs/` 안에 중첩된 경로임에 주의).

### 2026-07-24 — constellation 시각화 + 학습 가속 (근거 포함)
- **시각화**: `eval_goal_pose.py --record_video` 영상 좌상단에 top-down inset 추가 —
  초록 링=목표 constellation, 파랑 링=현재 로봇 constellation, 각 링의 큰 점=heading 방향점.
  두 링의 점 간격이 곧 constellation 오차(보상이 벌하는 그 양)라서 눈으로 오차를 직접 읽을 수 있음.
- **가속 1 — mini_epochs 20→5** (Goal_Pose.yaml): rsl_rl/legged_gym의 표준 학습 epoch 수는 5
  (Isaac Lab rsl_rl 설정 문서 기준). 우리 runner는 minibatch 분할 없이 full-batch를 20회 돌고
  있었으므로 표준 대비 학습 단계 연산이 4배 과잉이었음 → 4배 절감. (주의: 최적화 dynamics가
  달라지므로 이전 실행과의 iteration-수 비교는 무의미해짐 — 비교는 eval 하네스 결과로만.)
- **가속 2 — num_envs 2048→8192 권장** (CLI `--num_envs 8192`): Isaac Gym 논문(arXiv:2108.10470)이
  humanoid 최적 env 수를 4096~8192로 보고(16384는 이득 없음). 우리 실측 GPU 사용량이
  2048 env에서 3.6GB/49GB·util 42~57%로 여유가 커서 샘플 처리량 ~4배 기대.
- **가속 3 — TF32 활성화** (utils/runner.py): A6000(Ampere)에서 MLP 행렬곱 가속, 물리엔진 무관.
- 종합 기대: 학습 벽시계 시간 수 배 단축 (20000 iter ≈ 17시간 → 수 시간대 목표).

### 2026-07-24 — GPU util 67% 원인 조사 + 최적화 (8192 env 실행 기준)
- **조사 계기**: 8192 env에서 VRAM 6.4GB(계산상 정상, 여유 큼)인데 GPU-Util은 67%에 그침.
  [IsaacLab GitHub #3043](https://github.com/isaac-sim/IsaacLab/issues/3043)에서 동일 증상
  (GPU util 60~80%, 코어 1개만 100%) 다수 로봇 태스크에서 보고됨 — Isaac Gym류 프레임워크의
  구조적 특성(decimation 루프의 Python API 호출이 CPU 단일 코어를 병목시킴)으로 확인.
- **발견 (코드 근거)**: [envs/K1/goal_pose.py](htwk-gym/envs/K1/goal_pose.py) `step()`의
  decimation 루프(10회/스텝) 안에서 매번 `refresh_dof_force_tensor()`를 호출했는데,
  `dof_force` 텐서는 `acquire_dof_force_tensor()`조차 호출된 적 없어 **wrap된 버퍼가 존재하지
  않음** — 즉 완전히 죽은 코드였음(GPU→CPU sync만 매 서브스텝마다 10번 낭비).
- **조치**: 루프 안 `refresh_dof_force_tensor()` 제거. `sim.physx.num_velocity_iterations`
  1→0 (TGS 솔버는 position iteration만 필요, velocity iteration은 오히려 강성/수렴성 저하 —
  [Isaac Gym tuning 문서](https://docs.robotsfan.com/isaacgym/programming/tuning.html) 권장).
- **구조적 결론**: 남은 병목(67% util)은 근본적으로 decimation=10 루프의 Python 호출
  오버헤드(GPU 커널 launch 지연이 매 서브스텝마다 CPU 단일 코어에 걸림) — Isaac Gym 계열의
  알려진 한계라 완전 해소는 어려움. 가장 효과적인 완화책은 **num_envs를 계속 늘리는 것**
  (호출 횟수는 env 수와 무관하게 고정이라, env가 많을수록 커널당 연산량이 늘어 launch 지연
  비중이 상대적으로 줄어듦) — VRAM 여유(6.4/49GB) 감안 시 16384까지 시도 여지 있음.

### 2026-07-24 — GPU 효율: 단일코어 병목 → 병렬 스윕 (사용자 결정: GPU1만, 2~3개 + 200Hz 갈래)
- **핵심 인식**: 병목이 물리 스텝마다의 Python 호출(단일 CPU 코어)이라 한 프로세스가 GPU
  하나조차 못 채움(8192env에서 util 67%, VRAM 6.4/49GB). → 독립 프로세스를 더 띄우면 각자
  별도 코어/GIL을 쓰고 커널이 시분할돼 **총 처리량이 프로세스 수에 비례**. 근거:
  [NVIDIA MPS 분석](https://www.abhik.ai/concepts/gpu-computing/cuda-mps) — 개별 프로세스가
  GPU를 덜 쓸 때 멀티프로세스로 2~5배 처리량 보고.
- **사용자 결정**: (1) GPU 1 한 장에만 프로세스 2~3개(공유서버 "두 GPU 동시점유 금지" 규칙 유지,
  GPU 0 안 건드림), (2) 200Hz 물리를 스윕 한 갈래로 처음부터 테스트.
- **구현**:
  - `utils/runner.py`: `--config <yaml>` 추가 — 클래스는 `--task`(K1/Goal_Pose→GoalPose)로
    고르되 설정은 이 경로에서 로드. 같은 task를 서로 다른 설정으로 병렬 실행 가능.
  - `utils/recorder.py`: `basic.description`을 run 디렉토리 이름에 태깅 → 병렬 갈래가
    자기-라벨링된 별도 폴더에 안착(타임스탬프_armX).
  - `tools/make_sweep_configs.py`: base Goal_Pose.yaml + 갈래별 1-변수 override로 sweeps/*.yaml
    생성 + 갈래별 실행 명령 출력(DRY, 서버 pyyaml 사용). sweeps/는 .gitignore.
- **스윕 3갈래 (전부 model_20000에서 이어받아 1변수씩만 변경 → eval 델타 귀속 가능)**:
  - armA_continue: base 그대로 ("더 학습하면 된다" 가설의 기준선)
  - armB_goal_reached: `rewards.scales.goal_reached=1.0` (정지+위치 정밀도 직접 공략)
  - armC_200hz: `sim.dt=0.005, control.decimation=4` — 제어주파수는 50Hz 유지(dt×dec=0.02s)라
    정책 액션 의미 불변, 물리 서브스텝만 500→200Hz(2.5배 저렴). 물리 가속 가설 검증.
- **모니터링**: 각 갈래를 별도 tmux 창에서 → 서버 붐비면 갈래 하나만 개별 kill(양보 용이).
  고정 iter 예산(기본 20000)으로 공정 비교 후 eval 하네스로 3갈래 판정.
- **참고**: MPS로 커널 실제 겹침까지 가면 util 더 오르지만 compute mode 변경은 타 사용자
  프로세스를 깨므로 금지 — per-user MPS(DEFAULT 모드)만 선택지, 지금은 미적용(멀티프로세스만으로 충분).

### 2026-07-24 — 실기 통합 결정: lookahead/정지/BT 흔들림/사람 지지

#### 1. 2–3 m A* lookahead와 odom

- **고정 local waypoint를 한 번만 주는 경우**: 짧은 거리여도 partial observability가 남는다.
  현재 actor는 feed-forward MLP라 명시적 history가 없고 simulator GT reward는 배포 시 남은 거리를
  관측하게 해주지 않는다. 직전 action/몸 상태로 open-loop timing을 암묵적으로 외워 nominal sim에서
  성공할 수는 있지만 slip/push/사람 지지 때 누적오차를 위치 feedback으로 교정하지 못한다.
- **상위 localization/perception이 현재 robot-frame lookahead를 streaming으로 계속 주는 경우**:
  low-level policy 내부의 별도 적분은 생략 가능하다. 그러나 현재 축구 stack은 PF motion update,
  `robotPoseToField`, A*/local planner, TF에도 같은 egomotion을 쓰므로 **시스템 odom delta는 생략할 수
  없다.** 직접 상대 목표 sensor로 global localization/field 전술 전체를 대체하는 다른 stack일 때만
  예외다.
- 현재 actor는 이미 projected gravity와 gyro를 받는다. raw acceleration 한 프레임을 더 넣어도
  translation을 알 수 없고, 적분 memory/contact constraint가 없어 해결이 안 된다.
- 현재 범위는 `dx±2 m`, `dy±1.5 m`라 정면 3 m lookahead는 OOD(normalized x=1.5)다. 우선 BT가
  학습 사각형 안의 path point로 project/clip하고, 3 m가 실제로 필요하면 별도 분포로 재학습한다.
- 채택 구조: `low_state → contact-aided estimator → timestamped body-frame SE(2) delta+covariance →
  PF/continuous odom → A*/GoalPose target`의 공용 파이프라인이다. policy 안에 global pose를 넣을
  필요는 없지만, PF가 만든 map pose와 estimator의 local delta는 둘 다 유지한다. 필터 baseline과
  AutoOdom형 supervised estimator는 걷기 PPO와 병렬 개발한다([STATE_ESTIMATION.md](STATE_ESTIMATION.md)).
- history/RNN policy의 implicit odometry도 유효한 ablation이지만 현재 PF stack의 production 경로는
  별도 Δpose estimator가 선행 조건이다. policy 입력 방식의 우선순위만 streaming relative goal →
  delta로 보간한 relative goal → recurrent/open-loop policy 순서다.

#### sim2real branch 재감사 결과 (PF까지 포함)

- `../[07]sim2real`의 `sim2real` branch commit `9ffeb143`을 직접 추적했다. `brain.cpp`의
  `odometerCallback()`은 SDK 누적 odom을 차분/scale/clamp한 뒤 main PF predict, continuous field pose,
  TF/`localized_pose`, relocalization PF, orientation sentinel, local planner compute를 한꺼번에 실행한다.
- CUSTOM에서 `odometer_state`가 멈추면 landmark correction 시 data pose가 간헐적으로 바뀔 수는 있어도
  그 사이 PF propagation, TF, planner update가 멎는다. 별도 명목상 100 Hz `Brain::tick()`은 cached planner
  velocity를 계속 보낼 수 있으므로 odom/pose freshness watchdog과 stale-command STOP이 P0다.
- 새 estimator는 기존 `Odometer{x,y,theta}`를 흉내 내는 것보다 old-body-frame
  `(Δx,Δy,Δyaw,stamp,seq,epoch,Q,status)`를 source-neutral consumer에 주는 구조로 만든다. 기존 SDK
  보정값(forward 1.30, backward 1.31, lateral 1.17, yaw 1.5)은 SDK adapter에만 두고 새 estimator에는
  중복 적용하지 않는다.
- delta의 x/y는 SE(2) `Log`가 아니라 relative transform의 실제 old-body translation으로 고정하고 PF도
  Pose2D composition을 쓴다. Q는 pose clone/preintegration cross-covariance로 만들고 mocap으로
  scale/floor를 calibration한다. delayed vision의 production 경로는 PF rewind/correct/delta replay다.
  capture→now marker transform은 odom Q까지 measurement covariance에 반영해 보수적으로 inflate하는
  prototype 근사로만 두고 consistency를 replay에서 확인한다.
- 현 ROS `LowState`/`Odometer`에는 source timestamp가 없고, CUSTOM deploy의 timer는 callback 횟수×2 ms다.
  estimator는 detection과 같은 ROS clock receipt stamp를 쓰거나 명시적인 clock mapping+gap detection을
  사용해야 한다. worker는 PF를 직접 건드리지 않고 Brain single-writer callback에 delta를 넘긴다.
- 현재 `deploy/`는 `B1JointCnt` 23-slot/`[11:]` layout이고 K1 resource는 22 actuator/leg XML indices
  10..21이다. SDK가 placeholder slot을 유지하면 맞을 수도 있어 one-index bug로 단정하지 않는다.
  실제 K1 packet은 이름 없는 배열이므로 robot/SDK-version별 index table, packet schema,
  serial/parallel ankle 의미, `T_BI`를 boot-time assert+1-joint physical test로 먼저 승인한다.

#### 2. 제자리 정지와 walk→stand 자세

- stand category는 4–8초 resample 때 에피소드 중간에도 나와 **갑작스러운 walk→stand command**는
  일부 학습한다. 그러나 비영점 목표에 접근하며 자연스럽게 감속해 stand와 같은 자세로 들어가는
  arrival transition은 보장하지 않는다. `gait_frequency=0`일 때 phase도 임의 값에 멈추므로
  phase별 종단 자세가 갈릴 수 있다.
- 3-arm 결과를 먼저 판정한 뒤 다음 실험에서 `arrival_hold`를 추가한다. 진입/해제 threshold를
  다르게 둔 SE(2) hysteresis, double-support/canonical phase에서 clock 정지, gait frequency
  ramp-down, stand와 동일한 neutral joint/base/양발 보상을 walk 도착 후에도 적용한다.
- 평가는 순간 final speed뿐 아니라 1 s 연속 hold 성공률, hold 중 최대 위치 이탈, speed p90,
  stand 시작 vs walk→stand 종단 joint RMS, 발 step/slip을 기록한다.
- armB의 `goal_reached` radius는 10 cm인데 독립 median gate는 5 cm다. 5–10 cm에서 보상 파밍할
  수 있으므로 armB 결과 해석 시 주의하고, 후속 구현은 heading/yaw speed/dwell을 조건에 추가한다.

#### 3. 사람이 팔을 잡아주는 상황

- 현재 trunk Gaussian push는 사람 손 지지와 다르다. K1 locomotion URDF는 팔이 fixed/collapsed라
  우선 trunk에 equivalent wrench를 가하는 virtual hand-anchor spring-damper로 근사한다.
- 좌/우/양팔, lever arm, anchor, stiffness/damping, force cap, 유지시간, ramp/갑작스러운 release를
  env별 랜덤화한다. 대부분은 no-support로 두어 손에 의존하는 정책이 되지 않게 한다.
- no-support / support-on / release 후 1–2 s의 세 프로토콜을 분리 평가한다.

#### 4. domain randomization 운영법

- 현재 물성/PD/지연/noise/kick/push DR은 이미 넓게 존재한다. 다음 일은 항목 추가보다 **실기 로그로
  범위를 calibration**하는 것이다: nominal과 5–95 percentile을 train 분포로, 범위 밖은 OOD
  stress eval로 분리한다.
- 후속 구현에서 물성/구동계와 bias는 episode별, white noise는 step별, drift는 시간상관(AR/OU)으로
  모델링한다. 현재 mass/CoM/PD/contact property는 env 생성 시 1회 샘플이므로 episode별 재설정
  코드를 추가해야 한다.
  좌우 PD/encoder offset, 배터리 torque scale처럼 실제로 상관된 항목은 묶어 샘플링한다.
- 현재 모든 env가 같은 시점에 kick/push를 받는 구조는 phase/interval/duration을 env별로 바꾼다.
  no-DR / calibrated-DR / OOD-stress 평가를 고정해 “DR이 많음”과 “실기를 잘 덮음”을 구분한다.

#### 5. BT goal jitter와 RLkick handoff

- 학습에서는 reward용 `goal_true`와 actor 입력용 `goal_observed`를 분리한다. bias/random walk,
  sample-and-hold, latency, dropout, outlier/spurious planner jitter는 `goal_observed`에만 넣는다.
  실제 공 이동·유효한 replanning은 `goal_true`/`goal_observed`를 함께 바꾸되, 센서 noise 때문에
  reward target 자체를 흔드는 것은 금지한다.
- 실기에서는 raw robot-local goal을 곧바로 low-pass하지 않는다. map/odom-frame path arc-length에
  innovation/confidence gate, rate limit, deadband, last-valid hold, monotonic-index hysteresis를
  적용하고 최신 pose로 local 변환한다. yaw는 wrap-safe circular filter를 쓴다.
- GoalPose→RLkick은 RLkick의 실측 capture set(예: 거리/각도/ball confidence)을 N frame 연속
  만족할 때 진입하고, 더 큰 exit threshold와 minimum dwell/cooldown을 둔다. 진입 후 approach
  pose는 freeze/강한 smoothing하고 공 상대 최종 보정은 RLkick이 담당한다. 독립 GoalPose 5 cm
  benchmark와 축구 stack의 handoff 성공률은 서로 다른 게이트로 유지한다.

#### 6. GPU 0의 오늘 사용 순서와 중간 확인

- 2026-07-24 사용자 허용에 따라 GPU 1의 세 학습은 건드리지 않고, GPU 0에서는 먼저 세 arm의
  최신 **안정된** checkpoint를 순차 평가한다. 새 4번째 학습을 먼저 띄우면 어느 arm이 유망한지
  모른 채 계산을 쓰므로 preview 결과 뒤에 다음 lever를 정한다.
- 원클릭 도구: `python tools/preview_sweep.py --device cuda:0`. 기본은 64 env × 30 simulated s와
  arm별 12 s 영상(추세 확인용), `--full --no_video`는 256 env × 120 s 표준 프로토콜이다.
  `--status_only`는 TensorBoard 마지막 iteration/reward와 checkpoint만 출력한다.
- `eval_goal_pose.py`는 setup/rollout wall-clock, ETA, env당 real-time factor와 aggregate
  env·s/wall-s를 report.json/report.md에 기록한다. `--config <run>/config.yaml`로 armC의 native
  200 Hz behavior도 볼 수 있지만, 기본 sweep 비교는 공통 500 Hz dynamics를 써야 공정하다.
  armC **속도** 비교는 camera 비용을 빼기 위해 별도로 `--native --no_video`를 쓴다.
- 다음 GPU 0 우선순위: preview 승자 확인 → (a) arrival_hold 후속 arm 또는 (b) 세 policy의
  rollout을 모아 Δpose estimator 학습 데이터 생성. 정보 없이 “goal/odom 제거 MLP” arm을 돌리는
  것은 nominal 가능성을 부정하진 않지만 sim2real 신뢰성이 낮아 production 우선순위에서 제외하고,
  필요하면 마지막에 작은 ablation으로만 확인한다.

### 2026-07-24 — sim2real 요구 반영 + armD_v2_ultimate (GPU 0, 24h 창구)
역할 분담 확정: **오도메트리(레그+IMU 추정기)는 Codex 담당**(STATE_ESTIMATION.md가 Codex의
시스템 설계로 재작성됨 — PF/planner까지 포함한 전체 egomotion 계약). **이쪽(Claude)은 CUSTOM
mode에서 쓸 수 있는 센서(IMU gyro/gravity + 관절 엔코더)만으로 RLKick 직전 pose에 최적
도달하는 정책 학습**에 집중.
- **"odom 없이 도달" 질문 판정**: 현 네트워크는 history 없는 순수 MLP라 자기 이동량을 내부
  적분할 기억이 없음 → 목표를 한 번만 주고 끊는 완전 open-loop는 원리적으로 불가.
  단, 실전에서는 지각(공 관측·localization)이 상대 목표를 계속 재측정해 주므로 폐루프는
  유지됨 — 따라서 정답은 "상대 목표 관측에 강한 노이즈를 걸어 학습"(사용자 가설과 일치).
  IMU 가속도를 관측에 추가하는 것은 적분(=기억) 없이는 위치 정보가 안 되므로 도움 안 됨 —
  그 적분을 제대로 하는 것이 Codex의 estimator.
- **코드 변경** (base 동작은 완전 동일 유지, 전부 스위치):
  - `goal_pose.py` 관측에 `noise.goal_pos`/`noise.goal_heading` 적용(관측만 오염, 보상은
    ground-truth 유지). base yaml은 range [0,0] = OFF.
  - `_reward_stand_posture` 신설: 목표 반경 0.3m 안에서 기본 기립 자세와의 관절 편차 벌점
    → 감속·정지 자세가 PREP 기립과 비슷해져 RLKick 인계가 깨끗해짐. base scale 0 = OFF.
  - `make_sweep_configs.py`에 `--only` 필터 + **armD_v2_ultimate** 추가(의도적으로 다변수 통합):
    goal 지터 std 10cm/6°, goal_reached +1, stand_posture -2, 목표범위 ±3m/±2m,
    resample 4~10s(더 긴 보행), push_duration 3s(팔 잡힘류 지속 외란), armature 0.02(공식 USD값).
- **eval wall-clock**: Codex가 이미 eval_goal_pose.py에 setup/rollout wall-clock + 처리량
  출력을 반영함(+ --config/--exploratory 옵션) — 추가 작업 불필요.
- **운영**: GPU 1 = armA/B/C 단일변수 통제실험(계속), GPU 0(24h 유휴 보장) = armD 통합
  리허설. git pull은 실행 중 프로세스에 무해(코드는 시작 시 메모리에 적재 완료, logs/와
  sweeps/는 gitignore라 pull이 안 건드림).
