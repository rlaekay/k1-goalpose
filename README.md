# K1 Goal-Pose Locomotion RL

Local development + Server execution setup.

## 폴더 구조
~/Projects/k1-goalpose/
├── MASTERPLAN.md   # 마스터플랜 (목표/범위/성공기준 + 변경 이력)
├── UPSTREAM.md     # htwk-gym 포크 출처/시점 기록
├── README.md       # 이 파일
├── PULL.sh         # 서버 → 로컬 결과물(체크포인트/로그/영상) 다운로드 (rsync)
└── htwk-gym/       # NaoHTWK/htwk-gym 포크, 일반 추적 파일 (submodule 아님)
    └── envs/K1/
        ├── parameter_walk.py / Parameter_Walk.yaml   # 베이스라인
        └── goal_pose.py / Goal_Pose.yaml              # 우리가 만드는 태스크

htwk-gym에서 태스크는 `tasks/`가 아니라 `envs/<robot>/<task>.py` +
`envs/<robot>/<Task>.yaml` 쌍으로 정의된다 (`utils/runner.py`의
`get_task_class`가 `envs` 패키지를 스캔).

## 코드 동기화 (git, 2026-07-23부터)

코드는 더 이상 rsync로 push하지 않는다 — 로컬에서 GitHub로 push, 서버는
`git pull`만 한다. 학습 체크포인트/로그/영상처럼 git에 안 맞는 큰 바이너리만
`PULL.sh`로 따로 받는다 (이유: [MASTERPLAN.md](MASTERPLAN.md) 변경 이력 참고).

### 서버 최초 설정 (한 번만)
```bash
ssh user@user-ESC4000A-E12
cd /mnt/DATA/workspace/ws_eungkyu
git clone https://github.com/rlaekay/k1-goalpose.git
# private repo라면: git clone 시 GitHub PAT 또는 SSH deploy key 필요
#   HTTPS: git clone https://<PAT>@github.com/rlaekay/k1-goalpose.git
#   SSH:   git clone git@github.com:rlaekay/k1-goalpose.git (서버에 deploy key 등록 필요)
cd k1-goalpose/htwk-gym
# 이후 IsaacGym/PyTorch 등 환경 설치는 htwk-gym/README.md의 Installation 절 참고
```

### 코드 갱신 (로컬 → 서버)
```bash
# 로컬에서
git add -A && git commit -m "..." && git push

# 서버에서
cd /mnt/DATA/workspace/ws_eungkyu/k1-goalpose
git pull
```

### 결과물 회수 (서버 → 로컬)
```bash
./PULL.sh
# 다른 서버 경로/계정을 쓰면: SERVER=user@host SERVER_REPO=/path/to/k1-goalpose ./PULL.sh
```

## 서버에서 학습 실행 (전체 파이프라인)
```bash
# 서버 tmux 창 1 — 학습 (GPU 번호는 nvidia-smi로 빈 쪽 확인 후 지정)
cd /mnt/DATA/workspace/ws_eungkyu/k1-goalpose/htwk-gym
python train.py --task=K1/Goal_Pose --headless True \
  --checkpoint logs/warmstart/parameter_walk_actor_seed.pth \
  --num_envs 2048 --sim_device cuda:1 --rl_device cuda:1

# 서버 tmux 창 2 — 수렴 감시: 보상 정체 감지 → 안전 정지 → 자동 평가 리포트
python tools/auto_stop.py

# 평가만 따로 (최신 체크포인트 자동 탐색; report.md/json + segments.csv 생성)
python eval_goal_pose.py --task K1/Goal_Pose --checkpoint -1 \
  --sim_device cuda:1 --rl_device cuda:1

# 보상 조합 실험: envs/K1/Goal_Pose.yaml의 rewards.scales만 수정
#   goal_progress / goal_reached / heading_near_goal (기본 0 = 꺼짐)
```

