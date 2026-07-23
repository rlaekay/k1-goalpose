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

## 서버에서 학습 실행
```bash
# 서버 터미널
ssh user@user-ESC4000A-E12
cd /mnt/DATA/workspace/ws_eungkyu/k1-goalpose/htwk-gym
python train.py --task=K1/GoalPose --headless
```

