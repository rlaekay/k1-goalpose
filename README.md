# K1 Goal-Pose Locomotion RL

Local development + Server execution setup.

## 폴더 구조
~/Projects/k1-goalpose/
├── MASTERPLAN.md   # 마스터플랜 (목표/범위/성공기준 + 변경 이력)
├── UPSTREAM.md     # htwk-gym 포크 출처/시점 기록
├── README.md       # 이 파일
├── SYNC.sh         # 로컬 ↔ 서버 동기화 스크립트 (rsync)
└── htwk-gym/       # NaoHTWK/htwk-gym 포크, 일반 추적 파일 (submodule 아님)
    └── envs/K1/
        ├── parameter_walk.py / Parameter_Walk.yaml   # 베이스라인
        └── goal_pose.py / Goal_Pose.yaml              # 우리가 만드는 태스크

htwk-gym에서 태스크는 `tasks/`가 아니라 `envs/<robot>/<task>.py` +
`envs/<robot>/<Task>.yaml` 쌍으로 정의된다 (`utils/runner.py`의
`get_task_class`가 `envs` 패키지를 스캔).

## 로컬 ↔ 서버 동기화

### 맥 → 서버 (코드 업로드)
```bash
./SYNC.sh push
```

### 서버 → 맥 (결과 다운로드)
```bash
./SYNC.sh pull
```

## 서버에서 학습 실행
```bash
# 서버 터미널
ssh user@user-ESC4000A-E12
cd /mnt/DATA/workspace/ws_eungkyu/htwk-gym
python train.py --task=K1/GoalPose --headless
```

