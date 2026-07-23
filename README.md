# K1 Goal-Pose Locomotion RL

Local development + Server execution setup.

## 폴더 구조
~/Projects/k1-goalpose/
├── MASTERPLAN.md # 마스터플랜
├── README.md # 이 파일
├── SYNC.sh # 로컬 ↔ 서버 동기화 스크립트
└── htwk-gym/
├── tasks/K1/
│ ├── ParameterWalk/
│ └── GoalPose/ # 우리가 만드는 태스크
└── [나머지]
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

