# archive — 세대가 끝난 학습/평가 코드

`eval_*` / `train_*` / 배치 런처가 세대마다 하나씩 쌓여서 `tools/`를 열 때마다
"지금 쓰는 게 뭐지"를 다시 판단해야 했다. 그래서 **현행 한 벌만 남기고** 나머지를
여기로 옮겼다. 지운 것이 아니라 옮긴 것이므로 `git log --follow <path>`로 이력이
그대로 따라오고, 필요하면 `git mv`로 되돌리면 된다.

원래 경로를 그대로 유지한다: `archive/tools/run_g_suite.sh`는 예전 `tools/run_g_suite.sh`.

## 지금 살아 있는 한 벌 (여기 없는 것들)

```
train_v7.py                    학습 진입점 (RunnerV3: minibatch PPO + 대칭 손실)
train.py                       train_and_eval.sh 의 기본 TRAIN 값
eval_goal_pose.py              평가 본체
tools/tonight.sh               런처 (config 생성 -> 스모크 게이트 -> tmux 배치)
tools/make_v7_arms.py          arm config 생성기
tools/smoke_v7.py              학습 전 게이트
tools/train_and_eval.sh        학습 -> 최적 체크포인트 -> 영상 -> 공유 폴더
tools/select_best_checkpoint.py
tools/watch_eval.py            학습 중 연속 평가 + 조기 정지
tools/monitor.py / monitor_page.html / progress.py    진행 상황
tools/check_names.py           정적 이름 검사
tools/fetch_results.sh         Mac 로 결과 회수
envs/K1/goal_pose{,_v3,_v7,_v8}.py + 대응 yaml
utils/*, deploy/*
```

## 옮긴 것

### 루트

| 파일 | 세대 | 왜 대체되었나 |
|---|---|---|
| `train_v3.py` | v3 | `train_v7.py`와 실질적으로 같은 파일(둘 다 `utils.runner_v3.RunnerV3` 한 줄 래퍼). v7 쪽만 남긴다 |
| `train_v4.py` | v4 (CrossQ) | `utils.runner_crossq.RunnerCrossQ` 래퍼. goal-pose 라인은 PPO로 확정되어 미사용. **`envs/K1/get_up.py` 주석이 이 트레이너를 가리키므로, Get_Up 을 다시 학습하려면 되살려야 한다** |
| `train_hbatch.py` | H | 역시 `RunnerV3` 한 줄 래퍼. H 배치 종료로 별도 진입점이 필요 없다 |
| `seed_warmstart_checkpoint.py` | — | deploy 용 TorchScript `.pt` 에서 러너가 읽는 `.pth` 를 만들어 warm-start 를 심는 도구. 현행 워크플로는 `logs/` 의 `.pth` 를 직접 `--checkpoint` 로 넘기므로 호출하는 곳이 없다 |
| `export_tflite.py` | 업스트림 HTWK | `utils.model_thomas` 를 import 하는데 이 저장소에 그 모듈이 없다 — 현재 상태로 실행 불가. (`htwk-gym/README.md` 에 사용법만 남아 있다) |

### envs

| 파일 | 세대 | 왜 대체되었나 |
|---|---|---|
| `envs/K1/goal_pose_hbatch.py` | H | H 배치 전용 env (`GoalPoseV7` 상속). H 종료. `envs/__init__.py` 의 `GoalPoseHBatch` import 도 같이 제거했다 — 이 import 가 남아 있으면 `from envs import *` 를 하는 `eval_goal_pose.py` / `smoke_v7.py` / `utils/runner.py` 가 전부 죽는다 |

### sweeps

| 파일 | 세대 | 왜 대체되었나 |
|---|---|---|
| `sweeps/hbatch/H0-codex.yaml` ~ `H3-codex.yaml` | H | H0~H3 arm config (Codex 생성). H 종료. 현행 config 는 `tools/make_v7_arms.py` 가 `sweeps/` 에 그때그때 생성한다 |

### tools — H 배치

H 배치는 4 GPU-day 를 쓰고 arm 4종이 모두 나빠졌으며 선택기가 4/4 로 `model_0` 을
골랐다(`tools/watch_eval.py` 머리말). 배치가 끝났으므로 전용 도구도 함께 내린다.

| 파일 | 왜 대체되었나 |
|---|---|
| `tools/make_hbatch_configs.py` | H arm config 생성기. 현행은 `make_v7_arms.py` |
| `tools/smoke_hbatch.py` | H 전용 스모크. 현행은 `smoke_v7.py` |
| `tools/run_hbatch_suite.sh` | H 배치 런처. 현행은 `tonight.sh` |
| `tools/run_hbatch_arm.sh` | H arm 1개 실행. `run_hbatch_suite.sh` 하위 |
| `tools/train_and_eval_hbatch.sh` | `train_hbatch.py` 를 호출하는 H 전용 학습+평가. 현행은 `train_and_eval.sh` |
| `tools/verify_hbatch_health.py` | H 런 건전성 검사 |
| `tools/verify_hbatch_video.py` | H 영상이 sim teardown 전에 끝났는지 확인 |
| `tools/compare_hbatch_results.py` | H0~H3 결과 집계 + 게이트 |

### tools — M-cell 스크린 (H 후속, Codex)

H 가 disturbance / joint DR / mirror 중 무엇이 원인인지 구분하지 못해서, 레버를
하나씩만 바꾼 200-iteration 짧은 스크린으로 설계된 것들이다.

| 파일 | 왜 대체되었나 |
|---|---|
| `tools/make_mcell_configs.py` | M0~M3 cell config 생성기. 스크린 종료 |
| `tools/compare_mcells.py` | M-cell 페어드 비교 리포트 |
| `tools/run_mcells.sh` | M-cell 엔드투엔드 런처 |
| `tools/test_mcell_static_codex.py` | M-cell 생성기/비교기의 정적 단위 테스트. 대상 파일이 전부 여기로 내려왔다 |

### tools — v7 / E / G 배치

| 파일 | 세대 | 왜 대체되었나 |
|---|---|---|
| `tools/run_v7_suite.sh` | v7/E | 스모크 게이트 + tmux 배치 런처. `tonight.sh` 가 같은 일을 한다 |
| `tools/run_g_suite.sh` | G | `run_v7_suite.sh` 의 G 배치 판. `tonight.sh` 의 직전 세대이고, 사용법 주석까지 `run_v7_suite.sh` 를 그대로 물려받아 있다 |
| `tools/reeval_v7.sh` | E | 하네스 버그 수정 후 E 배치 arm 재평가 전용. 재평가가 끝나서 역할 종료 (`tonight.sh` 는 이 프로세스가 남아있는지 `pgrep` 으로 보기만 한다) |
| `tools/reeval_e_batch_gpu1.sh` | E | 위와 같은 재평가를 GPU 1 에 격리해 돌리는 판. 역시 역할 종료 |
| `tools/watch_reeval.sh` | E | `reeval_v7.sh` 전용 진행 표시기. 대상이 없어졌다 |
| `tools/diag_reset.py` | v7 | 리셋 경로 일회성 진단. 결과는 `smoke_v7.py` 체크 항목으로 흡수 |
| `tools/diag_seq.py` | v8 | 순차 내비게이션(seq) 일회성 진단. 마찬가지로 `smoke_v7.py` 가 상시 확인한다 |
| `tools/watch_gpu_waterfall.sh` | — | 읽기 전용 GPU 점유 뷰어. `tools/monitor.py` 웹/TUI 가 대체 |
| `tools/test_joint_probe_codex.py` | Codex | `eval_goal_pose` 의 joint-DR probe 를 AST 로만 읽는 정적 테스트. joint DR 실험 라인(H/M)이 종료 |

### tools/legacy — armA~D 세대

원래 `tools/legacy/` 에 있던 것을 경로만 `archive/tools/legacy/` 로 옮겼다.
내용과 사유는 그 폴더의 `README.md` 에 그대로 있다:
`make_sweep_configs.py`, `auto_stop.py`, `preview_sweep.py`, `run_e3.sh`,
`run_f_batch.sh`, `eval_suite.sh`.

## 되돌리는 법

```bash
git mv htwk-gym/archive/tools/run_g_suite.sh htwk-gym/tools/run_g_suite.sh
```

`train_v4.py` 를 되살릴 때는 `utils/runner_crossq.py` 와 `utils/crossq.py` 가
`utils/` 에 그대로 남아 있으므로 파일 하나만 옮기면 된다.
