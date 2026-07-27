# legacy — 더 이상 `tools/` 경로에 두지 않는 도구

지우면 이력이 흐려지고, 그냥 두면 `tools/`를 열 때마다 "이건 뭐지"를 반복한다.
그래서 옮겼다. `git log --follow`로 전부 추적되고, 필요하면 되돌리면 된다.

| 파일 | 왜 여기 있나 |
|---|---|
| `make_sweep_configs.py` | armA~D 세대 config 생성기. `make_v7_arms.py`가 대체 |
| `auto_stop.py` | armA~D 때 쓰던 수렴 자동정지. v7부터 고정 iteration 운용이라 미사용 |
| `preview_sweep.py` | armA~D 스윕 미리보기. 대상 세대가 종료 |
| `run_e3.sh` | E3(무커리큘럼 광역 샘플링) 런처. **한 번도 실행 안 함** — Margolis et al.(RSS 2022)이 "무커리큘럼은 저속에서도 학습이 아예 안 된다"를 ablation으로 보고해 착수 전 폐기 |
| `run_f_batch.sh` | F 배치 런처. F는 G로 재설계됨(masterplan3 §4). **실행된 적 없음** |
| `eval_suite.sh` | clean/perturbed/jitter 3조건 평가. `train_and_eval.sh`(STRESS=1) + `reeval_v7.sh`가 같은 일을 하고, 이쪽은 **실행된 적 없음** |
