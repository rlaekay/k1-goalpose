# 실기 로그 — **되찾을 수 없는 측정이다. 지우지 마라**

분석 세션이 2026-08-09 에 지적했다: **실기 시계열이 각각 한 곳에만 있고 전부
버전관리 밖이었다.** 서버 CSV 745 + 맥 1,236 파일 중 **실기는 단 둘**이고
(나머지는 시뮬 출력) 그 둘이 단일 사본이었다. 여기로 옮겨 커밋한다.

`server-dump/` 와 `logs/` 는 `.gitignore` 에 있다. **이 디렉터리는 아니다.**

## 파일

| 파일 | 행 | 길이 | 조건 |
|---|---:|---:|---|
| `2026-08-09_t_stand_i3b.csv` | **6,330** | 184 s | `fixed 0,0,0` 서기. `gait_process` 0 에 얼어 있고 `walking=0`. 낙상 1회(t≈150~165 s) 포함 |
| `2026-08-0x_real_walk_i3b.csv` | 91 | 2.38 s | 보행 중 붕괴 국면. `walking=1` |

둘 다 **`I3b_stance10 model_200`** 이다(md5 `60006446...`, R3 종결).

## ⛔ 이 데이터를 읽을 때

- **등간격이 아니다.** `tick_dt_s` CV **0.303**(11~48 ms). **FFT 를 그냥 걸지 마라** —
  나와 분석 세션이 둘 다 여기서 틀렸다(`ibatch` §8-64, `RETRACTIONS` C44~C46).
  주파수 주장을 하려면 Lomb-Scargle 이나 비균일 표본용 방법을 쓰고, **귀무분포를 같이 내라.**
- ~~`t_s` 는 벽시계가 아니다~~ → **정정(2026-08-09, Codex 감사 §19-6 이 잡았다):
  `t_s` 는 `time.monotonic()` 이다**(`00ed5b8` 의 `_log_timing`, `now - _timing_t0`).
  카운터 시계(`Timer.get_time()` = LowState 개수 × 0.002, §8-59)는 **정책 스케줄링과
  `gait_process`** 에 쓰인 것이지 이 열이 아니다. 그래서 이 파일의 t_s 로 계산한
  율(38.4 Hz 등)은 유효하다.
- 열 배치: `t_s, low_state_age_s, tick_dt_s, tilt_deg, roll, pitch, gx, gy, gz,
  walking, gait_freq, gait_process, pub_hz, goal_x, goal_y, heading_err,
  q0..q11, dq0..dq11, tau0..tau11, act0..act11`
  (`q0..q11` = 다리 12관절, `leg_dof_start=10`. 관절당 Hip_P/Hip_R/Hip_Y/Knee_P/Ankle_P/Ankle_R
  ⇒ **`q5` = Left_Ankle_Roll, `q11` = Right_Ankle_Roll**)
  ⚠️ **Hip_Roll ↔ Hip_Yaw 한 쌍만 미확정**이다(둘 다 default 0 / kp 100 / effort 30 이라
  기존 지문으로 못 가른다). 무부하에서 Hip_Roll 만 손으로 벌려 인덱스를 읽으면 닫힌다.

## 여기서 나온 판정

- `pub_hz` **185.1 Hz**(설계 500) — R6 종결
- 제어 **38.4 Hz**(설계 50), gait **1.5116 Hz**(지령 2.0)
- `low_state_age` **1.02 ms** ⇒ **센서가 아니라 루프가 느리다**
- tilt median **1.4°** ⇒ `b`~`r` 자세 수정이 맞았다
- 발목 roll 이 12채널 중 유일한 이상(`r1(q)` 0.665/0.694 대 나머지 0.945~0.991)
- 토크가 sim 클램프를 넘는 관절 **4/6**(Hip_P 37.8>30, Hip_R 34.0>20, Ankle_P 23.6>20)
