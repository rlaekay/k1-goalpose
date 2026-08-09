# 2026-08-09 Codex 실기 세션 원시 데이터

이 디렉터리는 2026-08-09 실기 세션 중 로봇 `/tmp`에서 회수한 CSV를 **바이트 변경 없이**
보존한 것이다. 해석 보고서는 저장소 루트의
`CODEX_SIM2REAL_REAL_TEST_AUDIT_20260809.md`를 참조한다.

## 원본 보존 원칙

- CSV 값, 열 순서, 정밀도는 수정하지 않았다.
- 복사 직후 원본 `/tmp/<name>.csv`와 보존본의 SHA-256이 동일함을 확인했다.
- `trial_manifest.csv`는 실행조건·유효성·해시를 연결하는 메타데이터다.
- `joint_zero_probe.csv`만 콘솔 출력 3회를 사람이 옮긴 별도 표이며 원시 DDS 덤프가 아니다.
- 이 폴더의 보존본을 분석 입력으로 사용하고, 파생값을 원본 CSV에 다시 쓰지 않는다.

## 파일과 판정 범위

| 파일 | 정책 | 목적 | 핵심 판정 |
|---|---|---|---|
| `np_stand.csv` | NP | 기존 counter-clock 제자리 1초 | 37.54 Hz 기준 참고 |
| `np_walk_02.csv` | NP | 기존 counter-clock 전방 0.2 m, 1.5초 | 정책 불안정 + 종료 DAMPING 교락 |
| `np_wallclock_stand.csv` | NP | wall-clock 변경 후 제자리 1초 | polling만으로 50 Hz 복원 실패 |
| `np_deadline_stand.csv` | NP | deadline sleep 후 제자리 1초 | NP 실기 퇴화 확인 |
| `nh_deadline_stand.csv` | NH | NP에서 armature arm 제거 후보 | NP보다 개선, I3b보다 동적 |
| `i3b_deadline_stand.csv` | I3b | 같은 배포기의 기준선 | 세 후보 중 가장 조용함 |
| `i3b_pub100_stand.csv` | I3b | 100 Hz LowCmd + tau 10 ms | **stale-state 버그로 성능 비교 무효** |
| `i3b_pub100_walk02.csv` | I3b | 전방 0.2 m, 1초 | **stale-state 버그로 성능 비교 무효** |
| `i3b_freshstate_stand.csv` | I3b | stale-state 수정 후 제자리 1초 | 50/50 추론에 증가한 상태 sequence |

`fixed 0,0,0`은 `walking=0`, `gait_freq=0`, `gait_process=0`인 balance stance다.
`fixed 0.2,0,0`은 0.2 m/s 속도 명령이 아니라 로봇 기준 전방 0.2 m 목표점이다.

## 열과 단위

- `t_s`: 로그 시작 뒤 monotonic wall time [s]
- `tick_dt_s`: 정책 로그 행 사이 wall time [s]
- `low_state_age_s`: 마지막 callback 도착 뒤 경과시간 [s]
- `policy_state_age_s`: inference가 소비한 snapshot 시각부터 로그 기록 시점까지 [s]
- `low_state_seq`: callback가 원자 snapshot을 갱신할 때마다 증가하는 sequence
- `tilt_deg`: 몸통 수직 기울기 [deg]
- `roll,pitch`: IMU 자세 [rad]
- `gx,gy,gz`: body angular velocity [rad/s]
- `q*`, `dq*`, `tau*`: 12개 다리 관절의 위치 [rad], 속도 [rad/s], 추정 토크 [N m]
- `act*`: actor action; 배포 target은 nominal pose와 이 action으로 구성된다.

정책 주기는 `1 / mean(tick_dt_s[1:])`로 계산했다. 첫 행의 `tick_dt_s=0`은 제외한다.
몸 각속도는 `sqrt(gx^2+gy^2+gz^2)`, 최대 토크비는 관절별 effort
`[45,30,30,45,20,20] x 2`로 나눈 절댓값의 최댓값이다.

## 관련 원자료

- 역사적 I3b 장기 로그: `../2026-08-09_t_stand_i3b.csv`
- 현재 세션 전체 대화·도구 출력:
  `/Users/dmdrb/.codex/sessions/2026/08/09/rollout-2026-08-09T11-09-13-019fe448-0547-7f11-9973-2d9c7493f3e7.jsonl`
- Claude Deployer 원세션:
  `/Users/dmdrb/.claude/projects/-Users-dmdrb-RoboCup-k1-goalpose/6b2a2a27-b7e8-4400-be55-c45c83d03289.jsonl`

