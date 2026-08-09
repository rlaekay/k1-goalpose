# 배포 모델 provenance

체크포인트 파일명에는 iteration 이 없다. **어느 iteration 인지가 파일명에서 안 보이는
것** 때문에 이 저장소는 이미 한 번 크게 데였다(best.pth 가 arm 마다 다른 iteration 을
가리켜 비교가 교락, RETRACTIONS C3). 여기 박아 둔다.

| 파일 | 원본 | md5 | 실기 이력 |
|---|---|---|---|
| `goal_pose_nh.pt` | `NH_zeroclock/model_6000` | `c9e206104ff2c147e4e6f951120f6cfb` | R5 deadline 서기 1 s (Codex 감사 §8) |
| `goal_pose_np.pt` | `NP_clockarm/model_6000` | `c72aae2b7e0f99b77778af55f7f2a27b` | R1~R4 (Codex 감사 §5~8) |
| (로봇의) `goal_pose_i3b.pt` | `I3b_stance10/model_200` | `60006446e638566c8245bc6013c8780c` | 현재 실기 후보. R3 종결 |

⛔ **`goal_pose_nh.pt` = it6000 은 그 arm 의 최고점이 아닐 수 있다** (Teacher,
2026-08-09 저녁): waypoint 축에서 **it100 이 도착률 1.8배(34.6 % 대 18.9 %) ·
위치오차 1.6배(5.96 대 9.33 cm)** 좋다. 생존 편향 아님(it100 이 낙상도 많은데 두 지표
다 이김). 보행 축 격자(`052-grid_walk_NH_NF`) 결과가 나오기 전까지 **"NH 는
model_6000 이 최고" 인용 금지**, export 교체도 보류.

⚠️ 실기 쪽 참고: 실물 서기 스크린(R4~R6)에서 NH(it6000)는 I3b 대비 몸 각속도 p95
**2.34배**였다 — sim 두 축을 다 이겨도 실기 후보 순위는 별도 문제다.
