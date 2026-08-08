# K1 액추에이터 벤더 공식 사양 (2026-08-08 확보)

**출처**: [`BoosterRobotics/booster_train`](https://github.com/BoosterRobotics/booster_train) —
Booster Robotics 공식 Isaac Lab 학습 저장소.
- `source/booster_train/booster_train/assets/robots/actuator.py` — 모터 모델 DB
- `source/booster_train/booster_train/assets/robots/booster.py` — `BOOSTER_K1_CFG` 관절↔모터 매핑

⛔ 이 문서가 **[DEPLOY_REQUESTS_FROM_TRAINING.md](DEPLOY_REQUESTS_FROM_TRAINING.md) R8 을 닫는다.**
그 전까지 이 값들은 저장소 어디에도 근거가 없었고, 나는 추측으로 메우다가 두 번 틀렸다
(RETRACTIONS **C31** "정격 20 N·m", **C30** 의 7.1 rad/s 전제).

---

## 1. K1 관절 ↔ 모터 매핑과 사양

`BOOSTER_K1_CFG.actuators` 에서 직접 읽었다. 발목은 `BoosterK1AnkleParaWrapperCfg`
(평행 기구)로 감싸여 `armature_ratio = (2.0, 2.0)`, `effort/velocity_ratio = (1.0, 1.0)` 이 곱해진다.

| 관절 | 모터 | effort (N·m) | velocity (rad/s) | **armature (kg·m²)** | 유도 kp | 유도 kd |
|---|---|---:|---:|---:|---:|---:|
| Hip_Pitch | E6408 | **68.0** | **14.66** | **0.047813** | 30.2 | 3.60 |
| Hip_Roll | E4315 | **76.0** | 12.57 | **0.033955** | 21.4 | 2.56 |
| Hip_Yaw | E4310 | **38.3** | 17.59 | **0.028253** | 17.8 | 2.13 |
| Knee_Pitch | E6416 | **112.0** | 12.57 | **0.095625** | 60.4 | 4.81 |
| Ankle_Pitch | E4310 ×평행 | **38.3** | 17.59 | **0.056506** | 35.7 | 4.26 |
| Ankle_Roll | E4310 ×평행 | **38.3** | 17.59 | **0.056506** | 35.7 | 4.26 |
| 팔 (4×2) | R14 | 14.0 | 33.51 | 0.001 | — | — |
| 머리 (2) | HT4438 | 6.0 | 7.85 | 0.001 | — | — |

`natural_freq = 4.0 Hz`, `damping_ratio = 1.5` (무릎만 1.0).

## 2. ⭐ 벤더는 게인을 armature 에서 **유도**한다 — 둘은 한 쌍이다

`actuator.py:161-163`:
```python
self.stiffness = self.armature * (2 * pi * self.natural_freq)**2
self.damping   = 2 * self.damping_ratio * self.armature * (2 * pi * self.natural_freq)
```

즉 **PD 게인은 자유 파라미터가 아니라 `armature × 목표 고유진동수`의 결과다.**
`armature = 0` 이면 이 관계가 성립하지 않는다 — 관성이 없는 관절에 유한한 kp 를 주면
고유진동수가 발산한다.

우리 게인을 벤더 armature 로 역산한 등가 고유진동수:

| 관절 | 우리 kp | 벤더 armature | 등가 f_n | 벤더 f_n |
|---|---:|---:|---:|---:|
| Ankle_Roll | 50 | 0.056506 | **4.73 Hz** | 4.0 |
| Knee | 100 | 0.095625 | **5.15 Hz** | 4.0 |
| Hip_Pitch | 100 | 0.047813 | **7.28 Hz** | 4.0 |

⇒ 우리 게인 자체는 벤더 대역과 같은 자리에 있다(4.7~7.3 대 4.0). **문제는 게인이 아니라
그 게인이 가정하는 관성이 시뮬에 없다는 것이다.**

## 3. ⛔ 우리 자산과의 차이 — 세 축 전부 어긋난다

| 관절 | 벤더 effort | `boxfoot` | `armsdown`(배포 계보) | 벤더 vel | `boxfoot` | 학습 계열 |
|---|---:|---:|---:|---:|---:|---:|
| Hip_Pitch | **68.0** | 30 | 30 | **14.66** | 7.1 | 18 |
| Hip_Roll | **76.0** | 35 | **20** | 12.57 | 12.9 | 18 |
| Hip_Yaw | **38.3** | 20 | 20 | 17.59 | 18.1 | 18 |
| Knee | **112.0** | 40 | 40 | 12.57 | 12.5 | 18 |
| Ankle_P | **38.3** | 20 | 20 | 17.59 | 18.1 | 18 |
| Ankle_R | **38.3** | 20 | **15** | 17.59 | 18.1 | 18 |

⭐ **우리 URDF 의 effort 가 벤더의 절반 이하다** — 무릎은 **2.8배** 차이(40 대 112).
학습이 토크 한계를 `goal_pose.py:866` 에서 **하드 클램프**하므로,
**정책은 로봇이 실제로 낼 수 있는 토크의 절반 이하로 학습됐다.**

⚠️ 방향에 주의: 이것은 "실기에 없는 토크를 쓰도록 배웠다"가 **아니라** 그 반대다.
그 자체로는 안전하지만, 굶긴 상태에서 최적화된 보행은 실기에서 최적이 아니다.

⚠️ **속도**: 벤더 Hip_Pitch 14.66 이다. `boxfoot` 의 7.1 은 **절반이고**, 학습 계열의 18 은
과하다. **둘 다 틀렸다.** 다만 학습은 속도를 강제하지 않으므로(하드 클램프 없음,
벌칙 스케일 `-0.`) 이 값은 **측정 해석에만** 쓰인다.
⇒ RETRACTIONS **C30 의 "Hip_Pitch 7.1 초과" 판정 기준은 무효**다. 실제 한계는 14.66 이고
측정된 peak 가 ~8 이므로 **고관절은 여유가 있다.** 발목 roll 만 남는다
(벤더 17.59, 측정 평균 7.5~12.8, peak 76~92).

## 4. armature 값의 역사 — 어느 것도 벤더 값이 아니었다

| 출처 | 값 | 벤더 대비 |
|---|---|---|
| `Goal_Pose_V7.yaml` · `Goal_Pose.yaml` (**현재 학습 + 배포된 정책**) | **0** | ⛔ 전부 틀림 |
| `Goal_Pose_V3` · `Safe_Fall` · `Get_Up` ("official Booster USD value") | 0.02 | ⛔ 어느 관절에도 없음 (실제 0.028~0.096) |
| MuJoCo `K1_serial*.xml` | 발목 **0.05** / 힙·무릎 0 | 발목은 **가장 근접**(벤더 0.0565), 힙·무릎은 틀림 |
| **벤더 공식** | 관절별 **0.0283~0.0956** | — |

⚠️ `Get_Up.yaml:92` 의 주석 *"official Booster USD value"* 는 **틀렸거나 다른 로봇 값이다.**
0.02 는 K1 의 어느 관절에도 해당하지 않는다.

⚠️ 참고로 **Booster 자신의 Isaac Gym 프레임워크(`booster_gym/envs/T1.yaml`)도 `armature: 0.`** 이다.
즉 벤더도 Isaac **Gym** 쪽에서는 0 을 쓴다. 관절별 armature 는 Isaac **Lab** 파이프라인
(`booster_train`)에서 도입됐다. **"벤더가 0 을 쓴다"와 "벤더가 관절별 값을 안다"는 둘 다 참이다** —
새 파이프라인이 옛 것보다 정확하다는 뜻으로 읽는 것이 타당하다.

## 5. 우리가 안 모델링하는 것 둘 (벤더는 기본으로 켠다)

1. **구동 지연** — `BoosterDelayedPDActuatorCfg(max_delay=8, min_delay=2)`.
   **모든 관절에 2~8 스텝 지연이 기본이다.** 우리 arm 대부분은 `obs_delay_steps: [0,0]` 이다.
2. **토크-속도 곡선** — `knee_point_velocity` 로 속도에 따른 토크 저하를 모델링한다.
   우리는 속도와 무관한 상수 클램프다. 실제 모터는 고속에서 토크가 준다.

## 6. ⛔ 미해결 — 벤더 문서끼리 어긋난다

공개 페이지([booster.tech](https://www.booster.tech/booster-k1/))는 **`Max Peak Torque 60 N·m`**
단일 값을 광고하는데, `booster_train` 의 액추에이터 DB 는 무릎에 **112 N·m** 를 준다.
- 60 이 연속 정격이고 112 가 피크인가?
- 아니면 60 이 마케팅 값이고 112 가 시뮬 상한(여유 포함)인가?
**미확인.** effort 를 벤더 값으로 올리기 전에 이것부터 갈라야 한다 —
112 로 학습해 놓고 실기가 60 에서 잘리면 지금과 반대 방향의 같은 실수다.

## 7. 다음 걸음

1. ⭐ **관절별 armature 지원을 넣고 벤더 값으로 arm 을 세운다.**
   `asset.armature` 는 스칼라라 지금 구조로는 못 넣는다. Isaac Gym 은 `dof_props["armature"]`
   로 DOF 별 설정이 가능하다.
   ⚠️ 지금 도는 `NG_armature`(0.02)는 **"0 이냐 아니냐"** 는 답하지만 **벤더 값이 아니다.**
   정성 판정용으로만 읽어라.
2. **effort 를 벤더 값으로 올리는 arm** — 단, §6 을 먼저 닫는다.
3. **구동 지연 2~8 스텝**을 기본으로 켜는 것을 검토한다(벤더 기본값이다).
4. 발목 roll 속도 이상(한계의 27~55 % 초과)이 armature 로 설명되는지는
   `NG_armature` 와 MuJoCo A/B 가 답한다.
