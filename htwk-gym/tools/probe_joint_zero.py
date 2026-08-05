"""관절 영점 오차를 로봇을 움직이지 않고 잰다 (읽기 전용).

FK는 쓰지 않는다 -- 평행 발목 때문에 직렬 체인으로 안 풀린다(2026-08-05에 시도했다가
몸통 pitch -34도, 발이 몸 앞 23 cm라는 값이 나왔다). 대신 **좌우 대칭**을 기준으로 쓴다.
URDF에서 좌우 관절 축이 동일하고(mirror 아님) 학습 기본자세는 roll/yaw가 전부 0이므로,
대칭으로 서 있으면 pitch는 L=R, roll/yaw는 L=R=0이어야 한다.

⛔ 값은 **반드시 콜백 안에서** 복사한다. 밖에서 msg.motor_state_serial을 반복해 읽으면
SDK가 그 버퍼를 재사용/해제해 떠 있는 참조가 된다 -- 2026-08-05에 그렇게 해서 7.7e28
같은 쓰레기 값, MemoryError, 그리고 segfault를 차례로 봤다.

⚠️ 한계: 좌우가 같은 방향으로 같이 틀어진 공통(common-mode) 오차는 안 잡힌다.
이 측정은 **하한**이다.
"""
import time, math, sys
import booster_robotics_sdk_python as B

NAMES = ["Hip_Pitch", "Hip_Roll", "Hip_Yaw", "Knee", "Ankle_Pitch", "Ankle_Roll"]
DEFAULT = [-0.2, 0.0, 0.0, 0.4, -0.25, 0.0]
samples = []

def cb(msg):
    try:
        ms = msg.motor_state_serial
        if len(ms) < 22:
            return
        q = [float(ms[i].q) for i in range(22)]          # 콜백 안에서 즉시 복사
        rpy = [float(v) for v in msg.imu_state.rpy]
        samples.append((q, rpy))
    except Exception:
        pass

B.ChannelFactory.Instance().Init(0, "127.0.0.1")
sub = B.B1LowStateSubscriber(cb); sub.InitChannel()
t0 = time.time()
while len(samples) < 150 and time.time() - t0 < 4:
    time.sleep(0.05)
# CloseChannel()이 걸리는 것을 봤다(rc=124). 프로세스 종료가 정리한다.

if len(samples) < 20:
    print("표본 부족: %d" % len(samples)); sys.stdout.flush(); __import__('os')._exit(1)
n = len(samples)
q = [sum(s[0][i] for s in samples) / n for i in range(22)]
rpy = [math.degrees(sum(s[1][k] for s in samples) / n) for k in range(3)]

bad = [i for i in range(10, 22) if not math.isfinite(q[i]) or abs(q[i]) > 3.2]
if bad:
    print("⛔ 관절 %s 읽기값이 물리 범위 밖: %s" % (bad, ["%.3g" % q[i] for i in bad]))
    sys.stdout.flush(); __import__('os')._exit(2)
if abs(rpy[1]) > 20.0 or abs(rpy[0]) > 20.0:
    print("⛔ 로봇이 서 있지 않다: roll %+.1f도 pitch %+.1f도" % (rpy[0], rpy[1]))
    print("   대칭 자세가 기준이라, 기울어져 있으면 영점이 아니라 기울기를 재게 된다.")
    sys.stdout.flush(); __import__('os')._exit(3)

print("표본 %d개, IMU roll %+.2f도  pitch %+.2f도" % (n, rpy[0], rpy[1]))
print()
print("%-13s %9s %9s %9s | %9s" % ("관절", "왼(도)", "오른(도)", "좌우차(도)", "기본자세"))
worst, wname = 0.0, ""
for k in range(6):
    L = math.degrees(q[10 + k]); R = math.degrees(q[16 + k])
    a = L - R
    if abs(a) > abs(worst): worst, wname = a, NAMES[k]
    flag = " ⛔" if abs(a) > 3.0 else (" ⚠️" if abs(a) > 2.0 else "")
    print("%-13s %+9.2f %+9.2f %+9.2f | %+9.2f%s" % (
        NAMES[k], L, R, a, math.degrees(DEFAULT[k]), flag))
print()
print("최대 좌우차: %s %+.2f도" % (wname, worst))
print()
print("== 판정 (MuJoCo 스윕: ±2도 낙상 0, ±3도 낙상 12) ==")
a = abs(worst)
if a >= 3.0:
    print("  ⛔ %.2f도 -- 절벽을 넘었다. 이 크기면 MuJoCo에서 넘어진다." % a)
elif a >= 2.0:
    print("  ⚠️ %.2f도 -- 버티는 상한 언저리." % a)
else:
    print("  ✅ %.2f도 -- ±2도 안. 차동 영점만으로는 증상이 설명되지 않는다." % a)
print("  ⚠️ 차동 성분만이다. 좌우 공통 오차는 안 잡힌다 -- 이 값은 하한.")
sys.stdout.flush()
import os; os._exit(0)   # SDK 정리 경로가 걸리므로 즉시 나간다
