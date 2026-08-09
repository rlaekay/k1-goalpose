"""`_reward_feet_cross` 부호 회귀 검사. 옛 식(abs)이 반드시 **실패**해야 한다 = 검출력 확인."""
def new(sep, lo=0.07, hi=0.26):
    return max(lo - sep, 0.0) + max(sep - hi, 0.0)
def old(sep, lo=0.07, hi=0.26):
    gap = abs(sep)
    return max(lo - gap, 0.0) + max(gap - hi, 0.0)

fails = 0
# 1. 교차는 깊어질수록 더 벌받아야 한다 (단조)
prev = -1.0
for s in (-0.03, -0.07, -0.10, -0.15, -0.20, -0.30):
    v = new(s)
    if v <= prev:
        print("⛔ 단조 아님: sep=%.2f 벌칙 %.3f <= 직전 %.3f" % (s, v, prev)); fails += 1
    prev = v
# 2. 실기 서명(-0.164)이 무벌칙이면 안 된다
if new(-0.164) <= 0: print("⛔ 실기 p1 -0.164 무벌칙"); fails += 1
# 3. 정상 자세는 무벌칙
for s in (0.10, 0.18, 0.25):
    if new(s) != 0.0: print("⛔ 정상 %.2f 가 벌받는다" % s); fails += 1
# 4. 너무 벌어지면 벌받는다
if new(0.30) <= 0: print("⛔ 0.30 무벌칙"); fails += 1
# 5. 음성 대조 -- 옛 식은 위 1·2를 **반드시** 어긴다
neg = 0
if old(-0.20) != 0.0: neg += 1
if old(-0.10) != 0.0: neg += 1
if old(-0.30) <= old(-0.20): neg += 1
if neg:
    print("⛔ 음성 대조 실패: 옛 식이 결함을 안 보인다"); fails += 1
else:
    print("음성 대조 OK -- 옛 식은 -0.10/-0.20 교차를 무벌칙으로 통과시킨다")
print("PASS" if not fails else "FAIL %d" % fails)
