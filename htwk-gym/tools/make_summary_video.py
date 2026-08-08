"""MJC 세션이 한 일과 앞으로 할 일을 2 분짜리 mp4 로 남긴다.

왜 파일로 만드나: 이 세션의 결과가 여러 문서(§8-45/47/48, RETRACTIONS, 인계문서)에
흩어져 있어 한 번에 보기 어렵다. 슬라이드를 코드로 만들면 숫자가 바뀔 때 다시 뽑을 수
있고, 손으로 만든 그림처럼 낡지 않는다.

숫자는 전부 이 저장소의 실측이다. 추정이나 예시는 넣지 않는다 -- 넣어야 한다면
슬라이드에 '미확인'이라고 적는다.

    python tools/make_summary_video.py --out logs/mujoco/summary.mp4
"""

import os
import sys
import argparse

import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
BG = (16, 18, 22)
FG = (232, 234, 238)
DIM = (150, 156, 166)
OK = (110, 200, 130)
BAD = (232, 108, 100)
WARN = (232, 190, 100)
ACC = (120, 175, 235)

FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_R = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def F(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT_R, size)


def slide(title, lines, kicker=None, bar=None):
    """lines: (text, colour, size, indent) 튜플 목록."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 10, H], fill=ACC)
    y = 56
    if kicker:
        d.text((60, y), kicker, font=F(24), fill=ACC)
        y += 40
    d.text((60, y), title, font=F(46, True), fill=FG)
    y += 84
    d.line([(60, y), (W - 60, y)], fill=(52, 56, 64), width=2)
    y += 34
    for text, col, size, ind in lines:
        if text == "":
            y += size // 2
            continue
        d.text((60 + ind, y), text, font=F(size, size >= 32), fill=col)
        y += size + 16
    if bar:
        draw_bars(d, bar, y + 6)
    return img


def draw_bars(d, spec, y0):
    """spec: (label, value, vmax, colour) 목록. 가로 막대."""
    x0, wmax = 300, 700
    for label, val, vmax, col in spec:
        d.text((60, y0 - 4), label, font=F(24), fill=DIM)
        w = int(wmax * min(val / vmax, 1.0))
        d.rectangle([x0, y0, x0 + wmax, y0 + 26], fill=(38, 42, 50))
        if w > 0:
            d.rectangle([x0, y0, x0 + w, y0 + 26], fill=col)
        d.text((x0 + wmax + 16, y0 - 2), ("%g" % val), font=F(22), fill=FG)
        y0 += 40


def build():
    S = []
    T = lambda t, c=FG, s=28, i=0: (t, c, s, i)

    S.append(slide(
        "MuJoCo 세션 — 한 일과 다음 수",
        [T("실기 증상: 서 있기는 완벽한데 세 걸음에서 무너진다", DIM, 30),
         T(""),
         T("근본 원인 둘을 재현했고, 둘 다 수정안이 코드에 이미 있다.", FG, 32),
         T(""),
         T("① 배포 필터가 발행 루프 속도에 의존해 보행을 삼킨다", OK, 28),
         T("② hold 자세(합 −0.05)가 개루프로 설 수 없다", OK, 28)],
        kicker="2026-08-07 ~ 08-08"))

    S.append(slide(
        "① 필터 — 계수는 상수인데 시정수는 상수가 아니다",
        [T("deploy_goal_pose.py:1430", ACC, 26),
         T("filtered = filtered * 0.8 + target * 0.2", FG, 30, 20),
         T(""),
         T("0.2 는 500 Hz 발행을 가정하고 고른 값 (시정수 10 ms).", DIM, 26),
         T("발행 루프는 파이썬이고 time.sleep(0.001) 이 있다.", DIM, 26),
         T(""),
         T("500 Hz → 시정수 10 ms · 차단 17.8 Hz · 2 Hz 통과 0.99", FG, 26),
         T(" 50 Hz → 시정수 90 ms · 차단 1.78 Hz · 2 Hz 통과 0.66", BAD, 26),
         T(""),
         T("보행이 2 Hz 다. 차단이 1.78 Hz 면 보행 신호가 걸린다.", WARN, 28)],
        kicker="발견 ①"))

    S.append(slide(
        "① 증거 — 발행률만 바꿔 재현했다",
        [T("배포 정책 그대로, 실물 질량 19.666 kg, 120 초", DIM, 26),
         T("낙상 / 구간", DIM, 24)],
        kicker="발견 ①",
        bar=[(" 50 Hz", 66, 70, BAD), (" 75 Hz", 43, 70, BAD),
             ("100 Hz", 10, 70, WARN), ("150 Hz", 4, 70, WARN),
             ("200 Hz", 0, 70, OK), ("500 Hz", 0, 70, OK)]))

    S.append(slide(
        "① 수정이 전 구간을 덮는다",
        [T("연속보행 · 관측지연 20 ms · IMU 잡음 · seed 3 개", DIM, 26),
         T(""),
         T("발행 Hz      현재(계수 0.2)      수정(시정수 고정)", DIM, 26),
         T("  150          0 / 6 / 0            0 / 0 / 0", FG, 28),
         T("  200        31 / 29 / 24           0 / 0 / 0", FG, 28),
         T(""),
         T("→ --rate-fixed-filter 는 50·150·200·500 Hz 전부 낙상 0", OK, 30),
         T("→ 500 Hz 에서는 켜나 안 켜나 같다 = 손해가 없다", OK, 28),
         T("→ pub_hz 실측(R6)이 선행조건에서 빠진다", OK, 28)],
        kicker="발견 ①"))

    S.append(slide(
        "② hold 자세 — 개루프로 설 수 없다",
        [T("b(CUSTOM 진입) ~ r(보행 시작) 구간엔 균형 제어기가 없다.", DIM, 26),
         T("정책이 균형 제어기인데 아직 안 돈다. 남는 건 위치 서보뿐.", DIM, 26),
         T(""),
         T("평면에서 곧게 서려면  hip + knee + ankle = 0", FG, 28),
         T("A prepare  −0.10 + 0.20 − 0.10 =  0.00", OK, 28, 20),
         T("B RL 자세  −0.20 + 0.40 − 0.25 = −0.05", BAD, 28, 20),
         T(""),
         T("MuJoCo 개루프 PD, prepare 게인, 실물 질량:", DIM, 26),
         T("A → tilt 0.07°   완벽히 섬", OK, 30, 20),
         T("B → tilt 14.11° (1 초) → 91.9° 완전 전도", BAD, 30, 20)],
        kicker="발견 ②"))

    S.append(slide(
        "② 실기 두 값이 다 재현됐다",
        [T("실기 실측            MuJoCo", DIM, 26),
         T(""),
         T("손 뗌      13.8°      14.11°   (차이 0.3°)", OK, 32),
         T("발 잡음     4.9°       4.5°    (2.86 기구학 + 1.6 기준)", OK, 32),
         T(""),
         T("기전: CoP 가 0.4 초 만에 +82.7 mm 로 가서 붙는다.", WARN, 28),
         T("발 앞경계는 +94 mm. 그 뒤 tilt 만 자란다.", WARN, 28),
         T(""),
         T("발목 토크는 8.8 / 20 N·m = 44 %. 여유가 있다.", FG, 28),
         T("→ 토크 한계가 아니라 지지면 한계다.", ACC, 30)],
        kicker="발견 ②"))

    S.append(slide(
        "기각한 가설 — 죽인 것도 결과다",
        [T("Isaac 특유 물리        MuJoCo 에서도 낙상 0", DIM, 27),
         T("토크 상한              벤더/학습 상한 바꿔도 동일", DIM, 27),
         T("배포 필터 존재 자체    500 Hz 로 넣으면 무해", DIM, 27),
         T("질량·관성 오차         실물 19.666 kg → 6.73 → 6.70 cm", DIM, 27),
         T("떨림·채터링            Nyquist 대역 0.3~1.8 %", DIM, 27),
         T("관측 지연 단독         수정 필터로 200 Hz 이상현상 소멸", DIM, 27),
         T("관절 속도 초과         hip pitch p99 = 한계의 68~75 %", DIM, 27),
         T(""),
         T("일곱 개가 죽었다. 남은 것은 필터의 속도 의존 하나.", FG, 30)],
        kicker="누적"))

    S.append(slide(
        "내가 틀렸던 것 — 셋",
        [T("① --seed 가 있으면 변동이 생긴다고 가정했다", BAD, 28),
         T("goal-hold + 잡음 0 은 결정론적. 세 번이 같은 실행이었다.", DIM, 25, 20),
         T(""),
         T("② 반전율 하나로 떨림을 판정하려 했다", BAD, 28),
         T("무릎 22.9 회/초는 채터링이 아니라 보행 조화성분이었다.", DIM, 25, 20),
         T(""),
         T("③ 한 축의 문턱을 다른 축이 있는 상황에 일반화했다", BAD, 28),
         T("\"≥200 Hz 면 안전\" 이 지연 20 ms 를 넣자 깨졌다.", DIM, 25, 20),
         T(""),
         T("셋 다 인계문서 §7 에 남겼다. 다음 세션이 안 밟도록.", FG, 27)],
        kicker="방법론"))

    S.append(slide(
        "지금 확인 중 — 관절 속도",
        [T("학습 자산은 전 관절 velocity = 18 (자리표시자로 보인다)", DIM, 26),
         T("실물 계열은 hip pitch 7.1 / knee 12.5 로 비균일", DIM, 26),
         T(""),
         T("배포 정책이 실제로 내는 |dq| 대 실물 한계:", FG, 27),
         T("R_HipP  p99 5.34 / 7.1     초과 0.17 %", FG, 27, 20),
         T("R_Knee  p99 9.40 / 12.5    초과 0.23 %", FG, 27, 20),
         T("나머지 8 관절              초과 0.00 %", FG, 27, 20),
         T(""),
         T("→ 사전 고정한 기각 조건에 걸린다. 이 축은 약하다.", WARN, 28),
         T("→ 다만 좌우 비대칭 3.3 배는 남는다 (미확인)", DIM, 26)],
        kicker="진행 중"))

    S.append(slide(
        "앞으로 — 시뮬에서",
        [T("8'  관절 속도 상한을 실물 값으로 걸고 같은 정책 실행", FG, 28),
         T("(B) 실물 상한 + (C) p99 상한으로 감도까지", DIM, 25, 20),
         T(""),
         T("6   개루프 hold 4 셀 + tilt FFT", FG, 28),
         T("{합 0, 합 −0.05} × {prepare 250, 정책 50}", DIM, 25, 20),
         T("0.35 Hz 흔들림이 정책 게인에서만 뜨는지", DIM, 25, 20),
         T(""),
         T("7   질량 감도 18.714 대 19.666 kg", FG, 28),
         T("여유 ×1.27 → ×1.21 이 거동에 보이는지", DIM, 25, 20)],
        kicker="다음 수"))

    S.append(slide(
        "앞으로 — 로봇에서",
        [T("가장 값싼 확인 한 줄:", DIM, 26),
         T("deploy_goal_pose.py --rate-fixed-filter --log-timing", ACC, 27),
         T(""),
         T("손해가 없다. 유죄면 즉시 해결된다.", OK, 28),
         T(""),
         T("⛔ 서 있기로는 필터 버그를 못 잡는다 (실측 확인)", BAD, 27),
         T("50 Hz 와 500 Hz 가 정지 상태에서 구별 안 됨", DIM, 25, 20),
         T("저역통과는 상수를 그대로 통과시키기 때문", DIM, 25, 20),
         T(""),
         T("다만 pub_hz 측정은 세워둔 채로 안전하게 가능하다.", FG, 27),
         T("hold 자세는 prepare(합 0)만 쓴다 — 수정안 ②", FG, 27)],
        kicker="다음 수"))

    S.append(slide(
        "한 줄로",
        [T(""),
         T("필터 수정을 켜고, hold 자세를 합 0 으로 바꾼다.", FG, 38),
         T(""),
         T("둘 다 코드에 이미 있고, 둘 다 손해가 없다.", OK, 32),
         T(""),
         T("남은 미지수는 실기에서 한 번 걸어보는 것뿐.", DIM, 30),
         T(""),
         T(""),
         T("근거: ibatch §8-45 / §8-47 / §8-48", DIM, 24),
         T("인계: HANDOFF_MUJOCO_FILTER_20260807.md", DIM, 24)],
        kicker="결론"))
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/mujoco/summary.mp4")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--total", type=float, default=120.0)
    a = ap.parse_args()

    slides = build()
    per = int(round(a.total * a.fps / len(slides)))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with imageio.get_writer(a.out, fps=a.fps, macro_block_size=1) as w:
        for img in slides:
            fr = np.asarray(img)
            for _ in range(per):
                w.append_data(fr)
    print("슬라이드 %d 장 x %.1f 초 = %.0f 초 -> %s"
          % (len(slides), per / a.fps, len(slides) * per / a.fps, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
