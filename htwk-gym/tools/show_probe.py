"""forward_hold(정상상태 속도) 프로브 결과를 읽는다.

이 프로브는 목표를 로컬 프레임 2 m 앞에 0.8-1.2초마다 다시 놓는다. 로봇은 도착하지
못하므로 감속하지 않고 계속 걷는다. 그래서:

  * `body_speed`가 곧 이 정책의 **정상상태 속도**다 -- 이것만이 이 실행의 결과다
  * `pos_err_m` / `success_rate_*` / 게이트는 **의미가 없다.** 도착이 정의되지 않는
    프로브에서 도착 오차를 읽으면 안 된다. 그래서 여기서 아예 출력하지 않는다

부수적으로 이 조건이 스윙 정점을 재기에 가장 좋다. 정상 과제는 구간의 44%가 0.5 m
미만이라 로봇이 거의 서 있는데(8-15), 여기서는 쉬지 않고 걷는다.

    python tools/show_probe.py                 # logs/steady_i2a
    python tools/show_probe.py logs/<다른 run>
"""

import os
import sys
import json
import glob


def _g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
    return default if d is None else d


def show(path):
    d = json.load(open(path, encoding="utf-8"))
    print("=" * 78)
    print(os.path.relpath(os.path.dirname(path)))
    print("  checkpoint : %s" % os.path.basename(d.get("checkpoint") or "?"))
    # goal_pattern은 report.json 최상위에 없다(effective_protocol 안에서 해시될 뿐).
    # 라벨 대신 실제로 샘플링된 값을 찍는다 -- 프로브가 정말 돌았는지의 증거는
    # 이름이 아니라 goal_dx와 resample 주기다.
    cm = d.get("sampled_commands") or d.get("commands") or {}
    print("  바닥 %s | 외력 %s | 구간 %s" % (
        d.get("eval_terrain"), d.get("force_profile"), d.get("segments_completed")))
    print("  goal_dx %s  goal_dy %s  resample %s s" % (
        cm.get("goal_dx"), cm.get("goal_dy"), cm.get("resampling_time_s")))
    jd = d.get("sampled_joint_dr")
    if jd is not None:
        eb = jd.get("joint_encoder_bias")
        import math as _mm
        deg = ("%.1f도" % _mm.degrees(eb[1])) if eb and eb[1] else "0"
        print("  관절 영점 DR: encoder %s / target %s  -> ±%s" % (
            eb, jd.get("joint_target_offset"), deg))
    else:
        print("  관절 영점 DR: 미기록 — 이 실행은 조건을 파일에 안 남겼다."
              " 경로/체크포인트로만 구분된다")
    dur = d.get("duration_s") or 0
    ne = d.get("num_envs") or 0
    n = d.get("segments_completed") or 0
    seg_s = dur * ne / float(n) if (dur and ne and n) else 0.0
    if seg_s:
        print("  구간 길이 실측 %.2f s -> 순항 최장연속은 이 값에 상한이 걸린다"
              % seg_s)

    # forward_hold 판별: 목표 범위가 한 점([2,2])이면 도착하지 않도록 만든 프로브다.
    # 정상 과제는 [-2, 2]다. 라벨이 아니라 실제 샘플링 범위로 판별한다.
    dx = cm.get("goal_dx") or []
    is_probe = bool(d.get("goal_pattern")) or (
        len(dx) == 2 and dx[0] == dx[1] and dx[0] != 0.0)
    # 옛 report.json에는 goal_pattern도 sampled_commands도 없다. 그때는 구간 길이로
    # 가른다: 정상 과제는 resample 4-8 s라 구간이 4 s를 훨씬 넘고, forward_hold는
    # 0.8-1.2 s다. 판별에 실패해 도착 오차를 찍는 쪽이 훨씬 나쁜 결과라 보수적으로 본다.
    if not is_probe and seg_s and seg_s < 2.0:
        is_probe = True
    if not is_probe:
        # 정상 과제에서는 도착 지표가 결과다. 프로브에서만 숨긴다.
        pe = d.get("pos_err_m") or {}
        he = d.get("heading_err_deg") or {}
        import math as _m
        te = (_m.hypot(pe.get("median", float("nan")),
                       _m.radians(he.get("median", float("nan")))) * 100
              if pe.get("median") is not None and he.get("median") is not None
              else float("nan"))
        print("\n  [도착 — 정상 과제의 결과]")
        print("    위치 median %.2f cm  p90 %.2f cm | heading %.2f° | 과제오차 %.2f cm"
              % (100 * pe.get("median", float("nan")),
                 100 * pe.get("p90", float("nan")),
                 he.get("median", float("nan")), te))
        print("    strict %.1f%%  |  낙상 %s / 구간 %s"
              % (100 * (d.get("success_rate_strict") or 0.0),
                 d.get("falls"), d.get("segments_completed")))

    b = d.get("body_speed") or {}
    print("\n  [몸통속도 — 이 프로브의 결과]")
    print("    median %.2f   p90 %.2f   p99 %.2f   max %.2f m/s"
          % (b.get("median", float("nan")), b.get("p90", float("nan")),
             b.get("p99", float("nan")), b.get("max_instant", float("nan"))))
    print("    0.5 초과 체류 %.1f%%   1.0 초과 체류 %.1f%%"
          % (100 * b.get("share_above_0p5", 0.0),
             100 * b.get("share_above_1p0", 0.0)))
    print("    구간최고  median %.2f  p90 %.2f  max %.2f"
          % (b.get("segment_peak_median", float("nan")),
             b.get("segment_peak_p90", float("nan")),
             b.get("segment_peak_max", float("nan"))))
    s = b.get("sustained_1p3")
    if s:
        print("    순항(>=%.1f m 구간 %d개): 최장연속 median %.2f s / p90 %.2f s / max %.2f s"
              % (s.get("long_segment_min_dist_m", 2.0), s.get("n_long_segments", 0),
                 s.get("cruise_median_s", float("nan")),
                 s.get("cruise_p90_s", float("nan")),
                 s.get("cruise_max_s", float("nan"))))
    else:
        print("    순항: 미기록")

    sa = d.get("swing_apex_m")
    print("\n  [스윙 정점 — 지형 arm의 대조군]")
    if sa:
        print("    p10 %.1f cm   median %.1f cm   p90 %.1f cm   (스윙 %d회)"
              % (100 * sa["p10"], 100 * sa["median"], 100 * sa["p90"],
                 sa["n_swings"]))
        print("    2 cm 미만 %.1f%%   3 cm 미만 %.1f%%   (접촉 판정선 %.0f cm)"
              % (100 * sa["share_below_0p02"], 100 * sa["share_below_0p03"],
                 100 * sa["contact_threshold_m"]))
    else:
        print("    미기록 (이 실행은 swing_apex 계측 이전이다)")

    th = d.get("trunk_height_m")
    if th:
        print("\n  [몸통 높이 — 걸을 때 주저앉는가]")
        print("    전체    median %.3f m  p10 %.3f m" % (th["median"], th["p10"]))
        print("    이동중  median %.3f m  p10 %.3f m  (%d 표본)"
              % (th["walking_median"], th["walking_p10"], th["n_walking"]))
        print("    목표 %.3f m | 종료 임계 %.3f m | 이동중 처짐 %.0f mm"
              % (th["target"], th["terminate_height"],
                 1000 * (th["target"] - th["walking_median"])))

    fs = d.get("foot_support")
    print("\n  [지지 상태 — 낙상 개수를 대체하는 연속량]")
    if fs:
        print("    비행 %.1f%%   단일지지 %.1f%%   양발 %.1f%%"
              % (100 * fs["flight_share"], 100 * fs["single_support_share"],
                 100 * fs["double_support_share"]))
        ss = fs.get("single_support_s")
        if ss:
            print("    단일지지 구간  median %.2f s  p90 %.2f s  p99 %.2f s  (%d회)"
                  % (ss["median"], ss["p90"], ss["p99"], ss["n"]))
        sw = fs.get("stance_width_m")
        if sw:
            print("    발 간격(양발 접지 중)  p10 %.1f cm  median %.1f cm  p90 %.1f cm"
                  " | 기준 %.1f cm"
                  % (100 * sw["p10"], 100 * sw["median"], 100 * sw["p90"],
                     100 * sw["reference"]))
        la = fs.get("load_asymmetry")
        if la:
            print("    하중 비대칭 |L-R|/(L+R)  median %.2f  p90 %.2f  (%d 표본)"
                  % (la["median"], la["p90"], la["n"]))
    else:
        print("    미기록 (이 실행은 foot_support 계측 이전이다)")
    print()


def main():
    roots = sys.argv[1:] or ["logs/steady_i2a"]
    found = 0
    for root in roots:
        for p in sorted(glob.glob(os.path.join(root, "**", "report.json"),
                                  recursive=True)):
            show(p)
            found += 1
    if not found:
        print("report.json이 없다. 아직 도는 중이거나 경로가 다르다: %s" % roots)
        return 1
    print("주의: goal_dx가 한 점인 실행(forward_hold)에서는 도착 지표를 찍지 않는다 --")
    print("      도착하지 않도록 설계된 프로브라 도착 오차가 정의되지 않기 때문이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
