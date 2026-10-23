"""
수구 전생애(Start-to-Stop) 궤적 방향·변곡점·극소점 돌출도(Prominence) 3쿠션 판정 검증 스위트
파일 경로: tests/test_cue_lifecycle_tracking.py
"""

from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scoring_fsm import (
    BallColor,
    CausalKissSafeFSM,
    CueBallLifecycleTracker,
    ShotResult,
    TABLE_H,
    TABLE_W,
)


def _interpolate_waypoints(
    waypoints: List[Tuple[float, float]], steps_per_seg: int = 8
) -> List[Tuple[float, float]]:
    """경유점 리스트를 등간격 선형 보간하여 프레임별 좌표 시퀀스를 생성합니다."""
    pts: List[Tuple[float, float]] = []
    for i in range(len(waypoints) - 1):
        p0 = np.array(waypoints[i], dtype=np.float64)
        p1 = np.array(waypoints[i + 1], dtype=np.float64)
        for s in range(steps_per_seg):
            alpha = s / float(steps_per_seg)
            p = p0 + alpha * (p1 - p0)
            pts.append((float(p[0]), float(p[1])))
    pts.append(waypoints[-1])
    return pts


def test_lifecycle_start_to_stop_full_event_timeline() -> None:
    """
    1. 수구(WHITE)가 정지 상태에서 출발 -> 1적구(RED) 충돌(각도 25도 굴절)
       -> 1쿠션(TOP 극소점) -> 2쿠션(RIGHT 극소점) -> 3쿠션(BOTTOM 극소점)
       -> 2적구(YELLOW) 충돌(즉시 득점!) -> 수구 완전 정지까지의 전생애 이벤트가
       시간순으로 정확히 기록되고 scored == True, cushions_at_second_hit == 3 검증.
    """
    immediate_results: List[ShotResult] = []
    fsm = CausalKissSafeFSM(
        fps=30.0,
        assigned_cue_ball=BallColor.WHITE,
        on_immediate_score=lambda r: immediate_results.append(r),
    )
    tracker = CueBallLifecycleTracker()
    dt = 1.0 / 30.0
    t = 0.0

    w_init = (400.0, 600.0)
    r_init = (900.0, 420.0)
    y_init = (1600.0, 650.0)

    # 1) 초기 정지 앵커 고정 (IDLE, 5프레임)
    for _ in range(5):
        coords = {BallColor.WHITE: w_init, BallColor.RED: r_init, BallColor.YELLOW: y_init}
        fsm.update(t, coords)
        t += dt

    # 2) WHITE 출발 -> RED(900, 420) 타격 후 약 25도 굴절하여 TOP(1450, 38) -> RIGHT(2804, 650)
    #    -> BOTTOM(2150, 1384) -> YELLOW(1600, 650) 타격 -> (1480, 520) 정지
    cue_waypoints = [
        w_init,
        (865.0, 432.0),    # 제1적구(RED) 접촉
        (1450.0, 38.0),    # 1쿠션: TOP 극소점
        (2804.0, 650.0),   # 2쿠션: RIGHT 극소점
        (2150.0, 1384.0),  # 3쿠션: BOTTOM 극소점
        (1625.0, 675.0),   # 제2적구(YELLOW) 접촉
        (1480.0, 520.0),   # 타격 후 감속 이동
    ]
    cue_pts = _interpolate_waypoints(cue_waypoints, steps_per_seg=8)

    r_pos = np.array(r_init, dtype=np.float64)
    y_pos = np.array(y_init, dtype=np.float64)
    final_res = None

    for idx, wp in enumerate(cue_pts):
        # RED 충돌(idx >= 8) 이후 RED가 법선 방향으로 밀려남
        if idx >= 8:
            r_pos = np.array([900.0 + min(idx - 7, 12) * 8.0, 420.0 + min(idx - 7, 12) * 5.0])
        # YELLOW 충돌(idx >= 40) 이후 YELLOW가 법선 방향으로 밀려남
        if idx >= 40:
            y_pos = np.array([1600.0 - min(idx - 39, 10) * 8.0, 650.0 - min(idx - 39, 10) * 5.0])

        r_tuple = (float(r_pos[0]), float(r_pos[1]))
        y_tuple = (float(y_pos[0]), float(y_pos[1]))
        v_est = (np.array(wp) - np.array(cue_pts[max(0, idx - 1)])) / dt
        tracker.append_sample(
            t,
            np.array(wp, dtype=np.float64),
            v_est,
            r_pos.copy(),
            y_pos.copy(),
        )

        res = fsm.update(t, {BallColor.WHITE: wp, BallColor.RED: r_tuple, BallColor.YELLOW: y_tuple})
        if res is not None and res.is_turn_end:
            final_res = res
        t += dt

    # 3) 수구 및 모든 공 정지 유지 (0.60초 이상)
    last_w = cue_pts[-1]
    last_r = (float(r_pos[0]), float(r_pos[1]))
    last_y = (float(y_pos[0]), float(y_pos[1]))
    for _ in range(22):
        res = fsm.update(t, {BallColor.WHITE: last_w, BallColor.RED: last_r, BallColor.YELLOW: last_y})
        if res is not None:
            final_res = res
        t += dt

    assert len(immediate_results) == 1, f"즉시 득점 콜백이 1회 호출되어야 합니다: {immediate_results}"
    imm = immediate_results[0]
    assert imm.scored is True
    assert imm.cushions_at_second_hit == 3, f"3쿠션 득점이어야 합니다: {imm.cushions_at_second_hit}"
    assert imm.shot_event_tokens == ["B1", "C", "C", "C", "B2"], f"이벤트 순서 불일치: {imm.shot_event_tokens}"
    assert fsm.assigned_cue_ball == BallColor.WHITE, "득점 성공 후 수구(WHITE)가 유지되어야 합니다."
    assert len(tracker.cue_lifecycle_path) > 30
    print("[PASS] test_lifecycle_start_to_stop_full_event_timeline")


def test_Dead_center_straight_hit_on_first_and_second_target() -> None:
    """
    2. 수구가 목적구를 정면(0도)으로 두껍게 쳐서 수구 각도는 꺾이지 않고(0도 직진) 속력만 감소하며
       목적구가 앞으로 튕겨 나가는 정면 타격(형태 3) 시나리오에서도 1적구 및 2적구 충돌을 정확히 인식하는지 검증.
    """
    dt = 1.0 / 30.0

    # CausalKissSafeFSM 통합 시나리오: 1적구 정면(0도) 타격 -> 3쿠션 -> 2적구 정면(0도) 감속 타격
    fsm = CausalKissSafeFSM(fps=30.0, assigned_cue_ball=BallColor.WHITE)
    t_fsm = 0.0
    w0, r0, y0 = (300.0, 300.0), (520.0, 300.0), (1500.0, 700.0)
    for _ in range(5):
        fsm.update(t_fsm, {BallColor.WHITE: w0, BallColor.RED: r0, BallColor.YELLOW: y0})
        t_fsm += dt

    # 1적구(520, 300)를 0도 직선으로 정면 타격 후 계속 y=300으로 조금 전진하다가
    # TOP -> RIGHT -> BOTTOM 3쿠션 후 y=700 수평선 위에서 제2적구(1500, 700)를 0도 정면 감속 타격!
    seg1 = _interpolate_waypoints([(300.0, 300.0), (462.0, 300.0)], steps_per_seg=5)
    seg1_follow = _interpolate_waypoints([(462.0, 300.0), (500.0, 300.0)], steps_per_seg=3)[1:]
    seg_cushions = _interpolate_waypoints(
        [
            (500.0, 300.0),
            (1100.0, 38.0),    # 1쿠션 (TOP)
            (2804.0, 500.0),   # 2쿠션 (RIGHT)
            (2100.0, 1384.0),  # 3쿠션 (BOTTOM)
            (1750.0, 700.0),   # 제2적구 정면 진입 정렬
            (1558.0, 700.0),   # 제2적구(1500, 700) 정면(0도) 타격 지점
        ],
        steps_per_seg=7,
    )[1:]
    # 제2적구 정면 타격 직후 각도 꺾임(0도) 없이 속력만 급감하여 천천히 전진
    seg2_decel = [(1550.0, 700.0), (1543.0, 700.0), (1537.0, 700.0), (1532.0, 700.0)]
    all_w_pts = seg1 + seg1_follow + seg_cushions + seg2_decel

    r_cur = np.array(r0, dtype=np.float64)
    y_cur = np.array(y0, dtype=np.float64)
    hit2_idx = len(seg1) + len(seg1_follow) + len(seg_cushions) - 1
    scored_flag = False

    for i, wp in enumerate(all_w_pts):
        if i >= len(seg1):
            r_cur = np.array([520.0 + min(i - len(seg1) + 1, 10) * 18.0, 300.0])
        if i > hit2_idx:
            y_cur = np.array([1500.0 - (i - hit2_idx) * 16.0, 700.0])
        res = fsm.update(
            t_fsm,
            {
                BallColor.WHITE: wp,
                BallColor.RED: (float(r_cur[0]), float(r_cur[1])),
                BallColor.YELLOW: (float(y_cur[0]), float(y_cur[1])),
            },
        )
        if res is not None and res.scored:
            scored_flag = True
        t_fsm += dt

    assert fsm.first_target_ball == BallColor.RED, "1적구 정면 타격이 인식되어야 합니다."
    assert fsm.second_target_ball == BallColor.YELLOW, "2적구 정면 타격이 인식되어야 합니다."
    assert scored_flag or fsm.score_dispatched, "정면 타격 3쿠션 후 즉시 득점이 확정되어야 합니다."
    print("[PASS] test_Dead_center_straight_hit_on_first_and_second_target")


def test_valley_prominence_ignores_rail_jitter_and_catches_shallow_bounce() -> None:
    """
    3. 벽면 근처 미세 떨림(±5mm)은 돌출도 미달로 무시하고,
       시차 오차로 벽에서 90mm 앞에서 튕긴 완만한 입사각 쿠션은 극소점 돌출도로 정확히 잡아내는지 검증.
    """
    # Part A: 벽면 근처(y=65mm ± 5mm)에서 평행 주행하며 미세 떨림이 발생하는 경우 -> 쿠션 0회
    fsm_jitter = CausalKissSafeFSM(fps=30.0, assigned_cue_ball=BallColor.WHITE)
    dt = 1.0 / 30.0
    t = 0.0
    w_start = (300.0, 65.0)
    r_far = (2200.0, 900.0)
    y_far = (2400.0, 1100.0)

    for _ in range(5):
        fsm_jitter.update(t, {BallColor.WHITE: w_start, BallColor.RED: r_far, BallColor.YELLOW: y_far})
        t += dt

    jitter_offsets = [0.0, -4.5, 4.0, -5.0, 4.8, -3.5, 4.2, -4.8, 3.0, 0.0]
    for i in range(20):
        wx = 300.0 + (i + 1) * 45.0
        wy = 65.0 + jitter_offsets[i % len(jitter_offsets)]
        fsm_jitter.update(t, {BallColor.WHITE: (wx, wy), BallColor.RED: r_far, BallColor.YELLOW: y_far})
        t += dt

    assert fsm_jitter.cushion_count == 0, (
        f"벽면 근처 ±5mm 미세 떨림은 쿠션으로 카운트되면 안 됩니다: {fsm_jitter.cushion_count}"
    )

    # Part B: 시차 오차로 TOP 레일에서 90mm 앞(y=90mm)에서 튕겨 나온 완만한 입사각 쿠션 -> 정확히 1쿠션 인식
    fsm_shallow = CausalKissSafeFSM(fps=30.0, assigned_cue_ball=BallColor.WHITE)
    t = 0.0
    w_s0 = (400.0, 260.0)
    for _ in range(5):
        fsm_shallow.update(t, {BallColor.WHITE: w_s0, BallColor.RED: r_far, BallColor.YELLOW: y_far})
        t += dt

    shallow_pts = _interpolate_waypoints(
        [(400.0, 260.0), (1050.0, 90.0), (1700.0, 260.0)],
        steps_per_seg=8,
    )
    for wp in shallow_pts:
        fsm_shallow.update(t, {BallColor.WHITE: wp, BallColor.RED: r_far, BallColor.YELLOW: y_far})
        t += dt

    assert fsm_shallow.cushion_count == 1, (
        f"벽에서 90mm 앞 극소점(돌출도 충분) 반사는 정확히 1쿠션으로 인식되어야 합니다: {fsm_shallow.cushion_count}"
    )
    print("[PASS] test_valley_prominence_ignores_rail_jitter_and_catches_shallow_bounce")


if __name__ == "__main__":
    test_lifecycle_start_to_stop_full_event_timeline()
    test_Dead_center_straight_hit_on_first_and_second_target()
    test_valley_prominence_ignores_rail_jitter_and_catches_shallow_bounce()
    print("=== ALL CUE LIFECYCLE TRACKING TESTS PASSED! ===")
