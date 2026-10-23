"""
GitHub pooltool 연속 사건 물리 시뮬레이션(TOI) 및 IMM 디지털 트윈 기반
수구 전생애 흐름 추적 모듈(cue_flow_physics_tracker.py) 검증 스위트
파일 경로: tests/test_cue_flow_physics.py
"""

from pathlib import Path
import sys
from typing import List, Optional, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cue_flow_physics_tracker import (
    CueFlowEvent,
    CueFlowPhysicsTwin,
    CueTrajectorySegment,
    FlowEventType,
)
from scoring_fsm import BallColor, CausalKissSafeFSM, ShotResult


def _interp_pts(
    waypoints: List[Tuple[float, float]], steps: int = 8
) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for i in range(len(waypoints) - 1):
        p0 = np.array(waypoints[i], dtype=np.float64)
        p1 = np.array(waypoints[i + 1], dtype=np.float64)
        for s in range(steps):
            alpha = s / float(steps)
            p = p0 + alpha * (p1 - p0)
            pts.append((float(p[0]), float(p[1])))
    pts.append(waypoints[-1])
    return pts


def test_detect_white_or_yellow_cue_ball_and_never_lose_flow() -> None:
    """
    1. 흰공 출발 시 WHITE로, 노란공 출발 시 YELLOW로 즉시 감지하고
       중간에 가려짐(None)이나 적구 충돌이 발생해도 정지할 때까지
       수구 흐름(flow_twin.segments 및 events)이 끊기지 않는지 검증.
    """
    dt = 1.0 / 60.0

    for expected_cue in (BallColor.WHITE, BallColor.YELLOW):
        fsm = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE)
        t = 0.0
        w0 = (400.0, 650.0) if expected_cue == BallColor.WHITE else (1650.0, 680.0)
        y0 = (1650.0, 680.0) if expected_cue == BallColor.WHITE else (400.0, 650.0)
        r0 = (900.0, 420.0)

        for _ in range(5):
            fsm.update(t, {BallColor.WHITE: w0, BallColor.YELLOW: y0, BallColor.RED: r0})
            t += dt

        cue_waypoints = [
            (400.0, 650.0),
            (865.0, 435.0),    # 제1적구(RED) 타격
            (1450.0, 38.0),    # 1쿠션(TOP)
            (2804.0, 650.0),   # 2쿠션(RIGHT)
            (2150.0, 1384.0),  # 3쿠션(BOTTOM)
            (1675.0, 700.0),   # 제2적구 타격
            (1520.0, 560.0),   # 감속 후 정지
        ]
        cue_pts = _interp_pts(cue_waypoints, steps=8)
        r_pos = np.array(r0, dtype=np.float64)
        other_pos = np.array(y0 if expected_cue == BallColor.WHITE else w0, dtype=np.float64)
        final_res: Optional[ShotResult] = None

        for idx, cp in enumerate(cue_pts):
            if idx >= 8:
                r_pos = np.array([900.0 + min(idx - 7, 12) * 8.0, 420.0 + min(idx - 7, 12) * 5.0])
            if idx >= 40:
                other_pos = np.array(
                    [1650.0 - min(idx - 39, 10) * 8.0, 680.0 - min(idx - 39, 10) * 5.0]
                )
            # 중간 주행 구간(idx 20, 21)에 2프레임 가려짐(None) 주입
            obs_cp: Optional[Tuple[float, float]] = None if idx in (20, 21) else cp
            r_t = (float(r_pos[0]), float(r_pos[1]))
            o_t = (float(other_pos[0]), float(other_pos[1]))
            coords = (
                {BallColor.WHITE: obs_cp, BallColor.YELLOW: o_t, BallColor.RED: r_t}
                if expected_cue == BallColor.WHITE
                else {BallColor.WHITE: o_t, BallColor.YELLOW: obs_cp, BallColor.RED: r_t}
            )
            res = fsm.update(t, coords)
            if res is not None:
                final_res = res
            t += dt

        last_c = cue_pts[-1]
        last_r = (float(r_pos[0]), float(r_pos[1]))
        last_o = (float(other_pos[0]), float(other_pos[1]))
        for _ in range(25):
            coords = (
                {BallColor.WHITE: last_c, BallColor.YELLOW: last_o, BallColor.RED: last_r}
                if expected_cue == BallColor.WHITE
                else {BallColor.WHITE: last_o, BallColor.YELLOW: last_c, BallColor.RED: last_r}
            )
            res = fsm.update(t, coords)
            if res is not None:
                final_res = res
            t += dt

        assert final_res is not None and final_res.scored is True
        assert fsm.flow_twin.locked_cue_ball == expected_cue, (
            f"수구 락킹 불일치: expected={expected_cue}, got={fsm.flow_twin.locked_cue_ball}"
        )
        ev_types = [ev.event_type for ev in fsm.flow_twin.events]
        assert ev_types[0] == FlowEventType.STROKE_START
        assert ev_types[-1] == FlowEventType.STROKE_STOP
        assert len(fsm.flow_twin.segments) >= 6
    print("[PASS] test_detect_white_or_yellow_cue_ball_and_never_lose_flow")


def test_cushion_first_vs_ball1_first_subframe_disambiguation() -> None:
    """
    2. 쿠션 바로 옆(y=85mm)의 1적구에 대해:
       - Case A (쿠션 먼저 -> 1적구): 초기 직선으로 맞힐 수 없는 각도로 들어가 쿠션을 먼저 스치고 1적구 타격
         -> flow_twin.first_contact_type == "CUSHION_FIRST", is_bank_shot == True, scored == True
       - Case B (1적구 먼저 -> 쿠션): 출발 직선으로 1적구를 먼저 직접 타격한 뒤 쿠션 접촉
         -> flow_twin.first_contact_type == "BALL1_FIRST", is_bank_shot == False, scored == True
    """
    dt = 1.0 / 60.0

    # --- Case A: 쿠션 먼저(TOP y=40mm) -> 1적구(RED: 650, 85) -> 2쿠션(RIGHT) -> 3쿠션(BOTTOM) -> 2적구(YELLOW) ---
    fsm_a = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE)
    t_a = 0.0
    w_a0, r_a0, y_a0 = (350.0, 260.0), (650.0, 85.0), (2050.0, 700.0)
    for _ in range(5):
        fsm_a.update(t_a, {BallColor.WHITE: w_a0, BallColor.RED: r_a0, BallColor.YELLOW: y_a0})
        t_a += dt

    # 초기 직선은 (350, 260) -> (575, 40)으로 향하여 RED(650, 85)와의 직선 수직 거리(b_perp)가 약 82mm (>= 64mm)
    case_a_frames = [
        (395.0, 215.0),
        (455.0, 155.0),
        (515.0, 95.0),
        (575.0, 40.0),   # TOP 쿠션 선접촉! (RED는 아직 y=85mm 정지)
        (615.0, 75.0),   # 쿠션 반사 후 RED(650, 85) 타격!
    ] + _interp_pts(
        [
            (615.0, 75.0),
            (2802.0, 850.0),   # 2쿠션(RIGHT)
            (2250.0, 1384.0),  # 3쿠션(BOTTOM)
            (2070.0, 745.0),   # 제2적구(YELLOW: 2050, 700) 접촉
            (2010.0, 560.0),
        ],
        steps=8,
    )[1:]

    r_a = np.array(r_a0, dtype=np.float64)
    y_a = np.array(y_a0, dtype=np.float64)
    imm_a: Optional[ShotResult] = None
    for i, wp in enumerate(case_a_frames):
        if i >= 5:
            r_a = np.array([650.0 + min(i - 4, 10) * 10.0, 85.0 + min(i - 4, 10) * 6.0])
        if i >= 29:
            y_a = np.array([2050.0 - min(i - 28, 8) * 10.0, 700.0 - min(i - 28, 8) * 6.0])
        out = fsm_a.update(
            t_a,
            {
                BallColor.WHITE: wp,
                BallColor.RED: (float(r_a[0]), float(r_a[1])),
                BallColor.YELLOW: (float(y_a[0]), float(y_a[1])),
            },
        )
        if out is not None and not out.already_dispatched:
            imm_a = out
        t_a += dt

    assert imm_a is not None and imm_a.scored is True
    assert fsm_a.flow_twin.first_contact_type == "CUSHION_FIRST", (
        f"Case A는 CUSHION_FIRST여야 합니다: {fsm_a.flow_twin.first_contact_type}"
    )
    assert imm_a.is_bank_shot is True, "Case A는 뱅크샷(is_bank_shot=True)이어야 합니다."

    # --- Case B: 1적구 먼저(RED: 650, 85 직격) -> 1쿠션(TOP) -> 2쿠션(RIGHT) -> 3쿠션(BOTTOM) -> 2적구(YELLOW) ---
    fsm_b = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE)
    t_b = 0.0
    w_b0, r_b0, y_b0 = (350.0, 320.0), (650.0, 85.0), (1950.0, 680.0)
    for _ in range(5):
        fsm_b.update(t_b, {BallColor.WHITE: w_b0, BallColor.RED: r_b0, BallColor.YELLOW: y_b0})
        t_b += dt

    # 초기 직선이 (350, 320) -> (615, 112)로 RED(650, 85) 중심을 정조준(b_perp < 5mm)하여 먼저 타격한 뒤 TOP 쿠션(950, 38) 반사!
    case_b_pts = _interp_pts(
        [
            w_b0,
            (615.0, 112.0),    # RED(650, 85) 먼저 직접 타격!
            (950.0, 38.0),     # 1쿠션(TOP)
            (2804.0, 680.0),   # 2쿠션(RIGHT)
            (2250.0, 1384.0),  # 3쿠션(BOTTOM)
            (1975.0, 720.0),   # 제2적구(YELLOW: 1950, 680) 타격
            (1880.0, 560.0),
        ],
        steps=8,
    )
    r_b = np.array(r_b0, dtype=np.float64)
    y_b = np.array(y_b0, dtype=np.float64)
    imm_b: Optional[ShotResult] = None
    for i, wp in enumerate(case_b_pts):
        if i >= 8:
            r_b = np.array([650.0 + min(i - 7, 10) * 12.0, 85.0 + min(i - 7, 10) * 4.0])
        if i >= 40:
            y_b = np.array([1950.0 - min(i - 39, 8) * 10.0, 680.0 - min(i - 39, 8) * 6.0])
        out = fsm_b.update(
            t_b,
            {
                BallColor.WHITE: wp,
                BallColor.RED: (float(r_b[0]), float(r_b[1])),
                BallColor.YELLOW: (float(y_b[0]), float(y_b[1])),
            },
        )
        if out is not None and not out.already_dispatched:
            imm_b = out
        t_b += dt

    assert imm_b is not None and imm_b.scored is True
    assert fsm_b.flow_twin.first_contact_type == "BALL1_FIRST", (
        f"Case B는 BALL1_FIRST여야 합니다: {fsm_b.flow_twin.first_contact_type}"
    )
    assert imm_b.is_bank_shot is False, "Case B는 일반샷(is_bank_shot=False)이어야 합니다."
    print("[PASS] test_cushion_first_vs_ball1_first_subframe_disambiguation")


def test_full_cue_flow_segments_and_angles_until_stop() -> None:
    """
    3. 수구 출발 -> 1적구 -> 1쿠션 -> 2쿠션 -> 3쿠션 -> 2적구 -> 정지까지
       총 6개의 직선 구간(CueTrajectorySegment)의 진행 각도(direction_deg)와
       충돌 이벤트 체인이 정확히 순서대로 생성되는지 검증.
    """
    fsm = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE)
    dt = 1.0 / 60.0
    t = 0.0

    w0 = (400.0, 600.0)
    r0 = (900.0, 420.0)
    y0 = (1600.0, 650.0)
    for _ in range(5):
        fsm.update(t, {BallColor.WHITE: w0, BallColor.RED: r0, BallColor.YELLOW: y0})
        t += dt

    waypoints = [
        w0,
        (865.0, 432.0),    # B1
        (1450.0, 38.0),    # C1 (TOP)
        (2804.0, 650.0),   # C2 (RIGHT)
        (2150.0, 1384.0),  # C3 (BOTTOM)
        (1625.0, 675.0),   # B2
        (1480.0, 520.0),   # STOP
    ]
    cue_pts = _interp_pts(waypoints, steps=8)
    r_pos = np.array(r0, dtype=np.float64)
    y_pos = np.array(y0, dtype=np.float64)

    for idx, wp in enumerate(cue_pts):
        if idx >= 8:
            r_pos = np.array([900.0 + min(idx - 7, 12) * 8.0, 420.0 + min(idx - 7, 12) * 5.0])
        if idx >= 40:
            y_pos = np.array([1600.0 - min(idx - 39, 10) * 8.0, 650.0 - min(idx - 39, 10) * 5.0])
        fsm.update(
            t,
            {
                BallColor.WHITE: wp,
                BallColor.RED: (float(r_pos[0]), float(r_pos[1])),
                BallColor.YELLOW: (float(y_pos[0]), float(y_pos[1])),
            },
        )
        t += dt

    for _ in range(25):
        fsm.update(
            t,
            {
                BallColor.WHITE: cue_pts[-1],
                BallColor.RED: (float(r_pos[0]), float(r_pos[1])),
                BallColor.YELLOW: (float(y_pos[0]), float(y_pos[1])),
            },
        )
        t += dt

    expected_event_chain = [
        FlowEventType.STROKE_START,
        FlowEventType.FIRST_BALL_HIT,
        FlowEventType.CUSHION_HIT,
        FlowEventType.CUSHION_HIT,
        FlowEventType.CUSHION_HIT,
        FlowEventType.SECOND_BALL_HIT,
        FlowEventType.STROKE_STOP,
    ]
    actual_event_chain = [ev.event_type for ev in fsm.flow_twin.events]
    assert actual_event_chain == expected_event_chain, (
        f"이벤트 체인 불일치: expected={expected_event_chain}, got={actual_event_chain}"
    )

    assert len(fsm.flow_twin.segments) == 6, (
        f"6개의 직선 구간(Segment)이 생성되어야 합니다: {len(fsm.flow_twin.segments)}"
    )
    for idx_s, seg in enumerate(fsm.flow_twin.segments):
        assert seg.seg_index == idx_s + 1
        assert seg.length_mm > 50.0
        assert -180.0 <= seg.direction_deg <= 180.0
    print("[PASS] test_full_cue_flow_segments_and_angles_until_stop")


if __name__ == "__main__":
    test_detect_white_or_yellow_cue_ball_and_never_lose_flow()
    test_cushion_first_vs_ball1_first_subframe_disambiguation()
    test_full_cue_flow_segments_and_angles_until_stop()
    print("=== ALL CUE FLOW PHYSICS TESTS PASSED! ===")
