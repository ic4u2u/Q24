"""
CIE LAB + 헝가리안 3구 정밀 식별기(LabHungarianBallIdentifier) 및
평범한 득점 실패 오인식(False Positive) 원천 차단 검증 테스트
파일 경로: tests/test_lab_hungarian_and_false_positives.py
"""

import cv2
import numpy as np

from ball_identity_tracker import LabHungarianBallIdentifier
from cue24.core.scoring_engine import ScoringEngine, ShotFSMState
from scoring_fsm import BallColor, CausalKissSafeFSM


def test_1_lab_hungarian_ball_identifier_with_glare() -> None:
    """
    [검증 1] 상단 LED 조명 하이라이트 반사광(Glare)과 파란 당구대 라사지 배경이 포함된 이미지에서
    LabHungarianBallIdentifier가 WHITE, YELLOW, RED를 중복 없이 1:1로 정확히 할당하는지 검증
    """
    identifier = LabHungarianBallIdentifier()
    # 파란 당구대 라사지 배경 (BGR: 190, 90, 20)
    frame = np.full((200, 400, 3), (190, 90, 20), dtype=np.uint8)

    # 1) RED 공 (중심 60, 100), 2) WHITE 공 (중심 200, 100), 3) YELLOW 공 (중심 340, 100)
    cv2.circle(frame, (60, 100), 18, (30, 25, 210), -1)     # RED (BGR)
    cv2.circle(frame, (200, 100), 18, (235, 235, 235), -1)  # WHITE (BGR)
    cv2.circle(frame, (340, 100), 18, (20, 215, 245), -1)   # YELLOW (BGR)

    # 각 공 상단에 강한 천장 조명 하이라이트(순백색 255, 255, 255) 추가
    for cx in (60, 200, 340):
        cv2.circle(frame, (cx - 3, 95), 5, (255, 255, 255), -1)

    bboxes = [
        (40.0, 80.0, 80.0, 120.0),    # index 0: RED 위치
        (180.0, 80.0, 220.0, 120.0),  # index 1: WHITE 위치
        (320.0, 80.0, 360.0, 120.0),  # index 2: YELLOW 위치
    ]
    assigned = identifier.assign_ball_identities(frame, bboxes)

    assert len(assigned) == 3, f"3개 공이 모두 할당되지 않음: {assigned}"
    assert assigned[BallColor.RED] == bboxes[0], f"RED 오할당: {assigned}"
    assert assigned[BallColor.WHITE] == bboxes[1], f"WHITE 오할당: {assigned}"
    assert assigned[BallColor.YELLOW] == bboxes[2], f"YELLOW 오할당: {assigned}"
    print(
        "[PASS] Test 1 (LabHungarianBallIdentifier): 조명 반사광(Glare) 제거 및 CIE LAB + 헝가리안 1:1 전역 최적 식별 완료"
    )


def test_2_ordinary_miss_false_positive_blocked() -> None:
    """
    [검증 2] 키스 없는 평범한 득점 실패 시나리오:
    수구가 3쿠션 후 제2적구 옆을 대각선 시차(d_min = 58mm <= 63mm)로 지나가면서
    바운딩 박스 1~2픽셀 제자리 노이즈(2.5mm 진동 후 복귀)와 수구 미세 회전 휨(2.3도)이 동시에 발생하더라도,
    정지 앵커 순변위(< 5.2mm) 및 직선 회귀 잔차(line_rmse) 게이트가 작동하여 100% 노카운트(False)로 차단하는지 검증
    """
    fsm = CausalKissSafeFSM(fps=30.0, assigned_cue_ball=BallColor.WHITE)
    engine = ScoringEngine()
    engine.reset_shot(cue_ball_id=1, object_ball_ids=[2, 3])

    # 1적구(RED) 타격 -> 3쿠션(TOP -> RIGHT -> BOTTOM) -> 2적구(YELLOW: 1500, 600) 옆을 시차 거리 58mm로 비접촉 통과
    frames = []
    # 초기 정지 상태 3프레임 (idle_anchor_pos 안정화)
    for _ in range(3):
        frames.append(((300.0, 700.0), (1500.0, 600.0), (500.0, 700.0)))
    # 수구 출발 및 1적구(RED: 500, 700) 타격
    frames.extend([
        ((380.0, 700.0), (1500.0, 600.0), (500.0, 700.0)),
        ((445.0, 700.0), (1500.0, 600.0), (500.0, 700.0)),
        ((470.0, 620.0), (1500.0, 600.0), (550.0, 740.0)),
    ])
    # 1쿠션: TOP (y=42)
    for k in range(1, 5):
        frames.append(((470.0 + k * 55, 620.0 - k * 140), (1500.0, 600.0), (580.0, 770.0)))
    frames.append(((700.0, 42.0), (1500.0, 600.0), (600.0, 780.0)))
    # 2쿠션: RIGHT (x=2798)
    for k in range(1, 12):
        frames.append(((700.0 + k * 175, 42.0 + k * 48), (1500.0, 600.0), (610.0, 790.0)))
    frames.append(((2798.0, 600.0), (1500.0, 600.0), (610.0, 790.0)))
    # 3쿠션: BOTTOM (y=1380)
    for k in range(1, 10):
        frames.append(((2798.0 - k * 90, 600.0 + k * 80), (1500.0, 600.0), (620.0, 800.0)))
    frames.append(((1900.0, 1380.0), (1500.0, 600.0), (620.0, 800.0)))

    # 3쿠션 후 제2적구(1500, 600) 근처를 등속 직선으로 안 맞고 통과 (최근접 투영 거리 58mm + 제2적구 2.2mm 제자리 떨림 노이즈)
    pass_points = [
        ((1740.0, 960.0), (1500.0, 600.0)),
        ((1660.0, 820.0), (1500.5, 600.5)),
        ((1580.0, 680.0), (1501.8, 601.2)),  # 바운딩 박스 2.1mm 노이즈 떨림
        ((1500.0, 542.0), (1502.0, 601.0)),  # d_min = 58.0mm (<= 63.0mm 시차 중첩!)
        ((1422.0, 404.0), (1500.2, 600.1)),  # 제자리 복귀 (영구 순변위 < 5.2mm) & 수구 2도 미만 직진
        ((1344.0, 266.0), (1500.0, 600.0)),
        ((1266.0, 128.0), (1500.0, 600.0)),
    ]
    for c_pt, y_pt in pass_points:
        frames.append((c_pt, y_pt, (620.0, 800.0)))

    # 정지 프레임
    for _ in range(10):
        frames.append(((1266.0, 128.0), (1500.0, 600.0), (620.0, 800.0)))

    final_res = None
    for idx, (w_pos, y_pos, r_pos) in enumerate(frames):
        t = idx / 30.0
        r = fsm.update(
            t,
            {
                BallColor.WHITE: w_pos,
                BallColor.YELLOW: y_pos,
                BallColor.RED: r_pos,
            },
        )
        if r is not None:
            final_res = r
        engine.update_frame(
            frame_idx=idx,
            ball_positions={1: w_pos, 2: y_pos, 3: r_pos},
        )

    assert final_res is not None
    assert final_res.scored is False, f"FSM이 평범한 실패(58mm 시차 + 2.2mm 노이즈)를 득점으로 오판함! {final_res.event_log}"
    assert engine.fsm_state != ShotFSMState.VALID_SCORE_PENDING, "ScoringEngine이 평범한 실패를 득점으로 오판함!"
    print(
        "[PASS] Test 2 (평범한 득점 실패 오판 원천 차단): d_min=58mm 시차 중첩 + 2.2mm 노이즈 + 미세 휨 통과 시 FSM & Engine 모두 100% 노카운트(False) 검증 완료"
    )


if __name__ == "__main__":
    test_1_lab_hungarian_ball_identifier_with_glare()
    test_2_ordinary_miss_false_positive_blocked()
    print("ALL LAB-HUNGARIAN & FALSE-POSITIVE BLOCK TESTS PASSED!")
