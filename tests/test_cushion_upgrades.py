"""
4대 쿠션 알고리즘 고도화(11쿠션 코너 폭주 차단 / 0쿠션 누락 복원 / 사후 쿠션 동결) 검증 테스트
파일 경로: tests/test_cushion_upgrades.py
"""

from cue24.core.scoring_engine import ScoringEngine, ShotFSMState
from scoring_fsm import BallColor, CausalKissSafeFSM, ShotState


def test_1_corner_ping_pong_and_hysteresis_lock() -> None:
    """
    [검증 1] 코너 근처에서 바운딩 박스가 미세하게 떨릴 때 10~11쿠션으로 폭주하지 않고
    슈미트 트리거 히스테리시스(92mm 이탈 / 80mm 이동) 및 코너 최소 변위(22mm) 잠금으로 차단되는지 검증
    """
    fsm = CausalKissSafeFSM(fps=30.0, assigned_cue_ball=BallColor.WHITE)
    engine = ScoringEngine()
    engine.reset_shot(cue_ball_id=1, object_ball_ids=[2, 3])

    pts = [
        (0, (300.0, 300.0)),
        (1, (220.0, 220.0)),
        (2, (140.0, 140.0)),
        (3, (70.0, 70.0)),
        (4, (48.0, 52.0)),
    ]
    # 코너 구석에서 25프레임 동안 x, y가 46~53mm 사이에서 지그재그로 진동 (기존에는 10~11쿠션 폭주 발생)
    for f in range(5, 30):
        if f % 2 == 0:
            pts.append((f, (52.0, 47.0)))
        else:
            pts.append((f, (47.0, 52.0)))

    for f_idx, (cx, cy) in pts:
        t = f_idx / 30.0
        fsm.update(
            t,
            {
                BallColor.WHITE: (cx, cy),
                BallColor.YELLOW: (1400.0, 700.0),
                BallColor.RED: (2000.0, 700.0),
            },
        )
        engine.update_frame(
            frame_idx=f_idx,
            ball_positions={
                1: (cx, cy),
                2: (1400.0, 700.0),
                3: (2000.0, 700.0),
            },
        )

    assert fsm.cushion_count <= 2, f"FSM 코너 핑퐁 폭주 발생: {fsm.cushion_count}쿠션"
    assert engine.total_cushion_count <= 2, f"ScoringEngine 코너 핑퐁 폭주 발생: {engine.total_cushion_count}쿠션"
    print(
        f"[PASS] Test 1 (슈미트 트리거 & 코너 핑퐁 차단): 25프레임 코너 진동 시 FSM={fsm.cushion_count}쿠션, Engine={engine.total_cushion_count}쿠션 (11쿠션 폭주 완벽 차단)"
    )


def test_2_wide_zone_kinematic_bounce_apex() -> None:
    """
    [검증 2] 호모그래피 시차나 고속 프레임 스킵으로 수구가 54mm 안으로 들어가지 못하고
    85mm~95mm 광역 대역(RAIL_KINEMATIC_ZONE=115mm 이내)에서 법선 속도가 반전될 때 유효 쿠션으로 감지하는지 검증
    """
    fsm = CausalKissSafeFSM(fps=30.0, assigned_cue_ball=BallColor.WHITE)

    cue_path = [
        (500.0, 260.0),
        (540.0, 215.0),
        (580.0, 170.0),
        (620.0, 125.0),
        (660.0, 88.0),   # 54mm 안으로 못 들어갔으나 115mm 광역 대역 내 변곡점
        (700.0, 130.0),  # 법선 속도 vy 부호 반전 (- -> +)
        (740.0, 175.0),
        (780.0, 220.0),
        (820.0, 265.0),
        (860.0, 310.0),
    ]
    for i, (cx, cy) in enumerate(cue_path):
        t = i / 30.0
        fsm.update(
            t,
            {
                BallColor.WHITE: (cx, cy),
                BallColor.YELLOW: (1800.0, 700.0),
                BallColor.RED: (2200.0, 900.0),
            },
        )

    assert fsm.cushion_count == 1, f"광역 궤적 반사(y=88mm) 쿠션 누락: {fsm.cushion_count}쿠션"
    print(
        f"[PASS] Test 2 (광역 궤적 변곡점 RAIL_KINEMATIC_ZONE=115mm 감지): y=88.0mm 반사 시 누락 없이 {fsm.cushion_count}쿠션 감지 완료"
    )


def test_3_cushion_freeze_after_valid_score() -> None:
    """
    [검증 3] 3쿠션 달성 후 제2적구를 맞혀 즉시 득점(VALID_SCORE_PENDING)이 확정된 이후,
    공이 멈추기 전까지 추가로 2개 쿠션을 더 치더라도 최종 쿠션 수가 5쿠션으로 늘어나지 않고 3쿠션으로 동결되는지 검증
    """
    engine = ScoringEngine()
    engine.reset_shot(cue_ball_id=1, object_ball_ids=[2, 3])

    frames = []
    # 0~3: 제1적구(500, 700)로 이동 및 충돌
    frames.extend([
        ((300.0, 700.0), (500.0, 700.0), (1500.0, 600.0)),
        ((380.0, 700.0), (500.0, 700.0), (1500.0, 600.0)),
        ((445.0, 700.0), (500.0, 700.0), (1500.0, 600.0)),
        ((470.0, 620.0), (540.0, 730.0), (1500.0, 600.0)),
    ])
    # 4~15: 1쿠션 TOP (y=42)
    for k in range(5):
        frames.append(((500.0 + k * 40, 500.0 - k * 90), (560.0, 750.0), (1500.0, 600.0)))
    frames.append(((700.0, 42.0), (580.0, 770.0), (1500.0, 600.0)))
    # 16~28: 2쿠션 RIGHT (x=2798) - 물리적 프레임당 이동량(<=180mm/frame, 12프레임 이상) 준수
    for k in range(1, 13):
        frames.append(((700.0 + k * 170, 42.0 + k * 45), (600.0, 780.0), (1500.0, 600.0)))
    frames.append(((2798.0, 600.0), (610.0, 790.0), (1500.0, 600.0)))
    # 29~40: 3쿠션 BOTTOM (y=1380)
    for k in range(1, 11):
        frames.append(((2798.0 - k * 85, 600.0 + k * 75), (620.0, 800.0), (1500.0, 600.0)))
    frames.append(((1900.0, 1380.0), (620.0, 800.0), (1500.0, 600.0)))
    for k in range(1, 8):
        frames.append(((1900.0 - k * 50, 1380.0 - k * 100), (620.0, 800.0), (1500.0, 600.0)))
    # 제2적구(1500, 600) 충돌! -> 즉시 득점 확정 및 3쿠션 동결 시점
    frames.append(((1540.0, 635.0), (620.0, 800.0), (1500.0, 600.0)))
    frames.append(((1400.0, 500.0), (620.0, 800.0), (1460.0, 560.0)))
    # 득점 확정 이후 굴러가며 맞은 사후 쿠션 2회 (TOP y=40, LEFT x=40)
    for k in range(5):
        frames.append(((1300.0 - k * 80, 400.0 - k * 70), (620.0, 800.0), (1440.0, 540.0)))
    frames.append(((900.0, 40.0), (620.0, 800.0), (1430.0, 530.0)))
    for k in range(1, 6):
        frames.append(((900.0 - k * 150, 40.0 + k * 70), (620.0, 800.0), (1430.0, 530.0)))
    frames.append(((40.0, 420.0), (620.0, 800.0), (1430.0, 530.0)))
    frames.append(((200.0, 500.0), (620.0, 800.0), (1430.0, 530.0)))

    for idx, (c_pos, b1_pos, b2_pos) in enumerate(frames):
        engine.update_frame(
            frame_idx=idx,
            ball_positions={1: c_pos, 2: b1_pos, 3: b2_pos},
        )

    assert engine.fsm_state == ShotFSMState.VALID_SCORE_PENDING
    assert engine.total_cushion_count == 3, f"Engine 사후 쿠션 미동결: {engine.total_cushion_count}쿠션 (기대값: 3)"
    eval_res = engine.evaluate_shot()
    assert "총 3쿠션" in str(eval_res["message"]), f"메시지 오류: {eval_res['message']}"
    print(
        f"[PASS] Test 3 (제2적구 충돌 즉시 쿠션 동결): 사후 2쿠션 추가 발생에도 Engine={engine.total_cushion_count}쿠션 동결 완료 -> '{eval_res['message']}'"
    )


if __name__ == "__main__":
    test_1_corner_ping_pong_and_hysteresis_lock()
    test_2_wide_zone_kinematic_bounce_apex()
    test_3_cushion_freeze_after_valid_score()
    print("ALL CUSHION UPGRADE TESTS PASSED!")
