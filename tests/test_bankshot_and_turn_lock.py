"""
[통합 패치 검증 테스트]
1) 뱅크샷 카운트 숫자 및 누적 쿠션 수 정밀 교정 (선쿠션 분리, 프레임 내 판정 순서 역전, 출발 직후 1쿠션 보존, 동일 레일 연속 중복 차단)
2) 가짜 턴 넘김(Turn Hijack) 및 상대방 가짜 득점(Ghost Score) 원천 차단
파일 경로: tests/test_bankshot_and_turn_lock.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "three_cushion_vision")
    ),
)

from three_cushion_vision.causal_kiss_fsm import BallColor, CausalKissSafeFSM, ShotResult


def test_1_cushion_and_2_cushion_bankshot_exact_count() -> None:
    """
    [테스트 1] 1쿠션 걸어치기 뱅크샷(1쿠션 -> 제1적구 -> 2쿠션 -> 제2적구)에서:
    - 출발 직후 0.05초 만에 맞은 첫 쿠션이 버퍼 초기화로 증발하지 않고
      cushions_before_first_hit == 1, total_cushions == 3, is_bank_shot == True로 정확히 기록되며,
    - 즉시 득점 콜백과 정지 완료 시점 통틀어 점수와 뱅크샷 카운트가 딱 1번만 올라가는지 검증합니다.
    """
    score_board = {"WHITE_score": 0, "WHITE_bank": 0, "YELLOW_score": 0, "last_shot_id": -1}

    def on_score(res: ShotResult) -> None:
        if res.scored and not res.already_dispatched and res.shot_id != score_board["last_shot_id"]:
            score_board["last_shot_id"] = res.shot_id
            score_board["WHITE_score"] += 1
            if res.is_bank_shot:
                score_board["WHITE_bank"] += 1

    fsm = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE, on_immediate_score=on_score)

    # t=0.00s ~ 0.03s: 정지 대기 후 수구(WHITE) 출발 (TOP 레일 근처 y=90mm에서 시작해 즉시 y=38mm 1쿠션 타격!)
    fsm.update(0.00, {BallColor.WHITE: (400.0, 90.0), BallColor.YELLOW: (700.0, 300.0), BallColor.RED: (2100.0, 600.0)})
    fsm.update(0.016, {BallColor.WHITE: (440.0, 62.0), BallColor.YELLOW: (700.0, 300.0), BallColor.RED: (2100.0, 600.0)})
    fsm.update(0.033, {BallColor.WHITE: (480.0, 38.0), BallColor.YELLOW: (700.0, 300.0), BallColor.RED: (2100.0, 600.0)})  # 1쿠션(TOP, 빈쿠션!)
    fsm.update(0.050, {BallColor.WHITE: (520.0, 85.0), BallColor.YELLOW: (700.0, 300.0), BallColor.RED: (2100.0, 600.0)})
    fsm.update(0.066, {BallColor.WHITE: (580.0, 160.0), BallColor.YELLOW: (700.0, 300.0), BallColor.RED: (2100.0, 600.0)})

    # t=0.08s ~ 0.15s: 제1적구(YELLOW) 충돌
    for f in range(5, 11):
        t = f / 60.0
        wx = 580.0 + (f - 4) * 24.0
        wy = 160.0 + (f - 4) * 28.0
        yx = 700.0 if f < 8 else 700.0 + (f - 7) * 30.0
        yy = 300.0 if f < 8 else 300.0 + (f - 7) * 15.0
        fsm.update(t, {BallColor.WHITE: (wx, wy), BallColor.YELLOW: (yx, yy), BallColor.RED: (2100.0, 600.0)})

    # 제1적구 충돌 후 2쿠션(BOTTOM y=1384), 3쿠션(RIGHT x=2802) 이동 (프레임당 이동량 <= 250mm 준수)
    pts = [
        (820.0, 520.0), (950.0, 740.0), (1080.0, 960.0), (1220.0, 1180.0), (1380.0, 1384.0),  # 2쿠션(BOTTOM)
        (1580.0, 1220.0), (1800.0, 1060.0), (2020.0, 920.0), (2250.0, 800.0), (2480.0, 700.0),
        (2680.0, 640.0), (2802.0, 620.0),                                                      # 3쿠션(RIGHT)
        (2620.0, 612.0), (2440.0, 606.0), (2280.0, 602.0), (2152.0, 600.0),                    # 제2적구(RED: 2100, 600) 충돌!
        (2080.0, 560.0), (2010.0, 510.0), (1950.0, 450.0),
    ]
    f_cur = 11
    final_res = None
    for idx, (wx, wy) in enumerate(pts):
        f_cur += 1
        t = f_cur / 60.0
        # idx == len(pts) - 4 (wx=2152, wy=600)에서 접촉 후 그 다음 프레임부터 RED가 밀려남
        if idx <= len(pts) - 4:
            rx, ry = 2100.0, 600.0
        else:
            step_after = idx - (len(pts) - 4)
            rx = 2100.0 - step_after * 35.0
            ry = 600.0 - step_after * 15.0
        r = fsm.update(t, {BallColor.WHITE: (wx, wy), BallColor.YELLOW: (820.0, 360.0), BallColor.RED: (rx, ry)})
        if r is not None:
            on_score(r)

    # 제2적구 충돌 이후 잔여 구름 중 쿠션(TOP y=38)을 하나 더 맞고 정지하더라도 3쿠션으로 동결되어야 함!
    post_roll = [
        (1920.0, 260.0), (1900.0, 120.0), (1890.0, 38.0), (1880.0, 110.0), (1870.0, 150.0)
    ]
    for wx, wy in post_roll:
        f_cur += 1
        t = f_cur / 60.0
        r = fsm.update(t, {BallColor.WHITE: (wx, wy), BallColor.YELLOW: (820.0, 360.0), BallColor.RED: (1995.0, 555.0)})
        if r is not None:
            on_score(r)

    for f_stop in range(40):
        f_cur += 1
        t = f_cur / 60.0
        r = fsm.update(t, {BallColor.WHITE: (1870.0, 150.0), BallColor.YELLOW: (820.0, 360.0), BallColor.RED: (1995.0, 555.0)})
        if r is not None:
            final_res = r
            on_score(r)

    assert final_res is not None
    assert final_res.scored is True
    assert final_res.is_bank_shot is True, "1쿠션 걸어치기가 뱅크샷으로 인식되지 않았습니다."
    assert final_res.cushions_before_first_hit == 1, f"선쿠션 수 오류: {final_res.cushions_before_first_hit}"
    assert final_res.total_cushions == 3, f"총 쿠션 수 오류(사후 쿠션 포함됨): {final_res.total_cushions}"
    assert score_board["WHITE_score"] == 1, f"점수 중복 카운트 오류: {score_board['WHITE_score']}"
    assert score_board["WHITE_bank"] == 1, f"뱅크샷 중복/누락 카운트 오류: {score_board['WHITE_bank']}"
    assert fsm.assigned_cue_ball == BallColor.WHITE, "득점했는데 공격권(턴)이 상대방으로 넘어갔습니다!"
    print("[PASS] Test 1 (1쿠션/2쿠션/3쿠션 뱅크샷 정밀 카운트 & 중복 가산 차단 & 턴 유지) 통과!")


def test_order_reversal_prevents_false_bankshot() -> None:
    """
    [테스트 2] 제1적구가 TOP 레일 근처(y=45mm)에 붙어 있을 때,
    수구가 제1적구를 먼저 맞히는 일반 3쿠션 샷이 '프레임 내 판정 순서 역전(목적구 충돌 -> 쿠션 검사)' 덕분에
    가짜 뱅크샷(cushions_before_first_hit=1)으로 둔갑하지 않고 일반 득점(is_bank_shot=False)으로 판정되는지 검증합니다.
    """
    fsm = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE)

    fsm.update(0.00, {BallColor.WHITE: (300.0, 150.0), BallColor.YELLOW: (500.0, 45.0), BallColor.RED: (2200.0, 700.0)})
    fsm.update(0.016, {BallColor.WHITE: (350.0, 120.0), BallColor.YELLOW: (500.0, 45.0), BallColor.RED: (2200.0, 700.0)})

    # t=0.05s ~ 0.15s: 수구(WHITE)가 제1적구(YELLOW)를 향해 이동하여 y=45mm(TOP 레일 54mm 마진 내부!)에서 충돌
    for f in range(3, 12):
        t = f / 60.0
        wx = 350.0 + (f - 2) * 18.0
        wy = 120.0 - (f - 2) * 8.3  # f=11일 때 wy ≈ 45.3mm (TOP 레일 마진 내부!)
        yx = 500.0 if f < 10 else 500.0 + (f - 9) * 28.0
        yy = 45.0 if f < 10 else 45.0 + (f - 9) * 10.0
        fsm.update(t, {BallColor.WHITE: (wx, wy), BallColor.YELLOW: (yx, yy), BallColor.RED: (2200.0, 700.0)})

    assert fsm.first_target_ball == BallColor.YELLOW
    assert fsm.cushions_before_first_hit == 0, f"일반 샷인데 선쿠션이 카운트됨: {fsm.cushions_before_first_hit}"
    assert fsm.is_bank_shot is False, "일반 샷이 가짜 뱅크샷으로 오인식됨!"

    # 이후 정상 3쿠션(TOP -> BOTTOM -> RIGHT) 후 제2적구(RED) 타격 (프레임당 이동량 <= 250mm 준수)
    post_pts = [
        (650.0, 120.0), (720.0, 70.0), (800.0, 38.0), (900.0, 120.0),          # 1쿠션(TOP)
        (1020.0, 350.0), (1140.0, 600.0), (1250.0, 860.0), (1330.0, 1120.0),
        (1400.0, 1384.0), (1560.0, 1180.0),                                    # 2쿠션(BOTTOM)
        (1750.0, 1000.0), (1960.0, 870.0), (2180.0, 790.0), (2400.0, 740.0),
        (2620.0, 715.0), (2802.0, 700.0), (2620.0, 700.0), (2430.0, 700.0),    # 3쿠션(RIGHT)
        (2252.0, 700.0),                                                       # 제2적구(RED: 2200, 700) 접촉!
        (2180.0, 675.0), (2110.0, 645.0), (2050.0, 615.0),
    ]
    f_cur = 12
    imm_res = None
    for idx, (wx, wy) in enumerate(post_pts):
        f_cur += 1
        t = f_cur / 60.0
        if idx <= len(post_pts) - 4:
            rx, ry = 2200.0, 700.0
        else:
            step_after = idx - (len(post_pts) - 4)
            rx = 2200.0 - step_after * 35.0
            ry = 700.0 + step_after * 12.0
        r = fsm.update(t, {BallColor.WHITE: (wx, wy), BallColor.YELLOW: (600.0, 90.0), BallColor.RED: (rx, ry)})
        if r is not None:
            imm_res = r

    assert imm_res is not None and imm_res.scored is True
    assert imm_res.is_bank_shot is False, "일반 3쿠션 득점이 뱅크샷으로 잘못 분류되었습니다!"
    assert imm_res.total_cushions == 3, f"일반 3쿠션 쿠션 수 오류: {imm_res.total_cushions}"
    print("[PASS] Test 2 (판정 순서 역전으로 레일 근처 제1적구 타격 시 가짜 뱅크샷 오인식 완벽 차단) 통과!")


def test_same_rail_sliding_no_extra_cushions() -> None:
    """
    [테스트 3] 수구가 TOP 쿠션을 맞고 얕은 각도로 레일을 따라 굴러가며 y=42mm -> 98mm -> 48mm로 흔들릴 때,
    다른 레일이나 목적구를 맞히지 않았으므로 동일 레일 중복 잠금(same_rail_locked)이 작동하여
    5쿠션/11쿠션으로 폭주하지 않고 정확히 1쿠션만 카운트되는지 검증합니다.
    """
    fsm = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE)
    fsm.update(0.00, {BallColor.WHITE: (300.0, 250.0), BallColor.YELLOW: (2000.0, 800.0), BallColor.RED: (2200.0, 800.0)})

    # TOP 레일을 따라 길게 미끄러지며 y가 40mm ~ 105mm 사이를 여러 번 왕복 (심층 이탈 220mm 미만)
    y_wave = [200.0, 140.0, 80.0, 42.0, 75.0, 102.0, 60.0, 45.0, 98.0, 48.0, 105.0, 44.0, 95.0, 49.0]
    for idx, wy in enumerate(y_wave, start=1):
        t = idx * 0.05
        wx = 300.0 + idx * 95.0
        fsm.update(t, {BallColor.WHITE: (wx, wy), BallColor.YELLOW: (2000.0, 800.0), BallColor.RED: (2200.0, 800.0)})

    assert fsm.cushion_count == 1, f"동일 레일 슬라이딩 중복 카운트 발생: {fsm.cushion_count}회 (기대값: 1회)"
    print(f"[PASS] Test 3 (동일 레일 슬라이딩 98mm~45mm 진동 시 중복 카운트 차단 -> 정확히 {fsm.cushion_count}쿠션 유지) 통과!")


def test_opponent_ball_noise_never_hijacks_turn_or_scores() -> None:
    """
    [테스트 4] WHITE(선공) 턴 대기(IDLE) 중에:
    1) 상대방 공(YELLOW)이나 적구(RED)가 바운딩 박스 노이즈로 순간 300mm/s로 흔들려도
       WRONG_CUE_BALL_FOUL로 턴이 넘어가거나 상대방(YELLOW) 수구로 바뀌지 않아야 하며,
    2) WHITE가 살짝(50mm) 흔들렸다가 멈춰도(< 120mm) 공타로 턴이 넘어가지 않고 WHITE 턴이 그대로 유지되는지 검증합니다.
    """
    fsm = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE)

    # 1) YELLOW 공이 바운딩 박스 노이즈로 12mm 순간 점프 (속도 > 200mm/s 이지만 지정 수구 WHITE는 정지)
    for f in range(20):
        t = f / 60.0
        yx = 1000.0 + (12.0 if f == 5 else 0.0)
        res = fsm.update(t, {BallColor.WHITE: (500.0, 500.0), BallColor.YELLOW: (yx, 500.0), BallColor.RED: (1500.0, 500.0)})
        assert res is None
        assert fsm.state.name == "IDLE", "상대방 공 노이즈에 의해 가짜 샷이 시작되었습니다!"
        assert fsm.assigned_cue_ball == BallColor.WHITE, "상대방 공 노이즈에 의해 턴이 강탈되었습니다!"

    # 2) WHITE 공이 큐대 그림자 때문에 45mm 움직였다가(< 120mm) 아무 공도 안 맞고 정지
    for f in range(20, 65):
        t = f / 60.0
        wx = 500.0 + min(45.0, (f - 20) * 15.0)
        res = fsm.update(t, {BallColor.WHITE: (wx, 500.0), BallColor.YELLOW: (1000.0, 500.0), BallColor.RED: (1500.0, 500.0)})
        assert res is None, "120mm 미만 노이즈 이동이 정식 샷 종료(공타 턴 넘김)로 처리되었습니다!"

    assert fsm.assigned_cue_ball == BallColor.WHITE, "노이즈 정지 후 공격권(턴)이 상대방에게 넘어갔습니다!"
    print("[PASS] Test 4 (상대방 공 노이즈 & 120mm 미만 가짜 출발 시 턴 넘김 및 상대방 가짜 득점 원천 차단) 통과!")


def test_residual_rolling_after_immediate_score_never_flips_turn() -> None:
    """
    [테스트 5] 3쿠션 후 제2적구 충돌 즉시 득점(+1)이 발생한 뒤,
    공들이 멈출 때까지 굴러가는 잔여 구름(Residual Rolling) 동안 제1적구와 제2적구가 키스하고
    수구가 추가 쿠션을 여러 번 치더라도:
    - 쿠션 수가 3쿠션에서 늘어나지 않고,
    - 공이 완전히 정지했을 때 공격권(assigned_cue_ball)이 절대 상대방(YELLOW)으로 넘어가지 않는지 검증합니다.
    """
    fsm = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE)

    # 일반 3쿠션 득점 궤적 (프레임당 이동량 <= 250mm 준수)
    frames = [
        (0.00, (300.0, 700.0), (500.0, 700.0), (2200.0, 600.0)),
        (0.02, (380.0, 700.0), (500.0, 700.0), (2200.0, 600.0)),
        (0.04, (445.0, 700.0), (500.0, 700.0), (2200.0, 600.0)),
        (0.06, (470.0, 620.0), (550.0, 740.0), (2200.0, 600.0)),  # 1적구(YELLOW) 충돌
        (0.10, (540.0, 440.0), (580.0, 760.0), (2200.0, 600.0)),
        (0.15, (620.0, 260.0), (610.0, 780.0), (2200.0, 600.0)),
        (0.20, (690.0, 120.0), (630.0, 800.0), (2200.0, 600.0)),
        (0.25, (750.0, 38.0),  (650.0, 820.0), (2200.0, 600.0)),  # 1쿠션(TOP)
        (0.30, (860.0, 220.0), (670.0, 830.0), (2200.0, 600.0)),
        (0.35, (1000.0, 420.0), (685.0, 840.0), (2200.0, 600.0)),
        (0.40, (1150.0, 620.0), (700.0, 850.0), (2200.0, 600.0)),
        (0.45, (1300.0, 820.0), (710.0, 860.0), (2200.0, 600.0)),
        (0.50, (1460.0, 1020.0), (725.0, 870.0), (2200.0, 600.0)),
        (0.55, (1650.0, 1210.0), (740.0, 875.0), (2200.0, 600.0)),
        (0.60, (1900.0, 1384.0), (750.0, 880.0), (2200.0, 600.0)), # 2쿠션(BOTTOM)
        (0.65, (2060.0, 1220.0), (765.0, 890.0), (2200.0, 600.0)),
        (0.70, (2240.0, 1060.0), (775.0, 895.0), (2200.0, 600.0)),
        (0.75, (2420.0, 910.0), (785.0, 900.0), (2200.0, 600.0)),
        (0.80, (2610.0, 770.0), (795.0, 905.0), (2200.0, 600.0)),
        (0.85, (2802.0, 650.0), (800.0, 910.0), (2200.0, 600.0)),  # 3쿠션(RIGHT)
        (0.90, (2610.0, 632.0), (800.0, 910.0), (2200.0, 600.0)),
        (0.95, (2420.0, 616.0), (800.0, 910.0), (2200.0, 600.0)),
        (1.00, (2250.0, 600.0), (800.0, 910.0), (2200.0, 600.0)),  # 2적구(RED: 2200, 600) 충돌! -> 즉시 득점!
        (1.04, (2180.0, 565.0), (800.0, 910.0), (2155.0, 585.0)),
        (1.08, (2110.0, 530.0), (800.0, 910.0), (2115.0, 570.0)),
        (1.12, (2040.0, 500.0), (800.0, 910.0), (2080.0, 550.0)),
    ]

    immediate_res = None
    for t, w, y, r in frames:
        out = fsm.update(t, {BallColor.WHITE: w, BallColor.YELLOW: y, BallColor.RED: r})
        if out is not None and not out.already_dispatched:
            immediate_res = out

    assert immediate_res is not None and immediate_res.scored is True
    assert immediate_res.is_turn_end is False

    # 잔여 구름 후 40프레임 완전 정지
    final_res = None
    for s in range(1, 45):
        t = 1.32 + s / 60.0
        out = fsm.update(t, {BallColor.WHITE: (2000.0, 500.0), BallColor.YELLOW: (800.0, 910.0), BallColor.RED: (2050.0, 530.0)})
        if out is not None:
            final_res = out

    assert final_res is not None
    assert final_res.scored is True
    assert final_res.already_dispatched is True
    assert final_res.is_turn_end is False
    assert fsm.assigned_cue_ball == BallColor.WHITE, "즉시 득점 후 공이 멈췄을 때 턴이 상대방으로 넘어갔습니다!"
    print("[PASS] Test 5 (즉시 득점 후 잔여 구름 동결 및 정지 시 공격권 100% 유지) 통과!")


def test_rule1_to_rule4_cue_identification_and_1cushion_bankshot() -> None:
    """
    [테스트 6: 4대 핵심 규칙 통합 검증]
    - 규칙 1: 3구 상대 색도 랭킹(WHITE vs YELLOW vs RED) 및 정지 상태(IDLE)에서 가장 먼저 움직이는 공(First-Mover) 수구 결정
    - 규칙 2: 득점 인정(scored == True) 시 현재 선수의 수구 그대로 유지
    - 규칙 3: 득점 실패(scored == False) 시 상대방 수구로 즉시 전환 (WHITE <-> YELLOW)
    - 규칙 4: 제1적구가 쿠션 바로 앞(y=62mm)에 서 있는 '1쿠션 넣어치기(원뱅크 걸어치기) 3쿠션 득점' 100% 인식
    """
    import numpy as np
    from ball_identity_tracker import LabHungarianBallIdentifier
    from cue24.core.color_locker import ColorLocker
    from cue24.core.scoring_engine import ScoringEngine

    # 1. 상대 색도 랭킹(Relative Chromaticity Ranking) 검증 (조명이 따뜻한 색온도여서 흰공도 약간 노란빛을 띠는 환경 시뮬레이션)
    frame_bgr = np.full((200, 300, 3), (180, 90, 20), dtype=np.uint8)
    # 공 1 (따뜻한 조명 아래의 흰공: BGR=(215, 232, 240))
    # 공 2 (노란공: BGR=(45, 215, 248))
    # 공 3 (적구: BGR=(35, 35, 225))
    import cv2
    cv2.circle(frame_bgr, (50, 100), 18, (215, 232, 240), -1)
    cv2.circle(frame_bgr, (150, 100), 18, (45, 215, 248), -1)
    cv2.circle(frame_bgr, (250, 100), 18, (35, 35, 225), -1)

    identifier = LabHungarianBallIdentifier()
    boxes = [
        (30.0, 80.0, 70.0, 120.0),
        (130.0, 80.0, 170.0, 120.0),
        (230.0, 80.0, 270.0, 120.0),
    ]
    assigned = identifier.assign_ball_identities(frame_bgr, boxes, is_idle_state=True)
    assert assigned[BallColor.WHITE] == (30.0, 80.0, 70.0, 120.0)
    assert assigned[BallColor.YELLOW] == (130.0, 80.0, 170.0, 120.0)
    assert assigned[BallColor.RED] == (230.0, 80.0, 270.0, 120.0)

    locker = ColorLocker()
    crops = {
        1: frame_bgr[80:120, 30:70],
        2: frame_bgr[80:120, 130:170],
        3: frame_bgr[80:120, 230:270],
    }
    locked_colors = locker.classify_balls(crops, use_temporal_smoothing=False)
    assert locked_colors[1] == "WHITE"
    assert locked_colors[2] == "YELLOW"
    assert locked_colors[3] == "RED"

    # 2. 규칙 1 (First-Mover Onset) + 규칙 4 (1쿠션 넣어치기 원뱅크 3쿠션 득점) + 규칙 2 (득점 시 수구 유지)
    # 초기 설정은 WHITE였으나, 정지 상태(IDLE)에서 YELLOW가 가장 먼저 출발하는 상황!
    fsm = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE)
    engine = ScoringEngine()
    engine.reset_shot(cue_ball_id=2, object_ball_ids=[1, 3])

    # 제1적구(RED)가 TOP 쿠션 바로 앞(x=620, y=62)에 위치!
    # YELLOW(수구)가 (400, 220)에서 출발해 TOP 쿠션(x=570, y=42 -> RED와의 거리가 53mm로 68mm 미만!)을 먼저 치고
    # 직후 RED(620, 62)를 타격한 뒤 RIGHT(x=2800) -> BOTTOM(y=1382)을 거쳐 WHITE(2100, 700)를 타격하는 1쿠션 넣어치기!
    shot_frames = [
        (0.00, (2100.0, 700.0), (400.0, 220.0), (620.0, 62.0)),
        (0.02, (2100.0, 700.0), (445.0, 170.0), (620.0, 62.0)),  # YELLOW 단독 출발 (disp > 12mm, speed > 65mm/s) -> First-Mover!
        (0.04, (2100.0, 700.0), (505.0, 105.0), (620.0, 62.0)),
        (0.06, (2100.0, 700.0), (568.0, 42.0),  (620.0, 62.0)),  # 1쿠션(TOP, 제1적구 RED와 거리 55.7mm 인접!)
        (0.08, (2100.0, 700.0), (595.0, 60.0),  (620.0, 62.0)),  # 제1적구(RED) 접촉!
        (0.10, (2100.0, 700.0), (680.0, 130.0), (660.0, 85.0)),  # RED 밀려남
        (0.14, (2100.0, 700.0), (900.0, 220.0), (690.0, 100.0)),
        (0.18, (2100.0, 700.0), (1140.0, 310.0), (710.0, 110.0)),
        (0.22, (2100.0, 700.0), (1380.0, 400.0), (720.0, 115.0)),
        (0.26, (2100.0, 700.0), (1620.0, 490.0), (720.0, 115.0)),
        (0.30, (2100.0, 700.0), (1860.0, 580.0), (720.0, 115.0)),
        (0.34, (2100.0, 700.0), (2100.0, 660.0), (720.0, 115.0)),
        (0.38, (2100.0, 700.0), (2340.0, 740.0), (720.0, 115.0)),
        (0.42, (2100.0, 700.0), (2580.0, 820.0), (720.0, 115.0)),
        (0.46, (2100.0, 700.0), (2802.0, 900.0), (720.0, 115.0)), # 2쿠션(RIGHT)
        (0.50, (2100.0, 700.0), (2620.0, 1060.0), (720.0, 115.0)),
        (0.54, (2100.0, 700.0), (2440.0, 1220.0), (720.0, 115.0)),
        (0.58, (2100.0, 700.0), (2260.0, 1384.0), (720.0, 115.0)), # 3쿠션(BOTTOM)
        (0.62, (2100.0, 700.0), (2210.0, 1160.0), (720.0, 115.0)),
        (0.66, (2100.0, 700.0), (2165.0, 940.0), (720.0, 115.0)),
        (0.70, (2100.0, 700.0), (2120.0, 750.0), (720.0, 115.0)), # 제2적구(WHITE: 2100, 700) 접촉!
        (0.74, (2060.0, 665.0), (2090.0, 620.0), (720.0, 115.0)),
        (0.78, (2020.0, 635.0), (2060.0, 540.0), (720.0, 115.0)),
    ]

    imm_out = None
    for f_i, (t, w_pt, y_pt, r_pt) in enumerate(shot_frames):
        out = fsm.update(t, {BallColor.WHITE: w_pt, BallColor.YELLOW: y_pt, BallColor.RED: r_pt})
        if out is not None and not out.already_dispatched:
            imm_out = out
        engine.update_frame(f_i + 1, {1: w_pt, 2: y_pt, 3: r_pt})

    for s in range(1, 45):
        t = 0.78 + s / 60.0
        fsm.update(t, {BallColor.WHITE: (2020.0, 635.0), BallColor.YELLOW: (2060.0, 540.0), BallColor.RED: (720.0, 115.0)})

    eval_eng = engine.evaluate_shot()
    assert imm_out is not None and imm_out.scored is True, "1쿠션 넣어치기 3쿠션 득점이 인식되지 않았습니다!"
    assert imm_out.cue_ball == BallColor.YELLOW, "First-Mover(YELLOW)가 수구로 결정되지 않았습니다!"
    assert imm_out.is_bank_shot is True, "1쿠션 넣어치기가 뱅크샷(is_bank_shot=True)으로 판정되지 않았습니다!"
    assert imm_out.cushions_before_first_hit == 1, f"선쿠션 수 오류: {imm_out.cushions_before_first_hit}"
    assert imm_out.total_cushions == 3, f"총 쿠션 수 오류: {imm_out.total_cushions}"
    assert eval_eng["is_scored"] is True and eval_eng["shot_type"] == "BANK_SHOT", f"eval_eng={eval_eng}"
    # 규칙 2: 득점 인정 시 수구(YELLOW) 그대로 유지!
    assert fsm.assigned_cue_ball == BallColor.YELLOW, "득점했는데 수구가 YELLOW에서 유지되지 않았습니다!"

    # 3. 규칙 3: 다음 샷에서 YELLOW가 득점 실패(MISS)하면 수구가 상대방(WHITE)으로 전환되는지 검증
    miss_frames = [
        (2.00, (2020.0, 635.0), (2060.0, 540.0), (720.0, 115.0)),
        (2.03, (2020.0, 635.0), (1900.0, 540.0), (720.0, 115.0)),
        (2.06, (2020.0, 635.0), (1700.0, 540.0), (720.0, 115.0)),
        (2.09, (2020.0, 635.0), (1500.0, 540.0), (720.0, 115.0)),
    ]
    for t, w_pt, y_pt, r_pt in miss_frames:
        fsm.update(t, {BallColor.WHITE: w_pt, BallColor.YELLOW: y_pt, BallColor.RED: r_pt})
    miss_res = None
    for s in range(1, 45):
        t = 2.09 + s / 60.0
        out = fsm.update(t, {BallColor.WHITE: (2020.0, 635.0), BallColor.YELLOW: (1500.0, 540.0), BallColor.RED: (720.0, 115.0)})
        if out is not None:
            miss_res = out

    assert miss_res is not None and miss_res.scored is False
    assert fsm.assigned_cue_ball == BallColor.WHITE, "득점 실패 후 상대방 수구(WHITE)로 전환되지 않았습니다!"
    print("[PASS] Test 6 (규칙 1~4: 3구 상대 색도 랭킹 + First-Mover 수구 결정 + 1쿠션 넣어치기 원뱅크 득점 + 득점유지/실패교체) 통과!")


def test_user_reported_3_critical_errors_regression() -> None:
    """
    [테스트 7: 사용자 제보 3대 치명적 오류 완벽 해결 회귀 검증]
    1. 오류 1: 수구가 레일 근처에서 출발하거나 쿠션 앞 제1적구를 먼저 맞힌 일반 3쿠션인데 뱅크샷으로 오인하는 현상 차단
    2. 오류 2: 공이 매우 빠른 경우(초고속 타구: 프레임 사이 Sub-frame 반사 및 사다리꼴 평탄 정점) 쿠션 터치 인식을 놓치는 현상 복원
    3. 오류 3: 제1적구 충돌 후 3쿠션 후 제2적구에 충돌하지 않아 득점 실패(Miss)인데 뱅크샷 득점으로 오인하는 현상 차단
    """
    from cue24.core.scoring_engine import ScoringEngine

    # ----------------------------------------------------------------------
    # 시나리오 A (오류 1 해결 검증):
    # 수구(WHITE)가 하단 레일 근처(y=52mm)에 서 있다가 테이블 안쪽의 제1적구(RED: 650, 220)를 먼저 직접 타격한 뒤
    # TOP -> RIGHT -> BOTTOM 3쿠션을 돌고 제2적구(YELLOW: 2100, 950)를 맞히는 '일반 3쿠션(REGULAR_3C)'
    # (과거 버그: 출발 위치가 y=52mm(<=58mm)여서 기하학적 원뱅크 복원이 잘못 켜지거나, 1적구 직후 반사가 선쿠션으로 잡혀 뱅크샷으로 오판됨)
    # ----------------------------------------------------------------------
    fsm_a = CausalKissSafeFSM(fps=30.0, assigned_cue_ball=BallColor.WHITE)
    eng_a = ScoringEngine()
    eng_a.reset_shot(cue_ball_id=1, object_ball_ids=[2, 3])

    frames_a = [
        (0.00, (300.0, 52.0),  (2100.0, 950.0), (650.0, 220.0)),
        (0.03, (420.0, 110.0), (2100.0, 950.0), (650.0, 220.0)),
        (0.06, (540.0, 168.0), (2100.0, 950.0), (650.0, 220.0)),
        (0.09, (620.0, 208.0), (2100.0, 950.0), (650.0, 220.0)),  # 제1적구(RED) 직접 먼저 타격!
        (0.12, (740.0, 130.0), (2100.0, 950.0), (695.0, 260.0)),  # RED 밀려남, WHITE는 TOP 레일로 진행
        (0.15, (880.0, 44.0),  (2100.0, 950.0), (725.0, 285.0)),  # 1쿠션 (TOP: 제1적구 타격 후!)
        (0.18, (1350.0, 240.0),(2100.0, 950.0), (745.0, 300.0)),
        (0.21, (1900.0, 470.0),(2100.0, 950.0), (755.0, 310.0)),
        (0.24, (2400.0, 680.0),(2100.0, 950.0), (760.0, 315.0)),
        (0.27, (2802.0, 860.0),(2100.0, 950.0), (760.0, 315.0)),  # 2쿠션 (RIGHT)
        (0.30, (2520.0, 1120.0),(2100.0, 950.0),(760.0, 315.0)),
        (0.33, (2260.0, 1384.0),(2100.0, 950.0),(760.0, 315.0)),  # 3쿠션 (BOTTOM)
        (0.36, (2180.0, 1170.0),(2100.0, 950.0),(760.0, 315.0)),
        (0.39, (2112.0, 990.0), (2100.0, 950.0),(760.0, 315.0)),  # 제2적구(YELLOW) 타격!
        (0.42, (2060.0, 890.0), (2070.0, 880.0),(760.0, 315.0)),
        (0.45, (2020.0, 820.0), (2040.0, 820.0),(760.0, 315.0)),
    ]

    res_a = None
    for idx_f, (t, w_p, y_p, r_p) in enumerate(frames_a):
        out = fsm_a.update(t, {BallColor.WHITE: w_p, BallColor.YELLOW: y_p, BallColor.RED: r_p})
        if out is not None and not out.already_dispatched:
            res_a = out
        eng_a.update_frame(idx_f + 1, {1: w_p, 2: y_p, 3: r_p})

    eval_a = eng_a.evaluate_shot()
    assert res_a is not None and res_a.scored is True, "시나리오 A: 일반 3쿠션 득점이 인식되지 않았습니다!"
    assert res_a.is_bank_shot is False, f"시나리오 A 오류: 일반 3쿠션이 뱅크샷으로 오인되었습니다! (c_before={res_a.cushions_before_first_hit})"
    assert res_a.cushions_before_first_hit == 0, f"시나리오 A 오류: 선쿠션 수가 0이어야 합니다 ({res_a.cushions_before_first_hit})"
    assert eval_a["is_scored"] is True and eval_a["shot_type"] == "REGULAR_3C", f"시나리오 A ScoringEngine 오류: {eval_a}"

    # ----------------------------------------------------------------------
    # 시나리오 B (오류 2 해결 검증):
    # 초고속 강타로 인해 수구가 프레임 사이(Sub-frame)에 레일을 찍고 나와서
    # 샘플링된 좌표가 레일에서 165mm~185mm 떨어져 있거나(기존 125mm/145mm 대역 밖),
    # 인접 두 프레임이 사다리꼴 평탄 정점(y=52 -> y=52)을 형성하는 초고속 3쿠션 득점
    # ----------------------------------------------------------------------
    fsm_b = CausalKissSafeFSM(fps=30.0, assigned_cue_ball=BallColor.WHITE)
    eng_b = ScoringEngine()
    eng_b.reset_shot(cue_ball_id=1, object_ball_ids=[2, 3])

    frames_b = [
        (0.00, (400.0, 700.0),  (2200.0, 500.0), (650.0, 680.0)),
        (0.03, (580.0, 685.0),  (2200.0, 500.0), (650.0, 680.0)),
        (0.06, (625.0, 680.0),  (2200.0, 500.0), (650.0, 680.0)),  # 제1적구(RED) 타격
        (0.09, (820.0, 430.0),  (2200.0, 500.0), (720.0, 740.0)),
        # 1쿠션(TOP): 초고속 사다리꼴 평탄 정점 (두 프레임 연속 y=52.0mm)
        (0.12, (1010.0, 52.0),  (2200.0, 500.0), (760.0, 770.0)),
        (0.15, (1190.0, 52.0),  (2200.0, 500.0), (780.0, 785.0)),
        (0.18, (1480.0, 310.0), (2200.0, 500.0), (790.0, 790.0)),
        (0.21, (1820.0, 610.0), (2200.0, 500.0), (790.0, 790.0)),
        (0.24, (2200.0, 940.0), (2200.0, 500.0), (790.0, 790.0)),
        # 2쿠션(BOTTOM): 초고속 서브프레임 반사로 인해 프레임 샘플이 y=1248mm (레일 거리 174mm > 125mm!)에서 찍힘
        (0.27, (2490.0, 1248.0),(2200.0, 500.0), (790.0, 790.0)),
        (0.30, (2670.0, 980.0), (2200.0, 500.0), (790.0, 790.0)),
        # 3쿠션(RIGHT): 초고속 서브프레임 반사로 인해 프레임 샘플이 x=2668mm (레일 거리 174mm > 125mm!)에서 찍힘
        (0.33, (2668.0, 730.0), (2200.0, 500.0), (790.0, 790.0)),
        (0.36, (2410.0, 600.0), (2200.0, 500.0), (790.0, 790.0)),
        (0.39, (2230.0, 515.0), (2200.0, 500.0), (790.0, 790.0)),  # 제2적구(YELLOW) 타격!
        (0.42, (2120.0, 480.0), (2140.0, 465.0), (790.0, 790.0)),
        (0.45, (2050.0, 460.0), (2090.0, 435.0), (790.0, 790.0)),
    ]

    res_b = None
    for idx_f, (t, w_p, y_p, r_p) in enumerate(frames_b):
        out = fsm_b.update(t, {BallColor.WHITE: w_p, BallColor.YELLOW: y_p, BallColor.RED: r_p})
        if out is not None and not out.already_dispatched:
            res_b = out
        eng_b.update_frame(idx_f + 1, {1: w_p, 2: y_p, 3: r_p})

    eval_b = eng_b.evaluate_shot()
    assert res_b is not None and res_b.scored is True, f"시나리오 B 오류: 초고속 쿠션 반사가 누락되어 득점 실패했습니다! (cushions={fsm_b.cushion_count})"
    assert res_b.total_cushions == 3 and res_b.is_bank_shot is False, f"시나리오 B 쿠션/뱅크샷 오류: {res_b}"
    assert eval_b["is_scored"] is True and eval_b["shot_type"] == "REGULAR_3C", f"시나리오 B ScoringEngine 오류: {eval_b}"

    # ----------------------------------------------------------------------
    # 시나리오 C (오류 3 해결 검증):
    # 제1적구(RED) 충돌 후 3쿠션을 돌고 제2적구(YELLOW: 2100, 95.0) 근처 레일(y=68mm)을 맞고 튕겨 나갔으나
    # 제2적구에는 닿지 않은 득점 실패(Miss) 상황 (제2적구에 미세한 트래커 지터 1.8mm 발생 포함)
    # -> 절대 뱅크샷 득점이나 일반 득점으로 인식되면 안 됨!
    # ----------------------------------------------------------------------
    fsm_c = CausalKissSafeFSM(fps=30.0, assigned_cue_ball=BallColor.WHITE)
    eng_c = ScoringEngine()
    eng_c.reset_shot(cue_ball_id=1, object_ball_ids=[2, 3])

    frames_c = [
        (0.00, (400.0, 700.0),  (2100.0, 95.0), (700.0, 700.0)),
        (0.03, (560.0, 700.0),  (2100.0, 95.0), (700.0, 700.0)),
        (0.06, (655.0, 695.0),  (2100.0, 95.0), (700.0, 700.0)),  # 제1적구(RED) 충돌
        (0.09, (840.0, 420.0),  (2100.0, 95.0), (760.0, 745.0)),
        (0.12, (1050.0, 45.0),  (2100.0, 95.0), (795.0, 770.0)),  # 1쿠션(TOP)
        (0.15, (1550.0, 380.0), (2100.0, 95.0), (810.0, 780.0)),
        (0.18, (2150.0, 780.0), (2100.0, 95.0), (815.0, 785.0)),
        (0.21, (2802.0, 1120.0),(2100.0, 95.0), (815.0, 785.0)),  # 2쿠션(RIGHT)
        (0.24, (2480.0, 1382.0),(2100.0, 95.0), (815.0, 785.0)),  # 3쿠션(BOTTOM)
        (0.27, (2280.0, 750.0), (2100.0, 95.0), (815.0, 785.0)),
        # 4쿠션(TOP: x=2152, y=68) -> 제2적구(2100, 95)와의 거리가 58.5mm로 가깝지만,
        # 실제로는 TOP 쿠션을 맞고 아래로 튕겨 나간 것이며 제2적구는 맞지 않음(지터 1.5mm만 존재)!
        (0.30, (2152.0, 68.0),  (2100.0, 95.0), (815.0, 785.0)),
        (0.33, (2040.0, 320.0), (2101.2, 94.2), (815.0, 785.0)),
        (0.36, (1920.0, 580.0), (2101.0, 94.5), (815.0, 785.0)),
    ]

    imm_c = None
    for idx_f, (t, w_p, y_p, r_p) in enumerate(frames_c):
        out = fsm_c.update(t, {BallColor.WHITE: w_p, BallColor.YELLOW: y_p, BallColor.RED: r_p})
        if out is not None and not out.already_dispatched:
            imm_c = out
        eng_c.update_frame(idx_f + 1, {1: w_p, 2: y_p, 3: r_p})

    final_c = None
    for s in range(1, 45):
        t = 0.36 + s / 30.0
        out = fsm_c.update(t, {BallColor.WHITE: (1920.0, 580.0), BallColor.YELLOW: (2101.0, 94.5), BallColor.RED: (815.0, 785.0)})
        if out is not None:
            final_c = out

    eval_c = eng_c.evaluate_shot()
    assert imm_c is None, f"시나리오 C 오류: 제2적구 미충돌(Miss)인데 실시간 득점이 발생했습니다! ({imm_c})"
    assert final_c is not None and final_c.scored is False, f"시나리오 C 오류: 최종 결과가 득점 실패(False)여야 합니다! ({final_c})"
    assert eval_c["is_scored"] is False, f"시나리오 C ScoringEngine 오류: {eval_c}"

    # ----------------------------------------------------------------------
    # 시나리오 D (초고속 수구 1적구 선타격 -> 직후 쿠션 반사 순서 역전 방지 검증):
    # 수구(WHITE)가 빠르게(첫 스텝 210mm+) 이동하여 레일 근처 제1적구(RED: 620, 78)를 얇게 먼저 치고(impact_param ≈ 48mm),
    # 바로 다음 프레임(f_post1)에 뒤쪽 TOP 쿠션(y=42mm)을 맞힌 뒤 RIGHT -> BOTTOM 3쿠션 후 제2적구(YELLOW) 득점!
    # (과거 버그: 2프레임 후행 컨텍스트 중 TOP 쿠션 y=42mm가 min_rail_dist_before_first_hit에 먼저 기록되어
    #  '1쿠션 => 1적구' 뱅크샷으로 순서가 뒤집히던 현상 완벽 차단 검증)
    # ----------------------------------------------------------------------
    fsm_d = CausalKissSafeFSM(fps=30.0, assigned_cue_ball=BallColor.WHITE)
    eng_d = ScoringEngine()
    eng_d.reset_shot(cue_ball_id=1, object_ball_ids=[2, 3])

    frames_d = [
        (0.00, (360.0, 240.0), (2100.0, 750.0), (620.0, 78.0)),
        (0.03, (575.0, 118.0), (2100.0, 750.0), (620.0, 78.0)),  # 초고속 접근 후 제1적구(RED: 620, 78) 얇은 두께 선타격!
        (0.06, (665.0, 42.0),  (2100.0, 750.0), (655.0, 102.0)), # 직후 프레임에 뒤쪽 TOP 레일(y=42mm) 1쿠션 도달 & RED 밀려남!
        (0.09, (1120.0, 210.0),(2100.0, 750.0), (685.0, 120.0)),
        (0.12, (1680.0, 430.0),(2100.0, 750.0), (705.0, 132.0)),
        (0.15, (2260.0, 660.0),(2100.0, 750.0), (715.0, 138.0)),
        (0.18, (2802.0, 880.0),(2100.0, 750.0), (720.0, 140.0)),  # 2쿠션(RIGHT)
        (0.21, (2520.0, 1140.0),(2100.0, 750.0),(720.0, 140.0)),
        (0.24, (2260.0, 1384.0),(2100.0, 750.0),(720.0, 140.0)),  # 3쿠션(BOTTOM)
        (0.27, (2175.0, 1050.0),(2100.0, 750.0),(720.0, 140.0)),
        (0.30, (2112.0, 790.0), (2100.0, 750.0),(720.0, 140.0)),  # 제2적구(YELLOW) 충돌!
        (0.33, (2060.0, 660.0), (2065.0, 690.0),(720.0, 140.0)),
        (0.36, (2010.0, 540.0), (2035.0, 640.0),(720.0, 140.0)),
    ]

    res_d = None
    for idx_f, (t, w_p, y_p, r_p) in enumerate(frames_d):
        out = fsm_d.update(t, {BallColor.WHITE: w_p, BallColor.YELLOW: y_p, BallColor.RED: r_p})
        if out is not None and not out.already_dispatched:
            res_d = out
        eng_d.update_frame(idx_f + 1, {1: w_p, 2: y_p, 3: r_p})

    eval_d = eng_d.evaluate_shot()
    assert res_d is not None and res_d.scored is True, "시나리오 D: 고속 일반 3쿠션 득점이 인식되지 않았습니다!"
    assert res_d.cushions_before_first_hit == 0 and res_d.is_bank_shot is False, (
        f"시나리오 D 오류: 고속 수구가 1적구를 먼저 맞혔는데 쿠션 먼저(뱅크샷)로 오인됨! (c_before={res_d.cushions_before_first_hit})"
    )
    assert res_d.total_cushions == 3, f"시나리오 D 쿠션 수 오류: {res_d.total_cushions}"
    assert eval_d["is_scored"] is True and eval_d["shot_type"] == "REGULAR_3C" and eval_d["cushions_before_first_hit"] == 0, (
        f"시나리오 D ScoringEngine 오류: {eval_d}"
    )
    print("[PASS] Test 7 (고속 수구 1적구 선타격->쿠션 순서 역전 방지 포함 4대 시나리오 전수 검증) 통과!")


if __name__ == "__main__":
    test_1_cushion_and_2_cushion_bankshot_exact_count()
    test_order_reversal_prevents_false_bankshot()
    test_same_rail_sliding_no_extra_cushions()
    test_opponent_ball_noise_never_hijacks_turn_or_scores()
    test_residual_rolling_after_immediate_score_never_flips_turn()
    test_rule1_to_rule4_cue_identification_and_1cushion_bankshot()
    test_user_reported_3_critical_errors_regression()
    print("\n=== ALL BANKSHOT & TURN-LOCK INTEGRATION TESTS PASSED! ===")
