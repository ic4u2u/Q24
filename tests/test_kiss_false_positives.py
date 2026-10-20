"""
Step 2: 8대 악성 키스 오판 방어 및 정상 득점 자동 검증 테스트 스위트 (tests/test_kiss_false_positives.py)
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scoring_fsm import BallColor, CausalKissSafeFSM, ShotResult


def _simulate_keyframes(
    fsm: CausalKissSafeFSM,
    keyframes: List[Tuple[float, Tuple[float, float], Tuple[float, float], Tuple[float, float]]],
    frame_overrides: Optional[Dict[int, Dict[BallColor, Optional[Tuple[float, float]]]]] = None
) -> ShotResult:
    dt = 1.0 / 60.0
    global_frame_idx = 0

    for i in range(len(keyframes) - 1):
        t0, w0, y0, r0 = keyframes[i]
        t1, w1, y1, r1 = keyframes[i + 1]
        steps = max(int(round((t1 - t0) / dt)), 1)
        for s in range(steps):
            a = s / steps
            t = t0 + (t1 - t0) * a
            coords: Dict[BallColor, Optional[Tuple[float, float]]] = {
                BallColor.WHITE: (w0[0] + (w1[0] - w0[0]) * a, w0[1] + (w1[1] - w0[1]) * a),
                BallColor.YELLOW: (y0[0] + (y1[0] - y0[0]) * a, y0[1] + (y1[1] - y0[1]) * a),
                BallColor.RED: (r0[0] + (r1[0] - r0[0]) * a, r0[1] + (r1[1] - r0[1]) * a),
            }
            if frame_overrides and global_frame_idx in frame_overrides:
                for c, override_val in frame_overrides[global_frame_idx].items():
                    coords[c] = override_val

            fsm.update(t, coords)
            global_frame_idx += 1

    last_t, w_end, y_end, r_end = keyframes[-1]
    for s in range(1, 35):
        res = fsm.update(last_t + s * dt, {
            BallColor.WHITE: w_end,
            BallColor.YELLOW: y_end,
            BallColor.RED: r_end,
        })
        if res is not None:
            return res
    raise RuntimeError("이닝이 종료되지 않았습니다.")


def test_1_kiss_shockwave_immediate_pass_after_2_frames():
    """
    [악성 오판 케이스 1: 키스 충격파 시간창 잔류]
    수구(W)가 3쿠션을 다 돌고 내려오는 순간, 1적구(R)가 2적구(Y)와 키스(t=1.20s)하여 Y가 급가속됨.
    그 직후 단 2프레임(0.033초, t=1.233s) 뒤에 수구(W)가 Y 바로 옆(거리 65mm)을 안 맞고 스쳐 지나감!
    기존 코드는 EMA 및 5프레임 윈도우에 남은 Y의 키스 가속도 때문에 100% 득점으로 오판했으나,
    새 코드는 last_impulse_frame 방화벽과 운동량 차감으로 정확히 노카운트(False) 판정해야 함.
    """
    fsm = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE)
    kf = [
        (0.00, (400, 700),   (2000, 600), (800, 700)),
        (0.20, (740, 700),   (2000, 600), (800, 700)),   # W -> R 1적구 타격
        (0.50, (1300, 35),   (2000, 600), (1250, 600)),  # W 1쿠션(TOP)
        (0.85, (2805, 500),  (2000, 600), (1650, 600)),  # W 2쿠션(RIGHT)
        (1.15, (2150, 1385), (2000, 600), (1900, 600)),  # W 3쿠션(BOTTOM) 완료!
        # t=1.20s: R(1940, 600)이 정지해 있던 Y(2000, 600)를 타격(키스)! W는 (2080, 700) 접근 중
        (1.20, (2080, 700),  (2000, 600), (1940, 600)),
        # t=1.233s (단 2프레임 뒤!): Y는 키스 충격으로 (2025, 600)으로 튀어나가고, W는 (2025, 535)로 거리 65mm 비접촉 직진 통과!
        (1.233,(2025, 535),  (2025, 600), (1930, 600)),
        (1.50, (1580, 200),  (2225, 600), (1900, 600)),  # W와 Y 모두 굴절 없이 각자 직진
    ]
    res = _simulate_keyframes(fsm, kf)
    assert res.scored is False, f"키스 충격파 잔류 오판 발생! 로그: {res.event_log}"
    assert res.object_ball_kiss_occurred is True
    print("[PASS] Case 1: 1적구-2적구 키스 직후 2프레임(0.03초) 뒤 수구 근접 통과 오판 차단 완료")


def test_2_chain_cannon_cue_hits_1st_into_2nd_ball():
    """
    [악성 오판 케이스 2: 3쿠션 후 A -> B -> C 간접 연쇄 키스]
    수구(W)가 3쿠션 후 모여 있는 1적구(R)와 2적구(Y) 쪽으로 진입하여,
    2적구(Y)는 못 맞히고 앞에 있던 1적구(R)만 맞혔는데, 밀려난 1적구(R)가 바로 뒤의 2적구(Y)를 타격함!
    Y가 실제로 강하게 튕겨 나갔더라도 수구(W)가 직접 친 것이 아니므로 노카운트(False)여야 함.
    """
    fsm = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE)
    kf = [
        (0.00, (400, 700),   (2000, 600), (800, 700)),
        (0.20, (740, 700),   (2000, 600), (800, 700)),   # W -> R 1적구 타격
        (0.50, (1300, 35),   (2000, 600), (1500, 600)),  # 1쿠션(TOP)
        (0.85, (2805, 500),  (2000, 600), (1935, 600)),  # 2쿠션(RIGHT), R이 Y 바로 앞(거리 65mm)에 정지
        (1.20, (2200, 1385), (2000, 600), (1935, 600)),  # 3쿠션(BOTTOM)
        # t=1.50s: W가 (1877, 600)으로 진입해 R(1935, 600)만 타격(거리 58mm)! (이 순간 W와 Y의 거리는 123mm, 이후 R이 Y를 침)
        (1.50, (1877, 600),  (2000, 600), (1935, 600)),
        # t=1.533s: R이 밀려나며 Y를 치고(R: 1942, Y: 2000 -> 2200), W는 R에 맞고 꺾여 (1935, 535)로 Y 옆 65mm를 스쳐 지나감
        (1.533,(1935, 535),  (2000, 600), (1942, 600)),
        (1.80, (1800, 350),  (2250, 600), (1945, 620)),
    ]
    res = _simulate_keyframes(fsm, kf)
    assert res.scored is False, f"A->B->C 연쇄 충돌 오판 발생! 로그: {res.event_log}"
    print("[PASS] Case 2: 3쿠션 후 수구가 1적구를 쳐서 1적구가 2적구를 맞힌 간접 연쇄 키스 차단 완료")


def test_3_kiss_id_swap_spot_replacement_foul():
    """
    [악성 오판 케이스 3: 적구 정면 키스 시 트래커 ID 스왑 (자리 바꿈 키스)]
    1적구(R)가 정지한 2적구(Y, 1800, 500)를 정면으로 쳐서 Y가 날아가고 R은 (1800, 500) 자리에 멈춰 섰는데,
    칼만 필터 트래커가 직진 운동을 따라가며 날아간 공을 R로, 멈춰 선 공(실제로는 1적구 R)을 Y로 ID 스왑함!
    이후 3쿠션을 돈 수구(W)가 (1800, 500)에 멈춰 있는 공(실제 1적구 R)을 맞혔을 때,
    ID 역스왑 복원기가 작동하여 이를 '수구-제1적구 재키스(실패)'로 정확히 판정해야 함!
    """
    fsm = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE)
    kf = [
        (0.00, (400, 700),   (1800, 500), (800, 700)),
        (0.20, (740, 700),   (1800, 500), (800, 700)),   # W -> R 1적구 타격
        # t=0.45s: R이 (1800, 500)에 있는 Y를 정면으로 관통!
        # 트래커 ID 스왑 발생 모사: 원래 멈춰야 할 R이 Y를 뚫고 (2200, 500)으로 계속 직진하고, Y는 (1800, 500)에 그대로 정지한 것처럼 입력됨!
        (0.45, (1200, 35),   (1800, 500), (1780, 500)),
        (0.50, (1350, 100),  (1800, 500), (1830, 500)),  # R과 Y 중심 거리 0mm 유령 관통(트래커 ID 스왑!)
        (0.85, (2805, 500),  (1800, 500), (2180, 500)),  # 2쿠션(RIGHT)
        (1.20, (2200, 1385), (1800, 500), (2250, 500)),  # 3쿠션(BOTTOM)
        # t=1.55s: W가 3쿠션 후 (1800, 500)에 남아 있는 공(트래커상 Y로 잘못 라벨링된 실제 1적구 R)을 타격!
        (1.55, (1800, 558),  (1800, 500), (2250, 500)),
        (1.85, (1700, 350),  (1800, 380), (2250, 500)),
    ]
    res = _simulate_keyframes(fsm, kf)
    assert res.kiss_id_swap_corrected is True, "적구 키스 ID 스왑이 감지되지 않았습니다."
    assert res.scored is False, f"자리바꿈 키스 후 1적구 재타격을 득점으로 오판함! 로그: {res.event_log}"
    print("[PASS] Case 3: 적구 정면 키스 시 트래커 ID 스왑(자리바꿈 키스) 자동 역스왑 및 오판 차단 완료")


def test_4_perspective_overlap_58mm_and_spin_curve_without_touch():
    """
    [악성 오판 케이스 4: 원근 시차로 인해 중심 거리 58mm(<=63mm) 통과 + 수구 스핀 휨]
    기존 코드는 d_min <= 63mm(HARD_CONTACT_DIST)이면 속도 변화를 아예 안 보고 무조건 득점 처리했음.
    수구(W)가 3쿠션 후 정지해 있는 2적구(Y) 옆을 투영 거리 58mm로 스쳐 지나가며 스핀으로 2.2도 휘어졌으나,
    Y는 미동도 하지 않은 경우(속도 0 유지) -> 무조건 노카운트(False)여야 함!
    """
    fsm = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE)
    kf = [
        (0.00, (400, 700),   (2000, 500), (800, 700)),
        (0.20, (740, 700),   (2000, 500), (800, 700)),
        (0.50, (1300, 35),   (2000, 500), (1000, 800)),
        (0.85, (2805, 500),  (2000, 500), (1050, 850)),
        (1.20, (2300, 1385), (2000, 500), (1050, 850)),
        (1.50, (2000, 558),  (2000, 500), (1050, 850)),  # 거리 58mm 통과 (Y는 미동도 안 함!)
        (1.80, (1710, 100),  (2000, 500), (1050, 850)),
    ]
    res = _simulate_keyframes(fsm, kf)
    assert res.scored is False, f"58mm 원근 왜곡 비접촉 통과를 득점으로 오판함! 로그: {res.event_log}"
    print("[PASS] Case 4: 투영 거리 58mm(<=63mm) 비접촉 통과 무조건 득점 버그 차단 완료")


def test_5_moving_second_ball_friction_and_bbox_jitter_after_kiss():
    """
    [악성 오판 케이스 5: 키스로 굴러가는 2적구의 마찰 감속 + 바운딩 박스 노이즈]
    1적구(R)와 키스하여 굴러가고 있는 2적구(Y)가 마찰로 감속 중일 때,
    3쿠션 후 수구(W)가 Y 옆(68mm)을 지나가는 순간 Y에 10mm 바운딩 박스 떨림이 발생해도 노카운트여야 함.
    """
    fsm = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE)
    kf = [
        (0.00, (400, 700),   (1800, 500), (800, 700)),
        (0.20, (740, 700),   (1800, 500), (800, 700)),
        (0.45, (1200, 35),   (1800, 500), (1742, 500)),  # R <-> Y 키스!
        (0.80, (2805, 500),  (1980, 500), (1700, 520)),
        (1.15, (2300, 1385), (2120, 500), (1680, 530)),
        (1.45, (2200, 568),  (2200, 500), (1670, 535)),  # W가 굴러가는 Y 옆 68mm 통과
        (1.75, (2100, 200),  (2250, 500), (1670, 535)),  # Y가 마찰 감속
    ]
    res = _simulate_keyframes(fsm, kf)
    assert res.scored is False
    print("[PASS] Case 5: 키스 후 이동 중인 2적구 마찰 감속 착시 방어 완료")


def test_6_kiss_pushes_2nd_ball_into_early_cue_kiss_locks_no_score():
    """
    [악성 오판 케이스 6: 1적구와 키스해 튕겨 나온 2적구가 2쿠션 수구와 조기 키스 후 3쿠션 뒤 재접촉]
    1적구(R)에 맞고 튕긴 2적구(Y)가 2쿠션 진행 중인 수구(W)와 부딪힘(조기 키스 -> NO_SCORE_LOCKED).
    이후 수구(W)가 3번째 쿠션을 맞고 다시 Y를 타격하더라도 이미 조기 키스로 잠겼으므로 노카운트여야 함!
    """
    fsm = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE)
    kf = [
        (0.00, (400, 700),   (1800, 500), (800, 700)),
        (0.20, (740, 700),   (1800, 500), (800, 700)),   # W -> R 타격
        (0.45, (1200, 35),   (1800, 500), (1742, 500)),  # 1쿠션(TOP) & R <-> Y 키스
        (0.80, (2805, 500),  (2100, 500), (1700, 520)),  # 2쿠션(RIGHT)
        (1.05, (2300, 555),  (2300, 500), (1700, 520)),  # 2쿠션 상태에서 굴러오던 Y와 W가 조기 충돌!
        (1.35, (2000, 1385), (2400, 420), (1700, 520)),  # 3쿠션(BOTTOM)
        (1.65, (2400, 475),  (2400, 420), (1700, 520)),  # 3쿠션 후 Y 재타격 (무효 잠금 상태)
        (1.90, (2300, 300),  (2400, 300), (1700, 520)),
    ]
    res = _simulate_keyframes(fsm, kf)
    assert res.scored is False
    assert "EARLY_SECOND_BALL_KISS" in (res.fail_reason or "")
    print(f"[PASS] Case 6: 적구 키스 유발 2쿠션 조기 접촉 잠금 검증 완료 -> {res.fail_reason}")


def test_7_kiss_then_genuine_thin_score_on_moving_ball():
    """
    [정상 득점 케이스 7: 1적구-2적구 키스 후 굴러가는 2적구를 수구가 3쿠션 뒤 실제로 얇게 타격]
    적구 간 키스로 인해 2적구(Y)가 이동 중이더라도, 3쿠션 후 수구(W)가 실제로 Y를 타격하여
    W와 Y 모두에 쌍방 작용-반작용 충격량(Δv_W 및 Δv_Y)이 발생하면 정확히 득점(True)으로 인정해야 함!
    """
    fsm = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE)
    kf = [
        (0.00, (400, 700),   (1800, 500), (800, 700)),
        (0.20, (740, 700),   (1800, 500), (800, 700)),   # W -> R 타격
        (0.45, (1200, 35),   (1800, 500), (1742, 500)),  # 1쿠션(TOP) & R <-> Y 키스
        (0.80, (2805, 500),  (2000, 500), (1700, 520)),  # 2쿠션(RIGHT), Y 우측 이동 중
        (1.15, (2300, 1385), (2120, 500), (1700, 520)),  # 3쿠션(BOTTOM), Y 계속 우측 이동 중
        (1.45, (2200, 560),  (2200, 500), (1700, 520)),  # W가 이동 중인 Y를 실제로 타격!
        (1.75, (2020, 320),  (2260, 360), (1700, 520)),  # 타격 후 Y는 아래(y: 500->360)로 꺾이고 W도 반작용으로 꺾임!
    ]
    res = _simulate_keyframes(fsm, kf)
    assert res.scored is True, f"키스 후 실제 2적구 타격 득점 인식 실패: {res.fail_reason}"
    assert res.object_ball_kiss_occurred is True
    print("[PASS] Case 7: 적구 간 키스 후 이동 중인 2적구 실제 타격 시 정상 득점(+1) 검증 완료")


def test_8_kiss_then_genuine_score_on_stationary_ball():
    """
    [정상 득점 케이스 8: 1적구-2적구 키스 후 정지한 2적구를 수구가 3쿠션 뒤 정상 타격]
    """
    fsm = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE)
    kf = [
        (0.00, (400, 700),   (1800, 500), (800, 700)),
        (0.20, (740, 700),   (1800, 500), (800, 700)),
        (0.45, (1200, 35),   (1800, 500), (1742, 500)),
        (0.80, (2805, 500),  (2100, 500), (1700, 520)),
        (1.15, (2300, 1385), (2100, 500), (1700, 520)),
        (1.50, (2100, 558),  (2100, 500), (1700, 520)),
        (1.80, (2000, 350),  (2100, 380), (1700, 520)),
    ]
    res = _simulate_keyframes(fsm, kf)
    assert res.scored is True
    assert res.object_ball_kiss_occurred is True
    print("[PASS] Case 8: 적구 간 키스 후 정지한 2적구 정상 타격 득점(+1) 검증 완료")


if __name__ == "__main__":
    test_1_kiss_shockwave_immediate_pass_after_2_frames()
    test_2_chain_cannon_cue_hits_1st_into_2nd_ball()
    test_3_kiss_id_swap_spot_replacement_foul()
    test_4_perspective_overlap_58mm_and_spin_curve_without_touch()
    test_5_moving_second_ball_friction_and_bbox_jitter_after_kiss()
    test_6_kiss_pushes_2nd_ball_into_early_cue_kiss_locks_no_score()
    test_7_kiss_then_genuine_thin_score_on_moving_ball()
    test_8_kiss_then_genuine_score_on_stationary_ball()
    print("\n=== 8대 악성 키스 오판 방어 및 정상 득점 테스트 전체 PASS 완료! ===")
