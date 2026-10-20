"""
Step 3 자동 검증 테스트 스위트 (tests/test_kiss_and_fouls.py)
1) 4대 키스 착시 방어 (오류 1 ~ 4)
2) 투터치(Double Hit / 밀어치기) 파울 자동 감지
3) 바운딩 박스 병합(Blob Merging) 중심점 점프 및 결측(None) 방어
4) BilliardNanoGPT.rollout_occlusion 직접 통합 가려짐 구간 쿠션 반사 복원 검증
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
    """
    키프레임 사이를 60fps(dt=1/60s)로 선형 보간하여 FSM에 공급합니다.
    frame_overrides를 통해 특정 프레임에 바운딩 박스 점프(Centroid Shift)나 결측(None)을 주입할 수 있습니다.
    """
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
    raise RuntimeError("이닝이 정상 종료되지 않았습니다.")


def test_1_object_ball_kiss_then_near_miss():
    """1. 1적구(RED)가 2적구(YELLOW)와 키스해 굴러갈 때 수구가 3쿠션 후 옆(71mm)을 안 맞고 통과 -> 노카운트"""
    fsm = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE)
    kf = [
        (0.00, (400, 700),   (1800, 500), (800, 700)),
        (0.20, (740, 700),   (1800, 500), (800, 700)),
        (0.45, (1200, 35),   (1800, 500), (1742, 500)),
        (0.80, (2805, 500),  (1950, 500), (1700, 520)),
        (1.15, (2300, 1385), (2100, 500), (1680, 530)),
        (1.45, (2200, 571),  (2200, 500), (1670, 535)),
        (1.75, (2100, 200),  (2300, 500), (1670, 535)),
    ]
    res = _simulate_keyframes(fsm, kf)
    assert res.scored is False
    assert res.object_ball_kiss_occurred is True
    print("[PASS] Test 1: 1적구-2적구 키스 후 수구 근접 비접촉 노카운트 검증 완료")


def test_2_cluster_rehit_first_target():
    """2. 3쿠션 후 1적구와 2적구 밀집 구역에서 1적구(RED)만 재타격하고 굴절 -> 노카운트"""
    fsm = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE)
    kf = [
        (0.00, (400, 700),   (2012, 590), (800, 700)),
        (0.20, (740, 700),   (2012, 590), (800, 700)),
        (0.50, (1300, 35),   (2012, 590), (1400, 650)),
        (0.85, (2805, 600),  (2012, 590), (1940, 650)),
        (1.20, (2300, 1385), (2012, 590), (1940, 650)),
        (1.50, (1940, 590),  (2012, 590), (1940, 650)),
        (1.80, (1750, 450),  (2012, 590), (1940, 740)),
    ]
    res = _simulate_keyframes(fsm, kf)
    assert res.scored is False
    print("[PASS] Test 2: 3구 밀집 구역 제1적구 재키스 착시 방어 검증 완료")


def test_3_rail_bounce_near_second_target():
    """3. 쿠션 근처에 서 있는 2적구(YELLOW) 옆 쿠션만 수구가 맞고 꺾여 나온 경우 -> 노카운트"""
    fsm = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE)
    kf = [
        (0.00, (400, 700),   (1800, 105), (800, 700)),
        (0.20, (740, 700),   (1800, 105), (800, 700)),
        (0.50, (35, 400),    (1800, 105), (900, 800)),
        (0.85, (1000, 1385), (1800, 105), (950, 850)),
        (1.20, (1800, 35),   (1800, 105), (950, 850)),
        (1.55, (2400, 500),  (1800, 105), (950, 850)),
    ]
    res = _simulate_keyframes(fsm, kf)
    assert res.scored is False
    print("[PASS] Test 3: 쿠션 밀착 제2적구 옆 쿠션 반사 착시 방어 검증 완료")


def test_4_object_ball_kiss_then_valid_hit():
    """4. 1적구와 2적구가 키스한 뒤 수구가 3쿠션을 채우고 실제로 2적구를 타격 -> 정상 득점(+1)"""
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
    print("[PASS] Test 4: 적구간 키스 후 정상 3쿠션 제2적구 타격 득점 검증 완료")


def test_5_double_hit_push_foul():
    """5. [신규 보정 1] 수구와 1적구가 95mm로 붙어 있을 때 밀어치기(투터치) 발생 -> 이후 3쿠션 맞혀도 파울(노카운트)"""
    fsm = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE)
    kf = [
        (0.00, (500, 700),   (2200, 400), (595, 700)),
        (0.05, (540, 700),   (2200, 400), (595, 700)),
        (0.12, (650, 700),   (2200, 400), (715, 700)),
        (0.45, (1400, 35),   (2200, 400), (1200, 700)),
        (0.80, (2805, 600),  (2200, 400), (1300, 700)),
        (1.15, (2400, 1385), (2200, 400), (1350, 700)),
        (1.50, (2200, 455),  (2200, 400), (1350, 700)),
        (1.80, (2100, 300),  (2200, 300), (1350, 700)),
    ]
    res = _simulate_keyframes(fsm, kf)
    assert res.scored is False
    assert res.double_hit_foul is True
    assert "DOUBLE_HIT_FOUL" in (res.fail_reason or "")
    print(f"[PASS] Test 5: 투터치(밀어치기) 파울 자동 감지 검증 완료 -> {res.fail_reason}")


def test_6_blob_merging_centroid_shift_immunity():
    """6. [신규 보정 2] 근접 통과 시 바운딩 박스 병합으로 2적구 중심점이 22mm 끌려가고 1프레임 결측(None)되어도 방어"""
    fsm = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE)
    kf = [
        (0.00, (400, 700),   (2000, 500), (800, 700)),
        (0.20, (740, 700),   (2000, 500), (800, 700)),
        (0.50, (1300, 35),   (2000, 500), (1000, 800)),
        (0.85, (2805, 500),  (2000, 500), (1050, 850)),
        (1.20, (2300, 1385), (2000, 500), (1050, 850)),
        (1.50, (2000, 572),  (2000, 500), (1050, 850)),
        (1.80, (1800, 200),  (2000, 500), (1050, 850)),
    ]
    overrides = {
        90: {BallColor.YELLOW: (2000.0, 522.0)},
        91: {BallColor.YELLOW: None},
    }
    res = _simulate_keyframes(fsm, kf, frame_overrides=overrides)
    assert res.scored is False
    assert res.centroid_shift_corrected_count >= 1
    print(
        f"[PASS] Test 6: 바운딩 박스 병합 중심점 점프 및 결측 방어 완료 "
        f"(보정 횟수: {res.centroid_shift_corrected_count}회)"
    )


def test_7_gpt_occlusion_cushion_bounce_and_blob_merge():
    """
    7. [신규 통합 검증] 수구(WHITE)가 1적구(RED) 타격 후 1쿠션(TOP, y=30.75mm)에 입사하는 순간
    선수 신체에 의해 8프레임 연속 가려짐(None)이 발생하더라도,
    _stabilize_blob_merging 내부의 BilliardNanoGPT.rollout_occlusion이 가려진 구간의 TOP 쿠션 반사를 복원하여
    최종 3쿠션 득점(scored == True)을 정상 카운트하는지 검증합니다.
    """
    fsm = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE)
    kf = [
        (0.00, (500, 700),   (2200, 400), (900, 700)),
        (0.20, (845, 700),   (2200, 400), (900, 700)),   # 제1적구(RED) 타격
        (0.50, (1400, 32),   (2200, 400), (950, 800)),   # 1쿠션(TOP) 접촉 시점! (전후 8프레임 동안 선수 몸에 가려짐)
        (0.90, (2805, 600),  (2200, 400), (1000, 850)),  # 2쿠션(RIGHT)
        (1.30, (2400, 1385), (2200, 400), (1020, 870)),  # 3쿠션(BOTTOM)
        (1.70, (2200, 455),  (2200, 400), (1020, 870)),  # 제2적구(YELLOW) 타격 -> 득점!
    ]
    # 0.50초(30번째 프레임) 부근의 27~34번 프레임(총 8프레임) 동안 WHITE 좌표를 None(완전 가려짐)으로 주입
    overrides = {f_idx: {BallColor.WHITE: None} for f_idx in range(27, 35)}
    res = _simulate_keyframes(fsm, kf, frame_overrides=overrides)

    assert res.scored is True, f"가려진 쿠션 반사 복원 실패: {res.fail_reason}, 쿠션수={res.total_cushions}"
    assert res.total_cushions >= 3
    assert res.gpt_occlusion_restored_frames == 8
    print(
        f"[PASS] Test 7: BilliardNanoGPT 가려짐 구간 쿠션 반사 복원 및 3쿠션 득점 판정 완료 "
        f"(복원 프레임: {res.gpt_occlusion_restored_frames}f, 총 쿠션: {res.total_cushions}회)"
    )


if __name__ == "__main__":
    test_1_object_ball_kiss_then_near_miss()
    test_2_cluster_rehit_first_target()
    test_3_rail_bounce_near_second_target()
    test_4_object_ball_kiss_then_valid_hit()
    test_5_double_hit_push_foul()
    test_6_blob_merging_centroid_shift_immunity()
    test_7_gpt_occlusion_cushion_bounce_and_blob_merge()
    print("\n=== 7대 핵심 시나리오 테스트 전체 PASS 완료! ===")
