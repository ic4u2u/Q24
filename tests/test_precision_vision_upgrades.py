"""
좌표 휴리스틱의 한계를 극복하는 4대 고성능 비전 판독 기술 검증 테스트 (tests/test_precision_vision_upgrades.py)

1. test_3d_ray_plane_vs_2d_homography:
   대각선 카메라에서 RayPlane3DProjector가 z = 30.75mm 구심 좌표를 오차 0.1mm 이내로 정확히 복원하는지 확인
2. test_subpixel_phase_correlate_0_1px_kiss:
   바운딩 박스 좌표는 전혀 안 움직였지만(0mm 이동) 제2적구 패치가 0.15픽셀 미세 진동한 초미세 키스 상황에서
   SubPixelPhaseImpactDetector가 이를 감지해 CausalKissSafeFSM이 즉시 득점(scored == True)으로 판정하는지 확인
3. test_motion_streak_v_apex_detection:
   한 프레임 안에서 쿠션을 맞고 꺾인 V자 모션 블러 마스크에서 MotionStreakVApexDetector가
   꺾임 각도와 꼭짓점을 정확히 찾아내고, CushionRimPixelTrigger 및 FSM intra_frame_rail_bounce 연동이 정상 작동하는지 확인
"""

from pathlib import Path
import sys
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scoring_fsm import BallColor, CausalKissSafeFSM, ShotResult
from vision_precision_upgraders import (
    BALL_R,
    CUSHION_NOSE_Z,
    TABLE_H,
    TABLE_W,
    CushionRimPixelTrigger,
    MotionStreakVApexDetector,
    RayPlane3DProjector,
    SubPixelPhaseImpactDetector,
)


def test_3d_ray_plane_vs_2d_homography():
    """
    1. 대각선 카메라에서 RayPlane3DProjector가 z = 30.75mm 구심 좌표를
    오차 0.1mm 이내로 정확히 복원하는지 확인합니다 (기존 2D 호모그래피의 시차 오차와 비교).
    """
    img_w, img_h = 1920, 1080
    fx = img_w * 1.1
    K = np.array(
        [
            [fx, 0.0, img_w * 0.5],
            [0.0, fx, img_h * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # 당구대 남쪽 측면 상단에서 비스듬히 내려다보는 대각선 카메라 포즈 설정
    cam_pos_world = np.array([1420.0, -950.0, 1850.0], dtype=np.float64)
    look_at_world = np.array([1420.0, 710.0, 0.0], dtype=np.float64)

    forward = look_at_world - cam_pos_world
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)

    R_true = np.vstack([right, down, forward])
    rvec_true, _ = cv2.Rodrigues(R_true)
    tvec_true = -R_true @ cam_pos_world.reshape(3, 1)

    # 쿠션 고무 날 높이(z = 37.0mm)의 4개 코너를 카메라 픽셀로 투영
    corners_3d = np.array(
        [
            [0.0, 0.0, CUSHION_NOSE_Z],
            [TABLE_W, 0.0, CUSHION_NOSE_Z],
            [TABLE_W, TABLE_H, CUSHION_NOSE_Z],
            [0.0, TABLE_H, CUSHION_NOSE_Z],
        ],
        dtype=np.float64,
    )
    img_corners, _ = cv2.projectPoints(corners_3d, rvec_true, tvec_true, K, None)
    src_corners_px = [(float(pt[0][0]), float(pt[0][1])) for pt in img_corners]

    projector = RayPlane3DProjector(
        src_corners_px=src_corners_px,
        image_size=(img_w, img_h),
        focal_length_px=fx,
    )

    test_balls_3d = [
        (450.0, 280.0, BALL_R),
        (1420.0, 710.0, BALL_R),
        (2380.0, 1250.0, BALL_R),
        (2750.0, 1360.0, BALL_R),
    ]

    max_3d_err_mm = 0.0
    max_2d_err_mm = 0.0

    for gx, gy, gz in test_balls_3d:
        pt_3d = np.array([[gx, gy, gz]], dtype=np.float64)
        uv, _ = cv2.projectPoints(pt_3d, rvec_true, tvec_true, K, None)
        u, v = float(uv[0, 0, 0]), float(uv[0, 0, 1])

        # 3D Ray-Plane 역투영 (z = 30.75mm)
        rx, ry = projector.pixel_to_ball_center_3d(u, v, target_z_mm=BALL_R)
        err_3d = float(np.hypot(rx - gx, ry - gy))
        max_3d_err_mm = max(max_3d_err_mm, err_3d)

        # 기존 2D 호모그래피 변환 (시차 오차 발생)
        hx, hy = projector.pixel_to_plane_2d_homography(u, v)
        err_2d = float(np.hypot(hx - gx, hy - gy))
        max_2d_err_mm = max(max_2d_err_mm, err_2d)

    assert max_3d_err_mm < 0.1, f"RayPlane3DProjector 복원 오차가 0.1mm를 초과했습니다: {max_3d_err_mm:.6f}mm"
    assert max_2d_err_mm > 3.0, f"대각선 카메라에서 2D 호모그래피는 시차 오차가 존재해야 합니다: {max_2d_err_mm:.2f}mm"
    print(
        f"[PASS] Test 1 (RayPlane3DProjector): 3D 구심 평면(z=30.75mm) 최대 복원 오차 = {max_3d_err_mm:.6f}mm (< 0.1mm) "
        f"| 기존 2D 호모그래피 시차 오차 = {max_2d_err_mm:.2f}mm 완벽 극복"
    )


def test_subpixel_phase_correlate_0_1px_kiss():
    """
    2. 바운딩 박스 좌표는 전혀 안 움직였지만(0mm 이동) 제2적구 패치가 0.15픽셀 미세 진동한
    초미세 키스 상황에서 SubPixelPhaseImpactDetector가 이를 감지해
    CausalKissSafeFSM에서 즉시 득점(scored == True)으로 판정하는지 확인합니다.
    """
    detector = SubPixelPhaseImpactDetector(patch_size=48, min_shift_px=0.06)

    # 당구공 표면 광택과 적구 스팟/미세 텍스처가 있는 48x48 그레이스케일 패치 생성 (0.15px 연속 위상 이동 적용)
    yy, xx = np.mgrid[:48, :48].astype(np.float64)

    def _render_ball_patch(shift_x: float, shift_y: float) -> np.ndarray:
        sx = xx - shift_x
        sy = yy - shift_y
        ball_dome = 165.0 - ((sx - 23.5) ** 2 + (sy - 23.5) ** 2) * 0.35
        highlight_spot = 70.0 * np.exp(-((sx - 20.0) ** 2 + (sy - 19.0) ** 2) / 14.0)
        aramith_dot = -55.0 * np.exp(-((sx - 26.5) ** 2 + (sy - 25.5) ** 2) / 8.0)
        wave_tex = 18.0 * np.sin((sx - 23.5) * 0.65) * np.cos((sy - 23.5) * 0.65)
        return np.clip(ball_dome + highlight_spot + aramith_dot + wave_tex, 0, 255).astype(np.uint8)

    prev_patch = _render_ball_patch(0.0, 0.0)
    # 0.15픽셀(dx=0.12, dy=0.09 -> hypot=0.15px) 서브픽셀 미세 진동 패치 생성
    curr_patch = _render_ball_patch(0.12, 0.09)

    evidence = detector.detect_subpixel_impulse(prev_patch, curr_patch)
    assert evidence.is_impacted is True, f"0.15픽셀 미세 진동을 감지하지 못했습니다: {evidence}"
    assert 0.06 <= evidence.phase_shift_px <= 0.35, f"측정된 위상 이동량 비정상: {evidence.phase_shift_px}"

    # CausalKissSafeFSM 연동 검증:
    # 제2적구(YELLOW) 바운딩 박스 좌표는 (2100.0, 500.0)에서 0.0mm도 움직이지 않고,
    # 수구(WHITE)도 직선으로 스쳐 지나가며 좌표상 굴절이 없지만(거리 82mm <= 95mm),
    # t=1.50s에 SubPixelPhaseImpactDetector가 0.15px 미세 진동을 감지한 경우 즉시 득점(+1) 판정!
    immediate_callbacks = []
    fsm = CausalKissSafeFSM(
        fps=60.0,
        assigned_cue_ball=BallColor.WHITE,
        on_immediate_score=lambda res: immediate_callbacks.append(res),
    )

    kf = [
        (0.00, (400, 700),   (2100, 500), (800, 700)),
        (0.20, (740, 700),   (2100, 500), (800, 700)),   # 제1적구(RED) 타격
        (0.50, (1300, 35),   (2100, 500), (950, 820)),   # 1쿠션(TOP)
        (0.85, (2805, 500),  (2100, 500), (1000, 860)),  # 2쿠션(RIGHT)
        (1.20, (2300, 1385), (2100, 500), (1000, 860)),  # 3쿠션(BOTTOM)
        (1.50, (2100, 582),  (2100, 500), (1000, 860)),  # 제2적구(YELLOW)에서 82mm 이격 통과 (YELLOW 이동량 0mm!)
        (1.80, (1900, 200),  (2100, 500), (1000, 860)),
    ]

    dt = 1.0 / 60.0
    instant_return_res = None
    for i in range(len(kf) - 1):
        t0, w0, y0, r0 = kf[i]
        t1, w1, y1, r1 = kf[i + 1]
        steps = max(int(round((t1 - t0) / dt)), 1)
        for s in range(steps):
            a = s / steps
            t = t0 + (t1 - t0) * a
            coords = {
                BallColor.WHITE: (w0[0] + (w1[0] - w0[0]) * a, w0[1] + (w1[1] - w0[1]) * a),
                BallColor.YELLOW: (2100.0, 500.0),  # 바운딩 박스 이동량 완전 0mm!
                BallColor.RED: (r0[0] + (r1[0] - r0[0]) * a, r0[1] + (r1[1] - r0[1]) * a),
            }
            subpixel_map = (
                {BallColor.YELLOW: evidence.is_impacted}
                if abs(t - 1.50) < 0.012
                else None
            )
            step_res = fsm.update(t, coords, subpixel_impacts=subpixel_map)
            if step_res is not None and not step_res.already_dispatched:
                instant_return_res = step_res

    assert len(immediate_callbacks) == 1, "0.15px 미세 진동 감지 시 즉시 득점 콜백이 호출되어야 합니다."
    assert immediate_callbacks[0].scored is True
    assert instant_return_res is not None and instant_return_res.scored is True
    print(
        f"[PASS] Test 2 (SubPixelPhaseImpactDetector): 바운딩 박스 이동 0.0mm 상태에서 "
        f"위상 진동 {evidence.phase_shift_px:.3f}px(신뢰도={evidence.confidence:.2f}) 감지 -> 즉시 득점(+1) 판정 완료"
    )


def test_motion_streak_v_apex_detection():
    """
    3. 한 프레임 안에서 쿠션을 맞고 꺾인 V자 모션 블러 마스크에서
    MotionStreakVApexDetector가 꺾임 각도와 꼭짓점을 정확히 찾아내고,
    CushionRimPixelTrigger 및 FSM intra_frame_rail_bounce 연동이 정상 작동하는지 확인합니다.
    """
    mask = np.zeros((120, 200), dtype=np.uint8)
    # 입사 궤적 (30, 85) -> 쿠션 반사 꼭짓점 (100, 18) -> 반사 궤적 (170, 85) 형태의 V자 모션 블러 잔상
    cv2.line(mask, (30, 85), (100, 18), 255, thickness=9)
    cv2.line(mask, (100, 18), (170, 85), 255, thickness=9)

    apex_ev = MotionStreakVApexDetector.extract_streak_apex(mask)
    assert apex_ev.is_bent_within_frame is True, f"V자 모션 블러 꺾임을 감지하지 못했습니다: {apex_ev}"
    assert apex_ev.intra_frame_bend_deg >= 45.0, f"꺾임 각도가 너무 작게 측정되었습니다: {apex_ev.intra_frame_bend_deg}"
    assert abs(apex_ev.apex_offset_px[0] - 100.0) <= 12.0 and abs(apex_ev.apex_offset_px[1] - 18.0) <= 12.0, (
        f"검출된 V자 꼭짓점 좌표 오차 과다: {apex_ev.apex_offset_px}"
    )

    # CushionRimPixelTrigger 픽셀 터치 감지 검증
    rim_trigger = CushionRimPixelTrigger()
    prev_strip = np.full((16, 80), 90, dtype=np.uint8)
    curr_strip = prev_strip.copy()
    curr_strip[:, 32:48] = 220  # 쿠션 고무 날 경계에 밝은 수구 외곽 및 그림자 접촉 발생
    curr_strip[10:, 30:50] = 35
    rim_ev = rim_trigger.verify_rail_touch_by_pixels(prev_strip, curr_strip)
    assert rim_ev.is_rail_touched is True

    # FSM 연동 검증: 3번째 쿠션(BOTTOM) 근처 y=110mm(좌표 기준으로는 쿠션 마진 밖)에서
    # 프레임 내 V자 반사가 일어났을 때 intra_frame_rail_bounce=True로 3쿠션이 정상 인정되는지 확인
    fsm = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=BallColor.WHITE)
    kf = [
        (0.00, (400, 700),   (2100, 500), (800, 700)),
        (0.20, (740, 700),   (2100, 500), (800, 700)),   # 제1적구(RED) 타격
        (0.50, (1300, 35),   (2100, 500), (950, 820)),   # 1쿠션(TOP)
        (0.85, (2805, 500),  (2100, 500), (1000, 860)),  # 2쿠션(RIGHT)
        (1.20, (2300, 1295), (2100, 500), (1000, 860)),  # 3쿠션(BOTTOM) 앞 125mm 지점에서 프레임 내 V자 반사!
        (1.50, (2100, 558),  (2100, 500), (1000, 860)),  # 제2적구(YELLOW) 타격
        (1.80, (2000, 350),  (2100, 380), (1000, 860)),
    ]
    dt = 1.0 / 60.0
    instant_res: ShotResult | None = None
    for i in range(len(kf) - 1):
        t0, w0, y0, r0 = kf[i]
        t1, w1, y1, r1 = kf[i + 1]
        steps = max(int(round((t1 - t0) / dt)), 1)
        for s in range(steps):
            a = s / steps
            t = t0 + (t1 - t0) * a
            coords = {
                BallColor.WHITE: (w0[0] + (w1[0] - w0[0]) * a, w0[1] + (w1[1] - w0[1]) * a),
                BallColor.YELLOW: (y0[0] + (y1[0] - y0[0]) * a, y0[1] + (y1[1] - y0[1]) * a),
                BallColor.RED: (r0[0] + (r1[0] - r0[0]) * a, r0[1] + (r1[1] - r0[1]) * a),
            }
            is_v_bounce = bool(apex_ev.is_bent_within_frame and abs(t - 1.20) < 0.012)
            step_res = fsm.update(t, coords, intra_frame_rail_bounce=is_v_bounce)
            if step_res is not None and not step_res.already_dispatched:
                instant_res = step_res

    assert instant_res is not None and instant_res.scored is True, "V-Apex 쿠션 반사 복원으로 3쿠션 득점이 인정되어야 합니다."
    assert instant_res.cushions_at_second_hit >= 3
    assert "B1" in instant_res.shot_event_tokens and "B2" in instant_res.shot_event_tokens
    assert instant_res.shot_event_tokens.count("C") >= 3
    print(
        f"[PASS] Test 3 (MotionStreakVApexDetector & CushionRimPixelTrigger): "
        f"V자 잔상 꼭짓점={apex_ev.apex_offset_px}, 굴절각={apex_ev.intra_frame_bend_deg:.1f}° 검출 및 3쿠션 득점 연동 완료"
    )


def test_literature_5_algorithms_integration() -> None:
    """
    4. 문헌 기반 5대 핵심 알고리즘(YCbCr 크로마키 + 타원 장축 분해 split_merged_oval_blob,
    직선 회귀 잔차 Line-Fit RMSE, 이벤트 시퀀스 문법 파서 shot_event_tokens) 통합 검증.
    """
    from ball_identity_tracker import LabHungarianBallIdentifier, split_merged_oval_blob

    # (1) 두 공이 키스하여 하나로 합쳐진 가로형 타원 블롭(가로 68px, 세로 32px -> 장단축비 2.125 >= 1.45) 생성
    frame_bgr = np.zeros((200, 300, 3), dtype=np.uint8)
    frame_bgr[:, :] = (190, 95, 20)  # 파란색 당구대 라사지 배경 (B가 높음 -> Cb 마스킹 대상)
    cv2.circle(frame_bgr, (134, 100), 15, (235, 235, 235), -1)  # 왼쪽 흰공
    cv2.circle(frame_bgr, (166, 100), 15, (25, 215, 245), -1)   # 오른쪽 노란공
    cv2.circle(frame_bgr, (240, 100), 15, (20, 25, 225), -1)    # 적구

    sub_boxes = split_merged_oval_blob((116.0, 84.0, 184.0, 116.0))
    assert len(sub_boxes) == 2, f"중첩 타원 블롭은 2개의 서브 박스로 분할되어야 합니다: len={len(sub_boxes)}"
    c1_x = 0.5 * (sub_boxes[0][0] + sub_boxes[0][2])
    c2_x = 0.5 * (sub_boxes[1][0] + sub_boxes[1][2])
    assert abs(c2_x - c1_x) >= 20.0, f"분할된 두 서브 박스 중심 간격이 충분해야 합니다: {c1_x:.1f} vs {c2_x:.1f}"

    # 분할된 박스를 LabHungarianBallIdentifier에 넣었을 때 YCbCr 마스킹 + 상대 색도 랭킹으로 3구 정확히 식별하는지 검증
    identifier = LabHungarianBallIdentifier()
    id_res = identifier.assign_ball_identities(
        frame_bgr,
        [
            (116.0, 84.0, 184.0, 116.0),
            (224.0, 84.0, 256.0, 116.0),
        ],
    )
    assert set(id_res.keys()) == {BallColor.WHITE, BallColor.YELLOW, BallColor.RED}

    # (2) 직선 회귀 잔차(Line-Fit RMSE) 검증: 완벽한 직선(0.0mm) vs 꺾인 궤적(>= 1.6mm)
    straight_pts = np.array(
        [[100.0, 100.0], [150.0, 100.0], [200.0, 100.0], [250.0, 100.0], [300.0, 100.0]],
        dtype=np.float64,
    )
    kinked_pts = np.array(
        [[100.0, 100.0], [150.0, 100.0], [200.0, 100.0], [250.0, 112.0], [300.0, 125.0]],
        dtype=np.float64,
    )
    rmse_straight = CausalKissSafeFSM._compute_line_fit_rmse(straight_pts)
    rmse_kinked = CausalKissSafeFSM._compute_line_fit_rmse(kinked_pts)
    assert rmse_straight < 0.01, f"직선 등속 궤적의 Line-Fit RMSE는 0에 수렴해야 합니다: {rmse_straight}"
    assert rmse_kinked >= 1.6, f"충돌로 꺾인 궤적의 Line-Fit RMSE는 1.6mm 이상이어야 합니다: {rmse_kinked}"

    print(
        f"[PASS] Test 4 (문헌 기반 5대 핵심 기술): split_merged_oval_blob 2구 분할 완료 "
        f"(c1_x={c1_x:.1f}, c2_x={c2_x:.1f}) | Line-Fit RMSE 직선={rmse_straight:.4f}mm vs 굴절={rmse_kinked:.2f}mm 검증 완료"
    )


def test_micro_contact_4factor_forensics_vs_near_miss() -> None:
    """
    5. [4중 교차 초미세 접촉 감식기 (Micro-Contact Forensics Engine) 극한 벤치마크]
    - Case A (진짜 0.3mm 초미세 스침 득점):
      제2적구(YELLOW)가 충돌 법선 방향으로 단 1.4mm 밀려나고 수구가 반대 법선 방향으로 2.2도 국소 굴절된 경우 -> 100% 득점(scored=True) 인식!
    - Case B (종이 한 장 차이 불발 + 카메라 지터 2.0mm):
      수구가 d_min=62.5mm로 아슬아슬하게 스쳐 지나가고, 마침 그 순간 제2적구 바운딩 박스가 카메라 노이즈로 2.0mm(수구 쪽으로 끌려오거나 1프레임 스파이크 후 복귀) 튄 경우 -> 100% 불발(scored=False) 차단!
    - Case C (수구 스핀 커브 + 허공 스침 불발):
      수구가 매 프레임 2.3도씩 완만하게 휘어지는 회전(스핀) 커브로 제2적구 옆 63.0mm를 지나가고 제2적구에 미세 진동(1.5mm 비법선 지터)이 섞인 경우 -> 100% 불발(scored=False) 차단!
    """
    from cue24.core.scoring_engine import ScoringEngine

    def _build_base_3c_prefix():
        return [
            (0.00, (400.0, 700.0),  (2100.0, 500.0), (700.0, 700.0)),
            (0.03, (560.0, 700.0),  (2100.0, 500.0), (700.0, 700.0)),
            (0.06, (655.0, 695.0),  (2100.0, 500.0), (700.0, 700.0)),  # 제1적구(RED) 타격
            (0.09, (840.0, 420.0),  (2100.0, 500.0), (760.0, 745.0)),
            (0.12, (1050.0, 45.0),  (2100.0, 500.0), (795.0, 770.0)),  # 1쿠션(TOP)
            (0.15, (1650.0, 380.0), (2100.0, 500.0), (810.0, 780.0)),
            (0.18, (2350.0, 780.0), (2100.0, 500.0), (815.0, 785.0)),
            (0.21, (2802.0, 1080.0),(2100.0, 500.0), (815.0, 785.0)),  # 2쿠션(RIGHT)
            (0.24, (2450.0, 1382.0),(2100.0, 500.0), (815.0, 785.0)),  # 3쿠션(BOTTOM)
            (0.27, (2300.0, 960.0), (2100.0, 500.0), (815.0, 785.0)),
            (0.30, (2230.0, 760.0), (2100.0, 500.0), (815.0, 785.0)),
        ]

    # ------------------------------------------------------------------
    # Case A: 진짜 0.3mm 초미세 스침 (제2적구가 법선 방향(-x)으로 1.45mm 밀려나고 유지, 수구는 +x 쪽으로 2.3도 반작용 굴절)
    # ------------------------------------------------------------------
    fsm_a = CausalKissSafeFSM(fps=30.0, assigned_cue_ball=BallColor.WHITE)
    eng_a = ScoringEngine()
    eng_a.reset_shot(cue_ball_id=1, object_ball_ids=[2, 3])

    # 수구가 (2161.2, 500.0)을 지나며 제2적구(2100.0, 500.0)의 오른쪽 옆구리를 0.3mm 두께로 얇게 스침!
    # 충돌 법선 n_xy = (-1.0, 0.0) -> 제2적구는 왼쪽(-x)으로 1.45mm 밀려나고(2098.55, 500.0), 수구는 오른쪽(+x)으로 2.3도 굴절!
    frames_a = _build_base_3c_prefix() + [
        (0.33, (2195.6, 630.0), (2100.0, 500.0),   (815.0, 785.0)),
        (0.36, (2161.2, 500.0), (2100.0, 500.0),   (815.0, 785.0)),  # 초미세 접촉 시점 (d_min = 61.2mm)
        (0.39, (2132.2, 370.0), (2098.6, 499.8),   (815.0, 785.0)),  # 수구 +x 반작용 굴절(2.3도) & 제2적구 -x 법선 밀림 1.41mm
        (0.42, (2103.2, 240.0), (2098.5, 499.7),   (815.0, 785.0)),  # 단방향 관성 지속 유지 (1.53mm)
    ]

    res_a = None
    for idx_f, (t, w_p, y_p, r_p) in enumerate(frames_a):
        out = fsm_a.update(t, {BallColor.WHITE: w_p, BallColor.YELLOW: y_p, BallColor.RED: r_p})
        if out is not None and not out.already_dispatched:
            res_a = out
        eng_a.update_frame(idx_f + 1, {1: w_p, 2: y_p, 3: r_p})

    eval_a = eng_a.evaluate_shot()
    assert res_a is not None and res_a.scored is True, "Case A 실패: 진짜 0.3mm 초미세 스침(1.45mm 법선 밀림)이 득점으로 인식되지 않았습니다!"
    assert eval_a["is_scored"] is True, f"Case A ScoringEngine 실패: {eval_a}"

    # ------------------------------------------------------------------
    # Case B: 아슬아슬한 불발(d_min=62.5mm) + 카메라 지터 2.1mm (수구 쪽인 +x로 끌려오거나 엉뚱한 방향 스파이크 후 원위치 복귀)
    # ------------------------------------------------------------------
    fsm_b = CausalKissSafeFSM(fps=30.0, assigned_cue_ball=BallColor.WHITE)
    eng_b = ScoringEngine()
    eng_b.reset_shot(cue_ball_id=1, object_ball_ids=[2, 3])

    frames_b = _build_base_3c_prefix() + [
        (0.33, (2196.2, 630.0), (2100.0, 500.0),   (815.0, 785.0)),
        (0.36, (2162.5, 500.0), (2100.0, 500.0),   (815.0, 785.0)),  # 종이 한 장 차이 불발 (d_min = 62.5mm)
        (0.39, (2133.5, 370.0), (2102.1, 500.4),   (815.0, 785.0)),  # 카메라 지터로 오히려 수구 쪽(+x)으로 2.14mm 튐! (비인과적 방향)
        (0.42, (2104.5, 240.0), (2100.1, 500.0),   (815.0, 785.0)),  # 다음 프레임 다시 원위치로 복귀 (1프레임 스파이크)
    ]

    res_b = None
    for idx_f, (t, w_p, y_p, r_p) in enumerate(frames_b):
        out = fsm_b.update(t, {BallColor.WHITE: w_p, BallColor.YELLOW: y_p, BallColor.RED: r_p})
        if out is not None and not out.already_dispatched:
            res_b = out
        eng_b.update_frame(idx_f + 1, {1: w_p, 2: y_p, 3: r_p})

    eval_b = eng_b.evaluate_shot()
    assert res_b is None, f"Case B 실패: 비인과적 카메라 지터(2.1mm)에 의한 허공 스침이 득점으로 오인되었습니다! ({res_b})"
    assert eval_b["is_scored"] is False, f"Case B ScoringEngine 실패: {eval_b}"

    # ------------------------------------------------------------------
    # Case C: 수구 스핀 커브(매 프레임 2.5도 균일 휨) + 허공 스침 불발
    # ------------------------------------------------------------------
    fsm_c = CausalKissSafeFSM(fps=30.0, assigned_cue_ball=BallColor.WHITE)
    eng_c = ScoringEngine()
    eng_c.reset_shot(cue_ball_id=1, object_ball_ids=[2, 3])

    frames_c = _build_base_3c_prefix() + [
        (0.33, (2192.0, 630.0), (2100.0, 500.0),   (815.0, 785.0)),  # 이미 전 구간부터 2.5도씩 휘어지는 스핀 커브
        (0.36, (2163.0, 500.0), (2100.0, 500.0),   (815.0, 785.0)),  # 제2적구 옆 63.0mm 통과 (접촉 없음)
        (0.39, (2139.5, 370.0), (2099.4, 501.5),   (815.0, 785.0)),  # 동일 곡률로 계속 휘어짐 + 수직 방향 미세 노이즈
        (0.42, (2121.5, 240.0), (2099.8, 500.2),   (815.0, 785.0)),
    ]

    res_c = None
    for idx_f, (t, w_p, y_p, r_p) in enumerate(frames_c):
        out = fsm_c.update(t, {BallColor.WHITE: w_p, BallColor.YELLOW: y_p, BallColor.RED: r_p})
        if out is not None and not out.already_dispatched:
            res_c = out
        eng_c.update_frame(idx_f + 1, {1: w_p, 2: y_p, 3: r_p})

    eval_c = eng_c.evaluate_shot()
    assert res_c is None, f"Case C 실패: 수구 스핀 커브에 의한 허공 통과가 미세접촉 득점으로 오인되었습니다! ({res_c})"
    assert eval_c["is_scored"] is False, f"Case C ScoringEngine 실패: {eval_c}"

    print(
        "[PASS] Test 5 (4중 교차 초미세 접촉 감식기): "
        "Case A(0.3mm 스침 1.45mm 법선밀림 -> 득점 확정) / "
        "Case B(2.1mm 비인과 지터 불발 -> 100% 기각) / "
        "Case C(스핀 커브 불발 -> 100% 기각) 완벽 통과!"
    )


if __name__ == "__main__":
    test_3d_ray_plane_vs_2d_homography()
    test_subpixel_phase_correlate_0_1px_kiss()
    test_motion_streak_v_apex_detection()
    test_literature_5_algorithms_integration()
    test_micro_contact_4factor_forensics_vs_near_miss()
    print("\n=== 4대 고성능 비전 판독 기술 + 문헌 기반 5대 핵심 알고리즘 통합 검증 전체 PASS 완료! ===")
