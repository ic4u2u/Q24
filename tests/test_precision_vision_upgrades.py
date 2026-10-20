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
    print(
        f"[PASS] Test 3 (MotionStreakVApexDetector & CushionRimPixelTrigger): "
        f"V자 잔상 꼭짓점={apex_ev.apex_offset_px}, 굴절각={apex_ev.intra_frame_bend_deg:.1f}° 검출 및 3쿠션 득점 연동 완료"
    )


if __name__ == "__main__":
    test_3d_ray_plane_vs_2d_homography()
    test_subpixel_phase_correlate_0_1px_kiss()
    test_motion_streak_v_apex_detection()
    print("\n=== 4대 고성능 비전 판독 기술(Sub-Pixel & 3D Ray-Plane) 통합 검증 전체 PASS 완료! ===")
