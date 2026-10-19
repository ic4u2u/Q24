"""
60fps Vision-Only 3쿠션 판정 코어 엔진 4대 핵심 시나리오 자동 검증 스크립트
파일 경로: three_cushion_vision/tests/test_scenarios.py

검증 시나리오:
1. 시나리오 1 (일반 3쿠션 득점): 수구 출발 -> 제1목적구 접촉 -> 3쿠션 회전 -> 제2목적구 접촉 -> 정지 (scored == True, is_bank_shot == False)
2. 시나리오 2 (뱅크샷/가락 득점): 수구 출발 -> 빈 쿠션 3회 먼저 접촉 -> 제1목적구 접촉 -> 제2목적구 접촉 -> 정지 (scored == True, is_bank_shot == True)
3. 시나리오 3 (2쿠션 후 조기 키스 노카운트 잠금): 수구 출발 -> 제1목적구 접촉 -> 2쿠션 접촉 -> 제2목적구 조기 접촉(무효 잠금) -> 3번째 쿠션 접촉 -> 제2목적구 재접촉 -> 정지 (scored == False, fail_reason == "EARLY_SECOND_BALL_KISS")
4. 시나리오 4 (근접 통과 Near-Miss vs 0.5mm 미세 스침 구분):
   - Case 4A: 제2목적구 옆을 중심 거리 72mm(표면 간격 10.5mm)로 굴절 없이 직진 통과 -> 노카운트 판정 (scored == False)
   - Case 4B: 프레임 사이에서 제2목적구 옆을 지나가며 수구 궤적이 2.5도 꺾이고 목적구가 5mm 밀림 -> 미세 스침 감지 성공으로 득점 판정 (scored == True)
"""

import math
import unittest
from typing import Dict, List, Optional, Tuple
import numpy as np

from three_cushion_vision.config import BallColor, FailReason, ShotState
from three_cushion_vision.homography import Point2D, TableHomography
from three_cushion_vision.scoreboard_bridge import ScoreboardBridge
from three_cushion_vision.scoring_fsm import ShotResult, ThreeCushionScoringFSM
from three_cushion_vision.tracker_adapter import OCSortTrackerAdapter


def _interpolate_waypoints_60fps(
    waypoints: List[Point2D], frames_per_segment: int = 8
) -> List[Point2D]:
    """
    주어진 경유점(waypoints) 리스트 사이를 60fps 등간격 프레임 좌표로 선형 보간합니다.
    """
    if len(waypoints) < 2:
        return list(waypoints)
    pts: List[Point2D] = [waypoints[0]]
    for idx in range(len(waypoints) - 1):
        x0, y0 = waypoints[idx]
        x1, y1 = waypoints[idx + 1]
        for step in range(1, frames_per_segment + 1):
            alpha = step / float(frames_per_segment)
            pts.append((x0 + (x1 - x0) * alpha, y0 + (y1 - y0) * alpha))
    return pts


def _run_60fps_simulation(
    fsm: ThreeCushionScoringFSM,
    cue_path: List[Point2D],
    red_path: List[Point2D],
    yellow_path: List[Point2D],
    stop_hold_frames: int = 24,
    enable_ema: bool = False,
) -> Optional[ShotResult]:
    """
    60fps(dt = 1/60s) 프레임 스트림을 OCSortTrackerAdapter -> ThreeCushionScoringFSM에 순차 주입하고
    모든 공이 완전 정지(0.3초 이상)한 뒤 반환되는 ShotResult를 획득합니다.
    """
    adapter = OCSortTrackerAdapter()
    dt = 1.0 / 60.0
    total_active_frames = max(len(cue_path), len(red_path), len(yellow_path))

    # 마지막 정지 위치 유지 프레임(stop_hold_frames = 24프레임 = 0.40초 >= 0.30초) 추가
    total_frames = total_active_frames + stop_hold_frames
    shot_result: Optional[ShotResult] = None

    for f_idx in range(total_frames):
        t_sec = f_idx * dt
        c_pt = cue_path[min(f_idx, len(cue_path) - 1)]
        r_pt = red_path[min(f_idx, len(red_path) - 1)]
        y_pt = yellow_path[min(f_idx, len(yellow_path) - 1)]

        detections = [
            {"color": BallColor.WHITE, "x_mm": c_pt[0], "y_mm": c_pt[1], "track_id": 1},
            {"color": BallColor.RED, "x_mm": r_pt[0], "y_mm": r_pt[1], "track_id": 2},
            {"color": BallColor.YELLOW, "x_mm": y_pt[0], "y_mm": y_pt[1], "track_id": 3},
        ]
        tracked_balls = adapter.update_from_detections(
            detections=detections,
            timestamp=t_sec,
            enable_ema=enable_ema,
        )
        res = fsm.update(balls=tracked_balls, timestamp=t_sec)
        if res is not None:
            shot_result = res
            break

    return shot_result


class TestThreeCushionVisionScenarios(unittest.TestCase):
    """60fps Vision-Only 3쿠션 백엔드 코어 엔진 4대 핵심 시나리오 테스트"""

    def test_module_a_homography_and_module_b_hsv_histogram(self) -> None:
        """
        [기본 모듈 검증]
        - Module A: 4점 및 다점(6점) 호모그래피 변환 + 대각선 카메라 시차 보정(cy + h * 0.12)
        - Module B: 원형 ROI HSV 색공간 히스토그램 기반 WHITE / YELLOW / RED 분류
        """
        homography = TableHomography(camera_tilt_ratio=0.12)
        # 바운딩 박스 (970, 470, 1030, 530) -> 중심 (1000, 500), 높이 h=60 -> cy_corrected = 500 + 60*0.12 = 507.2
        x_mm, y_mm = homography.bbox_to_table_mm((970.0, 470.0, 1030.0, 530.0))
        self.assertAlmostEqual(x_mm, 1000.0, places=2)
        self.assertAlmostEqual(y_mm, 507.2, places=2)

        # 다점(6점: 4코너 + 상하 중앙 다이아몬드 포인트) cv2.findHomography 지원 검증
        pts_6 = [
            (0.0, 0.0),
            (1420.0, 0.0),
            (2840.0, 0.0),
            (2840.0, 1420.0),
            (1420.0, 1420.0),
            (0.0, 1420.0),
        ]
        homography_multi = TableHomography(src_points_px=pts_6, dst_points_mm=pts_6)
        mx, my = homography_multi.pixel_to_table_mm(1420.0, 710.0)
        self.assertAlmostEqual(mx, 1420.0, places=1)
        self.assertAlmostEqual(my, 710.0, places=1)

        # 합성 BGR 프레임에서 WHITE, YELLOW, RED 원형 ROI HSV 히스토그램 분류 검증
        canvas = np.zeros((200, 600, 3), dtype=np.uint8)
        canvas[:, :] = (180, 90, 20)  # 파란색 라사지 배경 (BGR)
        import cv2

        cv2.circle(canvas, (100, 100), 30, (245, 245, 245), -1)  # WHITE
        cv2.circle(canvas, (300, 100), 30, (20, 230, 245), -1)   # YELLOW (BGR)
        cv2.circle(canvas, (500, 100), 30, (25, 25, 235), -1)    # RED (BGR)

        c_white, _ = OCSortTrackerAdapter.classify_roi_hsv_histogram(
            canvas, (70.0, 70.0, 130.0, 130.0)
        )
        c_yellow, _ = OCSortTrackerAdapter.classify_roi_hsv_histogram(
            canvas, (270.0, 70.0, 330.0, 130.0)
        )
        c_red, _ = OCSortTrackerAdapter.classify_roi_hsv_histogram(
            canvas, (470.0, 70.0, 530.0, 130.0)
        )
        self.assertEqual(c_white, BallColor.WHITE)
        self.assertEqual(c_yellow, BallColor.YELLOW)
        self.assertEqual(c_red, BallColor.RED)

    def test_scenario_1_regular_3cushion_score(self) -> None:
        """
        [시나리오 1: 일반 3쿠션 득점]
        수구(WHITE) 출발 -> 제1목적구(RED) 접촉 -> 3쿠션 회전(TOP -> RIGHT -> BOTTOM) -> 제2목적구(YELLOW) 접촉 -> 정지
        기대 결과: scored == True, is_bank_shot == False
        """
        fsm = ThreeCushionScoringFSM(initial_cue_ball=BallColor.WHITE)
        bridge = ScoreboardBridge()

        cue_waypoints = [
            (400.0, 700.0),    # 출발
            (800.0, 500.0),    # 제1목적구(RED: 800, 450) 정면/측면 충돌 (거리 50mm <= 63.5mm)
            (1500.0, 32.0),    # 1쿠션: TOP 레일 (y=32.0 <= 44.75mm)
            (2808.0, 650.0),   # 2쿠션: RIGHT 레일 (x=2808.0 >= 2795.25mm)
            (1900.0, 1388.0),  # 3쿠션: BOTTOM 레일 (y=1388.0 >= 1375.25mm)
            (1200.0, 900.0),   # 제2목적구(YELLOW: 1200, 850) 접촉 (거리 50mm <= 63.5mm)
            (1100.0, 820.0),   # 감속 후 정지
        ]
        cue_path = _interpolate_waypoints_60fps(cue_waypoints, frames_per_segment=8)
        red_path = [(800.0, 450.0)] * 9 + [(820.0, 380.0)] * (len(cue_path) - 9)
        yellow_path = [(1200.0, 850.0)] * 41 + [(1160.0, 800.0)] * (len(cue_path) - 41)

        result = _run_60fps_simulation(fsm, cue_path, red_path, yellow_path)
        self.assertIsNotNone(result)
        assert result is not None

        published = bridge.publish_shot_result(result)
        print(
            f"\n[Scenario 1 - Regular 3C] scored={result.scored}, is_bank_shot={result.is_bank_shot}, "
            f"cushions_before_2nd={result.cushions_before_second_ball}, total_cushions={result.total_cushions}"
        )

        self.assertTrue(result.scored)
        self.assertFalse(result.is_bank_shot)
        self.assertGreaterEqual(result.cushions_before_second_ball, 3)
        self.assertEqual(result.fail_reason, FailReason.NONE.value)
        self.assertEqual(fsm.assigned_cue_ball, BallColor.WHITE)  # 득점 성공 시 수구 유지
        self.assertEqual(published["event_type"], "SCORE_UPDATED")

    def test_scenario_2_bank_shot_score(self) -> None:
        """
        [시나리오 2: 뱅크샷(가락) 득점]
        수구(WHITE) 출발 -> 빈 쿠션 3회 먼저 접촉(TOP -> RIGHT -> BOTTOM) -> 제1목적구(RED) 접촉 -> 제2목적구(YELLOW) 접촉 -> 정지
        기대 결과: scored == True, is_bank_shot == True
        """
        fsm = ThreeCushionScoringFSM(initial_cue_ball=BallColor.WHITE)

        cue_waypoints = [
            (500.0, 700.0),    # 출발
            (1200.0, 32.0),    # 선행 1쿠션: TOP 레일
            (2808.0, 600.0),   # 선행 2쿠션: RIGHT 레일
            (2000.0, 1388.0),  # 선행 3쿠션: BOTTOM 레일
            (1400.0, 800.0),   # 제1목적구(RED: 1400, 750) 접촉 (거리 50mm <= 63.5mm)
            (950.0, 500.0),    # 제2목적구(YELLOW: 950, 450) 접촉 (거리 50mm <= 63.5mm)
            (880.0, 420.0),    # 정지
        ]
        cue_path = _interpolate_waypoints_60fps(cue_waypoints, frames_per_segment=8)
        red_path = [(1400.0, 750.0)] * 33 + [(1350.0, 700.0)] * (len(cue_path) - 33)
        yellow_path = [(950.0, 450.0)] * 41 + [(910.0, 400.0)] * (len(cue_path) - 41)

        result = _run_60fps_simulation(fsm, cue_path, red_path, yellow_path)
        self.assertIsNotNone(result)
        assert result is not None

        print(
            f"[Scenario 2 - Bank Shot] scored={result.scored}, is_bank_shot={result.is_bank_shot}, "
            f"cushions_before_2nd={result.cushions_before_second_ball}, total_cushions={result.total_cushions}"
        )

        self.assertTrue(result.scored)
        self.assertTrue(result.is_bank_shot)
        self.assertGreaterEqual(result.cushions_before_second_ball, 3)
        self.assertEqual(result.fail_reason, FailReason.NONE.value)

    def test_scenario_3_early_second_ball_kiss_no_score_locked(self) -> None:
        """
        [시나리오 3: 2쿠션 후 조기 키스 노카운트 잠금]
        수구(WHITE) 출발 -> 제1목적구(RED) 접촉 -> 2쿠션 접촉(TOP -> RIGHT) ->
        제2목적구(YELLOW) 조기 접촉(NO_SCORE_LOCKED 잠금!) -> 3번째 쿠션 접촉(BOTTOM) ->
        제2목적구(YELLOW) 재접촉 -> 정지
        기대 결과: scored == False, fail_reason == "EARLY_SECOND_BALL_KISS", 다음 이닝 수구 YELLOW로 교대
        """
        fsm = ThreeCushionScoringFSM(initial_cue_ball=BallColor.WHITE)

        cue_waypoints = [
            (400.0, 700.0),    # 출발
            (800.0, 500.0),    # 제1목적구(RED: 800, 450) 접촉
            (1500.0, 32.0),    # 1쿠션: TOP 레일
            (2808.0, 600.0),   # 2쿠션: RIGHT 레일
            (2100.0, 1000.0),  # 2쿠션 상태에서 제2목적구(YELLOW: 2100, 950) 조기 접촉! (NO_SCORE_LOCKED)
            (1800.0, 1388.0),  # 3번째 쿠션: BOTTOM 레일
            (1600.0, 950.0),   # 이동한 제2목적구(YELLOW: 1600, 900) 재접촉!
            (1500.0, 900.0),   # 정지
        ]
        cue_path = _interpolate_waypoints_60fps(cue_waypoints, frames_per_segment=8)
        red_path = [(800.0, 450.0)] * len(cue_path)
        yellow_path = (
            [(2100.0, 950.0)] * 33
            + [(1600.0, 900.0)] * (len(cue_path) - 33)
        )

        result = _run_60fps_simulation(fsm, cue_path, red_path, yellow_path)
        self.assertIsNotNone(result)
        assert result is not None

        print(
            f"[Scenario 3 - Early 2nd Ball Kiss Lock] scored={result.scored}, "
            f"fail_reason={result.fail_reason}, cushions_before_2nd={result.cushions_before_second_ball}, "
            f"total_cushions={result.total_cushions}, next_cue={fsm.assigned_cue_ball.value}"
        )

        self.assertFalse(result.scored)
        self.assertEqual(result.fail_reason, FailReason.EARLY_SECOND_BALL_KISS.value)
        self.assertEqual(result.cushions_before_second_ball, 2)
        self.assertGreaterEqual(result.total_cushions, 3)
        self.assertEqual(fsm.assigned_cue_ball, BallColor.YELLOW)  # 공타 시 상대방 수구로 자동 교대

    def test_scenario_4_near_miss_vs_micro_kiss_dual_gate(self) -> None:
        """
        [시나리오 4: 근접 통과(Near-Miss) vs 0.5mm 미세 스침(Micro-Kiss) 구분]
        - Case 4A: 수구가 3쿠션 후 제2목적구 옆을 중심 거리 72mm(표면 간격 10.5mm, 그레이존 내)로
                   굴절(0.0도) 및 목적구 밀림(0.0mm) 없이 직진 통과 -> 노카운트 판정 (scored == False)
        - Case 4B: 수구가 3쿠션 후 프레임 사이에서 제2목적구 옆을 중심 거리 ~70mm로 스쳐 지나가며
                   수구 궤적이 2.5도 꺾이고 목적구가 5.0mm 밀림 -> 미세 스침 감지 성공으로 득점 판정 (scored == True)
        """
        # ------------------------------------------------------------------
        # Case 4A: 중심 거리 72mm 근접 비접촉 직진 통과 (Near-Miss)
        # ------------------------------------------------------------------
        fsm_4a = ThreeCushionScoringFSM(initial_cue_ball=BallColor.WHITE)

        # 3쿠션 후 y=700.0 직선 라인을 따라 x=1800 -> x=900으로 일직선 주행 (굴절 0.0도)
        # 제2목적구(YELLOW)는 (1350.0, 628.0)에 정지 -> 수구와의 최소 중심 거리 d_min = 72.0mm (표면 간격 10.5mm)
        cue_waypoints_4a = [
            (400.0, 700.0),    # 출발
            (800.0, 500.0),    # 제1목적구(RED: 800, 450) 접촉
            (1500.0, 32.0),    # 1쿠션: TOP
            (2808.0, 600.0),   # 2쿠션: RIGHT
            (2000.0, 1388.0),  # 3쿠션: BOTTOM
            (1800.0, 700.0),   # 직선 진입 시작점
            (1350.0, 700.0),   # YELLOW(1350, 628) 옆을 중심거리 정확히 72.0mm로 무굴절 통과
            (900.0, 700.0),    # 굴절 없이 직진 유지 후 정지
        ]
        cue_path_4a = _interpolate_waypoints_60fps(cue_waypoints_4a, frames_per_segment=8)
        red_path_4a = [(800.0, 450.0)] * len(cue_path_4a)
        yellow_path_4a = [(1350.0, 628.0)] * len(cue_path_4a)  # 전혀 밀리지 않고 정지 유지

        result_4a = _run_60fps_simulation(fsm_4a, cue_path_4a, red_path_4a, yellow_path_4a)
        self.assertIsNotNone(result_4a)
        assert result_4a is not None

        print(
            f"[Scenario 4A - 72mm Near-Miss] scored={result_4a.scored}, "
            f"fail_reason={result_4a.fail_reason}, total_cushions={result_4a.total_cushions}"
        )
        self.assertFalse(result_4a.scored)
        self.assertEqual(result_4a.fail_reason, FailReason.MISSING_SECOND_BALL.value)

        # ------------------------------------------------------------------
        # Case 4B: 0.5mm 미세 스침 (수구 궤적 2.5도 굴절 + 제2목적구 5mm 미세 밀림)
        # ------------------------------------------------------------------
        fsm_4b = ThreeCushionScoringFSM(initial_cue_ball=BallColor.WHITE)

        # (1350, 700)에서 제2목적구(1350, 630 -> 중심거리 70mm 그레이존)를 스친 직후
        # 수구가 서쪽으로 450mm 이동할 때 y방향으로 +20.0mm 꺾임 -> atan2(20, 450) = 약 2.54도 굴절!
        # 동시에 제2목적구(YELLOW)가 (1350, 630) -> (1350, 625)로 5.0mm 밀려남!
        cue_waypoints_4b = [
            (400.0, 700.0),     # 출발
            (800.0, 500.0),     # 제1목적구(RED: 800, 450) 접촉
            (1500.0, 32.0),     # 1쿠션: TOP
            (2808.0, 600.0),    # 2쿠션: RIGHT
            (2000.0, 1388.0),   # 3쿠션: BOTTOM
            (1800.0, 700.0),    # 직선 진입 시작점
            (1350.0, 700.0),    # YELLOW(1350, 630) 최근접 스침 지점 (d_min = 70.0mm)
            (900.0, 720.0),     # 스침 직후 2.54도 궤적 굴절 발생!
        ]
        cue_path_4b = _interpolate_waypoints_60fps(cue_waypoints_4b, frames_per_segment=8)
        red_path_4b = [(800.0, 450.0)] * len(cue_path_4b)

        # 최근접 프레임(index 48) 직후 제2목적구가 (1350, 630)에서 (1350, 625)로 5.0mm 미세 밀림
        yellow_path_4b: List[Point2D] = []
        for idx in range(len(cue_path_4b)):
            if idx <= 48:
                yellow_path_4b.append((1350.0, 630.0))
            elif idx == 49:
                yellow_path_4b.append((1350.0, 628.0))
            elif idx == 50:
                yellow_path_4b.append((1350.0, 626.0))
            else:
                yellow_path_4b.append((1350.0, 625.0))  # 총 5.0mm 밀림

        result_4b = _run_60fps_simulation(fsm_4b, cue_path_4b, red_path_4b, yellow_path_4b)
        self.assertIsNotNone(result_4b)
        assert result_4b is not None

        print(
            f"[Scenario 4B - 0.5mm Micro-Kiss (2.5 deg + 5mm)] scored={result_4b.scored}, "
            f"cushions_before_2nd={result_4b.cushions_before_second_ball}, events={result_4b.events[-1]}"
        )
        self.assertTrue(result_4b.scored)
        self.assertFalse(result_4b.is_bank_shot)
        self.assertGreaterEqual(result_4b.cushions_before_second_ball, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
