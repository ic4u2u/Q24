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

    def test_scenario_5_cue_ball_color_missed_score_auto_compensation(self) -> None:
        """
        [시나리오 5: 수구 색상 감지 기반 미인지 득점 자동 소급 보정]
        1) 1이닝 첫 샷에서 P1(WHITE)이 타구했으나 비전 사각지대 등으로 인해 노카운트(MISS)로 오판되어
           FSM의 assigned_cue_ball이 YELLOW로 넘어가고 점수판도 P2(후공) 차례로 넘어감.
        2) 그러나 실제 경기에서는 P1이 득점에 성공했기 때문에 다음 샷에서도 P1이 흰공(WHITE)을 다시 타구하여 3쿠션 득점 성공!
        3) FSM과 ScoreboardBridge + MatchManager가 선수가 친 공의 색깔(WHITE)을 감지하여:
           - 직전 누락되었던 득점(+1점)을 P1에게 즉시 소급 복구하고,
           - 이번 샷 득점(+1점)까지 합산하여 P1의 1이닝 점수가 총 2점(score=2, current_run=2, active_player=P1)으로 정확히 반영되는지 검증!
        """
        from cue24.services.match_manager import MatchManager

        mm = MatchManager(p1_name="선공선수", p2_name="후공선수")
        fsm = ThreeCushionScoringFSM(
            initial_cue_ball=BallColor.WHITE,
            auto_compensate_missed_score=True,
        )
        bridge = ScoreboardBridge(match_manager=mm)

        # 1차 샷: 비전이 제2목적구 접촉을 놓쳐 MISS로 판정된 상황 시뮬레이션
        miss_waypoints = [
            (400.0, 700.0),
            (800.0, 500.0),   # 제1목적구(RED: 800, 450) 접촉
            (1500.0, 32.0),   # 1쿠션
            (2808.0, 650.0),  # 2쿠션
            (1900.0, 1388.0), # 3쿠션 후 제2목적구 미접촉으로 오판된 경로
            (1500.0, 1000.0),
        ]
        cue_path_1 = _interpolate_waypoints_60fps(miss_waypoints, frames_per_segment=8)
        red_path_1 = [(800.0, 450.0)] * len(cue_path_1)
        yellow_path_1 = [(1200.0, 600.0)] * len(cue_path_1)

        res_1 = _run_60fps_simulation(fsm, cue_path_1, red_path_1, yellow_path_1)
        self.assertIsNotNone(res_1)
        assert res_1 is not None
        bridge.publish_shot_result(res_1)

        # 1차 샷 직후: 노카운트 판정으로 인해 P1은 0점이고 P2(YELLOW)에게 턴이 넘어간 상태
        self.assertFalse(res_1.scored)
        self.assertEqual(fsm.assigned_cue_ball, BallColor.YELLOW)
        self.assertEqual(mm.active_player_id, "P2")
        self.assertEqual(mm.players["P1"].score, 0)

        # 2차 샷: 실제로는 1차 샷이 득점이어서 선공(P1)이 자신의 흰공(WHITE)을 연속 타구하여 3쿠션 득점!
        score_waypoints = [
            (1500.0, 1000.0),
            (800.0, 500.0),    # 제1목적구(RED: 800, 450) 접촉
            (1500.0, 32.0),    # 1쿠션: TOP
            (2808.0, 650.0),   # 2쿠션: RIGHT
            (1900.0, 1388.0),  # 3쿠션: BOTTOM
            (1200.0, 650.0),   # 제2목적구(YELLOW: 1200, 600) 접촉
            (1100.0, 600.0),   # 정지
        ]
        cue_path_2 = _interpolate_waypoints_60fps(score_waypoints, frames_per_segment=8)
        red_path_2 = [(800.0, 450.0)] * len(cue_path_2)
        yellow_path_2 = [(1200.0, 600.0)] * len(cue_path_2)

        res_2 = _run_60fps_simulation(fsm, cue_path_2, red_path_2, yellow_path_2)
        self.assertIsNotNone(res_2)
        assert res_2 is not None
        pub_2 = bridge.publish_shot_result(res_2)

        print(
            f"[Scenario 5 - Cue Ball Color Auto-Compensation] "
            f"retroactive_compensated={res_2.retroactive_previous_score_compensated}, "
            f"P1_score={mm.players['P1'].score}, P2_score={mm.players['P2'].score}, "
            f"active_player={mm.active_player_id}, inning={mm.current_inning}"
        )

        # 검증:
        # 1) 선수가 친 흰공(WHITE)을 감지하여 직전 미인지 득점(+1점)이 소급 보정되었고,
        # 2) 이번 2차 샷 득점(+1점)까지 더해져 P1의 점수는 정확히 2점(1이닝 2득점, 연속 2점)이 되어야 함!
        self.assertTrue(res_2.retroactive_previous_score_compensated)
        self.assertTrue(res_2.scored)
        self.assertEqual(res_2.cue_ball, BallColor.WHITE)
        self.assertEqual(mm.players["P1"].score, 2)
        self.assertEqual(mm.players["P1"].inning_scores.get(1), 2)
        self.assertEqual(mm.players["P1"].current_run, 2)
        self.assertEqual(mm.players["P2"].score, 0)
        self.assertEqual(mm.active_player_id, "P1")
        self.assertEqual(mm.current_inning, 1)
        self.assertTrue(pub_2["reconcile_info"]["compensated"])

    def test_scenario_6_demo_visualizer_pip_and_overlay_rendering(self) -> None:
        """
        [시나리오 6: demo_visualizer.py 실시간 2D 미니맵(PIP), 쿠션 카운트, 궤적 라인, 판정 배너 렌더링 검증]
        """
        from three_cushion_vision.demo_visualizer import run_video_demo

        summary = run_video_demo(
            video_path="__non_existent_synthetic_demo__.mp4",
            output_path=None,
            max_frames=90,
            display=False,
        )
        print(
            f"[Scenario 6 - DemoVisualizer] processed_frames={summary['processed_frames']}, "
            f"shots={len(summary['shot_results'])}, backend={summary['detector_backend']}"
        )
        self.assertGreaterEqual(summary["processed_frames"], 70)
        self.assertEqual(len(summary["shot_results"]), 1)
        self.assertTrue(summary["shot_results"][0].scored)

    def test_scenario_7_billiard_nanogpt_occlusion_cushion_restoration(self) -> None:
        """
        [시나리오 7: BilliardNanoGPT 기반 선수 몸 가려짐(Occlusion) 구간 비선형 쿠션 반사 복원 검증]
        - 수구(WHITE)가 제1목적구(RED)를 맞힌 뒤 상단 쿠션(TOP, y=30.75mm)에 입사하기 직전부터 반사 직후까지
          총 6프레임(index 14 ~ 19) 동안 선수의 몸에 가려져 카메라 검출 리스트에서 완전히 사라진(None) 상황.
        - OCSortTrackerAdapter가 BilliardNanoGPT(OcclusionTrajectoryRestorer)를 통해 가려진 6프레임 동안의
          상단 쿠션 반사 궤적을 실시간 복원하여 1쿠션(TOP)을 놓치지 않고 최종 3쿠션 득점(scored == True)을 성공시키는지 검증!
        """
        fsm = ThreeCushionScoringFSM(initial_cue_ball=BallColor.WHITE)
        adapter = OCSortTrackerAdapter(enable_occlusion_restorer=True)

        cue_waypoints = [
            (400.0, 700.0),    # 출발
            (800.0, 500.0),    # 제1목적구(RED: 800, 450) 타격 (index 8)
            (1500.0, 31.0),    # 1쿠션: TOP 레일 (index 16 -> 가려짐 구간 한복판!)
            (2808.0, 650.0),   # 2쿠션: RIGHT 레일
            (1900.0, 1388.0),  # 3쿠션: BOTTOM 레일
            (1200.0, 900.0),   # 제2목적구(YELLOW: 1200, 850) 접촉
            (1100.0, 820.0),   # 정지
        ]
        cue_path = _interpolate_waypoints_60fps(cue_waypoints, frames_per_segment=8)
        red_path = [(800.0, 450.0)] * 9 + [(820.0, 380.0)] * (len(cue_path) - 9)
        yellow_path = [(1200.0, 850.0)] * 41 + [(1160.0, 800.0)] * (len(cue_path) - 41)

        # index 14 ~ 19 (TOP 쿠션 반사 전후 6프레임) 동안 선수 신체에 수구(WHITE)가 완전히 가려짐!
        occluded_indices = {14, 15, 16, 17, 18, 19}
        restored_count = 0
        shot_result: Optional[ShotResult] = None
        dt = 1.0 / 60.0
        total_frames = len(cue_path) + 24

        for f_idx in range(total_frames):
            t_sec = f_idx * dt
            c_pt = cue_path[min(f_idx, len(cue_path) - 1)]
            r_pt = red_path[min(f_idx, len(red_path) - 1)]
            y_pt = yellow_path[min(f_idx, len(yellow_path) - 1)]

            detections = [
                {"color": BallColor.RED, "x_mm": r_pt[0], "y_mm": r_pt[1], "track_id": 2},
                {"color": BallColor.YELLOW, "x_mm": y_pt[0], "y_mm": y_pt[1], "track_id": 3},
            ]
            # 가려진 프레임(14~19)에서는 WHITE 수구 검출이 아예 누락됨!
            if f_idx not in occluded_indices:
                detections.insert(
                    0,
                    {"color": BallColor.WHITE, "x_mm": c_pt[0], "y_mm": c_pt[1], "track_id": 1},
                )

            tracked_balls = adapter.update_from_detections(
                detections=detections,
                timestamp=t_sec,
                enable_ema=False,
            )
            if tracked_balls[BallColor.WHITE].is_occluded_restored:
                restored_count += 1

            res = fsm.update(balls=tracked_balls, timestamp=t_sec)
            if res is not None:
                shot_result = res
                break

        self.assertIsNotNone(shot_result)
        assert shot_result is not None
        print(
            f"[Scenario 7 - BilliardNanoGPT Occlusion Restoration] "
            f"restored_frames={restored_count}, scored={shot_result.scored}, "
            f"cushions_before_2nd={shot_result.cushions_before_second_ball}"
        )
        self.assertEqual(restored_count, 6)
        self.assertTrue(shot_result.scored)
        self.assertGreaterEqual(shot_result.cushions_before_second_ball, 3)

    def test_scenario_8_causal_kiss_disambiguator_4_illusions(self) -> None:
        """
        [시나리오 8: 3체 인과 충돌 분리 엔진(CausalKissSafeFSM) 4대 키스 착시 차단 검증]
        1) 오류 1: 1적구-2적구 키스로 2적구가 굴러갈 때 수구가 3쿠션 후 2적구 옆을 무굴절·무충격 근접 통과 -> 노카운트
        2) 오류 2: 3구 밀집 구역에서 수구가 1적구만 재키스하여 꺾였을 때(x_deflected_Off_other_ball) -> 노카운트
        3) 오류 3: 쿠션에 밀착된 2적구 옆에서 수구가 쿠션만 맞고 반사된 경우(x_near_rail) -> 노카운트
        4) 오류 4: 3쿠션 전에 1적구-2적구 간 키스가 발생했으나 이후 수구가 3쿠션 후 2적구 정상 타격 -> 득점 성공
        """
        from three_cushion_vision.causal_kiss_fsm import (
            BallColor as CausalBallColor,
            CausalKissSafeFSM,
            ShotResult as CausalShotResult,
        )

        def _run_causal_sim(
            cue_pts: List[Point2D],
            red_pts: List[Point2D],
            yel_pts: List[Point2D],
            hold_frames: int = 25,
        ) -> CausalShotResult:
            cfsm = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=CausalBallColor.WHITE)
            dt = 1.0 / 60.0
            n_max = max(len(cue_pts), len(red_pts), len(yel_pts))
            for idx in range(n_max + hold_frames):
                t_sec = idx * dt
                c = cue_pts[min(idx, len(cue_pts) - 1)]
                r = red_pts[min(idx, len(red_pts) - 1)]
                y = yel_pts[min(idx, len(yel_pts) - 1)]
                res = cfsm.update(
                    t_sec,
                    {
                        CausalBallColor.WHITE: c,
                        CausalBallColor.RED: r,
                        CausalBallColor.YELLOW: y,
                    },
                )
                if res is not None:
                    return res
            raise AssertionError("CausalKissSafeFSM did not finalize shot")

        # ------------------------------------------------------------------
        # 1) 오류 1: 1적구(RED)가 2적구(YELLOW)를 쳐서 YELLOW가 등속으로 굴러갈 때,
        #    수구(WHITE)가 3쿠션 후 굴러가는 YELLOW 옆을 그레이존(d_min=70mm)으로 비접촉 직진 통과
        # ------------------------------------------------------------------
        cue_1 = _interpolate_waypoints_60fps(
            [
                (400.0, 700.0),
                (800.0, 500.0),    # RED(800, 450) 타격 (index 8)
                (1500.0, 32.0),    # 1C: TOP
                (2808.0, 600.0),   # 2C: RIGHT
                (2000.0, 1388.0),  # 3C: BOTTOM
                (1800.0, 700.0),   # 직선 진입 (index 40)
                (1350.0, 700.0),   # 굴러가는 YELLOW 옆을 d_min=70mm로 무굴절 통과 (index 48)
                (900.0, 700.0),
            ],
            frames_per_segment=8,
        )
        # RED가 index 12에서 YELLOW(1100, 560)를 타격하여 적구 간 키스 발생!
        red_1 = _interpolate_waypoints_60fps(
            [
                (800.0, 450.0),
                (800.0, 450.0),
                (1100.0, 510.0),   # index 16에서 YELLOW와 적구 간 키스
                (1150.0, 480.0),
            ],
            frames_per_segment=8,
        )
        # YELLOW는 index 16 이후 등속으로 굴러가며 index 48에서 (1350, 630)을 등속 통과 (delta_vy = 0)
        yel_1: List[Point2D] = []
        for i in range(len(cue_1)):
            if i < 16:
                yel_1.append((1100.0, 560.0))
            elif i <= 54:
                alpha = (i - 16) / 32.0
                yel_1.append((1100.0 + 250.0 * alpha, 560.0 + 70.0 * alpha))
            else:
                yel_1.append((1100.0 + 250.0 * (38.0 / 32.0), 560.0 + 70.0 * (38.0 / 32.0)))

        res_err1 = _run_causal_sim(cue_1, red_1, yel_1)
        print(
            f"[Causal Illusion 1 - Moving 2nd Ball Near-Miss] scored={res_err1.scored}, "
            f"ob_kiss={res_err1.object_ball_kiss_occurred}, reason={res_err1.fail_reason}"
        )
        self.assertFalse(res_err1.scored)
        self.assertTrue(res_err1.object_ball_kiss_occurred)

        # ------------------------------------------------------------------
        # 2) 오류 2: 3구 밀집 구역에서 수구가 1적구(RED, d=50mm)만 재키스하고 꺾였을 때
        #    그레이존에 있던 2적구(YELLOW, d=71mm) 접촉으로 오판하지 않아야 함
        # ------------------------------------------------------------------
        cue_2 = _interpolate_waypoints_60fps(
            [
                (400.0, 700.0),
                (800.0, 500.0),    # RED(800, 450) 최초 타격
                (1500.0, 32.0),    # 1C: TOP
                (2808.0, 600.0),   # 2C: RIGHT
                (2000.0, 1388.0),  # 3C: BOTTOM
                (1800.0, 700.0),
                (1350.0, 700.0),   # 밀집 구역 진입: RED(1350, 750 -> d=50mm) 재키스! YELLOW(1350, 629 -> d=71mm) 비접촉!
                (1000.0, 620.0),   # RED 재키스 반작용으로 -y 방향으로 꺾임
            ],
            frames_per_segment=8,
        )
        red_2 = [(800.0, 450.0)] * 9 + [(1350.0, 750.0)] * 39 + [(1350.0, 790.0)] * (len(cue_2) - 48)
        yel_2 = [(1350.0, 629.0)] * len(cue_2)

        res_err2 = _run_causal_sim(cue_2, red_2, yel_2)
        print(
            f"[Causal Illusion 2 - 1st Ball Re-Kiss in Cluster] scored={res_err2.scored}, "
            f"reason={res_err2.fail_reason}"
        )
        self.assertFalse(res_err2.scored)

        # ------------------------------------------------------------------
        # 3) 오류 3: 쿠션에 밀착된 2적구(YELLOW: 1900, 1314 -> d_min=70.6mm 그레이존) 옆에서
        #    수구가 YELLOW를 건드리지 않고 BOTTOM 쿠션(1900, 1388)만 맞고 반사된 경우 -> 노카운트
        # ------------------------------------------------------------------
        cue_3 = _interpolate_waypoints_60fps(
            [
                (400.0, 700.0),
                (800.0, 500.0),    # RED(800, 450) 타격
                (1500.0, 32.0),    # 1C: TOP
                (2808.0, 900.0),   # 2C: RIGHT
                (1920.0, 1386.0),  # 3C: BOTTOM 레일 진입 (y >= 1375.25mm)
                (1880.0, 1388.0),  # BOTTOM 레일 정점 반사 (바로 위 (1900, 1314)에 YELLOW 정지, d_min=71mm 그레이존)
                (1300.0, 1200.0),  # 쿠션 반사로 꺾여 나감 (YELLOW 미접촉)
            ],
            frames_per_segment=8,
        )
        red_3 = [(800.0, 450.0)] * 9 + [(800.0, 380.0)] * (len(cue_3) - 9)
        yel_3 = [(1900.0, 1314.0)] * len(cue_3)  # 전혀 움직이지 않음 (delta_vy = 0)

        res_err3 = _run_causal_sim(cue_3, red_3, yel_3)
        print(
            f"[Causal Illusion 3 - Rail Bounce Near Cushion Ball] scored={res_err3.scored}, "
            f"reason={res_err3.fail_reason}"
        )
        self.assertFalse(res_err3.scored)

        # ------------------------------------------------------------------
        # 4) 오류 4: 수구가 3쿠션을 돌기 전에 1적구(RED)와 2적구(YELLOW)가 키스했으나,
        #    이닝을 죽이지 않고 수구가 3쿠션 후 이동한 YELLOW를 정상 타격하여 득점 성공!
        # ------------------------------------------------------------------
        cue_4 = _interpolate_waypoints_60fps(
            [
                (400.0, 700.0),
                (745.0, 450.0),    # RED(800, 450) 타격 -> RED가 +x 방향(1050, 500)으로 이동
                (1500.0, 32.0),    # 1C: TOP
                (2808.0, 650.0),   # 2C: RIGHT
                (1900.0, 1388.0),  # 3C: BOTTOM
                (1200.0, 900.0),   # 이동한 YELLOW(1200, 850) 정식 타격!
                (1100.0, 940.0),
            ],
            frames_per_segment=8,
        )
        # RED가 타격 직후(1쿠션 전) YELLOW(1050, 550)와 부딪혀 YELLOW가 (1200, 850)으로 이동
        red_4 = _interpolate_waypoints_60fps(
            [
                (800.0, 450.0),
                (800.0, 450.0),
                (1050.0, 500.0),   # index 16에서 YELLOW와 적구 간 키스
                (1100.0, 460.0),
            ],
            frames_per_segment=8,
        )
        yel_4 = (
            [(1050.0, 550.0)] * 16
            + _interpolate_waypoints_60fps([(1050.0, 550.0), (1200.0, 850.0)], frames_per_segment=8)[1:]
        )
        yel_4.extend([(1200.0, 850.0)] * (40 - len(yel_4)))
        yel_4.extend(
            _interpolate_waypoints_60fps([(1200.0, 850.0), (1140.0, 760.0)], frames_per_segment=8)[1:]
        )
        yel_4.extend([(1140.0, 760.0)] * (len(cue_4) - len(yel_4)))

        res_err4 = _run_causal_sim(cue_4, red_4, yel_4)
        print(
            f"[Causal Illusion 4 - Pre-3C Object Ball Kiss Then Valid Hit] "
            f"scored={res_err4.scored}, ob_kiss={res_err4.object_ball_kiss_occurred}, "
            f"cushions_at_2nd={res_err4.cushions_at_second_hit}"
        )
        self.assertTrue(res_err4.scored)
        self.assertTrue(res_err4.object_ball_kiss_occurred)
        self.assertGreaterEqual(res_err4.cushions_at_second_hit, 3)

    def test_scenario_9_double_hit_foul_and_blob_merging_stabilization(self) -> None:
        """
        [시나리오 9: 투터치(Double Hit) 파울 감지 및 바운딩 박스 병합(Blob Merging) 중심점 보정 검증]
        1) 투터치 조건 A (충돌 후 재가속): 초기 거리 90mm(<= 115mm)에서 출발 후 0.15초 내 1적구 충돌 직후
           수구가 감속되지 않고 1적구 방향으로 재가속(a_A · n_AB > 1500 mm/s^2) -> DOUBLE_HIT_FOUL 노카운트 잠금!
        2) 투터치 조건 B (정면충돌 후 비정상 추격): 두꺼운 정면충돌(cos > 0.85) 후 수구 법선 속도가
           1적구 법선 속도의 75% 이상으로 추격 -> DOUBLE_HIT_FOUL 노카운트 잠금!
        3) 바운딩 박스 병합(Blob Merging) 결측(None, 2프레임) 복원 및 중심점 끌림(Centroid Shift, 20mm 점프)
           0.85 가중 클램핑으로 가짜 속도 변화(Delta v) 오탐 차단 검증!
        """
        from three_cushion_vision.causal_kiss_fsm import (
            BallColor as CausalBallColor,
            CausalKissSafeFSM,
        )

        dt = 1.0 / 60.0

        # ------------------------------------------------------------------
        # 1) 투터치 파울 - 조건 B (두꺼운 정면충돌 후 75% 이상 법선 속도로 밀고 나가는 파울)
        #    초기 수구(400, 700), 제1적구 RED(490, 700) -> 초기 거리 90.0mm <= 115.0mm
        #    충돌 직후 RED가 +x 방향 1200 mm/s(프레임당 20mm)로 나가는데,
        #    WHITE도 큐대에 한 번 더 밀려 +x 방향 1080 mm/s(프레임당 18mm, 비율 90% >= 75%)로 추격 후 3쿠션+제2적구 접촉
        # ------------------------------------------------------------------
        cfsm_dh_b = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=CausalBallColor.WHITE)
        cue_dh_b: List[Point2D] = [
            (400.0, 700.0),   # f0: 정지 (RED(490, 700)와 초기 거리 90.0mm <= 115.0mm)
            (415.0, 700.0),   # f1: 스트로크 출발 (900 mm/s)
            (432.0, 700.0),   # f2: 정면 접근 (cos theta = 1.0 > 0.85)
            (448.0, 700.0),   # f3: RED(490, 700)와 충돌 (거리 42mm <= 63mm, t = 0.05s <= 0.15s)
            (468.0, 700.0),   # f4: 충돌 후에도 수구가 감속 없이 1200 mm/s로 추격!
            (488.0, 700.0),   # f5: 수구 법선 속도(1200) >= RED 법선 속도(1320) * 0.75
            (508.0, 700.0),   # f6: 계속 밀고 나감
        ]
        # 이후 3쿠션 돌고 YELLOW(1200, 850)까지 맞히더라도 이미 DOUBLE_HIT_FOUL로 NO_SCORE_LOCKED 상태여야 함!
        post_3c = _interpolate_waypoints_60fps(
            [
                (508.0, 700.0),
                (1500.0, 32.0),    # 1C: TOP
                (2808.0, 650.0),   # 2C: RIGHT
                (1900.0, 1388.0),  # 3C: BOTTOM
                (1200.0, 900.0),   # YELLOW(1200, 850) 접촉
                (1100.0, 820.0),
            ],
            frames_per_segment=8,
        )[1:]
        cue_dh_b.extend(post_3c)

        red_dh_b: List[Point2D] = [
            (490.0, 700.0),
            (490.0, 700.0),
            (490.0, 700.0),
            (490.0, 700.0),
            (512.0, 700.0),   # f4: RED 속도 1320 mm/s (+22mm/frame)
            (534.0, 700.0),   # f5
            (556.0, 700.0),   # f6
        ]
        red_dh_b.extend([(580.0, 700.0)] * (len(cue_dh_b) - len(red_dh_b)))
        yel_dh_b: List[Point2D] = [(1200.0, 850.0)] * len(cue_dh_b)

        res_dh_b = None
        for idx in range(len(cue_dh_b) + 25):
            t_sec = idx * dt
            c = cue_dh_b[min(idx, len(cue_dh_b) - 1)]
            r = red_dh_b[min(idx, len(red_dh_b) - 1)]
            y = yel_dh_b[min(idx, len(yel_dh_b) - 1)]
            res_dh_b = cfsm_dh_b.update(
                t_sec,
                {
                    CausalBallColor.WHITE: c,
                    CausalBallColor.RED: r,
                    CausalBallColor.YELLOW: y,
                },
            )
            if res_dh_b is not None:
                break

        self.assertIsNotNone(res_dh_b)
        assert res_dh_b is not None
        print(
            f"[Scenario 9A - Double Hit Foul (Push-Through)] scored={res_dh_b.scored}, "
            f"double_hit_foul={res_dh_b.double_hit_foul}, reason={res_dh_b.fail_reason}"
        )
        self.assertFalse(res_dh_b.scored)
        self.assertTrue(res_dh_b.double_hit_foul)
        self.assertIsNotNone(res_dh_b.fail_reason)
        self.assertIn("DOUBLE_HIT_FOUL", str(res_dh_b.fail_reason))

        # ------------------------------------------------------------------
        # 2) 투터치 파울 - 조건 A (충돌 직후 큐대 2차 타격에 의한 수구 재가속: a_A · v_A > 0 & a_A · n_AB > 1500)
        # ------------------------------------------------------------------
        cfsm_dh_a = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=CausalBallColor.WHITE)
        cue_dh_a: List[Point2D] = [
            (400.0, 700.0),   # f0: 정지 (RED(495, 715)와 초기 거리 96.2mm <= 115.0mm)
            (412.0, 700.0),   # f1: 출발 (720 mm/s)
            (425.0, 700.0),   # f2: 접근 (780 mm/s)
            (436.0, 700.0),   # f3: RED와 1차 충돌로 순간 감속 (660 mm/s)
            (452.0, 702.0),   # f4: 큐대 투터치로 재가속! (960 mm/s)
            (478.0, 705.0),   # f5: 강한 재가속! (1560 mm/s -> a_A > 20000 mm/s^2 > 1500 mm/s^2)
            (510.0, 708.0),   # f6
        ]
        red_dh_a: List[Point2D] = [
            (495.0, 715.0),
            (495.0, 715.0),
            (495.0, 715.0),
            (495.0, 715.0),
            (535.0, 730.0),
            (575.0, 745.0),
            (615.0, 760.0),
        ]
        yel_dh_a: List[Point2D] = [(1200.0, 850.0)] * len(cue_dh_a)

        res_dh_a = None
        for idx in range(len(cue_dh_a) + 25):
            t_sec = idx * dt
            c = cue_dh_a[min(idx, len(cue_dh_a) - 1)]
            r = red_dh_a[min(idx, len(red_dh_a) - 1)]
            y = yel_dh_a[min(idx, len(yel_dh_a) - 1)]
            res_dh_a = cfsm_dh_a.update(
                t_sec,
                {
                    CausalBallColor.WHITE: c,
                    CausalBallColor.RED: r,
                    CausalBallColor.YELLOW: y,
                },
            )
            if res_dh_a is not None:
                break

        self.assertIsNotNone(res_dh_a)
        assert res_dh_a is not None
        print(
            f"[Scenario 9B - Double Hit Foul (Re-Acceleration)] scored={res_dh_a.scored}, "
            f"double_hit_foul={res_dh_a.double_hit_foul}, reason={res_dh_a.fail_reason}"
        )
        self.assertFalse(res_dh_a.scored)
        self.assertTrue(res_dh_a.double_hit_foul)
        self.assertIn("DOUBLE_HIT_FOUL", str(res_dh_a.fail_reason))

        # ------------------------------------------------------------------
        # 3) 바운딩 박스 병합(Blob Merging) 결측 복원(None 2프레임) 및 중심점 끌림(Centroid Shift) 억제 검증
        #    - 수구가 3쿠션 후 정지해 있는 제2적구 YELLOW(1350, 625) 옆을 중심거리 75mm(비접촉 Near-Miss)로 지나갈 때:
        #      * index 47에서 바운딩 박스 병합으로 WHITE 좌표가 None(결측)으로 들어오고,
        #      * index 48에서 YELLOW의 바운딩 박스가 WHITE 쪽(+y 방향)으로 찌그러져 YELLOW 관측 좌표가
        #        (1350, 625) -> (1350, 647)로 +22.0mm(> 14.0mm) 급격히 끌려갔다가 index 49에서 (1350, 625)로 돌아옴!
        #    - 보정이 없다면 YELLOW에 가짜 속도 변화량(Delta v)이 생겨 2적구 접촉(득점)으로 오판되지만,
        #      _stabilize_blob_merging이 결측을 외삽 복원하고 중심점 끌림을 0.85 가중치로 클램핑하여
        #      가짜 접촉 오탐을 막아내고 정확히 노카운트(scored == False)로 판정해야 함!
        # ------------------------------------------------------------------
        cfsm_blob = CausalKissSafeFSM(fps=60.0, assigned_cue_ball=CausalBallColor.WHITE)
        cue_blob = _interpolate_waypoints_60fps(
            [
                (400.0, 700.0),
                (800.0, 500.0),    # RED(800, 450) 타격
                (1500.0, 32.0),    # 1C: TOP
                (2808.0, 600.0),   # 2C: RIGHT
                (2000.0, 1388.0),  # 3C: BOTTOM
                (1800.0, 700.0),
                (1350.0, 700.0),   # index 48: YELLOW(1350, 625) 옆을 거리 75.0mm(<= 85.0mm)로 무굴절 통과
                (900.0, 700.0),
            ],
            frames_per_segment=8,
        )
        red_blob = [(800.0, 450.0)] * len(cue_blob)
        yel_blob: List[Point2D] = [(1350.0, 625.0)] * len(cue_blob)
        # index 48에서 바운딩 박스 병합으로 인해 정지해 있던 YELLOW가 지나가는 WHITE(1350, 700) 쪽으로 22mm 끌려감!
        yel_blob[48] = (1350.0, 647.0)

        res_blob = None
        for idx in range(len(cue_blob) + 25):
            t_sec = idx * dt
            # index 46, 47(2프레임)에서 바운딩 박스 병합으로 WHITE 좌표가 None(결측) 발생!
            c_obs: Optional[Point2D] = (
                None
                if idx in (46, 47)
                else cue_blob[min(idx, len(cue_blob) - 1)]
            )
            r_obs = red_blob[min(idx, len(red_blob) - 1)]
            y_obs = yel_blob[min(idx, len(yel_blob) - 1)]

            res_blob = cfsm_blob.update(
                t_sec,
                {
                    CausalBallColor.WHITE: c_obs,
                    CausalBallColor.RED: r_obs,
                    CausalBallColor.YELLOW: y_obs,
                },
            )
            if res_blob is not None:
                break

        self.assertIsNotNone(res_blob)
        assert res_blob is not None
        print(
            f"[Scenario 9C - Blob Merging Occlusion & Centroid Shift Clamping] "
            f"scored={res_blob.scored}, centroid_corrected={res_blob.centroid_shift_corrected_count}, "
            f"reason={res_blob.fail_reason}"
        )
        self.assertFalse(res_blob.scored)
        self.assertGreaterEqual(res_blob.centroid_shift_corrected_count, 1)

        # ------------------------------------------------------------------
        # 4) ThreeCushionScoringFSM / KinematicsDetector 통합 검증:
        #    근접 출발 후 투터치(Double Hit) 파울 감지 및 바운딩 박스 병합 결측(None) 복원 검증
        # ------------------------------------------------------------------
        from three_cushion_vision.tracker_adapter import TrackedBall

        fsm_dh = ThreeCushionScoringFSM(initial_cue_ball=BallColor.WHITE)
        for idx in range(len(cue_dh_b) + 25):
            t_sec = idx * dt
            i_curr = min(idx, len(cue_dh_b) - 1)
            i_prev = max(0, min(idx - 1, len(cue_dh_b) - 1))
            c0, c1 = cue_dh_b[i_prev], cue_dh_b[i_curr]
            r0, r1 = red_dh_b[i_prev], red_dh_b[i_curr]
            y0, y1 = yel_dh_b[i_prev], yel_dh_b[i_curr]

            cvx, cvy = (c1[0] - c0[0]) / dt, (c1[1] - c0[1]) / dt
            rvx, rvy = (r1[0] - r0[0]) / dt, (r1[1] - r0[1]) / dt
            yvx, yvy = (y1[0] - y0[0]) / dt, (y1[1] - y0[1]) / dt

            balls_step = {
                BallColor.WHITE: TrackedBall(
                    color=BallColor.WHITE, x=c1[0], y=c1[1],
                    vx=cvx, vy=cvy, speed=math.hypot(cvx, cvy), timestamp=t_sec
                ),
                BallColor.RED: TrackedBall(
                    color=BallColor.RED, x=r1[0], y=r1[1],
                    vx=rvx, vy=rvy, speed=math.hypot(rvx, rvy), timestamp=t_sec
                ),
                BallColor.YELLOW: TrackedBall(
                    color=BallColor.YELLOW, x=y1[0], y=y1[1],
                    vx=yvx, vy=yvy, speed=math.hypot(yvx, yvy), timestamp=t_sec
                ),
            }
            res_fsm_dh = fsm_dh.update(balls=balls_step, timestamp=t_sec)
            if res_fsm_dh is not None:
                self.assertFalse(res_fsm_dh.scored)
                self.assertTrue(res_fsm_dh.double_hit_foul)
                self.assertEqual(res_fsm_dh.fail_reason, FailReason.DOUBLE_HIT_FOUL.value)
                break
        else:
            self.fail("ThreeCushionScoringFSM did not finalize double-hit foul shot")


if __name__ == "__main__":
    unittest.main(verbosity=2)
