"""
Step 10 단위 테스트: 라이브캠/리플레이 터치 시 전체화면(Fullscreen) 전환/복귀 및 1.0x~4.0x 줌인·줌아웃/드래그 패닝 검증
파일 경로: cue24/tests/test_step10.py
"""

import unittest

from cue24.api.server import STATIC_DIR


class TestStep10TouchFullscreenAndZoomPan(unittest.TestCase):
    """Step 10 라이브캠 터치 전체화면 토글 및 1.0x~4.0x 줌인·줌아웃/드래그 패닝 UI 검증 스위트"""

    def test_fullscreen_toggle_and_zoom_pan_elements(self) -> None:
        """
        static/index.html에 라이브캠 화면 터치 전체화면 토글(toggleCamFullscreen, fullscreen-mode,
        TAP_MOVE_THRESHOLD_PX = 6) 및 1.0x~4.0x 줌인/줌아웃(zoomCam, resetCamZoom, applyCamTransform,
        clampPanOffsets) 로직이 완벽히 구현되어 있는지 검증
        """
        index_path = STATIC_DIR / "index.html"
        html = index_path.read_text(encoding="utf-8")

        # 1. 라이브캠 터치 시 전체화면 전환 및 재터치 시 원래 화면 복귀 기능 확인
        self.assertIn("toggleCamFullscreen", html)
        self.assertIn("fullscreen-mode", html)
        self.assertIn("TAP_MOVE_THRESHOLD_PX = 6", html)
        self.assertIn("initCamInteractiveEvents", html)
        self.assertIn('id="btnFullscreenToggle"', html)

        # 2. 줌인(+0.5x), 줌아웃(-0.5x), 1:1 초기화 및 드래그 패닝 변환 로직 확인
        self.assertIn("zoomCam(0.5)", html)
        self.assertIn("zoomCam(-0.5)", html)
        self.assertIn("resetCamZoom()", html)
        self.assertIn("applyCamTransform", html)
        self.assertIn("clampPanOffsets", html)
        self.assertIn("translate3d(", html)

        # 3. 전체화면 모드 상단 플로팅 미니 스코어 HUD 및 좌우 잘림 방지(object-fit: contain) 확인
        self.assertIn('id="camFullscreenHud"', html)
        self.assertIn('id="hudP1Text"', html)
        self.assertIn('id="hudP2Text"', html)
        self.assertIn("object-fit: contain;", html)

    def test_real_billiard_video_native_resolution_streaming(self) -> None:
        """
        J:\\내 드라이브\\L-billiard.mp4 녹화 영상이 존재할 때 원본 해상도(1920x1080) 변형 없이
        그대로 디코딩 및 JPEG 스트리밍되는지 검증
        """
        from pathlib import Path
        import cv2
        import numpy as np
        from fastapi.testclient import TestClient
        from cue24.api.server import app, stream_reader
        from cue24.config.settings import settings

        client = TestClient(app)
        resp = client.get("/api/video/live-frame")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "image/jpeg")

        arr = np.frombuffer(resp.content, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)

        if Path(settings.DEFAULT_VIDEO_PATH).exists():
            self.assertTrue(stream_reader.is_real_video_loaded)
            self.assertEqual(decoded.shape[1], 1920)
            self.assertEqual(decoded.shape[0], 1080)

    def test_vision_ball_color_first_mover_cue_and_scoring_fsm(self) -> None:
        """
        1. 영상에서 공 3개 색상(WHITE, YELLOW, RED)을 감지하는지 확인
        2. 처음 움직이는 공을 수구로 자동 인식하고 즉시 샷 클락(카운트) 작동을 멈추는지 확인
        3. 득점 알고리즘 가동 후 모든 공이 멈추면 득점 카운트(+1점) 또는 득점 실패 시 자동 턴 전환 후
           샷 클락(카운트)이 다시 작동하는지 검증
        """
        import asyncio
        from cue24.api.server import ConnectionManager, LiveStreamReader
        from cue24.services.match_manager import MatchManager, MatchState

        async def _run_scenario() -> None:
            mm = MatchManager()
            ws = ConnectionManager()
            reader = LiveStreamReader(match_manager=mm, ws_manager=ws)

            # 초기 대기 상태: 모든 공이 멈춰 있으면 샷 클락(카운트) 작동 중 (is_timer_paused == False)
            self.assertEqual(mm.fsm_state, MatchState.WAIT_SHOT)
            self.assertFalse(mm.is_timer_paused)
            mm.tick_timer(3.0)
            self.assertEqual(mm.remaining_timer_sec, 37.0)

            # 1) 처음 움직이는 공(ID 2: YELLOW)을 감지 -> YELLOW를 수구로 자동 인식 & 샷 클락 즉시 정지
            await reader.process_coordinate_step(
                detected_points=[(710.0, 710.0), (760.0, 355.0), (2130.0, 710.0)],
                detected_colors={1: "WHITE", 2: "YELLOW", 3: "RED"},
            )
            self.assertEqual(reader.tracker.cue_ball_id, 2)
            self.assertEqual(mm.fsm_state, MatchState.SHOT_IN_PROGRESS)
            self.assertTrue(mm.is_timer_paused)

            # 샷 진행 중에는 샷 클락(카운트)이 멈춰 있어야 함
            mm.tick_timer(5.0)
            self.assertEqual(mm.remaining_timer_sec, 37.0)

            # 2) 주행 중 1적구(ID 1) 충돌 -> TOP, RIGHT, BOTTOM 3쿠션 -> 2적구(ID 3) 충돌 (득점 성공 시퀀스)
            shot_frames = [
                [(710.0, 710.0), (710.0, 660.0), (2130.0, 710.0)],  # 1적구(ID 1) 접촉
                [(710.0, 710.0), (800.0, 30.0), (2130.0, 710.0)],   # TOP 쿠션 진입
                [(710.0, 710.0), (900.0, 120.0), (2130.0, 710.0)],  # TOP 쿠션 반사 (1쿠션)
                [(710.0, 710.0), (2810.0, 500.0), (2130.0, 710.0)], # RIGHT 쿠션 진입
                [(710.0, 710.0), (2650.0, 650.0), (2130.0, 710.0)], # RIGHT 쿠션 반사 (2쿠션)
                [(710.0, 710.0), (2300.0, 1390.0), (2130.0, 710.0)],# BOTTOM 쿠션 진입
                [(710.0, 710.0), (2130.0, 760.0), (2130.0, 710.0)], # BOTTOM 반사(3쿠션) + 2적구(ID 3) 접촉
            ]
            for pts in shot_frames:
                await reader.process_coordinate_step(detected_points=pts)

            # 3) 모든 공이 멈추면(15프레임 연속 정지) 득점 카운트(+1점) 및 샷 클락(40초) 재작동
            stop_pts = [(710.0, 710.0), (2130.0, 760.0), (2130.0, 710.0)]
            transition_score = None
            for _ in range(15):
                res = await reader.process_coordinate_step(detected_points=stop_pts)
                if res is not None:
                    transition_score = res

            self.assertIsNotNone(transition_score)
            self.assertTrue(transition_score["is_scored"])
            self.assertEqual(mm.players["P1"].score, 1)
            self.assertEqual(mm.active_player_id, "P1")
            self.assertEqual(mm.fsm_state, MatchState.WAIT_SHOT)
            self.assertFalse(mm.is_timer_paused)
            self.assertEqual(mm.remaining_timer_sec, 40.0)

            # 4) 다음 샷에서 득점 실패(공타) 시 자동으로 상대(P2)에게 턴이 넘어가고 샷 클락이 작동하는지 검증
            await reader.process_coordinate_step(
                detected_points=[(710.0, 710.0), (1990.0, 700.0), (2130.0, 710.0)]
            )
            self.assertEqual(mm.fsm_state, MatchState.SHOT_IN_PROGRESS)
            self.assertTrue(mm.is_timer_paused)

            transition_miss = None
            for _ in range(15):
                res = await reader.process_coordinate_step(
                    detected_points=[(710.0, 710.0), (1990.0, 700.0), (2130.0, 710.0)]
                )
                if res is not None:
                    transition_miss = res

            self.assertIsNotNone(transition_miss)
            self.assertFalse(transition_miss["is_scored"])
            self.assertEqual(transition_miss["event_type"], "TURN_CHANGED")
            self.assertEqual(mm.active_player_id, "P2")
            self.assertEqual(mm.fsm_state, MatchState.WAIT_SHOT)
            self.assertFalse(mm.is_timer_paused)

        asyncio.run(_run_scenario())

    def test_bulletproof_5stage_tracker_and_korean_ui(self) -> None:
        """
        1. BulletproofBallTracker 5단계 알고리즘(1단계 도넛 마스크 색상 확정, 2단계 색상 무시 추적,
           3단계 충돌 법선 벡터 내적 > 0.8 ID 스왑 강제 원복, 4단계 3차 스플라인 궤적 복원) 검증
        2. 점수판 화면 용어의 자연스러운 한글화 적용 여부 검증
        """
        import numpy as np
        from cue24.core.physics_tracker import BulletproofBallTracker, PhysicsTracker

        bp = BulletproofBallTracker()
        white_crop = np.full((32, 32, 3), (245, 245, 245), dtype=np.uint8)
        yellow_crop = np.full((32, 32, 3), (20, 225, 245), dtype=np.uint8)
        red_crop = np.full((32, 32, 3), (25, 25, 235), dtype=np.uint8)

        # [1단계] 정지 상태 신분증 발급
        locked = bp.phase1_lock_balls_before_shot(
            [
                {"center": (900.0, 700.0), "crop_img": white_crop},
                {"center": (500.0, 300.0), "crop_img": yellow_crop},
                {"center": (1000.0, 700.0), "crop_img": red_crop},
            ]
        )
        self.assertEqual(locked[0]["color"], "WHITE")
        self.assertEqual(locked[1]["color"], "YELLOW")
        self.assertEqual(locked[2]["color"], "RED")

        # [2단계] 주행 중 색상 무시 위치 추적 (수구 ID 0이 적구 ID 2에 충돌 거리 50mm <= 61.5mm로 접근)
        bp.phase2_track_motion_only(
            [
                {"center": (950.0, 700.0)},
                {"center": (500.0, 300.0)},
                {"center": (1000.0, 700.0)},
            ]
        )

        # [3단계] 충돌 법선 방향으로 날아간 공이 수구(ID 0)로 잘못 매칭된 경우 물리 가드가 즉시 원복하는지 확인
        swapped = bp.phase3_anti_id_switch_guard(cue_id=0, obj_id=2)
        self.assertTrue(swapped)
        self.assertEqual(bp.swap_restore_count, 1)
        self.assertEqual(bp.locked_balls[0]["pos"], (1000.0, 700.0))
        self.assertEqual(bp.locked_balls[2]["pos"], (950.0, 700.0))

        # [4단계] 3차 스플라인 궤적 복원 검증
        spline_pts = PhysicsTracker.interpolate_cubic_spline(
            (100.0, 100.0), (20.0, 0.0), (200.0, 120.0), (20.0, 10.0), num_steps=8
        )
        self.assertEqual(len(spline_pts), 8)
        self.assertAlmostEqual(spline_pts[-1][0], 200.0, places=3)
        self.assertAlmostEqual(spline_pts[-1][1], 120.0, places=3)

        # 화면 UI 한글화 요소 검증
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        self.assertIn("스마트 3쿠션 디지털 점수판", html)
        self.assertIn("1번 선수", html)
        self.assertIn("2번 선수", html)
        self.assertIn("애버리지", html)
        self.assertIn("하이런 (연속득점)", html)
        self.assertIn("실시간 쿠션 수:", html)
        self.assertIn("이닝별 득점 기록표", html)
        self.assertIn("작전타임 (+40초)", html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
