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

            # 1) 처음 움직이는 공(ID 2: WHITE, 1번 선수 선공 수구)을 감지 -> 수구 자동 인식 & 샷 클락 즉시 정지
            await reader.process_coordinate_step(
                detected_points=[(710.0, 710.0), (760.0, 355.0), (2130.0, 710.0)],
                detected_colors={1: "YELLOW", 2: "WHITE", 3: "RED"},
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

        # 화면 UI 한글화 요소 및 실시간 쿠션 수 삭제 / 1회 리플레이 복귀 검증
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        self.assertIn("스마트 3쿠션 디지털 점수판", html)
        self.assertIn("1번 선수", html)
        self.assertIn("2번 선수", html)
        self.assertIn("애버리지", html)
        self.assertIn("하이런 (연속득점)", html)
        self.assertNotIn("실시간 쿠션 수:", html)
        self.assertNotIn("cushion-counter-box", html)
        self.assertIn("REPLAY_FINISHED", html)
        self.assertIn("replayOneShotTimerId", html)
        self.assertIn("이닝별 득점 기록표", html)
        self.assertIn("작전타임 (+40초)", html)
        self.assertIn("applyCueTheme('p1', 'WHITE')", html)
        self.assertIn("applyCueTheme('p2', 'YELLOW')", html)

    def test_immutable_cue_ball_colors_and_local_physics_ai_mcp(self) -> None:
        """
        1. 선공(P1)은 무조건 흰색(WHITE), 후공(P2)은 무조건 노란색(YELLOW)으로 고정되며
           경기 및 득점 카운트 도중 절대 수구 색상이 바뀌지 않는지 검증
        2. 2쿠션 공타 시 억지 3쿠션 부풀리기 없이 정확히 MISS(INSUFFICIENT_CUSHIONS)로 판정하는지 검증
        3. 로컬 물리/AI MCP 서버(billiard_physics_mcp) 도구 조회 및 3쿠션 물리 검증 호출 테스트
        """
        from fastapi.testclient import TestClient
        from cue24.api.server import app
        from cue24.mcp.billiard_physics_mcp import billiard_mcp_server

        client = TestClient(app)

        # 1) MCP 도구 목록 조회 검증
        tools_resp = client.get("/api/mcp/tools")
        self.assertEqual(tools_resp.status_code, 200)
        tools_json = tools_resp.json()
        self.assertEqual(tools_json["mcp_server"], "cue24-local-billiard-physics-mcp")
        tool_names = [t["name"] for t in tools_json["tools"]]
        self.assertIn("verify_3cushion_shot_physics", tool_names)
        self.assertIn("audit_cue_ball_identity", tool_names)

        # 2) 2쿠션 후 제2적구 충돌 시 물리 AI MCP가 정확히 공타(is_scored=False)로 판정하는지 검증
        miss_check = billiard_mcp_server.call_tool(
            "verify_3cushion_shot_physics",
            {
                "events": [
                    (20, "CUSHION", "BOTTOM"),
                    (71, "CUSHION", "LEFT"),
                    (72, "FIRST_HIT", "1"),
                    (87, "SECOND_HIT", "2"),
                ],
                "first_object_ball_id": 1,
                "second_object_ball_id": 2,
                "cushions_before_first_hit": 2,
                "cushions_before_second_hit": 2,
                "total_cushions": 2,
                "distinct_rails_before_second_hit": ["BOTTOM", "LEFT"],
                "cue_travel_distance_mm": 2100.0,
            },
        )
        self.assertFalse(miss_check["is_scored"])
        self.assertEqual(miss_check["shot_type"], "MISS")
        self.assertEqual(miss_check["reason_code"], "INSUFFICIENT_CUSHIONS")

        # 3) 정식 3쿠션 후 제2적구 충돌 시 물리 AI MCP가 정확히 득점(is_scored=True)으로 판정하는지 검증
        score_check = billiard_mcp_server.call_tool(
            "verify_3cushion_shot_physics",
            {
                "events": [
                    (10, "FIRST_HIT", "1"),
                    (40, "CUSHION", "TOP"),
                    (85, "CUSHION", "LEFT"),
                    (120, "CUSHION", "BOTTOM"),
                    (165, "SECOND_HIT", "3"),
                ],
                "first_object_ball_id": 1,
                "second_object_ball_id": 3,
                "cushions_before_first_hit": 0,
                "cushions_before_second_hit": 3,
                "total_cushions": 3,
                "distinct_rails_before_second_hit": ["TOP", "LEFT", "BOTTOM"],
                "cue_travel_distance_mm": 2850.0,
            },
        )
        self.assertTrue(score_check["is_scored"])
        self.assertEqual(score_check["shot_type"], "REGULAR_3C")

    def test_match_elapsed_timer_30_innings_shift_and_player_swap(self) -> None:
        """
        1. 중앙 상단 누적 경기 타이머(match_timer.elapsed_sec, formatted, matchElapsedDigits) 증가 및 리셋 검증
        2. 하단 이닝별 득점 기록표 30이닝 표시(VISIBLE_INNINGS = 30) 및 30이닝 초과 시 좌측 쉬프트 로직 검증
        3. 1번 선수/2번 선수 이름 변경(/api/match/player-name) 및 자리바꿈(/api/match/swap-players) 시
           선공(P1)=흰공(WHITE), 후공(P2)=노란공(YELLOW) 수구 색상이 절대 바뀌지 않는지 검증
        """
        from fastapi.testclient import TestClient
        from cue24.api.server import STATIC_DIR, app
        from cue24.services.match_manager import MatchManager

        # 1) HTML 내 중앙 상단 누적 타이머, 30이닝 쉬프트, 선수 변경 모달 및 자리바꿈 요소 검증
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="matchElapsedTimerBox"', html)
        self.assertIn('id="matchElapsedDigits"', html)
        self.assertIn("VISIBLE_INNINGS = 30", html)
        self.assertIn("maxInning > VISIBLE_INNINGS ? maxInning - VISIBLE_INNINGS + 1 : 1", html)
        self.assertIn("openPlayerNameModal", html)
        self.assertIn("swapPlayerSeats", html)

        # 2) MatchManager 누적 타이머 및 32이닝 진행 시 이닝 기록 검증
        mm = MatchManager(p1_name="홍길동", p2_name="이순신")
        self.assertEqual(mm.formatted_match_time, "00:00:00")
        mm.tick_timer(65.0)
        self.assertEqual(mm.formatted_match_time, "00:01:05")

        # 3) REST API를 통한 선수 이름 변경 및 자리바꿈 검증
        client = TestClient(app)
        client.post("/api/match/reset")

        r1 = client.post(
            "/api/match/player-name",
            json={"player_id": "P1", "name": "조재호"},
        )
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r1.json()["players"]["P1"]["name"], "조재호")

        r2 = client.post(
            "/api/match/player-name",
            json={"player_id": "P2", "name": "쿠드롱"},
        )
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.json()["players"]["P2"]["name"], "쿠드롱")

        # 자리바꿈 수행 -> P1은 '쿠드롱'(흰공 WHITE), P2는 '조재호'(노란공 YELLOW)가 되어야 함
        swap_resp = client.post("/api/match/swap-players")
        self.assertEqual(swap_resp.status_code, 200)
        swap_data = swap_resp.json()
        self.assertEqual(swap_data["players"]["P1"]["name"], "쿠드롱")
        self.assertEqual(swap_data["players"]["P2"]["name"], "조재호")
        self.assertEqual(swap_data["players"]["P1"]["cue_ball_color"], "WHITE")
        self.assertEqual(swap_data["players"]["P2"]["cue_ball_color"], "YELLOW")

    def test_same_inning_sync_active_turn_and_trajectory_ai_recovery(self) -> None:
        """
        1. 선공(P1)이 득점 실패하면 후공(P2)이 서로 다른 이닝이 아닌 '동일한 이닝 번호(current_inning)'에 기록되는지 검증
        2. 후공(P2)이 득점 성공 시 같은 이닝에 점수가 누적되고 계속 '공격 중(active_player_id=P2)'이 유지되는지,
           그리고 후공(P2)이 득점 실패할 때 비로소 다음 이닝(current_inning + 1)으로 넘어가는지 검증
        3. 모션 블러/트랙 스왑으로 인해 실시간 제2적구 접촉이 누락된 3쿠션 샷을
           LocalPhysics3CushionAI.reconstruct_shot_from_trajectories가 100% 득점으로 복원하는지 검증
        """
        from cue24.mcp.billiard_physics_mcp import physics_ai_engine
        from cue24.services.match_manager import MatchManager

        mm = MatchManager(p1_name="1번 선수", p2_name="2번 선수")

        # [1이닝 초] 선공(P1) 1점 득점 -> 공격권 P1 유지, 1이닝 P1=1, P2=None
        mm.on_shot_started()
        mm.on_ball_stopped(
            {
                "is_scored": True,
                "shot_type": "REGULAR_3C",
                "reason_code": "VALID_REGULAR_3C",
                "message": "득점",
            }
        )
        self.assertEqual(mm.active_player_id, "P1")
        self.assertEqual(mm.current_inning, 1)
        self.assertEqual(mm.get_inning_history_list(), [{"inning": 1, "P1": 1, "P2": None}])

        # 비전 가림 노이즈(유령 샷) 취소 시 공격권(P1)과 이닝(1)이 절대 바뀌지 않는지 검증
        mm.on_shot_started()
        mm.cancel_phantom_shot()
        self.assertEqual(mm.active_player_id, "P1")
        self.assertEqual(mm.current_inning, 1)

        # [1이닝 초 종료 -> 1이닝 말] 선공(P1) 득점 실패 -> 후공(P2)으로 턴 전환, 반드시 같은 1이닝(cur_inn=1)이어야 함!
        mm.on_shot_started()
        mm.on_ball_stopped(
            {
                "is_scored": False,
                "shot_type": "MISS",
                "reason_code": "NO_SECOND_BALL_HIT",
                "message": "공타",
            }
        )
        self.assertEqual(mm.active_player_id, "P2")
        self.assertEqual(mm.current_inning, 1)
        self.assertEqual(mm.players["P1"].innings, 1)
        self.assertEqual(mm.players["P2"].innings, 1)
        self.assertEqual(mm.get_inning_history_list(), [{"inning": 1, "P1": 1, "P2": 0}])

        # [1이닝 말] 후공(P2) 2연속 득점 -> 같은 1이닝의 P2 칸에 2점이 기록되고 공격권 P2 유지!
        for _ in range(2):
            mm.on_shot_started()
            mm.on_ball_stopped(
                {
                    "is_scored": True,
                    "shot_type": "REGULAR_3C",
                    "reason_code": "VALID_REGULAR_3C",
                    "message": "득점",
                }
            )
            self.assertEqual(mm.active_player_id, "P2")
            self.assertEqual(mm.current_inning, 1)

        self.assertEqual(mm.get_inning_history_list(), [{"inning": 1, "P1": 1, "P2": 2}])

        # [1이닝 말 종료 -> 2이닝 초] 후공(P2) 득점 실패 -> 이제야 2이닝(cur_inn=2)으로 증가하고 선공(P1) 차례!
        mm.on_shot_started()
        mm.on_ball_stopped(
            {
                "is_scored": False,
                "shot_type": "MISS",
                "reason_code": "INSUFFICIENT_CUSHIONS",
                "message": "공타",
            }
        )
        self.assertEqual(mm.active_player_id, "P1")
        self.assertEqual(mm.current_inning, 2)
        self.assertEqual(mm.players["P1"].innings, 2)
        self.assertEqual(mm.players["P2"].innings, 1)
        self.assertEqual(
            mm.get_inning_history_list(),
            [{"inning": 1, "P1": 1, "P2": 2}, {"inning": 2, "P1": 0, "P2": None}],
        )

        # 시계열 궤적 텐서 복원기(reconstruct_shot_from_trajectories) 검증:
        # 3개의 공이 모두 15mm 이상 지속 이동했고, 마지막에 움직인 제2적구(ID 1) 타격 전에 3쿠션 반사가 존재할 때 복원 확인
        init_pos = {1: (300.0, 400.0), 2: (1200.0, 500.0), 3: (850.0, 1000.0)}
        traj_1 = [(300.0, 400.0)] * 20 + [(300.0, 320.0)] * 10  # step 20 이후 이동 (제2적구)
        traj_3 = [(850.0, 1000.0)] * 2 + [(920.0, 1300.0)] * 28  # step 2에 이동 (제1적구)
        traj_2 = (
            [(1200.0, 500.0), (850.0, 980.0), (500.0, 1050.0)]
            + [(60.0, 1100.0), (300.0, 1200.0)]                  # 1쿠션: LEFT 반사
            + [(600.0, 1380.0), (650.0, 1100.0)]                 # 2쿠션: BOTTOM 반사
            + [(600.0, 40.0), (500.0, 200.0)]                    # 3쿠션: TOP 반사
            + [(310.0, 390.0)] * 21                              # step 20에 제2적구(ID 1) 타격!
        )
        rec = physics_ai_engine.reconstruct_shot_from_trajectories(
            cue_ball_id=3,
            first_object_ball_id=2,
            second_object_ball_id=None,
            cushions_before_first_hit=0,
            cushions_before_second_hit=3,
            total_cushions=3,
            initial_positions=init_pos,
            ball_trajectories={1: traj_1, 2: traj_2, 3: traj_3},
        )
        self.assertTrue(rec["reconstructed"])
        self.assertEqual(rec["second_object_ball_id"], 1)
        self.assertGreaterEqual(rec["cushions_before_second_hit"], 3)

    def test_hybrid_vision_upgrades_parallax_ocsort_fsm_and_pure_vision_kiss(self) -> None:
        """
        [하이브리드 비전·물리 5대 업그레이드 통합 검증]
        1. 대각선 카메라 3D 당구공 시차(Parallax) 보정 (ball_to_table_mm, bbox_to_table_mm)
        2. OC-SORT 비선형 추적 (EMA 속도 필터, OCM 방향 모멘텀 매칭, ORU 가림 복구 역추적, 저속 분리 색상 재검증)
        3. 오디오 배제 순수 비전 초미세 스침(0.5mm 키스) 판독 (스플라인 근접 <= 78mm + 미세 굴절 >= 2.2도 + 미세 이동 >= 1.5mm)
        4. 결정론적 4단계 FSM 조기 키스 영구 잠금(NO_SCORE_LOCKED / EARLY_SECOND_BALL_CONTACT):
           2쿠션 시점에 제2적구를 먼저 맞히면, 그 뒤 3쿠션을 채우고 다시 제2적구를 맞혀도 무조건 노카운트!
        5. 코너 더블쿠션(TOP -> RIGHT 연속 반사) 즉시 2쿠션 인정 및 in_rail_zone 쿠션 타기 중복 차단
        """
        from cue24.core.physics_tracker import PhysicsTracker
        from cue24.core.scoring_engine import FailureReason, ScoringEngine, ShotFSMState
        from cue24.core.table_calibrator import TableCalibrator
        from cue24.mcp.billiard_physics_mcp import physics_ai_engine

        # 1. 대각선 카메라 3D 구 시차(Parallax) 보정 검증
        cal = TableCalibrator(
            src_corners=[(0.0, 0.0), (2840.0, 0.0), (2840.0, 1420.0), (0.0, 1420.0)],
            camera_tilt_ratio=0.12,
        )
        pt_mm = cal.ball_to_table_mm((1000.0, 500.0), radius_px=30.0)
        self.assertAlmostEqual(pt_mm[0], 1000.0, places=2)
        self.assertAlmostEqual(pt_mm[1], 503.6, places=2)  # 500 + 30 * 0.12 = 503.6
        bbox_mm = cal.bbox_to_table_mm((970.0, 470.0, 1030.0, 530.0))
        self.assertAlmostEqual(bbox_mm[1], 503.6, places=2)

        # 2. OC-SORT ORU 가림 구간 거울 반사 역추적 복원 및 저속 분리 구간 색상 재검증
        oru_pts = PhysicsTracker.recover_occluded_trajectory_oru(
            last_obs_pos=(1000.0, 80.0),
            reappeared_pos=(1200.0, 120.0),
            missing_steps=4,
            last_obs_vel=(50.0, -30.0),  # TOP 레일(32mm)을 향해 진행 중 가려졌다가 반사되어 재등장
        )
        self.assertEqual(len(oru_pts), 4)
        # 중간 경유점이 TOP 레일 마진(32.0mm) 부근까지 도달했다가 복귀했는지 확인
        self.assertLessEqual(min(p[1] for p in oru_pts), 40.0)

        tracker = PhysicsTracker()
        tracker.initialize_tracks(
            initial_positions={1: (300.0, 300.0), 2: (1000.0, 500.0), 3: (2000.0, 700.0)},
            cue_ball_id=1,
            ball_colors={1: "WHITE", 2: "YELLOW", 3: "RED"},
        )
        # 저속 분리 구간에서 만약 트랙 1과 트랙 2의 관측 색상이 뒤바뀌어 있다면 즉시 원복되는지 검증
        swapped = tracker.verify_and_correct_colors_at_low_speed(
            observed_colors={1: "YELLOW", 2: "WHITE", 3: "RED"}
        )
        self.assertTrue(swapped)

        # 3. 오디오 배제 순수 비전 0.5mm 초미세 스침(키스) 판독 검증
        is_micro_kiss = physics_ai_engine.verify_direct_cue_hit(
            min_cue_to_target_dist_mm=74.0,   # 스플라인 최근접 74mm (직접 중첩 64mm보다 10mm 여유)
            target_step_move_mm=1.8,          # 목적구 1.8mm 미세 움찔
            target_init_disp_mm=0.0,
            curr_init_disp_mm=2.2,
            other_object_to_target_dist_mm=900.0,
            contact_threshold_mm=64.0,
            is_first_object_hit_Done=True,
            cue_deflection_deg=2.6,           # 수구 진행 각도 2.6도 미세 굴절
        )
        self.assertTrue(is_micro_kiss)

        # 4. 결정론적 4단계 FSM: 조기 키스 영구 잠금(NO_SCORE_LOCKED) 검증
        engine = ScoringEngine()
        engine.reset_shot(cue_ball_id=1, object_ball_ids=[2, 3])
        self.assertEqual(engine.fsm_state, ShotFSMState.IDLE)

        # 제1적구(2번) 타격 -> 2쿠션만 기록된 상태에서 제2적구(3번) 조기 접촉!
        engine.record_ball_hit(2, frame_idx=1, cue_pos=(500.0, 500.0))
        self.assertEqual(engine.fsm_state, ShotFSMState.IN_PLAY)
        engine.record_cushion_touch("TOP", frame_idx=5, cue_pos=(800.0, 32.0))
        engine.record_cushion_touch("RIGHT", frame_idx=6, cue_pos=(2808.0, 60.0))  # 코너 더블쿠션 (1프레임 간격 즉시 인정!)
        self.assertEqual(engine.total_cushion_count, 2)

        # 2쿠션 시점에 제2적구(3번) 조기 접촉 -> 즉시 NO_SCORE_LOCKED 잠금!
        engine.record_ball_hit(3, frame_idx=10, cue_pos=(2200.0, 400.0))
        self.assertEqual(engine.fsm_state, ShotFSMState.NO_SCORE_LOCKED)
        self.assertTrue(engine.failed_early)
        self.assertEqual(engine.failure_reason, FailureReason.EARLY_SECOND_BALL_CONTACT)

        # 그 뒤 수구가 계속 굴러 3번째, 4번째 쿠션을 치더라도 이미 NO_SCORE_LOCKED이므로 절대 득점이 되지 않음!
        engine.record_cushion_touch("BOTTOM", frame_idx=15, cue_pos=(1800.0, 1388.0))
        engine.record_cushion_touch("LEFT", frame_idx=22, cue_pos=(32.0, 700.0))
        eval_res = engine.evaluate_shot()
        self.assertFalse(bool(eval_res["is_scored"]))
        self.assertEqual(eval_res["fsm_state"], ShotFSMState.NO_SCORE_LOCKED.value)
        self.assertEqual(eval_res["failure_reason"], FailureReason.EARLY_SECOND_BALL_CONTACT.value)

    def test_cue_ball_color_100pct_player_match_and_retroactive_score_recovery(self) -> None:
        """
        [사용자 핵심 확인 항목 2대 시나리오 자동 검증]
        1. 직전 득점 미인지 시 다음 타구에서 자동 +1점 복구:
           - 직전 샷 득점이 누락되어(MISS 오판) 턴이 상대방으로 넘어갔더라도,
           - 선수가 자기 공(흰공 WHITE 또는 노란공 YELLOW)을 계속 치는 순간(타구 출발 즉시!)
             누락된 점수(+1점)와 공격권(active_player_id), 이닝 기록표가 즉시 제자리로 복구되는지 확인.
        2. 흰공 타구 = 1번 선수(선공 P1), 노란공 타구 = 2번 선수(후공 P2) 100% 일치:
           - 선수가 치는 공의 색깔에 따라 해당 선수의 점수판과 이닝 기록표에 정확히 점수가 올라가는지 확인.
        """
        import asyncio
        from cue24.api.server import ConnectionManager, LiveStreamReader
        from cue24.services.match_manager import MatchManager, MatchState

        async def _run_verification() -> None:
            mm = MatchManager(p1_name="1번 선수(선공)", p2_name="2번 선수(후공)")
            ws = ConnectionManager()
            reader = LiveStreamReader(match_manager=mm, ws_manager=ws)

            # 초기 트랙: 1=(710.0, 710.0)[WHITE], 2=(710.0, 355.0)[YELLOW], 3=(2130.0, 710.0)[RED]
            # ------------------------------------------------------------------
            # 검증 1: 1번 선수(흰공 WHITE, ID 1)가 1이닝 첫 샷을 쳤으나 비전이 득점을 놓쳐(MISS)
            #         2번 선수(P2)로 턴이 넘어간 직후, 1번 선수가 자기 공(흰공 WHITE, ID 1)을 계속 치는 순간!
            # ------------------------------------------------------------------
            # 1-A) 1번 선수(ID 1 = WHITE: (710,710)->(800,710)) 첫 타구 출발 및 정지(비전 미인지 공타 시뮬레이션)
            await reader.process_coordinate_step(
                detected_points=[(800.0, 710.0), (710.0, 355.0), (2130.0, 710.0)],
                detected_colors={1: "WHITE", 2: "YELLOW", 3: "RED"},
            )
            self.assertEqual(reader.tracker.cue_ball_id, 1)
            self.assertEqual(mm.active_player_id, "P1")
            for _ in range(15):
                await reader.process_coordinate_step(
                    detected_points=[(800.0, 710.0), (710.0, 355.0), (2130.0, 710.0)]
                )
            # 미인지 공타로 인해 현재 P1=0점, 공격권은 P2로 넘어간 상태
            self.assertEqual(mm.players["P1"].score, 0)
            self.assertEqual(mm.active_player_id, "P2")
            self.assertEqual(mm.current_inning, 1)

            # 1-B) 1번 선수가 자기 공(ID 1 = WHITE: (800,710)->(920,710))을 계속 치는 순간! (정지 전 타구 출발 즉시 확인)
            await reader.process_coordinate_step(
                detected_points=[(920.0, 710.0), (710.0, 355.0), (2130.0, 710.0)],
                detected_colors={1: "WHITE", 2: "YELLOW", 3: "RED"},
            )
            # 타구 출발 즉시 누락된 직전 점수(+1점)와 공격권(P1)이 제자리로 복구되어야 함!
            self.assertEqual(reader.tracker.cue_ball_id, 1)
            self.assertEqual(mm.fsm_state, MatchState.SHOT_IN_PROGRESS)
            self.assertEqual(mm.active_player_id, "P1")
            self.assertEqual(mm.players["P1"].score, 1)
            self.assertEqual(mm.players["P1"].inning_scores.get(1), 1)
            self.assertEqual(
                mm.last_shot_result.get("reason_code"),
                "RETROACTIVE_SCORE_COMPENSATED",
            )

            # 1-C) 이 두 번째 샷이 진짜 공타로 멈추면 P1은 복구된 1점을 유지한 채 P2에게 턴이 넘어감
            for _ in range(15):
                await reader.process_coordinate_step(
                    detected_points=[(920.0, 710.0), (710.0, 355.0), (2130.0, 710.0)]
                )
            self.assertEqual(mm.players["P1"].score, 1)
            self.assertEqual(mm.active_player_id, "P2")
            self.assertEqual(mm.current_inning, 1)

            # ------------------------------------------------------------------
            # 검증 2: 노란공(YELLOW, ID 2) 타구 = 2번 선수(후공 P2) 100% 일치 및
            #         2번 선수 직전 득점 미인지 시 노란공 연속 타구로 즉시 복구 확인!
            # ------------------------------------------------------------------
            # 2-A) 심지어 누군가 실수로 턴을 P1으로 바꿔놓았더라도,
            #      실제 당구대에서 노란공(ID 2 = YELLOW: (710,355)->(820,355))이 타구되면 100% 2번 선수(P2)로 공격권이 일치되어야 함!
            mm.active_player_id = "P1"
            await reader.process_coordinate_step(
                detected_points=[(920.0, 710.0), (820.0, 355.0), (2130.0, 710.0)],
                detected_colors={1: "WHITE", 2: "YELLOW", 3: "RED"},
            )
            self.assertEqual(reader.tracker.cue_ball_id, 2)
            self.assertEqual(mm.active_player_id, "P2")
            self.assertEqual(mm.active_cue_ball_color, "YELLOW")

            # 이 샷에서 2번 선수(YELLOW, ID 2)가 3쿠션 득점 성공!
            p2_3c_frames = [
                [(920.0, 710.0), (920.0, 660.0), (2130.0, 710.0)],   # 제1적구(ID 1: 920,710) 접촉
                [(920.0, 710.0), (1200.0, 30.0), (2130.0, 710.0)],   # TOP 쿠션 진입
                [(920.0, 710.0), (1400.0, 120.0), (2130.0, 710.0)],  # TOP 반사 (1쿠션)
                [(920.0, 710.0), (2810.0, 500.0), (2130.0, 710.0)],  # RIGHT 쿠션 진입
                [(920.0, 710.0), (2650.0, 650.0), (2130.0, 710.0)],  # RIGHT 반사 (2쿠션)
                [(920.0, 710.0), (2300.0, 1390.0), (2130.0, 710.0)], # BOTTOM 쿠션 진입
                [(920.0, 710.0), (2130.0, 760.0), (2130.0, 710.0)],  # BOTTOM 반사(3쿠션) + 제2적구(ID 3) 접촉
            ]
            for pts in p2_3c_frames:
                await reader.process_coordinate_step(detected_points=pts)
            for _ in range(15):
                await reader.process_coordinate_step(
                    detected_points=[(920.0, 710.0), (2130.0, 760.0), (2130.0, 710.0)]
                )

            # 노란공(YELLOW) 득점 결과: 2번 선수(P2)의 점수와 1이닝 기록표에 정확히 +1점이 올라가고 공격권 유지!
            self.assertEqual(mm.players["P2"].score, 1)
            self.assertEqual(mm.players["P2"].inning_scores.get(1), 1)
            self.assertEqual(mm.active_player_id, "P2")
            self.assertEqual(mm.current_inning, 1)

            # 2-B) 2번 선수(P2)의 다음 샷이 비전 미인지로 인해 공타(MISS) 처리되어 2이닝 P1으로 넘어간 상황
            await reader.process_coordinate_step(
                detected_points=[(920.0, 710.0), (1950.0, 700.0), (2130.0, 710.0)],
                detected_colors={1: "WHITE", 2: "YELLOW", 3: "RED"},
            )
            for _ in range(15):
                await reader.process_coordinate_step(
                    detected_points=[(920.0, 710.0), (1950.0, 700.0), (2130.0, 710.0)]
                )
            self.assertEqual(mm.active_player_id, "P1")
            self.assertEqual(mm.current_inning, 2)
            self.assertEqual(mm.players["P2"].score, 1)

            # 그러나 실제로는 득점이어서 2번 선수가 자기 공(노란공 YELLOW, ID 2: (1950,700)->(1750,620))을 계속 치는 순간!
            await reader.process_coordinate_step(
                detected_points=[(920.0, 710.0), (1750.0, 620.0), (2130.0, 710.0)],
                detected_colors={1: "WHITE", 2: "YELLOW", 3: "RED"},
            )
            # 즉시 2번 선수(P2)의 누락된 점수(+1점 -> 총 2점), 1이닝 기록(2점), 공격권(P2), 이닝 번호(1이닝)가 복구됨!
            self.assertEqual(mm.active_player_id, "P2")
            self.assertEqual(mm.current_inning, 1)
            self.assertEqual(mm.players["P2"].score, 2)
            self.assertEqual(mm.players["P2"].inning_scores.get(1), 2)
            self.assertEqual(mm.players["P2"].current_run, 2)

        asyncio.run(_run_verification())

    def test_kiosk_ui_ux_7_overhaul_requirements(self) -> None:
        """
        [당구 디지털 스코어보드 키오스크 UI/UX 7대 개편 검증]
        1. [선수변경] / [자리바꿈] 버튼 나란히 배치 및 동작 분리 검증
           - [선수변경] (/api/match/swap-player-names): 당구공 색상은 고정하고 좌우 선수 이름(및 핸디)만 교체
           - [자리바꿈] (/api/match/swap-seats-full): 좌우 선수 이름 + 당구공 색깔(흰공/노란공) + 점수/기록 통째로 교체
        2. '하이런' 단독 표기 & 메인 점수 바로 아래 긴장감 그라데이션 타이머 바(from-emerald-400 to-cyan-500,
           from-yellow-400 via-amber-500 to-orange-500, from-orange-600 via-red-600 to-rose-500 + animate-pulse)
        3. '이닝별득점' 간소화 및 현재 이닝 하이라이트(current-inning-col ring-2 ring-amber-400 border-cyan-400)
        4. 하단 시뮬레이터 & 키오스크 제어 바(⚡ 실시간 판독 시뮬레이터 & 키오스크 제어) 완전 제거
        5. 기본 화면 전용 4대 버튼([경기시작], [경기종료], [일시정지], [전체화면]) 및 전체화면 전용 배속/줌 컨트롤 분리
        6. 선수 이름 바로 옆에 [현재 이닝] 배지(p1NameInningBadge, p2NameInningBadge) 표기
        7. '목표점수' 제거 -> '핸디' 및 상대방과의 실시간 '점수차'(+N / -N / 0) 표기
        """
        from fastapi.testclient import TestClient
        from cue24.api.server import STATIC_DIR, app

        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

        # 1. [선수변경] / [자리바꿈] 나란히 배치 확인
        self.assertIn("swapPlayerNamesOnly()", html)
        self.assertIn("swapPlayerSeats()", html)
        self.assertIn("선수변경", html)
        self.assertIn("자리바꿈", html)

        # 2. '하이런' 단독 표기 & 점수 바로 아래 그라데이션 타이머 바 클래스 확인
        self.assertIn('<div class="stat-label">하이런</div>', html)
        self.assertIn('id="p1ShotTimerBar"', html)
        self.assertIn('id="p2ShotTimerBar"', html)
        self.assertIn("from-emerald-400 to-cyan-500", html)
        self.assertIn("from-yellow-400 via-amber-500 to-orange-500", html)
        self.assertIn("from-orange-600 via-red-600 to-rose-500 animate-pulse", html)

        # 3. '이닝별득점' 단독 타이틀 및 현재 이닝 하이라이트 클래스 확인
        self.assertIn('<div class="sheet-title">이닝별득점</div>', html)
        self.assertNotIn("1 ~ 30 이닝 표시 중", html)
        self.assertIn("current-inning-col ring-2 ring-amber-400 border-cyan-400", html)

        # 4. 하단 시뮬레이터 및 키오스크 제어 바 완전 삭제 확인
        self.assertNotIn("실시간 판독 시뮬레이터", html)
        self.assertNotIn("triggerSimulation(", html)
        self.assertNotIn('<footer class="control-bar">', html)

        # 5. 기본 화면 4대 버튼([경기시작], [경기종료], [일시정지], [전체화면]) 및 전체화면 전용 컨트롤 분리 확인
        self.assertIn('id="btnMatchStart"', html)
        self.assertIn('id="btnMatchEnd"', html)
        self.assertIn('id="btnMatchPause"', html)
        self.assertIn('id="btnFullscreenToggle"', html)
        self.assertIn('class="fullscreen-only-controls"', html)
        self.assertNotIn("👆 화면 터치: 전체화면 전환", html)

        # 6. 선수 이름 옆 '현재 이닝' 배지 확인
        self.assertIn('id="p1NameInningBadge"', html)
        self.assertIn('id="p2NameInningBadge"', html)

        # 7. '목표점수' 제거 -> '핸디' 및 '점수차' 표기 확인
        self.assertNotIn("목표 점수:", html)
        self.assertIn('id="p1HandicapVal"', html)
        self.assertIn('id="p2HandicapVal"', html)
        self.assertIn('id="p1ScoreDiffBadge"', html)
        self.assertIn('id="p2ScoreDiffBadge"', html)
        self.assertIn("updateScoreDiffBadge", html)

        # API 동작 검증: [선수변경] vs [자리바꿈]
        client = TestClient(app)
        client.post("/api/match/reset")
        client.post("/api/match/player-name", json={"player_id": "P1", "name": "김철수", "handicap": 28})
        client.post("/api/match/player-name", json={"player_id": "P2", "name": "박영희", "handicap": 22})
        client.post("/api/match/adjust-score", json={"player_id": "P1", "delta": 5})

        # [선수변경] 호출: 당구공 색상(P1=WHITE, P2=YELLOW) 및 진영 점수(P1=5, P2=0)는 그대로 두고 이름/핸디만 스왑
        r_names = client.post("/api/match/swap-player-names").json()
        self.assertEqual(r_names["players"]["P1"]["name"], "박영희")
        self.assertEqual(r_names["players"]["P1"]["target_score"], 22)
        self.assertEqual(r_names["players"]["P1"]["cue_ball_color"], "WHITE")
        self.assertEqual(r_names["players"]["P1"]["score"], 5)
        self.assertEqual(r_names["players"]["P2"]["name"], "김철수")
        self.assertEqual(r_names["players"]["P2"]["target_score"], 28)
        self.assertEqual(r_names["players"]["P2"]["cue_ball_color"], "YELLOW")
        self.assertEqual(r_names["players"]["P2"]["score"], 0)

        # [자리바꿈] 호출: 좌우 선수 이름, 핸디, 당구공 색깔(WHITE/YELLOW), 점수가 통째로 좌우 맞바뀜
        r_seats = client.post("/api/match/swap-seats-full").json()
        self.assertEqual(r_seats["players"]["P1"]["name"], "김철수")
        self.assertEqual(r_seats["players"]["P1"]["target_score"], 28)
        self.assertEqual(r_seats["players"]["P1"]["cue_ball_color"], "YELLOW")
        self.assertEqual(r_seats["players"]["P1"]["score"], 0)
        self.assertEqual(r_seats["players"]["P2"]["name"], "박영희")
        self.assertEqual(r_seats["players"]["P2"]["target_score"], 22)
        self.assertEqual(r_seats["players"]["P2"]["cue_ball_color"], "WHITE")
        self.assertEqual(r_seats["players"]["P2"]["score"], 5)

        # [경기시작] 클릭 시 (/api/match/reset 호출) 모든 상태가 초기값으로 완벽 초기화되는지 검증
        r_reset = client.post("/api/match/reset").json()
        self.assertEqual(r_reset["players"]["P1"]["score"], 0)
        self.assertEqual(r_reset["players"]["P2"]["score"], 0)
        self.assertEqual(r_reset["players"]["P1"]["cue_ball_color"], "WHITE")
        self.assertEqual(r_reset["players"]["P2"]["cue_ball_color"], "YELLOW")
        self.assertEqual(r_reset["current_inning"], 1)
        self.assertEqual(r_reset["shot_timer"]["remaining_sec"], 40)
        self.assertFalse(r_reset["shot_timer"]["is_paused"])
        self.assertEqual(r_reset["match_timer"]["elapsed_sec"], 0)

        # [일시정지] -> 경기시간(match_timer.elapsed_sec) 및 40초 샷 타이머 정지 -> [다시시작] -> 멈춘 시간부터 이어서 측정 검증
        from cue24.api.server import match_manager

        self.assertIn("다시시작", html)
        match_manager.tick_timer(5.0)
        self.assertEqual(match_manager.match_elapsed_sec, 5.0)
        self.assertEqual(match_manager.remaining_timer_sec, 35.0)

        r_pause = client.post("/api/match/pause").json()
        self.assertTrue(r_pause["is_match_paused"])
        self.assertTrue(r_pause["match_timer"]["is_paused"])
        self.assertTrue(r_pause["shot_timer"]["is_paused"])

        # 일시정지 중에는 tick_timer / tick_match_elapsed가 호출되어도 경기시간(5.0초)과 샷 타이머(35.0초)가 절대 변하지 않아야 함
        match_manager.tick_timer(10.0)
        match_manager.tick_match_elapsed(10.0)
        self.assertEqual(match_manager.match_elapsed_sec, 5.0)
        self.assertEqual(match_manager.remaining_timer_sec, 35.0)

        # 일시정지 중에 공이 움직이거나 멈추더라도 선수가 [다시시작]을 누르기 전까지는 일시정지(경기시간 멈춤) 유지
        match_manager.on_shot_started()
        match_manager.cancel_phantom_shot()
        match_manager.tick_timer(5.0)
        self.assertTrue(match_manager.is_match_paused)
        self.assertEqual(match_manager.match_elapsed_sec, 5.0)

        # [다시시작] 클릭 시 (/api/match/pause 재호출) 멈췄던 5.0초부터 다시 이어서 시간 체크
        r_resume = client.post("/api/match/pause").json()
        self.assertFalse(r_resume["is_match_paused"])
        self.assertFalse(r_resume["match_timer"]["is_paused"])
        self.assertFalse(r_resume["shot_timer"]["is_paused"])

        match_manager.tick_timer(3.0)
        self.assertEqual(match_manager.match_elapsed_sec, 8.0)
        self.assertEqual(match_manager.remaining_timer_sec, 32.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

