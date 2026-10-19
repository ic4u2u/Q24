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


if __name__ == "__main__":
    unittest.main(verbosity=2)
