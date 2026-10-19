"""
Step 9 단위 및 통합 테스트: 보드판 중앙 상단 실시간 라이브 캠 모니터 & 3공 정지 시 직전 샷 자동 리플레이(Auto-Replay) 전환 검증
파일 경로: cue24/tests/test_step9.py
"""

import unittest
from fastapi.testclient import TestClient

from cue24.api.server import STATIC_DIR, app


class TestStep9LiveCamAndAutoReplayUI(unittest.TestCase):
    """Step 9 중앙 상단 라이브캠 모니터 및 3공 정지 시 자동 리플레이 전환 검증 스위트"""

    def test_top_center_live_cam_and_auto_replay_ui_elements(self) -> None:
        """
        static/index.html의 중앙 허브 상단에 실시간 카메라 모니터(camMonitorBox, camStreamImg),
        AUTO_REPLAY_READY 자동 전환 핸들러, 배속 조절(1.0x/0.5x/0.25x), LIVE 복귀 버튼이
        정상 탑재되어 있는지 검증
        """
        index_path = STATIC_DIR / "index.html"
        html = index_path.read_text(encoding="utf-8")

        # 1. 중앙 상단 라이브캠 모니터 DOM 요소 및 초기 스트림 경로 확인
        self.assertIn('id="camMonitorBox"', html)
        self.assertIn('id="camStreamImg"', html)
        self.assertIn("/api/video/live", html)
        self.assertIn("/api/video/replay", html)

        # 2. 3공 정지(AUTO_REPLAY_READY) 시 자동 리플레이 전환 및 새 샷 출발 시 LIVE 복귀 로직 확인
        self.assertIn("AUTO_REPLAY_READY", html)
        self.assertIn("switchToReplayCam", html)
        self.assertIn("switchToLiveCam", html)

        # 3. 리플레이 배속 조절 버튼(1.0x, 0.5x, 0.25x) 확인
        self.assertIn("changeReplaySpeed(1.0)", html)
        self.assertIn("changeReplaySpeed(0.5)", html)
        self.assertIn("changeReplaySpeed(0.25)", html)

    def test_websocket_auto_replay_ready_broadcast_on_shot_stop(self) -> None:
        """
        샷 시뮬레이션 완료(3공 정지) 시 WebSocket으로 SCORE_UPDATED(또는 TURN_CHANGED)와 함께
        AUTO_REPLAY_READY 이벤트가 브로드캐스트되는지 검증
        """
        client = TestClient(app)
        client.post("/api/match/reset")

        with client.websocket_connect("/ws/scoreboard") as ws:
            initial_msg = ws.receive_json()
            self.assertEqual(initial_msg["type"], "SCORE_UPDATED")

            # WebSocket으로 일반 3쿠션 득점 샷 실행
            ws.send_json({"action": "SIMULATE_SHOT", "scenario": "REGULAR_3C"})

            # 1) 샷 시작 시 SCORE_UPDATED (SHOT_IN_PROGRESS)
            msg1 = ws.receive_json()
            self.assertEqual(msg1["type"], "SCORE_UPDATED")

            # 2) 3공 완전 정지 시 SCORE_UPDATED (득점 반영)
            msg2 = ws.receive_json()
            self.assertEqual(msg2["type"], "SCORE_UPDATED")
            self.assertEqual(msg2["data"]["players"]["P1"]["score"], 1)

            # 3) 직전 샷 리플레이 자동 전환 이벤트 (AUTO_REPLAY_READY)
            msg3 = ws.receive_json()
            self.assertEqual(msg3["type"], "AUTO_REPLAY_READY")
            self.assertGreater(msg3["data"]["frame_count"], 0)
            self.assertEqual(msg3["data"]["shot_type"], "REGULAR_3C")
            self.assertTrue(msg3["data"]["is_scored"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
