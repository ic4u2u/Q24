"""
Step 5 단위 테스트: FSM 경기 관리자(MatchManager) 및 FastAPI WebSocket 실시간 브로드캐스트 검증
파일 경로: cue24/tests/test_step5.py
"""

import unittest
from typing import Any, Dict, List

from cue24.api.server import ConnectionManager, LiveStreamReader
from cue24.services.match_manager import MatchManager, MatchState


class DummyWebSocket:
    """WebSocket 브로드캐스트 수신 검증용 가상(Mock) 클라이언트 클래스"""

    def __init__(self) -> None:
        self.received_messages: List[Dict[str, Any]] = []

    async def accept(self) -> None:
        pass

    async def send_json(self, message: Dict[str, Any]) -> None:
        self.received_messages.append(message)


class TestStep5MatchManagerAndServer(unittest.IsolatedAsyncioTestCase):
    """MatchManager FSM 규칙 및 비동기 WebSocket 파이프라인 단위 테스트 스위트"""

    def test_match_manager_fsm_scoring_and_turn_switch(self) -> None:
        """
        1. 초기 상태(WAIT_SHOT, P1=WHITE, 타이머 40초) 확인
        2. 샷 시작 시 타이머 즉시 일시 정지(Pause) 및 SHOT_IN_PROGRESS 전환 확인
        3. 득점 성공 시 점수 +1, 에버리지/하이런 갱신, 타이머 40초 리셋, 공격권(P1) 유지 확인
        4. 공타 실패 시 P2로 턴 교체, 수구 색상 YELLOW 스위칭, 이닝 +1, 타이머 40초 리셋 확인
        """
        mm = MatchManager(p1_name="홍길동", p2_name="이순신")

        # 초기 상태 검증
        self.assertEqual(mm.fsm_state, MatchState.WAIT_SHOT)
        self.assertEqual(mm.active_player_id, "P1")
        self.assertEqual(mm.active_cue_ball_color, "WHITE")
        self.assertEqual(mm.remaining_timer_sec, 40.0)
        self.assertFalse(mm.is_timer_paused)

        # 타이머 12초 경과 모사 (40초 -> 28초)
        mm.tick_timer(12.0)
        self.assertEqual(mm.remaining_timer_sec, 28.0)

        # 샷 출발 감지 -> 타이머 즉시 일시 정지 확인
        mm.on_shot_started()
        self.assertEqual(mm.fsm_state, MatchState.SHOT_IN_PROGRESS)
        self.assertTrue(mm.is_timer_paused)
        # 일시 정지 중에는 tick_timer가 호출되어도 시간이 줄어들지 않아야 함
        mm.tick_timer(5.0)
        self.assertEqual(mm.remaining_timer_sec, 28.0)

        # [득점 1]: P1 일반 3쿠션 득점 성공
        res_score1 = mm.on_ball_stopped(
            {
                "is_scored": True,
                "shot_type": "REGULAR_3C",
                "reason_code": "VALID_REGULAR_3C",
                "message": "득점 인정",
                "cushions_before_second_hit": 3,
                "total_cushions": 3,
            }
        )
        self.assertEqual(res_score1["event_type"], "SCORE_UPDATED")
        self.assertEqual(mm.active_player_id, "P1")
        self.assertEqual(mm.active_cue_ball_color, "WHITE")
        self.assertEqual(mm.players["P1"].score, 1)
        self.assertEqual(mm.players["P1"].innings, 1)
        self.assertEqual(mm.players["P1"].average, 1.000)
        self.assertEqual(mm.players["P1"].high_run, 1)
        # 타이머 40초 리셋 및 재시작 확인
        self.assertEqual(mm.remaining_timer_sec, 40.0)
        self.assertFalse(mm.is_timer_paused)
        self.assertEqual(mm.fsm_state, MatchState.WAIT_SHOT)

        # [득점 2]: P1 연속 2점째 득점 (하이런 2, 에버리지 2.000)
        mm.on_shot_started()
        mm.on_ball_stopped(
            {
                "is_scored": True,
                "shot_type": "BANK_SHOT",
                "reason_code": "VALID_BANK_SHOT",
                "message": "뱅크샷 득점",
                "cushions_before_second_hit": 3,
                "total_cushions": 3,
            }
        )
        self.assertEqual(mm.players["P1"].score, 2)
        self.assertEqual(mm.players["P1"].high_run, 2)
        self.assertEqual(mm.players["P1"].average, 2.000)

        # [공타 발생]: P1 3번째 시도에서 2쿠션 공타 -> P2(YELLOW)로 턴 자동 교체 및 P2 이닝 1로 증가
        mm.on_shot_started()
        res_miss = mm.on_ball_stopped(
            {
                "is_scored": False,
                "shot_type": "MISS",
                "reason_code": "INSUFFICIENT_CUSHIONS",
                "message": "공타: 쿠션 부족",
                "cushions_before_second_hit": 2,
                "total_cushions": 2,
            }
        )
        self.assertEqual(res_miss["event_type"], "TURN_CHANGED")
        self.assertEqual(mm.active_player_id, "P2")
        self.assertEqual(mm.active_cue_ball_color, "YELLOW")
        self.assertEqual(mm.players["P2"].innings, 1)
        self.assertEqual(mm.players["P1"].current_run, 0)
        self.assertEqual(mm.players["P1"].high_run, 2)  # 하이런은 유지되어야 함
        self.assertEqual(mm.remaining_timer_sec, 40.0)

        # [P2 공타 발생]: P2도 공타 -> 다시 P1(WHITE)으로 턴 교체 및 P1 이닝 2로 증가 (P1 에버리지: 2점/2이닝 = 1.000)
        mm.on_shot_started()
        mm.on_ball_stopped(
            {
                "is_scored": False,
                "shot_type": "MISS",
                "reason_code": "NO_SECOND_BALL_HIT",
                "message": "공타: 2적구 불발",
            }
        )
        self.assertEqual(mm.active_player_id, "P1")
        self.assertEqual(mm.active_cue_ball_color, "WHITE")
        self.assertEqual(mm.players["P1"].innings, 2)
        self.assertEqual(mm.players["P1"].average, 1.000)

    async def test_websocket_broadcast_and_live_stream_pipeline(self) -> None:
        """
        ConnectionManager와 LiveStreamReader를 연동하여 실시간 프레임 처리 시
        SCORE_UPDATED, VAR_ALERT, TURN_CHANGED 웹소켓 이벤트가 정상 브로드캐스트되는지 검증
        """
        ws_mgr = ConnectionManager()
        mm = MatchManager()
        reader = LiveStreamReader(match_manager=mm, ws_manager=ws_mgr)

        dummy_client = DummyWebSocket()
        await ws_mgr.connect(dummy_client)  # type: ignore[arg-type]

        # 1. 수구(ID 1) 고속 출발 주입 -> 샷 감지 및 타이머 정지 SCORE_UPDATED 브로드캐스트 발생
        await reader.process_coordinate_step(
            detected_points=[(750.0, 710.0), (710.0, 355.0), (2130.0, 710.0)]
        )
        self.assertEqual(mm.fsm_state, MatchState.SHOT_IN_PROGRESS)
        self.assertTrue(
            any(m["type"] == "SCORE_UPDATED" for m in dummy_client.received_messages)
        )

        # 2. 주행 중 VAR_ALERT 발생 시나리오 주입
        var_sample = reader.var_referee.evaluate_cushion_double_touch(
            cushion_distances_mm=[35.0, 29.8, 31.2, 30.1, 36.0],
            timestamps_ms=[0.0, 20.0, 45.0, 70.0, 100.0],
        )
        await reader.process_coordinate_step(
            detected_points=[(760.0, 710.0), (710.0, 355.0), (2130.0, 710.0)],
            var_alert_payload=var_sample,
        )
        self.assertTrue(
            any(m["type"] == "VAR_ALERT" for m in dummy_client.received_messages)
        )

        # 3. 15프레임 연속 정지 주입 -> 공타 판정 후 TURN_CHANGED 브로드캐스트 발생 확인
        for _ in range(15):
            await reader.process_coordinate_step(
                detected_points=[(760.0, 710.0), (710.0, 355.0), (2130.0, 710.0)]
            )

        self.assertEqual(mm.fsm_state, MatchState.WAIT_SHOT)
        self.assertEqual(mm.active_player_id, "P2")
        self.assertTrue(
            any(m["type"] == "TURN_CHANGED" for m in dummy_client.received_messages)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
