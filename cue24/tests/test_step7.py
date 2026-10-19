"""
Step 7 단위 테스트: 직전 샷 자동 리플레이 링버퍼(ReplayRingBuffer), PBA 뱅크샷 2점 룰,
이닝 히스토리 기록지, 타임아웃(+40초), 수동 점수 보정 및 서버 엔드포인트 검증
파일 경로: cue24/tests/test_step7.py
"""

import unittest

from cue24.api.server import (
    AdjustScoreRequest,
    RuleModeRequest,
    SimulateShotRequest,
    adjust_player_score,
    change_rule_mode,
    get_replay_info,
    reset_match_state,
    simulate_shot,
    swap_player_cue_balls,
    use_match_timeout,
)
from cue24.core.replay_buffer import ReplayRingBuffer, SyntheticCamGenerator
from cue24.services.match_manager import MatchManager


class TestStep7ReplayAndPBARules(unittest.IsolatedAsyncioTestCase):
    """Step 7 리플레이 버퍼 및 PBA/큐스코 정밀 규칙 확장 단위 테스트 스위트"""

    def test_replay_ring_buffer_and_synthetic_cam(self) -> None:
        """
        1. SyntheticCamGenerator가 라이브 프레임과 시나리오별 리플레이 클립을 정상 생성하는지 검증
        2. ReplayRingBuffer가 샷 진행 프레임을 저장하고 3공 정지 시 직전 샷 클립을 확정(Lock)하는지 검증
        """
        cam = SyntheticCamGenerator(width=640, height=320)
        live_jpg = cam.render_live_frame(active_cue_color="WHITE", fsm_state="WAIT_SHOT")
        self.assertGreater(len(live_jpg), 1000)

        clip_frames = cam.generate_shot_replay_clip(
            scenario="BANK_SHOT", active_cue_color="YELLOW", num_frames=20
        )
        self.assertEqual(len(clip_frames), 20)

        buf = ReplayRingBuffer(max_frames=100)
        for f in clip_frames:
            buf.push_frame(f)

        meta = buf.lock_last_shot(shot_type="BANK_SHOT", is_scored=True, message="뱅크샷 2점!")
        self.assertEqual(meta["frame_count"], 20)
        self.assertEqual(meta["shot_type"], "BANK_SHOT")
        self.assertEqual(len(buf.get_locked_clip()), 20)

    def test_pba_bank_shot_2points_and_inning_history(self) -> None:
        """
        1. PBA 룰 모드("PBA")에서 뱅크샷 성공 시 +2점, 일반 3C 성공 시 +1점 가산 검증
        2. 이닝별 득점 기록지(inning_history)가 정확히 누적되는지 검증
        3. 타임아웃 사용 시 +40초 연장 및 잔여 타임아웃 차감 검증
        """
        mm = MatchManager(p1_name="조재호", p2_name="쿠드롱", rule_mode="PBA")

        # 1이닝 P1: 뱅크샷 성공(+2점) -> 일반 3C 성공(+1점) -> 공타 (1이닝 총 3점 기록)
        mm.on_shot_started()
        res_bank = mm.on_ball_stopped(
            {
                "is_scored": True,
                "shot_type": "BANK_SHOT",
                "reason_code": "VALID_BANK_SHOT",
                "message": "PBA 뱅크샷 2점!",
            }
        )
        self.assertEqual(res_bank["points_awarded"], 2)
        self.assertEqual(mm.players["P1"].score, 2)
        self.assertEqual(mm.players["P1"].bank_shots, 1)

        mm.on_shot_started()
        mm.on_ball_stopped(
            {
                "is_scored": True,
                "shot_type": "REGULAR_3C",
                "reason_code": "VALID_REGULAR_3C",
                "message": "일반 1점!",
            }
        )
        self.assertEqual(mm.players["P1"].score, 3)
        self.assertEqual(mm.players["P1"].high_run, 3)

        # P1 공타 -> P2로 턴 교체
        mm.on_shot_started()
        mm.on_ball_stopped(
            {
                "is_scored": False,
                "shot_type": "MISS",
                "reason_code": "INSUFFICIENT_CUSHIONS",
                "message": "공타",
            }
        )
        self.assertEqual(mm.active_player_id, "P2")

        # 이닝 히스토리 시트 확인 (1이닝 P1=3점, P2=0점)
        sheet = mm.get_inning_history_list()
        self.assertEqual(len(sheet), 1)
        self.assertEqual(sheet[0]["inning"], 1)
        self.assertEqual(sheet[0]["P1"], 3)
        self.assertEqual(sheet[0]["P2"], 0)

        # P2 타임아웃 사용 검증 (잔여 2 -> 1, 타이머 40초 -> 80초)
        mm.tick_timer(15.0)  # 40 -> 25초
        mm.use_timeout()
        self.assertEqual(mm.players["P2"].timeouts_remaining, 1)
        self.assertEqual(mm.remaining_timer_sec, 65.0)

    async def test_server_auto_replay_and_control_endpoints(self) -> None:
        """
        서버 API에서 PBA 룰 변경, 뱅크샷 시뮬레이션 후 AUTO_REPLAY_READY 메타데이터 생성,
        수동 점수 가감(+1/-1), 수구 스왑, 타임아웃 엔드포인트가 정상 작동하는지 검증
        """
        await reset_match_state()
        await change_rule_mode(RuleModeRequest(rule_mode="PBA"))

        sim_res = await simulate_shot(SimulateShotRequest(scenario="BANK_SHOT"))
        self.assertEqual(sim_res["transition"]["points_awarded"], 2)
        self.assertIn("replay_meta", sim_res)
        self.assertGreater(sim_res["replay_meta"]["frame_count"], 0)

        replay_info = await get_replay_info()
        self.assertEqual(replay_info["shot_type"], "BANK_SHOT")

        # 수동 점수 보정 (+1 / -1)
        adj_res = await adjust_player_score(AdjustScoreRequest(player_id="P1", delta=1))
        self.assertEqual(adj_res["players"]["P1"]["score"], 3)

        # 타임아웃 및 수구 색상 스왑
        to_res = await use_match_timeout()
        self.assertEqual(to_res["players"]["P1"]["timeouts_remaining"], 1)

        swap_res = await swap_player_cue_balls()
        self.assertEqual(swap_res["players"]["P1"]["cue_ball_color"], "YELLOW")
        self.assertEqual(swap_res["players"]["P2"]["cue_ball_color"], "WHITE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
