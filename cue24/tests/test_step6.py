"""
Step 6 단위 및 통합 테스트: 20인치 태블릿 UI(static/index.html) 및 통합 엔트리포인트(main.py) 검증
파일 경로: cue24/tests/test_step6.py
"""

import tempfile
import unittest
from pathlib import Path

from cue24.api.server import (
    STATIC_DIR,
    SimulateShotRequest,
    get_match_state,
    reset_match_state,
    simulate_shot,
)
from cue24.main import create_demo_3cushion_video, parse_args


class TestStep6FrontendAndMain(unittest.IsolatedAsyncioTestCase):
    """Step 6 프론트엔드 정적 파일 및 통합 엔트리포인트 검증 스위트"""

    def test_static_scoreboard_html_exists_and_contains_key_elements(self) -> None:
        """
        static/index.html 파일이 존재하며, CUESCO 스타일 UI 필수 요소와
        [CUE24 HAWK-EYE VAR] 팝업 및 WebSocket(/ws/scoreboard) 클라이언트 로직이 포함되어 있는지 검증
        """
        index_path = STATIC_DIR / "index.html"
        self.assertTrue(index_path.exists(), "static/index.html 파일이 존재해야 합니다.")

        html = index_path.read_text(encoding="utf-8")
        self.assertIn("CUE24 HAWK-EYE VAR", html)
        self.assertIn("/ws/scoreboard", html)
        self.assertNotIn("cushion-counter-box", html)
        self.assertIn("PLAYER 1", html)
        self.assertIn("PLAYER 2", html)

    async def test_api_simulation_and_reset_flow(self) -> None:
        """
        태블릿 UI에서 호출하는 시뮬레이션 엔드포인트(일반 3C, VAR 미세접촉, VAR 더블터치)가
        정상적으로 점수와 VAR 팝업 데이터를 반환하는지 검증
        """
        await reset_match_state()
        init_state = await get_match_state()
        self.assertEqual(init_state["players"]["P1"]["score"], 0)

        # 1. 일반 3C 득점 시뮬레이션
        sim_3c = await simulate_shot(SimulateShotRequest(scenario="REGULAR_3C"))
        self.assertTrue(sim_3c["transition"]["is_scored"])
        self.assertEqual(sim_3c["transition"]["payload"]["players"]["P1"]["score"], 1)

        # 2. VAR 미세 접촉 시뮬레이션
        sim_var = await simulate_shot(SimulateShotRequest(scenario="VAR_MICRO_CONTACT"))
        self.assertIsNotNone(sim_var["var_alert"])
        self.assertTrue(sim_var["var_alert"]["var_triggered"])
        self.assertEqual(sim_var["var_alert"]["status"], "CONTACT_CONFIRMED")

        # 3. VAR 쿠션 더블터치 파울 시뮬레이션 -> 공타 및 P2로 턴 교체
        sim_dt = await simulate_shot(SimulateShotRequest(scenario="VAR_DOUBLE_TOUCH"))
        self.assertIsNotNone(sim_dt["var_alert"])
        self.assertTrue(sim_dt["var_alert"]["is_double_touch"])
        self.assertEqual(sim_dt["transition"]["active_player"], "P2")

    def test_demo_video_generation_and_cli_args(self) -> None:
        """
        main.py의 CLI 인자 파싱 및 가상 데모 MP4 비디오 생성 기능이 정상 작동하는지 검증
        """
        args = parse_args(["--port", "8080", "--check"])
        self.assertEqual(args.port, 8080)
        self.assertTrue(args.check)

        with tempfile.TemporaryDirectory() as tmp_dir:
            video_file = Path(tmp_dir) / "demo_test.mp4"
            out_path = create_demo_3cushion_video(str(video_file), fps=15)
            self.assertTrue(Path(out_path).exists())
            self.assertGreater(Path(out_path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
