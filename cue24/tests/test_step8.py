"""
Step 8 단위 테스트: 큐스코(CUESCO) 초대형 스코어 폰트 및 빌리보드(PBA) 이닝 기록지/컨트롤 UI 검증
파일 경로: cue24/tests/test_step8.py
"""

import unittest

from cue24.api.server import STATIC_DIR


class TestStep8CuescoAndBilliBoardUI(unittest.TestCase):
    """Step 8 큐스코 초대형 폰트 및 빌리보드 이닝 스코어시트 UI 검증 스위트"""

    def test_cuesco_font_and_biliboard_sheet_elements(self) -> None:
        """
        static/index.html에 큐스코 스타일 고대비 스포츠 폰트(Oswald, Chakra Petch),
        빌리보드 스타일 이닝 기록지(INNING SCORE SHEET), PBA 룰 토글, 타임아웃, 수동 점수 보정 버튼이
        정상 포함되어 있는지 검증
        """
        index_path = STATIC_DIR / "index.html"
        html = index_path.read_text(encoding="utf-8")

        # 1. 큐스코 스타일 초대형 스포츠 폰트 및 수구 테마 클래스 확인
        self.assertIn("Oswald", html)
        self.assertIn("Chakra Petch", html)
        self.assertIn("white-theme", html)
        self.assertIn("yellow-theme", html)

        # 2. 빌리보드 스타일 하단 이닝 기록지(INNING SCORE SHEET) 확인
        self.assertIn("INNING SCORE SHEET", html)
        self.assertIn("sheetHeadRow", html)
        self.assertIn("renderInningSheet", html)

        # 3. PBA 룰 토글, 타임아웃(+40s), 수구 스왑, 수동 점수 보정 버튼 및 초대형 꽉 찬 득점 숫자 확인
        self.assertIn("toggleRuleMode", html)
        self.assertIn("TIME-OUT (+40s)", html)
        self.assertIn("adjustScore('P1', 1)", html)
        self.assertIn("swapCueBalls()", html)
        self.assertIn("clamp(11rem, min(48vh, 25vw), 32rem)", html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
