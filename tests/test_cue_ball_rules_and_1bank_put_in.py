"""
수구(Cue Ball) 결정 및 턴 유지/교대 규칙 + 1쿠션 넣어치기(원뱅크 걸어치기) 3쿠션 득점 검증 스위트
파일 경로: tests/test_cue_ball_rules_and_1bank_put_in.py
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests.test_bankshot_and_turn_lock import (
    test_1_cushion_and_2_cushion_bankshot_exact_count,
    test_order_reversal_prevents_false_bankshot,
    test_same_rail_sliding_no_extra_cushions,
    test_opponent_ball_noise_never_hijacks_turn_or_scores,
    test_residual_rolling_after_immediate_score_never_flips_turn,
    test_rule1_to_rule4_cue_identification_and_1cushion_bankshot,
)


def run_all_cue_ball_and_1bank_tests() -> None:
    test_1_cushion_and_2_cushion_bankshot_exact_count()
    test_order_reversal_prevents_false_bankshot()
    test_same_rail_sliding_no_extra_cushions()
    test_opponent_ball_noise_never_hijacks_turn_or_scores()
    test_residual_rolling_after_immediate_score_never_flips_turn()
    test_rule1_to_rule4_cue_identification_and_1cushion_bankshot()
    print("=== ALL CUE BALL RULES & 1-BANK PUT-IN TESTS PASSED! ===")


if __name__ == "__main__":
    run_all_cue_ball_and_1bank_tests()
