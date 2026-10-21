"""
3체 자리바꿈 키스(Spot Replacement Kiss) 및 간접 연쇄 키스 방어 검증 테스트 스위트
파일 경로: tests/test_3body_replacement_kiss.py
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests.test_kiss_false_positives import (
    test_1_kiss_shockwave_immediate_pass_after_2_frames,
    test_2_chain_cannon_cue_hits_1st_into_2nd_ball,
    test_3_kiss_id_swap_spot_replacement_foul,
    test_6_kiss_pushes_2nd_ball_into_early_cue_kiss_locks_no_score,
)


def run_all_3body_replacement_kiss_tests() -> None:
    test_1_kiss_shockwave_immediate_pass_after_2_frames()
    test_2_chain_cannon_cue_hits_1st_into_2nd_ball()
    test_3_kiss_id_swap_spot_replacement_foul()
    test_6_kiss_pushes_2nd_ball_into_early_cue_kiss_locks_no_score()
    print("=== ALL 3-BODY REPLACEMENT KISS TESTS PASSED! ===")


if __name__ == "__main__":
    run_all_3body_replacement_kiss_tests()
