"""
평범한 득점 실패의 득점 오인식(False Positive) 원천 차단 및 8대 악성 키스 방어 통합 검증
파일 경로: tests/test_false_positive_prevention.py
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
    test_4_perspective_overlap_58mm_and_spin_curve_without_touch,
    test_5_moving_second_ball_friction_and_bbox_jitter_after_kiss,
    test_6_kiss_pushes_2nd_ball_into_early_cue_kiss_locks_no_score,
    test_7_kiss_then_genuine_thin_score_on_moving_ball,
    test_8_kiss_then_genuine_score_on_stationary_ball,
)
from tests.test_lab_hungarian_and_false_positives import (
    test_1_lab_hungarian_ball_identifier_with_glare,
    test_2_ordinary_miss_false_positive_blocked,
)


def run_all_false_positive_prevention_tests() -> None:
    test_1_lab_hungarian_ball_identifier_with_glare()
    test_2_ordinary_miss_false_positive_blocked()
    test_1_kiss_shockwave_immediate_pass_after_2_frames()
    test_2_chain_cannon_cue_hits_1st_into_2nd_ball()
    test_3_kiss_id_swap_spot_replacement_foul()
    test_4_perspective_overlap_58mm_and_spin_curve_without_touch()
    test_5_moving_second_ball_friction_and_bbox_jitter_after_kiss()
    test_6_kiss_pushes_2nd_ball_into_early_cue_kiss_locks_no_score()
    test_7_kiss_then_genuine_thin_score_on_moving_ball()
    test_8_kiss_then_genuine_score_on_stationary_ball()
    print("=== ALL FALSE-POSITIVE PREVENTION TESTS PASSED! ===")


if __name__ == "__main__":
    run_all_false_positive_prevention_tests()
