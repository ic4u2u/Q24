"""
쿠션 정확도 및 cv2.approxPolyDP 폴리라인 꼭짓점 단순화 검증 테스트 스위트
파일 경로: tests/test_cushion_accuracy.py
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests.test_cushion_upgrades import (
    test_1_corner_ping_pong_and_hysteresis_lock,
    test_2_wide_zone_kinematic_bounce_apex,
    test_3_cushion_freeze_after_valid_score,
)


def run_all_cushion_accuracy_tests() -> None:
    test_1_corner_ping_pong_and_hysteresis_lock()
    test_2_wide_zone_kinematic_bounce_apex()
    test_3_cushion_freeze_after_valid_score()
    print("=== ALL CUSHION ACCURACY (cv2.approxPolyDP) TESTS PASSED! ===")


if __name__ == "__main__":
    run_all_cushion_accuracy_tests()
