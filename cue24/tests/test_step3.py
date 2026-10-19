"""
Step 3 단위 테스트: 진원도 후보 검출(BallCandidateDetector), 물리 충돌 가드 추적기(PhysicsTracker),
스핀 정지 감지기(SpinDetector) 검증
파일 경로: cue24/tests/test_step3.py
"""

import unittest
import cv2
import numpy as np

from cue24.config.settings import settings
from cue24.core.physics_tracker import (
    BallCandidateDetector,
    PhysicsTracker,
    SpinDetector,
)


class TestStep3PhysicsAndSpin(unittest.TestCase):
    """Step 3 물리 충돌 가드 및 스핀 감지 엔진 단위 테스트 스위트"""

    def test_ball_candidate_detector_circularity_filter(self) -> None:
        """
        파란색 라사지 배경에서 선수의 팔과 큐대(낮은 진원도)를 배제하고
        진원도(> 0.72)를 만족하는 당구공 3개만 정확히 검출하는지 검증
        """
        # 1420 x 2840 파란색 라사지 배경 생성 (BGR: 180, 90, 20 -> HSV Blue 영역)
        frame = np.full((1420, 2840, 3), (180, 90, 20), dtype=np.uint8)

        # 공 3개 그리기 (반지름 31px)
        true_centers = [(500, 700), (1400, 700), (2100, 500)]
        colors = [(245, 245, 245), (20, 220, 245), (30, 30, 230)]
        for center, color in zip(true_centers, colors):
            cv2.circle(frame, center, 31, color, thickness=-1)

        # 방해물 1: 길쭉한 큐대 (두께 12px, 길이 400px -> 면적은 공과 비슷하지만 진원도가 매우 낮음)
        cv2.line(frame, (100, 680), (450, 695), (190, 210, 230), thickness=12)

        # 방해물 2: 선수의 팔/손 그림자 (타원형 길쭉한 도형 -> 진원도 < 0.72)
        cv2.ellipse(frame, (250, 300), (95, 18), 15, 0, 360, (160, 180, 210), thickness=-1)

        detector = BallCandidateDetector()
        candidates = detector.detect(frame)

        # 정확히 3개의 공만 검출되어야 함
        self.assertEqual(len(candidates), 3)
        for cand in candidates:
            self.assertGreater(cand["circularity"], detector.CIRCULARITY_TH)

    def test_physics_tracker_collision_id_swap_guard(self) -> None:
        """
        가상 충돌 좌표 시퀀스를 주입하여 최단 거리 매칭으로 인한 ID 스왑이 발생했을 때,
        물리 충돌 가드(법선 벡터 N 코사인 유사도 >= 0.85)가 이를 감지하고
        수구와 적구의 ID/위치/속도를 강제 원복하는지 검증
        """
        tracker = PhysicsTracker()

        # 초기 상태 등록:
        # ID 1 (수구, WHITE): (900, 700)에서 오른쪽(+x 방향)으로 고속 이동 예정
        # ID 2 (제1목적구, RED): (1000, 700)에 정지 상태
        # ID 3 (제2목적구, YELLOW): (2000, 500)에 정지 상태
        tracker.initialize_tracks(
            initial_positions={
                1: (900.0, 700.0),
                2: (1000.0, 700.0),
                3: (2000.0, 500.0),
            },
            cue_ball_id=1,
            ball_colors={1: "WHITE", 2: "RED", 3: "YELLOW"},
        )

        # Frame 1: 수구(ID 1)가 (920, 700)으로 전진
        tracker.update([(920.0, 700.0), (1000.0, 700.0), (2000.0, 500.0)])

        # Frame 2 (충돌 접촉 순간):
        # 수구(ID 1)가 (940, 700)에 도달하여 적구(ID 2, 1000, 700)와 중심 거리 60.0mm (<= 61.5mm)로 충돌!
        # 이때 충돌 법선 벡터 N = (1.0, 0.0) (오른쪽 수평 방향)이 계산되어 물리 가드가 활성화됨
        tracker.update([(940.0, 700.0), (1000.0, 700.0), (2000.0, 500.0)])

        # Frame 3 (충돌 직후 - 최단 거리 매칭 ID 스왑 발생 시나리오):
        # 수구(ID 1)가 (940, 700)에서 적구(ID 2, 1000, 700)와 충돌한 직후:
        # - 실제 수구(ID 1)는 당점/스핀에 의해 약간 위쪽 측면인 (960, 745)로 꺾여 감속되고,
        # - 정지해 있던 적구(ID 2)는 충돌 법선 N=(1.0, 0.0) 방향인 (985, 700)으로 튕겨 나감!
        # 이때 최단 거리 매칭기는:
        #   * 정상 매칭 거리 합: dist((940,700)->(960,745))[49.2] + dist((1000,700)->(985,700))[15.0] = 64.2mm
        #   * 스왑 매칭 거리 합: dist((940,700)->(985,700))[45.0] + dist((1000,700)->(960,745))[59.3] = 104.3mm (만약 수구가 (940,700)에서 (998,700)으로, 적구가 (945,745)로 가는 게 아니라...)
        # 최단 거리 매칭기가 스왑을 선택하는 정확한 기하 조건:
        #   수구 직전 위치 A=(940, 700), 적구 직전 위치 B=(1000, 700)일 때,
        #   충돌 후 적구(실제 ID 2)가 법선 방향 N=(1,0)인 P_fast=(965, 700)에 위치하고,
        #   수구(실제 ID 1)가 관성으로 비스듬히 밀려 P_deflect=(1010, 735)에 위치한 경우:
        #   - 스왑 매칭 거리 합: dist(A, P_fast)[25.0] + dist(B, P_deflect)[36.4] = 61.4mm
        #   - 실제 매칭 거리 합: dist(A, P_deflect)[78.2] + dist(B, P_fast)[35.0] = 113.2mm
        #   따라서 최단 거리 매칭기는 무조건 ID 1을 P_fast(965, 700)에, ID 2를 P_deflect(1010, 735)에 잘못 매칭(ID 스왑)함!
        #   이때 스왑된 ID 1의 속도 벡터는 (965-940, 700-700) = (25, 0)으로 법선 벡터 N=(1, 0)과 코사인 유사도 1.0 (>= 0.85)이 되어
        #   물리 충돌 가드가 즉시 스왑을 감지하고 ID 1과 ID 2의 위치/속도를 강제 원복함!
        states = tracker.update([(965.0, 700.0), (1010.0, 735.0), (2000.0, 500.0)])

        # 검증 1: 물리 충돌 가드 원복 카운트가 1회 증가했는지 확인
        self.assertGreaterEqual(tracker.swap_correction_count, 1)

        # 검증 2: 강제 원복 후 수구(ID 1)는 (1010, 735)에 있고, 법선 방향으로 튕겨 나간 적구(ID 2)가 (965, 700)을 보유하는지 확인
        cue_pos = states[1].position
        red_pos = states[2].position
        self.assertAlmostEqual(float(cue_pos[0]), 1010.0, places=1)
        self.assertAlmostEqual(float(cue_pos[1]), 735.0, places=1)
        self.assertAlmostEqual(float(red_pos[0]), 965.0, places=1)
        self.assertAlmostEqual(float(red_pos[1]), 700.0, places=1)

    def test_spin_detector_15_frames_confirm(self) -> None:
        """
        위치 이동은 멈췄으나 제자리 스핀(도트 회전 에너지 >= 2.0) 중일 때는 정지로 보지 않고,
        선속도 < 2.5 및 스핀 에너지 < 2.0 상태가 정확히 15프레임 유지될 때 완전 정지 판정하는지 검증
        """
        tracker = PhysicsTracker()
        tracker.initialize_tracks(
            initial_positions={1: (500.0, 500.0), 2: (1000.0, 500.0), 3: (1500.0, 500.0)},
            cue_ball_id=1,
        )
        spin_detector = SpinDetector()

        base_patch = np.full((40, 40), 200, dtype=np.uint8)
        spin_patch_a = base_patch.copy()
        cv2.circle(spin_patch_a, (12, 20), 5, 30, -1)  # 왼쪽 도트
        spin_patch_b = base_patch.copy()
        cv2.circle(spin_patch_b, (28, 20), 5, 30, -1)  # 오른쪽 도트 (회전 모사)

        # 1. 제자리 스핀 구간 (위치는 고정되어 선속도=0 이지만 도트가 번갈아 바뀌어 스핀 에너지 > 2.0)
        for i in range(6):
            states = tracker.update([(500.0, 500.0), (1000.0, 500.0), (1500.0, 500.0)])
            active_patch = spin_patch_a if i % 2 == 0 else spin_patch_b
            res = spin_detector.update(
                states,
                ball_crops_gray={1: active_patch, 2: base_patch, 3: base_patch},
            )
            if i > 0:
                self.assertFalse(res["is_fully_stopped"])
                self.assertEqual(res["consecutive_stop_frames"], 0)

        # 2. 스핀이 멈춘 구간: 동일한 도트 패치가 입력되어 스핀 에너지 = 0.0, 선속도 = 0.0
        # 14프레임까지는 아직 is_fully_stopped == False 여야 함
        for frame_idx in range(1, settings.STOP_CONFIRM_FRAMES):
            states = tracker.update([(500.0, 500.0), (1000.0, 500.0), (1500.0, 500.0)])
            res = spin_detector.update(
                states,
                ball_crops_gray={1: spin_patch_b, 2: base_patch, 3: base_patch},
            )
            self.assertEqual(res["consecutive_stop_frames"], frame_idx)
            self.assertFalse(res["is_fully_stopped"])

        # 정확히 15프레임째(0.5초)에 완전 정지(is_fully_stopped = True) 확정!
        states = tracker.update([(500.0, 500.0), (1000.0, 500.0), (1500.0, 500.0)])
        final_res = spin_detector.update(
            states,
            ball_crops_gray={1: spin_patch_b, 2: base_patch, 3: base_patch},
        )
        self.assertEqual(
            final_res["consecutive_stop_frames"], settings.STOP_CONFIRM_FRAMES
        )
        self.assertTrue(final_res["is_fully_stopped"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
