"""
Step 2 단위 테스트: 호모그래피 평면 보정기(TableCalibrator) 및 색상 확정기(ColorLocker) 검증
파일 경로: cue24/tests/test_step2.py
"""

import unittest
import cv2
import numpy as np

from cue24.config.settings import settings
from cue24.core.color_locker import ColorLocker
from cue24.core.table_calibrator import TableCalibrator


def create_synthetic_ball_patch(
    base_bgr: tuple[int, int, int],
    size: int = 60,
) -> np.ndarray:
    """
    테스트용 가상 당구공 BGR 크롭 패치를 생성합니다.
    - 외곽 테두리 15%: 어두운 파란색 당구대 라사지 그림자 모사
    - 중간 링 영역 (35% ~ 85%): 당구공 고유의 색상 (base_bgr)
    - 상단 중심부 35%: 강한 천장 LED 조명 반사광 (순백색 하이라이트 255, 255, 255) 모사
    """
    # 기본 배경을 어두운 파란색 라사지 그림자(BGR: 80, 30, 10)로 초기화
    patch = np.full((size, size, 3), (80, 30, 10), dtype=np.uint8)
    center = (size // 2, size // 2)
    max_radius = size // 2

    # 1. 공 몸체 영역 (반지름 88%까지 고유 색상 채우기)
    body_radius = int(max_radius * 0.88)
    cv2.circle(patch, center, body_radius, base_bgr, thickness=-1)

    # 2. 중심부 30% 영역에 천장 LED 조명 하이라이트(순백색 반사광) 강제 주입
    # 환형 마스크가 없다면 적구나 황구도 흰색 쪽으로 평균이 크게 왜곡됨
    highlight_radius = int(max_radius * 0.30)
    cv2.circle(patch, center, highlight_radius, (255, 255, 255), thickness=-1)

    return patch


class TestStep2VisionModules(unittest.TestCase):
    """TableCalibrator 및 ColorLocker 단위 테스트 스위트"""

    def test_table_calibrator_warp_and_transform(self) -> None:
        """
        1. 비스듬한 사다리꼴 당구대 영상을 2840 x 1420 평면으로 정확히 펴는지(warp) 검증
        2. 픽셀 <-> 밀리미터(mm) 양방향 좌표 변환 정밀도 검증
        """
        # FHD(1920x1080) 가상 카메라 프레임 생성
        raw_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # 원근 왜곡이 있는 가상의 당구대 4개 모서리 픽셀 좌표 (좌상, 우상, 우하, 좌하)
        src_corners = [
            (320.0, 180.0),   # 좌상 (Top-Left)
            (1600.0, 180.0),  # 우상 (Top-Right)
            (1820.0, 960.0),  # 우하 (Bottom-Right)
            (100.0, 960.0),   # 좌하 (Bottom-Left)
        ]

        # 당구대 영역 내부에 초록색 마커 칠하기
        poly_pts = np.array(src_corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(raw_frame, [poly_pts], (0, 200, 0))

        calibrator = TableCalibrator(src_corners=src_corners)
        warped = calibrator.warp(raw_frame)

        # 검증 1: 결과 영상의 해상도가 정확히 (1420, 2840, 3)인지 확인
        self.assertEqual(
            warped.shape,
            (int(settings.TABLE_HEIGHT_MM), int(settings.TABLE_WIDTH_MM), 3),
        )

        # 검증 2: 네 모서리 픽셀 좌표를 변환했을 때 각각 (0,0), (2840,0), (2840,1420), (0,1420)으로 매핑되는지 확인
        expected_mm_corners = [
            (0.0, 0.0),
            (settings.TABLE_WIDTH_MM, 0.0),
            (settings.TABLE_WIDTH_MM, settings.TABLE_HEIGHT_MM),
            (0.0, settings.TABLE_HEIGHT_MM),
        ]
        for src_pt, expected_mm in zip(src_corners, expected_mm_corners):
            calc_mm = calibrator.pixel_to_mm(src_pt)
            self.assertAlmostEqual(calc_mm[0], expected_mm[0], places=1)
            self.assertAlmostEqual(calc_mm[1], expected_mm[1], places=1)

            # 역변환(mm -> pixel)도 원본 픽셀 좌표로 정확히 돌아오는지 확인
            recovered_px = calibrator.mm_to_pixel(calc_mm)
            self.assertAlmostEqual(recovered_px[0], src_pt[0], places=1)
            self.assertAlmostEqual(recovered_px[1], src_pt[1], places=1)

    def test_color_locker_annular_mask_and_classification(self) -> None:
        """
        상단 중심부 조명 반사광(하이라이트)과 외곽 라사지 그림자가 포함된
        가상 당구공 3개(백구, 황구, 적구)를 100% 정확하게 분류하고 수구를 고정하는지 검증
        """
        locker = ColorLocker()

        # 반사광(중심 백색)과 그림자(외곽 청색)가 포함된 3개 공 패치 생성
        # ID 101: 백구 (BGR: 240, 240, 240)
        # ID 202: 황구 (BGR: 20, 215, 245)
        # ID 303: 적구 (BGR: 25, 25, 230)
        ball_crops = {
            101: create_synthetic_ball_patch((240, 240, 240), size=64),
            202: create_synthetic_ball_patch((20, 215, 245), size=64),
            303: create_synthetic_ball_patch((25, 25, 230), size=64),
        }

        # 1. 개별 및 전체 색상 분류 검증
        classified = locker.classify_balls(ball_crops)
        self.assertEqual(classified[101], "WHITE")
        self.assertEqual(classified[202], "YELLOW")
        self.assertEqual(classified[303], "RED")

        # 2. 플레이어 1 (P1 = WHITE 수구) 공격 턴 수구 Lock 검증
        lock_p1 = locker.lock_cue_ball(ball_crops, active_player="P1")
        self.assertEqual(lock_p1["cue_ball_id"], 101)
        self.assertEqual(lock_p1["cue_ball_color"], "WHITE")
        self.assertEqual(lock_p1["object_ball_ids"], [202, 303])
        self.assertEqual(locker.locked_cue_ball_id, 101)

        # 3. 플레이어 2 (P2 = YELLOW 수구) 공격 턴 수구 Lock 검증
        lock_p2 = locker.lock_cue_ball(ball_crops, active_player="P2")
        self.assertEqual(lock_p2["cue_ball_id"], 202)
        self.assertEqual(lock_p2["cue_ball_color"], "YELLOW")
        self.assertEqual(lock_p2["object_ball_ids"], [101, 303])
        self.assertEqual(locker.locked_cue_ball_id, 202)


if __name__ == "__main__":
    unittest.main(verbosity=2)
