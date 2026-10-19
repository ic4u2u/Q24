"""
호모그래피(Homography) 기반 당구대 평면 보정 모듈
파일 경로: cue24/core/table_calibrator.py

카메라 원본 화각에서 찍힌 비스듬한 당구대 4개 모서리 픽셀 좌표를 입력받아,
국제식 대대 표준 내경 규격(2840mm x 1420mm, 1px = 1mm)의 평면으로 정규화합니다.
"""

from typing import List, Tuple, Union
import cv2
import numpy as np

from cue24.config.settings import settings

Point2D = Tuple[float, float]


class TableCalibrator:
    """
    카메라 원근 왜곡을 제거하고 실제 당구대 물리 평면(2840 x 1420 mm)으로
    변환하는 호모그래피 보정기 클래스입니다.
    """

    def __init__(
        self,
        src_corners: Union[List[Point2D], np.ndarray],
        width_mm: float = settings.TABLE_WIDTH_MM,
        height_mm: float = settings.TABLE_HEIGHT_MM,
    ) -> None:
        """
        TableCalibrator 초기화

        Args:
            src_corners: 카메라 원본 영상에서 선택한 당구대 내경 4개 모서리 픽셀 좌표
                         순서: [좌상(Top-Left), 우상(Top-Right), 우하(Bottom-Right), 좌하(Bottom-Left)]
            width_mm: 보정될 목표 평면의 가로 길이 (기본값: 2840.0 mm)
            height_mm: 보정될 목표 평면의 세로 길이 (기본값: 1420.0 mm)
        """
        self.width_mm: int = int(round(width_mm))
        self.height_mm: int = int(round(height_mm))

        # 입력 좌표 유효성 검증 및 float32 배열 변환
        self.src_points: np.ndarray = self._validate_and_format_corners(src_corners)

        # 변환 목표가 되는 표준 대대 평면 4개 모서리 좌표 (좌상, 우상, 우하, 좌하)
        # 1 Pixel = 1 mm 정밀 매핑을 위해 (0, 0) ~ (width_mm, height_mm)로 설정
        self.dst_points: np.ndarray = np.array(
            [
                [0.0, 0.0],
                [float(self.width_mm), 0.0],
                [float(self.width_mm), float(self.height_mm)],
                [0.0, float(self.height_mm)],
            ],
            dtype=np.float32,
        )

        # 3x3 원근 변환(Homography) 행렬 및 역변환 행렬 계산
        self.homography_matrix: np.ndarray = cv2.getPerspectiveTransform(
            self.src_points, self.dst_points
        )
        self.inverse_homography_matrix: np.ndarray = cv2.getPerspectiveTransform(
            self.dst_points, self.src_points
        )

    @staticmethod
    def _validate_and_format_corners(
        corners: Union[List[Point2D], np.ndarray]
    ) -> np.ndarray:
        """
        입력된 4개 모서리 좌표의 형태가 (4, 2)인지 검증하고 np.float32로 변환합니다.
        """
        pts = np.asarray(corners, dtype=np.float32)
        if pts.shape != (4, 2):
            raise ValueError(
                f"모서리 좌표는 정확히 4개의 (x, y) 점이어야 합니다. 입력된 형태: {pts.shape}"
            )
        return pts

    def update_corners(self, new_src_corners: Union[List[Point2D], np.ndarray]) -> None:
        """
        카메라 흔들림이나 재보정 요청 시 4개 모서리 좌표를 갱신하고 변환 행렬을 재계산합니다.
        """
        self.src_points = self._validate_and_format_corners(new_src_corners)
        self.homography_matrix = cv2.getPerspectiveTransform(
            self.src_points, self.dst_points
        )
        self.inverse_homography_matrix = cv2.getPerspectiveTransform(
            self.dst_points, self.src_points
        )

    def warp(self, frame: np.ndarray) -> np.ndarray:
        """
        원본 카메라 프레임을 2840 x 1420 밀리미터 표준 평면 영상으로 원근 보정(Warp)합니다.

        Args:
            frame: 카메라에서 입력된 원본 BGR 또는 Grayscale 영상 (np.ndarray)

        Returns:
            warped_frame: (1420, 2840, C) 크기로 정규화된 평면 보정 영상
        """
        if frame is None or frame.size == 0:
            raise ValueError("유효하지 않은 빈 프레임(frame)이 입력되었습니다.")

        warped_frame = cv2.warpPerspective(
            frame,
            self.homography_matrix,
            dsize=(self.width_mm, self.height_mm),
            flags=cv2.INTER_LINEAR,
        )
        return warped_frame

    def pixel_to_mm(self, point_px: Point2D) -> Point2D:
        """
        원본 카메라 화각의 특정 픽셀 좌표 (x, y)를 2840 x 1420 mm 물리 평면 좌표로 변환합니다.
        """
        pt = np.array([[[point_px[0], point_px[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.homography_matrix)
        x_mm, y_mm = float(transformed[0, 0, 0]), float(transformed[0, 0, 1])
        return (x_mm, y_mm)

    def mm_to_pixel(self, point_mm: Point2D) -> Point2D:
        """
        2840 x 1420 mm 물리 평면 좌표를 원본 카메라 화면의 픽셀 좌표로 역변환합니다.
        """
        pt = np.array([[[point_mm[0], point_mm[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.inverse_homography_matrix)
        x_px, y_px = float(transformed[0, 0, 0]), float(transformed[0, 0, 1])
        return (x_px, y_px)
