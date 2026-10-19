"""
Module A: 당구대 내측 코너 4점(및 다이아몬드 포인트 다점 확장) 기반 2D 평면 변환 + 대각선 카메라 시차(Parallax) 보정 모듈
파일 경로: three_cushion_vision/homography.py

1. 국제식 대대 내측 쿠션 규격(WIDTH = 2840.0mm, HEIGHT = 1420.0mm, BALL_RADIUS = 30.75mm) 기준 좌표계 사용
2. 4점 입력 시 cv2.getPerspectiveTransform, 5점 이상(다이아몬드 포인트 포함) 입력 시 cv2.findHomography 자동 지원
3. Parallax Offset 보정: 대각선 상단 카메라에서 바운딩 박스 높이 h에 대해 cy_corrected = cy + h * camera_tilt_ratio (기본값 0.12) 적용 후 mm 좌표로 변환
"""

from typing import List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from three_cushion_vision.config import DEFAULT_CONFIG, VisionConfig

Point2D = Tuple[float, float]
BBoxXYXY = Tuple[float, float, float, float]


class TableHomography:
    """
    카메라 픽셀 좌표를 국제식 대대 2D 물리 평면(2840.0mm x 1420.0mm)으로 변환하고,
    대각선 카메라의 3D 당구공 구면 시차(Parallax Offset)를 보정하는 호모그래피 변환기 클래스입니다.
    """

    WIDTH: float = DEFAULT_CONFIG.TABLE_WIDTH_MM
    HEIGHT: float = DEFAULT_CONFIG.TABLE_HEIGHT_MM
    BALL_RADIUS: float = DEFAULT_CONFIG.BALL_RADIUS_MM

    def __init__(
        self,
        src_points_px: Optional[Union[Sequence[Point2D], np.ndarray]] = None,
        dst_points_mm: Optional[Union[Sequence[Point2D], np.ndarray]] = None,
        camera_tilt_ratio: float = DEFAULT_CONFIG.CAMERA_TILT_RATIO,
        config: VisionConfig = DEFAULT_CONFIG,
    ) -> None:
        """
        TableHomography 초기화

        Args:
            src_points_px: 카메라 영상 내 당구대 내측 코너 4점(좌상, 우상, 우하, 좌하) 또는 다이아몬드 포인트 포함 다점 픽셀 좌표.
                           생략 시 (0,0)~(2840,1420) 1:1 항등 좌표계로 초기화됩니다.
            dst_points_mm: 목표 물리 평면 좌표 (mm). 생략 시 국제식 대대 4코너 표준 규격이 자동 적용됩니다.
            camera_tilt_ratio: 대각선 카메라 시차(Parallax) 보정 계수 (기본값: 0.12)
            config: VisionConfig 설정 객체
        """
        self.config: VisionConfig = config
        self.width_mm: float = config.TABLE_WIDTH_MM
        self.height_mm: float = config.TABLE_HEIGHT_MM
        self.ball_radius_mm: float = config.BALL_RADIUS_MM
        self.camera_tilt_ratio: float = float(camera_tilt_ratio)

        default_dst = np.array(
            [
                [0.0, 0.0],
                [self.width_mm, 0.0],
                [self.width_mm, self.height_mm],
                [0.0, self.height_mm],
            ],
            dtype=np.float32,
        )
        if src_points_px is None:
            src_arr = default_dst.copy()
        else:
            src_arr = np.asarray(src_points_px, dtype=np.float32)

        if dst_points_mm is None:
            dst_arr = default_dst
        else:
            dst_arr = np.asarray(dst_points_mm, dtype=np.float32)

        self.src_points_px: np.ndarray = src_arr
        self.dst_points_mm: np.ndarray = dst_arr
        self.H: np.ndarray = np.eye(3, dtype=np.float64)
        self.H_inv: np.ndarray = np.eye(3, dtype=np.float64)
        self._compute_homography(self.src_points_px, self.dst_points_mm)

    def _compute_homography(
        self, src_pts: np.ndarray, dst_pts: np.ndarray
    ) -> None:
        """
        입력 점 개수에 따라 4점은 cv2.getPerspectiveTransform,
        5점 이상(다이아몬드 포인트 확장)은 cv2.findHomography(RANSAC)로 변환 행렬 H를 계산합니다.
        """
        if src_pts.ndim != 2 or src_pts.shape[1] != 2 or src_pts.shape[0] < 4:
            raise ValueError(
                f"src_points_px는 최소 4개 이상의 (x, y) 좌표여야 합니다. 현재 형태: {src_pts.shape}"
            )
        if dst_pts.shape != src_pts.shape:
            raise ValueError(
                f"src_points_px({src_pts.shape})와 dst_points_mm({dst_pts.shape})의 크기가 일치해야 합니다."
            )

        if src_pts.shape[0] == 4:
            self.H = cv2.getPerspectiveTransform(
                src_pts.astype(np.float32), dst_pts.astype(np.float32)
            )
            self.H_inv = cv2.getPerspectiveTransform(
                dst_pts.astype(np.float32), src_pts.astype(np.float32)
            )
        else:
            h_mat, _ = cv2.findHomography(
                src_pts.astype(np.float32),
                dst_pts.astype(np.float32),
                method=cv2.RANSAC,
                ransacReprojThreshold=2.0,
            )
            if h_mat is None:
                raise RuntimeError("cv2.findHomography 행렬 계산에 실패했습니다.")
            self.H = h_mat
            self.H_inv = np.linalg.inv(self.H)

    def update_calibration(
        self,
        src_points_px: Union[Sequence[Point2D], np.ndarray],
        dst_points_mm: Optional[Union[Sequence[Point2D], np.ndarray]] = None,
    ) -> None:
        """카메라 캘리브레이션 키포인트를 갱신하고 호모그래피 행렬을 재계산합니다."""
        self.src_points_px = np.asarray(src_points_px, dtype=np.float32)
        if dst_points_mm is not None:
            self.dst_points_mm = np.asarray(dst_points_mm, dtype=np.float32)
        self._compute_homography(self.src_points_px, self.dst_points_mm)

    def pixel_to_table_mm(self, x_px: float, y_px: float) -> Point2D:
        """단일 픽셀 좌표 (x_px, y_px)를 2840x1420mm 당구대 물리 평면 좌표로 변환합니다."""
        pt = np.array([[[float(x_px), float(y_px)]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.H)[0][0]
        return float(mapped[0]), float(mapped[1])

    def bbox_to_table_mm(
        self,
        bbox_xyxy: BBoxXYXY,
        camera_tilt_ratio: Optional[float] = None,
    ) -> Point2D:
        """
        [Parallax Offset 보정 투영]
        대각선 상단 카메라에서 구(Sphere)의 바운딩 박스 중심 (cx, cy)는 당구대 바닥 접점과 어긋납니다.
        바운딩 박스 높이 h = (y2 - y1)에 대해 cy_corrected = cy + h * camera_tilt_ratio (기본값 0.12)를
        적용한 뒤 국제식 대대 물리 좌표(mm)로 변환합니다.
        """
        x1, y1, x2, y2 = bbox_xyxy
        cx = (float(x1) + float(x2)) * 0.5
        cy = (float(y1) + float(y2)) * 0.5
        h = max(0.0, float(y2) - float(y1))
        tilt = (
            self.camera_tilt_ratio
            if camera_tilt_ratio is None
            else float(camera_tilt_ratio)
        )
        cy_corrected = cy + h * tilt
        return self.pixel_to_table_mm(cx, cy_corrected)

    def table_mm_to_pixel(self, x_mm: float, y_mm: float) -> Point2D:
        """당구대 물리 평면 좌표 (x_mm, y_mm)를 원본 카메라 픽셀 좌표로 역변환합니다."""
        pt = np.array([[[float(x_mm), float(y_mm)]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.H_inv)[0][0]
        return float(mapped[0]), float(mapped[1])

    def warp_frame(self, bgr_frame: np.ndarray) -> np.ndarray:
        """원본 카메라 BGR 영상을 (1420, 2840, 3) 크기의 1px=1mm 탑뷰 평면 영상으로 워프합니다."""
        return cv2.warpPerspective(
            bgr_frame,
            self.H,
            dsize=(int(round(self.width_mm)), int(round(self.height_mm))),
            flags=cv2.INTER_LINEAR,
        )
