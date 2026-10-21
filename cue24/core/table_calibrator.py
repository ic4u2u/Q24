"""
호모그래피(Homography) 기반 당구대 평면 보정 모듈
파일 경로: cue24/core/table_calibrator.py

카메라 원본 화각에서 찍힌 비스듬한 당구대 4개 모서리 픽셀 좌표를 입력받아,
국제식 대대 표준 내경 규격(2840mm x 1420mm, 1px = 1mm)의 평면으로 정규화합니다.
"""

from typing import List, Optional, Tuple, Union
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
        camera_tilt_ratio: float = 0.0,
    ) -> None:
        """
        TableCalibrator 초기화

        Args:
            src_corners: 카메라 원본 영상에서 선택한 당구대 내경 4개 모서리 픽셀 좌표
                         순서: [좌상(Top-Left), 우상(Top-Right), 우하(Bottom-Right), 좌하(Bottom-Left)]
            width_mm: 보정될 목표 평면의 가로 길이 (기본값: 2840.0 mm)
            height_mm: 보정될 목표 평면의 세로 길이 (기본값: 1420.0 mm)
            camera_tilt_ratio: 대각선 카메라 시차(Parallax) 보정 비율 (수직 탑뷰=0.0, 대각선 뷰=0.10~0.15)
        """
        self.width_mm: int = int(round(width_mm))
        self.height_mm: int = int(round(height_mm))
        self.camera_tilt_ratio: float = float(camera_tilt_ratio)

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

    def auto_detect_table_corners(self, frame: np.ndarray) -> List[Point2D]:
        """
        카메라(또는 카메라 대용 영상) 프레임에서 파란색 당구대 라사지 영역을 비전으로 검출하고,
        고무 쿠션 레일 폭을 제외한 실제 당구대 내경(쿠션 코 끝 4모서리) 좌표를 자동으로 찾아 보정 행렬을 갱신합니다.
        """
        if frame is None or frame.size == 0:
            raise ValueError("유효하지 않은 빈 프레임이 입력되었습니다.")

        h_img, w_img = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, np.array([92, 70, 50], dtype=np.uint8), np.array([135, 255, 255], dtype=np.uint8))

        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            corners: List[Point2D] = [
                (0.0, 0.0),
                (float(w_img), 0.0),
                (float(w_img), float(h_img)),
                (0.0, float(h_img)),
            ]
            self.update_corners(corners)
            return corners

        largest_cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest_cnt))
        if area < (w_img * h_img * 0.20):
            corners = [
                (0.0, 0.0),
                (float(w_img), 0.0),
                (float(w_img), float(h_img)),
                (0.0, float(h_img)),
            ]
            self.update_corners(corners)
            return corners

        x_box, y_box, w_box, h_box = cv2.boundingRect(largest_cnt)

        # 이미 화면 전체가 내경으로 꽉 찬 합성/데모 영상인 경우 전체 프레임 모서리 사용
        if w_box >= w_img * 0.96 and h_box >= h_img * 0.96:
            corners = [
                (0.0, 0.0),
                (float(w_img), 0.0),
                (float(w_img), float(h_img)),
                (0.0, float(h_img)),
            ]
            self.update_corners(corners)
            return corners

        # 방송 로고(예: 좌상단 파란색 CUESCO 마크 등)가 당구대 테두리에 닿아 외곽 박스가 부풀어 오르는 현상을 방지하기 위해
        # 행별·열별 파란색 픽셀 밀도 프로파일(> 0.35)로 당구대 본체 외곽 사각형을 정밀 산출합니다.
        mask_bool = blue_mask > 0
        col_ratio = mask_bool.mean(axis=0)
        row_ratio = mask_bool.mean(axis=1)
        valid_cols = np.where(col_ratio > 0.35)[0]
        valid_rows = np.where(row_ratio > 0.35)[0]
        if len(valid_cols) >= int(w_img * 0.4) and len(valid_rows) >= int(h_img * 0.4):
            x_box = int(valid_cols[0])
            w_box = int(valid_cols[-1] - valid_cols[0])
            y_box = int(valid_rows[0])
            h_box = int(valid_rows[-1] - valid_rows[0])

        # 실제 탑뷰 카메라 영상: 파란색 외곽 사각형에서 상/하/좌/우 고무 쿠션 턱 두께를 안쪽으로 보정하여 내경 산출
        left_x = float(x_box) + float(w_box) * 0.018
        right_x = float(x_box + w_box) - float(w_box) * 0.018
        top_y = float(y_box) + float(h_box) * 0.042
        bottom_y = float(y_box + h_box) - float(h_box) * 0.039

        corners = [
            (round(left_x, 1), round(top_y, 1)),
            (round(right_x, 1), round(top_y, 1)),
            (round(right_x, 1), round(bottom_y, 1)),
            (round(left_x, 1), round(bottom_y, 1)),
        ]
        self.update_corners(corners)
        return corners

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

    def ball_to_table_mm(
        self,
        point_px: Point2D,
        radius_px: float = 0.0,
        camera_tilt_ratio: Optional[float] = None,
    ) -> Point2D:
        """
        [대각선 카메라 3D 당구공 시차(Parallax) 보정 투영]
        당구공은 3D 구(반지름 30.75mm)이므로 대각선 카메라에서 볼 때 구의 중심(cx, cy)이
        실제 당구대 바닥(라사지) 접점보다 위쪽(멀리)으로 투영됩니다.
        공의 픽셀 반지름(radius_px)과 카메라 기울기 비율(camera_tilt_ratio)을 반영하여
        실제 바닥 접점(cy_contact = cy + radius_px * tilt)을 2840 x 1420 mm 물리 평면 좌표로 변환합니다.
        """
        tilt = (
            self.camera_tilt_ratio
            if camera_tilt_ratio is None
            else float(camera_tilt_ratio)
        )
        cx_px = float(point_px[0])
        cy_contact_px = float(point_px[1]) + float(radius_px) * tilt
        return self.pixel_to_mm((cx_px, cy_contact_px))

    def bbox_to_table_mm(
        self,
        bbox_xyxy: Tuple[float, float, float, float],
        camera_tilt_ratio: Optional[float] = None,
    ) -> Point2D:
        """
        객체 검출기(YOLO/RF-DETR 등)의 바운딩 박스(x1, y1, x2, y2)를 입력받아
        대각선 시차(Parallax)를 보정한 실제 당구대 바닥 접점(mm) 좌표로 변환합니다.
        """
        x1, y1, x2, y2 = bbox_xyxy
        cx = (float(x1) + float(x2)) / 2.0
        cy = (float(y1) + float(y2)) / 2.0
        r_px = (float(y2) - float(y1)) / 2.0
        return self.ball_to_table_mm((cx, cy), radius_px=r_px, camera_tilt_ratio=camera_tilt_ratio)

    def mm_to_pixel(self, point_mm: Point2D) -> Point2D:
        """
        2840 x 1420 mm 물리 평면 좌표를 원본 카메라 화면의 픽셀 좌표로 역변환합니다.
        """
        pt = np.array([[[point_mm[0], point_mm[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.inverse_homography_matrix)
        x_px, y_px = float(transformed[0, 0, 0]), float(transformed[0, 0, 1])
        return (x_px, y_px)
