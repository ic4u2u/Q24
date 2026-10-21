"""
좌표 휴리스틱의 한계를 극복하는 4대 고성능 비전 판독 모듈 (vision_precision_upgraders.py)

1. RayPlane3DProjector: 2D 호모그래피의 시차(Parallax) 오차를 제거하는 solvePnP 기반 3D 구심 평면(z=30.75mm) 역투영기
2. SubPixelPhaseImpactDetector: cv2.phaseCorrelate 푸리에 위상 상관법 기반 0.05픽셀 초미세 스침/키스 감지기
3. MotionStreakVApexDetector: 고속 스트로크 모션 블러 잔상의 프레임 내 'V자 꺾임 꼭짓점(Apex)' 및 굴절각 검출기
4. CushionRimPixelTrigger: 좌표 오차에 영향받지 않는 4면 쿠션 고무 날(Nose) 경계선 픽셀 변화 & 그림자 접촉 감지기
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

TABLE_W = 2840.0
TABLE_H = 1420.0
BALL_R = 30.75
CUSHION_NOSE_Z = 37.0  # 국제식 대대 쿠션 고무 날 높이 (mm)


@dataclass
class SubPixelImpactEvidence:
    """푸리에 위상 상관법 기반 초미세 접촉 판독 결과 데이터 클래스"""

    is_impacted: bool
    phase_shift_px: float
    shift_vector_px: Tuple[float, float]
    confidence: float
    diff_energy: float = 0.0


@dataclass
class StreakApexEvidence:
    """단일 프레임 내 모션 블러 잔상 V자 꺾임 검출 결과 데이터 클래스"""

    is_bent_within_frame: bool
    apex_offset_px: Tuple[float, float]
    intra_frame_bend_deg: float
    aspect_ratio: float


@dataclass
class CushionRimEvidence:
    """4면 쿠션 고무 날(Nose) 경계선 픽셀 밝기 변화 및 그림자 접촉 결과 데이터 클래스"""

    is_rail_touched: bool
    temporal_diff_score: float
    shadow_contact_ratio: float
    peak_offset_px: int


class RayPlane3DProjector:
    """
    2D 호모그래피의 시차(Parallax) 왜곡을 제거하기 위해,
    solvePnP로 구한 3D 카메라 행렬에서 광선(Ray)을 쏘아
    당구공 중심 평면(z = 30.75mm)과의 정확한 3D 교점을 계산합니다.
    """

    def __init__(
        self,
        src_corners_px: List[Tuple[float, float]],
        image_size: Tuple[int, int] = (1920, 1080),
        focal_length_px: Optional[float] = None,
        camera_matrix: Optional[np.ndarray] = None,
    ):
        w, h = image_size
        if camera_matrix is not None:
            self.K = np.array(camera_matrix, dtype=np.float64)
        else:
            fx = focal_length_px or (w * 1.1)
            fy = fx
            cx, cy = w * 0.5, h * 0.5
            self.K = np.array(
                [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        self.K_inv = np.linalg.inv(self.K)

        # 사용자가 찍은 당구대 내측 코너 4점은 실제로는 쿠션 고무 날 높이(z = 37.0mm)에 위치함
        self.world_corners_3d = np.array(
            [
                [0.0, 0.0, CUSHION_NOSE_Z],
                [TABLE_W, 0.0, CUSHION_NOSE_Z],
                [TABLE_W, TABLE_H, CUSHION_NOSE_Z],
                [0.0, TABLE_H, CUSHION_NOSE_Z],
            ],
            dtype=np.float64,
        )

        img_pts = np.array(src_corners_px, dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(
            self.world_corners_3d,
            img_pts,
            self.K,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            success, rvec, tvec = cv2.solvePnP(
                self.world_corners_3d,
                img_pts,
                self.K,
                None,
                flags=cv2.SOLVEPNP_IPPE,
            )
        if not success:
            raise RuntimeError("3D 카메라 자세(solvePnP) 추정에 실패했습니다.")

        self.rvec = rvec
        self.R, _ = cv2.Rodrigues(rvec)
        self.R_inv = self.R.T
        self.tvec = tvec.reshape(3, 1)
        # 월드 좌표계 기준 카메라 렌즈 중심 C = -R^T * t
        self.cam_center_world = (-self.R_inv @ self.tvec).flatten()

        # 비교 검증용 기존 2D 평면 호모그래피 행렬 (고도 차이를 무시한 단순 2D 투영)
        dst_2d = np.array(
            [
                [0.0, 0.0],
                [TABLE_W, 0.0],
                [TABLE_W, TABLE_H],
                [0.0, TABLE_H],
            ],
            dtype=np.float64,
        )
        self.homography_2d = cv2.getPerspectiveTransform(
            img_pts.astype(np.float32), dst_2d.astype(np.float32)
        )

    def pixel_to_ball_center_3d(
        self, u: float, v: float, target_z_mm: float = BALL_R
    ) -> Tuple[float, float]:
        """
        픽셀 (u, v)를 지나는 3D 카메라 광선과 z = 30.75mm(공 반지름 높이) 평면의 교점을 반환합니다.
        """
        ray_cam = self.K_inv @ np.array([u, v, 1.0], dtype=np.float64)
        ray_world = self.R_inv @ ray_cam

        dz = float(ray_world[2])
        if abs(dz) < 1e-8:
            dz = -1e-8

        # C_z + s * d_z = target_z_mm  =>  s = (target_z_mm - C_z) / d_z
        s = (target_z_mm - float(self.cam_center_world[2])) / dz
        intersect_pt = self.cam_center_world + s * ray_world
        return float(intersect_pt[0]), float(intersect_pt[1])

    def pixel_to_plane_2d_homography(self, u: float, v: float) -> Tuple[float, float]:
        """
        [비교용] 기존 단순 2D 호모그래피로 변환했을 때의 (X, Y) 좌표를 반환합니다.
        쿠션 코너(z=37.0mm)와 당구공 구심(z=30.75mm)의 높이 차이로 인해 시차(Parallax) 오차가 포함됩니다.
        """
        pt = np.array([[[u, v]]], dtype=np.float64)
        warped = cv2.perspectiveTransform(pt, self.homography_2d)
        return float(warped[0, 0, 0]), float(warped[0, 0, 1])

    def project_world_3d_to_pixel(
        self, x_mm: float, y_mm: float, z_mm: float = BALL_R
    ) -> Tuple[float, float]:
        """3D 물리 좌표 (X, Y, Z)를 카메라 이미지 픽셀 (u, v)로 정투영합니다."""
        pts_3d = np.array([[x_mm, y_mm, z_mm]], dtype=np.float64)
        img_pts, _ = cv2.projectPoints(pts_3d, self.rvec, self.tvec, self.K, None)
        return float(img_pts[0, 0, 0]), float(img_pts[0, 0, 1])


class SubPixelPhaseImpactDetector:
    """
    푸리에 위상 상관법(cv2.phaseCorrelate)을 사용하여 바운딩 박스로는 절대 잡을 수 없는
    제2적구의 0.05픽셀 미세 진동 및 표면 스침 충격을 감지합니다.
    """

    def __init__(
        self,
        patch_size: int = 48,
        min_shift_px: float = 0.06,
        max_valid_shift_px: float = 12.0,
    ):
        self.patch_size = patch_size
        self.min_shift_px = min_shift_px
        self.max_valid_shift_px = max_valid_shift_px
        self.hann_win = cv2.createHanningWindow((patch_size, patch_size), cv2.CV_64F)

        # 원형 마스크 (배경이나 지나가는 수구 테두리가 위상 계산에 섞이지 않도록 목적구 중심부만 마스킹)
        yy, xx = np.ogrid[:patch_size, :patch_size]
        center = (patch_size - 1) * 0.5
        radius = patch_size * 0.36
        self.inner_circle_mask = (
            (xx - center) ** 2 + (yy - center) ** 2 <= radius**2
        ).astype(np.float64)
        self.window_mask = self.hann_win * self.inner_circle_mask

    def detect_subpixel_impulse(
        self,
        prev_gray_roi: np.ndarray,
        curr_gray_roi: np.ndarray,
        expected_push_dir: Optional[Tuple[float, float]] = None,
    ) -> SubPixelImpactEvidence:
        """
        푸리에 위상 상관법(cv2.phaseCorrelate) 및 반구형 안전 마스킹(Safe-Hemisphere Masking)을 통해
        수구 옆그림자/헤일로 간섭을 배제하고 제2적구 자체의 0.05픽셀 미세 충격만을 추출합니다.
        - expected_push_dir (nx, ny): 수구 중심에서 목적구 중심을 향하는 충돌 법선 단위벡터.
          주어질 경우 수구가 스쳐 지나가는 쪽(-expected_push_dir) 가장자리를 차폐하고,
          추출된 서브픽셀 이동 방향이 밀려나는 방향(expected_push_dir)과 부합하는지 교차 검증합니다.
        """
        if prev_gray_roi.shape != (self.patch_size, self.patch_size):
            prev_gray_roi = cv2.resize(
                prev_gray_roi, (self.patch_size, self.patch_size)
            )
        if curr_gray_roi.shape != (self.patch_size, self.patch_size):
            curr_gray_roi = cv2.resize(
                curr_gray_roi, (self.patch_size, self.patch_size)
            )

        raw1 = prev_gray_roi.astype(np.float64)
        raw2 = curr_gray_roi.astype(np.float64)

        active_window_mask = self.window_mask
        active_inner_mask = self.inner_circle_mask
        push_unit: Optional[Tuple[float, float]] = None

        if expected_push_dir is not None:
            p_norm = float(np.hypot(expected_push_dir[0], expected_push_dir[1]))
            if p_norm > 1e-6:
                nx, ny = float(expected_push_dir[0] / p_norm), float(expected_push_dir[1] / p_norm)
                push_unit = (nx, ny)
                yy, xx = np.ogrid[: self.patch_size, : self.patch_size]
                center = (self.patch_size - 1) * 0.5
                proj = (xx - center) * nx + (yy - center) * ny
                # 수구가 접근하는 쪽(-0.12 * patch_size 이하 영역)의 그림자/모션블러 유입 원천 차폐
                hemi_gate = np.clip((proj + 0.12 * self.patch_size) / (0.20 * self.patch_size), 0.0, 1.0)
                active_inner_mask = self.inner_circle_mask * hemi_gate
                active_window_mask = self.hann_win * active_inner_mask

        # 1) 저주파 구체 조명(DC Dome)이 고정 윈도우 경계와 곱해져 위상 이동량을 0으로 끌어당기는 현상을 막기 위해
        #    고주파 표면 성분(High-Pass Texture/Spot)을 추출한 뒤 해닝·반구형 윈도우 적용 및 cv2.phaseCorrelate 수행
        hp1 = raw1 - cv2.GaussianBlur(raw1, (9, 9), 2.0)
        hp2 = raw2 - cv2.GaussianBlur(raw2, (9, 9), 2.0)

        f1 = hp1 * active_window_mask
        f2 = hp2 * active_window_mask

        (dx_pc, dy_pc), response = cv2.phaseCorrelate(f1, f2)

        # 2) 1픽셀 미만의 초미세 진동(0.05~0.5px) 구간에서 고정 윈도우 감쇠(Window Bias)를 보정하는
        #    원형 마스크 내부 고주파 광류(Sub-Pixel Gradient Flow) 정밀 추정 결합
        gx = cv2.Scharr(raw1, cv2.CV_64F, 1, 0) / 32.0
        gy = cv2.Scharr(raw1, cv2.CV_64F, 0, 1) / 32.0
        it = raw2 - raw1

        w = active_window_mask
        ix_w = (gx * w).ravel()
        iy_w = (gy * w).ravel()
        it_w = (it * w).ravel()
        A = np.column_stack([ix_w, iy_w])
        ata = A.T @ A
        if np.linalg.det(ata) > 1e-3:
            flow_vec = -np.linalg.solve(ata + np.eye(2) * 1e-4, A.T @ it_w)
            dx_flow, dy_flow = float(flow_vec[0]), float(flow_vec[1])
        else:
            dx_flow, dy_flow = 0.0, 0.0

        pc_mag = float(np.hypot(dx_pc, dy_pc))
        flow_mag = float(np.hypot(dx_flow, dy_flow))
        if flow_mag > pc_mag and flow_mag <= 1.5 and response > 0.15:
            dx, dy = dx_flow, dy_flow
            shift_mag = flow_mag
        else:
            dx, dy = float(dx_pc), float(dy_pc)
            shift_mag = pc_mag

        # 내부 픽셀 밝기 변화(회전으로 인한 하이라이트/적구 점 이동) 에너지 보조 측정
        mask_sum = max(float(np.sum(active_inner_mask)), 1.0)
        diff_energy = float(np.sum(np.abs((raw2 - raw1) * active_inner_mask)) / mask_sum)

        # 법선 방향 정합성 검증: expected_push_dir가 주어진 경우 반대 방향(수구 쪽으로 끌려감) 위상 노이즈 차단
        dir_ok = True
        if push_unit is not None and shift_mag > 1e-5:
            cos_dir = (dx * push_unit[0] + dy * push_unit[1]) / shift_mag
            dir_ok = cos_dir >= 0.35

        is_hit = dir_ok and (
            (
                self.min_shift_px <= shift_mag <= self.max_valid_shift_px
                and response > 0.15
            )
            or (diff_energy > 2.5 and shift_mag >= self.min_shift_px * 0.5)
        )
        return SubPixelImpactEvidence(
            is_impacted=bool(is_hit),
            phase_shift_px=shift_mag,
            shift_vector_px=(float(dx), float(dy)),
            confidence=float(response),
            diff_energy=diff_energy,
        )


class MotionStreakVApexDetector:
    """
    단일 프레임 노출 시간 동안 공이 쿠션이나 목적구를 맞고 꺾일 때 생성되는
    모션 블러 잔상(Motion Blur Streak)의 'V자 굴절 꼭짓점'을 직접 검출합니다.
    """

    @staticmethod
    def extract_streak_apex(binary_streak_mask: np.ndarray) -> StreakApexEvidence:
        ys, xs = np.where(binary_streak_mask > 0)
        if len(xs) < 15:
            return StreakApexEvidence(False, (0.0, 0.0), 0.0, 1.0)

        pts = np.column_stack([xs, ys]).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        w, h = rect[1]
        if min(w, h) < 1e-3:
            return StreakApexEvidence(False, (0.0, 0.0), 0.0, 1.0)

        aspect_ratio = float(max(w, h) / min(w, h))
        # 잔상이 길게 늘어난 고속 이동 프레임(aspect_ratio >= 1.6)에서 꺾임 분석
        if aspect_ratio < 1.6:
            return StreakApexEvidence(False, (0.0, 0.0), 0.0, aspect_ratio)

        # 잔상의 양 끝점(가장 먼 두 점) 찾기
        mean_pt = np.mean(pts, axis=0)
        cov = np.cov((pts - mean_pt).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_axis = eigvecs[:, np.argmax(eigvals)]

        projections = (pts - mean_pt) @ major_axis
        p_start = pts[np.argmin(projections)]
        p_end = pts[np.argmax(projections)]

        # 양 끝점을 잇는 직선으로부터 가장 멀리 벗어난 잔상 중심선 점(V자 굴절 꼭짓점 후보) 탐색
        line_vec = p_end - p_start
        line_len = float(np.linalg.norm(line_vec))
        if line_len < 8.0:
            return StreakApexEvidence(False, (0.0, 0.0), 0.0, aspect_ratio)

        line_unit = line_vec / line_len
        normal_unit = np.array([-line_unit[1], line_unit[0]], dtype=np.float32)

        perp_dists = np.abs((pts - p_start) @ normal_unit)
        max_idx = int(np.argmax(perp_dists))
        p_apex = pts[max_idx]
        max_deviation_px = float(perp_dists[max_idx])

        # 꼭짓점을 기준으로 전반부 벡터(start -> apex)와 후반부 벡터(apex -> end)의 꺾임 각도 계산
        v1 = p_apex - p_start
        v2 = p_end - p_apex
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if n1 < 3.0 or n2 < 3.0:
            return StreakApexEvidence(False, (0.0, 0.0), 0.0, aspect_ratio)

        cos_ang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        bend_deg = float(np.degrees(np.arccos(cos_ang)))

        is_bent = (max_deviation_px >= min(w, h) * 0.45) and (bend_deg >= 12.0)
        return StreakApexEvidence(
            is_bent_within_frame=bool(is_bent),
            apex_offset_px=(float(p_apex[0]), float(p_apex[1])),
            intra_frame_bend_deg=bend_deg,
            aspect_ratio=aspect_ratio,
        )


class CushionRimPixelTrigger:
    """
    좌표 변환 오차 때문에 쿠션 앞 10mm에서 꺾인 공을 쿠션 미접촉으로 놓치거나 반대로 오판하는 문제를 막기 위해,
    4면 쿠션 고무 날(Nose) 경계선 영역의 프레임 간 픽셀 밝기 변화량(Temporal Frame Difference)과
    수구 그림자 접촉 여부를 직접 확인합니다.
    """

    def __init__(
        self,
        diff_threshold: float = 12.0,
        shadow_dark_th: int = 55,
        min_contact_ratio: float = 0.08,
    ):
        self.diff_threshold = diff_threshold
        self.shadow_dark_th = shadow_dark_th
        self.min_contact_ratio = min_contact_ratio

    def verify_rail_touch_by_pixels(
        self, prev_rail_strip: np.ndarray, curr_rail_strip: np.ndarray
    ) -> CushionRimEvidence:
        """
        쿠션 고무 날 경계선 스트립(prev_rail_strip, curr_rail_strip)의 픽셀 차분 에너지와
        수구-쿠션 접점 그림자 비율을 분석하여 실제 쿠션 접촉 여부를 판정합니다.
        """
        if prev_rail_strip is None or curr_rail_strip is None:
            return CushionRimEvidence(False, 0.0, 0.0, 0)
        if prev_rail_strip.size == 0 or curr_rail_strip.size == 0:
            return CushionRimEvidence(False, 0.0, 0.0, 0)

        if prev_rail_strip.ndim == 3:
            prev_gray = cv2.cvtColor(prev_rail_strip, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_rail_strip

        if curr_rail_strip.ndim == 3:
            curr_gray = cv2.cvtColor(curr_rail_strip, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_rail_strip

        if prev_gray.shape != curr_gray.shape:
            curr_gray = cv2.resize(
                curr_gray, (prev_gray.shape[1], prev_gray.shape[0])
            )

        abs_diff = np.abs(curr_gray.astype(np.float64) - prev_gray.astype(np.float64))
        col_profile = np.mean(abs_diff, axis=0)
        peak_offset_px = int(np.argmax(col_profile)) if col_profile.size > 0 else 0
        temporal_diff_score = float(np.percentile(abs_diff, 90))

        # 활성 변화 구간 주변의 그림자(어두운 픽셀) 또는 밝은 당구공 경계 접촉 비율 계산
        active_mask = abs_diff >= self.diff_threshold
        active_ratio = float(np.mean(active_mask))
        shadow_or_ball_mask = (curr_gray <= self.shadow_dark_th) | (curr_gray >= 185)
        contact_ratio = float(np.mean(active_mask & shadow_or_ball_mask))

        is_touched = (
            temporal_diff_score >= self.diff_threshold
            and (
                contact_ratio >= self.min_contact_ratio
                or active_ratio >= self.min_contact_ratio * 1.5
            )
        )
        return CushionRimEvidence(
            is_rail_touched=bool(is_touched),
            temporal_diff_score=temporal_diff_score,
            shadow_contact_ratio=max(contact_ratio, active_ratio),
            peak_offset_px=peak_offset_px,
        )
