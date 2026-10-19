"""
Micro-Hawk 초정밀 VAR(Video Assistant Referee) 현미경 판독 모듈
파일 경로: cue24/core/var_referee.py

1. [공-공 미세 접촉 판정]: 두 공 간격 < 12mm 진입 시 서브픽셀 간극(Gap = 중심거리 - 61.5mm) 계측 및
   수구 진행 각도 굴절(Deflection Angle > 0.35도) 동시 검증. 간극 <= 0mm 및 굴절 > 0.35도 시 접촉 확정.
2. [쿠션 이중 터치 판정]: 쿠션 표면 거리의 시계열 프로파일에서 90ms 이내에 압축 극솟점(Local Minima)이
   2회 검출될 경우 Double Touch(2쿠션 파울) 판정.
3. 판정 결과를 20인치 태블릿 팝업 렌더링용 표준 딕셔너리(상태, 메시지, 간극 mm, 각도) 형태로 반환.
"""

import math
from typing import Dict, List, Sequence, Tuple
import numpy as np

from cue24.config.settings import settings

Point2D = Tuple[float, float]


class VARReferee:
    """
    육안으로 식별하기 어려운 초미세 공-공 스침(Micro-Contact)과
    쿠션 이중 터치(Double Touch)를 서브픽셀 및 밀리초(ms) 단위로 분석하는 VAR 심판 클래스입니다.
    """

    NEAR_MISS_TRIGGER_GAP_MM: float = 12.0   # VAR 미세 판독 활성화 표면 간격 임계값 (12.0mm)
    MIN_DEFLECTION_ANGLE_DEG: float = 0.35   # 미세 접촉 인정 최소 굴절 각도 (0.35도)
    DOUBLE_TOUCH_WINDOW_MS: float = 90.0     # 쿠션 이중 터치 판정 시간 윈도우 (90.0ms)

    def __init__(
        self,
        ball_diameter_mm: float = settings.BALL_DIAMETER_MM,
        cushion_margin_mm: float = settings.CUSHION_MARGIN_MM,
    ) -> None:
        self.ball_diameter_mm: float = ball_diameter_mm
        self.cushion_margin_mm: float = cushion_margin_mm

    @staticmethod
    def compute_deflection_angle_deg(
        vec_before: Sequence[float], vec_after: Sequence[float]
    ) -> float:
        """
        최근접 통과 전후의 수구 이동 벡터 간 굴절 각도(Deflection Angle, 단위: 도)를 계산합니다.
        """
        v1 = np.asarray(vec_before, dtype=np.float64)
        v2 = np.asarray(vec_after, dtype=np.float64)
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0

        cos_theta = float(np.dot(v1, v2) / (n1 * n2))
        # 부동소수점 오차 방지를 위한 클리핑 (-1.0 ~ 1.0)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle_rad = math.acos(cos_theta)
        return math.degrees(angle_rad)

    def evaluate_micro_contact(
        self,
        cue_trajectory: Sequence[Point2D],
        target_trajectory: Sequence[Point2D],
    ) -> Dict[str, object]:
        """
        [공-공 미세 접촉 판정]
        수구와 목적구의 궤적 시퀀스를 분석하여 두 공의 간격이 12mm 미만으로 진입했을 때,
        최소 서브픽셀 간극(Gap = 거리 - 61.5mm)과 최근접 시점 전후의 수구 진행 각도 굴절(> 0.35도)을
        동시 검증하고 팝업 렌더링용 딕셔너리를 반환합니다.

        Args:
            cue_trajectory: 수구의 서브픽셀 (x_mm, y_mm) 시계열 좌표 리스트 (최소 3개 이상 권장)
            target_trajectory: 목적구의 서브픽셀 (x_mm, y_mm) 시계열 좌표 리스트 (길이가 1이면 정지 공으로 간주)

        Returns:
            popup_dict: {
                "var_triggered": bool,
                "status": "CONTACT_CONFIRMED" | "NEAR_MISS" | "NO_APPROACH",
                "is_contact": bool,
                "message": str,
                "gap_mm": float,
                "deflection_deg": float,
                "closest_frame_idx": int
            }
        """
        if len(cue_trajectory) < 3:
            raise ValueError("굴절각 및 최소 간극 계측을 위해 최소 3프레임 이상의 수구 궤적이 필요합니다.")

        if len(target_trajectory) == 1:
            target_pts = [target_trajectory[0]] * len(cue_trajectory)
        elif len(target_trajectory) == len(cue_trajectory):
            target_pts = list(target_trajectory)
        else:
            raise ValueError("target_trajectory는 길이 1(정지 공)이거나 cue_trajectory와 길이가 같아야 합니다.")

        # 1. 각 프레임별 중심 거리 및 서브픽셀 간극(Gap = 거리 - 61.5mm) 계산
        gaps: List[float] = []
        for c_pt, t_pt in zip(cue_trajectory, target_pts):
            center_dist = math.hypot(c_pt[0] - t_pt[0], c_pt[1] - t_pt[1])
            gap = center_dist - self.ball_diameter_mm
            gaps.append(gap)

        min_idx = int(np.argmin(gaps))
        min_gap_mm = float(gaps[min_idx])

        # 12mm 미만으로 근접하지 않은 경우 VAR 비활성화 반환
        if min_gap_mm >= self.NEAR_MISS_TRIGGER_GAP_MM:
            return {
                "var_triggered": False,
                "status": "NO_APPROACH",
                "is_contact": False,
                "message": "근접 이벤트 없음 (간격 12mm 이상)",
                "gap_mm": round(min_gap_mm, 3),
                "deflection_deg": 0.0,
                "closest_frame_idx": min_idx,
            }

        # 2. 최근접 시점(min_idx) 전후의 수구 진행 방향 벡터 추출 및 굴절각 계산
        pivot_idx = max(1, min(min_idx, len(cue_trajectory) - 2))
        pt_before = cue_trajectory[0]
        pt_pivot = cue_trajectory[pivot_idx]
        pt_after = cue_trajectory[-1]

        vec_in = (pt_pivot[0] - pt_before[0], pt_pivot[1] - pt_before[1])
        vec_out = (pt_after[0] - pt_pivot[0], pt_after[1] - pt_pivot[1])
        deflection_deg = self.compute_deflection_angle_deg(vec_in, vec_out)

        # 3. 간극 <= 0mm 이고 굴절각 > 0.35도 동시 만족 시 미세 접촉 확정
        is_contact = (min_gap_mm <= 0.0) and (
            deflection_deg > self.MIN_DEFLECTION_ANGLE_DEG
        )

        if is_contact:
            status = "CONTACT_CONFIRMED"
            message = (
                f"[Micro-Hawk VAR 판독: 접촉 인정] 서브픽셀 간극 {min_gap_mm:.2f}mm (<= 0mm), "
                f"수구 굴절각 {deflection_deg:.2f}° (> 0.35°) 감지"
            )
        else:
            status = "NEAR_MISS"
            message = (
                f"[Micro-Hawk VAR 판독: 노터치(스침 불발)] 최소 간극 {min_gap_mm:.2f}mm, "
                f"수구 굴절각 {deflection_deg:.2f}°"
            )

        return {
            "var_triggered": True,
            "status": status,
            "is_contact": is_contact,
            "message": message,
            "gap_mm": round(min_gap_mm, 3),
            "deflection_deg": round(deflection_deg, 3),
            "closest_frame_idx": min_idx,
        }

    def evaluate_cushion_double_touch(
        self,
        cushion_distances_mm: Sequence[float],
        timestamps_ms: Sequence[float],
    ) -> Dict[str, object]:
        """
        [쿠션 이중 터치 판정]
        쿠션 표면 거리의 시계열 프로파일을 분석하여 쿠션 접촉 영역(<= CUSHION_MARGIN_MM) 내에서
        90ms 이내에 압축 극솟점(Local Minima)이 2회 이상 검출될 경우 Double Touch(2쿠션 파울)로 판정합니다.

        Args:
            cushion_distances_mm: 시간에 따른 수구 중심과 쿠션 레일 간의 거리 시계열 리스트 (mm)
            timestamps_ms: 각 거리 측정 시점의 타임스탬프 리스트 (단위: ms)

        Returns:
            popup_dict: {
                "var_triggered": bool,
                "status": "DOUBLE_TOUCH_FOUL" | "CLEAN_CUSHION" | "NO_CUSHION_CONTACT",
                "is_double_touch": bool,
                "message": str,
                "gap_mm": float,
                "deflection_deg": float,
                "minima_count": int,
                "minima_interval_ms": float
            }
        """
        if len(cushion_distances_mm) != len(timestamps_ms) or len(cushion_distances_mm) < 3:
            raise ValueError(
                "쿠션 거리 프로파일과 타임스탬프는 길이가 같아야 하며 최소 3개 이상의 샘플이 필요합니다."
            )

        dists = np.asarray(cushion_distances_mm, dtype=np.float64)
        times = np.asarray(timestamps_ms, dtype=np.float64)

        # 1. 쿠션 접촉 마진(32.0mm) 이하 구간에서 압축 극솟점(Local Minima) 인덱스 탐색
        minima_indices: List[int] = []
        for i in range(1, len(dists) - 1):
            if dists[i] <= self.cushion_margin_mm:
                if (dists[i] < dists[i - 1] and dists[i] <= dists[i + 1]) or (
                    dists[i] <= dists[i - 1] and dists[i] < dists[i + 1]
                ):
                    minima_indices.append(i)

        min_dist_mm = float(np.min(dists))
        compression_gap_mm = min_dist_mm - self.cushion_margin_mm

        if len(minima_indices) == 0:
            return {
                "var_triggered": False,
                "status": "NO_CUSHION_CONTACT",
                "is_double_touch": False,
                "message": "쿠션 압축 극솟점이 감지되지 않았습니다.",
                "gap_mm": round(compression_gap_mm, 3),
                "deflection_deg": 0.0,
                "minima_count": 0,
                "minima_interval_ms": 0.0,
            }

        # 2. 인접한 두 압축 극솟점(Local Minima) 간의 시간 간격이 90ms 이내인지 검사
        is_double_touch = False
        detected_interval_ms = 0.0

        for idx in range(len(minima_indices) - 1):
            t1 = float(times[minima_indices[idx]])
            t2 = float(times[minima_indices[idx + 1]])
            dt_ms = abs(t2 - t1)
            if dt_ms <= self.DOUBLE_TOUCH_WINDOW_MS:
                is_double_touch = True
                detected_interval_ms = dt_ms
                break

        if is_double_touch:
            return {
                "var_triggered": True,
                "status": "DOUBLE_TOUCH_FOUL",
                "is_double_touch": True,
                "message": (
                    f"[Micro-Hawk VAR 판독: 쿠션 더블터치 파울] {detected_interval_ms:.1f}ms 이내(<= 90ms) "
                    f"쿠션 압축 극솟점 2회 연속 검출!"
                ),
                "gap_mm": round(compression_gap_mm, 3),
                "deflection_deg": 0.0,
                "minima_count": len(minima_indices),
                "minima_interval_ms": round(detected_interval_ms, 2),
            }

        return {
            "var_triggered": False,
            "status": "CLEAN_CUSHION",
            "is_double_touch": False,
            "message": "[Micro-Hawk VAR 판독: 정상 1쿠션 반사] 단일 압축 극솟점 확인",
            "gap_mm": round(compression_gap_mm, 3),
            "deflection_deg": 0.0,
            "minima_count": len(minima_indices),
            "minima_interval_ms": round(detected_interval_ms, 2),
        }
