"""
CIE LAB 색공간(하이라이트 25% 및 그림자 15% 제거) + 헝가리안 전역 최적 매칭(Optimal Bipartite Assignment) 기반
수구(WHITE) · 노란공(YELLOW) · 적구(RED) 3구 정밀 식별기
파일 경로: ball_identity_tracker.py
"""

from itertools import permutations
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from scoring_fsm import BallColor


class LabHungarianBallIdentifier:
    """
    [문헌 기반 3구 정밀 식별기]
    1) 당구장 상단 LED 조명 반사광(상위 25% 밝기)과 라사지 배경 픽셀을 원형 마스크로 제거한 뒤
       CIE LAB 색공간의 (a*: 적-녹, b*: 황-청) 색도 벡터를 추출합니다.
    2) 색상 비용(Color Cost)과 이전 프레임 위치 대비 마할라노비스 공간 이동 비용(Motion Cost)을 결합한
       3x3 비용 행렬을 구축하고 전역 최적 매칭(Hungarian Assignment)으로 WHITE, YELLOW, RED를 1:1 확정합니다.
    """

    def __init__(self) -> None:
        # OpenCV 8-bit LAB 기준 프로토타입 (L: 0~255, a: 0~255 [128이 중립], b: 0~255 [128이 중립])
        # WHITE: a≈128, b≈128 (고명도, 무채색)
        # YELLOW: a≈132, b≈178 (b* 채널 강한 양수)
        # RED: a≈175, b≈150 (a* 채널 강한 양수)
        self.lab_prototypes: Dict[BallColor, np.ndarray] = {
            BallColor.WHITE: np.array([215.0, 128.0, 130.0], dtype=np.float32),
            BallColor.YELLOW: np.array([195.0, 134.0, 178.0], dtype=np.float32),
            BallColor.RED: np.array([135.0, 178.0, 152.0], dtype=np.float32),
        }
        self.prev_positions_px: Dict[BallColor, np.ndarray] = {}

    def extract_glare_free_lab(
        self, frame_bgr: np.ndarray, bbox_xyxy: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        바운딩 박스 내부에서 원형 ROI 마스크로 파란 라사지 배경을 배제하고,
        천장 조명 하이라이트(상위 25% L* 밝기)와 하단 그림자(하위 15% L* 밝기)를 제거한
        순수 당구공 표면의 CIE LAB 중앙값(Median) 벡터를 추출합니다.
        """
        h_img, w_img = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
            return np.array([180.0, 128.0, 128.0], dtype=np.float32)

        crop_resized = cv2.resize(crop, (24, 24), interpolation=cv2.INTER_LINEAR)
        lab = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2LAB).astype(np.float32)

        # 중심 원형 ROI 마스크 (바운딩 박스 모서리의 파란 라사지 배경 배제)
        yy, xx = np.ogrid[:24, :24]
        circle_mask = ((xx - 11.5) ** 2 + (yy - 11.5) ** 2) <= (8.5 ** 2)
        pixels = lab[circle_mask]

        # 천장 조명 하이라이트(상위 25% L* 밝기) 및 하단 그림자(하위 15% L* 밝기) 제거
        l_vals = pixels[:, 0]
        low_p, high_p = np.percentile(l_vals, [15, 75])
        valid_mask = (l_vals >= low_p) & (l_vals <= high_p)
        filtered = pixels[valid_mask] if np.any(valid_mask) else pixels

        return np.median(filtered, axis=0)

    def assign_ball_identities(
        self,
        frame_bgr: np.ndarray,
        bboxes_xyxy: List[Tuple[float, float, float, float]],
    ) -> Dict[BallColor, Tuple[float, float, float, float]]:
        """
        검출된 바운딩 박스들을 [WHITE, YELLOW, RED]에 중복 없이 전역 최적 할당합니다.
        """
        if not bboxes_xyxy:
            return {}

        colors = [BallColor.WHITE, BallColor.YELLOW, BallColor.RED]
        n_det = min(len(bboxes_xyxy), 3)
        bboxes = bboxes_xyxy[:n_det]

        det_labs = [self.extract_glare_free_lab(frame_bgr, box) for box in bboxes]
        det_centers = [
            np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])
            for box in bboxes
        ]

        # 3(Color) x N(Detection) 비용 행렬 계산
        cost_matrix = np.zeros((3, n_det), dtype=np.float32)
        for c_idx, color in enumerate(colors):
            proto = self.lab_prototypes[color]
            for d_idx in range(n_det):
                # a*, b* 색도 채널에 L* 명도보다 2.5배 높은 가중치 부여
                diff = det_labs[d_idx] - proto
                color_cost = float(
                    np.sqrt(
                        0.4 * (diff[0] ** 2)
                        + 2.5 * (diff[1] ** 2)
                        + 2.5 * (diff[2] ** 2)
                    )
                )

                motion_cost = 0.0
                if color in self.prev_positions_px:
                    dist_px = float(
                        np.linalg.norm(
                            det_centers[d_idx] - self.prev_positions_px[color]
                        )
                    )
                    motion_cost = min(dist_px * 0.35, 60.0)

                cost_matrix[c_idx, d_idx] = color_cost + motion_cost

        # 3x3 순열 전수 탐색(Hungarian Global Minimum)으로 1:1 최적 매칭 수행
        best_perm = None
        best_score = float("inf")
        for perm in permutations(range(3), n_det):
            score = sum(cost_matrix[perm[d_i], d_i] for d_i in range(n_det))
            if score < best_score:
                best_score = score
                best_perm = perm

        assigned: Dict[BallColor, Tuple[float, float, float, float]] = {}
        if best_perm is not None:
            for d_i, c_idx in enumerate(best_perm):
                c_enum = colors[c_idx]
                assigned[c_enum] = bboxes[d_i]
                self.prev_positions_px[c_enum] = det_centers[d_i]

        return assigned
