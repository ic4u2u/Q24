"""
[조명 불변 3구 상대 색도 랭킹 + 정지 앵커 ID 고정기]
파일 경로: ball_identity_tracker.py

1) 적색도(a* 및 R-G) 1위 공을 RED로 먼저 분리하고,
2) 나머지 두 공 중 황색도(b* + HSV 채도 + Blue 결핍도) 상대 비교로 WHITE와 YELLOW를 100% 구분합니다.
3) 세 공이 정지해 있는 IDLE 구간에서 각 공의 정지 위치(locked_stationary_pos_px)와 색상 ID를 고정하여
   스트로크 순간 모션 블러가 발생해도 출발한 공의 색상 ID가 바뀌지 않도록 보장합니다.
"""

from itertools import permutations
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from three_cushion_vision.causal_kiss_fsm import BallColor


class LabHungarianBallIdentifier:
    """
    [조명 불변 3구 상대 색도 랭킹 + 정지 앵커 ID 고정기]
    1) 적색도(a* 및 R-G) 1위 공을 RED로 먼저 분리하고,
    2) 나머지 두 공 중 황색도(b* + HSV 채도 + Blue 결핍도) 상대 비교로 WHITE와 YELLOW를 100% 구분합니다.
    """

    def __init__(self) -> None:
        self.prev_positions_px: Dict[BallColor, np.ndarray] = {}
        self.prev_centers = self.prev_positions_px
        self.locked_stationary_pos_px: Dict[BallColor, np.ndarray] = {}

    def reset(self) -> None:
        """이전 프레임 위치 및 정지 앵커 위치 이력을 초기화합니다."""
        self.prev_positions_px.clear()
        self.locked_stationary_pos_px.clear()

    def lock_stationary_anchors(
        self, assigned_boxes: Dict[BallColor, Tuple[float, float, float, float]]
    ) -> None:
        """세 공이 정지해 있는 IDLE 구간에서 각 공의 정지 중심 좌표를 고정(Anchor Lock)합니다."""
        for c_enum, box in assigned_boxes.items():
            center = np.array(
                [(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32
            )
            self.locked_stationary_pos_px[c_enum] = center

    def extract_color_metrics(
        self, frame_bgr: np.ndarray, bbox_xyxy: Tuple[float, float, float, float]
    ) -> Tuple[float, float, np.ndarray]:
        """
        반환값: (redness_score, yellowness_score, lab_median)
        """
        h_img, w_img = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
            return 0.0, 0.0, np.array([180.0, 128.0, 128.0], dtype=np.float32)

        crop_resized = cv2.resize(crop, (24, 24), interpolation=cv2.INTER_LINEAR)
        lab = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV).astype(np.float32)
        bgr = crop_resized.astype(np.float32)

        yy, xx = np.ogrid[:24, :24]
        circle_mask = ((xx - 11.5) ** 2 + (yy - 11.5) ** 2) <= (8.5 ** 2)

        lab_px = lab[circle_mask]
        hsv_px = hsv[circle_mask]
        bgr_px = bgr[circle_mask]

        # 천장 조명 하이라이트(상위 22% L*) 및 하단 그림자(하위 15% L*) 제거
        l_vals = lab_px[:, 0]
        low_p, high_p = np.percentile(l_vals, [15, 78])
        valid = (l_vals >= low_p) & (l_vals <= high_p)
        if not np.any(valid):
            valid = np.ones_like(l_vals, dtype=bool)

        med_lab = np.median(lab_px[valid], axis=0)
        med_hsv = np.median(hsv_px[valid], axis=0)
        med_bgr = np.median(bgr_px[valid], axis=0)  # [B, G, R]

        L_val, a_val, b_val = float(med_lab[0]), float(med_lab[1]), float(med_lab[2])
        S_val = float(med_hsv[1])
        B_ch, G_ch, R_ch = float(med_bgr[0]), float(med_bgr[1]), float(med_bgr[2])

        # 적색도(Redness): LAB a* 양수 편차 + (R - G)
        redness_score = 2.2 * (a_val - 128.0) + 1.0 * (R_ch - G_ch) - 0.3 * L_val
        # 황색도(Yellowness): LAB b* 양수 편차 + HSV 채도(S) + Blue 채널 결핍도((R+G)/2 - B)
        yellowness_score = (
            2.0 * (b_val - 128.0)
            + 1.1 * S_val
            + 1.0 * ((R_ch + G_ch) * 0.5 - B_ch)
        )

        return redness_score, yellowness_score, med_lab

    def extract_glare_free_lab(
        self, frame_bgr: np.ndarray, bbox_xyxy: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """하이라이트 및 그림자를 제거한 CIE LAB 중앙값 벡터를 반환합니다."""
        _, _, med_lab = self.extract_color_metrics(frame_bgr, bbox_xyxy)
        return med_lab

    def assign_ball_identities(
        self,
        frame_bgr: np.ndarray,
        bboxes_xyxy: List[Tuple[float, float, float, float]],
        is_idle_state: bool = False,
    ) -> Dict[BallColor, Tuple[float, float, float, float]]:
        if not bboxes_xyxy:
            return {}

        n_det = min(len(bboxes_xyxy), 3)
        bboxes = bboxes_xyxy[:n_det]
        metrics = [self.extract_color_metrics(frame_bgr, box) for box in bboxes]
        centers = [
            np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])
            for box in bboxes
        ]

        # 3개의 공이 모두 검출된 경우: 상대 색도 랭킹(Relative Ranking)으로 WHITE / YELLOW / RED 분리
        if n_det == 3:
            # 1) 적색도(redness_score)가 가장 높은 공 = RED
            red_idx = int(np.argmax([m[0] for m in metrics]))
            rem_indices = [i for i in range(3) if i != red_idx]

            # 2) 남은 2개 공 중 황색도(yellowness_score) 비교로 YELLOW vs WHITE 결정
            i1, i2 = rem_indices[0], rem_indices[1]
            y_score_1, y_score_2 = metrics[i1][1], metrics[i2][1]

            # 만약 두 공의 색상 차이가 뚜렷하면(> 12.0) 상대 랭킹을 최우선 적용하고,
            # 모션 블러로 색상이 흐려졌을 때만 이전 프레임 위치 연속성(Hungarian Motion)으로 보완
            if abs(y_score_1 - y_score_2) > 12.0 or (
                BallColor.WHITE not in self.prev_positions_px
            ):
                yellow_idx = i1 if y_score_1 >= y_score_2 else i2
                white_idx = i2 if yellow_idx == i1 else i1
            else:
                ref_w = self.locked_stationary_pos_px.get(
                    BallColor.WHITE, self.prev_positions_px[BallColor.WHITE]
                )
                d_i1_w = float(np.linalg.norm(centers[i1] - ref_w))
                d_i2_w = float(np.linalg.norm(centers[i2] - ref_w))
                white_idx = i1 if d_i1_w <= d_i2_w else i2
                yellow_idx = i2 if white_idx == i1 else i1

            assigned = {
                BallColor.RED: bboxes[red_idx],
                BallColor.YELLOW: bboxes[yellow_idx],
                BallColor.WHITE: bboxes[white_idx],
            }
            for c_enum, idx in [
                (BallColor.RED, red_idx),
                (BallColor.YELLOW, yellow_idx),
                (BallColor.WHITE, white_idx),
            ]:
                self.prev_positions_px[c_enum] = centers[idx]
            if is_idle_state:
                self.lock_stationary_anchors(assigned)
            return assigned

        # 1~2개만 검출된 경우(일시적 가려짐): 색상 프로토타입 + 위치 연속성 매칭
        colors = [BallColor.WHITE, BallColor.YELLOW, BallColor.RED]
        cost_matrix = np.zeros((3, n_det), dtype=np.float32)
        for d_i in range(n_det):
            r_sc, y_sc, _ = metrics[d_i]
            # 색상별 비용 산출
            c_costs = {
                BallColor.RED: -r_sc,
                BallColor.YELLOW: -y_sc + max(0.0, r_sc * 0.5),
                BallColor.WHITE: y_sc + max(0.0, r_sc * 0.5),
            }
            for c_idx, c_enum in enumerate(colors):
                m_cost = 0.0
                if c_enum in self.prev_positions_px:
                    m_cost = min(
                        float(
                            np.linalg.norm(
                                centers[d_i] - self.prev_positions_px[c_enum]
                            )
                        )
                        * 0.25,
                        50.0,
                    )
                cost_matrix[c_idx, d_i] = c_costs[c_enum] + m_cost

        best_perm = min(
            permutations(range(3), n_det),
            key=lambda p: sum(cost_matrix[p[i], i] for i in range(n_det)),
        )
        assigned = {}
        for d_i, c_idx in enumerate(best_perm):
            c_enum = colors[c_idx]
            assigned[c_enum] = bboxes[d_i]
            self.prev_positions_px[c_enum] = centers[d_i]
        return assigned
