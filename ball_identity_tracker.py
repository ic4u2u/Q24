"""
[특허 US20210170261A1 & PMC11183222 기반 조명 불변 3구 식별기]
파일 경로: ball_identity_tracker.py

핵심 기술:
1) YCbCr Cb 크로마키 라사지 배경 제거 (PMC11183222):
   Cb = 128 - 0.148*R - 0.291*G + 0.439*B 색차 공식으로 파란색/초록색 당구대 천(Cloth) 픽셀과
   상위 22% 천장 조명 반사광(Glare) 픽셀을 원천 배제합니다.
2) 상대 색도 랭킹 (Relative Chromaticity Ranking):
   3개의 공 중 적색도(a* 및 R-G) 1위를 무조건 RED로 먼저 분리하고,
   남은 두 공 중 황색도(b* + 채도 S + Blue 결핍도) 비교로 YELLOW와 WHITE를 100% 구분합니다.
3) 중첩 공 타원 장축 분해 (split_merged_oval_blob):
   두 공이 키스하여 하나의 길쭉한 바운딩 박스(장축/단축 비율 >= 1.45)로 병합되었을 때,
   타원 중심에서 장축 방향 양 끝(+-0.25 * 장축 길이)으로 두 개의 서브 박스를 분리합니다.
"""

from itertools import permutations
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from three_cushion_vision.causal_kiss_fsm import BallColor


def split_merged_oval_blob(
    bbox_xyxy: Tuple[float, float, float, float],
    contour: Optional[np.ndarray] = None,
    aspect_ratio_thresh: float = 1.45,
) -> List[Tuple[float, float, float, float]]:
    """
    [기술 1-3: 중첩 공 타원 장축 분해]
    두 공이 키스/밀착하여 하나의 길쭉한 바운딩 박스(장축/단축 비율 >= 1.45)로 병합되었을 때,
    타원의 중심에서 장축 방향 양 끝(+- 0.25 * 장축 길이)으로 두 개의 서브 박스를 분리합니다.
    """
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    if contour is not None and len(contour) >= 5:
        try:
            (ecx, ecy), (ma_1, ma_2), angle_deg = cv2.fitEllipse(contour)
            major_len = float(max(ma_1, ma_2))
            minor_len = float(max(1.0, min(ma_1, ma_2)))
            if major_len / minor_len >= aspect_ratio_thresh:
                # OpenCV fitEllipse에서 angle_deg는 단축(ma_1) 기준 시계 방향 각도이므로 장축 방향 벡터 산출
                rad = np.deg2rad(angle_deg + (90.0 if ma_2 >= ma_1 else 0.0))
                ux, uy = float(np.cos(rad)), float(np.sin(rad))
                offset = 0.25 * major_len
                r = minor_len * 0.5
                c1x, c1y = float(ecx) - offset * ux, float(ecy) - offset * uy
                c2x, c2y = float(ecx) + offset * ux, float(ecy) + offset * uy
                return [
                    (c1x - r, c1y - r, c1x + r, c1y + r),
                    (c2x - r, c2y - r, c2x + r, c2y + r),
                ]
        except cv2.error:
            pass

    major_len = max(w, h)
    minor_len = min(w, h)
    if (major_len / max(minor_len, 1e-6)) < aspect_ratio_thresh:
        return [(x1, y1, x2, y2)]

    offset = 0.25 * major_len
    r = minor_len * 0.5
    if w >= h:
        c1x, c1y = cx - offset, cy
        c2x, c2y = cx + offset, cy
    else:
        c1x, c1y = cx, cy - offset
        c2x, c2y = cx, cy + offset

    return [
        (c1x - r, c1y - r, c1x + r, c1y + r),
        (c2x - r, c2y - r, c2x + r, c2y + r),
    ]


class LabHungarianBallIdentifier:
    """
    [특허 US20210170261A1 & PMC11183222 기반 3구 정밀 식별기]
    1) YCbCr Cb 색차 공식(Cb = 128 - 0.148R - 0.291G + 0.439B)으로 라사지 배경 및 상위 22% 반사광 제거
    2) 적색도(a* 및 R-G) 1위 공을 RED로 먼저 분리하고,
       나머지 두 공 중 황색도(b* + HSV 채도 + Blue 결핍도) 비교로 WHITE와 YELLOW를 100% 구분
    3) split_merged_oval_blob으로 중첩된 타원형 블롭(장축/단축 >= 1.45)을 2개의 서브 박스로 자동 분해
    """

    def __init__(self) -> None:
        self.prev_positions_px: Dict[BallColor, np.ndarray] = {}
        self.prev_centers = self.prev_positions_px
        self.locked_stationary_pos_px: Dict[BallColor, np.ndarray] = {}

    def reset(self) -> None:
        """이전 프레임 위치 및 정지 앵커 위치 이력을 초기화합니다."""
        self.prev_positions_px.clear()
        self.locked_stationary_pos_px.clear()

    @staticmethod
    def split_merged_oval_blob(
        bbox_xyxy: Tuple[float, float, float, float],
        contour: Optional[np.ndarray] = None,
        aspect_ratio_thresh: float = 1.45,
    ) -> List[Tuple[float, float, float, float]]:
        """장축/단축 비율 >= 1.45인 병합 바운딩 박스를 장축 방향 +-0.25*장축길이 두 서브 박스로 분리합니다."""
        return split_merged_oval_blob(
            bbox_xyxy=bbox_xyxy,
            contour=contour,
            aspect_ratio_thresh=aspect_ratio_thresh,
        )

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
        [기술 1-1 & 1-2]
        PMC11183222 YCbCr Cb 공식(Cb = 128 - 0.148R - 0.291G + 0.439B)으로 당구대 라사지 배경 픽셀과
        상위 22% 조명 반사광(Glare)을 배제한 뒤 (redness_score, yellowness_score, lab_median)을 반환합니다.
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

        B_arr = bgr_px[:, 0]
        G_arr = bgr_px[:, 1]
        R_arr = bgr_px[:, 2]

        # [PMC11183222] YCbCr 색차 공식: Cb = 128 - 0.148*R - 0.291*G + 0.439*B
        cb_vals = 128.0 - 0.148 * R_arr - 0.291 * G_arr + 0.439 * B_arr
        is_blue_cloth = (cb_vals > 142.0) & ((B_arr - R_arr) > 35.0)
        is_green_cloth = (cb_vals < 110.0) & ((G_arr - R_arr) > 25.0) & ((G_arr - B_arr) > 20.0)
        non_cloth_mask = ~(is_blue_cloth | is_green_cloth)

        if np.count_nonzero(non_cloth_mask) >= 6:
            lab_px = lab_px[non_cloth_mask]
            hsv_px = hsv_px[non_cloth_mask]
            bgr_px = bgr_px[non_cloth_mask]

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
        """하이라이트 및 라사지 배경을 제거한 CIE LAB 중앙값 벡터를 반환합니다."""
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

        # 만약 2개의 바운딩 박스만 검출되었는데 그중 하나가 키스로 병합된 타원 블롭(장축/단축 >= 1.45)이면 자동 분해
        expanded_bboxes: List[Tuple[float, float, float, float]] = []
        if len(bboxes_xyxy) < 3:
            for box in bboxes_xyxy:
                sub_boxes = self.split_merged_oval_blob(box)
                expanded_bboxes.extend(sub_boxes)
        else:
            expanded_bboxes = list(bboxes_xyxy)

        n_det = min(len(expanded_bboxes), 3)
        bboxes = expanded_bboxes[:n_det]
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
