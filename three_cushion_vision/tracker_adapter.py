"""
Module B: OC-SORT 트래커 연동, HSV 색공간 히스토그램 기반 Ball ID 고정 및 적응형 EMA 스무딩 어댑터
파일 경로: three_cushion_vision/tracker_adapter.py

1. 당구공 정면충돌 시 트래커의 track_id 뒤바뀜(ID Switching)을 방지하기 위해,
   바운딩 박스 중심부 원형 ROI의 HSV 색공간 히스토그램을 분석하여 매 프레임 WHITE / YELLOW / RED로 재매핑합니다.
2. 바운딩 박스 떨림(Jitter)을 제어하기 위해 좌표와 속도 벡터에 적응형 지수이동평균
   (Adaptive EMA: 정지/저속 구간 alpha=0.25, 고속 이동 구간 alpha=0.75)을 적용합니다.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from three_cushion_vision.config import BallColor, DEFAULT_CONFIG, VisionConfig
from three_cushion_vision.homography import BBoxXYXY, Point2D, TableHomography


@dataclass
class TrackedBall:
    """단일 프레임에서 색상 잠금 및 물리 좌표(mm)·속도(mm/s) 변환이 완료된 당구공 상태"""

    color: BallColor
    x: float
    y: float
    vx: float = 0.0          # mm/s 단위 x축 속도
    vy: float = 0.0          # mm/s 단위 y축 속도
    speed: float = 0.0       # mm/s 단위 속력
    raw_track_id: int = -1
    timestamp: float = 0.0
    is_occluded_restored: bool = False  # BilliardNanoGPT로 복원된 가려짐 프레임 여부


@dataclass
class _InternalColorState:
    """색상별 적응형 EMA 위치·속도 추적 내부 상태 및 15프레임 물리 시퀀스 버퍼"""

    x_ema: float
    y_ema: float
    vx_ema: float = 0.0
    vy_ema: float = 0.0
    raw_x: float = 0.0
    raw_y: float = 0.0
    last_timestamp: float = 0.0
    last_track_id: int = -1
    history_15: Optional[List[Tuple[float, float, float, float]]] = None


class OCSortTrackerAdapter:
    """
    YOLO/RF-DETR + OC-SORT 검출 결과를 입력받아:
    1) 원형 ROI의 HSV 색공간 히스토그램 분석으로 BallColor(WHITE, YELLOW, RED)를 식별하여 ID 스위칭을 원천 차단하고,
    2) TableHomography의 대각선 시차(Parallax) 보정을 거쳐 2840x1420mm 물리 좌표로 변환하며,
    3) 적응형 EMA(정지 구간 alpha=0.25, 고속 구간 alpha=0.75)로 떨림을 억제하고,
    4) 선수 몸 가려짐(Occlusion) 발생 시 BilliardNanoGPT로 비선형 쿠션 반사 궤적을 복원하는 트래커 어댑터입니다.
    """

    def __init__(
        self,
        homography: Optional[TableHomography] = None,
        config: VisionConfig = DEFAULT_CONFIG,
        enable_occlusion_restorer: bool = True,
    ) -> None:
        self.config: VisionConfig = config
        self.homography: TableHomography = homography or TableHomography(config=config)
        self._states: Dict[BallColor, _InternalColorState] = {}
        self.enable_occlusion_restorer: bool = enable_occlusion_restorer
        self._restorers: Dict[BallColor, Any] = {}

    def reset(self) -> None:
        """트래커 어댑터의 누적 EMA 상태 및 가려짐 복원 버퍼를 초기화합니다."""
        self._states.clear()
        self._restorers.clear()

    @staticmethod
    def classify_roi_hsv_histogram(
        bgr_frame: np.ndarray, bbox_xyxy: BBoxXYXY
    ) -> Tuple[BallColor, Dict[BallColor, float]]:
        """
        바운딩 박스 중심부 원형 ROI(반지름 60% 영역, 천장 조명 반사 하이라이트 제외)의
        HSV 색공간 히스토그램을 계산하여 WHITE / YELLOW / RED 점수와 최적 색상을 반환합니다.
        """
        h_img, w_img = bgr_frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
        x1 = max(0, min(w_img - 1, x1))
        x2 = max(x1 + 1, min(w_img, x2))
        y1 = max(0, min(h_img - 1, y1))
        y2 = max(y1 + 1, min(h_img, y2))

        crop = bgr_frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
            return BallColor.WHITE, {
                BallColor.WHITE: 1.0,
                BallColor.YELLOW: 0.0,
                BallColor.RED: 0.0,
            }

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        ch, cw = hsv.shape[:2]
        yy, xx = np.ogrid[:ch, :cw]
        cy, cx = ch * 0.5, cw * 0.5
        r_max = min(ch, cw) * 0.42
        r_min = min(ch, cw) * 0.12  # 정중앙 천장 조명 하이라이트(Specular) 배제

        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        roi_mask = ((dist_sq <= r_max ** 2) & (dist_sq >= r_min ** 2)).astype(np.uint8) * 255
        if cv2.countNonZero(roi_mask) < 4:
            roi_mask = (dist_sq <= (min(ch, cw) * 0.48) ** 2).astype(np.uint8) * 255

        # Hue(0~179) 18빈 및 Saturation(0~255) 8빈 2D/1D 히스토그램 분석
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        valid_pixels = (roi_mask > 0) & (v_channel >= 35)
        if not np.any(valid_pixels):
            valid_pixels = roi_mask > 0

        h_vals = h_channel[valid_pixels]
        s_vals = s_channel[valid_pixels]
        v_vals = v_channel[valid_pixels]
        total_px = float(max(1, len(h_vals)))

        # 1) RED 비율: Hue가 0~12 또는 165~180 이고 채도(S) >= 85인 픽셀 비율
        red_mask = ((h_vals <= 12) | (h_vals >= 165)) & (s_vals >= 85)
        red_score = float(np.sum(red_mask)) / total_px

        # 2) YELLOW 비율: Hue가 15~38 이고 채도(S) >= 75인 픽셀 비율
        yellow_mask = (h_vals >= 15) & (h_vals <= 38) & (s_vals >= 75)
        yellow_score = float(np.sum(yellow_mask)) / total_px

        # 3) WHITE 비율: 채도(S) < 65 이고 명도(V) >= 110인 무채색 고명도 픽셀 비율
        white_mask = (s_vals < 65) & (v_vals >= 110)
        white_score = float(np.sum(white_mask)) / total_px

        scores = {
            BallColor.RED: red_score,
            BallColor.YELLOW: yellow_score,
            BallColor.WHITE: white_score,
        }
        best_color = max(scores.keys(), key=lambda c: scores[c])
        return best_color, scores

    def _assign_unique_colors(
        self,
        detections: Sequence[Dict[str, Any]],
        bgr_frame: Optional[np.ndarray] = None,
    ) -> Dict[BallColor, Dict[str, Any]]:
        """
        검출된 최대 3개의 당구공 객체에 대해 HSV 색공간 히스토그램 점수(또는 명시된 color 힌트)와
        직전 프레임 위치를 결합하여 WHITE, YELLOW, RED가 중복 없이 1:1 매핑되도록 보장합니다.
        """
        assigned: Dict[BallColor, Dict[str, Any]] = {}
        all_colors = [BallColor.WHITE, BallColor.YELLOW, BallColor.RED]

        # 1) 이미 명시적 BallColor가 지정된 입력(시뮬레이션 또는 사전 분류 완료 데이터) 처리
        remaining_dets: List[Dict[str, Any]] = []
        for det in detections[:3]:
            explicit_color = det.get("color")
            if explicit_color is not None:
                c_enum = (
                    explicit_color
                    if isinstance(explicit_color, BallColor)
                    else BallColor(str(explicit_color))
                )
                if c_enum not in assigned:
                    assigned[c_enum] = det
                    continue
            remaining_dets.append(det)

        if not remaining_dets:
            return assigned

        # 2) BGR 프레임이 주어진 경우 원형 ROI HSV 히스토그램 점수로 1:1 최적 할당
        unassigned_colors = [c for c in all_colors if c not in assigned]
        if bgr_frame is not None and bgr_frame.size > 0:
            det_scores: List[Dict[BallColor, float]] = []
            for det in remaining_dets:
                bbox = det.get("bbox")
                if bbox is not None:
                    _, sc = self.classify_roi_hsv_histogram(bgr_frame, tuple(bbox))  # type: ignore[arg-type]
                else:
                    sc = {c: 0.33 for c in all_colors}
                det_scores.append(sc)

            # 남은 검출 객체들에 대해 점수 합이 최대가 되는 색상 할당
            used_det_indices = set()
            for color in sorted(
                unassigned_colors,
                key=lambda c: max((ds[c] for ds in det_scores), default=0.0),
                reverse=True,
            ):
                best_idx = -1
                best_val = -1.0
                for idx, ds in enumerate(det_scores):
                    if idx in used_det_indices:
                        continue
                    if ds[color] > best_val:
                        best_val = ds[color]
                        best_idx = idx
                if best_idx >= 0:
                    used_det_indices.add(best_idx)
                    assigned[color] = remaining_dets[best_idx]
            return assigned

        # 3) 프레임 없이 좌표/트랙만 주어진 경우: 직전 위치 최근접 매칭
        for det, color in zip(remaining_dets, unassigned_colors):
            assigned[color] = det
        return assigned

    def update_from_detections(
        self,
        detections: Sequence[Dict[str, Any]],
        timestamp: float,
        bgr_frame: Optional[np.ndarray] = None,
        enable_ema: bool = True,
    ) -> Dict[BallColor, TrackedBall]:
        """
        OC-SORT 트래커 출력 리스트를 받아 색상 고정(WHITE/YELLOW/RED), 시차 보정 호모그래피 변환,
        적응형 EMA 필터링을 수행한 뒤 {BallColor: TrackedBall} 딕셔너리를 반환합니다.

        Args:
            detections: 각 원소가 {"bbox": (x1,y1,x2,y2), "track_id": int, "color": Optional[BallColor]}
                        또는 {"x_mm": float, "y_mm": float, "color": BallColor} 형태인 리스트
            timestamp: 현재 프레임 타임스탬프 (초)
            bgr_frame: HSV 색상 히스토그램 분석을 위한 원본 BGR 프레임 (선택)
            enable_ema: 바운딩 박스 노이즈 제어를 위한 적응형 EMA 활성화 여부
        """
        mapped = self._assign_unique_colors(detections, bgr_frame=bgr_frame)
        output: Dict[BallColor, TrackedBall] = {}

        for color, det in mapped.items():
            track_id = int(det.get("track_id", -1))
            if "x_mm" in det and "y_mm" in det:
                raw_x = float(det["x_mm"])
                raw_y = float(det["y_mm"])
            elif "bbox" in det:
                raw_x, raw_y = self.homography.bbox_to_table_mm(tuple(det["bbox"]))  # type: ignore[arg-type]
            else:
                raise ValueError("검출 데이터에 ('x_mm', 'y_mm') 또는 'bbox'가 포함되어야 합니다.")

            if color not in self._states:
                self._states[color] = _InternalColorState(
                    x_ema=raw_x,
                    y_ema=raw_y,
                    vx_ema=0.0,
                    vy_ema=0.0,
                    raw_x=raw_x,
                    raw_y=raw_y,
                    last_timestamp=timestamp,
                    last_track_id=track_id,
                    history_15=[(raw_x, raw_y, 0.0, 0.0)],
                )
                output[color] = TrackedBall(
                    color=color,
                    x=raw_x,
                    y=raw_y,
                    vx=0.0,
                    vy=0.0,
                    speed=0.0,
                    raw_track_id=track_id,
                    timestamp=timestamp,
                    is_occluded_restored=False,
                )
                continue

            st = self._states[color]
            dt = timestamp - st.last_timestamp
            if dt <= 1e-6:
                dt = 1.0 / self.config.DEFAULT_FPS

            inst_vx = (raw_x - st.raw_x) / dt
            inst_vy = (raw_y - st.raw_y) / dt
            inst_speed = math.hypot(inst_vx, inst_vy)

            if not enable_ema:
                st.x_ema = raw_x
                st.y_ema = raw_y
                st.vx_ema = inst_vx
                st.vy_ema = inst_vy
            else:
                # 적응형 EMA: 정지/저속 구간은 alpha=0.25로 강한 스무딩, 고속 구간은 alpha=0.75로 즉각 반응
                alpha = (
                    self.config.EMA_ALPHA_FAST
                    if inst_speed >= self.config.EMA_FAST_SPEED_TH_MM_S
                    else self.config.EMA_ALPHA_STATIONARY
                )
                # 급격한 충돌/반사 직후에는 지연 없이 실제 물리 속도 반영
                prev_speed = math.hypot(st.vx_ema, st.vy_ema)
                if prev_speed >= 50.0 and inst_speed >= 50.0:
                    cos_angle = (st.vx_ema * inst_vx + st.vy_ema * inst_vy) / (
                        prev_speed * inst_speed
                    )
                    if cos_angle < 0.85:
                        alpha = 1.0

                st.x_ema = alpha * raw_x + (1.0 - alpha) * st.x_ema
                st.y_ema = alpha * raw_y + (1.0 - alpha) * st.y_ema
                st.vx_ema = alpha * inst_vx + (1.0 - alpha) * st.vx_ema
                st.vy_ema = alpha * inst_vy + (1.0 - alpha) * st.vy_ema

            st.raw_x = raw_x
            st.raw_y = raw_y
            st.last_timestamp = timestamp
            st.last_track_id = track_id

            final_x = st.x_ema if enable_ema else raw_x
            final_y = st.y_ema if enable_ema else raw_y
            if st.history_15 is None:
                st.history_15 = []
            st.history_15.append((final_x, final_y, st.vx_ema, st.vy_ema))
            if len(st.history_15) > 15:
                st.history_15.pop(0)
            if color in self._restorers:
                self._restorers[color].buffer = list(st.history_15)

            speed_final = math.hypot(st.vx_ema, st.vy_ema)
            output[color] = TrackedBall(
                color=color,
                x=final_x,
                y=final_y,
                vx=st.vx_ema,
                vy=st.vy_ema,
                speed=speed_final,
                raw_track_id=track_id,
                timestamp=timestamp,
                is_occluded_restored=False,
            )

        # [Step 2 핵심] 선수 몸/큐대 가려짐(Occlusion)으로 인해 이번 프레임에 누락된 공 복원
        if self.enable_occlusion_restorer and len(output) < 3:
            for color in (BallColor.WHITE, BallColor.YELLOW, BallColor.RED):
                if color in output or color not in self._states:
                    continue
                st = self._states[color]
                if not st.history_15:
                    continue

                if color not in self._restorers:
                    from three_cushion_vision.billiard_nanogpt import (
                        OcclusionTrajectoryRestorer,
                        get_or_train_billiard_nanogpt,
                    )

                    gpt_model = get_or_train_billiard_nanogpt()
                    restorer = OcclusionTrajectoryRestorer(gpt_model)
                    restorer.buffer = list(st.history_15)
                    self._restorers[color] = restorer

                rx, ry, rvx, rvy = self._restorers[color].step(None)
                st.x_ema = rx
                st.y_ema = ry
                st.vx_ema = rvx
                st.vy_ema = rvy
                st.raw_x = rx
                st.raw_y = ry
                st.last_timestamp = timestamp
                st.history_15.append((rx, ry, rvx, rvy))
                if len(st.history_15) > 15:
                    st.history_15.pop(0)

                output[color] = TrackedBall(
                    color=color,
                    x=rx,
                    y=ry,
                    vx=rvx,
                    vy=rvy,
                    speed=math.hypot(rvx, rvy),
                    raw_track_id=st.last_track_id,
                    timestamp=timestamp,
                    is_occluded_restored=True,
                )

        return output
