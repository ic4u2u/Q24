"""
OpenCV VideoCapture + Ultralytics YOLO + OC-SORT + three_cushion_vision 코어 엔진 통합 데모 비주얼라이저
파일 경로: three_cushion_vision/demo_visualizer.py

주요 기능:
1. 입력 MP4 영상(cv2.VideoCapture) 또는 실시간 합성 시뮬레이션 스트림에서 당구공 3개(WHITE, YELLOW, RED)를
   Ultralytics YOLO + OC-SORT(가중치 미설치 시 내장 HSV/컨투어 + OC-SORT 방향성 칼만 추적기 자동 폴백)로 검출·추적합니다.
2. TableHomography를 통해 카메라 픽셀 좌표를 국제식 대대 2D 물리 평면(2840mm x 1420mm)으로 변환하고
   시차(Parallax) 보정을 수행합니다.
3. ThreeCushionScoringFSM & KinematicsDetector로 서브프레임 연속 충돌(CCD), 0.5mm 미세 스침(굴절각+미세 밀림),
   법선 반전 쿠션 카운트, 그리고 수구 색상 감지 기반 미인지 득점 자동 보정을 수행합니다.
4. 원본 비디오 프레임 위에 4대 시각 요소를 실시간 오버레이하여 렌더링(및 MP4 저장)합니다:
   - [우측 상단 PIP] 실시간 2D 당구대 미니맵 (2840x1420mm 축소 투영, 쿠션 임계선, 3구 위치 및 수구 궤적)
   - [메인 화면 궤적] 수구 및 목적구 실시간 이동 궤적 라인(Trajectory Polylines)과 쿠션 임팩트 마커(C1, C2, C3...)
   - [좌측 상단 HUD] P1(WHITE) vs P2(YELLOW) 점수판, 현재 수구 색상, 실시간 쿠션 카운트 게이지(0~3+), FSM 상태
   - [하단 중앙 배너] 샷 판정 배너 (일반 3쿠션 득점 / 뱅크샷 득점 / 조기 키스 무효 잠금 / 미인지 득점 자동 보정)
"""

from __future__ import annotations

import argparse
import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from three_cushion_vision.config import (
    BallColor,
    DEFAULT_CONFIG,
    FailReason,
    ShotState,
    VisionConfig,
)
from three_cushion_vision.homography import BBoxXYXY, Point2D, TableHomography
from three_cushion_vision.scoreboard_bridge import ScoreboardBridge
from three_cushion_vision.scoring_fsm import ShotResult, ThreeCushionScoringFSM
from three_cushion_vision.tracker_adapter import OCSortTrackerAdapter, TrackedBall


@dataclass
class OCSortTrackState:
    """OC-SORT 관측 중심(Observation-Centric) 비선형 당구공 트랙 상태"""

    track_id: int
    bbox: BBoxXYXY
    center_px: Point2D
    velocity_px: Point2D = (0.0, 0.0)
    color_hint: Optional[BallColor] = None
    age: int = 0
    missed_frames: int = 0
    history_px: Deque[Point2D] = field(default_factory=lambda: deque(maxlen=30))


class UltralyticsOCSortDetector:
    """
    Ultralytics YOLO 객체 검출기 + OC-SORT(Observation-Centric SORT) 추적 래퍼 클래스입니다.
    - 로컬에 `ultralytics` 패키지 및 YOLO 가중치 파일이 존재하면 YOLO 추론을 수행합니다.
    - 가중치 파일이 없거나 오프라인 환경일 경우, 당구대 라사지 배경 분리 및 HSV/컨투어 기반
      고속 3구 검출기 + OC-SORT 관측 중심 모멘텀(OCM) 매칭기로 100% 끊김 없이 동작합니다.
    """

    def __init__(
        self,
        yolo_weights: Optional[str] = None,
        conf_threshold: float = 0.25,
        table_roi_px: Optional[List[Point2D]] = None,
    ) -> None:
        self.conf_threshold: float = conf_threshold
        self.table_roi_px: Optional[List[Point2D]] = table_roi_px
        self.yolo_model: Optional[Any] = None
        self.backend_name: str = "HSV_CONTOUR_OCSORT"
        self._tracks: Dict[int, OCSortTrackState] = {}
        self._next_track_id: int = 1

        if yolo_weights and os.path.isfile(yolo_weights):
            try:
                from ultralytics import YOLO  # type: ignore[import-not-found]

                self.yolo_model = YOLO(yolo_weights)
                self.backend_name = f"ULTRALYTICS_YOLO({os.path.basename(yolo_weights)})+OCSORT"
            except Exception:
                self.yolo_model = None

    def detect_and_track(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        입력 BGR 프레임에서 당구공 바운딩 박스들을 검출하고,
        OC-SORT 관측 중심 모멘텀(OCM) 매칭을 통해 일관된 track_id를 부여하여 반환합니다.
        """
        raw_bboxes: List[BBoxXYXY] = []

        if self.yolo_model is not None:
            try:
                results = self.yolo_model.predict(
                    source=frame_bgr,
                    conf=self.conf_threshold,
                    verbose=False,
                )
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    if boxes is not None and boxes.xyxy is not None:
                        xyxy_arr = boxes.xyxy.cpu().numpy()
                        for row in xyxy_arr:
                            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                            w = x2 - x1
                            h = y2 - y1
                            if 8.0 <= w <= 120.0 and 8.0 <= h <= 120.0:
                                raw_bboxes.append((x1, y1, x2, y2))
            except Exception:
                raw_bboxes = []

        if len(raw_bboxes) < 3:
            fallback_boxes = self._detect_balls_hsv_contour(frame_bgr)
            if len(fallback_boxes) >= len(raw_bboxes):
                raw_bboxes = fallback_boxes

        return self._associate_ocsort(frame_bgr, raw_bboxes[:3])

    def _detect_balls_hsv_contour(self, frame_bgr: np.ndarray) -> List[BBoxXYXY]:
        """
        파란색/초록색 당구대 라사지 위에서 흰공(WHITE), 노란공(YELLOW), 적구(RED)를
        HSV 색공간 마스크 및 원형 컨투어 필터링으로 정밀 검출합니다.
        """
        h_img, w_img = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # 당구대 영역 마스크 (table_roi_px가 지정된 경우 다각형 내부만 탐색)
        roi_mask = np.ones((h_img, w_img), dtype=np.uint8) * 255
        if self.table_roi_px and len(self.table_roi_px) >= 4:
            poly = np.array(self.table_roi_px, dtype=np.int32).reshape((-1, 1, 2))
            roi_mask.fill(0)
            cv2.fillPoly(roi_mask, [poly], 255)

        # 1) WHITE 마스크: 낮은 채도 + 높은 명도
        mask_white = cv2.inRange(hsv, np.array([0, 0, 155]), np.array([180, 75, 255]))
        # 2) YELLOW 마스크: Hue 14~38, 높은 채도/명도
        mask_yellow = cv2.inRange(hsv, np.array([14, 95, 120]), np.array([38, 255, 255]))
        # 3) RED 마스크: Hue 0~12 또는 165~180, 높은 채도
        mask_red1 = cv2.inRange(hsv, np.array([0, 110, 90]), np.array([12, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([165, 110, 90]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        combined = cv2.bitwise_or(mask_white, cv2.bitwise_or(mask_yellow, mask_red))
        combined = cv2.bitwise_and(combined, roi_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[Tuple[float, BBoxXYXY]] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 35.0 or area > 8500.0:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 6 or h < 6 or w > 120 or h > 120:
                continue
            aspect = float(w) / float(max(1, h))
            if not (0.45 <= aspect <= 2.2):
                continue
            perimeter = max(1.0, cv2.arcLength(cnt, True))
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < 0.30:
                continue

            score = area * min(1.0, circularity)
            candidates.append((score, (float(x), float(y), float(x + w), float(y + h))))

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected: List[BBoxXYXY] = []
        for _, bbox in candidates:
            cx = 0.5 * (bbox[0] + bbox[2])
            cy = 0.5 * (bbox[1] + bbox[3])
            if all(
                math.hypot(cx - 0.5 * (b[0] + b[2]), cy - 0.5 * (b[1] + b[3])) >= 14.0
                for b in selected
            ):
                selected.append(bbox)
            if len(selected) >= 3:
                break

        return selected

    def _associate_ocsort(
        self, frame_bgr: np.ndarray, bboxes: List[BBoxXYXY]
    ) -> List[Dict[str, Any]]:
        """
        OC-SORT의 Observation-Centric Momentum(OCM) 원리를 적용하여
        기존 트랙과 현재 검출 바운딩 박스를 매칭하고 track_id 리스트를 반환합니다.
        """
        if not bboxes:
            return []

        det_centers: List[Point2D] = [
            (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])) for b in bboxes
        ]
        det_colors: List[BallColor] = [
            OCSortTrackerAdapter.classify_roi_hsv_histogram(frame_bgr, b)[0]
            for b in bboxes
        ]

        if not self._tracks:
            output: List[Dict[str, Any]] = []
            for idx, bbox in enumerate(bboxes):
                tid = self._next_track_id
                self._next_track_id += 1
                tr = OCSortTrackState(
                    track_id=tid,
                    bbox=bbox,
                    center_px=det_centers[idx],
                    color_hint=det_colors[idx],
                    age=1,
                )
                tr.history_px.append(det_centers[idx])
                self._tracks[tid] = tr
                output.append(
                    {
                        "track_id": tid,
                        "bbox": bbox,
                        "color": det_colors[idx],
                    }
                )
            return output

        # 비용 행렬 계산: 위치 거리 + OCM(진행 방향 일관성) + 색상 힌트 페널티
        track_ids = list(self._tracks.keys())
        used_tracks = set()
        used_dets = set()
        pairs: List[Tuple[float, int, int]] = []

        for t_idx, tid in enumerate(track_ids):
            tr = self._tracks[tid]
            pred_x = tr.center_px[0] + tr.velocity_px[0]
            pred_y = tr.center_px[1] + tr.velocity_px[1]
            v_norm = math.hypot(tr.velocity_px[0], tr.velocity_px[1])

            for d_idx, d_center in enumerate(det_centers):
                dist = math.hypot(d_center[0] - pred_x, d_center[1] - pred_y)

                # OCM 방향 일관성 보정
                ocm_penalty = 0.0
                if v_norm >= 3.0 and len(tr.history_px) >= 2:
                    dx = d_center[0] - tr.center_px[0]
                    dy = d_center[1] - tr.center_px[1]
                    d_norm = math.hypot(dx, dy)
                    if d_norm >= 2.0:
                        cos_sim = (
                            tr.velocity_px[0] * dx + tr.velocity_px[1] * dy
                        ) / (v_norm * d_norm + 1e-6)
                        ocm_penalty = (1.0 - cos_sim) * 18.0

                color_penalty = 0.0 if tr.color_hint == det_colors[d_idx] else 45.0
                total_cost = dist + ocm_penalty + color_penalty
                pairs.append((total_cost, tid, d_idx))

        pairs.sort(key=lambda item: item[0])
        matched_output: List[Dict[str, Any]] = []

        for _, tid, d_idx in pairs:
            if tid in used_tracks or d_idx in used_dets:
                continue
            used_tracks.add(tid)
            used_dets.add(d_idx)
            tr = self._tracks[tid]
            new_c = det_centers[d_idx]
            vx = new_c[0] - tr.center_px[0]
            vy = new_c[1] - tr.center_px[1]
            tr.center_px = new_c
            tr.velocity_px = (0.6 * vx + 0.4 * tr.velocity_px[0], 0.6 * vy + 0.4 * tr.velocity_px[1])
            tr.bbox = bboxes[d_idx]
            tr.color_hint = det_colors[d_idx]
            tr.age += 1
            tr.missed_frames = 0
            tr.history_px.append(new_c)
            matched_output.append(
                {
                    "track_id": tid,
                    "bbox": bboxes[d_idx],
                    "color": det_colors[d_idx],
                }
            )

        # 미매칭 신규 검출 처리
        for d_idx, bbox in enumerate(bboxes):
            if d_idx in used_dets:
                continue
            tid = self._next_track_id
            self._next_track_id += 1
            tr = OCSortTrackState(
                track_id=tid,
                bbox=bbox,
                center_px=det_centers[d_idx],
                color_hint=det_colors[d_idx],
                age=1,
            )
            tr.history_px.append(det_centers[d_idx])
            self._tracks[tid] = tr
            matched_output.append(
                {
                    "track_id": tid,
                    "bbox": bbox,
                    "color": det_colors[d_idx],
                }
            )

        return matched_output


class BilliardDemoVisualizer:
    """
    `three_cushion_vision` 엔진(TableHomography + OCSortTrackerAdapter + ThreeCushionScoringFSM + ScoreboardBridge)과
    OpenCV 비디오 프레임(1920x1080 권장)을 결합하여:
    1) 우측 상단 실시간 2D 미니맵(PIP)
    2) 현재 쿠션 카운트 및 FSM 상태 HUD
    3) 메인 카메라 & 미니맵 공 궤적 라인(Trajectory Lines)
    4) 득점/노카운트/미인지 득점 자동 보정 판정 배너
    를 오버레이하여 렌더링하는 메인 비주얼라이저 클래스입니다.
    """

    BALL_BGR_MAP: Dict[BallColor, Tuple[int, int, int]] = {
        BallColor.WHITE: (245, 245, 245),
        BallColor.YELLOW: (20, 225, 250),
        BallColor.RED: (35, 35, 235),
    }

    def __init__(
        self,
        homography: Optional[TableHomography] = None,
        config: VisionConfig = DEFAULT_CONFIG,
        yolo_weights: Optional[str] = None,
        initial_cue_ball: BallColor = BallColor.WHITE,
        match_manager: Optional[Any] = None,
    ) -> None:
        self.config: VisionConfig = config
        self.homography: TableHomography = homography or TableHomography(
            src_points_px=[
                (180.0, 140.0),
                (1740.0, 140.0),
                (1820.0, 960.0),
                (100.0, 960.0),
            ],
            camera_tilt_ratio=config.CAMERA_TILT_RATIO,
        )
        self.detector = UltralyticsOCSortDetector(
            yolo_weights=yolo_weights,
            table_roi_px=self.homography.src_points_px,
        )
        self.tracker_adapter = OCSortTrackerAdapter(
            homography=self.homography,
            config=config,
        )
        self.fsm = ThreeCushionScoringFSM(
            initial_cue_ball=initial_cue_ball,
            config=config,
            auto_compensate_missed_score=True,
        )
        if match_manager is None:
            try:
                from cue24.services.match_manager import MatchManager

                match_manager = MatchManager(p1_name="PLAYER 1", p2_name="PLAYER 2")
            except Exception:
                match_manager = None

        self.match_manager = match_manager
        self.bridge = ScoreboardBridge(match_manager=self.match_manager)

        # 궤적 히스토리 (mm 좌표계 및 픽셀 좌표계)
        self.trajectories_mm: Dict[BallColor, Deque[Point2D]] = {
            BallColor.WHITE: deque(maxlen=180),
            BallColor.YELLOW: deque(maxlen=180),
            BallColor.RED: deque(maxlen=180),
        }
        self.cushion_impact_points_mm: List[Tuple[int, Point2D, str]] = []

        # 배너 표시용 상태
        self.last_shot_result: Optional[ShotResult] = None
        self.banner_text: str = "READY - WAITING FOR STROKE"
        self.banner_subtext: str = "Vision-Only 60fps CCD + Micro-Impulse + FSM Active"
        self.banner_color_bgr: Tuple[int, int, int] = (70, 130, 180)
        self.banner_hold_frames: int = 0
        self.frame_index: int = 0

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
        override_detections: Optional[List[Dict[str, Any]]] = None,
        enable_ema: bool = True,
    ) -> Tuple[np.ndarray, Optional[ShotResult]]:
        """
        단일 비디오 프레임을 입력받아 검출 -> 호모그래피 변환 -> 물리/FSM 판정 ->
        2D 미니맵(PIP), 쿠션 카운트, 궤적 라인, 득점 판정 배너 오버레이까지 완료된 프레임을 반환합니다.
        """
        self.frame_index += 1
        prev_state = self.fsm.state
        prev_cushions = self.fsm.cushion_count

        # 1. YOLO + OC-SORT 검출 또는 외부 주입 검출값 사용
        raw_dets = (
            override_detections
            if override_detections is not None
            else self.detector.detect_and_track(frame_bgr)
        )

        # 2. OCSortTrackerAdapter를 통해 mm 좌표 변환 + HSV 색상 락킹 + 적응형 EMA 속도 계산
        tracked_balls = self.tracker_adapter.update_from_detections(
            detections=raw_dets,
            timestamp=timestamp,
            bgr_frame=frame_bgr if override_detections is None else None,
            enable_ema=enable_ema,
        )

        # 3. ThreeCushionScoringFSM 업데이트
        shot_result = self.fsm.update(balls=tracked_balls, timestamp=timestamp)

        # 새 이닝 출발 시 이전 궤적 초기화 및 미인지 득점 자동 보정 배너 체크
        if prev_state == ShotState.IDLE and self.fsm.state != ShotState.IDLE:
            for q in self.trajectories_mm.values():
                q.clear()
            self.cushion_impact_points_mm.clear()

            if self.fsm.retroactive_compensated_in_shot:
                if self.match_manager is not None and hasattr(
                    self.match_manager, "reconcile_cue_ball_stroke"
                ):
                    self.match_manager.reconcile_cue_ball_stroke(
                        stroked_cue_color=self.fsm.assigned_cue_ball.value,
                        compensate_missed_score=True,
                    )
                self.banner_text = (
                    f"AUTO-COMPENSATED (+1 PT) | {self.fsm.assigned_cue_ball.value} CUE VERIFIED"
                )
                self.banner_subtext = (
                    "Previous missed score automatically restored by cue-ball color detection"
                )
                self.banner_color_bgr = (0, 165, 255)  # 골드/앰버 강조 배너
                self.banner_hold_frames = 120

        # 주행 중 궤적 및 쿠션 충돌 지점 누적
        for color, ball in tracked_balls.items():
            self.trajectories_mm[color].append((ball.x, ball.y))

        if self.fsm.cushion_count > prev_cushions and self.fsm.assigned_cue_ball in tracked_balls:
            cue_b = tracked_balls[self.fsm.assigned_cue_ball]
            last_ev = self.fsm.events[-1] if self.fsm.events else "RAIL"
            self.cushion_impact_points_mm.append(
                (self.fsm.cushion_count, (cue_b.x, cue_b.y), last_ev)
            )

        # 주행 중 실시간 상태 배너 갱신
        if self.fsm.state != ShotState.IDLE and self.banner_hold_frames <= 0:
            if self.fsm.state == ShotState.VALID_SCORE_PENDING:
                tag = "BANK SHOT" if self.fsm.is_bank_shot else "REGULAR 3-CUSHION"
                self.banner_text = f"VALID SCORE PENDING ({tag}) - WAITING FOR STOP"
                self.banner_subtext = (
                    f"Cue: {self.fsm.assigned_cue_ball.value} | "
                    f"Cushions: {self.fsm.cushion_count} | 1st & 2nd Target Hit Confirmed"
                )
                self.banner_color_bgr = (45, 175, 60)
            elif self.fsm.state == ShotState.NO_SCORE_LOCKED:
                self.banner_text = f"NO SCORE LOCKED ({self.fsm.fail_reason})"
                self.banner_subtext = (
                    f"Cue: {self.fsm.assigned_cue_ball.value} | "
                    f"Cushions: {self.fsm.cushion_count} | Early Kiss or Foul Detected"
                )
                self.banner_color_bgr = (40, 45, 210)
            else:
                self.banner_text = (
                    f"IN PLAY - {self.fsm.assigned_cue_ball.value} CUE BALL TRACKING"
                )
                h1 = self.fsm.first_target_ball.value if self.fsm.first_target_ball else "NONE"
                h2 = self.fsm.second_target_ball.value if self.fsm.second_target_ball else "NONE"
                self.banner_subtext = (
                    f"Cushions: {self.fsm.cushion_count} | 1st Target: {h1} | 2nd Target: {h2}"
                )
                self.banner_color_bgr = (180, 105, 30)

        # 샷 종료 확정(ShotResult 반환) 시 ScoreboardBridge 발행 및 최종 배너 고정
        if shot_result is not None:
            self.last_shot_result = shot_result
            self.bridge.publish_shot_result(shot_result)
            self.banner_hold_frames = 150  # 2.5초간 최종 판정 배너 유지

            if shot_result.scored:
                shot_label = "BANK SHOT (2 PTS)" if shot_result.is_bank_shot else "3-CUSHION SCORE (+1 PT)"
                self.banner_text = f"SCORE CONFIRMED! [{shot_label}]"
                self.banner_subtext = (
                    f"Cue: {shot_result.cue_ball.value} | "
                    f"Cushions before 2nd hit: {shot_result.cushions_before_second_ball} | "
                    f"Total Cushions: {shot_result.total_cushions}"
                )
                self.banner_color_bgr = (35, 165, 45)
            else:
                self.banner_text = f"NO SCORE [{shot_result.fail_reason}] - TURN SWITCHED"
                self.banner_subtext = (
                    f"Cue: {shot_result.cue_ball.value} -> Next Cue: {self.fsm.assigned_cue_ball.value} | "
                    f"Cushions before 2nd hit: {shot_result.cushions_before_second_ball}"
                )
                self.banner_color_bgr = (35, 40, 200)

        if self.banner_hold_frames > 0:
            self.banner_hold_frames -= 1

        annotated = self.render_overlay_frame(frame_bgr, tracked_balls, timestamp)
        return annotated, shot_result

    def render_overlay_frame(
        self,
        frame_bgr: np.ndarray,
        tracked_balls: Dict[BallColor, TrackedBall],
        timestamp: float,
    ) -> np.ndarray:
        """
        원본 BGR 프레임 위에 1) 메인 공 궤적 라인, 2) 우측 상단 2D 미니맵(PIP),
        3) 좌측 상단 쿠션 카운트/점수판 HUD, 4) 하단 득점 판정 배너를 렌더링합니다.
        """
        canvas = frame_bgr.copy()
        h_img, w_img = canvas.shape[:2]

        # 1. 메인 카메라 뷰: 당구대 호모그래피 경계선 및 공 궤적 라인 렌더링
        self._draw_main_camera_trajectories(canvas, tracked_balls)

        # 2. 우측 상단 PIP: 실시간 2D 당구대 미니맵 (2840 x 1420 mm 축소 투영)
        self._draw_top_right_minimap_pip(canvas, tracked_balls, w_img, h_img)

        # 3. 좌측 상단 HUD: P1(WHITE) vs P2(YELLOW) 스코어, 현재 쿠션 카운트, FSM 상태
        self._draw_top_left_cushion_and_score_hud(canvas, timestamp)

        # 4. 하단 중앙: 실시간 득점 판정 배너 (Scoring Decision Banner)
        self._draw_bottom_decision_banner(canvas, w_img, h_img)

        return canvas

    def _draw_main_camera_trajectories(
        self,
        canvas: np.ndarray,
        tracked_balls: Dict[BallColor, TrackedBall],
    ) -> None:
        """메인 카메라 화면 위에 당구대 내측 경계, 공 이동 궤적 폴리라인, 바운딩 박스/속도를 그립니다."""
        # 당구대 내측 쿠션 폴리곤 표시
        if len(self.homography.src_points_px) >= 4:
            table_poly = np.array(
                self.homography.src_points_px[:4], dtype=np.int32
            ).reshape((-1, 1, 2))
            cv2.polylines(canvas, [table_poly], isClosed=True, color=(180, 140, 40), thickness=2)

        # 각 공의 이동 궤적 폴리라인 (역호모그래피 변환으로 픽셀 투영)
        for color, pts_mm in self.trajectories_mm.items():
            if len(pts_mm) < 2:
                continue
            pts_px = [
                self.homography.table_mm_to_pixel(mx, my) for (mx, my) in pts_mm
            ]
            poly_px = np.array(pts_px, dtype=np.int32).reshape((-1, 1, 2))
            line_bgr = self.BALL_BGR_MAP.get(color, (200, 200, 200))
            thickness = 3 if color == self.fsm.assigned_cue_ball else 1
            cv2.polylines(
                canvas,
                [poly_px],
                isClosed=False,
                color=line_bgr,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

        # 쿠션 임팩트 마커 (C1, C2, C3...)
        for c_num, (cx_mm, cy_mm), _ in self.cushion_impact_points_mm:
            px, py = self.homography.table_mm_to_pixel(cx_mm, cy_mm)
            ipx, ipy = int(round(px)), int(round(py))
            cv2.circle(canvas, (ipx, ipy), 9, (0, 255, 180), -1, cv2.LINE_AA)
            cv2.circle(canvas, (ipx, ipy), 11, (15, 20, 15), 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"C{c_num}",
                (ipx - 10, ipy - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 180),
                2,
                cv2.LINE_AA,
            )

        # 현재 추적된 3개 공의 위치 및 속도 표시
        for color, ball in tracked_balls.items():
            px, py = self.homography.table_mm_to_pixel(ball.x, ball.y)
            cx, cy = int(round(px)), int(round(py))
            bgr = self.BALL_BGR_MAP[color]
            is_cue = color == self.fsm.assigned_cue_ball

            radius = 18 if is_cue else 15
            cv2.circle(canvas, (cx, cy), radius, bgr, 2, cv2.LINE_AA)
            if is_cue:
                cv2.circle(canvas, (cx, cy), radius + 5, (0, 255, 255), 2, cv2.LINE_AA)

            label = f"{color.value[:1]}({int(round(ball.speed))}mm/s)"
            cv2.putText(
                canvas,
                label,
                (cx - 28, cy - radius - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                bgr,
                1,
                cv2.LINE_AA,
            )

    def _draw_top_right_minimap_pip(
        self,
        canvas: np.ndarray,
        tracked_balls: Dict[BallColor, TrackedBall],
        w_img: int,
        h_img: int,
    ) -> None:
        """
        우측 상단 PIP(Picture-in-Picture) 영역에 국제식 대대(2840x1420mm) 실시간 2D 미니맵을 렌더링합니다.
        """
        pip_w = min(420, max(280, int(w_img * 0.28)))
        pip_h = int(pip_w * 0.5)  # 2:1 국제식 대대 비율
        margin = 20
        x0 = max(10, w_img - pip_w - margin)
        y0 = margin + 12

        # 외곽 우드/다크 프레임 베젤 (반투명 블렌딩)
        bezel_pad = 12
        bx1 = max(0, x0 - bezel_pad)
        by1 = max(0, y0 - bezel_pad - 22)
        bx2 = min(w_img, x0 + pip_w + bezel_pad)
        by2 = min(h_img, y0 + pip_h + bezel_pad)

        sub = canvas[by1:by2, bx1:bx2]
        bezel_bg = np.full_like(sub, (28, 32, 42))
        cv2.addWeighted(bezel_bg, 0.85, sub, 0.15, 0.0, sub)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (90, 110, 140), 2)

        # 미니맵 타이틀
        cv2.putText(
            canvas,
            "2D TOP-VIEW MINIMAP (2840 x 1420 mm)",
            (x0, y0 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 235, 255),
            1,
            cv2.LINE_AA,
        )

        # 파란색 국제식 대대 라사지 배경
        cv2.rectangle(canvas, (x0, y0), (x0 + pip_w, y0 + pip_h), (175, 85, 20), -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + pip_w, y0 + pip_h), (230, 230, 230), 2)

        # 쿠션 임계선 가이드 (44.75mm)
        scale_x = pip_w / self.config.TABLE_WIDTH_MM
        scale_y = pip_h / self.config.TABLE_HEIGHT_MM
        rail_px_x = max(3, int(round(self.config.RAIL_ZONE_MARGIN_MM * scale_x)))
        rail_px_y = max(3, int(round(self.config.RAIL_ZONE_MARGIN_MM * scale_y)))
        cv2.rectangle(
            canvas,
            (x0 + rail_px_x, y0 + rail_px_y),
            (x0 + pip_w - rail_px_x, y0 + pip_h - rail_px_y),
            (195, 130, 65),
            1,
        )

        # 다이아몬드 포인트 표시 (장축 8등분, 단축 4등분)
        for i in range(1, 8):
            dx = x0 + int(round((pip_w * i) / 8.0))
            cv2.circle(canvas, (dx, y0 - 4), 2, (210, 210, 210), -1)
            cv2.circle(canvas, (dx, y0 + pip_h + 4), 2, (210, 210, 210), -1)
        for j in range(1, 4):
            dy = y0 + int(round((pip_h * j) / 4.0))
            cv2.circle(canvas, (x0 - 4, dy), 2, (210, 210, 210), -1)
            cv2.circle(canvas, (x0 + pip_w + 4, dy), 2, (210, 210, 210), -1)

        def _mm_to_pip(mx: float, my: float) -> Tuple[int, int]:
            px = x0 + int(round(np.clip(mx, 0.0, self.config.TABLE_WIDTH_MM) * scale_x))
            py = y0 + int(round(np.clip(my, 0.0, self.config.TABLE_HEIGHT_MM) * scale_y))
            return px, py

        # 미니맵 위 궤적 라인 렌더링
        for color, pts_mm in self.trajectories_mm.items():
            if len(pts_mm) < 2:
                continue
            pip_pts = np.array(
                [_mm_to_pip(mx, my) for (mx, my) in pts_mm], dtype=np.int32
            ).reshape((-1, 1, 2))
            line_bgr = self.BALL_BGR_MAP[color]
            thick = 2 if color == self.fsm.assigned_cue_ball else 1
            cv2.polylines(
                canvas, [pip_pts], isClosed=False, color=line_bgr, thickness=thick, lineType=cv2.LINE_AA
            )

        # 미니맵 위 쿠션 충돌 마커
        for c_num, (cx_mm, cy_mm), _ in self.cushion_impact_points_mm:
            cpx, cpy = _mm_to_pip(cx_mm, cy_mm)
            cv2.circle(canvas, (cpx, cpy), 4, (0, 255, 160), -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"C{c_num}",
                (cpx + 5, cpy - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (0, 255, 160),
                1,
                cv2.LINE_AA,
            )

        # 미니맵 위 당구공 3개 렌더링
        ball_r_px = max(5, int(round(self.config.BALL_RADIUS_MM * scale_x * 1.45)))
        for color, ball in tracked_balls.items():
            bpx, bpy = _mm_to_pip(ball.x, ball.y)
            bgr = self.BALL_BGR_MAP[color]
            cv2.circle(canvas, (bpx, bpy), ball_r_px, bgr, -1, cv2.LINE_AA)
            cv2.circle(canvas, (bpx, bpy), ball_r_px, (15, 15, 15), 1, cv2.LINE_AA)
            if color == self.fsm.assigned_cue_ball:
                cv2.circle(canvas, (bpx, bpy), ball_r_px + 3, (0, 255, 255), 1, cv2.LINE_AA)

    def _draw_top_left_cushion_and_score_hud(
        self, canvas: np.ndarray, timestamp: float
    ) -> None:
        """좌측 상단에 P1/P2 점수, 현재 수구 색상, 실시간 쿠션 카운트 게이지(0~3+), FSM 상태를 표시합니다."""
        x0, y0 = 20, 20
        hud_w, hud_h = 430, 155

        sub = canvas[y0 : y0 + hud_h, x0 : x0 + hud_w]
        bg = np.full_like(sub, (20, 24, 32))
        cv2.addWeighted(bg, 0.84, sub, 0.16, 0.0, sub)
        cv2.rectangle(canvas, (x0, y0), (x0 + hud_w, y0 + hud_h), (80, 110, 145), 2)

        # P1(WHITE) vs P2(YELLOW) 스코어 추출
        p1_score, p2_score, inning_no, active_pid = 0, 0, 1, "P1"
        if self.match_manager is not None:
            p1_score = self.match_manager.players["P1"].score
            p2_score = self.match_manager.players["P2"].score
            inning_no = self.match_manager.current_inning
            active_pid = self.match_manager.active_player_id

        cv2.putText(
            canvas,
            f"P1(WHITE): {p1_score:02d}   |   INNING {inning_no}   |   P2(YELLOW): {p2_score:02d}",
            (x0 + 14, y0 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (240, 245, 255),
            2,
            cv2.LINE_AA,
        )

        cue_col = self.fsm.assigned_cue_ball.value
        fsm_name = self.fsm.state.value
        cv2.putText(
            canvas,
            f"ACTIVE: {active_pid} ({cue_col} CUE)  |  FSM: {fsm_name}  |  T={timestamp:.2f}s",
            (x0 + 14, y0 + 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (180, 220, 250),
            1,
            cv2.LINE_AA,
        )

        # 실시간 쿠션 카운트 게이지 박스 (CUSHION COUNT: 1, 2, 3+)
        c_cnt = self.fsm.cushion_count
        cv2.putText(
            canvas,
            f"CUSHION COUNT: {c_cnt}",
            (x0 + 14, y0 + 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 180) if c_cnt >= 3 else (255, 215, 90),
            2,
            cv2.LINE_AA,
        )

        # 1, 2, 3 쿠션 램프 아이콘
        for idx in range(1, 4):
            bx = x0 + 225 + (idx - 1) * 62
            by = y0 + 70
            active_lamp = c_cnt >= idx
            lamp_color = (40, 200, 95) if (active_lamp and c_cnt >= 3) else (
                (0, 190, 255) if active_lamp else (55, 60, 72)
            )
            cv2.rectangle(canvas, (bx, by), (bx + 52, by + 30), lamp_color, -1)
            cv2.rectangle(canvas, (bx, by), (bx + 52, by + 30), (210, 215, 225), 1)
            cv2.putText(
                canvas,
                f"{idx}C",
                (bx + 14, by + 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (15, 15, 15) if active_lamp else (150, 155, 165),
                2,
                cv2.LINE_AA,
            )

        # 제1목적구 / 제2목적구 타격 상태 표시
        t1_str = self.fsm.first_target_ball.value if self.fsm.first_target_ball else "---"
        t2_str = self.fsm.second_target_ball.value if self.fsm.second_target_ball else "---"
        cv2.putText(
            canvas,
            f"1st Object Hit: {t1_str}   |   2nd Object Hit: {t2_str}   |   Compensated: {self.fsm.total_retroactive_compensations}",
            (x0 + 14, y0 + 134),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (210, 220, 230),
            1,
            cv2.LINE_AA,
        )

    def _draw_bottom_decision_banner(
        self, canvas: np.ndarray, w_img: int, h_img: int
    ) -> None:
        """화면 하단 중앙에 3쿠션 득점 판정 결과 및 상태 배너를 렌더링합니다."""
        banner_w = min(860, w_img - 40)
        banner_h = 68
        bx1 = (w_img - banner_w) // 2
        by1 = h_img - banner_h - 18
        bx2 = bx1 + banner_w
        by2 = by1 + banner_h

        sub = canvas[by1:by2, bx1:bx2]
        overlay = np.full_like(sub, self.banner_color_bgr)
        cv2.addWeighted(overlay, 0.88, sub, 0.12, 0.0, sub)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (240, 245, 255), 2)

        cv2.putText(
            canvas,
            self.banner_text,
            (bx1 + 22, by1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            self.banner_subtext,
            (bx1 + 22, by1 + 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (230, 240, 250),
            1,
            cv2.LINE_AA,
        )


def render_synthetic_billiard_frame(
    homography: TableHomography,
    balls_mm: Dict[BallColor, Point2D],
    width: int = 1280,
    height: int = 720,
) -> np.ndarray:
    """
    테스트 또는 데모 비디오 생성용: 주어진 3구 물리 좌표(mm)를 카메라 시점 BGR 프레임으로 렌더링합니다.
    """
    frame = np.full((height, width, 3), (24, 28, 36), dtype=np.uint8)
    if len(homography.src_points_px) >= 4:
        poly = np.array(homography.src_points_px[:4], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(frame, [poly], (175, 85, 20))
        cv2.polylines(frame, [poly], True, (45, 95, 150), 8, cv2.LINE_AA)

    color_map = BilliardDemoVisualizer.BALL_BGR_MAP
    for color, (mx, my) in balls_mm.items():
        px, py = homography.table_mm_to_pixel(mx, my)
        cx, cy = int(round(px)), int(round(py))
        cv2.circle(frame, (cx, cy), 14, color_map[color], -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 14, (20, 20, 20), 1, cv2.LINE_AA)

    return frame


def run_video_demo(
    video_path: Optional[str] = None,
    output_path: Optional[str] = None,
    yolo_weights: Optional[str] = None,
    max_frames: int = 180,
    display: bool = False,
) -> Dict[str, Any]:
    """
    입력 MP4 비디오 파일(또는 파일 부재 시 내장 3쿠션 물리 시뮬레이션 스트림)을 읽어
    BilliardDemoVisualizer 오버레이를 수행하고 결과 통계 및 출력 MP4를 생성합니다.
    """
    visualizer = BilliardDemoVisualizer(yolo_weights=yolo_weights)
    writer: Optional[cv2.VideoWriter] = None
    processed_frames = 0
    shot_results: List[ShotResult] = []

    use_real_video = bool(video_path and os.path.isfile(video_path))
    cap: Optional[cv2.VideoCapture] = None
    fps = 60.0

    if use_real_video and video_path is not None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            use_real_video = False
        else:
            raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
            fps = raw_fps if 15.0 <= raw_fps <= 120.0 else 60.0

    dt = 1.0 / fps

    try:
        if use_real_video and cap is not None:
            while processed_frames < max_frames:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                t_sec = processed_frames * dt
                annotated, res = visualizer.process_frame(frame, timestamp=t_sec)
                if res is not None:
                    shot_results.append(res)

                if output_path and writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                if writer is not None:
                    writer.write(annotated)

                if display:
                    cv2.imshow("Three-Cushion Vision Demo Visualizer", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                processed_frames += 1
        else:
            # 내장 60fps 일반 3쿠션 득점 데모 궤적 생성 및 렌더링
            homography = TableHomography(
                src_points_px=[
                    (140.0, 110.0),
                    (1140.0, 110.0),
                    (1200.0, 630.0),
                    (80.0, 630.0),
                ]
            )
            visualizer = BilliardDemoVisualizer(
                homography=homography, yolo_weights=yolo_weights
            )
            waypoints = [
                (400.0, 700.0),
                (800.0, 500.0),
                (1500.0, 32.0),
                (2808.0, 650.0),
                (1900.0, 1388.0),
                (1200.0, 900.0),
                (1100.0, 820.0),
            ]
            cue_pts: List[Point2D] = [waypoints[0]]
            for i in range(len(waypoints) - 1):
                x0, y0 = waypoints[i]
                x1, y1 = waypoints[i + 1]
                for s in range(1, 9):
                    a = s / 8.0
                    cue_pts.append((x0 + (x1 - x0) * a, y0 + (y1 - y0) * a))

            total_sim = min(max_frames, len(cue_pts) + 28)
            for idx in range(total_sim):
                t_sec = idx * (1.0 / 60.0)
                c_pt = cue_pts[min(idx, len(cue_pts) - 1)]
                r_pt = (800.0, 450.0) if idx < 9 else (820.0, 380.0)
                y_pt = (1200.0, 850.0) if idx < 41 else (1160.0, 800.0)

                balls_mm = {
                    BallColor.WHITE: c_pt,
                    BallColor.RED: r_pt,
                    BallColor.YELLOW: y_pt,
                }
                synth_frame = render_synthetic_billiard_frame(homography, balls_mm)
                override_dets = [
                    {"color": BallColor.WHITE, "x_mm": c_pt[0], "y_mm": c_pt[1], "track_id": 1},
                    {"color": BallColor.RED, "x_mm": r_pt[0], "y_mm": r_pt[1], "track_id": 2},
                    {"color": BallColor.YELLOW, "x_mm": y_pt[0], "y_mm": y_pt[1], "track_id": 3},
                ]
                annotated, res = visualizer.process_frame(
                    synth_frame,
                    timestamp=t_sec,
                    override_detections=override_dets,
                    enable_ema=False,
                )
                if res is not None:
                    shot_results.append(res)

                if output_path and writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_path, fourcc, 60.0, (w, h))
                if writer is not None:
                    writer.write(annotated)

                if display:
                    cv2.imshow("Three-Cushion Vision Demo Visualizer", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                processed_frames += 1
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    return {
        "processed_frames": processed_frames,
        "shot_results": shot_results,
        "output_path": output_path,
        "detector_backend": visualizer.detector.backend_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-Cushion Vision-Only 60fps Demo Visualizer (2D PIP Minimap + Trajectory + Cushion HUD + Banner)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=r"J:\내 드라이브\L-billiard.mp4",
        help="입력 MP4 영상 경로 (존재하지 않으면 내장 60fps 3쿠션 시뮬레이션 스트림 자동 실행)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="오버레이 렌더링 결과를 저장할 출력 MP4 경로 (예: demo_output.mp4)",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default=None,
        help="Ultralytics YOLO 가중치 경로 (예: yolo11s.pt)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=180,
        help="처리할 최대 프레임 수 (기본값: 180)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="OpenCV 윈도우(cv2.imshow)로 실시간 화면 출력 여부",
    )
    args = parser.parse_args()

    summary = run_video_demo(
        video_path=args.video,
        output_path=args.output,
        yolo_weights=args.yolo_weights,
        max_frames=args.max_frames,
        display=args.display,
    )
    print(
        f"[DemoVisualizer 완료] 처리 프레임={summary['processed_frames']}, "
        f"검출 백엔드={summary['detector_backend']}, "
        f"확정된 샷 수={len(summary['shot_results'])}"
    )


if __name__ == "__main__":
    main()
