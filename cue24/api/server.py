"""
CUE24 FastAPI 실시간 WebSocket, 라이브캠/자동 리플레이 스트리밍 및 REST API 관제 서버
파일 경로: cue24/api/server.py

1. WebSocket 엔드포인트 (/ws/scoreboard) 및 ConnectionManager를 통한 실시간 태블릿 브로드캐스트
2. LiveStreamReader 카메라/시뮬레이션 영상 루프와 Step 2~4 비전·물리·VAR 엔진 및 Step 7 ReplayRingBuffer 통합
3. 실시간 이벤트 송출:
   - SCORE_UPDATED (득점 및 점수판 갱신)
   - TURN_CHANGED (공타 시 턴 및 수구 색상 전환)
   - VAR_ALERT (미세 접촉 또는 쿠션 더블터치 현미경 판독 팝업 트리거)
   - AUTO_REPLAY_READY (3공 완전 정지 시 직전 샷 자동 리플레이 전환 신호)
4. 라이브캠 & 직전 샷 리플레이 MJPEG/프레임 스트리밍 엔드포인트 (/api/video/live, /api/video/replay)
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ball_identity_tracker import LabHungarianBallIdentifier
from cue24.config.settings import settings
from cue24.core.color_locker import ColorLocker
from cue24.core.physics_tracker import (
    BallCandidateDetector,
    PhysicsTracker,
    SpinDetector,
)
from cue24.core.replay_buffer import ReplayRingBuffer, SyntheticCamGenerator
from cue24.core.scoring_engine import ScoringEngine
from cue24.core.table_calibrator import TableCalibrator
from cue24.core.var_referee import VARReferee
from cue24.mcp.billiard_physics_mcp import billiard_mcp_server, physics_ai_engine
from cue24.services.match_manager import MatchManager, MatchState
from three_cushion_vision.billiard_nanogpt import (
    OcclusionTrajectoryRestorer,
    get_or_train_billiard_nanogpt,
)
from three_cushion_vision.causal_kiss_fsm import (
    BallColor as VisionBallColor,
    CausalKissSafeFSM,
    ShotResult as CausalShotResult,
    ShotState as CausalShotState,
)
from three_cushion_vision.config import BallColor as KDColor
from three_cushion_vision.kinematics_detector import KinematicsDetector
from three_cushion_vision.tracker_adapter import TrackedBall

Point2D = Tuple[float, float]


class ConnectionManager:
    """
    연결된 20인치 태블릿 점수판 브라우저 클라이언트들을 관리하고,
    경기 상태 및 VAR 판독/자동 리플레이 이벤트를 실시간 브로드캐스트하는 웹소켓 매니저입니다.
    """

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """신규 태블릿 클라이언트 웹소켓 연결 수락 및 등록"""
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        """연결 해제된 클라이언트 제거"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_event(
        self, event_type: str, data: Dict[str, Any]
    ) -> None:
        """
        연결된 모든 태블릿 클라이언트에 표준 이벤트 포맷({type, data})으로 JSON 메시지를 브로드캐스트합니다.
        """
        message = {
            "type": event_type,
            "data": data,
        }
        stale_clients: List[WebSocket] = []
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                stale_clients.append(connection)

        for stale in stale_clients:
            self.disconnect(stale)


import ctypes
import math
import sys
import time
import cv2
from cue24.core.physics_tracker import BulletproofBallTracker

# Windows 환경에서 asyncio.sleep의 기본 타이머 해상도(15.6ms)를 1.0ms 고정밀 모드로 전환하여
# 실시간 30.0fps(33.33ms 주기) 1.0배속 스트리밍 지연을 원천 제거합니다.
if sys.platform == "win32":
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass

COLOR_KO_LABELS: Dict[str, str] = {
    "WHITE": "흰공",
    "YELLOW": "노란공",
    "RED": "빨간공",
}


class LiveStreamReader:
    """
    실제 당구대 카메라 스트림 또는 카메라 대용 테스트 녹화 영상("J:\\내 드라이브\\L-billiard.mp4")을
    원본 해상도(1920x1080) 크기·비율 변동 없이 실시간 1.0배속으로 공급하고,
    5단계 철통 3구 추적 알고리즘(BulletproofBallTracker + PhysicsTracker + ScoringEngine)을
    실시간 연동하는 파이프라인 리더입니다.
    """

    # 실제 카메라/비디오 비전 스트림에서 예비 큐질(그림자 흔들림)을 배제하고 실제 수구 출발만 감지하기 위한 임계값
    VISION_START_SPEED_TH: float = 4.5
    VISION_START_DISP_MM: float = 14.0
    MOTION_BLUR_CIRC_TH: float = 0.42
    # 테스트 녹화 영상(J:\내 드라이브\L-billiard.mp4)에서 경기 초기화/화면 리플레시 즉시 3구 색상 잠금 ->
    # 선공(흰공) 5쿠션 즉시 득점 -> 후공(노란공) 3쿠션 즉시 득점 -> 선공(흰공) 5쿠션 뱅크샷 즉시 득점이 이어지는 시작 프레임
    ACTIVE_SHOT_START_FRAME: int = 7920
    _INIT_VIDEO_CACHE: Dict[str, Dict[str, Any]] = {}
    _CACHED_NANOGPT_MODEL: Optional[Any] = None

    def __init__(
        self,
        match_manager: MatchManager,
        ws_manager: ConnectionManager,
        camera_index: Optional[int] = None,
        video_path: Optional[str] = None,
    ) -> None:
        self.match_manager: MatchManager = match_manager
        self.ws_manager: ConnectionManager = ws_manager
        self.camera_index: Optional[int] = camera_index
        self.video_path: str = video_path or settings.DEFAULT_VIDEO_PATH
        self.is_running: bool = False
        self.frame_counter: int = 0

        # Step 2 ~ Step 4 핵심 코어 모듈, 5단계 BulletproofBallTracker 및 Step 7 리플레이 버퍼 초기화
        default_corners = [
            (0.0, 0.0),
            (settings.TABLE_WIDTH_MM, 0.0),
            (settings.TABLE_WIDTH_MM, settings.TABLE_HEIGHT_MM),
            (0.0, settings.TABLE_HEIGHT_MM),
        ]
        self.calibrator = TableCalibrator(src_corners=default_corners)
        self.color_locker = ColorLocker()
        self.lab_hungarian_identifier = LabHungarianBallIdentifier()
        self.detector = BallCandidateDetector()
        self.tracker = PhysicsTracker()
        self.bulletproof_tracker = BulletproofBallTracker()
        self.spin_detector = SpinDetector()
        self.scoring_engine = ScoringEngine()
        self.var_referee = VARReferee()
        self.replay_buffer = ReplayRingBuffer(max_frames=300)
        self.cam_generator = SyntheticCamGenerator(width=960, height=480)

        # 비전 자동 캘리브레이션, 0.5x/0.25x 초고속 캐싱 행렬 및 정지 기준 위치 상태
        self._table_calibrated: bool = False
        self._cached_half_H: Optional[np.ndarray] = None
        self._cached_quarter_H: Optional[np.ndarray] = None
        self._morph_kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._morph_open_k5: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._channel_core_state: Dict[str, Dict[str, Any]] = {
            "WHITE": {"pos": None, "vel": np.zeros(2, dtype=np.float64), "missing": 0},
            "YELLOW": {"pos": None, "vel": np.zeros(2, dtype=np.float64), "missing": 0},
            "RED": {"pos": None, "vel": np.zeros(2, dtype=np.float64), "missing": 0},
        }
        self._vision_tracks_synced: bool = False
        self._rest_positions: Dict[int, Point2D] = {}
        self._missing_streak_by_id: Dict[int, int] = {}
        self._last_missing_wait_id: Optional[int] = None
        self.detected_ball_colors: Dict[int, str] = {
            1: "WHITE",
            2: "YELLOW",
            3: "RED",
        }

        # 기본 녹화 영상("J:\내 드라이브\L-billiard.mp4") 또는 카메라 원본 사이즈(1920x1080) 로더 초기화
        self._video_cap: Optional[cv2.VideoCapture] = None
        self.is_real_video_loaded: bool = False
        self.video_width: int = 1920
        self.video_height: int = 1080
        self.video_fps: float = 30.0
        self._latest_live_jpeg: Optional[bytes] = None
        self._latest_raw_frame: Optional[np.ndarray] = None

        # 실시간 1.0배속(Wall-Clock) 동기화 및 다중 클라이언트 중복 캡처 방지 타이밍 상태
        self._wall_clock_origin: float = time.perf_counter()
        self._frames_advanced_since_origin: int = 0
        self._last_frame_step_time: float = 0.0

        self._init_video_capture(self.video_path)

        # [Autoresearch 100% + BilliardNanoGPT + 3체 인과 키스 분리 엔진 통합]
        if LiveStreamReader._CACHED_NANOGPT_MODEL is None:
            LiveStreamReader._CACHED_NANOGPT_MODEL = get_or_train_billiard_nanogpt()
        self.nanogpt_model = LiveStreamReader._CACHED_NANOGPT_MODEL
        self.trajectory_restorers: Dict[int, OcclusionTrajectoryRestorer] = {
            1: OcclusionTrajectoryRestorer(self.nanogpt_model, device="cpu"),
            2: OcclusionTrajectoryRestorer(self.nanogpt_model, device="cpu"),
            3: OcclusionTrajectoryRestorer(self.nanogpt_model, device="cpu"),
        }
        self.causal_kiss_fsm = CausalKissSafeFSM(fps=max(15.0, self.video_fps))
        self._last_causal_shot_result: Optional[CausalShotResult] = None
        self.vision_kinematics = KinematicsDetector()

        self._cue_trajectory: List[Point2D] = []
        self._target_trajectories: Dict[int, List[Point2D]] = {}

        # 초기 트랙 상태 세팅 (1: WHITE, 2: YELLOW, 3: RED)
        self.tracker.initialize_tracks(
            initial_positions={
                1: (710.0, 710.0),
                2: (710.0, 355.0),
                3: (2130.0, 710.0),
            },
            cue_ball_id=1,
            ball_colors={1: "WHITE", 2: "YELLOW", 3: "RED"},
        )
        self.scoring_engine.reset_shot(cue_ball_id=1, object_ball_ids=[2, 3])

    def _predict_occluded_ball_position(self, bid: int) -> Point2D:
        """
        트랙 ID(bid)가 큐대·신체·충돌로 가려졌을 때(Occlusion),
        주행 중이고 BilliardNanoGPT 컨텍스트 버퍼가 3프레임 이상 존재하면
        직선 칼만 필터 대신 비선형 쿠션 반사 궤적 복원기(OcclusionTrajectoryRestorer)로 좌표를 복원합니다.
        """
        st = self.tracker.balls[bid]
        restorer = self.trajectory_restorers.get(bid)
        if restorer is not None and len(restorer.buffer) >= 3 and st.speed >= 4.0:
            pred_x, pred_y, _, _ = restorer.step(None)
            return (float(pred_x), float(pred_y))
        return st.predict_kalman_position()

    def _update_cached_half_homography(self) -> None:
        """0.5x(1420x710) 및 0.25x(710x355, 프레임당 2.4ms 초고속 워프)용 호모그래피 행렬을 사전 계산합니다."""
        base_H = self.calibrator.homography_matrix.astype(np.float64)
        scale_half = np.array(
            [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        scale_quarter = np.array(
            [[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        self._cached_half_H = scale_half @ base_H
        self._cached_quarter_H = scale_quarter @ base_H

    def _reset_wall_clock_sync(self) -> None:
        """실시간 1.0배속(Wall-Clock) 프레임 동기화 기준점을 현재 시각으로 리셋합니다."""
        self._wall_clock_origin = time.perf_counter()
        self._frames_advanced_since_origin = 0
        self._last_frame_step_time = 0.0

    def _init_video_capture(self, path_str: str) -> None:
        """
        실제 카메라(camera_index) 또는 카메라 대용 테스트 영상 파일(예: J:\\내 드라이브\\L-billiard.mp4)을 오픈하고,
        당구대 내경 4모서리를 자동 보정한 뒤 타구 준비 구간에 대기시킵니다.
        (외부 가상 드라이브 J:\\ 접근 병목 방지를 위해 최초 1회 로드 후 메모리 캐시 재사용)
        """
        if self.camera_index is None and path_str in LiveStreamReader._INIT_VIDEO_CACHE:
            cached = LiveStreamReader._INIT_VIDEO_CACHE[path_str]
            self.is_real_video_loaded = bool(cached["is_real_video_loaded"])
            self.video_width = int(cached["video_width"])
            self.video_height = int(cached["video_height"])
            self.video_fps = float(cached["video_fps"])
            if self.is_real_video_loaded:
                self.calibrator = TableCalibrator(src_corners=cached["src_points"].copy())
                self._update_cached_half_homography()
                self._table_calibrated = True
                self._latest_raw_frame = cached["raw_frame"]
                self._latest_live_jpeg = cached["jpg_bytes"]
                for _ in range(10):
                    self.replay_buffer.push_frame(cached["jpg_bytes"])
                self.replay_buffer.lock_last_shot(
                    "REGULAR_3C", True, "카메라/테스트 영상 초기 리플레이 준비 완료"
                )
                return

        cap: Optional[cv2.VideoCapture] = None
        if self.camera_index is not None:
            cap = cv2.VideoCapture(self.camera_index)
        else:
            video_file = Path(path_str)
            try:
                if video_file.exists():
                    cap = cv2.VideoCapture(str(video_file))
            except OSError:
                cap = None

        if cap is not None and cap.isOpened():
            self._video_cap = cap
            self.is_real_video_loaded = True
            self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
            self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
            self.video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

            is_virtual_drive = str(path_str).upper().startswith("J:")
            if self.camera_index is None and not is_virtual_drive:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.ACTIVE_SHOT_START_FRAME)

            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, raw_frame = cap.read()

            if ret and raw_frame is not None:
                self.calibrator.auto_detect_table_corners(raw_frame)
                self._update_cached_half_homography()
                self._table_calibrated = True
                self._latest_raw_frame = raw_frame
                ok, buf = cv2.imencode(
                    ".jpg", raw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75]
                )
                if ok:
                    jpg_bytes = buf.tobytes()
                    self._latest_live_jpeg = jpg_bytes
                    for _ in range(10):
                        self.replay_buffer.push_frame(jpg_bytes)
                    if self.camera_index is None:
                        LiveStreamReader._INIT_VIDEO_CACHE[path_str] = {
                            "is_real_video_loaded": True,
                            "video_width": self.video_width,
                            "video_height": self.video_height,
                            "video_fps": self.video_fps,
                            "src_points": self.calibrator.src_points.copy(),
                            "raw_frame": raw_frame,
                            "jpg_bytes": jpg_bytes,
                        }

            self.replay_buffer.lock_last_shot(
                "REGULAR_3C", True, "카메라/테스트 영상 초기 리플레이 준비 완료"
            )
            return

        # 비디오 소스가 없을 경우에만 합성 카메라 프레임으로 폴백
        init_clip = self.cam_generator.generate_shot_replay_clip("REGULAR_3C", "WHITE")
        for f_bytes in init_clip:
            self.replay_buffer.push_frame(f_bytes)
        self.replay_buffer.lock_last_shot("REGULAR_3C", True, "초기 리플레이 준비 완료")

    def _extract_ball_candidates_fast(
        self, raw_frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict[str, float]], List[Dict[str, float]]]:
        """
        0.25x(710x355) 초고속 호모그래피 워프를 사용해 프레임당 ~2.4ms 만에
        2840x1420mm 물리 좌표계 기준 정밀 후보(strict_candidates)와 블러 후보(blur_candidates)를 추출합니다.
        """
        if not self._table_calibrated or self._cached_quarter_H is None:
            self.calibrator.auto_detect_table_corners(raw_frame)
            self._update_cached_half_homography()
            self._table_calibrated = True

        assert self._cached_quarter_H is not None
        warped_quarter = cv2.warpPerspective(
            raw_frame,
            self._cached_quarter_H,
            dsize=(710, 355),
            flags=cv2.INTER_LINEAR,
        )

        hsv = cv2.cvtColor(warped_quarter, cv2.COLOR_BGR2HSV)
        cloth_mask = cv2.inRange(
            hsv, self.detector.hsv_blue_lower, self.detector.hsv_blue_upper
        )
        fg_mask = cv2.bitwise_not(cloth_mask)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        strict_candidates: List[Dict[str, float]] = []
        blur_candidates: List[Dict[str, float]] = []
        cluster_contours: List[np.ndarray] = []
        tilt = self.calibrator.camera_tilt_ratio

        for cnt in contours:
            raw_area = float(cv2.contourArea(cnt))
            area = raw_area * 16.0  # 0.25x 스케일이므로 면적은 x16 복원
            if area < self.detector.min_area or area > self.detector.max_area * 2.1:
                continue
            perim = float(cv2.arcLength(cnt, True))
            if perim <= 0.0:
                continue
            circ = (4.0 * math.pi * raw_area) / (perim * perim)
            (cx, cy), rad = cv2.minEnclosingCircle(cnt)
            cand_item = {
                "x": float(cx) * 4.0,
                "y": (float(cy) + float(rad) * tilt) * 4.0,
                "radius": float(rad) * 4.0,
                "circularity": float(circ),
                "area": area,
            }
            if circ > self.detector.CIRCULARITY_TH and area <= max(
                4600.0, self.detector.max_area * 0.72
            ):
                strict_candidates.append(cand_item)
            elif 4000.0 <= area <= 13500.0 and circ > 0.35:
                cluster_contours.append(cnt)
                blur_candidates.append(cand_item)
            elif circ > self.MOTION_BLUR_CIRC_TH and area <= self.detector.max_area * 1.45:
                blur_candidates.append(cand_item)

        # 공 2~3개가 서로 밀착/충돌하여 하나의 컨투어로 합쳐진 경우 CIE LAB 색도 분리로 개별 공 추출
        if len(strict_candidates) < 3 and cluster_contours:
            lab_q = cv2.cvtColor(warped_quarter, cv2.COLOR_BGR2LAB)
            L_ch, A_ch, B_ch = lab_q[:, :, 0], lab_q[:, :, 1], lab_q[:, :, 2]
            for cnt in cluster_contours:
                cmask = np.zeros(fg_mask.shape, dtype=np.uint8)
                cv2.drawContours(cmask, [cnt], -1, 255, -1)
                sub_masks = [
                    (cmask > 0) & (A_ch >= 142),
                    (cmask > 0) & (A_ch < 142) & (B_ch >= 140),
                    (cmask > 0) & (A_ch < 142) & (B_ch < 140) & (L_ch >= 135),
                ]
                for sm in sub_masks:
                    scnts, _ = cv2.findContours(
                        sm.astype(np.uint8) * 255,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    if not scnts:
                        continue
                    sc = max(scnts, key=cv2.contourArea)
                    s_area = float(cv2.contourArea(sc)) * 16.0
                    if s_area >= 1100.0:
                        (cx, cy), rad = cv2.minEnclosingCircle(sc)
                        cand_pt = (
                            float(cx) * 4.0,
                            (float(cy) + float(rad) * tilt) * 4.0,
                        )
                        if all(
                            math.hypot(cand_pt[0] - ec["x"], cand_pt[1] - ec["y"]) >= 42.0
                            for ec in strict_candidates
                        ):
                            strict_candidates.append(
                                {
                                    "x": cand_pt[0],
                                    "y": cand_pt[1],
                                    "radius": float(rad) * 4.0,
                                    "circularity": 0.78,
                                    "area": s_area,
                                }
                            )

        strict_candidates.sort(
            key=lambda c: (-c["circularity"], abs(c["area"] - self.detector.ideal_area))
        )
        return warped_quarter, strict_candidates[:3], blur_candidates

    def _extract_3_channel_ball_cores(
        self, warped_quarter: np.ndarray
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """
        [Step 1: 큐대·손·모자를 원천 배제하는 3색 채널 독립 원형 코어 검출기]
        - WHITE 채널: L >= 178, S <= 88, B < 148, A < 140 마스크에 5x5 타원 모폴로지 열림(MORPH_OPEN)을 적용하여
          수구에 맞닿은 가느다란 하얀색 큐대(샤프트)와 선수 손/장갑을 100% 절단하고 둥근 흰공 코어만 추출합니다.
        - YELLOW 채널: B >= 150, A < 148, L >= 115, S >= 90 마스크에서 노란공 코어만 독립 추출합니다.
        - RED 채널: A >= 152, S >= 110, V >= 65 마스크에서 빨간공 코어만 독립 추출합니다.
        - 각 색상 채널별로 직전 위치 및 속도 예측 기반 물리 근접 게이트(max_gate)를 적용하여
          샷 도중에도 WHITE/YELLOW/RED ID 뒤바뀜(ID Swap) 및 선수 신체(모자·팔) 오인식을 0%로 차단합니다.
        """
        lab = cv2.cvtColor(warped_quarter, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(warped_quarter, cv2.COLOR_BGR2HSV)
        L_ch, A_ch, B_ch = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        _, S_ch, V_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        tilt = self.calibrator.camera_tilt_ratio

        red_m = (
            (A_ch >= 144) & (B_ch >= 118) & (S_ch >= 55) & (V_ch >= 52)
        ).astype(np.uint8) * 255
        if self._channel_core_state["RED"]["pos"] is not None:
            rp = self._channel_core_state["RED"]["pos"]
            rx_q = int(round(float(rp[0]) * 0.25))
            ry_q = int(round(float(rp[1]) * 0.25))
            x0_r, x1_r = max(0, rx_q - 28), min(710, rx_q + 29)
            y0_r, y1_r = max(0, ry_q - 28), min(355, ry_q + 29)
            if x1_r > x0_r and y1_r > y0_r:
                local_red = (
                    (A_ch[y0_r:y1_r, x0_r:x1_r] >= 140)
                    & (B_ch[y0_r:y1_r, x0_r:x1_r] >= 114)
                    & (S_ch[y0_r:y1_r, x0_r:x1_r] >= 48)
                    & (V_ch[y0_r:y1_r, x0_r:x1_r] >= 40)
                ).astype(np.uint8) * 255
                red_m[y0_r:y1_r, x0_r:x1_r] = np.maximum(
                    red_m[y0_r:y1_r, x0_r:x1_r], local_red
                )
        red_m = cv2.morphologyEx(red_m, cv2.MORPH_CLOSE, self._morph_open_k5)

        yel_m = (
            (B_ch >= 150) & (A_ch < 146) & (L_ch >= 118) & (S_ch >= 82)
        ).astype(np.uint8) * 255
        yel_m = cv2.morphologyEx(yel_m, cv2.MORPH_CLOSE, self._morph_kernel)

        whi_m = (
            (L_ch >= 178)
            & (S_ch <= 86)
            & (B_ch >= 112)
            & (B_ch < 148)
            & (A_ch < 140)
        ).astype(np.uint8) * 255
        whi_m = cv2.morphologyEx(whi_m, cv2.MORPH_OPEN, self._morph_open_k5)

        out: Dict[str, Dict[str, float]] = {}
        for col, mask in (("WHITE", whi_m), ("YELLOW", yel_m), ("RED", red_m)):
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            st = self._channel_core_state[col]
            prev_p: Optional[np.ndarray] = st["pos"]
            prev_v: np.ndarray = st["vel"]
            pred_p: Optional[np.ndarray] = (prev_p + prev_v * 0.95) if prev_p is not None else None

            best_pt: Optional[np.ndarray] = None
            best_rad: float = 31.0
            best_area: float = 2970.0
            best_circ: float = 0.88
            best_score: float = -1e9

            for c in cnts:
                area_q = float(cv2.contourArea(c))
                if area_q < 52.0 or area_q > 360.0:
                    continue
                x_b, y_b, w_b, h_b = cv2.boundingRect(c)
                touches_border = (
                    x_b <= 1
                    or y_b <= 1
                    or (x_b + w_b) >= 708
                    or (y_b + h_b) >= 353
                )
                (cx, cy), rad_q = cv2.minEnclosingCircle(c)
                if rad_q < 3.8 or rad_q > 15.0:
                    continue
                perim = float(cv2.arcLength(c, True))
                circ = (4.0 * math.pi * area_q) / (perim * perim + 1e-5)
                if circ < 0.40:
                    continue
                pt_mm = np.array(
                    [float(cx) * 4.0, (float(cy) + float(rad_q) * tilt) * 4.0],
                    dtype=np.float64,
                )
                border_penalty = 85.0 if touches_border else 0.0
                if pred_p is not None and prev_p is not None:
                    dist_pred = float(np.linalg.norm(pt_mm - pred_p))
                    dist_prev = float(np.linalg.norm(pt_mm - prev_p))
                    dist = min(dist_pred, dist_prev)
                    max_gate = 320.0 + min(900.0, int(st["missing"]) * 220.0)
                    if dist > max_gate:
                        continue
                    score = (
                        circ * 100.0
                        - abs(area_q - 150.0) * 0.25
                        - dist * 0.5
                        - border_penalty
                    )
                else:
                    score = (
                        circ * 100.0
                        - abs(area_q - 150.0) * 0.25
                        - border_penalty
                    )

                if score > best_score:
                    best_score = score
                    best_pt = pt_mm
                    best_rad = float(rad_q) * 4.0
                    best_area = area_q * 16.0
                    best_circ = float(circ)

            if best_pt is not None:
                if prev_p is not None and int(st["missing"]) == 0:
                    st["vel"] = 0.6 * (best_pt - prev_p) + 0.4 * prev_v
                elif prev_p is not None:
                    step_v = (best_pt - prev_p) / float(int(st["missing"]) + 1)
                    st["vel"] = step_v if float(np.linalg.norm(step_v)) <= 120.0 else np.zeros(2, dtype=np.float64)
                st["pos"] = best_pt
                st["missing"] = 0
            elif prev_p is not None:
                # 정지해 있던 공(속도 < 8.0mm/frame)이 선수 몸/팔에 가려진 경우 절대 스스로 움직이거나 순간이동하지 않고 제자리 고정!
                if float(np.linalg.norm(prev_v)) < 8.0:
                    st["pos"] = prev_p
                    st["vel"] = np.zeros(2, dtype=np.float64)
                    st["missing"] = int(st["missing"]) + 1
                else:
                    nxt = prev_p + prev_v * 0.94
                    if nxt[0] < 31.0:
                        nxt[0] = 31.0 + (31.0 - nxt[0])
                        prev_v[0] *= -0.82
                    elif nxt[0] > 2809.0:
                        nxt[0] = 2809.0 - (nxt[0] - 2809.0)
                        prev_v[0] *= -0.82
                    if nxt[1] < 31.0:
                        nxt[1] = 31.0 + (31.0 - nxt[1])
                        prev_v[1] *= -0.82
                    elif nxt[1] > 1389.0:
                        nxt[1] = 1389.0 - (nxt[1] - 1389.0)
                        prev_v[1] *= -0.82
                    st["pos"] = nxt
                    st["vel"] = prev_v * 0.94
                    st["missing"] = int(st["missing"]) + 1
                if int(st["missing"]) > 35 and cnts:
                    for c in cnts:
                        area_q = float(cv2.contourArea(c))
                        if 85.0 <= area_q <= 250.0:
                            x_b, y_b, w_b, h_b = cv2.boundingRect(c)
                            if x_b <= 2 or y_b <= 2 or (x_b + w_b) >= 706 or (y_b + h_b) >= 351:
                                continue
                            perim = float(cv2.arcLength(c, True))
                            circ = (4.0 * math.pi * area_q) / (perim * perim + 1e-5)
                            if circ < 0.72:
                                continue
                            (cx, cy), rad_q = cv2.minEnclosingCircle(c)
                            cand_x = float(cx) * 4.0
                            cand_y = (float(cy) + float(rad_q) * tilt) * 4.0
                            if cand_x < 2450.0 or cand_y < 1100.0:
                                st["pos"] = np.array([cand_x, cand_y], dtype=np.float64)
                                st["vel"] = np.zeros(2, dtype=np.float64)
                                st["missing"] = 0
                                break

            if st["pos"] is None:
                return None

            p_final = st["pos"]
            out[col] = {
                "x": float(p_final[0]),
                "y": float(p_final[1]),
                "radius": best_rad,
                "circularity": best_circ,
                "area": best_area,
            }

        return out if len(out) == 3 else None

    def sync_vision_and_lock_balls_now(
        self, seek_to_active_shot: bool = True, rewind_to_start: bool = False
    ) -> Dict[str, Any]:
        """
        [경기 초기화 또는 화면 리플레시 시 즉시 호출]
        기다림 없이 즉시 현재 카메라/영상 프레임에서 공 3개(흰공·노란공·빨간공)를 검출하여
        1단계 도넛 마스크 색상 잠금(Stationary ID Lock)을 완료하고,
        샷 클락 카운트다운 및 수구 출발 감시(득점 시스템)를 즉시 가동 상태로 전환합니다.
        """
        self._vision_tracks_synced = False
        self.color_locker.set_shot_lock(False)
        self.lab_hungarian_identifier.prev_positions_px.clear()
        self.bulletproof_tracker.is_shot_in_progress = False
        self.spin_detector.reset()
        self._missing_streak_by_id.clear()
        self._last_missing_wait_id = None
        for col_k in ("WHITE", "YELLOW", "RED"):
            self._channel_core_state[col_k] = {
                "pos": None,
                "vel": np.zeros(2, dtype=np.float64),
                "missing": 0,
            }
        for restorer in self.trajectory_restorers.values():
            restorer.reset()
        self.causal_kiss_fsm = CausalKissSafeFSM(fps=max(15.0, self.video_fps))
        self._last_causal_shot_result = None
        self._cue_trajectory.clear()
        self._target_trajectories.clear()
        self.vision_kinematics.reset()
        self._reset_wall_clock_sync()

        if self.is_real_video_loaded and self._video_cap is not None:
            is_virtual_drive = str(self.video_path).upper().startswith("J:")
            if self.camera_index is None and rewind_to_start:
                self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, raw_frame = self._video_cap.read()
            elif is_virtual_drive and self._latest_raw_frame is not None and not rewind_to_start:
                ret = True
                raw_frame = self._latest_raw_frame
            else:
                if self.camera_index is None and seek_to_active_shot and not is_virtual_drive:
                    self._video_cap.set(
                        cv2.CAP_PROP_POS_FRAMES, self.ACTIVE_SHOT_START_FRAME
                    )

                ret, raw_frame = self._video_cap.read()
                if not ret or raw_frame is None:
                    self._video_cap.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        0 if (is_virtual_drive or rewind_to_start) else self.ACTIVE_SHOT_START_FRAME,
                    )
                    ret, raw_frame = self._video_cap.read()

            if ret and raw_frame is not None:
                self._latest_raw_frame = raw_frame
                ok, buf = cv2.imencode(
                    ".jpg", raw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75]
                )
                if ok:
                    self._latest_live_jpeg = buf.tobytes()
                    self.replay_buffer.push_frame(self._latest_live_jpeg)

                warped_q, strict_cands, _ = self._extract_ball_candidates_fast(
                    raw_frame
                )
                channel_cores = self._extract_3_channel_ball_cores(warped_q)
                if channel_cores is not None:
                    fixed_colors = {1: "WHITE", 2: "YELLOW", 3: "RED"}
                    init_pts_ch: Dict[int, Point2D] = {
                        1: (channel_cores["WHITE"]["x"], channel_cores["WHITE"]["y"]),
                        2: (channel_cores["YELLOW"]["x"], channel_cores["YELLOW"]["y"]),
                        3: (channel_cores["RED"]["x"], channel_cores["RED"]["y"]),
                    }
                    self.detected_ball_colors = fixed_colors.copy()
                    active_color = self.match_manager.active_cue_ball_color
                    cue_id = 2 if active_color == "YELLOW" else 1
                    self.tracker.initialize_tracks(
                        initial_positions=init_pts_ch,
                        cue_ball_id=cue_id,
                        ball_colors=fixed_colors,
                    )
                    obj_ids = [bid for bid in (1, 2, 3) if bid != cue_id]
                    self.scoring_engine.reset_shot(
                        cue_ball_id=cue_id, object_ball_ids=obj_ids
                    )
                    self._rest_positions = init_pts_ch.copy()
                    self._vision_tracks_synced = True
                elif len(strict_cands) == 3:
                    init_pts: Dict[int, Point2D] = {
                        i + 1: (float(c["x"]), float(c["y"]))
                        for i, c in enumerate(strict_cands)
                    }
                    init_crops = {
                        bid: self._crop_ball_patch(
                            warped_q,
                            (pt[0] * 0.25, pt[1] * 0.25),
                            patch_radius=9,
                        )
                        for bid, pt in init_pts.items()
                    }
                    colors = self.color_locker.classify_balls(
                        init_crops, use_temporal_smoothing=True
                    )
                    self.detected_ball_colors = colors.copy()
                    self.bulletproof_tracker.phase1_lock_balls_before_shot(
                        [
                            {"center": init_pts[bid], "crop_img": init_crops[bid]}
                            for bid in sorted(init_pts.keys())
                        ]
                    )
                    active_color = self.match_manager.active_cue_ball_color
                    cue_id = 1
                    for bid, col in colors.items():
                        if col == active_color:
                            cue_id = bid
                            break

                    self.tracker.initialize_tracks(
                        initial_positions=init_pts,
                        cue_ball_id=cue_id,
                        ball_colors=colors,
                    )
                    obj_ids = [bid for bid in init_pts.keys() if bid != cue_id]
                    self.scoring_engine.reset_shot(
                        cue_ball_id=cue_id, object_ball_ids=obj_ids
                    )
                    self._rest_positions = init_pts.copy()
                    self._vision_tracks_synced = True

        # 만약 샷 도중 리플레시/초기화되었더라도 즉시 WAIT_SHOT으로 복귀하되, 선수가 수동 일시정지를 눌러둔 상태면 그대로 유지
        self.match_manager.fsm_state = MatchState.WAIT_SHOT
        self.match_manager.is_timer_paused = self.match_manager.is_match_paused
        active_cue_ko = COLOR_KO_LABELS.get(
            self.match_manager.active_cue_ball_color, "흰공"
        )
        p_label = "1번 선수" if self.match_manager.active_player_id == "P1" else "2번 선수"
        self.match_manager.last_shot_result = {
            "is_scored": False,
            "points_awarded": 0,
            "shot_type": "READY",
            "reason_code": "VISION_LOCKED_READY",
            "message": (
                f"[판독 가동] 실시간 득점시스템 작동 중 - 3구 색상 잠금 완료(흰공, 노란공, 빨간공) "
                f"| {p_label}({active_cue_ko}) 타구 대기 및 40초 샷 클락 작동"
            ),
            "cushions_before_second_hit": 0,
            "total_cushions": 0,
        }
        return self.match_manager.get_match_payload()

    def reset_match_and_vision(self, full_factory_reset: bool = True) -> Dict[str, Any]:
        """
        [경기시작 / 경기 초기화] 클릭 또는 /api/match/reset 호출 시:
        점수판 전체(점수, 이닝, 타이머, 선수 진영) 초기화와 동시에 영상을 처음(0프레임)으로 되감고
        비전 추적 엔진 및 3구 색상 잠금을 즉시 재가동합니다.
        """
        self.match_manager.reset_match(full_factory_reset=full_factory_reset)
        return self.sync_vision_and_lock_balls_now(
            seek_to_active_shot=False, rewind_to_start=True
        )

    @staticmethod
    def _crop_ball_patch(
        warped_bgr: np.ndarray, center_pt: Point2D, patch_radius: int = 31
    ) -> np.ndarray:
        """평면 보정 영상에서 특정 공 중심 주변의 정사각형 BGR 패치를 안전하게 추출합니다."""
        h, w = warped_bgr.shape[:2]
        cx, cy = int(round(center_pt[0])), int(round(center_pt[1]))
        x1 = max(0, cx - patch_radius)
        y1 = max(0, cy - patch_radius)
        x2 = min(w, cx + patch_radius)
        y2 = min(h, cy + patch_radius)
        crop = warped_bgr[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
            return np.full((patch_radius * 2, patch_radius * 2, 3), 128, dtype=np.uint8)
        return crop

    def _detect_with_occlusion_fallback(
        self, warped_bgr: np.ndarray
    ) -> List[Point2D]:
        """
        평면 보정 영상에서 당구공 3개를 검출하되,
        주행 중 고속 이동 잔상이나 두 공 충돌/큐대 가림으로 1~2개만 검출된 경우
        칼만 필터 예측 위치(predict_kalman_position)를 결합하여 3개 공의 좌표를 끊김 없이 반환합니다.
        """
        candidates = self.detector.detect(warped_bgr, max_balls=3)
        if len(candidates) == 3:
            return [(float(c["x"]), float(c["y"])) for c in candidates]

        if not self.tracker.balls or len(self.tracker.balls) != 3:
            return []

        detected_pts: List[Point2D] = [
            (float(c["x"]), float(c["y"])) for c in candidates
        ]
        if not detected_pts:
            return [
                self._predict_occluded_ball_position(bid)
                for bid in sorted(self.tracker.balls.keys())
            ]

        remaining_ids = set(self.tracker.balls.keys())
        completed_pts: List[Point2D] = list(detected_pts)

        for pt in detected_pts:
            if not remaining_ids:
                break
            nearest_id = min(
                remaining_ids,
                key=lambda bid: math.hypot(
                    pt[0] - float(self.tracker.balls[bid].position[0]),
                    pt[1] - float(self.tracker.balls[bid].position[1]),
                ),
            )
            remaining_ids.remove(nearest_id)

        for missing_id in sorted(remaining_ids):
            completed_pts.append(self._predict_occluded_ball_position(missing_id))

        return completed_pts[:3]

    async def process_video_frame(
        self, raw_frame: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        실제 카메라(또는 카메라 대용 영상) 단일 BGR 프레임을 입력받아
        0.25x(710x355) 초고속 호모그래피(프레임당 2.4ms)와 5단계 철통 3구 추적 알고리즘을 실시간 구동합니다:
        - [1단계] 정지 상태(WAIT_SHOT): 도넛 마스크 + 다중 프레임 투표로 WHITE, YELLOW, RED 확정 및 잠금
        - [2단계] 타구 순간(SHOT_IN_PROGRESS): 색상 감지기 완전 OFF! 위치만 초고속 추적
        - [3단계] 공-공 충돌 순간: 운동량 보존 법칙(법선 벡터 N)으로 ID 스왑 강제 원복
        - [4단계] 카메라 블러/터널링: 타원 블러 후보 구출 + 칼만 필터 예측 + 3차 스플라인 궤적 복원
        - [5단계] 완전 정지 확인: 이동+스핀 정지 시 득점 판정 및 수구 핸드오버
        """
        if raw_frame is None or raw_frame.size == 0:
            return None

        warped_quarter, strict_candidates, blur_candidates = (
            self._extract_ball_candidates_fast(raw_frame)
        )
        channel_cores = self._extract_3_channel_ball_cores(warped_quarter)
        if channel_cores is not None:
            fixed_colors = {1: "WHITE", 2: "YELLOW", 3: "RED"}
            fixed_pts: List[Point2D] = [
                (channel_cores["WHITE"]["x"], channel_cores["WHITE"]["y"]),
                (channel_cores["YELLOW"]["x"], channel_cores["YELLOW"]["y"]),
                (channel_cores["RED"]["x"], channel_cores["RED"]["y"]),
            ]
            if not self._vision_tracks_synced:
                init_pts_ch = {
                    1: fixed_pts[0],
                    2: fixed_pts[1],
                    3: fixed_pts[2],
                }
                self.detected_ball_colors = fixed_colors.copy()
                active_color = self.match_manager.active_cue_ball_color
                cue_id = 2 if active_color == "YELLOW" else 1
                self.tracker.initialize_tracks(
                    initial_positions=init_pts_ch,
                    cue_ball_id=cue_id,
                    ball_colors=fixed_colors,
                )
                obj_ids = [bid for bid in (1, 2, 3) if bid != cue_id]
                self.scoring_engine.reset_shot(
                    cue_ball_id=cue_id, object_ball_ids=obj_ids
                )
                self._rest_positions = init_pts_ch.copy()
                self._vision_tracks_synced = True
                return None

            # 3색 채널 독립 추적 좌표로 트랙 색상/위치를 1:1 고정 동기화 (ID 스왑 원천 차단)
            for bid, col_name in fixed_colors.items():
                if bid in self.tracker.balls:
                    self.tracker.balls[bid].color = col_name
            if self.match_manager.fsm_state == MatchState.WAIT_SHOT:
                max_step = max(
                    math.hypot(
                        fixed_pts[bid - 1][0] - float(self.tracker.balls[bid].position[0]),
                        fixed_pts[bid - 1][1] - float(self.tracker.balls[bid].position[1]),
                    )
                    for bid in (1, 2, 3)
                    if bid in self.tracker.balls
                )
                if max_step < self.VISION_START_SPEED_TH:
                    for bid in (1, 2, 3):
                        coord = fixed_pts[bid - 1]
                        if bid not in self._rest_positions:
                            self._rest_positions[bid] = coord
                        else:
                            rx, ry = self._rest_positions[bid]
                            self._rest_positions[bid] = (
                                rx * 0.85 + coord[0] * 0.15,
                                ry * 0.85 + coord[1] * 0.15,
                            )

            return await self.process_coordinate_step(
                detected_points=fixed_pts,
                detected_colors=fixed_colors,
                start_velocity_th=self.VISION_START_SPEED_TH,
                require_rest_displacement=True,
            )

        # [1단계: 최초 비전 동기화] 3개의 공이 온전히 검출되었을 때 도넛 마스크로 색상(WHITE, YELLOW, RED)과 초기 ID 잠금
        if not self._vision_tracks_synced:
            if len(strict_candidates) == 3:
                init_pts: Dict[int, Point2D] = {
                    i + 1: (float(c["x"]), float(c["y"]))
                    for i, c in enumerate(strict_candidates)
                }
                init_crops = {
                    bid: self._crop_ball_patch(
                        warped_quarter, (pt[0] * 0.25, pt[1] * 0.25), patch_radius=9
                    )
                    for bid, pt in init_pts.items()
                }
                self.color_locker.set_shot_lock(False)
                colors = self.color_locker.classify_balls(
                    init_crops, use_temporal_smoothing=True
                )
                self.detected_ball_colors = colors.copy()

                # BulletproofBallTracker 1단계 동기화
                self.bulletproof_tracker.phase1_lock_balls_before_shot(
                    [
                        {"center": init_pts[bid], "crop_img": init_crops[bid]}
                        for bid in sorted(init_pts.keys())
                    ]
                )

                active_color = self.match_manager.active_cue_ball_color
                cue_id = 1
                for bid, col in colors.items():
                    if col == active_color:
                        cue_id = bid
                        break

                self.tracker.initialize_tracks(
                    initial_positions=init_pts,
                    cue_ball_id=cue_id,
                    ball_colors=colors,
                )
                obj_ids = [bid for bid in init_pts.keys() if bid != cue_id]
                self.scoring_engine.reset_shot(
                    cue_ball_id=cue_id, object_ball_ids=obj_ids
                )
                self._rest_positions = init_pts.copy()
                self._vision_tracks_synced = True
            return None

        # [2단계 & 4단계] 주행 중 모션 블러 구출 및 칼만 필터 예측 위치 복원
        if len(strict_candidates) == 3:
            self._missing_streak_by_id.clear()
            pts: List[Point2D] = [
                (float(c["x"]), float(c["y"])) for c in strict_candidates
            ]
        elif self.tracker.balls and len(self.tracker.balls) == 3:
            detected_pts = [(float(c["x"]), float(c["y"])) for c in strict_candidates]
            remaining_ids = set(self.tracker.balls.keys())
            unassigned_pts = list(detected_pts)
            pts = list(detected_pts)

            # 정지해 있는 목적구가 자신의 트랙 ID를 뺏기지 않도록 모든 (검출점, 트랙ID) 쌍 중 최단 거리부터 우선 매칭
            while unassigned_pts and remaining_ids:
                best_pair: Optional[Tuple[int, int]] = None
                best_dist = float("inf")
                for pt_idx, pt in enumerate(unassigned_pts):
                    for bid in remaining_ids:
                        d = math.hypot(
                            pt[0] - float(self.tracker.balls[bid].position[0]),
                            pt[1] - float(self.tracker.balls[bid].position[1]),
                        )
                        if d < best_dist:
                            best_dist = d
                            best_pair = (pt_idx, bid)
                if best_pair is None:
                    break
                pt_idx, matched_bid = best_pair
                unassigned_pts.pop(pt_idx)
                remaining_ids.remove(matched_bid)
                self._missing_streak_by_id[matched_bid] = 0

            search_radius = (
                620.0
                if (
                    self.match_manager.fsm_state == MatchState.WAIT_SHOT
                    or len(self._cue_trajectory) <= 12
                )
                else 260.0
            )
            for missing_id in sorted(remaining_ids):
                st = self.tracker.balls[missing_id]
                self._missing_streak_by_id[missing_id] = (
                    self._missing_streak_by_id.get(missing_id, 0) + 1
                )
                if (
                    self.match_manager.fsm_state == MatchState.WAIT_SHOT
                    and st.color in ("WHITE", "YELLOW")
                ):
                    self._last_missing_wait_id = missing_id

                kalman_pt = self._predict_occluded_ball_position(missing_id)
                rescued_pt: Optional[Point2D] = None
                if blur_candidates:
                    near_blurs = [
                        bc
                        for bc in blur_candidates
                        if math.hypot(bc["x"] - kalman_pt[0], bc["y"] - kalman_pt[1])
                        <= search_radius
                        and all(
                            math.hypot(bc["x"] - ep[0], bc["y"] - ep[1]) >= 45.0
                            for ep in pts
                        )
                    ]
                    if near_blurs:
                        best_b = min(
                            near_blurs,
                            key=lambda bc: math.hypot(
                                bc["x"] - kalman_pt[0], bc["y"] - kalman_pt[1]
                            ),
                        )
                        rescued_pt = (float(best_b["x"]), float(best_b["y"]))

                pts.append(rescued_pt if rescued_pt is not None else kalman_pt)
            pts = pts[:3]
        else:
            return None

        # [1단계 vs 2단계 분리]
        # WAIT_SHOT(정지 상태)일 때만 도넛 마스크 색상 확정(1단계)을 수행하고,
        # SHOT_IN_PROGRESS(주행 중)일 때는 색상 감지기를 완전히 OFF(2단계)합니다!
        detected_colors: Optional[Dict[int, str]] = None
        if (
            self.match_manager.fsm_state == MatchState.WAIT_SHOT
            and not self.color_locker.is_color_locked_for_shot
            and len(strict_candidates) == 3
        ):
            matched_preview = self.tracker._match_nearest_positions(pts)
            min_pair_dist = min(
                math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                for i, p1 in enumerate(pts)
                for p2 in pts[i + 1 :]
            )
            if min_pair_dist >= 68.0:
                crops_by_id = {
                    bid: self._crop_ball_patch(
                        warped_quarter, (coord[0] * 0.25, coord[1] * 0.25), patch_radius=9
                    )
                    for bid, coord in matched_preview.items()
                }
                detected_colors = self.color_locker.classify_balls(
                    crops_by_id, use_temporal_smoothing=True
                )
                self.detected_ball_colors = detected_colors.copy()

                max_step = max(
                    math.hypot(
                        coord[0] - float(self.tracker.balls[bid].position[0]),
                        coord[1] - float(self.tracker.balls[bid].position[1]),
                    )
                    for bid, coord in matched_preview.items()
                )
                if max_step < self.VISION_START_SPEED_TH:
                    for bid, coord in matched_preview.items():
                        if bid not in self._rest_positions:
                            self._rest_positions[bid] = coord
                        else:
                            rx, ry = self._rest_positions[bid]
                            self._rest_positions[bid] = (
                                rx * 0.85 + coord[0] * 0.15,
                                ry * 0.85 + coord[1] * 0.15,
                            )

        return await self.process_coordinate_step(
            detected_points=pts,
            detected_colors=detected_colors,
            start_velocity_th=self.VISION_START_SPEED_TH,
            require_rest_displacement=True,
        )

    def _read_single_raw_frame(self) -> Optional[np.ndarray]:
        """카메라 또는 테스트 녹화 영상에서 다음 BGR 프레임 1장을 읽고, 영상 끝 도달 시 타구 준비 프레임으로 루프합니다."""
        if not self.is_real_video_loaded or self._video_cap is None:
            return None
        ret, raw_frame = self._video_cap.read()
        if not ret or raw_frame is None:
            loop_frame = self.ACTIVE_SHOT_START_FRAME if self.camera_index is None else 0
            self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, loop_frame)
            self._vision_tracks_synced = False
            self.color_locker.set_shot_lock(False)
            self._reset_wall_clock_sync()
            ret, raw_frame = self._video_cap.read()
        if ret and raw_frame is not None:
            self._latest_raw_frame = raw_frame
            return raw_frame
        return None

    def read_live_frame_jpeg(self, skip_frames: int = 0) -> bytes:
        """
        실제 카메라 또는 테스트 녹화 영상(L-billiard.mp4)에서 다음 프레임을 원본 해상도(1920x1080) 그대로 읽어
        크기/비율 변형 없이 JPEG 바이트로 반환하며, 링버퍼에 상시 기록합니다.
        """
        if self.is_real_video_loaded and self._video_cap is not None:
            for _ in range(max(0, min(3, skip_frames))):
                self._video_cap.grab()
                self._frames_advanced_since_origin += 1

            raw_frame = self._read_single_raw_frame()
            if raw_frame is not None:
                self._frames_advanced_since_origin += 1
                ok, buf = cv2.imencode(
                    ".jpg", raw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75]
                )
                if ok:
                    jpg_bytes = buf.tobytes()
                    self._latest_live_jpeg = jpg_bytes
                    self.replay_buffer.push_frame(jpg_bytes)
                    return jpg_bytes

            if self._latest_live_jpeg is not None:
                return self._latest_live_jpeg

        # 폴백: 가상 대대 프레임 생성
        fallback_jpg = self.cam_generator.render_live_frame(
            active_cue_color=self.match_manager.active_cue_ball_color,
            fsm_state=self.match_manager.fsm_state.value,
        )
        self.replay_buffer.push_frame(fallback_jpg)
        return fallback_jpg

    async def read_and_process_live_frame(self, skip_frames: int = 0) -> bytes:
        """
        절대 시계(Wall-Clock, time.perf_counter) 기준으로 정확히 실시간 1.0배속(30.0fps)을 유지하면서
        5단계 비전·물리·3쿠션 득점 파이프라인을 구동합니다.
        - 다중 브라우저 탭이나 리플레시 직후 중복 스트림 요청이 동시에 들어올 경우 25ms 내 캐시를 공유하여 부하를 0으로 차단합니다.
        - 만약 순간적인 지연으로 Wall-Clock보다 뒤처진 프레임이 있으면, 중간 프레임은 무거운 JPEG 인코딩(11ms)을 생략하고
          초경량 0.25x 비전 추적(2.4ms)만 수행한 뒤 최종 표시 프레임만 JPEG로 인코딩하여
          '궤적 추적 누락 0%'와 '정확한 1.00배속 실시간 재생'을 동시에 보장합니다.
        """
        now = time.perf_counter()
        fps = max(15.0, min(60.0, self.video_fps))
        frame_interval = 1.0 / fps

        # 1) 다중 연결 중복 호출 방지 (직전 스텝 후 75% 주기 이내 재호출 시 캐시 반환)
        if (
            self._latest_live_jpeg is not None
            and (now - self._last_frame_step_time) < (frame_interval * 0.72)
        ):
            return self._latest_live_jpeg

        self._last_frame_step_time = now

        if self.is_real_video_loaded and self._video_cap is not None:
            if self._frames_advanced_since_origin == 0 or (now - self._wall_clock_origin) > 600.0:
                self._wall_clock_origin = now
                self._frames_advanced_since_origin = 0

            elapsed = now - self._wall_clock_origin
            target_frames = max(1, int(elapsed * fps) + 1)
            behind_frames = target_frames - self._frames_advanced_since_origin
            catchup_count = max(skip_frames, min(3, max(0, behind_frames - 1)))

            # 뒤처진 중간 프레임은 JPEG 인코딩 없이 2.4ms 초경량 비전 추적만 수행하여 1.0배속 따라잡기
            for _ in range(catchup_count):
                mid_frame = self._read_single_raw_frame()
                if mid_frame is not None:
                    self._frames_advanced_since_origin += 1
                    await self.process_video_frame(mid_frame)

            # 현재 화면에 송출할 프레임 읽기 + 비전 처리 + JPEG 인코딩
            raw_frame = self._read_single_raw_frame()
            if raw_frame is not None:
                self._frames_advanced_since_origin += 1
                await self.process_video_frame(raw_frame)
                ok, buf = cv2.imencode(
                    ".jpg", raw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75]
                )
                if ok:
                    jpg_bytes = buf.tobytes()
                    self._latest_live_jpeg = jpg_bytes
                    self.replay_buffer.push_frame(jpg_bytes)
                    return jpg_bytes

            if self._latest_live_jpeg is not None:
                return self._latest_live_jpeg

        return self.read_live_frame_jpeg()

    def _build_vision_color_positions(
        self, ball_states: Dict[int, Any], use_prev: bool = False
    ) -> Tuple[Dict[VisionBallColor, Point2D], Dict[KDColor, TrackedBall]]:
        """
        현재 트랙 상태(ball_states)를 CausalKissSafeFSM(VisionBallColor) 및
        KinematicsDetector(KDColor -> TrackedBall) 입력용 딕셔너리로 변환합니다.
        """
        v_map: Dict[VisionBallColor, Point2D] = {}
        kd_map: Dict[KDColor, TrackedBall] = {}
        fps_scale = max(15.0, self.video_fps)
        color_lookup = {
            "WHITE": (VisionBallColor.WHITE, KDColor.WHITE),
            "YELLOW": (VisionBallColor.YELLOW, KDColor.YELLOW),
            "RED": (VisionBallColor.RED, KDColor.RED),
        }
        fallback_order = [
            (VisionBallColor.WHITE, KDColor.WHITE),
            (VisionBallColor.YELLOW, KDColor.YELLOW),
            (VisionBallColor.RED, KDColor.RED),
        ]

        for bid in sorted(ball_states.keys()):
            st = ball_states[bid]
            if use_prev and bid in self._rest_positions:
                pt = self._rest_positions[bid]
                vx, vy, spd = 0.0, 0.0, 0.0
            elif use_prev:
                pt = (float(st.prev_position[0]), float(st.prev_position[1]))
                vx, vy, spd = 0.0, 0.0, 0.0
            else:
                pt = (float(st.position[0]), float(st.position[1]))
                vx = float(st.velocity[0]) * fps_scale
                vy = float(st.velocity[1]) * fps_scale
                spd = float(st.speed) * fps_scale

            col_pair = color_lookup.get(str(st.color).upper())
            if col_pair is not None and col_pair[0] not in v_map:
                v_map[col_pair[0]] = pt
                kd_map[col_pair[1]] = TrackedBall(
                    raw_track_id=bid,
                    color=col_pair[1],
                    x=pt[0],
                    y=pt[1],
                    vx=vx,
                    vy=vy,
                    speed=spd,
                )

        # 혹시 색상 미할당 트랙이 있을 경우 남은 색상 슬롯에 순서대로 매핑 보장
        for bid in sorted(ball_states.keys()):
            if len(v_map) >= 3:
                break
            st = ball_states[bid]
            pt = (
                (float(st.prev_position[0]), float(st.prev_position[1]))
                if use_prev
                else (float(st.position[0]), float(st.position[1]))
            )
            if pt in v_map.values():
                continue
            for vc, kc in fallback_order:
                if vc not in v_map:
                    v_map[vc] = pt
                    kd_map[kc] = TrackedBall(
                        raw_track_id=bid,
                        color=kc,
                        x=pt[0],
                        y=pt[1],
                        vx=0.0 if use_prev else float(st.velocity[0]) * fps_scale,
                        vy=0.0 if use_prev else float(st.velocity[1]) * fps_scale,
                        speed=0.0 if use_prev else float(st.speed) * fps_scale,
                    )
                    break

        return v_map, kd_map

    async def process_coordinate_step(
        self,
        detected_points: List[Point2D],
        ball_crops_gray: Optional[Dict[int, np.ndarray]] = None,
        has_double_touch_foul: bool = False,
        var_alert_payload: Optional[Dict[str, Any]] = None,
        detected_colors: Optional[Dict[int, str]] = None,
        start_velocity_th: Optional[float] = None,
        require_rest_displacement: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        단일 프레임에서 검출된 공 3개의 물리 좌표를 받아 5단계 파이프라인을 수행합니다:
        1) WAIT_SHOT 상태에서 처음 움직이는 공을 수구(Cue Ball)로 자동 인식하고,
           색상 감지기를 즉시 잠근(OFF) 뒤 샷 클락을 멈추고 3쿠션 득점 알고리즘을 가동합니다.
        2) SHOT_IN_PROGRESS 상태에서 색상 무시 위치 추적 + 물리 충돌 가드(ID 스왑 원복) + 3차 스플라인 판독을 수행합니다.
        3) 모든 공이 멈추면(BALL_STOPPED) 색상 잠금을 풀고 득점 가산 또는 자동 턴 전환 후 샷 클락을 재작동합니다.
        """
        self.frame_counter += 1

        # 1. [2단계 & 3단계] 3채널 고정 코어 좌표인 경우 ID 1:1 직결 업데이트(충돌 시 ID 스왑 0% 원천 차단),
        # 일반 좌표인 경우 최단 거리 위치 추적 및 물리 충돌 가드 업데이트
        if (
            require_rest_displacement
            and detected_colors is not None
            and detected_colors == {1: "WHITE", 2: "YELLOW", 3: "RED"}
            and len(detected_points) == 3
            and len(self.tracker.balls) == 3
        ):
            for bid in (1, 2, 3):
                self.tracker.balls[bid].color = detected_colors[bid]
                self.tracker.balls[bid].update_kinematics(detected_points[bid - 1])
            ball_states = self.tracker.balls
        else:
            ball_states = self.tracker.update(detected_points)

        # 정지 상태(1단계)에서만 색상 갱신 허용, 주행 중(2단계)에는 기존 잠긴 색상 100% 유지
        if (
            detected_colors
            and self.match_manager.fsm_state == MatchState.WAIT_SHOT
            and not self.color_locker.is_color_locked_for_shot
        ):
            for bid, col in detected_colors.items():
                if bid in ball_states:
                    ball_states[bid].color = col

        # 가시 상태인 각 공의 시계열 궤적을 BilliardNanoGPT 복원기(OcclusionTrajectoryRestorer) 버퍼에 적재
        fps_scale = max(15.0, self.video_fps)
        for bid, st in ball_states.items():
            if self._missing_streak_by_id.get(bid, 0) == 0:
                restorer = self.trajectory_restorers.get(bid)
                if restorer is not None:
                    restorer.step(
                        (
                            float(st.position[0]),
                            float(st.position[1]),
                            float(st.velocity[0]) * fps_scale,
                            float(st.velocity[1]) * fps_scale,
                        )
                    )

        # 2. WAIT_SHOT 상태에서 처음 움직이는 공을 수구(Cue Ball)로 인식 및 2단계(색상 감지기 OFF + 타이머 정지) 진입
        if self.match_manager.fsm_state == MatchState.WAIT_SHOT:
            # [불변 규칙 강제] 선공(P1)은 무조건 흰색(WHITE), 후공(P2)은 무조건 노란색(YELLOW)으로 영구 고정
            self.match_manager.players["P1"].cue_ball_color = "WHITE"
            self.match_manager.players["P2"].cue_ball_color = "YELLOW"

            vel_th = (
                start_velocity_th
                if start_velocity_th is not None
                else settings.STOP_VELOCITY_TH
            )
            active_expected_color = self.match_manager.active_cue_ball_color
            expected_cue_id: Optional[int] = None
            for bid, st in ball_states.items():
                if st.color == active_expected_color:
                    expected_cue_id = bid
                    break

            first_moving_id: Optional[int] = None
            occluded_first_hit_id: Optional[int] = None

            # [규칙 1: 세 공이 멈춘 상태(IDLE)에서 '가장 먼저 출발한 공(First-Mover Onset)' 추적]
            # 아직 공끼리 충돌하기 전(min_pair_dist >= 66.0mm) 상태에서 적구(RED)를 제외한 WHITE/YELLOW 중
            # 가장 먼저 정지 위치로부터 출발한 공을 추적합니다.
            pts_list = [
                (float(st.position[0]), float(st.position[1]))
                for st in ball_states.values()
            ]
            min_pair_dist_now = (
                min(
                    math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                    for i, p1 in enumerate(pts_list)
                    for p2 in pts_list[i + 1 :]
                )
                if len(pts_list) >= 2
                else 999.0
            )
            is_opening_unlocked = (
                not self.causal_kiss_fsm.turn_locked_by_previous_shot
                and self.match_manager.shot_seq_id <= 1
                and self.match_manager.current_inning == 1
                and self.match_manager.players["P1"].score == 0
                and self.match_manager.players["P2"].score == 0
            )

            # 1순위: 현재 턴 선수의 지정 수구(active_expected_color)가 출발했는지 확인
            if expected_cue_id is not None and expected_cue_id in ball_states:
                exp_st = ball_states[expected_cue_id]
                exp_disp = 999.0
                if require_rest_displacement and expected_cue_id in self._rest_positions:
                    rx, ry = self._rest_positions[expected_cue_id]
                    exp_disp = math.hypot(
                        float(exp_st.position[0]) - rx, float(exp_st.position[1]) - ry
                    )
                if exp_st.speed >= vel_th and exp_disp >= self.VISION_START_DISP_MM:
                    first_moving_id = expected_cue_id

            # 1-B순위: 아직 공끼리 충돌하기 전(>= 66mm) 상태에서 WHITE/YELLOW 중 실제로 정지 위치에서
            # 먼저 출발한 공(First-Mover)을 수구로 결정! (가만히 있는 공을 수구로 오인하는 현상 차단)
            if first_moving_id is None and min_pair_dist_now >= 66.0:
                best_onset_id: Optional[int] = None
                best_onset_speed: float = 0.0
                for bid, st in ball_states.items():
                    if str(st.color).upper() not in ("WHITE", "YELLOW"):
                        continue
                    cand_disp = 999.0
                    if require_rest_displacement and bid in self._rest_positions:
                        rx, ry = self._rest_positions[bid]
                        cand_disp = math.hypot(
                            float(st.position[0]) - rx, float(st.position[1]) - ry
                        )
                    if st.speed >= vel_th and cand_disp >= max(12.0, self.VISION_START_DISP_MM):
                        if st.speed > best_onset_speed:
                            best_onset_speed = float(st.speed)
                            best_onset_id = bid
                if best_onset_id is not None:
                    first_moving_id = best_onset_id

            # 2순위: 만약 스트로크 순간 큐대/손 가림으로 인해 지정 수구는 안 보였지만
            # 다른 공(제1적구 또는 상대방 수구)이 강하게(>= 35.0mm) 튀어나간 경우:
            # 절대 수구(턴)를 상대방으로 넘기지 않고, 지정 수구(expected_cue_id)를 수구로 잠근 채 튀어나간 공을 제1적구로 등록!
            if first_moving_id is None and expected_cue_id is not None:
                for bid, st in ball_states.items():
                    if bid == expected_cue_id or st.speed < max(vel_th * 1.5, 11.5):
                        continue
                    disp_other = 999.0
                    if bid in self._rest_positions:
                        rx, ry = self._rest_positions[bid]
                        disp_other = math.hypot(
                            float(st.position[0]) - rx, float(st.position[1]) - ry
                        )
                    elif require_rest_displacement:
                        disp_other = math.hypot(
                            float(st.position[0]) - float(st.prev_position[0]),
                            float(st.position[1]) - float(st.prev_position[1]),
                        )
                    if disp_other >= 35.0:
                        first_moving_id = expected_cue_id
                        occluded_first_hit_id = bid
                        break

            if first_moving_id is not None:
                # 샷 시작 직전 원래 공격 선수와 이닝 상태를 백업 (유령 샷 취소 시 100% 원복 보장)
                self.match_manager._pre_shot_player_id = self.match_manager.active_player_id
                self.match_manager._pre_shot_inning = self.match_manager.current_inning
                self.match_manager._pre_shot_p2_had_slot = (
                    self.match_manager.current_inning
                    in self.match_manager.players["P2"].inning_scores
                )

                # 현재 턴의 지정 수구를 이번 샷의 수구(Cue Ball)로 영구 고정! (상대방 공 노이즈 턴 하이재킹 원천 차단)
                self.tracker.cue_ball_id = first_moving_id
                cue_id = first_moving_id
                cue_state = ball_states[cue_id]

                # [2단계 핵심] 수구가 출발했으므로 샷이 끝날 때까지 색상 감지기를 완전히 OFF!
                self.color_locker.set_shot_lock(True)
                self.bulletproof_tracker.is_shot_in_progress = True

                # [절대 원칙]
                # 1) 흰공(WHITE) = 1번 선수(선공 P1), 노란공(YELLOW) = 2번 선수(후공 P2) 100% 고정!
                # 2) 후공(P2) 샷 이후 선공(P1)이 치면 반드시 다음 이닝(current_inning + 1)으로 동기화!
                self.match_manager.players["P1"].cue_ball_color = "WHITE"
                self.match_manager.players["P2"].cue_ball_color = "YELLOW"
                reconcile_res = self.match_manager.reconcile_cue_ball_stroke(
                    stroked_cue_color=str(cue_state.color).upper(),
                    compensate_missed_score=(not require_rest_displacement),
                    force_sync_player=True,
                )
                compensated_prev_shot = bool(reconcile_res.get("compensated", False))
                self.match_manager.sync_active_player_to_current_inning(
                    self.match_manager.active_player_id
                )

                # 수구가 움직였으므로 샷 클락(카운트) 작동을 즉시 멈추고 득점 알고리즘 가동
                self.match_manager.on_shot_started()
                self.spin_detector.reset()
                self.replay_buffer.clear_active()
                obj_ids = [bid for bid in ball_states.keys() if bid != cue_id]
                self.scoring_engine.reset_shot(
                    cue_ball_id=cue_id, object_ball_ids=obj_ids
                )
                if occluded_first_hit_id is not None:
                    self.scoring_engine.record_ball_hit(
                        occluded_first_hit_id,
                        self.frame_counter,
                        (float(cue_state.position[0]), float(cue_state.position[1])),
                    )
                # 3체 인과 키스 분리 엔진(CausalKissSafeFSM) 및 100% Autoresearch 물리 감지기 초기화
                shooter_vcol = (
                    VisionBallColor.YELLOW
                    if str(cue_state.color).upper() == "YELLOW"
                    else VisionBallColor.WHITE
                )
                self.causal_kiss_fsm._reset_shot()
                self.causal_kiss_fsm.assigned_cue_ball = shooter_vcol
                self.causal_kiss_fsm.active_cue_ball = shooter_vcol
                self.causal_kiss_fsm.state = CausalShotState.IN_PLAY
                self.causal_kiss_fsm.shot_seq_id = self.match_manager.shot_seq_id
                if occluded_first_hit_id is not None and occluded_first_hit_id in ball_states:
                    occ_col_str = str(ball_states[occluded_first_hit_id].color).upper()
                    if occ_col_str in VisionBallColor.__members__:
                        self.causal_kiss_fsm.first_target_ball = VisionBallColor[occ_col_str]
                        self.causal_kiss_fsm.first_target_hit_time = self.frame_counter / fps_scale
                self._last_causal_shot_result = None
                self.vision_kinematics.reset()
                init_vpos, init_kdpos = self._build_vision_color_positions(
                    ball_states, use_prev=True
                )
                if len(init_vpos) == 3:
                    t_init = max(0, self.frame_counter - 1) / fps_scale
                    self.causal_kiss_fsm.stroke_start_time = t_init
                    id_to_vcol = {
                        1: VisionBallColor.WHITE,
                        2: VisionBallColor.YELLOW,
                        3: VisionBallColor.RED,
                    }
                    rest_vpos: Dict[VisionBallColor, np.ndarray] = {}
                    for bid, vcol in id_to_vcol.items():
                        if self._rest_positions and bid in self._rest_positions:
                            rp = self._rest_positions[bid]
                            rest_vpos[vcol] = np.array([float(rp[0]), float(rp[1])], dtype=np.float64)
                        elif vcol in init_vpos:
                            pt = init_vpos[vcol]
                            rest_vpos[vcol] = np.array([float(pt[0]), float(pt[1])], dtype=np.float64)
                    if len(rest_vpos) == 3:
                        self.causal_kiss_fsm.rest_pos = {
                            vc: arr.copy() for vc, arr in rest_vpos.items()
                        }
                        self.causal_kiss_fsm.initial_positions = {
                            vc: arr.copy() for vc, arr in rest_vpos.items()
                        }
                        self.causal_kiss_fsm.target_rest_anchor = {
                            vc: arr.copy() for vc, arr in rest_vpos.items()
                        }
                        cue_origin = rest_vpos[shooter_vcol].copy()
                        self.causal_kiss_fsm.stroke_origin_pos = cue_origin
                        self.causal_kiss_fsm.update(
                            t=max(0.0, t_init - (1.0 / fps_scale)),
                            coords_mm=rest_vpos,
                        )
                    self.causal_kiss_fsm.update(
                        t=t_init,
                        coords_mm=init_vpos,
                    )
                    self.vision_kinematics.step(
                        balls=init_kdpos,
                        assigned_cue_ball=(
                            KDColor.YELLOW
                            if str(cue_state.color).upper() == "YELLOW"
                            else KDColor.WHITE
                        ),
                        timestamp=t_init,
                        current_cushion_count=0,
                    )
                # 샷 출발 전 정지 기준점(_rest_positions)이 존재하면 득점 엔진 초기 위치 및 시계열 궤적 시작점으로 보존
                if self._rest_positions and len(self._rest_positions) == 3:
                    self.scoring_engine._initial_ball_positions = {
                        bid: self._rest_positions[bid]
                        for bid in ball_states.keys()
                        if bid in self._rest_positions
                    }
                    self.scoring_engine._ball_trajectories = {
                        bid: [self._rest_positions[bid]]
                        for bid in ball_states.keys()
                        if bid in self._rest_positions
                    }
                self._cue_trajectory = [
                    (
                        float(cue_state.prev_position[0]),
                        float(cue_state.prev_position[1]),
                    )
                ]
                self._target_trajectories = {
                    bid: [
                        (
                            float(ball_states[bid].position[0]),
                            float(ball_states[bid].position[1]),
                        )
                    ]
                    for bid in obj_ids
                }
                active_cue_color = self.match_manager.active_cue_ball_color
                cue_ko = COLOR_KO_LABELS.get(active_cue_color, active_cue_color)
                start_msg = f"수구({cue_ko}) 출발 감지 - 타이머 일시정지 및 5단계 초고속 궤적 추적 중"
                start_reason = (
                    "RETROACTIVE_SCORE_COMPENSATED"
                    if compensated_prev_shot
                    else "CUE_MOVING"
                )

                self.match_manager.last_shot_result = {
                    "shot_id": self.match_manager.shot_seq_id,
                    "is_scored": False,
                    "points_awarded": 0,
                    "shot_type": "IN_PROGRESS",
                    "reason_code": start_reason,
                    "message": start_msg,
                    "cushions_before_first_hit": 0,
                    "cushions_before_second_hit": 0,
                    "total_cushions": 0,
                }
                await self.ws_manager.broadcast_event(
                    "SCORE_UPDATED", self.match_manager.get_match_payload()
                )

        # 3. 샷 진행 중(SHOT_IN_PROGRESS)일 때 3쿠션 이벤트 기록 및 5단계 완전 정지 감지
        if self.match_manager.fsm_state == MatchState.SHOT_IN_PROGRESS:
            cue_id = self.tracker.cue_ball_id or 1
            cue_state = ball_states[cue_id]
            positions_map: Dict[int, Point2D] = {
                bid: (float(st.position[0]), float(st.position[1]))
                for bid, st in ball_states.items()
            }
            cue_vel = (float(cue_state.velocity[0]), float(cue_state.velocity[1]))

            prev_cushions = max(
                int(self.scoring_engine.total_cushion_count),
                int(
                    self._last_causal_shot_result.total_cushions
                    if (
                        self._last_causal_shot_result is not None
                        and self._last_causal_shot_result.is_turn_end
                    )
                    else self.causal_kiss_fsm.cushion_count
                ),
            )
            prev_causal_cushions = int(self.causal_kiss_fsm.cushion_count)
            prev_causal_c_before = int(self.causal_kiss_fsm.cushions_before_first_hit)
            prev_events_count = len(self.scoring_engine.events)
            prev_causal_events_count = len(self.causal_kiss_fsm.event_log)

            # [득점 후 잔여 구름 동결(Freeze)]
            # 제2적구 충돌로 즉시 득점이 이미 발송된 이후에는 사후 쿠션/충돌 평가를 완전히 중단하고 공이 멈추기만 기다립니다!
            if not self.match_manager.score_dispatched_for_current_shot:
                self.scoring_engine.update_frame(
                    frame_idx=self.frame_counter,
                    ball_positions=positions_map,
                    cue_velocity=cue_vel,
                )

            # 3체 인과 키스 분리 엔진 및 Autoresearch 100% 물리 감지기 실시간 동기화
            cur_vpos, cur_kdpos = self._build_vision_color_positions(
                ball_states, use_prev=False
            )
            maybe_causal = None
            if len(cur_vpos) == 3:
                t_curr = self.frame_counter / fps_scale
                if (
                    self._last_causal_shot_result is None
                    or not self._last_causal_shot_result.is_turn_end
                ):
                    maybe_causal = self.causal_kiss_fsm.update(
                        t=t_curr,
                        coords_mm=cur_vpos,
                    )
                    if maybe_causal is not None:
                        self._last_causal_shot_result = maybe_causal
                if not self.match_manager.score_dispatched_for_current_shot:
                    fsm_c_now = (
                        int(self._last_causal_shot_result.total_cushions)
                        if (
                            self._last_causal_shot_result is not None
                            and self._last_causal_shot_result.is_turn_end
                        )
                        else int(self.causal_kiss_fsm.cushion_count)
                    )
                    se_c_now = int(self.scoring_engine.total_cushion_count)
                    live_total_cushions = max(fsm_c_now, se_c_now)
                    if (
                        self.causal_kiss_fsm.state == CausalShotState.IN_PLAY
                        and self.causal_kiss_fsm.second_target_ball is None
                    ):
                        self.causal_kiss_fsm.cushion_count = max(
                            int(self.causal_kiss_fsm.cushion_count),
                            live_total_cushions,
                        )
                    self.scoring_engine.total_cushion_count = max(
                        int(self.scoring_engine.total_cushion_count),
                        live_total_cushions,
                    )
                    self.vision_kinematics.step(
                        balls=cur_kdpos,
                        assigned_cue_ball=(
                            KDColor.YELLOW
                            if str(cue_state.color).upper() == "YELLOW"
                            else KDColor.WHITE
                        ),
                        timestamp=t_curr,
                        current_cushion_count=live_total_cushions,
                    )
            else:
                live_total_cushions = max(
                    int(self.scoring_engine.total_cushion_count),
                    int(self.causal_kiss_fsm.cushion_count),
                )

            self._cue_trajectory.append(positions_map[cue_id])
            for bid, tr_list in self._target_trajectories.items():
                if bid in positions_map:
                    tr_list.append(positions_map[bid])

            # [Step 4: 단일 판정 엔진 일원화 (Single Source of Truth)]
            # 실제 영상/카메라 스트림(require_rest_displacement=True)에서는 구형 엔진들의 OR 투표를 완전히 배제하고
            # 3체 인과·폴리라인·정지 앵커 검증기(CausalKissSafeFSM) 단독 판정만 신뢰합니다!
            causal_instant_scored = bool(
                (
                    maybe_causal is not None
                    and maybe_causal.scored
                    and not maybe_causal.already_dispatched
                )
                or self.causal_kiss_fsm.score_dispatched
                or (
                    self._last_causal_shot_result is not None
                    and self._last_causal_shot_result.scored
                )
            )
            if require_rest_displacement:
                engine_instant_scored = False
                kinematics_instant_scored = False
            else:
                engine_instant_scored = bool(
                    self.scoring_engine.fsm_state.value == "VALID_SCORE_PENDING"
                    and not self.causal_kiss_fsm.double_hit_foul
                    and (
                        not self.causal_kiss_fsm.object_ball_kiss_occurred
                        or self.causal_kiss_fsm.score_dispatched
                        or self.scoring_engine.second_object_ball_id is not None
                    )
                )
                kinematics_instant_scored = bool(
                    getattr(self.vision_kinematics, "second_ball_hit", False)
                    and int(getattr(self.vision_kinematics, "cushions_before_second_ball", 0)) >= 3
                    and not self.causal_kiss_fsm.double_hit_foul
                    and not self.causal_kiss_fsm.object_ball_kiss_occurred
                )
            if (
                not self.match_manager.score_dispatched_for_current_shot
                and not has_double_touch_foul
                and (causal_instant_scored or engine_instant_scored or kinematics_instant_scored)
            ):
                if require_rest_displacement:
                    cush_before_1st = (
                        int(maybe_causal.cushions_before_first_hit)
                        if maybe_causal is not None
                        else int(self.causal_kiss_fsm.cushions_before_first_hit)
                    )
                    is_bank = bool(
                        (maybe_causal.is_bank_shot if maybe_causal is not None else self.causal_kiss_fsm.is_bank_shot)
                        or cush_before_1st >= 1
                    )
                    cush_at_2nd = max(
                        3,
                        int(maybe_causal.cushions_at_second_hit)
                        if maybe_causal is not None
                        else int(self.causal_kiss_fsm.cushions_at_second_hit),
                    )
                else:
                    if self.causal_kiss_fsm.first_target_ball is not None or maybe_causal is not None:
                        cush_before_1st = (
                            int(maybe_causal.cushions_before_first_hit)
                            if maybe_causal is not None
                            else int(self.causal_kiss_fsm.cushions_before_first_hit)
                        )
                    else:
                        cush_before_1st = int(self.scoring_engine.cushion_count_before_first_hit)
                    is_bank = bool(cush_before_1st >= 1)
                    cush_at_2nd = max(
                        3,
                        int(self.scoring_engine.cushion_count_before_second_hit),
                        int(maybe_causal.cushions_at_second_hit) if maybe_causal is not None else 3,
                    )
                tot_cush = cush_at_2nd
                if is_bank:
                    if cush_before_1st >= cush_at_2nd:
                        seq_msg = f"{cush_at_2nd}쿠션 ⇒ 1적구 ⇒ 2적구 ⇒ 뱅크샷성공"
                    else:
                        seq_msg = f"{cush_before_1st}쿠션 ⇒ 1적구 ⇒ {cush_at_2nd}쿠션 ⇒ 2적구 ⇒ 뱅크샷성공"
                else:
                    seq_msg = f"1적구 ⇒ {cush_at_2nd}쿠션 ⇒ 2적구 ⇒ 득점성공"
                immediate_eval = {
                    "is_scored": True,
                    "shot_type": "BANK_SHOT" if is_bank else "REGULAR_3C",
                    "reason_code": "IMMEDIATE_SECOND_BALL_HIT",
                    "message": seq_msg,
                    "cushions_before_first_hit": cush_before_1st,
                    "cushions_before_second_hit": cush_at_2nd,
                    "total_cushions": tot_cush,
                    "first_ball_hit": True,
                    "second_ball_hit": True,
                }
                imm_transition = self.match_manager.on_immediate_second_ball_hit(
                    immediate_eval
                )
                await self.ws_manager.broadcast_event(
                    "SCORE_UPDATED", imm_transition["payload"]
                )
            # 주행 중 쿠션 횟수나 목적구 타격이 새로 감지되면 점수판 UI에 실시간 반영
            elif (
                self.scoring_engine.total_cushion_count != prev_cushions
                or int(self.causal_kiss_fsm.cushion_count) != prev_causal_cushions
                or int(self.causal_kiss_fsm.cushions_before_first_hit) != prev_causal_c_before
                or len(self.scoring_engine.events) != prev_events_count
                or len(self.causal_kiss_fsm.event_log) != prev_causal_events_count
            ):
                b1_done = (
                    bool(self.causal_kiss_fsm.first_target_ball is not None)
                    if require_rest_displacement
                    else bool(
                        self.causal_kiss_fsm.first_target_ball is not None
                        or self.scoring_engine.first_object_ball_id is not None
                    )
                )
                b2_done = (
                    bool(self.causal_kiss_fsm.second_target_ball is not None)
                    if require_rest_displacement
                    else bool(
                        self.causal_kiss_fsm.second_target_ball is not None
                        or self.scoring_engine.second_object_ball_id is not None
                    )
                )
                if self.causal_kiss_fsm.first_target_ball is not None:
                    c_before_now = int(self.causal_kiss_fsm.cushions_before_first_hit)
                elif require_rest_displacement:
                    c_before_now = int(self.causal_kiss_fsm.cushions_before_first_hit)
                else:
                    c_before_now = max(
                        int(self.causal_kiss_fsm.cushions_before_first_hit),
                        int(self.scoring_engine.cushion_count_before_first_hit),
                    )
                tot_c_now = (
                    int(self.causal_kiss_fsm.cushion_count)
                    if require_rest_displacement
                    else max(
                        int(self.scoring_engine.total_cushion_count),
                        int(self.causal_kiss_fsm.cushion_count),
                    )
                )
                fsm_val = self.scoring_engine.fsm_state.value
                if fsm_val == "VALID_SCORE_PENDING" or self.causal_kiss_fsm.score_dispatched:
                    fsm_ko = "즉시득점완료"
                elif fsm_val == "NO_SCORE_LOCKED":
                    fsm_ko = "조기키스잠금"
                else:
                    fsm_ko = "진행중"

                if c_before_now >= 1:
                    if not b1_done:
                        live_seq = f"{tot_c_now}쿠션 ⇒ 1적구(대기) ⇒ 2적구(대기)"
                    elif tot_c_now > c_before_now:
                        live_seq = f"{c_before_now}쿠션 ⇒ 1적구 ⇒ {tot_c_now}쿠션 ⇒ 2적구({'접촉' if b2_done else '대기'})"
                    else:
                        live_seq = f"{c_before_now}쿠션 ⇒ 1적구 ⇒ 2적구({'접촉' if b2_done else '대기'})"
                else:
                    if not b1_done:
                        live_seq = f"1적구(대기) ⇒ {tot_c_now}쿠션 ⇒ 2적구(대기)"
                    else:
                        live_seq = f"1적구 ⇒ {tot_c_now}쿠션 ⇒ 2적구({'접촉' if b2_done else '대기'})"

                if not self.match_manager.score_dispatched_for_current_shot:
                    self.match_manager.last_shot_result = {
                        "is_scored": False,
                        "points_awarded": 0,
                        "shot_type": "IN_PROGRESS",
                        "reason_code": "TRACKING",
                        "fsm_state": fsm_val,
                        "message": f"실시간 추적 중[{fsm_ko}]: {live_seq}",
                        "cushions_before_first_hit": c_before_now,
                        "cushions_before_second_hit": max(
                            int(self.scoring_engine.cushion_count_before_second_hit),
                            int(self.causal_kiss_fsm.cushions_at_second_hit),
                        ),
                        "total_cushions": tot_c_now,
                        "first_ball_hit": b1_done,
                        "second_ball_hit": b2_done,
                    }
                await self.ws_manager.broadcast_event(
                    "SCORE_UPDATED", self.match_manager.get_match_payload()
                )

            # 외부에서 명시적 VAR 알림이 주입되었거나 미세 접촉 판독이 필요한 경우 브로드캐스트
            if var_alert_payload is not None and var_alert_payload.get("var_triggered"):
                await self.ws_manager.broadcast_event("VAR_ALERT", var_alert_payload)

            # [5단계: 완전 정지 확인] 이동 속도 + 제자리 팽이 회전(스핀) 정지 감지 업데이트
            spin_status = self.spin_detector.update(
                ball_states=ball_states,
                ball_crops_gray=ball_crops_gray,
            )

            # 15프레임(0.5초) 연속 정지 확정(BALL_STOPPED) 시 룰 평가 및 스코어/턴 갱신 + 신분증 반납/인계
            if bool(spin_status["is_fully_stopped"]):
                self.color_locker.set_shot_lock(False)
                self.bulletproof_tracker.is_shot_in_progress = False
                self._missing_streak_by_id.clear()
                self._last_missing_wait_id = None

                # [불변 규칙 강제] 득점/공타 정산 전후 선공(P1)=WHITE, 후공(P2)=YELLOW 영구 고정
                self.match_manager.players["P1"].cue_ball_color = "WHITE"
                self.match_manager.players["P2"].cue_ball_color = "YELLOW"

                shot_eval = self.scoring_engine.evaluate_shot(
                    has_double_touch_foul=has_double_touch_foul
                )

                # [3체 인과 키스 분리(CausalKissSafeFSM) & Autoresearch 100% 초미세 판독 결합]
                if self.causal_kiss_fsm.state != CausalShotState.IDLE:
                    causal_res = self.causal_kiss_fsm._finalize_shot(
                        self.frame_counter / fps_scale
                    )
                else:
                    causal_res = self._last_causal_shot_result or self.causal_kiss_fsm._finalize_shot(
                        self.frame_counter / fps_scale
                    )
                self._last_causal_shot_result = None
                shot_eval["already_dispatched"] = bool(
                    causal_res.already_dispatched
                    or self.match_manager.score_dispatched_for_current_shot
                )
                ob_kiss = bool(
                    causal_res.object_ball_kiss_occurred
                    or self.vision_kinematics.object_ball_kiss_occurred
                )
                shot_eval["object_ball_kiss_occurred"] = ob_kiss
                shot_eval["double_hit_foul"] = False
                shot_eval["centroid_shift_corrected_count"] = int(
                    causal_res.centroid_shift_corrected_count
                    + self.vision_kinematics.centroid_shift_corrected_count
                )
                shot_eval["causal_event_log"] = list(causal_res.event_log)

                # [Step 4: 실제 영상(require_rest_displacement=True)에서는 CausalKissSafeFSM 단독 판정 적용]
                if require_rest_displacement:
                    is_valid_causal = bool(
                        causal_res.scored or shot_eval.get("already_dispatched", False)
                    )
                    shot_eval["is_scored"] = is_valid_causal
                    final_tot_cush = max(
                        int(causal_res.total_cushions),
                        int(self.scoring_engine.total_cushion_count),
                    )
                    shot_eval["cushions_before_first_hit"] = int(causal_res.cushions_before_first_hit)
                    shot_eval["cushions_before_second_hit"] = int(causal_res.cushions_at_second_hit)
                    shot_eval["total_cushions"] = final_tot_cush
                    shot_eval["first_ball_hit"] = bool(
                        self.causal_kiss_fsm.first_target_ball is not None
                        or self.scoring_engine.first_object_ball_id is not None
                        or is_valid_causal
                    )
                    shot_eval["second_ball_hit"] = bool(
                        self.causal_kiss_fsm.second_target_ball is not None
                        or self.scoring_engine.second_object_ball_id is not None
                        or is_valid_causal
                    )
                    if is_valid_causal:
                        is_bank_final = bool(
                            causal_res.is_bank_shot or causal_res.cushions_before_first_hit >= 1
                        )
                        shot_eval["shot_type"] = "BANK_SHOT" if is_bank_final else "REGULAR_3C"
                        shot_eval["reason_code"] = (
                            "VALID_BANK_SHOT" if is_bank_final else "VALID_REGULAR_3C"
                        )
                        c_b1_final = int(causal_res.cushions_before_first_hit) if is_bank_final else 0
                        c_b2 = max(3, int(causal_res.cushions_at_second_hit))
                        shot_eval["cushions_before_second_hit"] = c_b2
                        shot_eval["total_cushions"] = c_b2
                        if is_bank_final:
                            if c_b1_final >= c_b2:
                                shot_eval["message"] = f"{c_b2}쿠션 ⇒ 1적구 ⇒ 2적구 ⇒ 뱅크샷성공"
                            else:
                                shot_eval["message"] = f"{c_b1_final}쿠션 ⇒ 1적구 ⇒ {c_b2}쿠션 ⇒ 2적구 ⇒ 뱅크샷성공"
                        else:
                            shot_eval["message"] = f"1적구 ⇒ {c_b2}쿠션 ⇒ 2적구 ⇒ 득점성공"
                    else:
                        shot_eval["shot_type"] = "MISS"
                        shot_eval["reason_code"] = str(causal_res.fail_reason or "SECOND_BALL_MISSED")
                        shot_eval["message"] = (
                            f"공타: 제2목적구 불발 (누적 {final_tot_cush}쿠션)"
                            if "SECOND_BALL" in str(causal_res.fail_reason)
                            else f"공타: {causal_res.fail_reason or '목적구 불발'} (누적 {final_tot_cush}쿠션)"
                        )
                elif bool(shot_eval.get("already_dispatched", False)):
                    shot_eval["is_scored"] = True
                    is_bank_final = bool(
                        int(shot_eval.get("cushions_before_first_hit", 0)) >= 1
                        or self.scoring_engine.cushion_count_before_first_hit >= 1
                        or causal_res.is_bank_shot
                        or str(shot_eval.get("shot_type", "")) == "BANK_SHOT"
                    )
                    shot_eval["shot_type"] = (
                        "BANK_SHOT" if is_bank_final else "REGULAR_3C"
                    )
                    shot_eval["reason_code"] = (
                        "VALID_BANK_SHOT"
                        if is_bank_final
                        else "VALID_REGULAR_3C"
                    )
                    c_b2 = max(3, int(shot_eval.get("cushions_before_second_hit", 3)))
                    shot_eval["cushions_before_second_hit"] = c_b2
                    shot_eval["total_cushions"] = c_b2
                    if "득점" not in str(shot_eval.get("message", "")):
                        shot_eval["message"] = (
                            f"득점 인정 (뱅크 샷): 총 {c_b2}쿠션 후 제2적구 충돌 성공!"
                            if is_bank_final
                            else f"득점 인정 (일반 3쿠션): 총 {c_b2}쿠션 후 제2적구 충돌 성공!"
                        )

                # 0.3mm 초미세 스침 / 1쿠션 넣어치기 뱅크샷 / 시속 35km 고속 코너 반사 등 정밀 구출 및 동기화
                if (
                    bool(shot_eval.get("is_scored", False))
                    and causal_res.is_bank_shot
                    and str(shot_eval.get("shot_type", "")) != "BANK_SHOT"
                ):
                    shot_eval["shot_type"] = "BANK_SHOT"
                    shot_eval["reason_code"] = "VALID_BANK_SHOT"
                    shot_eval["cushions_before_first_hit"] = max(
                        1,
                        int(shot_eval.get("cushions_before_first_hit", 0)),
                        int(causal_res.cushions_before_first_hit),
                    )
                elif (
                    not bool(shot_eval.get("is_scored", False))
                    and not has_double_touch_foul
                    and causal_res.scored
                ):
                    shot_eval["is_scored"] = True
                    shot_eval["shot_type"] = (
                        "BANK_SHOT" if causal_res.is_bank_shot else "REGULAR_3C"
                    )
                    shot_eval["reason_code"] = (
                        "VALID_BANK_SHOT"
                        if causal_res.is_bank_shot
                        else "VALID_REGULAR_3C"
                    )
                    shot_eval["cushions_before_first_hit"] = max(
                        int(shot_eval.get("cushions_before_first_hit", 0)),
                        int(causal_res.cushions_before_first_hit),
                    )
                    shot_eval["cushions_before_second_hit"] = max(
                        int(shot_eval.get("cushions_before_second_hit", 0)),
                        int(causal_res.cushions_at_second_hit),
                    )
                    shot_eval["total_cushions"] = int(shot_eval["cushions_before_second_hit"])
                    bank_label = "뱅크 샷" if causal_res.is_bank_shot else "초미세 판독"
                    shot_eval["message"] = (
                        f"득점 인정 ({bank_label}): "
                        f"{shot_eval['cushions_before_second_hit']}쿠션 후 제2적구 충돌 성공!"
                    )

                # [유령 샷(Phantom Trigger) 및 120mm 미만 헛스트로크/노이즈 필터]
                # 1) 수구 총 이동거리가 120.0mm 미만이면서 제1적구 접촉이 없는 경우(카메라 떨림·큐대 노이즈), 또는
                # 2) 실제 카메라/영상 스트림(require_rest_displacement=True)에서 공이 제자리에서 잠깐 흔들렸다가 멈춘 경우,
                # 공격권(턴)이나 이닝을 절대 변경하지 않고 기존 선수의 '공격 중' 상태를 그대로 복원합니다!
                if not bool(shot_eval.get("is_scored", False)):
                    init_map = self.scoring_engine._initial_ball_positions
                    max_net_disp = 0.0
                    for bid, st in ball_states.items():
                        if bid in init_map:
                            d_net = math.hypot(
                                float(st.position[0]) - init_map[bid][0],
                                float(st.position[1]) - init_map[bid][1],
                            )
                            if d_net > max_net_disp:
                                max_net_disp = d_net

                    is_short_noise_stroke = (
                        require_rest_displacement
                        and self.scoring_engine.first_object_ball_id is None
                        and self.scoring_engine.cue_travel_distance_mm < 120.0
                    )
                    is_phantom_occlusion = is_short_noise_stroke or (
                        require_rest_displacement
                        and (
                            max_net_disp < 38.0
                            or (
                                self.scoring_engine.first_object_ball_id is None
                                and self.scoring_engine.cue_travel_distance_mm < 550.0
                            )
                        )
                    )
                    if is_phantom_occlusion:
                        self._rest_positions = {
                            bid: (float(st.position[0]), float(st.position[1]))
                            for bid, st in ball_states.items()
                        }
                        restored_payload = self.match_manager.cancel_phantom_shot()
                        await self.ws_manager.broadcast_event(
                            "SCORE_UPDATED", restored_payload
                        )
                        return None

                transition = self.match_manager.on_ball_stopped(shot_eval)

                self.match_manager.players["P1"].cue_ball_color = "WHITE"
                self.match_manager.players["P2"].cue_ball_color = "YELLOW"

                # 정지 직후 공 3개의 좌표를 다음 샷 감지를 위한 새로운 정지 기준점(_rest_positions)으로 저장
                self._rest_positions = {
                    bid: (float(st.position[0]), float(st.position[1]))
                    for bid, st in ball_states.items()
                }

                # 직전 샷 리플레이 클립 확정 및 AUTO_REPLAY_READY 브로드캐스트
                replay_meta = self.replay_buffer.lock_last_shot(
                    shot_type=str(shot_eval.get("shot_type", "MISS")),
                    is_scored=bool(shot_eval.get("is_scored", False)),
                    message=str(shot_eval.get("message", "")),
                )

                new_cue_color = str(transition["active_cue_ball_color"])
                for bid, st in self.tracker.balls.items():
                    if st.color == new_cue_color:
                        self.tracker.cue_ball_id = bid
                        break

                event_name = str(transition["event_type"])  # "SCORE_UPDATED" or "TURN_CHANGED"
                await self.ws_manager.broadcast_event(
                    event_name, dict(transition["payload"])  # type: ignore[arg-type]
                )
                await self.ws_manager.broadcast_event("AUTO_REPLAY_READY", replay_meta)
                return transition

        return None

    async def run_timer_loop(self) -> None:
        """
        백그라운드에서 1초마다:
        - 경기 시작부터 누적 경과 타이머(match_elapsed_sec)를 상시 1초씩 증가시키고,
        - 모든 공이 멈춰 있는 대기 상태(WAIT_SHOT)이면 40초 샷 클락(카운트)을 1초씩 작동시키고,
        - 만약 샷 클락(40초)이 모두 소진(0초)되면 시간 초과(TIME FOUL)로 자동 턴 교체를 수행합니다.
        """
        self.is_running = True
        while self.is_running:
            await asyncio.sleep(1.0)
            if (
                self.match_manager.fsm_state == MatchState.WAIT_SHOT
                and not self.match_manager.is_timer_paused
                and not self.match_manager.is_match_paused
            ):
                self.match_manager.tick_timer(1.0)
                if self.match_manager.remaining_timer_sec <= 0.0:
                    payload = self.match_manager.force_switch_turn()
                    self.match_manager.last_shot_result = {
                        "is_scored": False,
                        "points_awarded": 0,
                        "shot_type": "TIME_OUT_FOUL",
                        "reason_code": "SHOT_CLOCK_EXPIRED",
                        "message": "40초 공격 제한시간 초과 - 상대 선수로 공격권이 자동 교체되었습니다!",
                        "cushions_before_second_hit": 0,
                        "total_cushions": 0,
                    }
                    await self.ws_manager.broadcast_event(
                        "TURN_CHANGED", self.match_manager.get_match_payload()
                    )
                else:
                    await self.ws_manager.broadcast_event(
                        "SCORE_UPDATED", self.match_manager.get_match_payload()
                    )
            elif not self.match_manager.is_match_paused:
                # 샷 주행 중(SHOT_IN_PROGRESS)에는 누적 경기 시간만 1초씩 증가 (선수 수동 일시정지 중에는 멈춤!)
                self.match_manager.tick_match_elapsed(1.0)


class SimulateShotRequest(BaseModel):
    """프론트엔드 태블릿이나 테스트에서 즉시 시뮬레이션 시나리오를 실행하기 위한 요청 스키마"""

    scenario: str = Field(
        default="REGULAR_3C",
        description="실행할 시나리오: REGULAR_3C | BANK_SHOT | MISS_2C | VAR_MICRO_CONTACT | VAR_DOUBLE_TOUCH",
    )


class AdjustScoreRequest(BaseModel):
    """수동 점수 가감(+1 / -1) 요청 스키마"""

    player_id: str = Field(default="P1", description="P1 또는 P2")
    delta: int = Field(default=1, description="점수 변경량 (+1 또는 -1)")


class PlayerNameRequest(BaseModel):
    """선수 이름 및 핸디 변경 요청 스키마"""

    player_id: str = Field(default="P1", description="P1 또는 P2")
    name: str = Field(default="1번 선수", description="변경할 선수 이름")
    handicap: Optional[int] = Field(default=None, description="변경할 선수 핸디 점수")


class RuleModeRequest(BaseModel):
    """룰 모드 변경(STANDARD / PBA) 요청 스키마"""

    rule_mode: str = Field(default="STANDARD", description="STANDARD 또는 PBA")


# 전역 싱글톤 서비스 및 웹소켓 매니저 인스턴스 생성
ws_manager = ConnectionManager()
match_manager = MatchManager()
stream_reader = LiveStreamReader(
    match_manager=match_manager, ws_manager=ws_manager
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 서버 시작 시 백그라운드 타이머/스트림 루프를 가동하고 종료 시 정리합니다."""
    timer_task = asyncio.create_task(stream_reader.run_timer_loop())
    try:
        yield
    finally:
        stream_reader.is_running = False
        timer_task.cancel()


app = FastAPI(
    title="CUE24 Smart 3-Cushion SaaS Server",
    version=settings.VERSION,
    description="대대 3쿠션 무인 관제, 라이브캠/자동 리플레이 및 Micro-Hawk 자동 판독 서버",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 디렉토리(20인치 태블릿 UI) 마운트
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index() -> Any:
    """루트 접속 시 static/index.html이 존재하면 반환하고, 없으면 서버 상태 JSON을 반환합니다."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse(
        {
            "project": settings.PROJECT_NAME,
            "status": "running",
            "match": match_manager.get_match_payload(),
        }
    )


@app.get("/api/match/state")
async def get_match_state() -> Dict[str, Any]:
    """현재 스코어보드 전체 상태를 조회하는 REST 엔드포인트"""
    return match_manager.get_match_payload()


@app.post("/api/match/reset")
async def reset_match_state() -> Dict[str, Any]:
    """경기를 초기화하고 즉시 3구 색상 잠금 및 실시간 득점 시스템을 가동하여 모든 태블릿 점수판에 브로드캐스트합니다."""
    payload = stream_reader.reset_match_and_vision()
    await ws_manager.broadcast_event("SCORE_UPDATED", payload)
    return payload


@app.post("/api/match/activate-vision")
async def activate_vision_now() -> Dict[str, Any]:
    """화면 리플레시(새로고침) 시 즉시 3구 색상을 잠그고 실시간 득점 시스템 및 샷 클락을 가동합니다."""
    payload = stream_reader.sync_vision_and_lock_balls_now(seek_to_active_shot=True)
    await ws_manager.broadcast_event("SCORE_UPDATED", payload)
    return payload


@app.post("/api/match/timeout")
async def use_match_timeout() -> Dict[str, Any]:
    """현재 공격 선수의 타임아웃을 사용하여 타이머를 +40초 연장합니다."""
    payload = match_manager.use_timeout()
    await ws_manager.broadcast_event("SCORE_UPDATED", payload)
    return payload


@app.post("/api/match/rule-mode")
async def change_rule_mode(req: RuleModeRequest) -> Dict[str, Any]:
    """경기 룰 모드를 STANDARD(일반 1점) 또는 PBA(뱅크샷 2점)로 전환합니다."""
    payload = match_manager.set_rule_mode(req.rule_mode)
    await ws_manager.broadcast_event("SCORE_UPDATED", payload)
    return payload


@app.post("/api/match/adjust-score")
async def adjust_player_score(req: AdjustScoreRequest) -> Dict[str, Any]:
    """현장 수동 점수 가감(+1 / -1)을 반영하고 브로드캐스트합니다."""
    payload = match_manager.adjust_score(req.player_id, req.delta)
    await ws_manager.broadcast_event("SCORE_UPDATED", payload)
    return payload


@app.post("/api/match/player-name")
async def update_player_name(req: PlayerNameRequest) -> Dict[str, Any]:
    """1번 선수(P1) 또는 2번 선수(P2)의 이름(및 핸디)을 변경하고 모든 태블릿에 브로드캐스트합니다."""
    payload = match_manager.set_player_name(req.player_id, req.name, req.handicap)
    await ws_manager.broadcast_event("SCORE_UPDATED", payload)
    return payload


@app.post("/api/match/swap-player-names")
async def swap_player_names_only_endpoint() -> Dict[str, Any]:
    """[선수변경] 좌우 당구공 색상(흰공/노란공) 위치는 그대로 두고 좌우 선수의 이름(및 핸디)만 맞바꿉니다."""
    payload = match_manager.swap_player_names_only()
    await ws_manager.broadcast_event("SCORE_UPDATED", payload)
    return payload


@app.post("/api/match/swap-seats-full")
async def swap_seats_and_balls_endpoint() -> Dict[str, Any]:
    """[자리바꿈] 좌우 선수의 이름과 당구공 색깔(흰공/노란공 및 해당 진영의 점수·기록 상태)이 함께 통째로 좌우 위치가 바뀌도록 스왑합니다."""
    payload = match_manager.swap_seats_with_balls()
    await ws_manager.broadcast_event("SCORE_UPDATED", payload)
    return payload


@app.post("/api/match/pause")
async def toggle_match_pause_endpoint() -> Dict[str, Any]:
    """[일시정지] 경기 및 샷 클락 타이머 일시정지/재개 상태를 토글합니다."""
    payload = match_manager.toggle_pause_match()
    await ws_manager.broadcast_event("SCORE_UPDATED", payload)
    return payload


@app.post("/api/match/end")
async def end_match_endpoint() -> Dict[str, Any]:
    """[경기종료] 현재 경기를 종료 처리하고 결과를 브로드캐스트합니다."""
    payload = match_manager.end_match()
    await ws_manager.broadcast_event("SCORE_UPDATED", payload)
    return payload


@app.post("/api/match/swap-players")
async def swap_players_seat_endpoint() -> Dict[str, Any]:
    """선공(P1=흰공)/후공(P2=노란공) 수구 색상은 고정한 채 1번 선수와 2번 선수의 자리를 맞바꿉니다."""
    payload = match_manager.swap_player_seats()
    await ws_manager.broadcast_event("SCORE_UPDATED", payload)
    return payload


@app.post("/api/match/swap-cues")
async def swap_player_cue_balls() -> Dict[str, Any]:
    """P1과 P2의 수구 색상(WHITE <-> YELLOW)을 맞바꿉니다."""
    payload = match_manager.swap_cue_balls()
    await ws_manager.broadcast_event("SCORE_UPDATED", payload)
    return payload


@app.get("/api/video/live-frame")
async def get_single_live_frame() -> Response:
    """실시간 라이브캠(L-billiard.mp4 원본 해상도) 단일 JPEG 프레임을 반환합니다."""
    jpg_bytes = stream_reader.read_live_frame_jpeg()
    return Response(content=jpg_bytes, media_type="image/jpeg")


async def _mjpeg_live_generator() -> AsyncGenerator[bytes, None]:
    """
    실시간 카메라 또는 테스트 녹화 영상(L-billiard.mp4 원본 1920x1080)을
    절대 시계(Wall-Clock, time.perf_counter) 기준으로 처리 시간을 자동 보정하여
    정확히 실시간 1.0배속(30.0fps 정배속)으로 스트리밍 및 판독하는 제너레이터입니다.
    """
    if not stream_reader._vision_tracks_synced:
        stream_reader.sync_vision_and_lock_balls_now(seek_to_active_shot=True)
    stream_reader._reset_wall_clock_sync()

    fps = max(15.0, min(60.0, stream_reader.video_fps))
    frame_interval = 1.0 / fps
    next_frame_deadline = time.perf_counter()

    while True:
        jpg = await stream_reader.read_and_process_live_frame()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        )

        next_frame_deadline += frame_interval
        now = time.perf_counter()
        if next_frame_deadline < now - frame_interval:
            next_frame_deadline = now

        sleep_sec = next_frame_deadline - now
        if sleep_sec > 0.002:
            await asyncio.sleep(sleep_sec)
        else:
            # 2ms 이하일 때는 OS 타이머 대기 없이 이벤트 루프만 즉시 양보 (0.01ms 소요)
            await asyncio.sleep(0)


@app.get("/api/video/live")
async def stream_live_video() -> StreamingResponse:
    """보드 중앙 상단 모니터용 실시간 1.0배속 MJPEG 라이브캠 스트림 엔드포인트"""
    return StreamingResponse(
        _mjpeg_live_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


async def _mjpeg_replay_generator(speed: float = 1.0) -> AsyncGenerator[bytes, None]:
    """
    3공 정지 후 확정된 직전 샷 클립(원본 해상도)을 지정된 배속(1.0x / 0.5x / 0.25x)으로 **단 1회만** 재생하며,
    1회 재생 완료 시(또는 도중 다음 샷 출발 시) 즉시 LIVE 화면 및 실시간 득점 판독으로 자동 복귀합니다.
    """
    fps = max(15.0, min(60.0, stream_reader.video_fps))
    base_interval = 1.0 / fps
    target_interval = max(0.015, min(0.35, base_interval / max(0.25, speed)))

    clip = stream_reader.replay_buffer.get_locked_clip()
    if clip:
        for jpg in clip:
            t0 = time.perf_counter()
            # 리플레이 재생 중에도 실시간 카메라 프레임을 감시하여 수구 출발 시 즉시 SHOT_IN_PROGRESS로 전환
            if stream_reader.match_manager.fsm_state == MatchState.WAIT_SHOT:
                await stream_reader.read_and_process_live_frame()
                if stream_reader.match_manager.fsm_state == MatchState.SHOT_IN_PROGRESS:
                    break
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )
            rem = target_interval - (time.perf_counter() - t0)
            if rem > 0.002:
                await asyncio.sleep(rem)
            else:
                await asyncio.sleep(0)

        # 직전 샷 리플레이 1회 재생 완료 즉시 클라이언트에 실시간 복귀 신호 브로드캐스트
        meta = stream_reader.replay_buffer.get_last_shot_meta()
        await stream_reader.ws_manager.broadcast_event(
            "REPLAY_FINISHED",
            {
                "clip_id": meta.get("clip_id", 0),
                "frame_count": len(clip),
            },
        )

    # 1회 재생 완료 후(또는 클립이 없을 때)에는 반복 재생 없이 실시간 LIVE 프레임을 송출
    while True:
        t0 = time.perf_counter()
        jpg = await stream_reader.read_and_process_live_frame()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        )
        rem = base_interval - (time.perf_counter() - t0)
        if rem > 0.002:
            await asyncio.sleep(rem)
        else:
            await asyncio.sleep(0)


@app.get("/api/video/replay")
async def stream_replay_video(speed: float = 1.0) -> StreamingResponse:
    """3공 완전 정지 시 직전 샷을 딱 1번만 보여주는 MJPEG 리플레이 엔드포인트"""
    return StreamingResponse(
        _mjpeg_replay_generator(speed=speed),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/video/replay-info")
async def get_replay_info() -> Dict[str, Any]:
    """현재 저장된 직전 샷 리플레이 클립 메타데이터 및 원본 비디오 소스 정보를 반환합니다."""
    meta = stream_reader.replay_buffer.get_last_shot_meta()
    meta["video_source"] = stream_reader.video_path
    meta["is_real_video"] = stream_reader.is_real_video_loaded
    meta["video_width"] = stream_reader.video_width
    meta["video_height"] = stream_reader.video_height
    return meta


@app.post("/api/match/simulate-shot")
async def simulate_shot(req: SimulateShotRequest) -> Dict[str, Any]:
    """
    태블릿 화면에서 득점(일반 3C/뱅크샷), 공타(2쿠션),
    VAR 현미경 판독(미세 접촉/더블터치) 및 3공 정지 후 직전 샷 자동 리플레이를
    즉시 시연할 수 있는 시뮬레이션 엔드포인트입니다.
    실제 녹화 영상(L-billiard.mp4)이 로드된 경우 원본 1920x1080 프레임을 그대로 리플레이로 확정합니다.
    """
    scenario = req.scenario.upper()
    cue_color_before = match_manager.active_cue_ball_color

    # 1. 샷 출발 감지 -> 타이머 즉시 일시 정지
    match_manager.on_shot_started()
    await ws_manager.broadcast_event(
        "SCORE_UPDATED", match_manager.get_match_payload()
    )

    # 실제 녹화 영상(L-billiard.mp4)이 연결된 경우 원본 영상 프레임을 리플레이 버퍼에 확보
    if stream_reader.is_real_video_loaded:
        if len(stream_reader.replay_buffer._active_buffer) < 30:
            for _ in range(30):
                stream_reader.read_live_frame_jpeg()
    else:
        stream_reader.replay_buffer.clear_active()
        generated_clip = stream_reader.cam_generator.generate_shot_replay_clip(
            scenario=scenario, active_cue_color=cue_color_before
        )
        for frame_jpg in generated_clip:
            stream_reader.replay_buffer.push_frame(frame_jpg)

    var_alert: Optional[Dict[str, Any]] = None

    if scenario == "REGULAR_3C":
        shot_eval = {
            "is_scored": True,
            "shot_type": "REGULAR_3C",
            "reason_code": "VALID_REGULAR_3C",
            "message": "득점 인정 (일반 3쿠션): 총 3쿠션 후 제2적구 충돌 성공!",
            "cushions_before_first_hit": 0,
            "cushions_before_second_hit": 3,
            "total_cushions": 3,
        }
    elif scenario == "BANK_SHOT":
        pts_label = "2점" if match_manager.rule_mode == "PBA" else "1점"
        shot_eval = {
            "is_scored": True,
            "shot_type": "BANK_SHOT",
            "reason_code": "VALID_BANK_SHOT",
            "message": f"득점 인정 (뱅크 샷 +{pts_label}): 선행 3쿠션 후 제1·2적구 충돌 성공!",
            "cushions_before_first_hit": 3,
            "cushions_before_second_hit": 3,
            "total_cushions": 3,
        }
    elif scenario == "VAR_MICRO_CONTACT":
        var_alert = stream_reader.var_referee.evaluate_micro_contact(
            cue_trajectory=[(900.0, 500.0), (1000.0, 500.0), (1100.0, 501.2)],
            target_trajectory=[(1000.0, 561.35)],
        )
        await ws_manager.broadcast_event("VAR_ALERT", var_alert)
        shot_eval = {
            "is_scored": True,
            "shot_type": "REGULAR_3C",
            "reason_code": "VALID_REGULAR_3C",
            "message": str(var_alert["message"]),
            "cushions_before_first_hit": 0,
            "cushions_before_second_hit": 3,
            "total_cushions": 3,
        }
    elif scenario == "VAR_DOUBLE_TOUCH":
        var_alert = stream_reader.var_referee.evaluate_cushion_double_touch(
            cushion_distances_mm=[35.0, 29.8, 31.2, 30.1, 36.0],
            timestamps_ms=[0.0, 20.0, 45.0, 70.0, 100.0],
        )
        await ws_manager.broadcast_event("VAR_ALERT", var_alert)
        shot_eval = {
            "is_scored": False,
            "shot_type": "FOUL",
            "reason_code": "DOUBLE_TOUCH_FOUL",
            "message": str(var_alert["message"]),
            "cushions_before_first_hit": 0,
            "cushions_before_second_hit": 2,
            "total_cushions": 2,
        }
    else:
        shot_eval = {
            "is_scored": False,
            "shot_type": "MISS",
            "reason_code": "INSUFFICIENT_CUSHIONS",
            "message": "공타: 쿠션 부족 (제2적구 접촉 시점 2쿠션 < 3쿠션).",
            "cushions_before_first_hit": 0,
            "cushions_before_second_hit": 2,
            "total_cushions": 2,
        }

    # 2. 득점 인정 시 제2적구 충돌 즉시(+1점) 점수판에 선반영 후, 공 정지 시점에는 중복 가산 없이 상태만 정리
    if bool(shot_eval.get("is_scored", False)):
        imm_transition = match_manager.on_immediate_second_ball_hit(shot_eval)
        await ws_manager.broadcast_event(
            "SCORE_UPDATED", dict(imm_transition["payload"])  # type: ignore[arg-type]
        )
        shot_eval["already_dispatched"] = True

    transition = match_manager.on_ball_stopped(shot_eval)
    replay_meta = stream_reader.replay_buffer.lock_last_shot(
        shot_type=str(shot_eval["shot_type"]),
        is_scored=bool(shot_eval["is_scored"]),
        message=str(shot_eval["message"]),
    )

    await ws_manager.broadcast_event(
        str(transition["event_type"]), dict(transition["payload"])  # type: ignore[arg-type]
    )
    await ws_manager.broadcast_event("AUTO_REPLAY_READY", replay_meta)

    return {
        "transition": transition,
        "var_alert": var_alert,
        "replay_meta": replay_meta,
    }


class MCPToolCallRequest(BaseModel):
    """로컬 물리/AI 3쿠션 MCP 도구 호출 요청 스키마"""

    name: str = Field(
        default="verify_3cushion_shot_physics",
        description="실행할 MCP 도구명: verify_3cushion_shot_physics | audit_cue_ball_identity",
    )
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="MCP 도구 입력 인자 딕셔너리",
    )


@app.get("/api/mcp/tools")
async def list_mcp_tools() -> Dict[str, Any]:
    """로컬 3쿠션 물리/AI 판독 MCP 서버가 제공하는 도구 스키마 목록을 반환합니다."""
    return {
        "mcp_server": "cue24-local-billiard-physics-mcp",
        "hf_model_loaded": physics_ai_engine.hf_backend_loaded,
        "tools": billiard_mcp_server.list_tools(),
    }


@app.post("/api/mcp/call")
async def call_mcp_tool_endpoint(req: MCPToolCallRequest) -> Dict[str, Any]:
    """로컬 3쿠션 물리/AI 판독 MCP 도구를 실행하고 검증 결과를 반환합니다."""
    return billiard_mcp_server.call_tool(req.name, req.arguments)


class ReconcileCueBallRequest(BaseModel):
    """선수가 친 수구 색상 감지 기반 미인지 득점 자동 보정 및 공격자 동기화 요청 스키마"""

    stroked_cue_color: str = Field(
        default="WHITE",
        description="선수가 실제로 타구한 수구의 색상: WHITE | YELLOW",
    )
    compensate_missed_score: bool = Field(
        default=True,
        description="직전 샷이 공타로 판정된 후 동일 선수가 연속 타구한 경우 미인지 득점(+1점) 자동 복구 여부",
    )


@app.post("/api/match/reconcile-cue-ball")
async def reconcile_cue_ball_endpoint(req: ReconcileCueBallRequest) -> Dict[str, Any]:
    """
    선수가 치는 공의 색깔(WHITE=P1, YELLOW=P2)을 감지하여
    직전 미인지 득점(+1점)을 자동 소급 보정하고 현재 공격 선수를 수구 색깔에 맞게 동기화합니다.
    """
    res = match_manager.reconcile_cue_ball_stroke(
        stroked_cue_color=req.stroked_cue_color,
        compensate_missed_score=req.compensate_missed_score,
        force_sync_player=True,
    )
    await ws_manager.broadcast_event(
        "SCORE_UPDATED", dict(res["payload"])  # type: ignore[arg-type]
    )
    return res


@app.post("/api/match/switch-turn")
async def switch_player_turn() -> Dict[str, Any]:
    """선공(P1=흰공)/후공(P2=노란공)의 수구 색상은 절대 바꾸지 않고 공격권(턴)만 교대합니다."""
    match_manager.players["P1"].cue_ball_color = "WHITE"
    match_manager.players["P2"].cue_ball_color = "YELLOW"
    payload = match_manager.force_switch_turn()
    await ws_manager.broadcast_event("TURN_CHANGED", payload)
    return payload


@app.websocket("/ws/scoreboard")
async def websocket_scoreboard_endpoint(websocket: WebSocket) -> None:
    """
    20인치 태블릿 점수판 전용 실시간 양방향 WebSocket 엔드포인트 (/ws/scoreboard)
    """
    await ws_manager.connect(websocket)
    try:
        await websocket.send_json(
            {
                "type": "SCORE_UPDATED",
                "data": match_manager.get_match_payload(),
            }
        )

        while True:
            msg = await websocket.receive_json()
            action = str(msg.get("action", "")).upper()

            if action == "RESET_MATCH":
                payload = stream_reader.reset_match_and_vision()
                await ws_manager.broadcast_event("SCORE_UPDATED", payload)
            elif action == "ACTIVATE_VISION":
                payload = stream_reader.sync_vision_and_lock_balls_now(
                    seek_to_active_shot=True
                )
                await ws_manager.broadcast_event("SCORE_UPDATED", payload)
            elif action == "SIMULATE_SHOT":
                scenario_name = str(msg.get("scenario", "REGULAR_3C"))
                await simulate_shot(SimulateShotRequest(scenario=scenario_name))
            elif action == "USE_TIMEOUT":
                payload = match_manager.use_timeout()
                await ws_manager.broadcast_event("SCORE_UPDATED", payload)
            elif action == "SET_RULE_MODE":
                mode_name = str(msg.get("rule_mode", "STANDARD"))
                payload = match_manager.set_rule_mode(mode_name)
                await ws_manager.broadcast_event("SCORE_UPDATED", payload)
            elif action == "ADJUST_SCORE":
                pid = str(msg.get("player_id", "P1"))
                delta = int(msg.get("delta", 1))
                payload = match_manager.adjust_score(pid, delta)
                await ws_manager.broadcast_event("SCORE_UPDATED", payload)
            elif action == "SET_PLAYER_NAME":
                pid = str(msg.get("player_id", "P1"))
                new_name = str(msg.get("name", ""))
                handicap_val = msg.get("handicap")
                handicap_int = int(handicap_val) if handicap_val is not None else None
                payload = match_manager.set_player_name(pid, new_name, handicap_int)
                await ws_manager.broadcast_event("SCORE_UPDATED", payload)
            elif action == "SWAP_PLAYER_NAMES":
                payload = match_manager.swap_player_names_only()
                await ws_manager.broadcast_event("SCORE_UPDATED", payload)
            elif action == "SWAP_SEATS_FULL":
                payload = match_manager.swap_seats_with_balls()
                await ws_manager.broadcast_event("SCORE_UPDATED", payload)
            elif action == "START_MATCH":
                payload = stream_reader.reset_match_and_vision()
                await ws_manager.broadcast_event("SCORE_UPDATED", payload)
            elif action == "END_MATCH":
                payload = match_manager.end_match()
                await ws_manager.broadcast_event("SCORE_UPDATED", payload)
            elif action == "TOGGLE_PAUSE":
                payload = match_manager.toggle_pause_match()
                await ws_manager.broadcast_event("SCORE_UPDATED", payload)
            elif action == "SWAP_PLAYERS":
                payload = match_manager.swap_player_seats()
                await ws_manager.broadcast_event("SCORE_UPDATED", payload)
            elif action == "RECONCILE_CUE_BALL":
                cue_col = str(msg.get("stroked_cue_color", "WHITE"))
                comp = bool(msg.get("compensate_missed_score", True))
                res = match_manager.reconcile_cue_ball_stroke(
                    stroked_cue_color=cue_col,
                    compensate_missed_score=comp,
                    force_sync_player=True,
                )
                await ws_manager.broadcast_event(
                    "SCORE_UPDATED", dict(res["payload"])  # type: ignore[arg-type]
                )
            elif action == "SWITCH_TURN":
                match_manager.players["P1"].cue_ball_color = "WHITE"
                match_manager.players["P2"].cue_ball_color = "YELLOW"
                payload = match_manager.force_switch_turn()
                await ws_manager.broadcast_event("TURN_CHANGED", payload)
            elif action == "SWAP_CUES":
                payload = match_manager.swap_cue_balls()
                await ws_manager.broadcast_event("SCORE_UPDATED", payload)
            elif action == "PING":
                await websocket.send_json(
                    {
                        "type": "PONG",
                        "data": match_manager.get_match_payload(),
                    }
                )
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)
