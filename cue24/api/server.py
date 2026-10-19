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
from cue24.services.match_manager import MatchManager, MatchState

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


import math
import cv2


class LiveStreamReader:
    """
    실제 당구대 카메라 스트림 또는 카메라 대용 테스트 녹화 영상("J:\\내 드라이브\\L-billiard.mp4")을
    원본 해상도(1920x1080) 크기·비율 변동 없이 공급하고,
    Step 2~4 컴퓨터 비전·물리·VAR 모듈, Step 5 MatchManager, Step 7 ReplayRingBuffer를
    실시간 연동하는 파이프라인 리더입니다.
    """

    # 실제 카메라/비디오 비전 스트림에서 예비 큐질(그림자 흔들림)을 배제하고 실제 수구 출발만 감지하기 위한 임계값
    VISION_START_SPEED_TH: float = 4.5
    VISION_START_DISP_MM: float = 14.0

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

        # Step 2 ~ Step 4 핵심 코어 모듈 및 Step 7 리플레이 버퍼 초기화
        default_corners = [
            (0.0, 0.0),
            (settings.TABLE_WIDTH_MM, 0.0),
            (settings.TABLE_WIDTH_MM, settings.TABLE_HEIGHT_MM),
            (0.0, settings.TABLE_HEIGHT_MM),
        ]
        self.calibrator = TableCalibrator(src_corners=default_corners)
        self.color_locker = ColorLocker()
        self.detector = BallCandidateDetector()
        self.tracker = PhysicsTracker()
        self.spin_detector = SpinDetector()
        self.scoring_engine = ScoringEngine()
        self.var_referee = VARReferee()
        self.replay_buffer = ReplayRingBuffer(max_frames=300)
        self.cam_generator = SyntheticCamGenerator(width=960, height=480)

        # 비전 자동 캘리브레이션 및 정지 기준 위치 상태
        self._table_calibrated: bool = False
        self._vision_tracks_synced: bool = False
        self._rest_positions: Dict[int, Point2D] = {}
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

        self._init_video_capture(self.video_path)

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

    def _init_video_capture(self, path_str: str) -> None:
        """
        실제 카메라(camera_index) 또는 카메라 대용 테스트 영상 파일(예: J:\\내 드라이브\\L-billiard.mp4)을 오픈하고,
        당구대 내경 4모서리를 자동 보정한 뒤 초기 프레임 및 리플레이 버퍼를 준비합니다.
        """
        cap: Optional[cv2.VideoCapture] = None
        if self.camera_index is not None:
            cap = cv2.VideoCapture(self.camera_index)
        else:
            video_file = Path(path_str)
            if video_file.exists():
                cap = cv2.VideoCapture(str(video_file))

        if cap is not None and cap.isOpened():
            self._video_cap = cap
            self.is_real_video_loaded = True
            self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
            self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
            self.video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

            # 초기 40프레임을 원본 사이즈 그대로 읽어 리플레이 버퍼에 적재 및 첫 프레임으로 당구대 자동 보정
            for idx in range(40):
                ret, raw_frame = cap.read()
                if not ret or raw_frame is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    break
                if idx == 0:
                    self.calibrator.auto_detect_table_corners(raw_frame)
                    self._table_calibrated = True
                self._latest_raw_frame = raw_frame
                ok, buf = cv2.imencode(
                    ".jpg", raw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88]
                )
                if ok:
                    jpg_bytes = buf.tobytes()
                    self._latest_live_jpeg = jpg_bytes
                    self.replay_buffer.push_frame(jpg_bytes)

            self.replay_buffer.lock_last_shot(
                "REGULAR_3C", True, "카메라/테스트 영상 초기 리플레이 준비 완료"
            )
            return

        # 비디오 소스가 없을 경우에만 합성 카메라 프레임으로 폴백
        init_clip = self.cam_generator.generate_shot_replay_clip("REGULAR_3C", "WHITE")
        for f_bytes in init_clip:
            self.replay_buffer.push_frame(f_bytes)
        self.replay_buffer.lock_last_shot("REGULAR_3C", True, "초기 리플레이 준비 완료")

    @staticmethod
    def _crop_ball_patch(
        warped_bgr: np.ndarray, center_pt: Point2D, patch_radius: int = 31
    ) -> np.ndarray:
        """2840x1420 평면 보정 영상에서 특정 공 중심 주변의 정사각형 BGR 패치를 안전하게 추출합니다."""
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
        완화된 진원도 후보 및 직전 추적 위치를 결합하여 3개 공의 좌표를 끊김 없이 반환합니다.
        """
        candidates = self.detector.detect(warped_bgr, max_balls=3)
        if len(candidates) == 3:
            return [(float(c["x"]), float(c["y"])) for c in candidates]

        # 아직 비전 트랙이 초기화되지 않았으면 3개가 온전히 검출될 때까지 빈 리스트 반환
        if not self.tracker.balls or len(self.tracker.balls) != 3:
            return []

        detected_pts: List[Point2D] = [
            (float(c["x"]), float(c["y"])) for c in candidates
        ]
        if not detected_pts:
            return [
                (float(st.position[0]), float(st.position[1]))
                for st in self.tracker.balls.values()
            ]

        # 검출된 점들과 가장 가까운 기존 트랙을 먼저 매칭하고, 누락된 공은 직전 위치로 유지
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
            st = self.tracker.balls[missing_id]
            completed_pts.append((float(st.position[0]), float(st.position[1])))

        return completed_pts[:3]

    async def process_video_frame(
        self, raw_frame: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        실제 카메라(또는 카메라 대용 영상) 단일 BGR 프레임을 입력받아:
        1) 당구대 내경 2840x1420mm 호모그래피 보정
        2) 진원도 필터 기반 당구공 3개 검출 및 ColorLocker 3색(WHITE, YELLOW, RED) 분류
        3) 처음 움직이는 공을 수구로 자동 인식 -> 샷 클락 정지 -> 3쿠션 득점 판정 -> 모든 공 정지 시 득점/턴 전환 & 샷 클락 재작동
        전체 비전 파이프라인을 실시간 구동합니다.
        """
        if raw_frame is None or raw_frame.size == 0:
            return None

        if not self._table_calibrated:
            self.calibrator.auto_detect_table_corners(raw_frame)
            self._table_calibrated = True

        # 고속 비전 처리를 위해 0.5x 스케일(1420x710) 행렬로 단일 패스 워프 후 2.0x(2840x1420mm) 물리 좌표로 변환
        scale_mat = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        half_H = scale_mat @ self.calibrator.homography_matrix.astype(np.float64)
        warped_half = cv2.warpPerspective(
            raw_frame,
            half_H,
            dsize=(1420, 710),
            flags=cv2.INTER_LINEAR,
        )

        # 1420x710 영상에서 검출할 때는 면적이 1/4이므로 스케일 보정 검출 수행
        hsv = cv2.cvtColor(warped_half, cv2.COLOR_BGR2HSV)
        cloth_mask = cv2.inRange(hsv, self.detector.hsv_blue_lower, self.detector.hsv_blue_upper)
        fg_mask = cv2.bitwise_not(cloth_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        strict_candidates: List[Dict[str, float]] = []
        for cnt in contours:
            raw_area = float(cv2.contourArea(cnt))
            area = raw_area * 4.0
            if area < self.detector.min_area or area > self.detector.max_area:
                continue
            perim = float(cv2.arcLength(cnt, True))
            if perim <= 0.0:
                continue
            circ = (4.0 * math.pi * raw_area) / (perim * perim)
            if circ <= self.detector.CIRCULARITY_TH:
                continue
            (cx, cy), rad = cv2.minEnclosingCircle(cnt)
            strict_candidates.append(
                {
                    "x": float(cx) * 2.0,
                    "y": float(cy) * 2.0,
                    "radius": float(rad) * 2.0,
                    "circularity": float(circ),
                    "area": area,
                }
            )
        strict_candidates.sort(
            key=lambda c: (-c["circularity"], abs(c["area"] - self.detector.ideal_area))
        )
        strict_candidates = strict_candidates[:3]

        # 최초 비전 동기화: 3개의 공이 온전히 검출되었을 때 색상(WHITE, YELLOW, RED)과 초기 위치를 확정
        if not self._vision_tracks_synced:
            if len(strict_candidates) == 3:
                init_pts: Dict[int, Point2D] = {
                    i + 1: (float(c["x"]), float(c["y"]))
                    for i, c in enumerate(strict_candidates)
                }
                init_crops = {
                    bid: self._crop_ball_patch(
                        warped_half, (pt[0] * 0.5, pt[1] * 0.5), patch_radius=16
                    )
                    for bid, pt in init_pts.items()
                }
                colors = self.color_locker.classify_balls(init_crops)
                self.detected_ball_colors = colors.copy()

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

        if len(strict_candidates) == 3:
            pts: List[Point2D] = [
                (float(c["x"]), float(c["y"])) for c in strict_candidates
            ]
        elif self.tracker.balls and len(self.tracker.balls) == 3:
            detected_pts = [(float(c["x"]), float(c["y"])) for c in strict_candidates]
            remaining_ids = set(self.tracker.balls.keys())
            pts = list(detected_pts)
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
                st = self.tracker.balls[missing_id]
                pts.append((float(st.position[0]), float(st.position[1])))
            pts = pts[:3]
        else:
            return None

        # 정지 대기 상태(WAIT_SHOT)이고 3개 공이 모두 선명하게 검출되었을 때 ColorLocker로 색상 지속 검증
        detected_colors: Optional[Dict[int, str]] = None
        if (
            self.match_manager.fsm_state == MatchState.WAIT_SHOT
            and len(strict_candidates) == 3
        ):
            matched_preview = self.tracker._match_nearest_positions(pts)
            # 두 공이 너무 밀착해 있지 않을 때만 색상 재분류 (영역 침범 방지)
            min_pair_dist = min(
                math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                for i, p1 in enumerate(pts)
                for p2 in pts[i + 1 :]
            )
            if min_pair_dist >= 68.0:
                crops_by_id = {
                    bid: self._crop_ball_patch(
                        warped_half, (coord[0] * 0.5, coord[1] * 0.5), patch_radius=16
                    )
                    for bid, coord in matched_preview.items()
                }
                detected_colors = self.color_locker.classify_balls(crops_by_id)
                self.detected_ball_colors = detected_colors.copy()

                # 모든 공이 정지해 있는 동안 기준 정지 위치를 부드럽게 유지 (큐대 근접 시 미세 흔들림 기준점)
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

    def read_live_frame_jpeg(self) -> bytes:
        """
        실제 카메라 또는 테스트 녹화 영상(L-billiard.mp4)에서 다음 프레임을 원본 해상도(1920x1080) 그대로 읽어
        크기/비율 변형 없이 JPEG 바이트로 반환하며, 링버퍼에 상시 기록합니다.
        """
        if self.is_real_video_loaded and self._video_cap is not None:
            ret, raw_frame = self._video_cap.read()
            if not ret or raw_frame is None:
                self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._vision_tracks_synced = False
                ret, raw_frame = self._video_cap.read()

            if ret and raw_frame is not None:
                self._latest_raw_frame = raw_frame
                # 카메라/비디오 원본 해상도 그대로 인코딩 (리사이즈/크롭 절대 없음)
                ok, buf = cv2.imencode(
                    ".jpg", raw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88]
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

    async def read_and_process_live_frame(self) -> bytes:
        """
        실시간 스트림에서 프레임을 읽고 즉시 비전·물리·3쿠션 득점 파이프라인을 구동한 뒤 JPEG 바이트를 반환합니다.
        """
        jpg_bytes = self.read_live_frame_jpeg()
        if self.is_real_video_loaded and self._latest_raw_frame is not None:
            await self.process_video_frame(self._latest_raw_frame)
        return jpg_bytes

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
        단일 프레임에서 검출된 공 3개의 물리 좌표를 받아:
        1) WAIT_SHOT 상태에서 처음 움직이는 공을 수구(Cue Ball)로 자동 인식하고,
           샷 클락(타이머)을 즉시 멈춘 뒤 3쿠션 득점 알고리즘(ScoringEngine)을 가동합니다.
        2) SHOT_IN_PROGRESS 상태에서 쿠션 반사 및 제1·2적구 충돌을 실시간 카운트합니다.
        3) 모든 공이 멈추면(BALL_STOPPED) 득점 성공 시 점수 가산, 실패 시 자동 턴 전환 후 샷 클락을 재작동합니다.
        """
        self.frame_counter += 1

        # 1. 물리 충돌 가드 위치 추적기 업데이트 및 감지된 공 색상 반영
        ball_states = self.tracker.update(detected_points)
        if detected_colors:
            for bid, col in detected_colors.items():
                if bid in ball_states:
                    ball_states[bid].color = col

        # 2. WAIT_SHOT 상태에서 처음 움직이는 공을 수구(Cue Ball)로 인식 및 샷 시작/타이머 정지
        if self.match_manager.fsm_state == MatchState.WAIT_SHOT:
            vel_th = (
                start_velocity_th
                if start_velocity_th is not None
                else settings.STOP_VELOCITY_TH
            )
            moving_candidates: List[Tuple[float, int]] = []

            for bid, st in ball_states.items():
                if st.speed < vel_th:
                    continue
                if require_rest_displacement and bid in self._rest_positions:
                    rx, ry = self._rest_positions[bid]
                    disp = math.hypot(
                        float(st.position[0]) - rx, float(st.position[1]) - ry
                    )
                    if disp < self.VISION_START_DISP_MM:
                        continue
                # WHITE 또는 YELLOW 공에 우선순위를 부여하되 가장 빠르게 출발한 공을 선택
                color_bonus = 1000.0 if st.color in ("WHITE", "YELLOW") else 0.0
                moving_candidates.append((color_bonus + st.speed, bid))

            if moving_candidates:
                moving_candidates.sort(reverse=True)
                first_moving_id = moving_candidates[0][1]

                # 처음 움직이는 공을 이번 샷의 수구(Cue Ball)로 고정!
                self.tracker.cue_ball_id = first_moving_id
                cue_id = first_moving_id
                cue_state = ball_states[cue_id]

                # 현재 공격 플레이어의 수구 색상 표시를 실제 출발한 수구 색상(WHITE/YELLOW)과 동기화
                if cue_state.color in ("WHITE", "YELLOW"):
                    if self.match_manager.active_cue_ball_color != cue_state.color:
                        self.match_manager.swap_cue_balls()

                # 수구가 움직였으므로 샷 클락(카운트) 작동을 즉시 멈추고 득점 알고리즘 가동
                self.match_manager.on_shot_started()
                self.spin_detector.reset()
                self.replay_buffer.clear_active()
                obj_ids = [bid for bid in ball_states.keys() if bid != cue_id]
                self.scoring_engine.reset_shot(
                    cue_ball_id=cue_id, object_ball_ids=obj_ids
                )
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
                self.match_manager.last_shot_result = {
                    "is_scored": False,
                    "points_awarded": 0,
                    "shot_type": "IN_PROGRESS",
                    "reason_code": "CUE_MOVING",
                    "message": (
                        f"수구({cue_state.color}) 출발 감지 — 샷 클락 정지 및 3쿠션 판독 가동 중"
                    ),
                    "cushions_before_second_hit": 0,
                    "total_cushions": 0,
                }
                await self.ws_manager.broadcast_event(
                    "SCORE_UPDATED", self.match_manager.get_match_payload()
                )

        # 3. 샷 진행 중(SHOT_IN_PROGRESS)일 때 3쿠션 이벤트 기록 및 정지 감지
        if self.match_manager.fsm_state == MatchState.SHOT_IN_PROGRESS:
            cue_id = self.tracker.cue_ball_id or 1
            cue_state = ball_states[cue_id]
            positions_map: Dict[int, Point2D] = {
                bid: (float(st.position[0]), float(st.position[1]))
                for bid, st in ball_states.items()
            }
            cue_vel = (float(cue_state.velocity[0]), float(cue_state.velocity[1]))

            prev_cushions = self.scoring_engine.total_cushion_count
            prev_events_count = len(self.scoring_engine.events)

            self.scoring_engine.update_frame(
                frame_idx=self.frame_counter,
                ball_positions=positions_map,
                cue_velocity=cue_vel,
            )

            self._cue_trajectory.append(positions_map[cue_id])
            for bid, tr_list in self._target_trajectories.items():
                if bid in positions_map:
                    tr_list.append(positions_map[bid])

            # 주행 중 쿠션 횟수나 목적구 타격이 새로 감지되면 점수판 UI에 실시간 반영
            if (
                self.scoring_engine.total_cushion_count != prev_cushions
                or len(self.scoring_engine.events) != prev_events_count
            ):
                hit1 = "완료" if self.scoring_engine.first_object_ball_id is not None else "대기"
                hit2 = "완료" if self.scoring_engine.second_object_ball_id is not None else "대기"
                self.match_manager.last_shot_result = {
                    "is_scored": False,
                    "points_awarded": 0,
                    "shot_type": "IN_PROGRESS",
                    "reason_code": "TRACKING",
                    "message": (
                        f"판독 중: {self.scoring_engine.total_cushion_count}쿠션 "
                        f"| 1적구({hit1}) · 2적구({hit2})"
                    ),
                    "cushions_before_second_hit": self.scoring_engine.cushion_count_before_second_hit,
                    "total_cushions": self.scoring_engine.total_cushion_count,
                }
                await self.ws_manager.broadcast_event(
                    "SCORE_UPDATED", self.match_manager.get_match_payload()
                )

            # 외부에서 명시적 VAR 알림이 주입되었거나 미세 접촉 판독이 필요한 경우 브로드캐스트
            if var_alert_payload is not None and var_alert_payload.get("var_triggered"):
                await self.ws_manager.broadcast_event("VAR_ALERT", var_alert_payload)

            # 스핀 및 선속도 정지 감지 업데이트
            spin_status = self.spin_detector.update(
                ball_states=ball_states,
                ball_crops_gray=ball_crops_gray,
            )

            # 15프레임 연속 정지 확정(BALL_STOPPED) 시 룰 평가 및 스코어/턴 갱신 + 샷 클락 재작동
            if bool(spin_status["is_fully_stopped"]):
                shot_eval = self.scoring_engine.evaluate_shot(
                    has_double_touch_foul=has_double_touch_foul
                )
                transition = self.match_manager.on_ball_stopped(shot_eval)

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
        - 모든 공이 멈춰 있는 대기 상태(WAIT_SHOT)이면 40초 샷 클락(카운트)을 1초씩 작동시키고,
        - 만약 샷 클락(40초)이 모두 소진(0초)되면 시간 초과(TIME FOUL)로 자동 턴 교체를 수행합니다.
        """
        self.is_running = True
        while self.is_running:
            await asyncio.sleep(1.0)
            if (
                self.match_manager.fsm_state == MatchState.WAIT_SHOT
                and not self.match_manager.is_timer_paused
            ):
                self.match_manager.tick_timer(1.0)
                if self.match_manager.remaining_timer_sec <= 0.0:
                    payload = self.match_manager.force_switch_turn()
                    self.match_manager.last_shot_result = {
                        "is_scored": False,
                        "points_awarded": 0,
                        "shot_type": "TIME_OUT_FOUL",
                        "reason_code": "SHOT_CLOCK_EXPIRED",
                        "message": "40초 샷 클락 시간 초과 — 상대 선수로 자동 턴 교체!",
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
    """경기를 초기화하고 연결된 모든 태블릿 점수판에 브로드캐스트합니다."""
    payload = match_manager.reset_match()
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
    """실시간 카메라 또는 테스트 녹화 영상(L-billiard.mp4 원본 1920x1080) MJPEG 스트림 및 비전 판독 제너레이터"""
    frame_interval = 1.0 / max(15.0, min(60.0, stream_reader.video_fps))
    while True:
        jpg = await stream_reader.read_and_process_live_frame()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        )
        await asyncio.sleep(frame_interval)


@app.get("/api/video/live")
async def stream_live_video() -> StreamingResponse:
    """보드 중앙 상단 모니터용 실시간 MJPEG 라이브캠 스트림 엔드포인트"""
    return StreamingResponse(
        _mjpeg_live_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


async def _mjpeg_replay_generator(speed: float = 1.0) -> AsyncGenerator[bytes, None]:
    """
    3공 정지 후 확정된 직전 샷 클립(원본 해상도)을 반복 재생하며,
    동시에 카메라의 다음 샷(수구 출발)을 감시하여 수구가 움직이면 즉시 LIVE 및 득점 판독으로 전환합니다.
    """
    base_interval = 1.0 / max(15.0, min(60.0, stream_reader.video_fps))
    delay = max(0.02, min(0.35, base_interval / max(0.25, speed)))
    while True:
        clip = stream_reader.replay_buffer.get_locked_clip()
        if not clip:
            jpg = await stream_reader.read_and_process_live_frame()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )
            await asyncio.sleep(0.08)
            continue

        for jpg in clip:
            # 리플레이 재생 중에도 실시간 카메라 프레임을 감시하여 수구 출발 시 즉시 SHOT_IN_PROGRESS로 전환
            if stream_reader.match_manager.fsm_state == MatchState.WAIT_SHOT:
                await stream_reader.read_and_process_live_frame()
                if stream_reader.match_manager.fsm_state == MatchState.SHOT_IN_PROGRESS:
                    break
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )
            await asyncio.sleep(delay)
        await asyncio.sleep(0.35)


@app.get("/api/video/replay")
async def stream_replay_video(speed: float = 1.0) -> StreamingResponse:
    """3공 완전 정지 시 직전 샷을 자동으로 반복 보여주는 MJPEG 리플레이 엔드포인트"""
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

    # 2. 공 3개 완전 정지(BALL_STOPPED) 판정 반영 및 직전 샷 리플레이 클립 Lock
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
                payload = match_manager.reset_match()
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
