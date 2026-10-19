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


class LiveStreamReader:
    """
    카메라 영상 스트림(또는 카메라 미연결 시 가상 대대 프레임)을 비동기적으로 공급하고,
    Step 2~4 컴퓨터 비전·물리·VAR 모듈, Step 5 MatchManager, Step 7 ReplayRingBuffer를
    실시간 연동하는 파이프라인 리더입니다.
    """

    def __init__(
        self,
        match_manager: MatchManager,
        ws_manager: ConnectionManager,
        camera_index: Optional[int] = None,
    ) -> None:
        self.match_manager: MatchManager = match_manager
        self.ws_manager: ConnectionManager = ws_manager
        self.camera_index: Optional[int] = camera_index
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

        # 초기 기본 리플레이 클립 준비 (첫 접속 시에도 리플레이 테스트 가능하도록)
        init_clip = self.cam_generator.generate_shot_replay_clip("REGULAR_3C", "WHITE")
        for f_bytes in init_clip:
            self.replay_buffer.push_frame(f_bytes)
        self.replay_buffer.lock_last_shot("REGULAR_3C", True, "초기 리플레이 준비 완료")

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

    async def process_coordinate_step(
        self,
        detected_points: List[Point2D],
        ball_crops_gray: Optional[Dict[int, np.ndarray]] = None,
        has_double_touch_foul: bool = False,
        var_alert_payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        단일 프레임에서 검출된 공 3개의 물리 좌표를 받아 PhysicsTracker, SpinDetector,
        ScoringEngine, MatchManager, ReplayRingBuffer를 순차 구동하고 이벤트 발생 시 브로드캐스트합니다.
        """
        self.frame_counter += 1

        # 1. 물리 충돌 가드 위치 추적기 업데이트
        ball_states = self.tracker.update(detected_points)
        cue_id = self.tracker.cue_ball_id or 1
        cue_state = ball_states[cue_id]

        # 2. WAIT_SHOT 상태에서 수구 속도가 임계값을 초과하면 즉시 샷 시작으로 전환 및 타이머 정지
        if (
            self.match_manager.fsm_state == MatchState.WAIT_SHOT
            and cue_state.speed >= settings.STOP_VELOCITY_TH
        ):
            self.match_manager.on_shot_started()
            self.spin_detector.reset()
            self.replay_buffer.clear_active()
            obj_ids = [bid for bid in ball_states.keys() if bid != cue_id]
            self.scoring_engine.reset_shot(cue_ball_id=cue_id, object_ball_ids=obj_ids)
            self._cue_trajectory = [
                (float(cue_state.prev_position[0]), float(cue_state.prev_position[1]))
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
            await self.ws_manager.broadcast_event(
                "SCORE_UPDATED", self.match_manager.get_match_payload()
            )

        # 3. 샷 진행 중(SHOT_IN_PROGRESS)일 때 3쿠션 이벤트 기록 및 정지 감지
        if self.match_manager.fsm_state == MatchState.SHOT_IN_PROGRESS:
            positions_map: Dict[int, Point2D] = {
                bid: (float(st.position[0]), float(st.position[1]))
                for bid, st in ball_states.items()
            }
            cue_vel = (float(cue_state.velocity[0]), float(cue_state.velocity[1]))
            self.scoring_engine.update_frame(
                frame_idx=self.frame_counter,
                ball_positions=positions_map,
                cue_velocity=cue_vel,
            )

            self._cue_trajectory.append(positions_map[cue_id])
            for bid, tr_list in self._target_trajectories.items():
                if bid in positions_map:
                    tr_list.append(positions_map[bid])

            # 외부에서 명시적 VAR 알림이 주입되었거나 미세 접촉 판독이 필요한 경우 브로드캐스트
            if var_alert_payload is not None and var_alert_payload.get("var_triggered"):
                await self.ws_manager.broadcast_event("VAR_ALERT", var_alert_payload)

            # 스핀 및 선속도 정지 감지 업데이트
            spin_status = self.spin_detector.update(
                ball_states=ball_states,
                ball_crops_gray=ball_crops_gray,
            )

            # 15프레임 연속 정지 확정(BALL_STOPPED) 시 룰 평가 및 스코어/턴 갱신 + 자동 리플레이 브로드캐스트
            if bool(spin_status["is_fully_stopped"]):
                shot_eval = self.scoring_engine.evaluate_shot(
                    has_double_touch_foul=has_double_touch_foul
                )
                transition = self.match_manager.on_ball_stopped(shot_eval)

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
        백그라운드에서 1초마다 40초 샷 타이머를 차감하고 태블릿 점수판에 동기화합니다.
        """
        self.is_running = True
        while self.is_running:
            await asyncio.sleep(1.0)
            if (
                self.match_manager.fsm_state == MatchState.WAIT_SHOT
                and not self.match_manager.is_timer_paused
            ):
                self.match_manager.tick_timer(1.0)
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
    """실시간 라이브캠 단일 JPEG 프레임을 반환합니다."""
    jpg_bytes = stream_reader.cam_generator.render_live_frame(
        active_cue_color=match_manager.active_cue_ball_color,
        fsm_state=match_manager.fsm_state.value,
    )
    return Response(content=jpg_bytes, media_type="image/jpeg")


async def _mjpeg_live_generator() -> AsyncGenerator[bytes, None]:
    """실시간 카메라(또는 가상 대대) MJPEG 멀티파트 스트림 제너레이터"""
    while True:
        jpg = stream_reader.cam_generator.render_live_frame(
            active_cue_color=match_manager.active_cue_ball_color,
            fsm_state=match_manager.fsm_state.value,
        )
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        )
        await asyncio.sleep(0.08)


@app.get("/api/video/live")
async def stream_live_video() -> StreamingResponse:
    """보드 중앙 상단 모니터용 실시간 MJPEG 라이브캠 스트림 엔드포인트"""
    return StreamingResponse(
        _mjpeg_live_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


async def _mjpeg_replay_generator(speed: float = 1.0) -> AsyncGenerator[bytes, None]:
    """3공 정지 후 확정된 직전 샷 클립을 반복 재생하는 MJPEG 스트림 제너레이터"""
    delay = max(0.03, min(0.4, 0.06 / max(0.25, speed)))
    while True:
        clip = stream_reader.replay_buffer.get_locked_clip()
        if not clip:
            jpg = stream_reader.cam_generator.render_live_frame()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )
            await asyncio.sleep(0.1)
            continue

        for jpg in clip:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )
            await asyncio.sleep(delay)
        await asyncio.sleep(0.4)


@app.get("/api/video/replay")
async def stream_replay_video(speed: float = 1.0) -> StreamingResponse:
    """3공 완전 정지 시 직전 샷을 자동으로 반복 보여주는 MJPEG 리플레이 엔드포인트"""
    return StreamingResponse(
        _mjpeg_replay_generator(speed=speed),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/video/replay-info")
async def get_replay_info() -> Dict[str, Any]:
    """현재 저장된 직전 샷 리플레이 클립의 메타데이터를 반환합니다."""
    return stream_reader.replay_buffer.get_last_shot_meta()


@app.post("/api/match/simulate-shot")
async def simulate_shot(req: SimulateShotRequest) -> Dict[str, Any]:
    """
    실제 카메라 없이도 태블릿 화면에서 득점(일반 3C/뱅크샷), 공타(2쿠션),
    VAR 현미경 판독(미세 접촉/더블터치) 및 3공 정지 후 직전 샷 자동 리플레이를
    즉시 시연할 수 있는 시뮬레이션 엔드포인트입니다.
    """
    scenario = req.scenario.upper()
    cue_color_before = match_manager.active_cue_ball_color

    # 1. 샷 출발 감지 -> 타이머 즉시 일시 정지 및 리플레이 링버퍼 초기화
    match_manager.on_shot_started()
    stream_reader.replay_buffer.clear_active()
    await ws_manager.broadcast_event(
        "SCORE_UPDATED", match_manager.get_match_payload()
    )

    # 시나리오에 맞는 주행 프레임들을 링버퍼에 녹화
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
