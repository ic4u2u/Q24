"""
CUE24 FastAPI 실시간 WebSocket 및 REST API 관제 서버
파일 경로: cue24/api/server.py

1. WebSocket 엔드포인트 (/ws/scoreboard) 및 ConnectionManager를 통한 실시간 태블릿 브로드캐스트
2. LiveStreamReader 카메라/시뮬레이션 영상 루프와 Step 2~4 비전·물리·VAR 엔진 통합 파이프라인
3. 실시간 이벤트 송출:
   - SCORE_UPDATED (득점 및 점수판 갱신)
   - TURN_CHANGED (공타 시 턴 및 수구 색상 전환)
   - VAR_ALERT (미세 접촉 또는 쿠션 더블터치 현미경 판독 팝업 트리거)
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cue24.config.settings import settings
from cue24.core.color_locker import ColorLocker
from cue24.core.physics_tracker import (
    BallCandidateDetector,
    PhysicsTracker,
    SpinDetector,
)
from cue24.core.scoring_engine import ScoringEngine
from cue24.core.table_calibrator import TableCalibrator
from cue24.core.var_referee import VARReferee
from cue24.services.match_manager import MatchManager, MatchState

Point2D = Tuple[float, float]


class ConnectionManager:
    """
    연결된 20인치 태블릿 점수판 브라우저 클라이언트들을 관리하고,
    경기 상태 및 VAR 판독 이벤트를 실시간 브로드캐스트하는 웹소켓 매니저입니다.
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
    Step 2~4 컴퓨터 비전·물리·VAR 모듈과 Step 5 MatchManager를 실시간 연동하는 파이프라인 리더입니다.
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

        # Step 2 ~ Step 4 핵심 코어 모듈 인스턴스 초기화
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

        # 현재 샷 주행 중 궤적 버퍼 (VAR 미세 접촉 판독용)
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
        ScoringEngine, MatchManager를 순차 구동하고 이벤트 발생 시 WebSocket으로 즉시 브로드캐스트합니다.
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

            # 15프레임 연속 정지 확정(BALL_STOPPED) 시 룰 평가 및 스코어/턴 갱신 브로드캐스트
            if bool(spin_status["is_fully_stopped"]):
                shot_eval = self.scoring_engine.evaluate_shot(
                    has_double_touch_foul=has_double_touch_foul
                )
                transition = self.match_manager.on_ball_stopped(shot_eval)

                # 만약 턴이 바뀌었다면 다음 턴 수구(WHITE <-> YELLOW)로 트래커의 cue_ball_id 갱신
                new_cue_color = str(transition["active_cue_ball_color"])
                for bid, st in self.tracker.balls.items():
                    if st.color == new_cue_color:
                        self.tracker.cue_ball_id = bid
                        break

                event_name = str(transition["event_type"])  # "SCORE_UPDATED" or "TURN_CHANGED"
                await self.ws_manager.broadcast_event(
                    event_name, dict(transition["payload"])  # type: ignore[arg-type]
                )
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
    description="대대 3쿠션 무인 관제 및 Micro-Hawk 자동 판독 실시간 WebSocket 서버",
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


@app.post("/api/match/simulate-shot")
async def simulate_shot(req: SimulateShotRequest) -> Dict[str, Any]:
    """
    실제 카메라 없이도 태블릿 화면에서 득점(일반 3C/뱅크샷), 공타(2쿠션),
    VAR 현미경 판독(미세 접촉/더블터치)을 즉시 시연할 수 있는 시뮬레이션 엔드포인트입니다.
    """
    scenario = req.scenario.upper()

    # 1. 샷 출발 감지 -> 타이머 즉시 일시 정지
    match_manager.on_shot_started()
    await ws_manager.broadcast_event(
        "SCORE_UPDATED", match_manager.get_match_payload()
    )

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
        shot_eval = {
            "is_scored": True,
            "shot_type": "BANK_SHOT",
            "reason_code": "VALID_BANK_SHOT",
            "message": "득점 인정 (뱅크 샷): 선행 3쿠션 후 제1·2적구 충돌 성공!",
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
        # 기본 공타 시나리오 (2쿠션 부족)
        shot_eval = {
            "is_scored": False,
            "shot_type": "MISS",
            "reason_code": "INSUFFICIENT_CUSHIONS",
            "message": "공타: 쿠션 부족 (제2적구 접촉 시점 2쿠션 < 3쿠션).",
            "cushions_before_first_hit": 0,
            "cushions_before_second_hit": 2,
            "total_cushions": 2,
        }

    # 2. 공 정지(BALL_STOPPED) 판정 반영 및 SCORE_UPDATED / TURN_CHANGED 브로드캐스트
    transition = match_manager.on_ball_stopped(shot_eval)
    await ws_manager.broadcast_event(
        str(transition["event_type"]), dict(transition["payload"])  # type: ignore[arg-type]
    )

    return {
        "transition": transition,
        "var_alert": var_alert,
    }


@app.websocket("/ws/scoreboard")
async def websocket_scoreboard_endpoint(websocket: WebSocket) -> None:
    """
    20인치 태블릿 점수판 전용 실시간 양방향 WebSocket 엔드포인트 (/ws/scoreboard)
    - 연결 즉시 현재 스코어보드 상태(SCORE_UPDATED)를 전송합니다.
    - 태블릿에서 제어 명령(RESET_MATCH, SIMULATE_SHOT 등)을 보낼 경우 즉시 처리하여 브로드캐스트합니다.
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
