"""
디지털 점수판(WebSocket / REST / CUE24 MatchManager) 이벤트 브릿지 및 리플레이 타임스탬프 트리거 모듈
파일 경로: three_cushion_vision/scoreboard_bridge.py

ThreeCushionScoringFSM에서 확정된 ShotResult를 디지털 점수판 표준 페이로드로 변환하여
등록된 콜백(WebSocket 브로드캐스터 등) 및 CUE24 MatchManager에 즉시 전달하고,
득점/주요 장면의 자동 리플레이 구간 타임스탬프(start_sec ~ end_sec)를 트리거합니다.
"""

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

from three_cushion_vision.scoring_fsm import ShotResult


@dataclass
class ReplayClipTrigger:
    """직전 샷 자동 리플레이 재생 구간 타임스탬프 트리거 정보"""

    clip_start_sec: float
    clip_end_sec: float
    duration_sec: float
    scored: bool
    is_bank_shot: bool
    shot_type: str
    cue_ball: str
    summary_ko: str


class ScoreboardBridge:
    """
    ThreeCushionScoringFSM의 판정 결과를 디지털 점수판(WebSocket/REST) 및
    CUE24 MatchManager에 실시간 발행하는 브릿지 클래스입니다.
    """

    def __init__(
        self,
        match_manager: Optional[Any] = None,
        pre_roll_sec: float = 0.5,
        post_roll_sec: float = 0.5,
    ) -> None:
        self.match_manager: Optional[Any] = match_manager
        self.pre_roll_sec: float = pre_roll_sec
        self.post_roll_sec: float = post_roll_sec
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []
        self.published_events: List[Dict[str, Any]] = []
        self.replay_triggers: List[ReplayClipTrigger] = []

    def register_listener(
        self, callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """WebSocket 또는 REST 점수판 이벤트 수신 콜백을 등록합니다."""
        self._listeners.append(callback)

    def publish_shot_result(self, result: ShotResult) -> Dict[str, Any]:
        """
        확정된 ShotResult를 기반으로 점수판 갱신 이벤트(SCORE_UPDATED / TURN_CHANGED)와
        리플레이 구간 타임스탬프 트리거(AUTO_REPLAY_READY)를 발행합니다.
        """
        if result.scored:
            shot_type = "BANK_SHOT" if result.is_bank_shot else "REGULAR_3C"
            reason_code = "VALID_BANK_SHOT" if result.is_bank_shot else "VALID_REGULAR_3C"
            summary_ko = (
                f"득점 인정 ({'뱅크샷' if result.is_bank_shot else '일반 3쿠션'}): "
                f"제2적구 접촉 시 {result.cushions_before_second_ball}쿠션 달성!"
            )
            event_type = "SCORE_UPDATED"
        else:
            shot_type = "MISS"
            reason_code = result.fail_reason
            summary_ko = (
                f"노카운트 ({result.fail_reason}): "
                f"제2적구 접촉 시 {result.cushions_before_second_ball}쿠션 / 총 {result.total_cushions}쿠션"
            )
            event_type = "TURN_CHANGED"

        clip_start = max(0.0, result.inning_start_time - self.pre_roll_sec)
        clip_end = max(clip_start, result.inning_end_time + self.post_roll_sec)
        replay_trigger = ReplayClipTrigger(
            clip_start_sec=round(clip_start, 3),
            clip_end_sec=round(clip_end, 3),
            duration_sec=round(clip_end - clip_start, 3),
            scored=result.scored,
            is_bank_shot=result.is_bank_shot,
            shot_type=shot_type,
            cue_ball=result.cue_ball.value,
            summary_ko=summary_ko,
        )
        self.replay_triggers.append(replay_trigger)

        # CUE24 MatchManager가 연결된 경우 즉시 이닝/점수/턴 상태 동기화
        match_payload: Optional[Dict[str, Any]] = None
        if self.match_manager is not None:
            if getattr(self.match_manager, "fsm_state", None) is not None:
                from cue24.services.match_manager import MatchState

                if self.match_manager.fsm_state == MatchState.WAIT_SHOT:
                    self.match_manager.on_shot_started()
                match_payload = self.match_manager.on_ball_stopped(
                    {
                        "is_scored": result.scored,
                        "shot_type": shot_type,
                        "reason_code": reason_code,
                        "message": summary_ko,
                        "cushions_before_second_hit": result.cushions_before_second_ball,
                        "total_cushions": result.total_cushions,
                    }
                )

        bridge_event: Dict[str, Any] = {
            "event_type": event_type,
            "shot_result": {
                "scored": result.scored,
                "cue_ball": result.cue_ball.value,
                "cushions_before_second_ball": result.cushions_before_second_ball,
                "total_cushions": result.total_cushions,
                "is_bank_shot": result.is_bank_shot,
                "fail_reason": result.fail_reason,
                "inning_start_time": round(result.inning_start_time, 3),
                "inning_end_time": round(result.inning_end_time, 3),
                "events": result.events,
            },
            "replay_trigger": asdict(replay_trigger),
            "match_payload": match_payload,
        }
        self.published_events.append(bridge_event)

        for listener in self._listeners:
            listener(bridge_event)

        return bridge_event
