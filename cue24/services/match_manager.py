"""
대대 3쿠션 경기 규칙 FSM(유한 상태 머신) 및 스코어/타이머/이닝 관리자
파일 경로: cue24/services/match_manager.py

1. FSM 상태 관리: WAIT_SHOT (샷 대기 및 타이머 작동) -> SHOT_IN_PROGRESS (주행 중, 타이머 정지) -> BALL_STOPPED (판정 및 정산)
2. 플레이어 P1(WHITE), P2(YELLOW) 상태 관리: 점수, 이닝, 에버리지(점수/이닝), 현재 런(Run), 하이런(High Run), 수구 색상
3. 샷 감지 시 40초 타이머 즉시 일시 정지(Pause)
4. BALL_STOPPED 수신 시 득점/공타 판정에 따른 점수 가산, 턴 교체, 수구 색상 스위칭, 타이머 40초 리셋 후 재시작
5. get_match_payload(): 20인치 태블릿 스코어보드 렌더링용 전체 JSON 페이로드 반환
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from cue24.config.settings import settings


class MatchState(str, Enum):
    """3쿠션 경기 진행 유한 상태 머신(FSM) 상태 정의"""

    WAIT_SHOT = "WAIT_SHOT"
    SHOT_IN_PROGRESS = "SHOT_IN_PROGRESS"
    BALL_STOPPED = "BALL_STOPPED"


@dataclass
class PlayerState:
    """개별 플레이어(P1, P2)의 경기 기록 및 수구 상태 데이터 클래스"""

    player_id: str          # "P1" 또는 "P2"
    name: str               # 선수 이름
    cue_ball_color: str     # "WHITE" 또는 "YELLOW"
    score: int = 0          # 현재 득점
    innings: int = 1        # 소화한 이닝 수 (선공 P1은 1이닝부터 시작)
    current_run: int = 0    # 현재 이닝 연속 득점
    high_run: int = 0       # 경기 최고 연속 득점 (하이런)
    target_score: int = 40  # 목표 점수 (대대 표준 40점 등)

    @property
    def average(self) -> float:
        """에버리지(Avg = 점수 / 이닝) 계산 (소수점 셋째 자리 반올림)"""
        if self.innings <= 0:
            return 0.000
        return round(self.score / float(self.innings), 3)

    def to_dict(self) -> Dict[str, object]:
        """JSON 직렬화용 딕셔너리 변환"""
        return {
            "player_id": self.player_id,
            "name": self.name,
            "cue_ball_color": self.cue_ball_color,
            "score": self.score,
            "innings": self.innings,
            "average": self.average,
            "current_run": self.current_run,
            "high_run": self.high_run,
            "target_score": self.target_score,
        }


class MatchManager:
    """
    3쿠션 경기의 FSM 상태, 턴(공격권), 점수, 이닝, 에버리지, 하이런, 40초 샷 타이머를
    통합 관리하는 핵심 서비스 클래스입니다.
    """

    def __init__(
        self,
        p1_name: str = "PLAYER 1",
        p2_name: str = "PLAYER 2",
        default_shot_timer: int = settings.DEFAULT_SHOT_TIMER,
        target_score: int = 40,
    ) -> None:
        self.default_shot_timer: int = default_shot_timer
        self.target_score: int = target_score

        self.players: Dict[str, PlayerState] = {
            "P1": PlayerState(
                player_id="P1",
                name=p1_name,
                cue_ball_color="WHITE",
                score=0,
                innings=1,
                target_score=target_score,
            ),
            "P2": PlayerState(
                player_id="P2",
                name=p2_name,
                cue_ball_color="YELLOW",
                score=0,
                innings=0,
                target_score=target_score,
            ),
        }

        self.fsm_state: MatchState = MatchState.WAIT_SHOT
        self.active_player_id: str = "P1"
        self.current_inning: int = 1

        # 샷 타이머 관리 (초 단위)
        self.remaining_timer_sec: float = float(self.default_shot_timer)
        self.is_timer_paused: bool = False

        # 직전 샷 판정 요약 메시지 및 코드
        self.last_shot_result: Dict[str, object] = {
            "is_scored": False,
            "shot_type": "READY",
            "reason_code": "MATCH_START",
            "message": "경기 시작 대기 중 (P1 선공 - WHITE 수구)",
        }

    @property
    def active_player(self) -> PlayerState:
        """현재 공격권을 가진 플레이어 객체 반환"""
        return self.players[self.active_player_id]

    @property
    def active_cue_ball_color(self) -> str:
        """현재 공격 플레이어의 수구 색상("WHITE" | "YELLOW") 반환"""
        return self.active_player.cue_ball_color

    def reset_match(
        self, p1_name: Optional[str] = None, p2_name: Optional[str] = None
    ) -> Dict[str, object]:
        """경기 전체 기록 및 타이머를 초기 상태로 리셋합니다."""
        name1 = p1_name or self.players["P1"].name
        name2 = p2_name or self.players["P2"].name
        self.__init__(
            p1_name=name1,
            p2_name=name2,
            default_shot_timer=self.default_shot_timer,
            target_score=self.target_score,
        )
        return self.get_match_payload()

    def on_shot_started(self) -> Dict[str, object]:
        """
        [샷 감지 이벤트]
        수구가 움직이기 시작하면 FSM을 SHOT_IN_PROGRESS로 전환하고,
        진행 중이던 40초 샷 타이머를 즉시 일시 정지(Pause)합니다.
        """
        self.fsm_state = MatchState.SHOT_IN_PROGRESS
        self.is_timer_paused = True
        return self.get_match_payload()

    def tick_timer(self, elapsed_sec: float = 1.0) -> Dict[str, object]:
        """
        WAIT_SHOT 상태에서 타이머가 일시 정지되지 않았을 때 남은 시간을 차감합니다.
        """
        if self.fsm_state == MatchState.WAIT_SHOT and not self.is_timer_paused:
            self.remaining_timer_sec = max(
                0.0, round(self.remaining_timer_sec - elapsed_sec, 2)
            )
        return self.get_match_payload()

    def on_ball_stopped(
        self, shot_eval: Dict[str, object]
    ) -> Dict[str, object]:
        """
        [BALL_STOPPED 수신 및 판정 반영]
        공 3개가 완전히 정지(15프레임 확정)했을 때 ScoringEngine의 판정 결과를 받아 처리합니다.

        - 득점 성공(is_scored == True):
          현재 플레이어 점수 +1, 런/하이런 및 에버리지 갱신, 타이머 40초 리셋 후 재시작, 공격권 유지.
        - 공타 실패(is_scored == False):
          상대방으로 턴 자동 교체, 수구 색상 스위칭(WHITE <-> YELLOW), 이닝 수 +1, 타이머 40초 리셋 후 재시작.

        Returns:
            transition_summary: {
                "event_type": "SCORE_UPDATED" | "TURN_CHANGED",
                "previous_player": str,
                "active_player": str,
                "active_cue_ball_color": str,
                "is_scored": bool,
                "payload": Dict[str, object]
            }
        """
        # 1. BALL_STOPPED 상태 진입 기록
        self.fsm_state = MatchState.BALL_STOPPED
        is_scored = bool(shot_eval.get("is_scored", False))
        prev_player_id = self.active_player_id
        current_player = self.players[self.active_player_id]

        # 2. 플레이어 이닝 보장 (최소 1이닝 이상으로 에버리지 분모 보호)
        if current_player.innings <= 0:
            current_player.innings = 1

        if is_scored:
            # [득점 성공 처리]
            current_player.score += 1
            current_player.current_run += 1
            if current_player.current_run > current_player.high_run:
                current_player.high_run = current_player.current_run

            event_type = "SCORE_UPDATED"
        else:
            # [공타 실패 처리 -> 상대방으로 턴 교체 및 이닝 증가]
            current_player.current_run = 0
            next_player_id = "P2" if self.active_player_id == "P1" else "P1"
            self.active_player_id = next_player_id

            next_player = self.players[self.active_player_id]
            next_player.innings += 1
            next_player.current_run = 0

            self.current_inning = max(
                self.players["P1"].innings, self.players["P2"].innings
            )
            event_type = "TURN_CHANGED"

        # 3. 샷 타이머 40초 리셋 후 재시작 (WAIT_SHOT 상태로 복귀)
        self.remaining_timer_sec = float(self.default_shot_timer)
        self.is_timer_paused = False
        self.fsm_state = MatchState.WAIT_SHOT

        self.last_shot_result = {
            "is_scored": is_scored,
            "shot_type": shot_eval.get("shot_type", "MISS"),
            "reason_code": shot_eval.get("reason_code", ""),
            "message": shot_eval.get("message", ""),
            "cushions_before_second_hit": shot_eval.get(
                "cushions_before_second_hit", 0
            ),
            "total_cushions": shot_eval.get("total_cushions", 0),
        }

        payload = self.get_match_payload()
        return {
            "event_type": event_type,
            "previous_player": prev_player_id,
            "active_player": self.active_player_id,
            "active_cue_ball_color": self.active_cue_ball_color,
            "is_scored": is_scored,
            "payload": payload,
        }

    def get_match_payload(self) -> Dict[str, object]:
        """
        20인치 태블릿 스코어보드 전체 상태를 JSON 직렬화 가능한 딕셔너리로 반환합니다.
        """
        return {
            "fsm_state": self.fsm_state.value,
            "active_player_id": self.active_player_id,
            "active_cue_ball_color": self.active_cue_ball_color,
            "current_inning": self.current_inning,
            "shot_timer": {
                "remaining_sec": int(round(self.remaining_timer_sec)),
                "default_sec": self.default_shot_timer,
                "is_paused": self.is_timer_paused,
            },
            "players": {
                "P1": self.players["P1"].to_dict(),
                "P2": self.players["P2"].to_dict(),
            },
            "last_shot_result": self.last_shot_result,
        }
