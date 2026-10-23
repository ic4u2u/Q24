"""
대대 3쿠션 경기 규칙 FSM(유한 상태 머신) 및 스코어/타이머/이닝/PBA·큐스코 확장 관리자
파일 경로: cue24/services/match_manager.py

1. FSM 상태 관리: WAIT_SHOT (샷 대기 및 타이머 작동) -> SHOT_IN_PROGRESS (주행 중, 타이머 정지) -> BALL_STOPPED (판정 및 정산)
2. 플레이어 P1(WHITE), P2(YELLOW) 상태 관리: 점수, 이닝, 에버리지(점수/이닝), 현재 런(Run), 하이런(High Run), 수구 색상, 잔여 타임아웃
3. 샷 감지 시 40초 타이머 즉시 일시 정지(Pause)
4. BALL_STOPPED 수신 시 득점/공타 판정에 따른 점수 가산(일반룰 1점 / PBA룰 뱅크샷 2점), 턴 교체, 수구 색상 스위칭, 이닝 히스토리 기록
5. 큐스코/빌리보드 현장 기능: 타임아웃(+40초 연장), 수동 점수 가감(+1/-1), 수구 색상 스왑, 강제 턴 전환
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

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
    timeouts_remaining: int = 2  # 선수별 잔여 타임아웃 횟수 (기본 2회)
    bank_shots: int = 0     # 성공한 뱅크샷 횟수
    inning_scores: Dict[int, int] = field(default_factory=dict)  # 이닝 번호별 획득 점수 기록

    @property
    def average(self) -> float:
        """에버리지(Avg = 점수 / 이닝) 계산 (소수점 셋째 자리 반올림)"""
        if self.innings <= 0:
            return 0.000
        return round(self.score / float(self.innings), 3)

    @property
    def progress_pct(self) -> float:
        """목표 점수 대비 현재 달성률 (0.0 ~ 100.0%)"""
        if self.target_score <= 0:
            return 0.0
        return round(min(100.0, (self.score / float(self.target_score)) * 100.0), 1)

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
            "progress_pct": self.progress_pct,
            "timeouts_remaining": self.timeouts_remaining,
            "bank_shots": self.bank_shots,
        }


class MatchManager:
    """
    3쿠션 경기의 FSM 상태, 턴(공격권), 점수, 이닝 기록지(Inning Sheet),
    PBA/일반 룰 모드, 타임아웃, 40초 샷 타이머를 통합 관리하는 핵심 서비스 클래스입니다.
    """

    def __init__(
        self,
        p1_name: str = "PLAYER 1",
        p2_name: str = "PLAYER 2",
        default_shot_timer: int = settings.DEFAULT_SHOT_TIMER,
        target_score: int = 40,
        rule_mode: str = "STANDARD",
    ) -> None:
        self.default_shot_timer: int = default_shot_timer
        self.target_score: int = target_score
        # rule_mode: "STANDARD" (모든 득점 1점) | "PBA" (뱅크샷 2점, 일반 3C 1점)
        self.rule_mode: str = rule_mode.upper() if rule_mode.upper() in ("STANDARD", "PBA") else "STANDARD"

        self.players: Dict[str, PlayerState] = {
            "P1": PlayerState(
                player_id="P1",
                name=p1_name,
                cue_ball_color="WHITE",
                score=0,
                innings=1,
                target_score=target_score,
                inning_scores={1: 0},
            ),
            "P2": PlayerState(
                player_id="P2",
                name=p2_name,
                cue_ball_color="YELLOW",
                score=0,
                innings=0,
                target_score=target_score,
                inning_scores={},
            ),
        }

        self.fsm_state: MatchState = MatchState.WAIT_SHOT
        self.active_player_id: str = "P1"
        self.current_inning: int = 1

        # 샷 타이머 관리 (초 단위) 및 경기 시작부터 누적 경과 타이머 (초 단위)
        self.remaining_timer_sec: float = float(self.default_shot_timer)
        self.is_timer_paused: bool = False
        self.is_match_paused: bool = False
        self.match_elapsed_sec: float = 0.0

        # 직전 완료 샷 기록 (수구 색상 감지 기반 미인지 득점 자동 소급 보정용)
        self._last_shot_player_id: Optional[str] = None
        self._last_shot_inning: int = 1
        self._last_shot_was_miss: bool = False
        self._last_shot_prev_run: int = 0
        self._pre_shot_player_id: str = "P1"
        self._pre_shot_inning: int = 1
        self._pre_shot_p2_had_slot: bool = False
        self.shot_seq_id: int = 0
        self.last_scored_shot_id: int = -1
        self.score_dispatched_for_current_shot: bool = False
        self.bank_shot_dispatched_for_current_shot: bool = False

        # 직전 샷 판정 요약 메시지 및 코드
        self.last_shot_result: Dict[str, object] = {
            "is_scored": False,
            "points_awarded": 0,
            "shot_type": "READY",
            "reason_code": "MATCH_START",
            "message": "경기 시작 대기 중 (P1 선공 - WHITE 수구)",
        }

    @property
    def formatted_match_time(self) -> str:
        """경기 시작부터 누적된 경과 시간을 HH:MM:SS 포맷 문자열로 반환합니다."""
        total_sec = max(0, int(round(self.match_elapsed_sec)))
        hours = total_sec // 3600
        minutes = (total_sec % 3600) // 60
        seconds = total_sec % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @property
    def active_player(self) -> PlayerState:
        """현재 공격권을 가진 플레이어 객체 반환"""
        return self.players[self.active_player_id]

    @property
    def active_cue_ball_color(self) -> str:
        """현재 공격 플레이어의 수구 색상("WHITE" | "YELLOW") 반환"""
        return self.active_player.cue_ball_color

    def reset_match(
        self,
        p1_name: Optional[str] = None,
        p2_name: Optional[str] = None,
        rule_mode: Optional[str] = None,
        full_factory_reset: bool = False,
    ) -> Dict[str, object]:
        """경기 전체 기록 및 타이머를 초기 상태로 리셋합니다."""
        if full_factory_reset:
            name1 = "PLAYER 1"
            name2 = "PLAYER 2"
            mode = "STANDARD"
            p1_handicap = self.target_score
            p2_handicap = self.target_score
        else:
            name1 = p1_name or self.players["P1"].name
            name2 = p2_name or self.players["P2"].name
            mode = rule_mode or self.rule_mode
            p1_handicap = self.players["P1"].target_score
            p2_handicap = self.players["P2"].target_score

        self.__init__(
            p1_name=name1,
            p2_name=name2,
            default_shot_timer=self.default_shot_timer,
            target_score=self.target_score,
            rule_mode=mode,
        )
        self.players["P1"].target_score = p1_handicap
        self.players["P2"].target_score = p2_handicap
        return self.get_match_payload()

    def set_rule_mode(self, rule_mode: str) -> Dict[str, object]:
        """경기 룰 모드를 STANDARD(일반 1점) 또는 PBA(뱅크샷 2점)로 전환합니다."""
        upper_mode = rule_mode.upper()
        if upper_mode not in ("STANDARD", "PBA"):
            raise ValueError(f"지원하지 않는 rule_mode입니다: {rule_mode}")
        self.rule_mode = upper_mode
        return self.get_match_payload()

    def use_timeout(self, player_id: Optional[str] = None) -> Dict[str, object]:
        """
        현재 공격 플레이어(또는 지정된 플레이어)의 타임아웃을 1회 차감하고
        샷 타이머에 +40초(default_shot_timer)를 즉시 연장합니다.
        """
        pid = player_id or self.active_player_id
        player = self.players[pid]
        if player.timeouts_remaining > 0:
            player.timeouts_remaining -= 1
            self.remaining_timer_sec += float(self.default_shot_timer)
        return self.get_match_payload()

    def adjust_score(self, player_id: str, delta: int) -> Dict[str, object]:
        """
        현장 수동 점수 보정(+1 / -1): 플레이어 점수와 현재 이닝 득점을 안전하게 보정합니다.
        """
        if player_id not in self.players:
            raise ValueError(f"존재하지 않는 플레이어 ID입니다: {player_id}")
        player = self.players[player_id]
        new_score = max(0, player.score + delta)
        actual_delta = new_score - player.score
        player.score = new_score

        inning_idx = max(1, player.innings)
        prev_inning_pts = player.inning_scores.get(inning_idx, 0)
        player.inning_scores[inning_idx] = max(0, prev_inning_pts + actual_delta)
        return self.get_match_payload()

    def set_player_name(
        self, player_id: str, new_name: str, handicap: Optional[int] = None
    ) -> Dict[str, object]:
        """
        1번 선수(P1) 또는 2번 선수(P2)의 이름(및 핸디)을 변경합니다.
        """
        pid = player_id.upper()
        if pid not in self.players:
            raise ValueError(f"존재하지 않는 플레이어 ID입니다: {player_id}")
        cleaned = new_name.strip()
        if not cleaned:
            cleaned = "1번 선수" if pid == "P1" else "2번 선수"
        self.players[pid].name = cleaned
        if handicap is not None and int(handicap) > 0:
            self.players[pid].target_score = int(handicap)
        self.last_shot_result = {
            "is_scored": False,
            "points_awarded": 0,
            "shot_type": "READY",
            "reason_code": "PLAYER_NAME_CHANGED",
            "message": f"선수 정보가 '{cleaned}' (핸디 {self.players[pid].target_score})로 변경되었습니다.",
        }
        return self.get_match_payload()

    def swap_player_names_only(self) -> Dict[str, object]:
        """
        [선수변경 버튼 동작]
        좌·우측에 배치된 당구공 색상(흰공 / 노란공) 및 현재 위치의 점수/기록은 그대로 두고,
        좌우 선수의 이름(및 핸디)만 서로 좌우로 맞바꿉니다.
        """
        p1, p2 = self.players["P1"], self.players["P2"]
        n1 = "1번 선수" if p1.name.strip().upper() in ("PLAYER 1", "P1") else p1.name
        n2 = "2번 선수" if p2.name.strip().upper() in ("PLAYER 2", "P2") else p2.name
        p1.name, p2.name = n2, n1
        p1.target_score, p2.target_score = p2.target_score, p1.target_score

        self.last_shot_result = {
            "is_scored": False,
            "points_awarded": 0,
            "shot_type": "READY",
            "reason_code": "PLAYER_NAMES_SWAPPED",
            "message": f"선수변경 완료: 좌측({p1.name}) ↔ 우측({p2.name}) (당구공 색상 위치 유지)",
        }
        return self.get_match_payload()

    def swap_seats_with_balls(self) -> Dict[str, object]:
        """
        [자리바꿈 버튼 동작]
        좌우 선수의 이름(및 핸디)과 당구공 색깔(흰공/노란공 및 해당 진영의 점수·기록 상태)이
        함께 통째로 좌우 위치가 바뀌도록 스왑합니다.
        """
        p1, p2 = self.players["P1"], self.players["P2"]
        n1 = "1번 선수" if p1.name.strip().upper() in ("PLAYER 1", "P1") else p1.name
        n2 = "2번 선수" if p2.name.strip().upper() in ("PLAYER 2", "P2") else p2.name
        p1.name, p2.name = n2, n1
        p1.target_score, p2.target_score = p2.target_score, p1.target_score
        p1.cue_ball_color, p2.cue_ball_color = p2.cue_ball_color, p1.cue_ball_color
        p1.score, p2.score = p2.score, p1.score
        p1.innings, p2.innings = p2.innings, p1.innings
        p1.current_run, p2.current_run = p2.current_run, p1.current_run
        p1.high_run, p2.high_run = p2.high_run, p1.high_run
        p1.timeouts_remaining, p2.timeouts_remaining = (
            p2.timeouts_remaining,
            p1.timeouts_remaining,
        )
        p1.bank_shots, p2.bank_shots = p2.bank_shots, p1.bank_shots
        p1.inning_scores, p2.inning_scores = (
            dict(p2.inning_scores),
            dict(p1.inning_scores),
        )
        self.active_player_id = "P2" if self.active_player_id == "P1" else "P1"
        self._last_shot_was_miss = False

        c1_ko = "흰공" if p1.cue_ball_color == "WHITE" else "노란공"
        c2_ko = "흰공" if p2.cue_ball_color == "WHITE" else "노란공"
        self.last_shot_result = {
            "is_scored": False,
            "points_awarded": 0,
            "shot_type": "READY",
            "reason_code": "SEATS_AND_BALLS_SWAPPED",
            "message": f"자리바꿈 완료: 좌측({p1.name}·{c1_ko}) ↔ 우측({p2.name}·{c2_ko})",
        }
        return self.get_match_payload()

    def toggle_pause_match(self) -> Dict[str, object]:
        """
        선수가 [일시정지] / [다시시작] 버튼을 클릭했을 때
        경기시간(match_elapsed_sec)과 40초 샷 타이머의 일시정지 / 다시시작 상태를 토글합니다.
        선수가 직접 [다시시작]을 클릭하기 전까지는 공이 움직여도 일시정지 상태가 유지됩니다.
        """
        self.is_match_paused = not self.is_match_paused
        self.is_timer_paused = self.is_match_paused or (
            self.fsm_state == MatchState.SHOT_IN_PROGRESS
        )
        state_txt = "일시정지됨 (다시시작 클릭 시 시간 재개)" if self.is_match_paused else "다시시작됨"
        self.last_shot_result = {
            "is_scored": False,
            "points_awarded": 0,
            "shot_type": "READY",
            "reason_code": "MATCH_PAUSED" if self.is_match_paused else "MATCH_RESUMED",
            "message": f"경기가 {state_txt} 상태입니다.",
        }
        return self.get_match_payload()

    def end_match(self) -> Dict[str, object]:
        """현재 경기를 종료 처리하고 경기시간 및 샷 타이머를 정지합니다."""
        self.is_match_paused = True
        self.is_timer_paused = True
        p1, p2 = self.players["P1"], self.players["P2"]
        if p1.score > p2.score:
            winner_msg = f"{p1.name} 승리! ({p1.score} : {p2.score})"
        elif p2.score > p1.score:
            winner_msg = f"{p2.name} 승리! ({p2.score} : {p1.score})"
        else:
            winner_msg = f"무승부 ({p1.score} : {p2.score})"
        self.last_shot_result = {
            "is_scored": False,
            "points_awarded": 0,
            "shot_type": "READY",
            "reason_code": "MATCH_ENDED",
            "message": f"[경기 종료] {winner_msg}",
        }
        return self.get_match_payload()

    def swap_player_seats(self) -> Dict[str, object]:
        """
        1번 선수(좌측 선공 자리)와 2번 선수(우측 후공 자리)의 자리를 맞바꿉니다.
        - 두 선수의 이름 및 경기 스탯(점수, 하이런, 뱅크샷, 이닝 기록 등)을 서로 교체하되,
        - 좌측 1번 자리(P1)는 무조건 흰공(WHITE), 우측 2번 자리(P2)는 무조건 노란공(YELLOW)으로 고정합니다!
        """
        p1, p2 = self.players["P1"], self.players["P2"]

        # 기본 영문 플레이스홀더일 경우 자연스러운 한글 기본명으로 변환 후 스왑
        n1 = "1번 선수" if p1.name.strip().upper() in ("PLAYER 1", "P1") else p1.name
        n2 = "2번 선수" if p2.name.strip().upper() in ("PLAYER 2", "P2") else p2.name
        p1.name, p2.name = n2, n1

        # 만약 경기가 이미 진행 중(득점이 있거나 2이닝 이상)이면 점수/스탯도 함께 자리 이동
        is_initial_state = (
            p1.score == 0
            and p2.score == 0
            and self.current_inning == 1
            and p2.innings == 0
        )
        if not is_initial_state:
            p1.score, p2.score = p2.score, p1.score
            p1.innings, p2.innings = p2.innings, p1.innings
            p1.current_run, p2.current_run = p2.current_run, p1.current_run
            p1.high_run, p2.high_run = p2.high_run, p1.high_run
            p1.timeouts_remaining, p2.timeouts_remaining = (
                p2.timeouts_remaining,
                p1.timeouts_remaining,
            )
            p1.bank_shots, p2.bank_shots = p2.bank_shots, p1.bank_shots
            p1.inning_scores, p2.inning_scores = (
                dict(p2.inning_scores),
                dict(p1.inning_scores),
            )
            self.active_player_id = "P2" if self.active_player_id == "P1" else "P1"
        else:
            # 경기 시작 직전(초기 상태)에 자리바꿈을 한 경우: 새로 1번 자리에 온 선수가 1이닝 선공(흰공)으로 시작
            p1.innings = 1
            p1.inning_scores = {1: 0}
            p2.innings = 0
            p2.inning_scores = {}
            self.active_player_id = "P1"

        # [불변 규칙 강제] 1번 자리(선공) = 무조건 흰공(WHITE), 2번 자리(후공) = 무조건 노란공(YELLOW)
        p1.cue_ball_color = "WHITE"
        p2.cue_ball_color = "YELLOW"
        self._last_shot_was_miss = False

        self.last_shot_result = {
            "is_scored": False,
            "points_awarded": 0,
            "shot_type": "READY",
            "reason_code": "PLAYERS_SWAPPED",
            "message": f"선수 자리바꿈 완료: 선공({p1.name}·흰공) vs 후공({p2.name}·노란공)",
        }
        return self.get_match_payload()

    def swap_cue_balls(self) -> Dict[str, object]:
        """P1과 P2의 수구 색상(WHITE <-> YELLOW)을 맞바꿉니다."""
        p1, p2 = self.players["P1"], self.players["P2"]
        p1.cue_ball_color, p2.cue_ball_color = p2.cue_ball_color, p1.cue_ball_color
        return self.get_match_payload()

    def sync_active_player_to_current_inning(self, target_player_id: str) -> None:
        """
        현재 통합 이닝(self.current_inning)을 기준으로 공격권이 부여된 선수의 이닝 번호와
        이닝 기록표(inning_scores) 슬롯을 어긋남 없이 동기화합니다.
        - P1(선공) 공격 시: current_inning의 P1 슬롯 보장
        - P2(후공) 공격 시: 같은 current_inning에서 P1 슬롯과 P2 슬롯이 모두 존재하도록 보장
        """
        self.current_inning = max(1, self.current_inning)
        p1 = self.players["P1"]
        p2 = self.players["P2"]

        if target_player_id == "P1":
            self.active_player_id = "P1"
            p1.innings = self.current_inning
            if self.current_inning not in p1.inning_scores:
                p1.inning_scores[self.current_inning] = 0
        else:
            self.active_player_id = "P2"
            p1.innings = max(p1.innings, self.current_inning)
            if self.current_inning not in p1.inning_scores:
                p1.inning_scores[self.current_inning] = 0
            p2.innings = self.current_inning
            if self.current_inning not in p2.inning_scores:
                p2.inning_scores[self.current_inning] = 0

    def reconcile_cue_ball_stroke(
        self,
        stroked_cue_color: str,
        compensate_missed_score: bool = True,
        force_sync_player: bool = True,
    ) -> Dict[str, object]:
        """
        [수구 색상 감지 기반 미인지 득점 자동 보정 및 공격자 100% 동기화]
        선수가 실제로 치는 공(수구)의 색깔("WHITE"=1번 선공 P1, "YELLOW"=2번 후공 P2)을 감지하여:
        1) 직전 샷에서 비전이 미세 득점을 놓쳐 공타(MISS)로 판정하고 턴을 넘겼으나,
           실제로는 득점이어서 동일한 선수가 자신의 수구("WHITE"=P1, "YELLOW"=P2)를 계속 친 경우:
           - 누락되었던 직전 이닝 득점(+1점)과 연속 득점(Run)을 즉시 소급 복구하고,
           - 잘못 넘어갔던 이닝/턴을 해당 선수의 원래 이닝(공격 중)으로 완벽히 되돌립니다.
        2) 흰공 타구 = 1번 선수(P1), 노란공 타구 = 2번 선수(P2) 100% 일치:
           - 선수가 치는 공의 색깔에 따라 해당 선수의 점수판과 이닝 기록표에 정확히 점수가 올라가도록
             공격권(active_player_id)과 이닝 슬롯을 즉시 동기화합니다.
        """
        self.players["P1"].cue_ball_color = "WHITE"
        self.players["P2"].cue_ball_color = "YELLOW"

        color_upper = str(stroked_cue_color or "").strip().upper()
        if color_upper not in ("WHITE", "YELLOW"):
            return {
                "compensated": False,
                "points_compensated": 0,
                "active_player_id": self.active_player_id,
                "stroked_cue_color": color_upper,
                "payload": self.get_match_payload(),
            }

        stroked_player_id = "P1" if color_upper == "WHITE" else "P2"

        # Case 1: 직전 샷이 공타(MISS)로 판정되어 상대방에게 턴이 넘어갔는데,
        # 직전 샷을 쳤던 선수가 자기 공(흰공 또는 노란공)을 계속 치는 순간 -> 누락된 +1점과 공격권 즉시 복구!
        if (
            compensate_missed_score
            and self._last_shot_was_miss
            and self._last_shot_player_id == stroked_player_id
            and self.active_player_id != stroked_player_id
        ):
            target_inn = max(1, self._last_shot_inning)
            if stroked_player_id == "P1":
                # P1 공타 판정으로 인해 같은 이닝(target_inn)의 P2 슬롯이 임시 0점으로 열렸던 상태 정리
                p2 = self.players["P2"]
                if p2.inning_scores.get(target_inn, 0) == 0 and p2.innings == target_inn:
                    p2.inning_scores.pop(target_inn, None)
                    p2.innings = max(0, target_inn - 1)
            else:
                # P2 공타 판정으로 인해 다음 이닝(target_inn + 1)의 P1 슬롯이 임시 0점으로 열렸던 상태 정리
                p1 = self.players["P1"]
                next_inn = target_inn + 1
                if p1.inning_scores.get(next_inn, 0) == 0 and p1.innings == next_inn:
                    p1.inning_scores.pop(next_inn, None)
                    p1.innings = target_inn

            self.current_inning = target_inn
            self.active_player_id = stroked_player_id
            player = self.players[stroked_player_id]
            player.innings = target_inn
            player.score += 1
            player.inning_scores[target_inn] = (
                player.inning_scores.get(target_inn, 0) + 1
            )
            player.current_run = max(1, self._last_shot_prev_run + 1)
            if player.current_run > player.high_run:
                player.high_run = player.current_run

            self._last_shot_was_miss = False
            cue_ko = "흰공" if color_upper == "WHITE" else "노란공"
            self.last_shot_result = {
                "is_scored": True,
                "points_awarded": 1,
                "shot_type": "RETROACTIVE_3C",
                "reason_code": "RETROACTIVE_SCORE_COMPENSATED",
                "message": (
                    f"수구({cue_ko}) 연속 타구 감지: 직전 미인지 득점(+1점) 자동 보정 및 "
                    f"{player.name} 공격 유지"
                ),
            }
            return {
                "compensated": True,
                "points_compensated": 1,
                "active_player_id": self.active_player_id,
                "stroked_cue_color": color_upper,
                "payload": self.get_match_payload(),
            }

        # Case 2: 흰공 타구 = 1번 선수(P1), 노란공 타구 = 2번 선수(P2) 100% 일치 동기화
        if force_sync_player and self.active_player_id != stroked_player_id:
            if self.active_player_id == "P2" and stroked_player_id == "P1":
                # 후공(P2)이 이미 현재 이닝에 타구한 기록이 있거나 현재 이닝의 P2 슬롯이 열려 있었다면 다음 이닝(current_inning + 1)의 P1 차례로 진행
                p2 = self.players["P2"]
                if (
                    (
                        self._last_shot_player_id == "P2"
                        and self._last_shot_inning == self.current_inning
                    )
                    or p2.innings >= self.current_inning
                    or self.current_inning in p2.inning_scores
                ):
                    self.current_inning += 1
            self.players[self.active_player_id].current_run = 0
            self._last_shot_was_miss = False
            self.sync_active_player_to_current_inning(stroked_player_id)
        else:
            if stroked_player_id == "P1":
                p2 = self.players["P2"]
                if (
                    self._last_shot_player_id == "P2"
                    and self._last_shot_inning == self.current_inning
                ) or (
                    p2.innings >= self.current_inning
                    and self.current_inning in p2.inning_scores
                ):
                    self.current_inning += 1
            self.sync_active_player_to_current_inning(self.active_player_id)

        return {
            "compensated": False,
            "points_compensated": 0,
            "active_player_id": self.active_player_id,
            "stroked_cue_color": color_upper,
            "payload": self.get_match_payload(),
        }

    def cancel_phantom_shot(self) -> Dict[str, object]:
        """
        카메라 조명 반사나 선수 손/큐대 가림으로 인해 실제 타구가 아닌 미세 흔들림이 감지되었을 때,
        공격권(active_player_id)이나 이닝 번호를 절대 변경하지 않고 기존 공격 선수의 '공격 중' 상태를 그대로 복원합니다.
        """
        self.fsm_state = MatchState.WAIT_SHOT
        self.is_timer_paused = self.is_match_paused
        self.players["P1"].cue_ball_color = "WHITE"
        self.players["P2"].cue_ball_color = "YELLOW"
        restored_pid = self._pre_shot_player_id or self.active_player_id
        restored_inn = max(1, self._pre_shot_inning or self.current_inning)
        if restored_pid == "P1" and not self._pre_shot_p2_had_slot:
            p2 = self.players["P2"]
            if p2.inning_scores.get(restored_inn, 0) == 0 and p2.innings == restored_inn:
                p2.inning_scores.pop(restored_inn, None)
                p2.innings = max(0, restored_inn - 1)
        self.current_inning = restored_inn
        self.active_player_id = restored_pid
        self.sync_active_player_to_current_inning(self.active_player_id)
        return self.get_match_payload()

    def force_switch_turn(self) -> Dict[str, object]:
        """
        수동으로 공격 턴을 상대방에게 넘기고 타이머를 40초로 리셋합니다.
        - 선공(P1)에서 후공(P2)으로 넘어갈 때는 같은 이닝(self.current_inning 유지)의 후공 칸으로 이동합니다.
        - 후공(P2)에서 선공(P1)으로 넘어갈 때는 다음 이닝(self.current_inning + 1)으로 이동합니다.
        """
        self.current_inning = max(1, self.current_inning)
        prev_player_id = self.active_player_id
        current_player = self.players[prev_player_id]
        current_player.innings = self.current_inning
        if self.current_inning not in current_player.inning_scores:
            current_player.inning_scores[self.current_inning] = 0
        current_player.current_run = 0
        self._last_shot_was_miss = False

        if prev_player_id == "P1":
            # 선공(P1) 턴 종료 -> 같은 이닝(current_inning)의 후공(P2) 차례
            self.active_player_id = "P2"
            p2 = self.players["P2"]
            p2.innings = self.current_inning
            p2.current_run = 0
            if self.current_inning not in p2.inning_scores:
                p2.inning_scores[self.current_inning] = 0
        else:
            # 후공(P2) 턴 종료 -> 다음 이닝(current_inning + 1)의 선공(P1) 차례
            self.current_inning += 1
            self.active_player_id = "P1"
            p1 = self.players["P1"]
            p1.innings = self.current_inning
            p1.current_run = 0
            if self.current_inning not in p1.inning_scores:
                p1.inning_scores[self.current_inning] = 0

        self.remaining_timer_sec = float(self.default_shot_timer)
        self.is_timer_paused = self.is_match_paused
        self.fsm_state = MatchState.WAIT_SHOT
        return self.get_match_payload()

    def on_shot_started(self) -> Dict[str, object]:
        """
        [샷 감지 이벤트]
        수구가 움직이기 시작하면 FSM을 SHOT_IN_PROGRESS로 전환하고,
        진행 중이던 40초 샷 타이머를 즉시 일시 정지(Pause)합니다.
        """
        self.shot_seq_id += 1
        self.sync_active_player_to_current_inning(self.active_player_id)
        self.fsm_state = MatchState.SHOT_IN_PROGRESS
        self.is_timer_paused = True
        self.score_dispatched_for_current_shot = False
        self.bank_shot_dispatched_for_current_shot = False
        return self.get_match_payload()

    def on_immediate_second_ball_hit(
        self, shot_eval: Dict[str, object]
    ) -> Dict[str, object]:
        """
        [3쿠션 이상 충족 후 제2적구 충돌 즉시 실시간 득점(+1) 반영]
        공이 완전히 멈추기를 기다리지 않고, 제2적구와 충돌하는 해당 프레임에 즉시:
        - 공격 선수의 메인 점수 +1 (PBA 뱅크샷은 +2)
        - 하이런(연속득점) 갱신
        - 이닝별득점 테이블의 현재 이닝 셀 +1
        - 메인 점수 아래 그라데이션 타이머 바를 40초(100%, 에메랄드 그린)로 즉시 리셋
        을 수행하고 중복 가산 방지 플래그(score_dispatched_for_current_shot = True)를 설정합니다.
        """
        if (
            self.score_dispatched_for_current_shot
            or self.last_scored_shot_id == self.shot_seq_id
        ):
            payload = self.get_match_payload()
            return {
                "event_type": "SCORE_UPDATED",
                "previous_player": self.active_player_id,
                "active_player": self.active_player_id,
                "active_cue_ball_color": self.active_cue_ball_color,
                "is_scored": True,
                "already_dispatched": True,
                "shot_id": self.shot_seq_id,
                "points_awarded": 0,
                "payload": payload,
            }

        self.score_dispatched_for_current_shot = True
        self.last_scored_shot_id = self.shot_seq_id
        shot_type = str(shot_eval.get("shot_type", "REGULAR_3C"))
        if shot_type not in ("REGULAR_3C", "BANK_SHOT"):
            shot_type = "REGULAR_3C"

        prev_player_id = self.active_player_id
        prev_inning = max(1, self.current_inning)
        self.current_inning = prev_inning
        current_player = self.players[prev_player_id]
        current_player.innings = self.current_inning
        if self.current_inning not in current_player.inning_scores:
            current_player.inning_scores[self.current_inning] = 0

        if self.rule_mode == "PBA" and shot_type == "BANK_SHOT":
            points_awarded = 2
        else:
            points_awarded = 1

        if shot_type == "BANK_SHOT" and not self.bank_shot_dispatched_for_current_shot:
            current_player.bank_shots += 1
            self.bank_shot_dispatched_for_current_shot = True

        current_player.score += points_awarded
        current_player.current_run += points_awarded
        current_player.inning_scores[self.current_inning] = (
            current_player.inning_scores.get(self.current_inning, 0) + points_awarded
        )
        if current_player.current_run > current_player.high_run:
            current_player.high_run = current_player.current_run

        # 제2적구 타격 즉시 타이머 바를 40초(100%, 에메랄드 그린)로 리셋 (단, 선수 수동 일시정지 중이면 정지 유지)
        self.remaining_timer_sec = float(self.default_shot_timer)
        self.is_timer_paused = self.is_match_paused

        self.last_shot_result = {
            "shot_id": self.shot_seq_id,
            "is_scored": True,
            "immediate_score": True,
            "already_dispatched": False,
            "points_awarded": points_awarded,
            "shot_type": shot_type,
            "reason_code": shot_eval.get("reason_code", "IMMEDIATE_SECOND_BALL_HIT"),
            "message": shot_eval.get(
                "message",
                f"⚡ 제2적구 충돌 즉시 득점! (+{points_awarded}점 실시간 반영)",
            ),
            "cushions_before_first_hit": shot_eval.get(
                "cushions_before_first_hit", 0
            ),
            "cushions_before_second_hit": shot_eval.get(
                "cushions_before_second_hit", 3
            ),
            "total_cushions": shot_eval.get("total_cushions", 3),
            "first_ball_hit": bool(shot_eval.get("first_ball_hit", True)),
            "second_ball_hit": bool(shot_eval.get("second_ball_hit", True)),
        }

        payload = self.get_match_payload()
        return {
            "event_type": "SCORE_UPDATED",
            "previous_player": prev_player_id,
            "active_player": self.active_player_id,
            "active_cue_ball_color": self.active_cue_ball_color,
            "is_scored": True,
            "immediate_score": True,
            "already_dispatched": False,
            "shot_id": self.shot_seq_id,
            "points_awarded": points_awarded,
            "payload": payload,
        }

    def tick_match_elapsed(self, elapsed_sec: float = 1.0) -> None:
        """
        경기 시작부터 누적된 총 경기 시간을 증가시킵니다.
        선수가 [일시정지] 버튼을 클릭한 상태(is_match_paused == True)에서는
        [다시시작]을 클릭하기 전까지 경기시간이 멈춥니다.
        """
        if self.is_match_paused:
            return
        self.match_elapsed_sec = round(
            self.match_elapsed_sec + max(0.0, elapsed_sec), 2
        )

    def tick_timer(self, elapsed_sec: float = 1.0) -> Dict[str, object]:
        """
        선수가 [일시정지]를 누르지 않은 상태(not is_match_paused)에서만 경기 누적 경과 시간을 갱신하고,
        WAIT_SHOT 상태에서 타이머가 일시 정지되지 않았을 때 남은 샷 시간을 차감합니다.
        """
        if not self.is_match_paused:
            self.tick_match_elapsed(elapsed_sec)
        if (
            self.fsm_state == MatchState.WAIT_SHOT
            and not self.is_timer_paused
            and not self.is_match_paused
        ):
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
          일반 룰("STANDARD")은 +1점, PBA 룰("PBA")에서 BANK_SHOT일 때는 +2점 가산,
          현재 이닝(self.current_inning)의 본인 칸에 점수 누적, 타이머 40초 리셋 후 재시작,
          공격권(active_player_id) 및 '공격 중' 표시를 절대 변경하지 않고 그대로 유지!
        - 공타 실패(is_scored == False):
          * 선공(P1)이 실패한 경우: 같은 이닝(self.current_inning 유지)의 후공(P2)으로 턴 전환!
            (P2의 같은 이닝 칸이 0점으로 생성되어 P1과 동일한 이닝 열에 기록됨)
          * 후공(P2)이 실패한 경우: 해당 이닝이 종료되므로 다음 이닝(self.current_inning + 1)으로 증가하며 선공(P1)으로 턴 전환!
        """
        self.fsm_state = MatchState.BALL_STOPPED
        already_dispatched = (
            self.score_dispatched_for_current_shot
            or (self.shot_seq_id > 0 and self.last_scored_shot_id == self.shot_seq_id)
            or bool(shot_eval.get("already_dispatched", False))
        )
        was_bank_dispatched = self.bank_shot_dispatched_for_current_shot
        # [불변 원칙] 제2적구 충돌 즉시 한 번 올라간 점수(+1)는 절대 롤백(차감)하지 않으며,
        # 공이 멈췄을 때도 해당 선수의 득점 인정과 공격권을 100% 그대로 유지합니다!
        is_scored = True if already_dispatched else bool(shot_eval.get("is_scored", False))
        self.score_dispatched_for_current_shot = False
        self.bank_shot_dispatched_for_current_shot = False
        shot_type = str(shot_eval.get("shot_type", "REGULAR_3C" if is_scored else "MISS"))
        if is_scored and shot_type not in ("REGULAR_3C", "BANK_SHOT"):
            shot_type = "REGULAR_3C"
        if is_scored and was_bank_dispatched:
            shot_type = "BANK_SHOT"
        prev_player_id = self.active_player_id
        prev_inning = max(1, self.current_inning)

        self.current_inning = prev_inning
        current_player = self.players[prev_player_id]
        prev_run_before_shot = current_player.current_run
        current_player.innings = self.current_inning
        if self.current_inning not in current_player.inning_scores:
            current_player.inning_scores[self.current_inning] = 0

        points_awarded = 0
        if is_scored:
            self.last_scored_shot_id = self.shot_seq_id
            if already_dispatched:
                # 이미 제2적구 충돌 순간에 즉시 득점(+1)이 반영되었으므로 중복 가산 및 롤백을 모두 차단하고 턴을 유지!
                # 단, 즉시 득점 시점에는 일반 3C였다가 정지 시점에 뱅크샷으로 최종 확인된 경우 뱅크샷 카운트를 보정합니다.
                if shot_type == "BANK_SHOT" and not was_bank_dispatched:
                    current_player.bank_shots += 1
                    if self.rule_mode == "PBA":
                        current_player.score += 1
                        current_player.current_run += 1
                        current_player.inning_scores[self.current_inning] = (
                            current_player.inning_scores.get(self.current_inning, 0) + 1
                        )
                        if current_player.current_run > current_player.high_run:
                            current_player.high_run = current_player.current_run
                points_awarded = 0
                self.active_player_id = prev_player_id
                event_type = "SCORE_UPDATED"
            else:
                # PBA 모드이고 뱅크샷인 경우 2점, 그 외에는 1점 부여
                if self.rule_mode == "PBA" and shot_type == "BANK_SHOT":
                    points_awarded = 2
                else:
                    points_awarded = 1

                if shot_type == "BANK_SHOT":
                    current_player.bank_shots += 1

                current_player.score += points_awarded
                current_player.current_run += points_awarded
                current_player.inning_scores[self.current_inning] = (
                    current_player.inning_scores.get(self.current_inning, 0)
                    + points_awarded
                )
                if current_player.current_run > current_player.high_run:
                    current_player.high_run = current_player.current_run

                # 득점 성공 시 반드시 현재 공격 선수의 턴과 '공격 중' 상태를 유지!
                self.active_player_id = prev_player_id
                event_type = "SCORE_UPDATED"
        else:
            current_player.current_run = 0
            if prev_player_id == "P1":
                # 선공(P1) 득점 실패 -> 같은 이닝(self.current_inning)의 후공(P2) 차례로 턴 교체
                self.active_player_id = "P2"
                p2 = self.players["P2"]
                p2.innings = self.current_inning
                p2.current_run = 0
                if self.current_inning not in p2.inning_scores:
                    p2.inning_scores[self.current_inning] = 0
            else:
                # 후공(P2) 득점 실패 -> 다음 이닝(self.current_inning + 1)의 선공(P1) 차례로 턴 교체
                self.current_inning += 1
                self.active_player_id = "P1"
                p1 = self.players["P1"]
                p1.innings = self.current_inning
                p1.current_run = 0
                if self.current_inning not in p1.inning_scores:
                    p1.inning_scores[self.current_inning] = 0

            event_type = "TURN_CHANGED"

        # 직전 완료 샷 정보 기록 (다음 샷에서 수구 색상 감지 시 미인지 득점 소급 보정용)
        self._last_shot_player_id = prev_player_id
        self._last_shot_inning = prev_inning
        self._last_shot_was_miss = not is_scored
        self._last_shot_prev_run = prev_run_before_shot

        self.remaining_timer_sec = float(self.default_shot_timer)
        self.is_timer_paused = self.is_match_paused
        self.fsm_state = MatchState.WAIT_SHOT

        self.last_shot_result = {
            "shot_id": self.shot_seq_id,
            "is_scored": is_scored,
            "already_dispatched": already_dispatched,
            "points_awarded": points_awarded,
            "shot_type": shot_type,
            "reason_code": shot_eval.get("reason_code", ""),
            "message": shot_eval.get("message", ""),
            "cushions_before_first_hit": shot_eval.get(
                "cushions_before_first_hit", 0
            ),
            "cushions_before_second_hit": shot_eval.get(
                "cushions_before_second_hit", 0
            ),
            "total_cushions": shot_eval.get("total_cushions", 0),
            "first_ball_hit": bool(shot_eval.get("first_ball_hit", is_scored)),
            "second_ball_hit": bool(shot_eval.get("second_ball_hit", is_scored)),
        }

        payload = self.get_match_payload()
        return {
            "event_type": event_type,
            "previous_player": prev_player_id,
            "active_player": self.active_player_id,
            "active_cue_ball_color": self.active_cue_ball_color,
            "is_scored": is_scored,
            "already_dispatched": already_dispatched,
            "shot_id": self.shot_seq_id,
            "points_awarded": points_awarded,
            "payload": payload,
        }

    def get_inning_history_list(self) -> List[Dict[str, Optional[int]]]:
        """
        1이닝부터 현재 이닝까지 P1, P2의 이닝별 득점 시트(Inning Score Sheet) 리스트를 반환합니다.
        """
        max_inning = max(1, self.current_inning)
        p1_map = self.players["P1"].inning_scores
        p2_map = self.players["P2"].inning_scores

        sheet: List[Dict[str, Optional[int]]] = []
        for inn in range(1, max_inning + 1):
            p1_val = p1_map.get(inn, None) if inn <= self.players["P1"].innings else None
            p2_val = p2_map.get(inn, None) if inn <= self.players["P2"].innings else None
            sheet.append(
                {
                    "inning": inn,
                    "P1": p1_val,
                    "P2": p2_val,
                }
            )
        return sheet

    def get_match_payload(self) -> Dict[str, object]:
        """
        20인치 태블릿 스코어보드 전체 상태를 JSON 직렬화 가능한 딕셔너리로 반환합니다.
        """
        return {
            "fsm_state": self.fsm_state.value,
            "rule_mode": self.rule_mode,
            "active_player_id": self.active_player_id,
            "active_cue_ball_color": self.active_cue_ball_color,
            "current_inning": self.current_inning,
            "is_match_paused": self.is_match_paused,
            "match_timer": {
                "elapsed_sec": int(round(self.match_elapsed_sec)),
                "formatted": self.formatted_match_time,
                "is_paused": self.is_match_paused,
            },
            "shot_timer": {
                "remaining_sec": int(round(self.remaining_timer_sec)),
                "default_sec": self.default_shot_timer,
                "is_paused": self.is_timer_paused or self.is_match_paused,
            },
            "players": {
                "P1": self.players["P1"].to_dict(),
                "P2": self.players["P2"].to_dict(),
            },
            "inning_history": self.get_inning_history_list(),
            "last_shot_result": self.last_shot_result,
        }
