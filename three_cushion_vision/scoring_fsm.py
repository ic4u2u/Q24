"""
Module D: 쓰리쿠션 공식 규칙 결정론적 유한 상태 머신 (ThreeCushionScoringFSM)
파일 경로: three_cushion_vision/scoring_fsm.py

1. 상태 정의:
   - IDLE: 3개의 공이 모두 정지(speed < 15.0 mm/s가 0.3초 이상 유지)된 상태
   - IN_PLAY: 지정 수구(assigned_cue_ball)가 180.0 mm/s 이상으로 출발하여 이닝이 진행 중인 상태
              (적구 또는 상대방 수구가 먼저 출발 시 WRONG_CUE_BALL_FOUL로 즉시 NO_SCORE_LOCKED 전이)
   - VALID_SCORE_PENDING: 3쿠션 이상 득점 조건을 충족하고 모든 공이 정지하기를 기다리는 상태
   - NO_SCORE_LOCKED: 조기 제2목적구 접촉(EARLY_SECOND_BALL_KISS) 또는 당구대 이탈 파울 등으로 무효가 영구 잠금된 상태
2. 제2목적구 최초 접촉 게이트:
   - 제2목적구 최초 접촉 순간 cushion_count >= 3이면 VALID_SCORE_PENDING(선행 3쿠션 이상 시 is_bank_shot=True),
     cushion_count < 3이면 즉시 NO_SCORE_LOCKED(fail_reason="EARLY_SECOND_BALL_KISS")로 영구 잠금합니다.
3. 턴 종료 및 결과 반환:
   - 모든 공이 완전 정지하면 ShotResult를 반환하고, 노카운트 시 다음 이닝의 assigned_cue_ball을 상대방 수구로 자동 교대합니다.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from three_cushion_vision.config import (
    BallColor,
    DEFAULT_CONFIG,
    FailReason,
    Rail,
    ShotState,
    VisionConfig,
)
from three_cushion_vision.kinematics_detector import (
    ConfirmedContactEvent,
    CushionImpactEvent,
    KinematicsDetector,
)
from three_cushion_vision.tracker_adapter import TrackedBall


@dataclass
class ShotResult:
    """한 이닝(샷)의 모든 공이 정지한 후 확정되는 최종 쓰리쿠션 판정 결과 데이터클래스"""

    scored: bool
    cue_ball: BallColor
    cushions_before_second_ball: int
    total_cushions: int
    is_bank_shot: bool
    fail_reason: str
    events: List[str] = field(default_factory=list)
    inning_start_time: float = 0.0
    inning_end_time: float = 0.0
    first_target_ball: Optional[BallColor] = None
    second_target_ball: Optional[BallColor] = None
    retroactive_previous_score_compensated: bool = False
    object_ball_kiss_occurred: bool = False


class ThreeCushionScoringFSM:
    """
    KinematicsDetector와 연동하여 60fps 프레임 스트림에서 쓰리쿠션 공식 규칙을
    오차 없이 결정론적으로 판정하는 유한 상태 머신(FSM) 클래스입니다.
    """

    def __init__(
        self,
        initial_cue_ball: BallColor = BallColor.WHITE,
        config: VisionConfig = DEFAULT_CONFIG,
        kinematics: Optional[KinematicsDetector] = None,
        auto_compensate_missed_score: bool = True,
    ) -> None:
        self.config: VisionConfig = config
        self.assigned_cue_ball: BallColor = initial_cue_ball
        self.kinematics: KinematicsDetector = kinematics or KinematicsDetector(config=config)
        self.auto_compensate_missed_score: bool = auto_compensate_missed_score
        self._last_completed_cue_ball: Optional[BallColor] = None
        self._last_completed_was_miss: bool = False
        self.total_retroactive_compensations: int = 0
        self._reset_inning_state()

    def _reset_inning_state(self) -> None:
        """이닝 내부 상태 변수 초기화 (assigned_cue_ball은 유지)"""
        self.state: ShotState = ShotState.IDLE
        self.first_target_ball: Optional[BallColor] = None
        self.second_target_ball: Optional[BallColor] = None
        self.cushion_count: int = 0
        self.cushions_before_first_ball: int = 0
        self.cushions_at_second_ball: int = 0
        self.is_bank_shot: bool = False
        self.fail_reason: str = FailReason.NONE.value
        self.events: List[str] = []
        self.inning_start_time: float = 0.0
        self.all_stopped_since: Optional[float] = None
        self.retroactive_compensated_in_shot: bool = False
        self.kinematics.reset()

    def _check_balls_in_bounds(self, balls: Dict[BallColor, TrackedBall]) -> bool:
        """3개의 공이 모두 당구대 내측 경계(허용 오차 포함) 안에 있는지 검사합니다."""
        tol = self.config.OUT_OF_TABLE_TOLERANCE_MM
        w = self.config.TABLE_WIDTH_MM
        h = self.config.TABLE_HEIGHT_MM
        for b in balls.values():
            if not (-tol <= b.x <= (w + tol) and -tol <= b.y <= (h + tol)):
                return False
        return True

    def on_cushion_impact(self, event: CushionImpactEvent) -> None:
        """수구의 유효 쿠션 반사 이벤트를 FSM에 반영합니다."""
        self.cushion_count += 1
        if self.first_target_ball is None:
            self.cushions_before_first_ball += 1
        self.events.append(
            f"[{event.timestamp:.3f}s] CUSHION #{self.cushion_count} ({event.rail.value})"
        )

    def on_ball_contact(self, event: ConfirmedContactEvent) -> None:
        """
        수구와 목적구(제1목적구 또는 제2목적구)의 접촉 이벤트를 처리하고,
        [제2목적구 최초 접촉 게이트] 규칙에 따라 VALID_SCORE_PENDING 또는 NO_SCORE_LOCKED로 즉시 전이합니다.
        """
        target = event.target_color

        # 1) 제1목적구 최초 접촉
        if self.first_target_ball is None:
            self.first_target_ball = target
            self.events.append(
                f"[{event.timestamp:.3f}s] 1st_BALL_HIT ({target.value}, type={event.contact_type}, "
                f"d_min={event.d_min_mm:.1f}mm, cushions={self.cushion_count})"
            )
            return

        # 이미 맞힌 제1목적구와의 재접촉(키스)은 제2목적구 접촉이 아니므로 로그만 남김
        if target == self.first_target_ball:
            self.events.append(
                f"[{event.timestamp:.3f}s] 1st_BALL_RECONTACT ({target.value})"
            )
            return

        # 2) [핵심] 제2목적구 최초 접촉 게이트 (First Contact with 2nd Object Ball)
        if self.second_target_ball is None:
            self.second_target_ball = target
            cushions_now = max(self.cushion_count, event.cushions_at_impact)
            self.cushions_at_second_ball = cushions_now

            if self.state == ShotState.NO_SCORE_LOCKED:
                self.events.append(
                    f"[{event.timestamp:.3f}s] 2nd_BALL_HIT_IGNORED_LOCKED ({target.value})"
                )
                return

            if cushions_now >= 3:
                self.state = ShotState.VALID_SCORE_PENDING
                self.is_bank_shot = self.cushions_before_first_ball >= 3
                self.fail_reason = FailReason.NONE.value
                shot_tag = "BANK_SHOT" if self.is_bank_shot else "REGULAR_3C"
                self.events.append(
                    f"[{event.timestamp:.3f}s] 2nd_BALL_HIT_VALID ({target.value}, {shot_tag}, "
                    f"cushions={cushions_now}, type={event.contact_type})"
                )
            else:
                # 3쿠션 미만 상태에서 제2목적구와 먼저 접촉 -> 즉시 NO_SCORE_LOCKED 영구 잠금!
                self.state = ShotState.NO_SCORE_LOCKED
                self.fail_reason = FailReason.EARLY_SECOND_BALL_KISS.value
                self.events.append(
                    f"[{event.timestamp:.3f}s] EARLY_2nd_BALL_KISS_LOCKED ({target.value}, "
                    f"cushions={cushions_now} < 3)"
                )
        else:
            # 이미 제2목적구와 최초 접촉이 끝난 이후의 재접촉:
            # 만약 앞서 EARLY_SECOND_BALL_KISS로 NO_SCORE_LOCKED 상태가 되었다면 절대 득점으로 바뀌지 않음!
            self.events.append(
                f"[{event.timestamp:.3f}s] 2nd_BALL_RECONTACT ({target.value}, state={self.state.value})"
            )

    def update(
        self,
        balls: Dict[BallColor, TrackedBall],
        timestamp: float,
    ) -> Optional[ShotResult]:
        """
        매 프레임(60fps) 추적된 3구 상태(balls)와 타임스탬프를 입력받아
        타구 시작, 쿠션·목적구 접촉, 완전 정지 및 이닝 종료(ShotResult 반환)를 수행합니다.
        """
        if len(balls) < 3 or self.assigned_cue_ball not in balls:
            return None

        cue = balls[self.assigned_cue_ball]
        other_balls = [b for c, b in balls.items() if c != self.assigned_cue_ball]

        # ------------------------------------------------------------------
        # 1. IDLE 상태: 지정 수구의 스트로크 출발(>= 180.0 mm/s) 또는
        #    수구 색상 감지 기반 미인지 득점 자동 보정 / 파울 출발 감시
        # ------------------------------------------------------------------
        if self.state == ShotState.IDLE:
            if cue.speed >= self.config.STROKE_START_SPEED_MM_S:
                self.state = ShotState.IN_PLAY
                self.inning_start_time = timestamp
                self.events.append(
                    f"[{timestamp:.3f}s] STROKE_START (cue={self.assigned_cue_ball.value}, "
                    f"speed={cue.speed:.1f}mm/s)"
                )
            else:
                for ob in other_balls:
                    if ob.speed >= self.config.STROKE_START_SPEED_MM_S:
                        # [핵심] 선수가 치는 공의 색깔(WHITE 또는 YELLOW)을 감지하여
                        # 직전 샷에서 득점 인지를 못해 턴이 넘어갔던 경우 미인지 득점(+1점)을 자동 보정하고 수구를 복원!
                        if (
                            self.auto_compensate_missed_score
                            and ob.color in (BallColor.WHITE, BallColor.YELLOW)
                        ):
                            if (
                                self._last_completed_was_miss
                                and self._last_completed_cue_ball == ob.color
                            ):
                                self.retroactive_compensated_in_shot = True
                                self.total_retroactive_compensations += 1
                                self.assigned_cue_ball = ob.color
                                self.state = ShotState.IN_PLAY
                                self.inning_start_time = timestamp
                                self.events.append(
                                    f"[{timestamp:.3f}s] RETROACTIVE_SCORE_COMPENSATED "
                                    f"(cue={ob.color.value}, speed={ob.speed:.1f}mm/s)"
                                )
                            else:
                                self.assigned_cue_ball = ob.color
                                self.state = ShotState.IN_PLAY
                                self.inning_start_time = timestamp
                                self.events.append(
                                    f"[{timestamp:.3f}s] CUE_COLOR_SYNCED_STROKE_START "
                                    f"(cue={ob.color.value}, speed={ob.speed:.1f}mm/s)"
                                )
                        else:
                            self.state = ShotState.NO_SCORE_LOCKED
                            self.inning_start_time = timestamp
                            self.fail_reason = FailReason.WRONG_CUE_BALL_FOUL.value
                            self.events.append(
                                f"[{timestamp:.3f}s] WRONG_CUE_BALL_FOUL (moved={ob.color.value}, "
                                f"expected={self.assigned_cue_ball.value})"
                            )
                        break

            # 만약 출발 직후 같은 프레임에서 쿠션/접촉 검사를 위한 초기 위치를 KinematicsDetector에 동기화
            if self.state != ShotState.IDLE:
                self.kinematics.step(
                    balls=balls,
                    assigned_cue_ball=self.assigned_cue_ball,
                    timestamp=timestamp,
                    current_cushion_count=self.cushion_count,
                )
            return None

        # ------------------------------------------------------------------
        # 2. IN_PLAY / VALID_SCORE_PENDING / NO_SCORE_LOCKED 주행 중 물리 판정
        # ------------------------------------------------------------------
        if not self._check_balls_in_bounds(balls):
            if self.state != ShotState.NO_SCORE_LOCKED:
                self.state = ShotState.NO_SCORE_LOCKED
                self.fail_reason = FailReason.BALL_OUT_OF_TABLE.value
                self.events.append(f"[{timestamp:.3f}s] BALL_OUT_OF_TABLE_FOUL")

        cushion_events, contact_events = self.kinematics.step(
            balls=balls,
            assigned_cue_ball=self.assigned_cue_ball,
            timestamp=timestamp,
            current_cushion_count=self.cushion_count,
        )

        # 동일 프레임 내에서 쿠션과 목적구 접촉의 시간순 정렬 처리
        for c_ev in cushion_events:
            self.on_cushion_impact(c_ev)

        for hit_ev in contact_events:
            self.on_ball_contact(hit_ev)

        # ------------------------------------------------------------------
        # 3. 모든 공(3개) 완전 정지 감지 (< 15.0 mm/s 상태가 0.3초 이상 지속)
        # ------------------------------------------------------------------
        all_stopped = all(
            b.speed < self.config.IDLE_STOP_SPEED_MM_S for b in balls.values()
        )
        if all_stopped:
            if self.all_stopped_since is None:
                self.all_stopped_since = timestamp
            elif (
                timestamp - self.all_stopped_since
            ) >= self.config.IDLE_CONFIRM_DURATION_SEC:
                return self._finalize_inning(timestamp)
        else:
            self.all_stopped_since = None

        return None

    def _finalize_inning(self, end_timestamp: float) -> ShotResult:
        """
        모든 공이 정지했을 때 최종 ShotResult를 생성하고,
        노카운트(scored == False)인 경우 다음 이닝의 assigned_cue_ball을 상대방 수구로 자동 교대합니다.
        """
        scored = self.state == ShotState.VALID_SCORE_PENDING
        completed_cue_ball = self.assigned_cue_ball

        if not scored and self.fail_reason == FailReason.NONE.value:
            if self.first_target_ball is None:
                self.fail_reason = FailReason.MISSING_FIRST_BALL.value
            elif self.second_target_ball is None:
                if self.cushion_count < 3:
                    self.fail_reason = FailReason.INSUFFICIENT_CUSHIONS.value
                else:
                    self.fail_reason = FailReason.MISSING_SECOND_BALL.value
            else:
                self.fail_reason = FailReason.INSUFFICIENT_CUSHIONS.value

        result = ShotResult(
            scored=scored,
            cue_ball=completed_cue_ball,
            cushions_before_second_ball=self.cushions_at_second_ball,
            total_cushions=self.cushion_count,
            is_bank_shot=self.is_bank_shot if scored else False,
            fail_reason=FailReason.NONE.value if scored else self.fail_reason,
            events=list(self.events),
            inning_start_time=self.inning_start_time,
            inning_end_time=end_timestamp,
            first_target_ball=self.first_target_ball,
            second_target_ball=self.second_target_ball,
            retroactive_previous_score_compensated=self.retroactive_compensated_in_shot,
            object_ball_kiss_occurred=self.kinematics.object_ball_kiss_occurred,
        )

        self._last_completed_cue_ball = completed_cue_ball
        self._last_completed_was_miss = not scored

        # 노카운트 시 다음 이닝의 수구(assigned_cue_ball)를 상대방 수구(WHITE <-> YELLOW)로 자동 교대
        if not scored:
            self.assigned_cue_ball = (
                BallColor.YELLOW
                if self.assigned_cue_ball == BallColor.WHITE
                else BallColor.WHITE
            )

        self._reset_inning_state()
        return result
