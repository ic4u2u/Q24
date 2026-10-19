"""
대대 3쿠션 이벤트 시퀀스 기록 및 득점/공타 룰 판정 엔진
파일 경로: cue24/core/scoring_engine.py

1. 이벤트 시퀀스 기록: 제1적구 충돌(FIRST_HIT), 4면 쿠션 접촉(CUSHION), 제2적구 충돌(SECOND_HIT)
2. 쿠션 판정: 쿠션 마진(32.0mm) 진입 + 법선 속도 부호 반전(Velocity Reversal) 발생 시에만 유효 1쿠션 인정
3. 룰 평가: 일반 3쿠션 득점, 뱅크 샷(공 접촉 전 3쿠션 이상 선행) 득점, 공타 사유(쿠션 부족, 2적구 불발 등) 정밀 분류
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from cue24.config.settings import settings

Point2D = Tuple[float, float]


class ShotFSMState(str, Enum):
    """결정론적 3쿠션 샷 판정 유한 상태 머신(FSM) 4단계 상태"""

    IDLE = "IDLE"                                # 정지 대기 상태 (저전력 감시 모드)
    IN_PLAY = "IN_PLAY"                          # 스트로크 후 샷 진행 중 (초고속 정밀 추적 모드)
    VALID_SCORE_PENDING = "VALID_SCORE_PENDING"  # 3쿠션 이상 달성 후 제2목적구 타격 성공 (정지 시 1점 확정)
    NO_SCORE_LOCKED = "NO_SCORE_LOCKED"          # 3쿠션 미만 조기 제2목적구 키스 또는 파울 발생 (영구 노카운트 잠금)


class FailureReason(str, Enum):
    """노카운트(공타) 및 파울 상세 사유 코드"""

    NONE = "NONE"
    INSUFFICIENT_CUSHIONS = "INSUFFICIENT_CUSHIONS"              # 3쿠션 미만
    EARLY_SECOND_BALL_CONTACT = "EARLY_SECOND_BALL_CONTACT"      # 3쿠션 달성 전 제2목적구 조기 접촉(키스 파울)
    MISSING_SECOND_BALL = "MISSING_SECOND_BALL"                  # 제2목적구 미접촉
    WRONG_CUE_BALL_FOUL = "WRONG_CUE_BALL_FOUL"                  # 상대방 수구 타격 파울
    BALL_OUT_OF_TABLE = "BALL_OUT_OF_TABLE"                      # 당구대 장외 이탈 파울


@dataclass
class ShotEvent:
    """샷 진행 중 발생한 단일 물리 이벤트(공 충돌 또는 쿠션 반사) 기록 데이터 클래스"""

    frame_idx: int
    event_type: str        # "FIRST_HIT" | "SECOND_HIT" | "CUSHION"
    detail: str            # 목적구 ID 문자열 또는 쿠션 면 이름 ("LEFT" | "RIGHT" | "TOP" | "BOTTOM")
    position_mm: Point2D   # 이벤트 발생 시점의 수구 (x, y) 좌표 (mm)


class ScoringEngine:
    """
    한 번의 샷(이닝) 동안 수구(Cue Ball)의 궤적과 충돌 이벤트를 실시간 추적하여
    3쿠션 국제 규칙에 따른 득점 여부(일반 3C / 뱅크샷) 및 상세 공타 사유를 판정하는 클래스입니다.
    """

    def __init__(
        self,
        table_width_mm: float = settings.TABLE_WIDTH_MM,
        table_height_mm: float = settings.TABLE_HEIGHT_MM,
        cushion_margin_mm: float = settings.CUSHION_MARGIN_MM,
        ball_diameter_mm: float = settings.BALL_DIAMETER_MM,
        contact_tolerance_mm: float = 2.5,
    ) -> None:
        """
        ScoringEngine 초기화

        Args:
            table_width_mm: 대대 가로 내경 (2840.0 mm)
            table_height_mm: 대대 세로 내경 (1420.0 mm)
            cushion_margin_mm: 쿠션 접촉 판정 마진 (32.0 mm)
            ball_diameter_mm: 당구공 직경 (61.5 mm)
            contact_tolerance_mm: 프레임 간 공-공 접촉 검출 허용 여유치 (mm)
        """
        self.table_width_mm: float = table_width_mm
        self.table_height_mm: float = table_height_mm
        self.cushion_margin_mm: float = cushion_margin_mm
        self.ball_contact_th: float = ball_diameter_mm + contact_tolerance_mm

        # 30fps 카메라 원근 투영 및 쿠션 레일 그림자 마진을 고려한 유효 쿠션 반사 감지 경계
        effective_rail_zone_x = max(cushion_margin_mm, 140.0)
        effective_rail_zone_y = max(cushion_margin_mm, 115.0)
        self.left_bound: float = effective_rail_zone_x
        self.right_bound: float = table_width_mm - effective_rail_zone_x
        self.top_bound: float = effective_rail_zone_y
        self.bottom_bound: float = table_height_mm - effective_rail_zone_y

        # 샷 상태 관리 변수 초기화
        self.cue_ball_id: Optional[int] = None
        self.object_ball_ids: List[int] = []
        self.reset_shot()

    def reset_shot(
        self,
        cue_ball_id: Optional[int] = None,
        object_ball_ids: Optional[List[int]] = None,
    ) -> None:
        """
        새로운 샷 시작 시 이벤트 타임라인, 결정론적 4단계 FSM 및 쿠션 집합(in_rail_zone) 상태를 초기화합니다.
        """
        if cue_ball_id is not None:
            self.cue_ball_id = cue_ball_id
        if object_ball_ids is not None:
            self.object_ball_ids = list(object_ball_ids)

        self.fsm_state: ShotFSMState = ShotFSMState.IDLE
        self.failure_reason: FailureReason = FailureReason.NONE
        self.failed_early: bool = False
        self.in_rail_zone: Set[str] = set()

        self.events: List[ShotEvent] = []
        self.first_object_ball_id: Optional[int] = None
        self.second_object_ball_id: Optional[int] = None

        self.cushion_count_before_first_hit: int = 0
        self.cushion_count_before_second_hit: int = 0
        self.total_cushion_count: int = 0
        self.cue_travel_distance_mm: float = 0.0

        # 쿠션 법선 속도 반전(Velocity Reversal) 및 목적구 실제 충격 이동 추적용 상태
        self._prev_cue_pos: Optional[Point2D] = None
        self._prev_cue_vel: Optional[Point2D] = None
        self._prev_ball_positions: Dict[int, Point2D] = {}
        self._initial_ball_positions: Dict[int, Point2D] = {}
        self._ball_trajectories: Dict[int, List[Point2D]] = {}
        self._target_first_move_frame: Dict[int, int] = {}
        self._last_rail_hit: Optional[str] = None
        self._last_cushion_frame: Dict[str, int] = {
            "LEFT": -999,
            "RIGHT": -999,
            "TOP": -999,
            "BOTTOM": -999,
        }
        # 쿠션 마진 내부에서 아직 반전 카운트가 처리되지 않은 접근 상태 기록
        self._approaching_rail: Dict[str, bool] = {
            "LEFT": False,
            "RIGHT": False,
            "TOP": False,
            "BOTTOM": False,
        }
        # 동일 목적구 연속 프레임 중복 접촉 방지용 플래그
        self._in_contact_with_ball: Dict[int, bool] = {}

    def record_cushion_touch(
        self, rail_name: str, frame_idx: int, cue_pos: Point2D
    ) -> None:
        """
        유효 1쿠션 접촉(법선 속도 반전 확인 완료)을 이벤트 시퀀스에 기록합니다.
        - 코너 더블쿠션(예: TOP -> RIGHT 연속 반사)은 서로 다른 쿠션이므로 쿨타임 없이 즉시 2쿠션으로 인정합니다.
        - 같은 레일을 따라 굴러가는 '쿠션 타기(Rail-hugging)'는 동일 레일 연속 카운트 방지 로직으로 차단합니다.
        """
        if self._last_rail_hit == rail_name:
            return
        if frame_idx - self._last_cushion_frame.get(rail_name, -999) <= 2:
            return
        self._last_rail_hit = rail_name
        self._last_cushion_frame[rail_name] = frame_idx

        self.total_cushion_count += 1
        if self.first_object_ball_id is None:
            self.cushion_count_before_first_hit += 1
        if self.second_object_ball_id is None and not self.failed_early:
            self.cushion_count_before_second_hit += 1

        self.events.append(
            ShotEvent(
                frame_idx=frame_idx,
                event_type="CUSHION",
                detail=rail_name,
                position_mm=cue_pos,
            )
        )

    def record_ball_hit(
        self, target_ball_id: int, frame_idx: int, cue_pos: Point2D
    ) -> None:
        """
        수구와 목적구(제1적구 또는 제2적구)의 충돌을 이벤트 시퀀스에 기록하고,
        결정론적 4단계 FSM(VALID_SCORE_PENDING vs NO_SCORE_LOCKED) 상태를 전이합니다.
        """
        if self.first_object_ball_id is None:
            self.first_object_ball_id = target_ball_id
            self._last_rail_hit = None
            if self.fsm_state == ShotFSMState.IDLE:
                self.fsm_state = ShotFSMState.IN_PLAY
            self.events.append(
                ShotEvent(
                    frame_idx=frame_idx,
                    event_type="FIRST_HIT",
                    detail=str(target_ball_id),
                    position_mm=cue_pos,
                )
            )
        elif (
            target_ball_id != self.first_object_ball_id
            and self.second_object_ball_id is None
        ):
            self.second_object_ball_id = target_ball_id
            self._last_rail_hit = None

            # [결정론적 FSM 잠금 규칙]
            # 제2목적구 최초 접촉 시점에 누적 쿠션이 3회 미만이면 즉시 NO_SCORE_LOCKED(조기 키스 파울)로 영구 잠금!
            # 그 이후 공이 굴러 3쿠션을 채우고 다시 제2목적구를 맞혀도 절대 득점으로 인정되지 않습니다.
            if not self.failed_early and self.fsm_state != ShotFSMState.NO_SCORE_LOCKED:
                if self.cushion_count_before_second_hit >= 3:
                    self.fsm_state = ShotFSMState.VALID_SCORE_PENDING
                    self.failure_reason = FailureReason.NONE
                else:
                    self.failed_early = True
                    self.fsm_state = ShotFSMState.NO_SCORE_LOCKED
                    self.failure_reason = FailureReason.EARLY_SECOND_BALL_CONTACT

            self.events.append(
                ShotEvent(
                    frame_idx=frame_idx,
                    event_type="SECOND_HIT",
                    detail=str(target_ball_id),
                    position_mm=cue_pos,
                )
            )

    @staticmethod
    def _segment_to_point_dist(p1: Point2D, p2: Point2D, target: Point2D) -> float:
        """프레임 간 수구 이동 선분(p1 -> p2)과 목적구 중심(target) 사이의 최단 거리를 계산합니다."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        len_sq = dx * dx + dy * dy
        if len_sq < 1e-6:
            return math.hypot(target[0] - p1[0], target[1] - p1[1])
        t = max(0.0, min(1.0, ((target[0] - p1[0]) * dx + (target[1] - p1[1]) * dy) / len_sq))
        proj_x = p1[0] + t * dx
        proj_y = p1[1] + t * dy
        return math.hypot(target[0] - proj_x, target[1] - proj_y)

    def _check_cushion_velocity_reversal(
        self, frame_idx: int, cue_pos: Point2D, cue_vel: Point2D
    ) -> List[str]:
        """
        단순 마진 진입이 아닌, 쿠션 마진 영역 내에서 벽면을 향하던 법선 속도의 부호가
        반전(Velocity Reversal)되었을 때만 유효 1쿠션으로 판정합니다.

        - LEFT 쿠션 (x <= left_bound): 접근 속도 vx < 0 -> 반사 속도 vx > 0
        - RIGHT 쿠션 (x >= right_bound): 접근 속도 vx > 0 -> 반사 속도 vx < 0
        - TOP 쿠션 (y <= top_bound): 접근 속도 vy < 0 -> 반사 속도 vy > 0
        - BOTTOM 쿠션 (y >= bottom_bound): 접근 속도 vy > 0 -> 반사 속도 vy < 0
        """
        triggered_rails: List[str] = []
        x, y = cue_pos
        vx, vy = cue_vel
        prev_vx = self._prev_cue_vel[0] if self._prev_cue_vel is not None else 0.0
        prev_vy = self._prev_cue_vel[1] if self._prev_cue_vel is not None else 0.0
        prev_vel_tuple = (prev_vx, prev_vy)

        from cue24.mcp.billiard_physics_mcp import physics_ai_engine

        current_zones: Set[str] = set()
        if x <= self.left_bound:
            current_zones.add("LEFT")
        if x >= self.right_bound:
            current_zones.add("RIGHT")
        if y <= self.top_bound:
            current_zones.add("TOP")
        if y >= self.bottom_bound:
            current_zones.add("BOTTOM")

        # 1. LEFT 레일 검사
        if x <= self.left_bound:
            if vx < -1e-4 or prev_vx < -1e-4:
                self._approaching_rail["LEFT"] = True
            if self._approaching_rail["LEFT"] and vx > 1e-4:
                if physics_ai_engine.verify_cushion_impulse("LEFT", cue_pos, prev_vel_tuple, cue_vel):
                    self.record_cushion_touch("LEFT", frame_idx, cue_pos)
                    triggered_rails.append("LEFT")
                self._approaching_rail["LEFT"] = False
        else:
            if self._approaching_rail["LEFT"] and vx > 1e-4:
                if physics_ai_engine.verify_cushion_impulse("LEFT", cue_pos, prev_vel_tuple, cue_vel):
                    self.record_cushion_touch("LEFT", frame_idx, cue_pos)
                    triggered_rails.append("LEFT")
            self._approaching_rail["LEFT"] = False

        # 2. RIGHT 레일 검사
        if x >= self.right_bound:
            if vx > 1e-4 or prev_vx > 1e-4:
                self._approaching_rail["RIGHT"] = True
            if self._approaching_rail["RIGHT"] and vx < -1e-4:
                if physics_ai_engine.verify_cushion_impulse("RIGHT", cue_pos, prev_vel_tuple, cue_vel):
                    self.record_cushion_touch("RIGHT", frame_idx, cue_pos)
                    triggered_rails.append("RIGHT")
                self._approaching_rail["RIGHT"] = False
        else:
            if self._approaching_rail["RIGHT"] and vx < -1e-4:
                if physics_ai_engine.verify_cushion_impulse("RIGHT", cue_pos, prev_vel_tuple, cue_vel):
                    self.record_cushion_touch("RIGHT", frame_idx, cue_pos)
                    triggered_rails.append("RIGHT")
            self._approaching_rail["RIGHT"] = False

        # 3. TOP 레일 검사
        if y <= self.top_bound:
            if vy < -1e-4 or prev_vy < -1e-4:
                self._approaching_rail["TOP"] = True
            if self._approaching_rail["TOP"] and vy > 1e-4:
                if physics_ai_engine.verify_cushion_impulse("TOP", cue_pos, prev_vel_tuple, cue_vel):
                    self.record_cushion_touch("TOP", frame_idx, cue_pos)
                    triggered_rails.append("TOP")
                self._approaching_rail["TOP"] = False
        else:
            if self._approaching_rail["TOP"] and vy > 1e-4:
                if physics_ai_engine.verify_cushion_impulse("TOP", cue_pos, prev_vel_tuple, cue_vel):
                    self.record_cushion_touch("TOP", frame_idx, cue_pos)
                    triggered_rails.append("TOP")
            self._approaching_rail["TOP"] = False

        # 4. BOTTOM 레일 검사
        if y >= self.bottom_bound:
            if vy > 1e-4 or prev_vy > 1e-4:
                self._approaching_rail["BOTTOM"] = True
            if self._approaching_rail["BOTTOM"] and vy < -1e-4:
                if physics_ai_engine.verify_cushion_impulse("BOTTOM", cue_pos, prev_vel_tuple, cue_vel):
                    self.record_cushion_touch("BOTTOM", frame_idx, cue_pos)
                    triggered_rails.append("BOTTOM")
                self._approaching_rail["BOTTOM"] = False
        else:
            if self._approaching_rail["BOTTOM"] and vy < -1e-4:
                if physics_ai_engine.verify_cushion_impulse("BOTTOM", cue_pos, prev_vel_tuple, cue_vel):
                    self.record_cushion_touch("BOTTOM", frame_idx, cue_pos)
                    triggered_rails.append("BOTTOM")
            self._approaching_rail["BOTTOM"] = False

        self.in_rail_zone = current_zones
        return triggered_rails

    def update_frame(
        self,
        frame_idx: int,
        ball_positions: Dict[int, Point2D],
        cue_velocity: Optional[Point2D] = None,
    ) -> None:
        """
        매 프레임 수구 및 목적구들의 좌표를 입력받아 쿠션 법선 속도 반전과 목적구 충돌을 감지합니다.

        Args:
            frame_idx: 현재 프레임 번호
            ball_positions: {ball_id: (x_mm, y_mm)} 딕셔너리
            cue_velocity: 수구의 현재 프레임 속도 벡터 (vx, vy). 생략 시 이전 위치와의 차분으로 자동 계산
        """
        if self.cue_ball_id is None or self.cue_ball_id not in ball_positions:
            raise ValueError("수구(cue_ball_id)가 설정되지 않았거나 좌표 딕셔너리에 없습니다.")

        if not self._initial_ball_positions:
            self._initial_ball_positions = {
                bid: (float(pt[0]), float(pt[1])) for bid, pt in ball_positions.items()
            }
            self._ball_trajectories = {
                bid: [(float(pt[0]), float(pt[1]))] for bid, pt in ball_positions.items()
            }
        else:
            for bid, pt in ball_positions.items():
                if bid not in self._ball_trajectories:
                    self._ball_trajectories[bid] = []
                self._ball_trajectories[bid].append((float(pt[0]), float(pt[1])))

        cue_pos = ball_positions[self.cue_ball_id]

        if cue_velocity is None:
            if self._prev_cue_pos is None:
                cue_vel = (0.0, 0.0)
            else:
                cue_vel = (
                    cue_pos[0] - self._prev_cue_pos[0],
                    cue_pos[1] - self._prev_cue_pos[1],
                )
        else:
            cue_vel = cue_velocity

        if self.fsm_state == ShotFSMState.IDLE and math.hypot(cue_vel[0], cue_vel[1]) > 0.5:
            self.fsm_state = ShotFSMState.IN_PLAY

        if self._prev_cue_pos is not None:
            self.cue_travel_distance_mm += math.hypot(
                cue_pos[0] - self._prev_cue_pos[0],
                cue_pos[1] - self._prev_cue_pos[1],
            )

        # 1. 4면 쿠션 법선 속도 부호 반전(Velocity Reversal) + 물리 충격량(Impulse) 판정
        triggered_rails = self._check_cushion_velocity_reversal(frame_idx, cue_pos, cue_vel)

        # 수구의 허공 진행 각도 변화량(Deflection Angle) 계산 (쿠션 반사 프레임이 아닐 때)
        cue_deflection_deg = 0.0
        if not triggered_rails and self._prev_cue_vel is not None:
            v1_norm = math.hypot(self._prev_cue_vel[0], self._prev_cue_vel[1])
            v2_norm = math.hypot(cue_vel[0], cue_vel[1])
            if v1_norm >= 2.5 and v2_norm >= 2.5:
                cos_ang = (
                    self._prev_cue_vel[0] * cue_vel[0]
                    + self._prev_cue_vel[1] * cue_vel[1]
                ) / (v1_norm * v2_norm)
                cos_ang = max(-1.0, min(1.0, cos_ang))
                cue_deflection_deg = math.degrees(math.acos(cos_ang))

        # 2. 목적구(제1적구, 제2적구) 충돌 판정 (수구 직접 접촉 + 스플라인 복원 + 적구 간 키스 분리 검증)
        spline_pts: List[Point2D] = []
        if self._prev_cue_pos is not None and self._prev_cue_vel is not None:
            from cue24.core.physics_tracker import PhysicsTracker

            spline_pts = PhysicsTracker.interpolate_cubic_spline(
                self._prev_cue_pos, self._prev_cue_vel, cue_pos, cue_vel, num_steps=8
            )

        from cue24.mcp.billiard_physics_mcp import physics_ai_engine

        for bid, pos in ball_positions.items():
            if bid == self.cue_ball_id:
                continue

            dist = math.hypot(cue_pos[0] - pos[0], cue_pos[1] - pos[1])
            seg_dist = (
                self._segment_to_point_dist(self._prev_cue_pos, cue_pos, pos)
                if self._prev_cue_pos is not None
                else dist
            )
            spline_dist = (
                min(math.hypot(sp[0] - pos[0], sp[1] - pos[1]) for sp in spline_pts)
                if spline_pts
                else dist
            )
            min_approach_dist = min(dist, seg_dist, spline_dist)

            # 다른 목적구(제1적구 등)와 현재 목적구 사이의 거리 계산 -> 적구끼리 부딪힌 키스(Kiss) 오판 차단용
            other_obj_dist = 9999.0
            for other_id, other_pos in ball_positions.items():
                if other_id != self.cue_ball_id and other_id != bid:
                    d_obj = math.hypot(pos[0] - other_pos[0], pos[1] - other_pos[1])
                    if d_obj < other_obj_dist:
                        other_obj_dist = d_obj

            step_move = 0.0
            prev_init_disp = 0.0
            curr_init_disp = 0.0
            if bid in self._initial_ball_positions:
                init_t = self._initial_ball_positions[bid]
                curr_init_disp = math.hypot(pos[0] - init_t[0], pos[1] - init_t[1])
                if curr_init_disp >= 18.0 and bid not in self._target_first_move_frame:
                    self._target_first_move_frame[bid] = frame_idx

                prev_t = self._prev_ball_positions.get(bid, init_t)
                step_move = math.hypot(pos[0] - prev_t[0], pos[1] - prev_t[1])
                prev_init_disp = math.hypot(
                    prev_t[0] - init_t[0], prev_t[1] - init_t[1]
                )

            is_direct_hit = physics_ai_engine.verify_direct_cue_hit(
                min_cue_to_target_dist_mm=min_approach_dist,
                target_step_move_mm=step_move,
                target_init_disp_mm=prev_init_disp,
                curr_init_disp_mm=curr_init_disp,
                other_object_to_target_dist_mm=other_obj_dist,
                contact_threshold_mm=self.ball_contact_th,
                is_first_object_hit_Done=(self.first_object_ball_id is not None),
                cue_deflection_deg=cue_deflection_deg,
            )

            if is_direct_hit:
                if not self._in_contact_with_ball.get(bid, False):
                    self.record_ball_hit(bid, frame_idx, cue_pos)
                    self._in_contact_with_ball[bid] = True
            else:
                self._in_contact_with_ball[bid] = False

        self._prev_cue_pos = cue_pos
        self._prev_cue_vel = cue_vel
        self._prev_ball_positions = {
            bid: (float(pt[0]), float(pt[1])) for bid, pt in ball_positions.items()
        }

    def evaluate_shot(self, has_double_touch_foul: bool = False) -> Dict[str, object]:
        """
        샷이 완전히 정지한 후 기록된 물리 이벤트 시퀀스, 결정론적 4단계 FSM 상태(NO_SCORE_LOCKED 등) 및
        전체 시계열 궤적 텐서를 기반으로 로컬 물리/AI 3쿠션 검증 엔진을 통해 최종 3쿠션 룰을 평가합니다.

        - 일반 3C: 1적구 충돌 -> (2적구 충돌 전 총 쿠션 접촉 >= 3회) -> 2적구 충돌 성립 시에만 득점 인정
        - 뱅크 샷: 공 접촉 전 쿠션 접촉 >= 3회 선행 -> 1적구 충돌 -> 2적구 충돌 성립 시에만 득점 인정
        - 조기 키스 파울(NO_SCORE_LOCKED): 3쿠션 전에 제2적구를 먼저 건드린 경우 이후 3쿠션을 채워도 영구 노카운트 처리
        """
        from cue24.mcp.billiard_physics_mcp import physics_ai_engine

        # 시계열 궤적 텐서 역추적을 통해 모션 블러/충돌 직후 트랙 스왑으로 누락된 실제 3쿠션 타격을 복원
        reconstructed = physics_ai_engine.reconstruct_shot_from_trajectories(
            cue_ball_id=self.cue_ball_id,
            first_object_ball_id=self.first_object_ball_id,
            second_object_ball_id=self.second_object_ball_id,
            cushions_before_first_hit=self.cushion_count_before_first_hit,
            cushions_before_second_hit=self.cushion_count_before_second_hit,
            total_cushions=self.total_cushion_count,
            initial_positions=self._initial_ball_positions,
            ball_trajectories=self._ball_trajectories,
            left_bound=self.left_bound,
            right_bound=self.right_bound,
            top_bound=self.top_bound,
            bottom_bound=self.bottom_bound,
            failed_early=self.failed_early,
        )
        if bool(reconstructed.get("reconstructed", False)):
            self.first_object_ball_id = reconstructed["first_object_ball_id"]
            self.second_object_ball_id = reconstructed["second_object_ball_id"]
            self.cushion_count_before_first_hit = int(
                reconstructed["cushions_before_first_hit"]
            )
            self.cushion_count_before_second_hit = int(
                reconstructed["cushions_before_second_hit"]
            )
            self.total_cushion_count = int(reconstructed["total_cushions"])
            if self.second_object_ball_id is None:
                self.failed_early = False
                self.fsm_state = ShotFSMState.IN_PLAY
                self.failure_reason = FailureReason.MISSING_SECOND_BALL
            elif self.cushion_count_before_second_hit >= 3:
                self.failed_early = False
                self.fsm_state = ShotFSMState.VALID_SCORE_PENDING
                self.failure_reason = FailureReason.NONE

        verified = physics_ai_engine.verify_3cushion_shot(
            first_object_ball_id=self.first_object_ball_id,
            second_object_ball_id=self.second_object_ball_id,
            cushions_before_first_hit=self.cushion_count_before_first_hit,
            cushions_before_second_hit=self.cushion_count_before_second_hit,
            total_cushions=self.total_cushion_count,
            has_double_touch_foul=has_double_touch_foul,
            failed_early=self.failed_early,
        )

        if bool(verified["is_scored"]):
            self.fsm_state = ShotFSMState.VALID_SCORE_PENDING
            self.failure_reason = FailureReason.NONE
        else:
            if self.failed_early:
                self.fsm_state = ShotFSMState.NO_SCORE_LOCKED
                self.failure_reason = FailureReason.EARLY_SECOND_BALL_CONTACT
            elif self.second_object_ball_id is None:
                self.failure_reason = FailureReason.MISSING_SECOND_BALL
            elif self.cushion_count_before_second_hit < 3:
                self.failure_reason = FailureReason.INSUFFICIENT_CUSHIONS

        return {
            "is_scored": bool(verified["is_scored"]),
            "shot_type": str(verified["shot_type"]),
            "reason_code": str(verified["reason_code"]),
            "failure_reason": self.failure_reason.value,
            "fsm_state": self.fsm_state.value,
            "message": str(verified["message"]),
            "confidence": float(verified.get("confidence", 1.0)),
            "cushions_before_first_hit": self.cushion_count_before_first_hit,
            "cushions_before_second_hit": self.cushion_count_before_second_hit,
            "total_cushions": self.total_cushion_count,
            "events": list(self.events),
        }
