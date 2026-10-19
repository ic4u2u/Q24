"""
대대 3쿠션 이벤트 시퀀스 기록 및 득점/공타 룰 판정 엔진
파일 경로: cue24/core/scoring_engine.py

1. 이벤트 시퀀스 기록: 제1적구 충돌(FIRST_HIT), 4면 쿠션 접촉(CUSHION), 제2적구 충돌(SECOND_HIT)
2. 쿠션 판정: 쿠션 마진(32.0mm) 진입 + 법선 속도 부호 반전(Velocity Reversal) 발생 시에만 유효 1쿠션 인정
3. 룰 평가: 일반 3쿠션 득점, 뱅크 샷(공 접촉 전 3쿠션 이상 선행) 득점, 공타 사유(쿠션 부족, 2적구 불발 등) 정밀 분류
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from cue24.config.settings import settings

Point2D = Tuple[float, float]


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

        # 30fps 카메라 샘플링 간격 및 쿠션 그림자 마진을 고려한 유효 쿠션 반사 감지 경계 (82.0mm와 설정 마진 중 큰 값 적용)
        effective_rail_zone = max(cushion_margin_mm, 82.0)
        self.left_bound: float = effective_rail_zone
        self.right_bound: float = table_width_mm - effective_rail_zone
        self.top_bound: float = effective_rail_zone
        self.bottom_bound: float = table_height_mm - effective_rail_zone

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
        새로운 샷 시작 시 이벤트 타임라인 및 쿠션/목적구 충돌 상태를 초기화합니다.
        """
        if cue_ball_id is not None:
            self.cue_ball_id = cue_ball_id
        if object_ball_ids is not None:
            self.object_ball_ids = list(object_ball_ids)

        self.events: List[ShotEvent] = []
        self.first_object_ball_id: Optional[int] = None
        self.second_object_ball_id: Optional[int] = None

        self.cushion_count_before_first_hit: int = 0
        self.cushion_count_before_second_hit: int = 0
        self.total_cushion_count: int = 0

        # 쿠션 법선 속도 반전(Velocity Reversal) 추적용 상태
        self._prev_cue_pos: Optional[Point2D] = None
        self._prev_cue_vel: Optional[Point2D] = None
        self._prev_ball_positions: Dict[int, Point2D] = {}
        self._initial_ball_positions: Dict[int, Point2D] = {}
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
        다른 레일이나 목적구 접촉 없이 동일 레일을 따라 구르며 연속 중복 카운트되는 현상을 방지합니다.
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
        if self.second_object_ball_id is None:
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
        수구와 목적구(제1적구 또는 제2적구)의 충돌을 이벤트 시퀀스에 기록합니다.
        """
        if self.first_object_ball_id is None:
            self.first_object_ball_id = target_ball_id
            self._last_rail_hit = None
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

        # 1. LEFT 레일 검사
        if x <= self.left_bound:
            if vx < -1e-4 or prev_vx < -1e-4:
                self._approaching_rail["LEFT"] = True
            if self._approaching_rail["LEFT"] and vx > 1e-4:
                self.record_cushion_touch("LEFT", frame_idx, cue_pos)
                self._approaching_rail["LEFT"] = False
                triggered_rails.append("LEFT")
        else:
            if self._approaching_rail["LEFT"] and vx > 1e-4:
                self.record_cushion_touch("LEFT", frame_idx, cue_pos)
                triggered_rails.append("LEFT")
            self._approaching_rail["LEFT"] = False

        # 2. RIGHT 레일 검사
        if x >= self.right_bound:
            if vx > 1e-4 or prev_vx > 1e-4:
                self._approaching_rail["RIGHT"] = True
            if self._approaching_rail["RIGHT"] and vx < -1e-4:
                self.record_cushion_touch("RIGHT", frame_idx, cue_pos)
                self._approaching_rail["RIGHT"] = False
                triggered_rails.append("RIGHT")
        else:
            if self._approaching_rail["RIGHT"] and vx < -1e-4:
                self.record_cushion_touch("RIGHT", frame_idx, cue_pos)
                triggered_rails.append("RIGHT")
            self._approaching_rail["RIGHT"] = False

        # 3. TOP 레일 검사
        if y <= self.top_bound:
            if vy < -1e-4 or prev_vy < -1e-4:
                self._approaching_rail["TOP"] = True
            if self._approaching_rail["TOP"] and vy > 1e-4:
                self.record_cushion_touch("TOP", frame_idx, cue_pos)
                self._approaching_rail["TOP"] = False
                triggered_rails.append("TOP")
        else:
            if self._approaching_rail["TOP"] and vy > 1e-4:
                self.record_cushion_touch("TOP", frame_idx, cue_pos)
                triggered_rails.append("TOP")
            self._approaching_rail["TOP"] = False

        # 4. BOTTOM 레일 검사
        if y >= self.bottom_bound:
            if vy > 1e-4 or prev_vy > 1e-4:
                self._approaching_rail["BOTTOM"] = True
            if self._approaching_rail["BOTTOM"] and vy < -1e-4:
                self.record_cushion_touch("BOTTOM", frame_idx, cue_pos)
                self._approaching_rail["BOTTOM"] = False
                triggered_rails.append("BOTTOM")
        else:
            if self._approaching_rail["BOTTOM"] and vy < -1e-4:
                self.record_cushion_touch("BOTTOM", frame_idx, cue_pos)
                triggered_rails.append("BOTTOM")
            self._approaching_rail["BOTTOM"] = False

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

        # 1. 4면 쿠션 법선 속도 부호 반전(Velocity Reversal) 판정
        self._check_cushion_velocity_reversal(frame_idx, cue_pos, cue_vel)

        # 2. 목적구(제1적구, 제2적구) 충돌 판정 (중심 거리 + 선분 접촉 + 3차 스플라인 곡선 복원 + 정지 목적구 피격 감지)
        spline_pts: List[Point2D] = []
        if self._prev_cue_pos is not None and self._prev_cue_vel is not None:
            from cue24.core.physics_tracker import PhysicsTracker

            spline_pts = PhysicsTracker.interpolate_cubic_spline(
                self._prev_cue_pos, self._prev_cue_vel, cue_pos, cue_vel, num_steps=8
            )

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

            # 목적구가 직전 프레임 대비 충격을 받아 갑자기 움직였는지 확인 (30fps 샘플링 보완)
            target_impact_moved = False
            if bid in self._prev_ball_positions and bid in self._initial_ball_positions:
                prev_t = self._prev_ball_positions[bid]
                init_t = self._initial_ball_positions[bid]
                step_move = math.hypot(pos[0] - prev_t[0], pos[1] - prev_t[1])
                init_disp = math.hypot(prev_t[0] - init_t[0], prev_t[1] - init_t[1])
                # 정지해 있던 목적구(초기 위치에서 15mm 이내)가 1프레임에 6mm 이상 튕겨 나가고 수구가 근처에 있었던 경우
                if init_disp <= 15.0 and step_move >= 6.0 and min_approach_dist <= 220.0:
                    target_impact_moved = True

            if min_approach_dist <= self.ball_contact_th or target_impact_moved:
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
        샷이 완전히 정지한 후 기록된 이벤트 시퀀스를 기반으로 최종 3쿠션 룰을 평가합니다.

        - 일반 3C: 1적구 충돌 -> (2적구 충돌 전 총 쿠션 접촉 >= 3회) -> 2적구 충돌 성립 시 득점 인정
        - 뱅크 샷: 공 접촉 전 쿠션 접촉 >= 3회 선행 -> 1적구 충돌 -> 2적구 충돌 성립 시 득점 인정
        - 그 외: 공타(Miss) 사유(쿠션 부족, 2적구 불발, 1적구 불발, 더블터치 파울) 분류

        Returns:
            result: {
                "is_scored": bool,
                "shot_type": "REGULAR_3C" | "BANK_SHOT" | "MISS" | "FOUL",
                "reason_code": str,
                "message": str,
                "cushions_before_first_hit": int,
                "cushions_before_second_hit": int,
                "total_cushions": int,
                "events": List[ShotEvent],
            }
        """
        if has_double_touch_foul:
            return {
                "is_scored": False,
                "shot_type": "FOUL",
                "reason_code": "DOUBLE_TOUCH_FOUL",
                "message": "쿠션 이중 터치(Double Touch) 파울로 공타 처리되었습니다.",
                "cushions_before_first_hit": self.cushion_count_before_first_hit,
                "cushions_before_second_hit": self.cushion_count_before_second_hit,
                "total_cushions": self.total_cushion_count,
                "events": list(self.events),
            }

        # 케이스 1: 제1적구조차 맞히지 못한 경우
        if self.first_object_ball_id is None:
            return {
                "is_scored": False,
                "shot_type": "MISS",
                "reason_code": "NO_BALL_HIT",
                "message": "공타: 제1목적구 충돌이 발생하지 않았습니다.",
                "cushions_before_first_hit": self.cushion_count_before_first_hit,
                "cushions_before_second_hit": self.cushion_count_before_second_hit,
                "total_cushions": self.total_cushion_count,
                "events": list(self.events),
            }

        # 케이스 2: 제1적구는 맞혔으나 제2적구를 맞히지 못한 경우
        if self.second_object_ball_id is None:
            return {
                "is_scored": False,
                "shot_type": "MISS",
                "reason_code": "NO_SECOND_BALL_HIT",
                "message": f"공타: 제2목적구 불발 (현재 누적 {self.total_cushion_count}쿠션).",
                "cushions_before_first_hit": self.cushion_count_before_first_hit,
                "cushions_before_second_hit": self.cushion_count_before_second_hit,
                "total_cushions": self.total_cushion_count,
                "events": list(self.events),
            }

        # 케이스 3: 제1적구와 제2적구를 모두 맞혔으나, 제2적구 충돌 시점까지 누적 쿠션이 3회 미만인 경우
        if self.cushion_count_before_second_hit < 3:
            return {
                "is_scored": False,
                "shot_type": "MISS",
                "reason_code": "INSUFFICIENT_CUSHIONS",
                "message": (
                    f"공타: 쿠션 부족 (제2적구 접촉 시점 {self.cushion_count_before_second_hit}쿠션 < 3쿠션)."
                ),
                "cushions_before_first_hit": self.cushion_count_before_first_hit,
                "cushions_before_second_hit": self.cushion_count_before_second_hit,
                "total_cushions": self.total_cushion_count,
                "events": list(self.events),
            }

        # 케이스 4: 득점 성공! (공 접촉 전 3쿠션 이상 선행 여부에 따라 뱅크 샷 / 일반 3C 분류)
        if self.cushion_count_before_first_hit >= 3:
            return {
                "is_scored": True,
                "shot_type": "BANK_SHOT",
                "reason_code": "VALID_BANK_SHOT",
                "message": (
                    f"득점 인정 (뱅크 샷): 선행 {self.cushion_count_before_first_hit}쿠션 후 제1·2적구 충돌 성공!"
                ),
                "cushions_before_first_hit": self.cushion_count_before_first_hit,
                "cushions_before_second_hit": self.cushion_count_before_second_hit,
                "total_cushions": self.total_cushion_count,
                "events": list(self.events),
            }

        return {
            "is_scored": True,
            "shot_type": "REGULAR_3C",
            "reason_code": "VALID_REGULAR_3C",
            "message": (
                f"득점 인정 (일반 3쿠션): 총 {self.cushion_count_before_second_hit}쿠션 후 제2적구 충돌 성공!"
            ),
            "cushions_before_first_hit": self.cushion_count_before_first_hit,
            "cushions_before_second_hit": self.cushion_count_before_second_hit,
            "total_cushions": self.total_cushion_count,
            "events": list(self.events),
        }
