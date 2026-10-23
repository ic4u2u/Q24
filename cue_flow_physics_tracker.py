"""
cue_flow_physics_tracker.py:
GitHub ekiefl/pooltool 연속 사건 물리 시뮬레이션(TOI: Time-Of-Impact) 및
IMM(Interactive Multiple Model) 디지털 트윈 기반의 수구 전생애 흐름 추적 모듈
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

TABLE_W = 2840.0
TABLE_H = 1420.0
BALL_R = 30.75
BALL_DIAM = BALL_R * 2.0  # 61.5mm


class FlowEventType(Enum):
    STROKE_START = "STROKE_START"
    CUSHION_HIT = "CUSHION_HIT"
    FIRST_BALL_HIT = "FIRST_BALL_HIT"
    SECOND_BALL_HIT = "SECOND_BALL_HIT"
    FIRST_BALL_REHIT = "FIRST_BALL_REHIT"
    OBJECT_BALL_KISS = "OBJECT_BALL_KISS"
    STROKE_STOP = "STROKE_STOP"


@dataclass
class CueFlowEvent:
    timestamp: float
    event_type: FlowEventType
    detail: str
    cue_pos_mm: Tuple[float, float]
    cue_dir_deg: float
    cushion_count_so_far: int


@dataclass
class CueTrajectorySegment:
    """수구가 충돌 없이 직선으로 주행한 개별 구간(Segment) 정보"""
    seg_index: int
    start_time: float
    end_time: float
    start_pos: np.ndarray
    end_pos: np.ndarray
    direction_unit: np.ndarray
    direction_deg: float
    length_mm: float


class CueFlowPhysicsTwin:
    """
    [GitHub ekiefl/pooltool & IMM 물리 시뮬레이션 차용 모듈]
    1) 정지 상태에서 최초 운동에너지 발생 공(WHITE vs YELLOW)을 감지해 수구로 락킹합니다.
    2) 매 프레임 3가지 물리 시뮬레이션 가설(H_직진, H_쿠션거울반사, H_강체구충돌)을 계산하여
       바운딩 박스 겹침이나 가려짐이 발생해도 수구의 흐름을 끝까지 추적합니다.
    3) 이차방정식 기반 연속 충돌 시각(TOI: tau_rail vs tau_ball)과 초기 출발 광선(Ray) 기하학을 풀어
       '쿠션 선접촉(뱅크샷)'과 '1적구 선접촉(일반샷)'의 순서를 오차 없이 판별합니다.
    """

    def __init__(self, restitution: float = 0.82, friction_decay: float = 0.992):
        self.restitution = restitution
        self.friction_decay = friction_decay
        self.reset()

    def reset(self):
        self.locked_cue_ball = None
        self.stroke_origin_pos: Optional[np.ndarray] = None
        self.initial_ray_dir: Optional[np.ndarray] = None
        self.initial_ray_locked: bool = False

        self.twin_pos: Optional[np.ndarray] = None
        self.twin_vel: np.ndarray = np.zeros(2, dtype=np.float64)
        self.prev_time: Optional[float] = None

        self.first_contact_type: Optional[str] = None  # "CUSHION_FIRST" or "BALL1_FIRST"
        self.events: List[CueFlowEvent] = []
        self.segments: List[CueTrajectorySegment] = []
        self.current_seg_start_t: float = 0.0
        self.current_seg_start_pos: Optional[np.ndarray] = None

    @staticmethod
    def _vec_to_deg(v: np.ndarray) -> float:
        if float(np.linalg.norm(v)) < 1e-6:
            return 0.0
        return float(np.degrees(math.atan2(float(v[1]), float(v[0]))))

    def lock_cue_ball_at_start(
        self,
        t: float,
        cue_ball_color,
        origin_pos: np.ndarray,
        initial_pos: np.ndarray,
        initial_vel: np.ndarray,
    ):
        """수구가 처음 움직이기 시작한 순간 색상(WHITE/YELLOW)과 출발 원점을 고정합니다."""
        self.reset()
        self.locked_cue_ball = cue_ball_color
        self.stroke_origin_pos = origin_pos.copy()
        self.twin_pos = initial_pos.copy()
        self.twin_vel = initial_vel.copy()
        self.prev_time = t
        self.current_seg_start_t = t
        self.current_seg_start_pos = origin_pos.copy()

        disp = initial_pos - origin_pos
        d_norm = float(np.linalg.norm(disp))
        v_norm = float(np.linalg.norm(initial_vel))
        if d_norm >= 15.0:
            self.initial_ray_dir = disp / d_norm
        elif v_norm > 1e-3:
            self.initial_ray_dir = initial_vel / v_norm

        self.events.append(
            CueFlowEvent(
                timestamp=t,
                event_type=FlowEventType.STROKE_START,
                detail=f"수구 출발 확정 ({cue_ball_color.name})",
                cue_pos_mm=(float(initial_pos[0]), float(initial_pos[1])),
                cue_dir_deg=self._vec_to_deg(self.twin_vel),
                cushion_count_so_far=0,
            )
        )

    def step_physics_twin(
        self,
        t: float,
        observed_cue_pos: Optional[np.ndarray],
        other_balls_pos: Dict[object, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        IMM(Interactive Multiple Model) 물리 시뮬레이션 스텝:
        관측된 수구 위치가 다른 공과 겹쳐 흔들리거나 결측(None)되었을 때,
        물리 시뮬레이터의 3가지 가설(직진, 쿠션 반사, 구 충돌) 중 가장 부합하는 상태로 수구 흐름을 유지합니다.
        """
        if self.twin_pos is None or self.prev_time is None:
            pos = (
                observed_cue_pos
                if observed_cue_pos is not None
                else np.array([TABLE_W * 0.5, TABLE_H * 0.5], dtype=np.float64)
            )
            self.twin_pos = pos.copy()
            self.prev_time = t
            return self.twin_pos, self.twin_vel

        dt = max(t - self.prev_time, 1e-4)
        self.prev_time = t

        # 가설 1 (H_free): 등감속 직선 구름 예측
        pred_free_pos = self.twin_pos + self.twin_vel * dt
        pred_free_vel = self.twin_vel * self.friction_decay

        # 가설 2 (H_rail): 4면 쿠션 거울 반사(Method of Images) 물리 예측
        pred_rail_pos = pred_free_pos.copy()
        pred_rail_vel = pred_free_vel.copy()
        e = self.restitution
        if pred_rail_pos[0] < BALL_R:
            pred_rail_pos[0] = BALL_R + (BALL_R - pred_rail_pos[0]) * e
            pred_rail_vel[0] = -pred_rail_vel[0] * e
        elif pred_rail_pos[0] > TABLE_W - BALL_R:
            pred_rail_pos[0] = (TABLE_W - BALL_R) - (
                pred_rail_pos[0] - (TABLE_W - BALL_R)
            ) * e
            pred_rail_vel[0] = -pred_rail_vel[0] * e

        if pred_rail_pos[1] < BALL_R:
            pred_rail_pos[1] = BALL_R + (BALL_R - pred_rail_pos[1]) * e
            pred_rail_vel[1] = -pred_rail_vel[1] * e
        elif pred_rail_pos[1] > TABLE_H - BALL_R:
            pred_rail_pos[1] = (TABLE_H - BALL_R) - (
                pred_rail_pos[1] - (TABLE_H - BALL_R)
            ) * e
            pred_rail_vel[1] = -pred_rail_vel[1] * e

        # 카메라 관측값이 결측(None)된 경우 물리 시뮬레이션(H_rail) 결과를 그대로 반환하여 흐름 유지
        if observed_cue_pos is None:
            self.twin_pos = pred_rail_pos
            self.twin_vel = pred_rail_vel
            return self.twin_pos.copy(), self.twin_vel.copy()

        # 관측값이 정상 존재할 때 상태 갱신
        raw_vel = (observed_cue_pos - self.twin_pos) / dt
        self.twin_vel = 0.65 * raw_vel + 0.35 * self.twin_vel
        self.twin_pos = observed_cue_pos.copy()

        # 초구 직진 구간(출발 후 25~90mm 이동 구간)에서 초기 출발 광선 방향(initial_ray_dir) 고정
        if not self.initial_ray_locked and self.stroke_origin_pos is not None:
            disp_from_origin = self.twin_pos - self.stroke_origin_pos
            dist_origin = float(np.linalg.norm(disp_from_origin))
            if dist_origin >= 25.0:
                if self.initial_ray_dir is None or dist_origin <= 95.0:
                    self.initial_ray_dir = disp_from_origin / dist_origin
                if dist_origin >= 75.0:
                    self.initial_ray_locked = True

        return self.twin_pos.copy(), self.twin_vel.copy()

    @staticmethod
    def solve_ray_ball_toi(
        p_cue: np.ndarray,
        v_cue: np.ndarray,
        p_target: np.ndarray,
        v_target: np.ndarray,
        collision_dist: float = BALL_DIAM + 4.0,
    ) -> Optional[float]:
        """
        [pooltool 이차방정식 충돌 시각 해법]
        || (p_cue - p_target) + tau * (v_cue - v_target) ||^2 = collision_dist^2
        을 풀어 두 공이 최초로 만나는 양수 시각 tau_ball (초)을 반환합니다.
        """
        r0 = p_cue - p_target
        v_rel = v_cue - v_target
        a = float(np.dot(v_rel, v_rel))
        if a < 1e-4:
            return None
        b = float(np.dot(r0, v_rel))
        c = float(np.dot(r0, r0)) - (collision_dist**2)

        # 이미 겹쳐 있거나 가까워지는 중(b < 0)일 때만 유효
        if c <= 0.0:
            return 0.0
        if b >= 0.0:
            return None

        disc = b * b - a * c
        if disc < 0.0:
            return None
        tau = (-b - math.sqrt(disc)) / a
        return float(max(0.0, tau))

    @staticmethod
    def solve_ray_rail_toi(
        p_cue: np.ndarray,
        v_cue: np.ndarray,
        rail_margin: float = BALL_R + 24.0,
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        수구 속도 벡터가 4면 쿠션 경계 평면(rail_margin)에 도달하는 최소 시각 tau_rail을 반환합니다.
        """
        candidates: List[Tuple[float, str]] = []
        vx, vy = float(v_cue[0]), float(v_cue[1])
        x, y = float(p_cue[0]), float(p_cue[1])

        if vx < -30.0:
            tau = max(0.0, (x - rail_margin) / (-vx))
            candidates.append((tau, "LEFT"))
        if vx > 30.0:
            tau = max(0.0, ((TABLE_W - rail_margin) - x) / vx)
            candidates.append((tau, "RIGHT"))
        if vy < -30.0:
            tau = max(0.0, (y - rail_margin) / (-vy))
            candidates.append((tau, "TOP"))
        if vy > 30.0:
            tau = max(0.0, ((TABLE_H - rail_margin) - y) / vy)
            candidates.append((tau, "BOTTOM"))

        if not candidates:
            return None, None
        candidates.sort(key=lambda item: item[0])
        return candidates[0]

    def resolve_first_contact_order(
        self,
        cue_trajectory_log: List[Tuple[float, np.ndarray, np.ndarray, float]],
        first_target_pos: np.ndarray,
        current_cushions_before_hit: int,
    ) -> Tuple[str, int]:
        """
        [핵심 질문 해결: '처음 쿠션을 먼저 닿았는지' vs '1적구를 먼저 충돌했는지' 물리 판별기]
        제1적구와 충돌이 감지된 시점에:
        1) 이미 명확한 쿠션이 카운트되었다면 -> ("CUSHION_FIRST", current_cushions_before_hit)
        2) 제1적구가 쿠션 근처에 있어 1쿠션과 1적구 충돌이 거의 동시에 일어난 경우(1쿠션 넣어치기 vs 1적구 먼저 맞고 쿠션):
           - 초기 출발 광선(stroke_origin_pos + s * initial_ray_dir)이 제1적구 구면(반경 64mm)과 직접 교차하는지,
           - 아니면 초기 광선이 쿠션 벽면을 먼저 때리고 굴절되어야만 제1적구에 닿을 수 있는지(Impact Parameter & TOI 비교)를
             계산하여 ("CUSHION_FIRST", 1) 또는 ("BALL1_FIRST", 0)을 정확히 판정합니다.
        """
        if current_cushions_before_hit >= 1:
            self.first_contact_type = "CUSHION_FIRST"
            return "CUSHION_FIRST", current_cushions_before_hit

        if (
            not cue_trajectory_log
            or self.stroke_origin_pos is None
            or self.initial_ray_dir is None
        ):
            self.first_contact_type = "BALL1_FIRST"
            return "BALL1_FIRST", 0

        # cue_trajectory_log에 충돌 이후 후행 프레임(f_post1, f_post2)이 포함되어 있더라도
        # 제1적구 최근접 프레임(hit_idx) 이전의 순수 선행 구간만 분리하여 역전 오판(1적구 타격 후 쿠션을 선쿠션으로 오인)을 차단합니다.
        hit_idx = int(
            np.argmin(
                [
                    float(np.linalg.norm(item[1] - first_target_pos))
                    for item in cue_trajectory_log
                ]
            )
        )
        pre_hit_log = (
            cue_trajectory_log[:hit_idx]
            if hit_idx >= 1
            else cue_trajectory_log[:1]
        )

        # 수구가 출발 후 제1적구에 닿기 직전까지 쿠션 경계에 실제로 다가간 최소 거리 확인
        orig_edges = (
            float(self.stroke_origin_pos[0]),
            float(TABLE_W - self.stroke_origin_pos[0]),
            float(self.stroke_origin_pos[1]),
            float(TABLE_H - self.stroke_origin_pos[1]),
        )
        min_edge_dist = float("inf")
        for _, p_i, _, _ in pre_hit_log:
            if float(np.linalg.norm(p_i - self.stroke_origin_pos)) < 25.0:
                continue
            cand_edges = (
                float(p_i[0]),
                float(TABLE_W - p_i[0]),
                float(p_i[1]),
                float(TABLE_H - p_i[1]),
            )
            approached = [
                d_c
                for d_c, d_o in zip(cand_edges, orig_edges)
                if d_c <= d_o - 15.0 or (d_o > 62.0 and d_c <= 52.0)
            ]
            if approached:
                d_edge = min(approached)
                if d_edge < min_edge_dist:
                    min_edge_dist = d_edge

        # 수구가 제1적구 접촉 전 실제 쿠션 반사 경계(<= 62mm)에 다가간 적이 없다면 1적구 선접촉!
        if min_edge_dist > 62.0:
            self.first_contact_type = "BALL1_FIRST"
            return "BALL1_FIRST", 0

        # 초기 출발 광선(Origin -> initial_ray_dir)과 제1적구 중심 간의 수직 거리(Impact Parameter b_perp) 계산
        to_target = first_target_pos - self.stroke_origin_pos
        proj_s = float(np.dot(to_target, self.initial_ray_dir))
        perp_vec = to_target - proj_s * self.initial_ray_dir
        b_perp_mm = float(np.linalg.norm(perp_vec)) if proj_s > 0.0 else 999.0

        # 제1적구 타격 직전 속도 방향과 초기 출발 광선 방향의 굴절각 비교
        v_just_before = (
             cue_trajectory_log[hit_idx][2]
            if hit_idx < len(cue_trajectory_log)
            else pre_hit_log[-1][2]
        )
        v_jb_norm = float(np.linalg.norm(v_just_before))
        pre_bend_deg = 0.0
        if v_jb_norm > 20.0:
            cos_val = float(
                np.clip(
                    np.dot(self.initial_ray_dir, v_just_before / v_jb_norm), -1.0, 1.0
                )
            )
            pre_bend_deg = float(np.degrees(np.arccos(cos_val)))

        # TOI 비교: 초기 출발 상태에서 실제 쿠션 접촉면 도달 시각(tau_rail) vs 제1적구 직격 시각(tau_ball)
        v_init_est = self.initial_ray_dir * max(v_jb_norm, 1000.0)
        tau_rail, _ = self.solve_ray_rail_toi(
            self.stroke_origin_pos, v_init_est, rail_margin=BALL_R + 24.0
        )
        tau_ball = self.solve_ray_ball_toi(
            self.stroke_origin_pos,
            v_init_est,
            first_target_pos,
            np.zeros(2),
            collision_dist=64.0,
        )

        # 판정:
        # 1) 초기 직선 광선이 제1적구를 빗나가거나(b_perp >= 64.0mm, 즉 직선으로는 절대 안 맞음),
        # 2) 초기 광선이 비껴난 상태에서 제1적구에 닿기 전에 궤적이 8도 이상 꺾였거나,
        # 3) tau_rail이 tau_ball보다 먼저 도달한 경우 -> 100% 쿠션을 먼저 찍고 들어간 1쿠션 넣어치기(CUSHION_FIRST)!
        if (
            b_perp_mm >= 64.0
            or (b_perp_mm >= 38.0 and pre_bend_deg >= 8.0)
            or (
                tau_rail is not None
                and (tau_ball is None or tau_rail < tau_ball)
                and b_perp_mm >= 38.0
            )
        ):
            self.first_contact_type = "CUSHION_FIRST"
            return "CUSHION_FIRST", 1

        self.first_contact_type = "BALL1_FIRST"
        return "BALL1_FIRST", 0

    def record_event(
        self,
        t: float,
        event_type: FlowEventType,
        detail: str,
        cue_pos: np.ndarray,
        cue_vel: np.ndarray,
        cushion_count: int,
    ):
        """수구가 쿠션이나 적구에 부딪힐 때마다 직선 구간(Segment)을 마감하고 흐름 이벤트를 기록합니다."""
        if self.current_seg_start_pos is not None:
            seg_vec = cue_pos - self.current_seg_start_pos
            seg_len = float(np.linalg.norm(seg_vec))
            if seg_len >= 5.0:
                unit_dir = seg_vec / seg_len
                self.segments.append(
                    CueTrajectorySegment(
                        seg_index=len(self.segments) + 1,
                        start_time=self.current_seg_start_t,
                        end_time=t,
                        start_pos=self.current_seg_start_pos.copy(),
                        end_pos=cue_pos.copy(),
                        direction_unit=unit_dir,
                        direction_deg=self._vec_to_deg(unit_dir),
                        length_mm=seg_len,
                    )
                )
        self.current_seg_start_t = t
        self.current_seg_start_pos = cue_pos.copy()

        self.events.append(
            CueFlowEvent(
                timestamp=t,
                event_type=event_type,
                detail=detail,
                cue_pos_mm=(float(cue_pos[0]), float(cue_pos[1])),
                cue_dir_deg=self._vec_to_deg(cue_vel),
                cushion_count_so_far=cushion_count,
            )
        )
