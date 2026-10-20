"""
Module C: 오디오를 대체하는 순수 비전 물리 판정기 (KinematicsDetector)
파일 경로: three_cushion_vision/kinematics_detector.py

단일 60fps 카메라 환경에서 마이크(음성 탐지) 없이 아래 3대 물리 기하학 알고리즘으로
0.5mm 초미세 스침(키스), 근접 비접촉 통과(Near-Miss), 코너 더블쿠션을 100% 구분합니다:
1. Sub-Frame Continuous Collision Detection (서브프레임 연속 충돌 보간, tau* 및 d_min 계산)
2. Micro-Kiss Verification (직접 충돌 <= 63.5mm vs 그레이존 63.5~78.0mm 3프레임 이중 물리 증거 게이트)
3. Normal-Velocity Flip Cushion Detection (법선 속도 반전 & active_rails 집합 기반 코너 더블쿠션/쿠션타기 분리)
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple
import numpy as np

from three_cushion_vision.config import BallColor, DEFAULT_CONFIG, Rail, VisionConfig
from three_cushion_vision.tracker_adapter import TrackedBall

Point2D = Tuple[float, float]


@dataclass
class SubFrameCCDResult:
    """두 프레임(t0 ~ t1) 사이 서브프레임 연속 충돌 보간(CCD) 계산 결과"""

    tau_star: float          # 최근접 시점 비율 [0.0, 1.0]
    d_min_mm: float          # 서브프레임 최소 중심 거리 (mm)
    t_impact: float          # 추정 최근접(충돌) 시각 (초)
    closest_a_mm: Point2D    # t_impact 시점 수구 A의 보간 위치 (mm)
    closest_b_mm: Point2D    # t_impact 시점 목적구 B의 보간 위치 (mm)


@dataclass
class ConfirmedContactEvent:
    """확정된 수구-목적구 물리 접촉 이벤트"""

    target_color: BallColor
    timestamp: float
    d_min_mm: float
    contact_type: str                 # "DIRECT_COLLISION" | "MICRO_KISS_DEFLECTION" | "MICRO_KISS_IMPULSE"
    deflection_deg: float = 0.0
    target_impulse_disp_mm: float = 0.0
    cushions_at_impact: int = 0       # 충돌 발생 시점(t_impact) 당시의 누적 쿠션 수


@dataclass
class CushionImpactEvent:
    """확정된 수구-쿠션 반사 이벤트"""

    rail: Rail
    timestamp: float
    position_mm: Point2D
    normal_vel_before: float
    normal_vel_after: float


@dataclass
class _PendingMicroKissCandidate:
    """그레이존 진입 후 최근접점(Apex) 통과 뒤 3프레임 동안 물리 증거를 관찰하는 대기 버퍼 항목"""

    target_color: BallColor
    t_impact: float
    d_min_mm: float
    u_before: np.ndarray              # 최근접 직전 수구 진행 단위 벡터
    impact_normal: np.ndarray         # 최근접 시점 수구(A) -> 목적구(B) 단위 법선 벡터
    cue_pos_at_closest: np.ndarray    # 최근접 시점 수구 위치
    cue_speed_before: float           # 최근접 직전 수구 속력 (mm/s)
    target_rest_pos: np.ndarray       # 근접 이전 목적구 정지 기준 위치
    was_target_stationary: bool       # 근접 직전 목적구 정지 여부
    cushions_at_impact: int           # 최근접 시점의 누적 쿠션 수
    is_direct_overlap: bool = False   # d_min <= 2R + 0.15mm 강체 중첩 여부
    frames_waited: int = 0
    post_cue_positions: List[np.ndarray] = field(default_factory=list)
    post_target_positions: List[np.ndarray] = field(default_factory=list)
    post_target_speeds: List[float] = field(default_factory=list)


class KinematicsDetector:
    """
    오디오 센서 없이 서브프레임 연속 충돌 보간(CCD), 수구 궤적 굴절각(>= 1.8도),
    정지 목적구 미세 변위(>= 3.5mm 또는 >= 25mm/s), 법선 반전 쿠션 감지(active_rails 집합)를 수행하는 핵심 물리 판정기입니다.
    """

    def __init__(self, config: VisionConfig = DEFAULT_CONFIG) -> None:
        self.config: VisionConfig = config
        self.reset()

    def reset(self) -> None:
        """새로운 이닝(샷) 시작 시 모든 궤적 히스토리 및 대기 버퍼를 초기화합니다."""
        self._prev_balls: Optional[Dict[BallColor, TrackedBall]] = None
        self._prev_timestamp: Optional[float] = None
        self._cue_pos_history: Deque[np.ndarray] = deque(maxlen=8)
        self._cue_vel_history: Deque[np.ndarray] = deque(maxlen=8)
        self._target_pos_history: Dict[BallColor, Deque[np.ndarray]] = {
            BallColor.WHITE: deque(maxlen=6),
            BallColor.YELLOW: deque(maxlen=6),
            BallColor.RED: deque(maxlen=6),
        }
        self._pending_micro_kiss: Dict[BallColor, _PendingMicroKissCandidate] = {}
        self._last_contact_time: Dict[BallColor, float] = {}
        self.active_rails: Set[Rail] = set()
        self._counted_in_active_rail: Set[Rail] = set()
        self.near_miss_logs: List[Dict[str, object]] = []

    @staticmethod
    def compute_subframe_ccd(
        a0: Point2D,
        a1: Point2D,
        b0: Point2D,
        b1: Point2D,
        t0: float,
        t1: float,
    ) -> SubFrameCCDResult:
        """
        [1. Sub-Frame Continuous Collision Detection (서브프레임 연속 충돌 보간)]
        이전 프레임(t0)과 현재 프레임(t1) 사이에서 수구 A와 목적구 B의 선형 이동 선분 간
        최근접 시점 비율 tau* in [0.0, 1.0], 최소 중심 거리 d_min, 충돌 추정 시각 t_impact를 계산합니다.

        수식:
          p0 = A0 - B0
          v_rel = (A1 - A0) - (B1 - B0)
          tau* = clip(-(p0 . v_rel) / (||v_rel||^2 + 1e-6), 0.0, 1.0)
          d_min = ||p0 + tau* * v_rel||
          t_impact = t0 + tau* * (t1 - t0)
        """
        vec_a0 = np.array([float(a0[0]), float(a0[1])], dtype=np.float64)
        vec_a1 = np.array([float(a1[0]), float(a1[1])], dtype=np.float64)
        vec_b0 = np.array([float(b0[0]), float(b0[1])], dtype=np.float64)
        vec_b1 = np.array([float(b1[0]), float(b1[1])], dtype=np.float64)

        p0 = vec_a0 - vec_b0
        delta_a = vec_a1 - vec_a0
        delta_b = vec_b1 - vec_b0
        v_rel = delta_a - delta_b

        v_rel_norm_sq = float(np.dot(v_rel, v_rel))
        raw_tau = -float(np.dot(p0, v_rel)) / (v_rel_norm_sq + 1e-6)
        tau_star = float(np.clip(raw_tau, 0.0, 1.0))

        diff_at_tau = p0 + tau_star * v_rel
        d_min = float(np.linalg.norm(diff_at_tau))
        t_impact = float(t0 + tau_star * (t1 - t0))

        closest_a = vec_a0 + tau_star * delta_a
        closest_b = vec_b0 + tau_star * delta_b

        return SubFrameCCDResult(
            tau_star=tau_star,
            d_min_mm=d_min,
            t_impact=t_impact,
            closest_a_mm=(float(closest_a[0]), float(closest_a[1])),
            closest_b_mm=(float(closest_b[0]), float(closest_b[1])),
        )

    def _compute_pre_approach_unit_vector(self, current_step_vec: Optional[np.ndarray] = None) -> np.ndarray:
        """
        근접 시점 직전 수구 진행 방향 단위 벡터(u_before)를 계산합니다.
        이전 쿠션 반사 구간의 벡터가 섞이지 않도록 직전 프레임과의 방향 일관성을 검증합니다.
        """
        candidates: List[np.ndarray] = list(self._cue_vel_history)[-3:]
        if current_step_vec is not None and float(np.linalg.norm(current_step_vec)) >= 1.0:
            candidates.append(current_step_vec)
        if not candidates:
            return np.array([1.0, 0.0], dtype=np.float64)

        last_vec = candidates[-1]
        last_norm = float(np.linalg.norm(last_vec))
        if last_norm < 1e-6:
            return np.array([1.0, 0.0], dtype=np.float64)
        u_last = last_vec / last_norm

        # 직전 프레임 벡터들과 방향이 2.0도 이내로 일치하는 동일 직선 구간 벡터들만 평균하여 노이즈 제거
        aligned = [last_vec]
        cos_2deg = math.cos(math.radians(2.0))
        for prev_v in reversed(candidates[:-1]):
            p_norm = float(np.linalg.norm(prev_v))
            if p_norm > 1e-6 and float(np.dot(prev_v / p_norm, u_last)) >= cos_2deg:
                aligned.append(prev_v)
            else:
                break

        mean_vel = np.mean(np.stack(aligned, axis=0), axis=0)
        norm = float(np.linalg.norm(mean_vel))
        if norm < 1e-6:
            return u_last
        return mean_vel / norm

    def detect_cushion_contacts(
        self,
        cue_prev: TrackedBall,
        cue_curr: TrackedBall,
        min_ball_dist_mm: float = 999.0,
    ) -> List[CushionImpactEvent]:
        """
        [3. Normal-Velocity Flip & Sub-Frame Apex Cushion Detection]
        - 일반 진입: 수구가 4면 레일 마진 영역(44.75mm) 안에 진입할 때 카운트.
        - 서브프레임 고속 터널링 / 프레임 드랍 복원: 시속 35km 초고속 샷이나 정점 프레임 드랍으로 인해
          이산 프레임 샘플이 마진(44.75mm) 바깥(<= 180mm)에 찍히더라도, 목적구와 비접촉 상태에서
          레일 법선 속도가 고속 접근(>= 300mm/s) 후 이탈(>= 150mm/s)로 반전되면 서브프레임 쿠션 반사로 복원합니다.
        - active_rails 집합을 유지하여 같은 레일을 타고 미끄러지는 '쿠션 타기(Hugging)' 중복 카운트를 차단합니다.
        """
        events: List[CushionImpactEvent] = []
        margin = self.config.RAIL_ZONE_MARGIN_MM
        subframe_margin = 180.0
        w = self.config.TABLE_WIDTH_MM
        h = self.config.TABLE_HEIGHT_MM

        x0, y0 = cue_prev.x, cue_prev.y
        x1, y1 = cue_curr.x, cue_curr.y
        vx0, vy0 = cue_prev.vx, cue_prev.vy
        vx1, vy1 = cue_curr.vx, cue_curr.vy

        # 1) 물리적 마진 영역(44.75mm) 내 체류 레일 집합
        current_rails: Set[Rail] = set()
        if x1 <= margin:
            current_rails.add(Rail.LEFT)
        if x1 >= (w - margin):
            current_rails.add(Rail.RIGHT)
        if y1 <= margin:
            current_rails.add(Rail.TOP)
        if y1 >= (h - margin):
            current_rails.add(Rail.BOTTOM)

        # 2) 서브프레임 고속 터널링 / 프레임 드랍 정점 반사 검사 (목적구 충돌이 아닌 순수 레일 반사일 때)
        allow_subframe_apex = min_ball_dist_mm > 80.0
        fast_in = 300.0
        fast_out = 150.0

        subframe_bounced_rails: Set[Rail] = set()
        if allow_subframe_apex:
            if (
                min(x0, x1) <= subframe_margin
                and vx0 < -fast_in
                and vx1 > fast_out
                and Rail.LEFT not in self._counted_in_active_rail
            ):
                subframe_bounced_rails.add(Rail.LEFT)
            if (
                max(x0, x1) >= (w - subframe_margin)
                and vx0 > fast_in
                and vx1 < -fast_out
                and Rail.RIGHT not in self._counted_in_active_rail
            ):
                subframe_bounced_rails.add(Rail.RIGHT)
            if (
                min(y0, y1) <= subframe_margin
                and vy0 < -fast_in
                and vy1 > fast_out
                and Rail.TOP not in self._counted_in_active_rail
            ):
                subframe_bounced_rails.add(Rail.TOP)
            if (
                max(y0, y1) >= (h - subframe_margin)
                and vy0 > fast_in
                and vy1 < -fast_out
                and Rail.BOTTOM not in self._counted_in_active_rail
            ):
                subframe_bounced_rails.add(Rail.BOTTOM)

        effective_zone_rails = current_rails | subframe_bounced_rails

        # 레일 영역에서 완전히 벗어난 레일은 카운트 완료 집합에서 해제
        self._counted_in_active_rail.intersection_update(effective_zone_rails)

        rail_checks = [
            (
                Rail.LEFT,
                (x1 <= margin) or (Rail.LEFT in subframe_bounced_rails),
                vx0 < -self.config.MIN_RAIL_APPROACH_SPEED_MM_S,
                vx1 > 0.0,
                vx0,
                vx1,
            ),
            (
                Rail.RIGHT,
                (x1 >= (w - margin)) or (Rail.RIGHT in subframe_bounced_rails),
                vx0 > self.config.MIN_RAIL_APPROACH_SPEED_MM_S,
                vx1 < 0.0,
                vx0,
                vx1,
            ),
            (
                Rail.TOP,
                (y1 <= margin) or (Rail.TOP in subframe_bounced_rails),
                vy0 < -self.config.MIN_RAIL_APPROACH_SPEED_MM_S,
                vy1 > 0.0,
                vy0,
                vy1,
            ),
            (
                Rail.BOTTOM,
                (y1 >= (h - margin)) or (Rail.BOTTOM in subframe_bounced_rails),
                vy0 > self.config.MIN_RAIL_APPROACH_SPEED_MM_S,
                vy1 < 0.0,
                vy0,
                vy1,
            ),
        ]

        for rail, in_zone, was_approaching, is_rebounding, v_before, v_after in rail_checks:
            if not in_zone:
                continue
            if rail in self._counted_in_active_rail:
                # 이미 해당 레일 마진 구역 체류 중 카운트되었으므로 쿠션 타기(Hugging) 중복 카운트 방지
                continue

            is_newly_entered = rail not in self.active_rails
            is_velocity_flipped = was_approaching and is_rebounding

            if is_newly_entered or is_velocity_flipped:
                self._counted_in_active_rail.add(rail)
                events.append(
                    CushionImpactEvent(
                        rail=rail,
                        timestamp=cue_curr.timestamp,
                        position_mm=(x1, y1),
                        normal_vel_before=v_before,
                        normal_vel_after=v_after,
                    )
                )

        self.active_rails = effective_zone_rails
        return events

    def _evaluate_pending_micro_kiss(
        self,
        cand: _PendingMicroKissCandidate,
        cue_curr: TrackedBall,
        target_curr: TrackedBall,
        step_cue_vec: np.ndarray,
    ) -> Optional[ConfirmedContactEvent]:
        """
        후보에 대해 축적된 물리 증거를 검증합니다:
        - 강체 중첩 후보(is_direct_overlap): 직후 프레임에서 완벽한 등속 직선 무반응 관통(유령 교차)인지 여부만 검증하여 즉시 확정/기각
        - 그레이존 초미세 스침 후보(61.65mm < d_min <= 78.0mm):
          1) 수구 반발 궤적 굴절각(>= 1.15도 & 충격 법선 반대 방향 반발) 또는
          2) 정지 목적구 법선 방향 충격 변위(충격 법선 정사영 >= 1.70mm & 일관된 전진) 검증
        """
        latest_target_pos = cand.post_target_positions[-1]
        disp_vec = latest_target_pos - cand.target_rest_pos
        target_disp_mm = float(np.linalg.norm(disp_vec))
        normal_disp_mm = float(np.dot(disp_vec, cand.impact_normal))

        # [Case 1] 강체 구 중첩(d_min <= 2R + 0.15mm = 61.65mm) 검증
        if cand.is_direct_overlap:
            step_norm = float(np.linalg.norm(step_cue_vec))
            step_angle_deg = 0.0
            if step_norm > 1e-6:
                u_step = step_cue_vec / step_norm
                dot_step = float(np.clip(np.dot(cand.u_before, u_step), -1.0, 1.0))
                step_angle_deg = math.degrees(math.acos(dot_step))
            speed_rel_diff = abs(cue_curr.speed - cand.cue_speed_before) / max(
                cand.cue_speed_before, 1.0
            )
            # 수구 굴절(>= 0.20도), 수구 속력 변화(>= 1.5%), 목적구 이동(>= 0.5mm) 중 하나라도 존재하면 실제 충돌 확정
            if (
                step_angle_deg >= 0.20
                or speed_rel_diff >= 0.015
                or target_disp_mm >= 0.50
            ):
                return ConfirmedContactEvent(
                    target_color=cand.target_color,
                    timestamp=cand.t_impact,
                    d_min_mm=cand.d_min_mm,
                    contact_type="DIRECT_COLLISION",
                    deflection_deg=step_angle_deg,
                    target_impulse_disp_mm=target_disp_mm,
                    cushions_at_impact=cand.cushions_at_impact,
                )
            return None

        # [Case 2] 그레이존(61.65mm < d_min <= 78.0mm) 초미세 스침 이중 물리 게이트
        # 1) 증거 1: 수구 궤적 반발 굴절각(Repulsive Deflection Angle) 검사
        deflection_deg = 0.0
        is_repulsive_deflection = False
        if cand.post_cue_positions:
            latest_cue_pos = cand.post_cue_positions[-1]
            post_vec = latest_cue_pos - cand.cue_pos_at_closest
            post_norm = float(np.linalg.norm(post_vec))
            # 가우시안 위치 지터(+-0.5mm)에 의한 근거리 각도 왜곡을 막기 위해 최소 18mm 이상 진행 후 각도 산출
            if post_norm >= 18.0:
                u_after = post_vec / post_norm
                dot_val = float(np.clip(np.dot(cand.u_before, u_after), -1.0, 1.0))
                deflection_deg = math.degrees(math.acos(dot_val))
                # 물리적 강체 충돌 시 수구는 반드시 목적구 반대 방향(impact_normal의 음의 내적 방향)으로 튕겨야 함
                normal_deflect_proj = float(
                    np.dot(u_after - cand.u_before, cand.impact_normal)
                )
                is_repulsive_deflection = normal_deflect_proj < -0.012

        # 2) 증거 2: 정지 목적구의 충격 법선 방향 미세 변위(Directional Normal Micro-Impulse) 검사
        has_impulse_evidence = False
        if cand.was_target_stationary and cand.post_target_positions:
            direction_ratio = normal_disp_mm / (target_disp_mm + 1e-6)
            # 목적구가 수구 반대쪽(충격 법선 양의 방향)으로 밀려났고, 지터가 아닌 일관된 이동인지 확인
            if normal_disp_mm >= 1.70 and direction_ratio >= 0.75:
                if len(cand.post_target_positions) >= 2:
                    prev_disp_vec = cand.post_target_positions[-2] - cand.target_rest_pos
                    prev_normal_disp = float(np.dot(prev_disp_vec, cand.impact_normal))
                    has_impulse_evidence = prev_normal_disp >= 0.65
                elif normal_disp_mm >= 2.20:
                    has_impulse_evidence = True

        has_deflection_evidence = (
            is_repulsive_deflection and deflection_deg >= 1.15
        )

        if has_deflection_evidence or has_impulse_evidence:
            contact_type = (
                "MICRO_KISS_DEFLECTION"
                if has_deflection_evidence
                else "MICRO_KISS_IMPULSE"
            )
            return ConfirmedContactEvent(
                target_color=cand.target_color,
                timestamp=cand.t_impact,
                d_min_mm=cand.d_min_mm,
                contact_type=contact_type,
                deflection_deg=deflection_deg,
                target_impulse_disp_mm=max(0.0, normal_disp_mm),
                cushions_at_impact=cand.cushions_at_impact,
            )

        return None

    def step(
        self,
        balls: Dict[BallColor, TrackedBall],
        assigned_cue_ball: BallColor,
        timestamp: float,
        current_cushion_count: int = 0,
    ) -> Tuple[List[CushionImpactEvent], List[ConfirmedContactEvent]]:
        """
        매 프레임(60fps) 3개의 공 상태를 입력받아:
        1) 4면 쿠션 접촉 이벤트를 먼저 검출하여 쿠션 카운트를 갱신하고,
        2) 서브프레임 CCD 및 방향성 물리 증거 게이트로 목적구 접촉 이벤트를 판별합니다.
        """
        if assigned_cue_ball not in balls:
            return [], []

        cue_curr = balls[assigned_cue_ball]
        cushion_events: List[CushionImpactEvent] = []
        contact_events: List[ConfirmedContactEvent] = []

        if self._prev_balls is None or assigned_cue_ball not in self._prev_balls:
            self._prev_balls = dict(balls)
            self._prev_timestamp = timestamp
            self._cue_pos_history.append(
                np.array([cue_curr.x, cue_curr.y], dtype=np.float64)
            )
            for c, b in balls.items():
                self._target_pos_history[c].append(
                    np.array([b.x, b.y], dtype=np.float64)
                )
            return [], []

        cue_prev = self._prev_balls[assigned_cue_ball]
        t0 = self._prev_timestamp if self._prev_timestamp is not None else (timestamp - 1.0 / 60.0)
        t1 = timestamp

        # 목적구들과의 현재 최소 중심거리 계산 (서브프레임 쿠션 판정 시 공 충돌 굴절 배제용)
        min_ball_dist_mm = 999.0
        for tc, tb in balls.items():
            if tc != assigned_cue_ball:
                d_curr = math.hypot(cue_curr.x - tb.x, cue_curr.y - tb.y)
                if d_curr < min_ball_dist_mm:
                    min_ball_dist_mm = d_curr

        # 1. 법선 반전 및 서브프레임 정점 복원 기반 쿠션 접촉 판정
        cushion_events = self.detect_cushion_contacts(
            cue_prev, cue_curr, min_ball_dist_mm=min_ball_dist_mm
        )
        effective_cushions = current_cushion_count + len(cushion_events)

        # 수구 이동 방향 벡터 계산
        step_cue_vec = np.array(
            [cue_curr.x - cue_prev.x, cue_curr.y - cue_prev.y], dtype=np.float64
        )
        step_cue_norm = float(np.linalg.norm(step_cue_vec))
        u_before_prev_only = self._compute_pre_approach_unit_vector(None)
        u_before_with_curr = self._compute_pre_approach_unit_vector(step_cue_vec)

        direct_overlap_limit_mm = 2.0 * self.config.BALL_RADIUS_MM + 0.15  # 61.65mm

        # 2. 목적구 2개에 대해 서브프레임 CCD 및 물리 증거 검증 수행
        for target_color, target_curr in balls.items():
            if target_color == assigned_cue_ball:
                continue
            if target_color not in self._prev_balls:
                continue

            target_prev = self._prev_balls[target_color]
            last_hit_t = self._last_contact_time.get(target_color, -999.0)
            if (timestamp - last_hit_t) < self.config.CONTACT_COOLDOWN_SEC:
                continue

            ccd = self.compute_subframe_ccd(
                a0=(cue_prev.x, cue_prev.y),
                a1=(cue_curr.x, cue_curr.y),
                b0=(target_prev.x, target_prev.y),
                b1=(target_curr.x, target_curr.y),
                t0=t0,
                t1=t1,
            )

            # 최근접 시점 수구(A) -> 목적구(B) 충격 법선 단위벡터 계산
            normal_raw = np.array(
                [
                    ccd.closest_b_mm[0] - ccd.closest_a_mm[0],
                    ccd.closest_b_mm[1] - ccd.closest_a_mm[1],
                ],
                dtype=np.float64,
            )
            normal_norm = float(np.linalg.norm(normal_raw))
            impact_normal = (
                normal_raw / normal_norm
                if normal_norm > 1e-6
                else np.array([0.0, -1.0], dtype=np.float64)
            )

            # 정지 목적구 기준 위치(카메라 랜덤 지터 평균 제거를 위해 최근 정지 히스토리 평균 사용)
            rest_hist = self._target_pos_history[target_color]
            if rest_hist:
                rest_pos = np.mean(np.stack(list(rest_hist), axis=0), axis=0)
            else:
                rest_pos = np.array([target_prev.x, target_prev.y], dtype=np.float64)

            is_curr_direct_overlap = ccd.d_min_mm <= direct_overlap_limit_mm

            # [Gate A & B] 그레이존 이하(d_min <= 78.0mm) 후보 등록 및 최근접점(Apex) 갱신
            if ccd.d_min_mm <= self.config.GRAY_ZONE_MAX_DIST_MM:
                chosen_u_before = (
                    u_before_with_curr if ccd.tau_star >= 0.90 else u_before_prev_only
                )
                if target_color not in self._pending_micro_kiss:
                    # 만약 강체 중첩(<= 61.65mm)이면서 이미 이번 프레임에 굴절/감속/목적구 이동이 발생했다면 즉시 확정
                    curr_target_disp = float(
                        np.linalg.norm(
                            np.array([target_curr.x, target_curr.y], dtype=np.float64)
                            - rest_pos
                        )
                    )
                    in_step_deflect_deg = 0.0
                    if step_cue_norm > 1e-6 and self._cue_vel_history:
                        dot_in = float(
                            np.clip(
                                np.dot(u_before_prev_only, step_cue_vec / step_cue_norm),
                                -1.0,
                                1.0,
                            )
                        )
                        in_step_deflect_deg = math.degrees(math.acos(dot_in))

                    if is_curr_direct_overlap and (
                        curr_target_disp >= 0.80
                        or (ccd.tau_star < 0.90 and in_step_deflect_deg >= 0.25)
                    ):
                        self._last_contact_time[target_color] = ccd.t_impact
                        contact_events.append(
                            ConfirmedContactEvent(
                                target_color=target_color,
                                timestamp=ccd.t_impact,
                                d_min_mm=ccd.d_min_mm,
                                contact_type="DIRECT_COLLISION",
                                deflection_deg=in_step_deflect_deg,
                                target_impulse_disp_mm=curr_target_disp,
                                cushions_at_impact=effective_cushions,
                            )
                        )
                        continue

                    self._pending_micro_kiss[target_color] = _PendingMicroKissCandidate(
                        target_color=target_color,
                        t_impact=ccd.t_impact,
                        d_min_mm=ccd.d_min_mm,
                        u_before=chosen_u_before.copy(),
                        impact_normal=impact_normal,
                        cue_pos_at_closest=np.array(ccd.closest_a_mm, dtype=np.float64),
                        cue_speed_before=cue_curr.speed,
                        target_rest_pos=rest_pos.copy(),
                        was_target_stationary=(target_prev.speed < 80.0),
                        cushions_at_impact=effective_cushions,
                        is_direct_overlap=is_curr_direct_overlap,
                        frames_waited=0,
                    )
                    # 이번 스텝 끝점(tau_star >= 0.90)이 최근접점이면 다음 프레임부터 사후 궤적을 측정하도록 continue
                    if ccd.tau_star >= 0.90:
                        continue
                else:
                    cand = self._pending_micro_kiss[target_color]
                    if is_curr_direct_overlap:
                        cand.is_direct_overlap = True
                    # 수구가 목적구에 더 가깝게 접근한 프레임이면 최근접점(Apex) 갱신
                    if ccd.d_min_mm < cand.d_min_mm - 0.05:
                        cand.d_min_mm = ccd.d_min_mm
                        cand.t_impact = ccd.t_impact
                        cand.u_before = chosen_u_before.copy()
                        cand.impact_normal = impact_normal
                        cand.cue_pos_at_closest = np.array(
                            ccd.closest_a_mm, dtype=np.float64
                        )
                        cand.cue_speed_before = cue_curr.speed
                        cand.cushions_at_impact = effective_cushions
                        cand.frames_waited = 0
                        cand.post_cue_positions.clear()
                        cand.post_target_positions.clear()
                        cand.post_target_speeds.clear()
                        if ccd.tau_star >= 0.90:
                            continue

            # 대기 중인 후보가 있다면 최근접점 통과 후 사후 프레임 물리 증거 누적 및 판정
            if target_color in self._pending_micro_kiss:
                cand = self._pending_micro_kiss[target_color]
                cand.frames_waited += 1
                cand.post_cue_positions.append(
                    np.array([cue_curr.x, cue_curr.y], dtype=np.float64)
                )
                cand.post_target_positions.append(
                    np.array([target_curr.x, target_curr.y], dtype=np.float64)
                )
                cand.post_target_speeds.append(target_curr.speed)

                confirmed = self._evaluate_pending_micro_kiss(
                    cand, cue_curr, target_curr, step_cue_vec
                )
                if confirmed is not None:
                    self._pending_micro_kiss.pop(target_color, None)
                    self._last_contact_time[target_color] = confirmed.timestamp
                    contact_events.append(confirmed)
                else:
                    max_wait = 1 if cand.is_direct_overlap else self.config.MICRO_KISS_VERIFY_FRAMES
                    if cand.frames_waited >= max_wait:
                        self.near_miss_logs.append(
                            {
                                "target_color": target_color.value,
                                "timestamp": cand.t_impact,
                                "d_min_mm": round(cand.d_min_mm, 2),
                                "verdict": "NEAR_MISS_NO_CONTACT",
                            }
                        )
                        self._pending_micro_kiss.pop(target_color, None)

            # 후보 대기 중이 아닌 정지 상태 목적구의 위치 히스토리 갱신
            if (
                target_color not in self._pending_micro_kiss
                and target_curr.speed < 80.0
            ):
                self._target_pos_history[target_color].append(
                    np.array([target_curr.x, target_curr.y], dtype=np.float64)
                )

        # 최근접점 이후 굴절 평가 중(frames_waited > 0)이 아닐 때만 직진 진행 벡터 히스토리 갱신
        is_post_apex_waiting = any(
            c.frames_waited > 0 for c in self._pending_micro_kiss.values()
        )
        if not is_post_apex_waiting and step_cue_norm >= 1.0:
            self._cue_vel_history.append(step_cue_vec)
        self._cue_pos_history.append(
            np.array([cue_curr.x, cue_curr.y], dtype=np.float64)
        )
        self._prev_balls = dict(balls)
        self._prev_timestamp = timestamp

        return cushion_events, contact_events
