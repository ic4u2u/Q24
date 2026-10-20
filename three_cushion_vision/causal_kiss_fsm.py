"""
Module H: 3체 인과 충돌 분리 엔진 (3-Body Causal Kiss Disambiguator)
파일 경로: three_cushion_vision/causal_kiss_fsm.py

당구대 위의 3개 공(수구 A, 제1적구 B, 제2적구 C)과 4면 쿠션의 상대적 인과관계(Causality)를 교차 검증하여
다음 4가지 키스 착시 오류를 원천 차단합니다:
1. 오류 1 (1적구-2적구 키스 후 수구 근접 통과): C의 속도 변화량 벡터(delta_vy)와 법선 내적(impulse_along_normal_y) 검증으로 노카운트 처리
2. 오류 2 (3구 밀집 구역 1적구 재키스 착시): x_deflected_Off_other_ball 조건으로 C 접촉 판정에서 수구 굴절각 증거 무효화
3. 오류 3 (쿠션 밀착구 옆 쿠션 반사 착시): 수구가 쿠션 반사 중(x_near_rail == True)일 때는 수구 굴절각 증거 제외
4. 오류 4 (3쿠션 전 적구 간 키스 정상 진행): 1적구(B)와 2적구(C) 간 키스는 object_ball_kiss_occurred = True만 기록하고 이닝 유지
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Set, Tuple
import numpy as np

TABLE_W = 2840.0
TABLE_H = 1420.0
BALL_R = 30.75
BALL_DIAM = BALL_R * 2.0  # 61.5mm


class BallColor(Enum):
    WHITE = 0
    YELLOW = 1
    RED = 2


class Rail(Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    TOP = "TOP"
    BOTTOM = "BOTTOM"


class ShotState(Enum):
    IDLE = auto()
    IN_PLAY = auto()
    VALID_SCORE_PENDING = auto()
    NO_SCORE_LOCKED = auto()


@dataclass
class FrameSnapshot:
    t: float
    pos: Dict[BallColor, np.ndarray]
    vel: Dict[BallColor, np.ndarray]


@dataclass
class ShotResult:
    scored: bool
    shooter_ball: BallColor
    cushions_at_second_hit: int
    total_cushions: int
    is_bank_shot: bool
    object_ball_kiss_occurred: bool
    fail_reason: Optional[str] = None
    event_log: List[str] = field(default_factory=list)


class CausalKissSafeFSM:
    def __init__(self, fps: float = 60.0, assigned_cue_ball: BallColor = BallColor.WHITE):
        self.fps = fps
        self.assigned_cue_ball = assigned_cue_ball

        self.STROKE_START_SPEED = 180.0
        self.STOP_SPEED = 15.0
        self.STOP_HOLD_FRAMES = int(fps * 0.25)

        self.HARD_CONTACT_DIST = BALL_DIAM + 1.5     # 63.0mm
        self.GRAY_ZONE_MAX_DIST = BALL_DIAM + 15.5   # 77.0mm
        self.RAIL_MARGIN = BALL_R + 14.0             # 44.75mm

        self.MIN_TARGET_DELTA_V = 22.0
        self.MIN_CUE_DEFLECTION_DEG = 1.8
        self.PAIR_COOLDOWN_SEC = 0.12

        self.history: Deque[FrameSnapshot] = deque(maxlen=7)
        self.prev_raw_pos: Dict[BallColor, Tuple[np.ndarray, float]] = {}
        self.prev_ema_vel: Dict[BallColor, np.ndarray] = {}
        self._reset_shot()

    def _reset_shot(self):
        self.state = ShotState.IDLE
        self.active_cue_ball: Optional[BallColor] = None
        self.first_target_ball: Optional[BallColor] = None
        self.second_target_ball: Optional[BallColor] = None
        self.cushion_count = 0
        self.cushions_at_second_hit = 0
        self.is_bank_shot = False
        self.object_ball_kiss_occurred = False
        self.fail_reason: Optional[str] = None
        self.event_log: List[str] = []
        self.stop_counter = 0
        self.in_rail_zone: Set[Rail] = set()
        self.last_rail_hit: Optional[Rail] = None
        self.last_rail_time = -999.0
        self.pair_last_hit_time: Dict[frozenset, float] = {}

    @staticmethod
    def _subframe_ccd(pA0: np.ndarray, pA1: np.ndarray, pB0: np.ndarray, pB1: np.ndarray) -> Tuple[float, float, np.ndarray]:
        r0 = pB0 - pA0
        v_rel = (pB1 - pB0) - (pA1 - pA0)
        v_rel_sq = float(np.dot(v_rel, v_rel))
        tau = 0.0 if v_rel_sq < 1e-6 else float(np.clip(-np.dot(r0, v_rel) / v_rel_sq, 0.0, 1.0))
        diff = r0 + tau * v_rel
        d_min = float(np.linalg.norm(diff))
        normal_ab = diff / max(d_min, 1e-6)
        return d_min, tau, normal_ab

    @staticmethod
    def _angle_between_deg(v1: np.ndarray, v2: np.ndarray) -> float:
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if n1 < 10.0 or n2 < 10.0:
            return 0.0
        cos_val = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_val)))

    def update(self, t: float, coords_mm: Dict[BallColor, Tuple[float, float]]) -> Optional[ShotResult]:
        if len(coords_mm) < 3:
            return None

        pos_map: Dict[BallColor, np.ndarray] = {}
        vel_map: Dict[BallColor, np.ndarray] = {}

        for color, (x, y) in coords_mm.items():
            p = np.array([float(x), float(y)], dtype=np.float64)
            v = np.zeros(2, dtype=np.float64)
            if color in self.prev_raw_pos:
                p_prev, t_prev = self.prev_raw_pos[color]
                dt = max(t - t_prev, 1e-4)
                raw_v = (p - p_prev) / dt
                v_prev = self.prev_ema_vel.get(color, raw_v)
                v = 0.65 * raw_v + 0.35 * v_prev
            self.prev_raw_pos[color] = (p, t)
            self.prev_ema_vel[color] = v
            pos_map[color] = p
            vel_map[color] = v

        self.history.append(FrameSnapshot(t=t, pos=pos_map, vel=vel_map))

        if self.state == ShotState.IDLE:
            fastest_color = max(vel_map.keys(), key=lambda c: float(np.linalg.norm(vel_map[c])))
            if float(np.linalg.norm(vel_map[fastest_color])) >= self.STROKE_START_SPEED:
                self._reset_shot()
                self.active_cue_ball = fastest_color
                if fastest_color != self.assigned_cue_ball:
                    self.state = ShotState.NO_SCORE_LOCKED
                    self.fail_reason = f"WRONG_CUE_BALL_FOUL ({fastest_color.name})"
                else:
                    self.state = ShotState.IN_PLAY
                    self.event_log.append(f"[{t:.2f}s] 스트로크 출발 (수구: {fastest_color.name})")
            return None

        if len(self.history) >= 5:
            self._evaluate_frame_with_context()

        all_stopped = all(float(np.linalg.norm(v)) < self.STOP_SPEED for v in vel_map.values())
        self.stop_counter = (self.stop_counter + 1) if all_stopped else 0

        if self.stop_counter >= self.STOP_HOLD_FRAMES:
            return self._finalize_shot(t)
        return None

    def _evaluate_frame_with_context(self):
        f_pre2, f_prev, f_curr, f_post2 = self.history[-5], self.history[-4], self.history[-3], self.history[-1]
        t_eval = f_curr.t
        cue_color = self.active_cue_ball
        if cue_color is None:
            return

        cue_near_rail = self._check_cushion_contact(t_eval, f_prev, f_curr, f_post2, cue_color)
        other_balls = [c for c in BallColor if c != cue_color]
        b1, b2 = other_balls[0], other_balls[1]

        if self._verify_pair_collision(b1, b2, f_pre2, f_prev, f_curr, f_post2, False, cue_color):
            self.object_ball_kiss_occurred = True
            self.event_log.append(f"[{t_eval:.2f}s] [적구 간 키스 감지] {b1.name} <-> {b2.name}")

        for target_color in other_balls:
            third_ball = b2 if target_color == b1 else b1
            if self._verify_pair_collision(cue_color, target_color, f_pre2, f_prev, f_curr, f_post2, cue_near_rail, third_ball):
                self._register_cue_target_hit(t_eval, target_color)

    def _verify_pair_collision(
        self, ball_x: BallColor, ball_y: BallColor,
        f_pre2: FrameSnapshot, f_prev: FrameSnapshot, f_curr: FrameSnapshot, f_post2: FrameSnapshot,
        x_near_rail: bool, other_ball: BallColor
    ) -> bool:
        pair_key = frozenset([ball_x, ball_y])
        if (f_curr.t - self.pair_last_hit_time.get(pair_key, -999.0)) < self.PAIR_COOLDOWN_SEC:
            return False

        d_min, _, n_xy = self._subframe_ccd(f_prev.pos[ball_x], f_curr.pos[ball_x], f_prev.pos[ball_y], f_curr.pos[ball_y])
        if d_min > self.GRAY_ZONE_MAX_DIST:
            return False

        vx_pre, vx_post = f_pre2.vel[ball_x], f_post2.vel[ball_x]
        vy_pre, vy_post = f_pre2.vel[ball_y], f_post2.vel[ball_y]
        delta_vx = vx_post - vx_pre
        delta_vy = vy_post - vy_pre

        d_x_other, _, n_x_other = self._subframe_ccd(f_prev.pos[ball_x], f_curr.pos[ball_x], f_prev.pos[other_ball], f_curr.pos[other_ball])
        d_y_other, _, _ = self._subframe_ccd(f_prev.pos[ball_y], f_curr.pos[ball_y], f_prev.pos[other_ball], f_curr.pos[other_ball])

        delta_vy_mag = float(np.linalg.norm(delta_vy))
        impulse_along_normal_y = float(np.dot(delta_vy, n_xy))
        y_received_causal_impulse = (
            delta_vy_mag >= self.MIN_TARGET_DELTA_V
            and impulse_along_normal_y >= self.MIN_TARGET_DELTA_V * 0.55
            and d_min <= d_y_other + 4.0
        )

        deflection_deg_x = self._angle_between_deg(vx_pre, vx_post)
        recoil_along_normal_x = float(np.dot(delta_vx, -n_xy))
        x_deflected_Off_other_ball = (
            d_x_other < d_min - 2.0
            and float(np.dot(delta_vx, -n_x_other)) > recoil_along_normal_x
        )

        x_has_valid_deflection = (
            not x_near_rail
            and not x_deflected_Off_other_ball
            and deflection_deg_x >= self.MIN_CUE_DEFLECTION_DEG
            and recoil_along_normal_x > 15.0
        )

        is_hard_overlap = (d_min <= self.HARD_CONTACT_DIST) and (not x_deflected_Off_other_ball)
        if is_hard_overlap or y_received_causal_impulse or x_has_valid_deflection:
            self.pair_last_hit_time[pair_key] = f_curr.t
            return True
        return False

    def _check_cushion_contact(self, t: float, f_prev: FrameSnapshot, f_curr: FrameSnapshot, f_post2: FrameSnapshot, cue_color: BallColor) -> bool:
        p = f_curr.pos[cue_color]
        v_in, v_out = f_prev.vel[cue_color], f_post2.vel[cue_color]
        current_zones: Set[Rail] = set()
        if p[0] <= self.RAIL_MARGIN: current_zones.add(Rail.LEFT)
        if p[0] >= TABLE_W - self.RAIL_MARGIN: current_zones.add(Rail.RIGHT)
        if p[1] <= self.RAIL_MARGIN: current_zones.add(Rail.TOP)
        if p[1] >= TABLE_H - self.RAIL_MARGIN: current_zones.add(Rail.BOTTOM)

        newly_entered = current_zones - self.in_rail_zone
        self.in_rail_zone = current_zones
        for rail in newly_entered:
            if (rail != self.last_rail_hit) or ((t - self.last_rail_time) >= 0.08):
                self.cushion_count += 1
                self.last_rail_hit = rail
                self.last_rail_time = t
                self.event_log.append(f"[{t:.2f}s] 쿠션 접촉 #{self.cushion_count} ({rail.value})")

        normal_flipped = (v_in[0] * v_out[0] < -100.0) or (v_in[1] * v_out[1] < -100.0)
        return (len(current_zones) > 0) and normal_flipped

    def _register_cue_target_hit(self, t: float, target_color: BallColor):
        if self.first_target_ball is None:
            self.first_target_ball = target_color
            if self.cushion_count >= 3:
                self.is_bank_shot = True
            self.event_log.append(f"[{t:.2f}s] 제1적구 접촉 ({target_color.name}, 현재 쿠션: {self.cushion_count})")
        elif target_color == self.first_target_ball:
            self.event_log.append(f"[{t:.2f}s] [수구-제1적구 재키스] {target_color.name} 재접촉")
        elif self.second_target_ball is None:
            self.second_target_ball = target_color
            self.cushions_at_second_hit = self.cushion_count
            if self.state == ShotState.IN_PLAY:
                if self.cushion_count >= 3:
                    self.state = ShotState.VALID_SCORE_PENDING
                    self.event_log.append(f"[{t:.2f}s] ★ 제2적구 정상 접촉 ({target_color.name}) -> 득점 확정 대기")
                else:
                    self.state = ShotState.NO_SCORE_LOCKED
                    self.fail_reason = f"EARLY_SECOND_BALL_KISS (쿠션 {self.cushion_count}회 < 3회)"
                    self.event_log.append(f"[{t:.2f}s] ✖ 제2적구 조기 키스 ({target_color.name}) -> 노카운트 잠금")

    def _finalize_shot(self, t: float) -> ShotResult:
        scored = (self.state == ShotState.VALID_SCORE_PENDING)
        if not scored and self.fail_reason is None:
            self.fail_reason = "NO_FIRST_BALL_HIT" if self.first_target_ball is None else "SECOND_BALL_MISSED"
        result = ShotResult(
            scored=scored,
            shooter_ball=self.active_cue_ball or self.assigned_cue_ball,
            cushions_at_second_hit=self.cushions_at_second_hit,
            total_cushions=self.cushion_count,
            is_bank_shot=self.is_bank_shot,
            object_ball_kiss_occurred=self.object_ball_kiss_occurred,
            fail_reason=None if scored else self.fail_reason,
            event_log=list(self.event_log)
        )
        self._reset_shot()
        return result
