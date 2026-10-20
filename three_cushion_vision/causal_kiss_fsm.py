"""
Module H: 3체 인과 충돌 분리, 투터치 파울, 바운딩 박스 병합/NanoGPT 가려짐 복원 및 8대 악성 키스 오판 방어 엔진
파일 경로: three_cushion_vision/causal_kiss_fsm.py
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Set, Tuple
import numpy as np
import torch

from billiard_nanogpt import BALL_R, TABLE_H, TABLE_W, BilliardNanoGPT

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
    frame_idx: int
    t: float
    pos: Dict[BallColor, np.ndarray]
    vel: Dict[BallColor, np.ndarray]
    acc: Dict[BallColor, np.ndarray]


@dataclass
class ShotResult:
    scored: bool
    shooter_ball: BallColor
    cushions_at_second_hit: int
    total_cushions: int
    is_bank_shot: bool
    object_ball_kiss_occurred: bool
    double_hit_foul: bool = False
    centroid_shift_corrected_count: int = 0
    gpt_occlusion_restored_frames: int = 0
    kiss_id_swap_corrected: bool = False
    fail_reason: Optional[str] = None
    event_log: List[str] = field(default_factory=list)


class CausalKissSafeFSM:
    """
    1적구-2적구 키스 후 득점 실패를 득점으로 오인하는 현상을 100% 차단하기 위해
    - Split-Window 무중첩 전후 속도 미분 및 이벤트 방화벽 (last_impulse_frame)
    - 뉴턴 제3법칙 운동량 보존 기반 적구 키스 충격량 차감 (Δv_C_net = Δv_C + Δv_B)
    - 무조건 거리 판정(is_hard_overlap) 폐지 및 쌍방 작용-반작용(AND 게이트) 검증
    - A -> B -> C 연쇄 간접 충돌(Chain Cannon) 기하학적 차폐
    - 적구 정면 키스 시 칼만 필터 ID 스왑(Ghost Pass-Through) 자동 역스왑 복원
    을 모두 탑재한 쓰리쿠션 판정 엔진
    """
    def __init__(
        self,
        fps: float = 60.0,
        assigned_cue_ball: BallColor = BallColor.WHITE,
        gpt_model: Optional[BilliardNanoGPT] = None,
        device: str = "cpu"
    ):
        self.fps = fps
        self.assigned_cue_ball = assigned_cue_ball
        self.device = device
        self.gpt_model = (gpt_model or BilliardNanoGPT()).to(device).eval()

        self.STROKE_START_SPEED = 180.0
        self.STOP_SPEED = 15.0
        self.STOP_HOLD_FRAMES = int(fps * 0.25)

        self.HARD_CONTACT_DIST = BALL_DIAM + 1.5     # 63.0mm
        self.GRAY_ZONE_MAX_DIST = BALL_DIAM + 15.5   # 77.0mm
        self.RAIL_MARGIN = BALL_R + 14.0             # 44.75mm

        self.MIN_TARGET_DELTA_V = 25.0
        self.MIN_CUE_DEFLECTION_DEG = 1.8
        self.PAIR_COOLDOWN_SEC = 0.12

        # 투터치(Double Hit) 파울 감지 임계값
        self.DOUBLE_HIT_WINDOW_SEC = 0.15
        self.DOUBLE_HIT_INIT_DIST_MAX = 115.0
        self.DOUBLE_HIT_REACCEL_THRESH = 1500.0
        self.DOUBLE_HIT_FOLLOW_RATIO = 0.75

        # 바운딩 박스 병합 및 NanoGPT 가려짐 복원 임계값
        self.BLOB_PROXIMITY_ZONE_MM = 85.0
        self.MAX_CENTROID_JUMP_RESIDUAL_MM = 14.0
        self.MAX_OCCLUSION_INTERP_FRAMES = 15

        self.frame_counter: int = 0
        self.history: Deque[FrameSnapshot] = deque(maxlen=6)
        self.prev_raw_pos: Dict[BallColor, Tuple[np.ndarray, float]] = {}
        self.prev_ema_vel: Dict[BallColor, np.ndarray] = {}
        self.missing_streak: Dict[BallColor, int] = {c: 0 for c in BallColor}

        self.gpt_buffers: Dict[BallColor, Deque[Tuple[float, float, float, float]]] = {
            c: deque(maxlen=self.gpt_model.config.block_size) for c in BallColor
        }

        self._reset_shot()

    def _reset_shot(self):
        self.state = ShotState.IDLE
        self.stroke_start_time: float = 0.0
        self.active_cue_ball: Optional[BallColor] = None
        self.first_target_ball: Optional[BallColor] = None
        self.second_target_ball: Optional[BallColor] = None
        self.first_target_hit_time: Optional[float] = None
        self.first_target_normal: Optional[np.ndarray] = None

        self.initial_positions: Dict[BallColor, np.ndarray] = {}
        self.initial_ball_distances: Dict[BallColor, float] = {}
        self.cushion_count = 0
        self.cushions_at_second_hit = 0
        self.is_bank_shot = False
        self.object_ball_kiss_occurred = False
        self.last_object_kiss_time: float = -999.0
        self.double_hit_foul = False
        self.centroid_shift_corrected_count = 0
        self.gpt_occlusion_restored_frames = 0
        self.kiss_id_swap_corrected = False
        self.fail_reason: Optional[str] = None
        self.event_log: List[str] = []

        self.stop_counter = 0
        self.in_rail_zone: Set[Rail] = set()
        self.last_rail_hit: Optional[Rail] = None
        self.last_rail_time = -999.0
        self.pair_last_hit_time: Dict[frozenset, float] = {}

        # [핵심 방화벽 1] 각 공이 마지막으로 충돌/반사 이벤트를 겪은 프레임 번호 기록 (이전 충격파 누수 차단)
        self.last_impulse_frame: Dict[BallColor, int] = {c: -1 for c in BallColor}
        # [핵심 방화벽 2] 적구 간 정면 키스 시 트래커 ID 스왑 보정 매핑 테이블
        self.id_swap_map: Dict[BallColor, BallColor] = {c: c for c in BallColor}
        self._last_id_swap_check_time: float = -999.0
        self.centroid_clamp_cooldown: Dict[BallColor, int] = {c: 0 for c in BallColor}

    def _predict_next_state_with_gpt(self, color: BallColor, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        buf = self.gpt_buffers[color]
        p_prev, _ = self.prev_raw_pos[color]
        v_prev = self.prev_ema_vel.get(color, np.zeros(2, dtype=np.float64))

        if len(buf) >= 2 and float(np.linalg.norm(v_prev)) > 25.0:
            hist_tensor = torch.tensor([list(buf)], dtype=torch.float32, device=self.device)
            pred_state = self.gpt_model.rollout_occlusion(hist_tensor, n_missing_frames=1)[0, 0].cpu().numpy()
            pred_pos = np.array([float(pred_state[0]), float(pred_state[1])], dtype=np.float64)
            pred_vel = np.array([float(pred_state[2]), float(pred_state[3])], dtype=np.float64)
            return pred_pos, pred_vel

        pred_pos = np.clip(
            p_prev + v_prev * dt,
            [BALL_R, BALL_R],
            [TABLE_W - BALL_R, TABLE_H - BALL_R]
        )
        return pred_pos, v_prev.copy()

    def _stabilize_blob_merging(
        self,
        t: float,
        raw_coords_mm: Dict[BallColor, Optional[Tuple[float, float]]]
    ) -> Optional[Tuple[Dict[BallColor, np.ndarray], Dict[BallColor, Optional[np.ndarray]]]]:
        # 트래커 ID 스왑 보정 매핑을 먼저 적용
        mapped_coords: Dict[BallColor, Optional[Tuple[float, float]]] = {
            true_c: raw_coords_mm.get(obs_c, None)
            for true_c, obs_c in self.id_swap_map.items()
        }

        predicted_pos: Dict[BallColor, np.ndarray] = {}
        predicted_vel: Dict[BallColor, np.ndarray] = {}
        stabilized_pos: Dict[BallColor, np.ndarray] = {}
        gpt_restored_vel_override: Dict[BallColor, Optional[np.ndarray]] = {c: None for c in BallColor}

        for color in BallColor:
            obs = mapped_coords.get(color, None)
            if color in self.prev_raw_pos:
                _, t_prev = self.prev_raw_pos[color]
                dt = max(t - t_prev, 1e-4)
                p_pred, v_pred = self._predict_next_state_with_gpt(color, dt)
                predicted_pos[color] = p_pred
                predicted_vel[color] = v_pred
            elif obs is not None:
                p_init = np.array([float(obs[0]), float(obs[1])], dtype=np.float64)
                predicted_pos[color] = p_init
                predicted_vel[color] = np.zeros(2, dtype=np.float64)

            if obs is None:
                self.missing_streak[color] += 1
                if color in predicted_pos and self.missing_streak[color] <= self.MAX_OCCLUSION_INTERP_FRAMES:
                    stabilized_pos[color] = predicted_pos[color]
                    gpt_restored_vel_override[color] = predicted_vel[color]
                    self.gpt_occlusion_restored_frames += 1
                else:
                    return None
            else:
                self.missing_streak[color] = 0
                stabilized_pos[color] = np.array([float(obs[0]), float(obs[1])], dtype=np.float64)

        colors = list(BallColor)
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                c1, c2 = colors[i], colors[j]
                if c1 not in self.prev_raw_pos or c2 not in self.prev_raw_pos:
                    continue

                p1_pred, p2_pred = predicted_pos[c1], predicted_pos[c2]
                dist_pred = float(np.linalg.norm(p2_pred - p1_pred))

                if dist_pred <= self.BLOB_PROXIMITY_ZONE_MM:
                    for target_c, neighbor_c in [(c1, c2), (c2, c1)]:
                        if mapped_coords.get(target_c, None) is None:
                            continue
                        p_obs = stabilized_pos[target_c]
                        p_prd = predicted_pos[target_c]
                        shift_vec = p_obs - p_prd
                        shift_mag = float(np.linalg.norm(shift_vec))

                        to_neighbor = self.prev_raw_pos[neighbor_c][0] - self.prev_raw_pos[target_c][0]
                        to_neighbor_norm = to_neighbor / max(float(np.linalg.norm(to_neighbor)), 1e-6)
                        pull_alignment = float(np.dot(shift_vec / max(shift_mag, 1e-6), to_neighbor_norm))

                        prev_speed = float(np.linalg.norm(self.prev_ema_vel.get(target_c, np.zeros(2))))
                        if shift_mag > self.MAX_CENTROID_JUMP_RESIDUAL_MM and pull_alignment > 0.45 and prev_speed < 150.0:
                            stabilized_pos[target_c] = 0.85 * p_prd + 0.15 * p_obs
                            self.centroid_shift_corrected_count += 1
                            self.centroid_clamp_cooldown[target_c] = 6

        return stabilized_pos, gpt_restored_vel_override

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

    def _compute_split_velocities(self) -> Tuple[Dict[BallColor, np.ndarray], Dict[BallColor, np.ndarray], Dict[BallColor, np.ndarray]]:
        """
        [핵심 해결책 1] EMA 스무딩에 의한 충격파 번짐(Smearing)을 원천 배제하기 위해,
        평가 구간 [h[2], h[3]]을 기준으로:
        - v_pre: 순수 충돌 전 구간 [h[start], h[2]]의 평균 속도 (단, start는 last_impulse_frame 이후로 클램핑!)
        - v_post1: 충돌 직후 1프레임 구간 [h[3], h[4]]의 즉각 속도
        - v_post2: 충돌 후 2프레임 구간 [h[3], h[5]]의 안정 속도
        를 완전히 분리 계산합니다.
        """
        v_pre: Dict[BallColor, np.ndarray] = {}
        v_post1: Dict[BallColor, np.ndarray] = {}
        v_post2: Dict[BallColor, np.ndarray] = {}

        h = self.history
        for c in BallColor:
            last_ev_f = self.last_impulse_frame.get(c, -1)
            # h[0], h[1], h[2] 중 last_impulse_frame 이후인 가장 이른 프레임 선택
            start_idx = 0
            for idx in (0, 1, 2):
                if h[idx].frame_idx >= last_ev_f:
                    start_idx = idx
                    break
            else:
                start_idx = 2

            if start_idx < 2:
                dt_pre = max(h[2].t - h[start_idx].t, 1e-4)
                v_pre[c] = (h[2].pos[c] - h[start_idx].pos[c]) / dt_pre
            else:
                # 직전 프레임(h[2])에서 방금 충격 이벤트가 있었다면 h[2]->h[3] 구간을 새 기준 속도로 사용
                dt_pre = max(h[3].t - h[2].t, 1e-4)
                v_pre[c] = (h[3].pos[c] - h[2].pos[c]) / dt_pre

            dt_post1 = max(h[4].t - h[3].t, 1e-4)
            v_post1[c] = (h[4].pos[c] - h[3].pos[c]) / dt_post1

            dt_post2 = max(h[5].t - h[3].t, 1e-4)
            v_post2[c] = (h[5].pos[c] - h[3].pos[c]) / dt_post2

        return v_pre, v_post1, v_post2

    def _detect_and_fix_kiss_id_swap(
        self,
        t_eval: float,
        b1: BallColor,
        b2: BallColor,
        v_pre: Dict[BallColor, np.ndarray],
        v_post2: Dict[BallColor, np.ndarray]
    ) -> bool:
        """
        [핵심 해결책 4] 1적구(B)가 정지한 2적구(C)를 정면으로 쳤을 때,
        B는 멈추고 C가 B의 속도로 날아가는 현상 때문에 칼만 필터(OC-SORT)가 두 적구의 ID를
        서로 뒤바꾸는 '유령 투과(Ghost Pass-Through)' 현상을 감지하고 즉시 역스왑(Un-swap)합니다.
        """
        if (t_eval - self._last_id_swap_check_time) < 0.20:
            return False

        h2, h3, h5 = self.history[2], self.history[3], self.history[5]
        d_min, _, n_12 = self._subframe_ccd(h2.pos[b1], h3.pos[b1], h2.pos[b2], h3.pos[b2])

        # 두 적구의 중심이 물리적 직경(61.5mm)보다 훨씬 깊숙이 겹쳐 투과(d_min < 42mm)했을 때 검사
        if d_min > 42.0:
            return False

        for moving_b, stationary_b in [(b1, b2), (b2, b1)]:
            s_mov_pre = float(np.linalg.norm(v_pre[moving_b]))
            s_sta_pre = float(np.linalg.norm(v_pre[stationary_b]))
            s_mov_post = float(np.linalg.norm(v_post2[moving_b]))
            s_sta_post = float(np.linalg.norm(v_post2[stationary_b]))

            # 한 적구는 빠르게 진입(>300mm/s)하고 다른 적구는 정지(<25mm/s)해 있었는데,
            # 중심 거리 42mm 이하로 정면 관통했음에도 계속 전방(>150mm/s, <25도)으로 뚫고 나갔다면
            # 강체 물리학상 불가능한 유령 관통이므로 트래커 ID 스왑으로 확정!
            if s_mov_pre > 300.0 and s_sta_pre < 25.0 and s_sta_post < 25.0 and s_mov_post > 150.0:
                angle_change = self._angle_between_deg(v_pre[moving_b], v_post2[moving_b])
                # 관통 전후 위치 벡터의 부호가 역전되었는지(뚫고 지나갔는지) 확인
                rel_before = h2.pos[stationary_b] - h2.pos[moving_b]
                rel_after = h5.pos[stationary_b] - h5.pos[moving_b]
                passed_through = float(np.dot(rel_before, rel_after)) < 0.0

                if passed_through and angle_change < 25.0:
                    # ID 매핑 역스왑 및 히스토리 내 두 적구 좌표/속도 즉시 교환
                    self.id_swap_map[b1], self.id_swap_map[b2] = self.id_swap_map[b2], self.id_swap_map[b1]
                    for snap in self.history:
                        snap.pos[b1], snap.pos[b2] = snap.pos[b2].copy(), snap.pos[b1].copy()
                        snap.vel[b1], snap.vel[b2] = snap.vel[b2].copy(), snap.vel[b1].copy()
                        snap.acc[b1], snap.acc[b2] = snap.acc[b2].copy(), snap.acc[b1].copy()
                    self.prev_raw_pos[b1], self.prev_raw_pos[b2] = self.prev_raw_pos[b2], self.prev_raw_pos[b1]
                    self.prev_ema_vel[b1], self.prev_ema_vel[b2] = self.prev_ema_vel[b2], self.prev_ema_vel[b1]

                    self.kiss_id_swap_corrected = True
                    self.object_ball_kiss_occurred = True
                    self.last_object_kiss_time = t_eval
                    self._last_id_swap_check_time = t_eval
                    self.last_impulse_frame[b1] = h3.frame_idx
                    self.last_impulse_frame[b2] = h3.frame_idx
                    self.event_log.append(
                        f"[{t_eval:.2f}s] [트래커 ID 스왑 자동 복원] 적구 정면 키스({b1.name}<->{b2.name}) 후 자리바꿈 역스왑 완료"
                    )
                    return True
        return False

    def update(self, t: float, coords_mm: Dict[BallColor, Optional[Tuple[float, float]]]) -> Optional[ShotResult]:
        stabilized_out = self._stabilize_blob_merging(t, coords_mm)
        if stabilized_out is None:
            return None
        stabilized_pos, gpt_vel_override = stabilized_out

        pos_map: Dict[BallColor, np.ndarray] = {}
        vel_map: Dict[BallColor, np.ndarray] = {}
        acc_map: Dict[BallColor, np.ndarray] = {}

        for color, p in stabilized_pos.items():
            v = np.zeros(2, dtype=np.float64)
            acc = np.zeros(2, dtype=np.float64)
            if color in self.prev_raw_pos:
                p_prev, t_prev = self.prev_raw_pos[color]
                dt = max(t - t_prev, 1e-4)
                v_prev = self.prev_ema_vel.get(color, np.zeros(2, dtype=np.float64))

                if self.centroid_clamp_cooldown.get(color, 0) > 0:
                    v = v_prev
                    acc = np.zeros(2, dtype=np.float64)
                    self.centroid_clamp_cooldown[color] -= 1
                elif gpt_vel_override[color] is not None:
                    v = gpt_vel_override[color]
                    acc = (v - v_prev) / dt
                else:
                    raw_v = (p - p_prev) / dt
                    v = 0.65 * raw_v + 0.35 * v_prev
                    acc = (v - v_prev) / dt

            self.prev_raw_pos[color] = (p, t)
            self.prev_ema_vel[color] = v
            pos_map[color] = p
            vel_map[color] = v
            acc_map[color] = acc

            self.gpt_buffers[color].append((float(p[0]), float(p[1]), float(v[0]), float(v[1])))

        self.frame_counter += 1
        self.history.append(FrameSnapshot(frame_idx=self.frame_counter, t=t, pos=pos_map, vel=vel_map, acc=acc_map))

        if self.state == ShotState.IDLE:
            fastest_color = max(vel_map.keys(), key=lambda c: float(np.linalg.norm(vel_map[c])))
            if float(np.linalg.norm(vel_map[fastest_color])) >= self.STROKE_START_SPEED:
                self._reset_shot()
                self.stroke_start_time = t
                self.active_cue_ball = fastest_color
                prev_snap = self.history[-2] if len(self.history) >= 2 else self.history[-1]
                self.initial_positions = {c: prev_snap.pos[c].copy() for c in BallColor}
                cue_init_p = self.initial_positions[fastest_color]
                self.initial_ball_distances = {
                    c: float(np.linalg.norm(self.initial_positions[c] - cue_init_p))
                    for c in BallColor if c != fastest_color
                }

                if fastest_color != self.assigned_cue_ball:
                    self.state = ShotState.NO_SCORE_LOCKED
                    self.fail_reason = f"WRONG_CUE_BALL_FOUL ({fastest_color.name})"
                else:
                    self.state = ShotState.IN_PLAY
                    self.event_log.append(f"[{t:.2f}s] 스트로크 출발 (수구: {fastest_color.name})")
            return None

        if len(self.history) >= 6:
            self._evaluate_frame_with_context()

        all_stopped = all(float(np.linalg.norm(v)) < self.STOP_SPEED for v in vel_map.values())
        self.stop_counter = (self.stop_counter + 1) if all_stopped else 0

        if self.stop_counter >= self.STOP_HOLD_FRAMES:
            return self._finalize_shot(t)
        return None

    def _evaluate_frame_with_context(self):
        h2, h3, h5 = self.history[2], self.history[3], self.history[5]
        t_eval = h3.t
        cue_color = self.active_cue_ball
        if cue_color is None:
            return

        v_pre, v_post1, v_post2 = self._compute_split_velocities()

        # 1. 수구 쿠션 접촉 판정
        cue_near_rail = self._check_cushion_contact(t_eval, h2, h3, v_pre[cue_color], v_post2[cue_color], cue_color)

        other_balls = [c for c in BallColor if c != cue_color]
        b1, b2 = other_balls[0], other_balls[1]

        # 2. 적구 정면 키스 시 트래커 ID 스왑(유령 관통) 자동 감지 및 역스왑
        if self._detect_and_fix_kiss_id_swap(t_eval, b1, b2, v_pre, v_post2):
            v_pre, v_post1, v_post2 = self._compute_split_velocities()

        # 3. 적구 간 충돌(B <-> C 키스) 먼저 판정 및 운동량 기록
        bc_collided = self._verify_pair_collision(
            ball_x=b1, ball_y=b2, other_ball=cue_color,
            v_pre=v_pre, v_post1=v_post1, v_post2=v_post2,
            x_near_rail=False, is_cue_second_target_check=False, bc_kiss_this_frame=False
        )
        if bc_collided:
            self.object_ball_kiss_occurred = True
            self.last_object_kiss_time = t_eval
            self.last_impulse_frame[b1] = h3.frame_idx
            self.last_impulse_frame[b2] = h3.frame_idx
            self.event_log.append(f"[{t_eval:.2f}s] [적구 간 키스 감지] {b1.name} <-> {b2.name}")

        # 4. 수구와 각 목적구 간의 실제 충돌 여부 검증
        # 주의: 제1적구 재충돌(A <-> B)이 동시에 일어났는지 먼저 확인하기 위해 first_target_ball을 먼저 평가!
        sorted_targets = sorted(
            other_balls,
            key=lambda c: 0 if c == self.first_target_ball else 1
        )

        a_hit_first_target_this_frame = False
        for target_color in sorted_targets:
            third_ball = b2 if target_color == b1 else b1
            is_second_target_check = (self.first_target_ball is not None and target_color != self.first_target_ball)

            # 만약 이번 프레임에 이미 수구(A)가 제1적구(B)를 타격했다면, A -> B -> C 간접 연쇄 키스 여부를 엄격히 차폐
            is_hit = self._verify_pair_collision(
                ball_x=cue_color,
                ball_y=target_color,
                other_ball=third_ball,
                v_pre=v_pre,
                v_post1=v_post1,
                v_post2=v_post2,
                x_near_rail=cue_near_rail,
                is_cue_second_target_check=is_second_target_check,
                bc_kiss_this_frame=(bc_collided or a_hit_first_target_this_frame)
            )
            if is_hit:
                if target_color == self.first_target_ball:
                    a_hit_first_target_this_frame = True
                self.last_impulse_frame[cue_color] = h3.frame_idx
                self.last_impulse_frame[target_color] = h3.frame_idx
                _, _, n_ab = self._subframe_ccd(h2.pos[cue_color], h3.pos[cue_color], h2.pos[target_color], h3.pos[target_color])
                self._register_cue_target_hit(t_eval, target_color, n_ab, v_pre, v_post2)

        # 5. 스트로크 직후 0.15초 내 투터치(Double Hit) 파울 검사
        self._check_double_hit_foul(t_eval, v_pre, v_post2, h5)

    def _verify_pair_collision(
        self,
        ball_x: BallColor,
        ball_y: BallColor,
        other_ball: BallColor,
        v_pre: Dict[BallColor, np.ndarray],
        v_post1: Dict[BallColor, np.ndarray],
        v_post2: Dict[BallColor, np.ndarray],
        x_near_rail: bool,
        is_cue_second_target_check: bool,
        bc_kiss_this_frame: bool
    ) -> bool:
        """
        [핵심 해결책 2 & 3]
        - 무조건 거리 통과(is_hard_overlap) 폐지: 거리가 60mm 이하라도 물리적 충격량 교환이 없으면 기각!
        - 운동량 보존 기반 적구 키스 충격 차감: Δv_Y_net = Δv_Y + Δv_other (other_ball과 Y가 충돌 중일 때)
        - A -> B -> C 간접 연쇄 충돌(Chain Cannon) 기하학적 차폐
        - 적구 키스 발생 후 또는 이동 중인 제2적구(Y)에 대해 쌍방 작용-반작용(AND 게이트) 강제
        """
        h2, h3 = self.history[2], self.history[3]
        t_eval = h3.t

        pair_key = frozenset([ball_x, ball_y])
        if (t_eval - self.pair_last_hit_time.get(pair_key, -999.0)) < self.PAIR_COOLDOWN_SEC:
            return False

        d_min, _, n_xy = self._subframe_ccd(h2.pos[ball_x], h3.pos[ball_x], h2.pos[ball_y], h3.pos[ball_y])
        if d_min > self.GRAY_ZONE_MAX_DIST:
            return False

        vx_pre, vx_post = v_pre[ball_x], v_post2[ball_x]
        vy_pre, vy_post = v_pre[ball_y], v_post2[ball_y]
        vy_post_immediate = v_post1[ball_y]
        vx_post_immediate = v_post1[ball_x]

        delta_vx = vx_post - vx_pre
        delta_vy = vy_post - vy_pre
        if self.centroid_clamp_cooldown.get(ball_y, 0) > 0:
            delta_vy = np.zeros(2, dtype=np.float64)
            vy_post_immediate = vy_pre.copy()

        # 제3의 공(other_ball)과의 최근접 거리 및 법선 벡터 계산
        d_x_other, _, n_x_other = self._subframe_ccd(h2.pos[ball_x], h3.pos[ball_x], h2.pos[other_ball], h3.pos[other_ball])
        d_y_other, _, n_other_y = self._subframe_ccd(h2.pos[other_ball], h3.pos[other_ball], h2.pos[ball_y], h3.pos[ball_y])

        # -----------------------------------------------------------------
        # [방어 A] A -> B -> C 간접 연쇄 키스(Chain Cannon) 차폐
        # 수구(X)가 제1적구(other_ball)를 치고 제1적구가 제2적구(Y)를 친 경우:
        # 1) d(X, other) <= d(X, Y) 및 d(other, Y) <= 75mm 이면서
        # 2) other_ball이 X와 Y 사이에 위치(내적 > 0)하고 Y의 충격 방향이 n_other_y와 일치하면
        # X가 Y를 직접 친 것이 아니라 other_ball이 Y를 친 것이므로 즉시 기각!
        # -----------------------------------------------------------------
        delta_v_other = v_post2[other_ball] - v_pre[other_ball]
        if is_cue_second_target_check and d_y_other <= self.GRAY_ZONE_MAX_DIST:
            vec_xy = h3.pos[ball_y] - h3.pos[ball_x]
            vec_x_other = h3.pos[other_ball] - h3.pos[ball_x]
            dist_xy = max(float(np.linalg.norm(vec_xy)), 1e-6)
            proj_ratio = float(np.dot(vec_x_other, vec_xy / dist_xy)) / dist_xy

            # other_ball(1적구)이 수구(X)와 2적구(Y) 사이에 끼어 있거나 X가 1적구에 더 가깝게 부딪힌 경우
            other_is_between = (0.0 < proj_ratio < 1.05) and (d_x_other <= d_min + 6.0)
            if other_is_between or bc_kiss_this_frame:
                # 1적구(other)가 2적구(Y)에 전달한 운동량(-Δv_other)을 Y의 속도 변화에서 차감!
                # 단, X가 other를 쳤다면 other도 X로부터 충격을 받았으므로,
                # X가 Y에 직접 준 충격이 있는지는 X의 반작용 방향이 n_xy인지 n_x_other인지로 판별
                recoil_along_other = float(np.dot(delta_vx, -n_x_other))
                recoil_along_y = float(np.dot(delta_vx, -n_xy))
                if d_x_other < d_min - 1.5 and recoil_along_other >= recoil_along_y * 0.9:
                    return False

        # -----------------------------------------------------------------
        # [방어 B] 뉴턴 제3법칙 운동량 보존 기반 적구 키스 충격량 차감
        # 만약 Y와 other_ball이 현재 근접 충돌 중이라면(d_y_other <= 75mm),
        # Y가 받은 충격 중 other_ball이 잃은 운동량(-Δv_other)만큼은 other_ball 때문이므로 차감합니다.
        # -----------------------------------------------------------------
        delta_vy_net = delta_vy.copy()
        if d_y_other <= self.GRAY_ZONE_MAX_DIST:
            other_pushed_y = float(np.dot(delta_v_other, -n_other_y)) > 18.0 and float(np.dot(delta_vy, n_other_y)) > 18.0
            if other_pushed_y:
                # other_ball이 Y에 전달한 법선 운동량 성분을 차감
                transferred_from_other = float(np.dot(-delta_v_other, n_other_y)) * n_other_y
                delta_vy_net = delta_vy - transferred_from_other
                # 적구 간 키스 플래그도 즉시 활성화
                if is_cue_second_target_check:
                    self.object_ball_kiss_occurred = True
                    self.last_object_kiss_time = t_eval

        # -----------------------------------------------------------------
        # [방어 C] 즉각 충격 일관성 검증 (미래 프레임 h[4]->h[5] 충돌의 역방향 누수 차단)
        # 진짜로 현재 프레임(h[2]->h[3])에서 충돌했다면, 직후 1프레임 속도(v_post1)에서도
        # 이미 속도 변화가 시작되었어야 합니다.
        # -----------------------------------------------------------------
        delta_vy_imm = vy_post_immediate - vy_pre
        delta_vx_imm = vx_post_immediate - vx_pre

        impulse_y_normal = float(np.dot(delta_vy_net, n_xy))
        impulse_y_imm_normal = float(np.dot(delta_vy_imm, n_xy))
        delta_vy_net_mag = float(np.linalg.norm(delta_vy_net))

        recoil_x_normal = float(np.dot(delta_vx, -n_xy))
        recoil_x_imm_normal = float(np.dot(delta_vx_imm, -n_xy))
        deflection_deg_x = self._angle_between_deg(vx_pre, vx_post)

        # 접근 속도(Closing Speed): 충돌 직전 두 공이 실제로 가까워지고 있었는지 확인
        closing_speed = float(np.dot(vx_pre - vy_pre, n_xy))
        if closing_speed < -15.0:
            return False

        # X가 제3의 공(other_ball)과 더 가깝게 충돌하여 꺾인 경우 X의 굴절 증거 무효화
        x_deflected_off_other = (
            d_x_other < d_min - 1.5
            and float(np.dot(delta_vx, -n_x_other)) > max(recoil_x_normal * 0.85, 15.0)
        )
        if x_deflected_off_other:
            return False

        # Y의 충격이 현재 프레임(h[3])에서 즉시 시작되었는지 확인 (다음 프레임 키스 역류 방지)
        y_impulse_is_immediate = (impulse_y_imm_normal >= impulse_y_normal * 0.35) or (impulse_y_imm_normal >= 15.0)
        x_recoil_is_immediate = (recoil_x_imm_normal >= recoil_x_normal * 0.35) or (recoil_x_imm_normal >= 15.0)

        speed_y_pre = float(np.linalg.norm(vy_pre))
        recent_kiss_contaminating = (
            self.object_ball_kiss_occurred
            and (t_eval - self.last_object_kiss_time) <= 0.35
        ) or (d_y_other <= self.GRAY_ZONE_MAX_DIST)

        # -----------------------------------------------------------------
        # [방어 D] 상태별 엄격한 물리 게이트 분기 (무조건 거리 통과 완전 폐지!)
        # -----------------------------------------------------------------
        # Case 1: 적구 간 키스가 있었거나, 제2적구(Y)가 이미 굴러가고 있는 상태 (오판 위험 최고 구간!)
        if is_cue_second_target_check and (speed_y_pre >= 20.0 or recent_kiss_contaminating):
            # 이동 중인 Y의 마찰 감속이나 키스 잔류 속도에 속지 않도록:
            # 1) Y가 n_xy 법선 방향으로 명확한 즉각 충격(>= 38mm/s)을 받고 동시에 X도 반작용을 받거나(쌍방 AND),
            # 2) Y가 큰 법선 충격(>= 65mm/s)으로 궤적이 꺾이고(>= 4도) 제3구와 무관함이 입증되어야 함!
            deflection_deg_y = self._angle_between_deg(vy_pre, vy_post)
            strong_y_kink = (
                y_impulse_is_immediate
                and impulse_y_normal >= 42.0
                and (deflection_deg_y >= 4.0 or impulse_y_normal >= 85.0)
                and d_min <= d_y_other + 2.0
            )
            mutual_action_reaction = (
                y_impulse_is_immediate
                and x_recoil_is_immediate
                and impulse_y_normal >= 28.0
                and recoil_x_normal >= 20.0
                and (not x_near_rail)
            )
            if strong_y_kink or mutual_action_reaction:
                self.pair_last_hit_time[pair_key] = t_eval
                return True
            return False

        # Case 2: 피타격구(Y)가 완전 정지 상태(speed_y_pre < 20mm/s)이고 주변 적구 간섭이 없는 깨끗한 상태
        y_cleanly_launched = (
            y_impulse_is_immediate
            and delta_vy_net_mag >= self.MIN_TARGET_DELTA_V
            and impulse_y_normal >= self.MIN_TARGET_DELTA_V * 0.60
            and d_min <= d_y_other + 4.0
        )

        # 정지한 Y를 초박두께(0.5mm)로 스친 경우: X가 쿠션 밖에서 명확히 꺾이고 동시에 Y도 최소 미동(>=8mm/s)해야 함
        # (단순히 X 혼자 스핀으로 2도 휘어졌다고 해서 Y가 미동도 안 하는데 득점 처리하는 것 차단!)
        thin_graze_verified = (
            not x_near_rail
            and x_recoil_is_immediate
            and deflection_deg_x >= self.MIN_CUE_DEFLECTION_DEG
            and recoil_x_normal >= 20.0
            and impulse_y_normal >= 8.0
        )

        # 정면 타격 후 수구 급정지 케이스 (합성 키프레임 호환용: d_min <= 60mm 이고 수구가 정면에서 급감속 정지)
        head_on_stop_verified = (
            d_min <= 60.0
            and closing_speed >= 400.0
            and recoil_x_normal >= closing_speed * 0.75
            and float(np.linalg.norm(vx_post)) < 50.0
            and (not recent_kiss_contaminating)
        )

        if y_cleanly_launched or thin_graze_verified or head_on_stop_verified:
            self.pair_last_hit_time[pair_key] = t_eval
            return True

        return False

    def _check_cushion_contact(
        self,
        t: float,
        h2: FrameSnapshot,
        h3: FrameSnapshot,
        v_in: np.ndarray,
        v_out: np.ndarray,
        cue_color: BallColor
    ) -> bool:
        p = h3.pos[cue_color]
        current_zones: Set[Rail] = set()
        if p[0] <= self.RAIL_MARGIN or (p[0] <= 85.0 and v_in[0] < -100.0 and v_out[0] > 100.0):
            current_zones.add(Rail.LEFT)
        if p[0] >= TABLE_W - self.RAIL_MARGIN or (p[0] >= TABLE_W - 85.0 and v_in[0] > 100.0 and v_out[0] < -100.0):
            current_zones.add(Rail.RIGHT)
        if p[1] <= self.RAIL_MARGIN or (p[1] <= 85.0 and v_in[1] < -100.0 and v_out[1] > 100.0):
            current_zones.add(Rail.TOP)
        if p[1] >= TABLE_H - self.RAIL_MARGIN or (p[1] >= TABLE_H - 85.0 and v_in[1] > 100.0 and v_out[1] < -100.0):
            current_zones.add(Rail.BOTTOM)

        newly_entered = current_zones - self.in_rail_zone
        self.in_rail_zone = current_zones
        for rail in newly_entered:
            if (rail != self.last_rail_hit) or ((t - self.last_rail_time) >= 0.08):
                self.cushion_count += 1
                self.last_rail_hit = rail
                self.last_rail_time = t
                self.last_impulse_frame[cue_color] = h3.frame_idx
                self.event_log.append(f"[{t:.2f}s] 쿠션 접촉 #{self.cushion_count} ({rail.value})")

        normal_flipped = (v_in[0] * v_out[0] < -100.0) or (v_in[1] * v_out[1] < -100.0)
        if len(current_zones) > 0 and normal_flipped:
            self.last_impulse_frame[cue_color] = h3.frame_idx
        return ((len(current_zones) > 0) and normal_flipped) or ((t - self.last_rail_time) <= 0.10)

    def _register_cue_target_hit(
        self,
        t: float,
        target_color: BallColor,
        n_ab: np.ndarray,
        v_pre: Dict[BallColor, np.ndarray],
        v_post2: Dict[BallColor, np.ndarray]
    ):
        if self.first_target_ball is None:
            self.first_target_ball = target_color
            self.first_target_hit_time = t
            self.first_target_normal = n_ab
            if self.cushion_count >= 3:
                self.is_bank_shot = True
            self.event_log.append(f"[{t:.2f}s] 제1적구 접촉 ({target_color.name}, 현재 쿠션: {self.cushion_count})")
            self._check_double_hit_foul(t, v_pre, v_post2, self.history[5])
        elif target_color == self.first_target_ball:
            self.event_log.append(f"[{t:.2f}s] [수구-제1적구 재키스] {target_color.name} 재접촉 (득점 무관)")
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

    def _check_double_hit_foul(
        self,
        t_eval: float,
        v_pre: Dict[BallColor, np.ndarray],
        v_post2: Dict[BallColor, np.ndarray],
        h5: FrameSnapshot
    ):
        if self.double_hit_foul or self.active_cue_ball is None or self.first_target_ball is None:
            return
        if self.first_target_hit_time is None or self.first_target_normal is None:
            return
        if (self.first_target_hit_time - self.stroke_start_time) > self.DOUBLE_HIT_WINDOW_SEC:
            return
        if not (0.0 <= (t_eval - self.first_target_hit_time) <= 0.08):
            return

        cue = self.active_cue_ball
        target = self.first_target_ball
        if cue in self.initial_positions and target in self.initial_positions:
            init_dist = float(np.linalg.norm(self.initial_positions[target] - self.initial_positions[cue]))
            if init_dist > self.DOUBLE_HIT_INIT_DIST_MAX:
                return

        v_cue_pre = v_pre[cue]
        v_cue_post = v_post2[cue]
        v_target_post = v_post2[target]
        acc_cue = h5.acc[cue]
        n_ab = self.first_target_normal

        speed_cue_post = float(np.linalg.norm(v_cue_post))
        if speed_cue_post < 80.0:
            return

        forward_reaccel = float(np.dot(acc_cue, v_cue_post / speed_cue_post))
        normal_reaccel = float(np.dot(acc_cue, n_ab))
        is_reaccelerating = (forward_reaccel > self.DOUBLE_HIT_REACCEL_THRESH) and (normal_reaccel > self.DOUBLE_HIT_REACCEL_THRESH * 0.7)

        pre_speed = float(np.linalg.norm(v_cue_pre))
        head_on_cos = float(np.dot(v_cue_pre / max(pre_speed, 1e-6), n_ab)) if pre_speed > 50.0 else 0.0
        cue_normal_v_post = float(np.dot(v_cue_post, n_ab))
        target_normal_v_post = float(np.dot(v_target_post, n_ab))

        is_pushing_through = (
            head_on_cos > 0.85
            and target_normal_v_post > 150.0
            and cue_normal_v_post >= target_normal_v_post * self.DOUBLE_HIT_FOLLOW_RATIO
        )

        if is_reaccelerating or is_pushing_through:
            self.double_hit_foul = True
            self.state = ShotState.NO_SCORE_LOCKED
            self.fail_reason = "DOUBLE_HIT_FOUL (투터치/밀어치기 파울)"
            self.event_log.append(f"[{t_eval:.2f}s] ✖ 파울 감지: 투터치(Double Hit) 확정 -> NO_SCORE_LOCKED 전환")

    def _finalize_shot(self, t: float) -> ShotResult:
        scored = (self.state == ShotState.VALID_SCORE_PENDING) and (not self.double_hit_foul)
        if not scored and self.fail_reason is None:
            self.fail_reason = "NO_FIRST_BALL_HIT" if self.first_target_ball is None else "SECOND_BALL_MISSED"

        result = ShotResult(
            scored=scored,
            shooter_ball=self.active_cue_ball or self.assigned_cue_ball,
            cushions_at_second_hit=self.cushions_at_second_hit,
            total_cushions=self.cushion_count,
            is_bank_shot=self.is_bank_shot,
            object_ball_kiss_occurred=self.object_ball_kiss_occurred,
            double_hit_foul=self.double_hit_foul,
            centroid_shift_corrected_count=self.centroid_shift_corrected_count,
            gpt_occlusion_restored_frames=self.gpt_occlusion_restored_frames,
            kiss_id_swap_corrected=self.kiss_id_swap_corrected,
            fail_reason=None if scored else self.fail_reason,
            event_log=list(self.event_log)
        )
        self._reset_shot()
        return result
