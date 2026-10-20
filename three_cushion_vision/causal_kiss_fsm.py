"""
Module H: 3체 인과 충돌 분리, 투터치(Double Hit) 파울 감지 및 BilliardNanoGPT 통합 바운딩 박스 병합(Blob Merging) 보정 엔진
파일 경로: three_cushion_vision/causal_kiss_fsm.py
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Set, Tuple
import numpy as np
import torch

from three_cushion_vision.billiard_nanogpt import BALL_R, TABLE_H, TABLE_W, BilliardNanoGPT

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
    fail_reason: Optional[str] = None
    event_log: List[str] = field(default_factory=list)


class CausalKissSafeFSM:
    """
    1) 3체 인과 충돌 분리 (1적구-2적구 키스 및 클러스터 착시 차단)
    2) 투터치(Double Hit / 밀어치기 파울) 감지
    3) BilliardNanoGPT.rollout_occlusion 직접 연동을 통한 가려짐(Occlusion) 비선형 쿠션 반사 복원
       및 초근접 바운딩 박스 병합(Blob Merging) 중심점 점프(Centroid Shift) 억제
    가 모두 통합된 쓰리쿠션 비전 판정 엔진
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

        self.MIN_TARGET_DELTA_V = 22.0
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
        self.MAX_OCCLUSION_INTERP_FRAMES = 15        # NanoGPT 연동으로 최대 15프레임(0.25초) 비선형 복원 지원

        self.history: Deque[FrameSnapshot] = deque(maxlen=7)
        self.prev_raw_pos: Dict[BallColor, Tuple[np.ndarray, float]] = {}
        self.prev_ema_vel: Dict[BallColor, np.ndarray] = {}
        self.missing_streak: Dict[BallColor, int] = {c: 0 for c in BallColor}
        self.centroid_clamp_cooldown: Dict[BallColor, int] = {c: 0 for c in BallColor}

        # [핵심 통합] 각 당구공의 최근 15프레임 (x, y, vx, vy) 시퀀스를 저장하는 NanoGPT 전용 버퍼
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
        self.double_hit_foul = False
        self.centroid_shift_corrected_count = 0
        self.gpt_occlusion_restored_frames = 0
        self.fail_reason: Optional[str] = None
        self.event_log: List[str] = []

        self.stop_counter = 0
        self.in_rail_zone: Set[Rail] = set()
        self.last_rail_hit: Optional[Rail] = None
        self.last_rail_time = -999.0
        self.pair_last_hit_time: Dict[frozenset, float] = {}

    def _predict_next_state_with_gpt(self, color: BallColor, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        직전 프레임 버퍼(최대 15프레임)를 BilliardNanoGPT.rollout_occlusion()에 입력하여
        현재 프레임의 기대 위치 (pred_x, pred_y)와 기대 속도 (pred_vx, pred_vy)를 추론합니다.
        """
        buf = self.gpt_buffers[color]
        p_prev, _ = self.prev_raw_pos[color]
        v_prev = self.prev_ema_vel.get(color, np.zeros(2, dtype=np.float64))

        # 이동 중인 공이고 버퍼에 최소 2프레임 이상 쌓여 있으면 BilliardNanoGPT rollout 호출
        if len(buf) >= 2 and float(np.linalg.norm(v_prev)) > 25.0:
            hist_tensor = torch.tensor([list(buf)], dtype=torch.float32, device=self.device)
            pred_state = self.gpt_model.rollout_occlusion(hist_tensor, n_missing_frames=1)[0, 0].cpu().numpy()
            pred_pos = np.array([float(pred_state[0]), float(pred_state[1])], dtype=np.float64)
            pred_vel = np.array([float(pred_state[2]), float(pred_state[3])], dtype=np.float64)
            return pred_pos, pred_vel

        # 정지 상태이거나 초기 1프레임인 경우 경량 운동학 외삽 수행
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
        """
        [BilliardNanoGPT 통합 보정 필터]
        1) 결측 복원 (Occlusion): obs가 None일 때 BilliardNanoGPT.rollout_occlusion으로
           비선형 쿠션 반사 위치와 반전된 속도 벡터까지 복원합니다.
        2) 바운딩 박스 병합 억제 (Blob Merging): 두 공이 85mm 이내로 근접했을 때
           관측 중심점이 NanoGPT 예측 궤적 대비 상대 공 쪽으로 비정상 점프(>14mm)하면 클램핑합니다.
        """
        predicted_pos: Dict[BallColor, np.ndarray] = {}
        predicted_vel: Dict[BallColor, np.ndarray] = {}
        stabilized_pos: Dict[BallColor, np.ndarray] = {}
        gpt_restored_vel_override: Dict[BallColor, Optional[np.ndarray]] = {c: None for c in BallColor}

        # Step 1: 각 공의 NanoGPT 기반 예측 상태 계산 및 결측(None) 복원
        for color in BallColor:
            obs = raw_coords_mm.get(color, None)
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
                    # 가려진 구간에서 쿠션 반사 시 속도가 반전되므로 NanoGPT 예측 속도를 직접 반영
                    gpt_restored_vel_override[color] = predicted_vel[color]
                    self.gpt_occlusion_restored_frames += 1
                else:
                    return None
            else:
                self.missing_streak[color] = 0
                stabilized_pos[color] = np.array([float(obs[0]), float(obs[1])], dtype=np.float64)

        # Step 2: 근접 구간(d <= 85mm) 바운딩 박스 병합 중심점 끌림(Centroid Pull) 억제
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
                        # 이미 결측 복원된 공은 스킵
                        if raw_coords_mm.get(target_c, None) is None:
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
                            self.centroid_clamp_cooldown[target_c] = 5

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
                    # 가려짐 구간에서는 NanoGPT가 복원한 물리 반사 속도를 직접 사용
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

            # 매 프레임 복원/보정된 (x, y, vx, vy)를 NanoGPT 15프레임 버퍼에 축적
            self.gpt_buffers[color].append((float(p[0]), float(p[1]), float(v[0]), float(v[1])))

        self.history.append(FrameSnapshot(t=t, pos=pos_map, vel=vel_map, acc=acc_map))

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
                _, _, n_ab = self._subframe_ccd(f_prev.pos[cue_color], f_curr.pos[cue_color], f_prev.pos[target_color], f_curr.pos[target_color])
                self._register_cue_target_hit(t_eval, target_color, n_ab, f_pre2, f_post2)

        self._check_double_hit_foul(t_eval, f_pre2, f_curr, f_post2)

    def _check_double_hit_foul(self, t_eval: float, f_pre2: FrameSnapshot, f_curr: FrameSnapshot, f_post2: FrameSnapshot):
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

        v_cue_pre = f_pre2.vel[cue]
        if float(np.linalg.norm(v_cue_pre)) < 50.0:
            v_cue_pre = f_curr.vel[cue]
        v_cue_post = f_post2.vel[cue]
        v_target_post = f_post2.vel[target]
        acc_cue = f_post2.acc[cue]
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

        delta_v_other = f_post2.vel[other_ball] - f_pre2.vel[other_ball]
        other_received_impulse = float(np.linalg.norm(delta_v_other)) >= self.MIN_TARGET_DELTA_V

        deflection_deg_x = self._angle_between_deg(vx_pre, vx_post)
        recoil_along_normal_x = float(np.dot(delta_vx, -n_xy))
        x_deflected_Off_other_ball = (
            (d_x_other < d_min - 2.0 and float(np.dot(delta_vx, -n_x_other)) > recoil_along_normal_x)
            or (d_x_other <= self.GRAY_ZONE_MAX_DIST and other_received_impulse and not y_received_causal_impulse)
        )

        y_moving_unperturbed = (float(np.linalg.norm(vy_pre)) > 30.0) and (delta_vy_mag < 5.0)
        still_approaching_y = float(np.dot(vx_post - vy_post, n_xy)) > 0.0

        x_has_valid_deflection = (
            not x_near_rail
            and not x_deflected_Off_other_ball
            and not y_moving_unperturbed
            and not still_approaching_y
            and deflection_deg_x >= self.MIN_CUE_DEFLECTION_DEG
            and recoil_along_normal_x > 15.0
        )

        x_stopped_on_hard_contact = (
            (d_min <= self.HARD_CONTACT_DIST)
            and (not x_near_rail)
            and (not x_deflected_Off_other_ball)
            and (not y_moving_unperturbed)
            and (float(np.linalg.norm(f_post2.pos[ball_x] - f_curr.pos[ball_x])) < 2.0)
            and (recoil_along_normal_x > 15.0)
        )

        is_hard_overlap = (
            (d_min <= self.HARD_CONTACT_DIST)
            and (not x_near_rail)
            and (not x_deflected_Off_other_ball)
            and (y_received_causal_impulse or x_has_valid_deflection or x_stopped_on_hard_contact)
        )
        if is_hard_overlap or y_received_causal_impulse or x_has_valid_deflection:
            self.pair_last_hit_time[pair_key] = f_curr.t
            return True
        return False

    def _check_cushion_contact(self, t: float, f_prev: FrameSnapshot, f_curr: FrameSnapshot, f_post2: FrameSnapshot, cue_color: BallColor) -> bool:
        p = f_curr.pos[cue_color]
        v_in, v_curr, v_out = f_prev.vel[cue_color], f_curr.vel[cue_color], f_post2.vel[cue_color]
        current_zones: Set[Rail] = set()
        if p[0] <= self.RAIL_MARGIN or (p[0] <= 85.0 and v_in[0] < -100.0 and v_curr[0] > 100.0):
            current_zones.add(Rail.LEFT)
        if p[0] >= TABLE_W - self.RAIL_MARGIN or (p[0] >= TABLE_W - 85.0 and v_in[0] > 100.0 and v_curr[0] < -100.0):
            current_zones.add(Rail.RIGHT)
        if p[1] <= self.RAIL_MARGIN or (p[1] <= 85.0 and v_in[1] < -100.0 and v_curr[1] > 100.0):
            current_zones.add(Rail.TOP)
        if p[1] >= TABLE_H - self.RAIL_MARGIN or (p[1] >= TABLE_H - 85.0 and v_in[1] > 100.0 and v_curr[1] < -100.0):
            current_zones.add(Rail.BOTTOM)

        newly_entered = current_zones - self.in_rail_zone
        self.in_rail_zone = current_zones
        for rail in newly_entered:
            if (rail != self.last_rail_hit) or ((t - self.last_rail_time) >= 0.08):
                self.cushion_count += 1
                self.last_rail_hit = rail
                self.last_rail_time = t
                self.event_log.append(f"[{t:.2f}s] 쿠션 접촉 #{self.cushion_count} ({rail.value})")

        normal_flipped = (v_in[0] * v_out[0] < -100.0) or (v_in[1] * v_out[1] < -100.0)
        return ((len(current_zones) > 0) and normal_flipped) or ((t - self.last_rail_time) <= 0.10)

    def _register_cue_target_hit(self, t: float, target_color: BallColor, n_ab: np.ndarray, f_pre2: FrameSnapshot, f_post2: FrameSnapshot):
        if self.first_target_ball is None:
            self.first_target_ball = target_color
            self.first_target_hit_time = t
            self.first_target_normal = n_ab
            if self.cushion_count >= 3:
                self.is_bank_shot = True
            self.event_log.append(f"[{t:.2f}s] 제1적구 접촉 ({target_color.name}, 현재 쿠션: {self.cushion_count})")
            self._check_double_hit_foul(t, f_pre2, f_post2, f_post2)
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
            fail_reason=None if scored else self.fail_reason,
            event_log=list(self.event_log)
        )
        self._reset_shot()
        return result
