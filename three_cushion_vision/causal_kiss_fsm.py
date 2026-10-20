"""
Module H: 3체 인과 충돌 분리, 투터치 파울, 바운딩 박스 병합/NanoGPT 가려짐 복원 및 8대 악성 키스 오판 방어 엔진
파일 경로: three_cushion_vision/causal_kiss_fsm.py
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple
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
    subpixel_impacts: Dict[BallColor, bool] = field(default_factory=dict)
    intra_frame_rail_bounce: bool = False


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
    already_dispatched: bool = False


class CausalKissSafeFSM:
    """
    1적구-2적구 키스 후 득점 실패를 득점으로 오인하는 현상을 100% 차단하기 위해
    - Split-Window 무중첩 전후 속도 미분 및 이벤트 방화벽 (last_impulse_frame)
    - 뉴턴 제3법칙 운동량 보존 기반 적구 키스 충격량 차감 (Δv_C_net = Δv_C + Δv_B)
    - 무조건 거리 판정(is_hard_overlap) 폐지 및 쌍방 작용-반작용(AND 게이트) 검증
    - A -> B -> C 연쇄 간접 충돌(Chain Cannon) 기하학적 차폐
    - 적구 정면 키스 시 칼만 필터 ID 스왑(Ghost Pass-Through) 자동 역스왑 복원
    - 3쿠션 이상 충족 상태에서 제2적구 충돌 즉시 실시간 득점(+1) 발송 (on_immediate_score & score_dispatched)
    을 모두 탑재한 쓰리쿠션 판정 엔진
    """
    def __init__(
        self,
        fps: float = 60.0,
        assigned_cue_ball: BallColor = BallColor.WHITE,
        gpt_model: Optional[BilliardNanoGPT] = None,
        device: str = "cpu",
        on_immediate_score: Optional[Callable[[ShotResult], None]] = None,
    ):
        self.fps = fps
        self.assigned_cue_ball = assigned_cue_ball
        self.device = device
        self.gpt_model = (gpt_model or BilliardNanoGPT()).to(device).eval()
        self.on_immediate_score: Optional[Callable[[ShotResult], None]] = on_immediate_score

        self.STROKE_START_SPEED = 180.0
        self.STOP_SPEED = 15.0
        self.STOP_HOLD_FRAMES = int(fps * 0.25)

        self.HARD_CONTACT_DIST = BALL_DIAM + 1.5     # 63.0mm
        self.GRAY_ZONE_MAX_DIST = BALL_DIAM + 15.5   # 77.0mm
        self.RAIL_MARGIN = BALL_R + 14.0             # 44.75mm

        # [4대 쿠션 알고리즘 고도화 상수]
        # 1) 슈미트 트리거 히스테리시스 3단계 대역
        self.RAIL_ENTER_MARGIN = BALL_R + 23.25      # 54.0mm: 기본 쿠션 진입 경계
        self.RAIL_KINEMATIC_ZONE = BALL_R + 84.25    # 115.0mm: 시차 오차 보정용 광역 법선 반전 대역
        self.RAIL_EXIT_HYSTERESIS = BALL_R + 61.25   # 92.0mm: 재카운트 허용을 위한 슈미트 트리거 이탈 경계
        # 2) 코너 핑퐁 방지 및 최저 속도 게이트
        self.MIN_CUSHION_SPEED = 65.0                # 65mm/s 미만 저속 떨림 시 쿠션 카운트 차단
        self.MIN_CUSHION_BOUNCE_SPEED = 65.0         # 최저 유효 쿠션 반사 속도 (mm/s)
        self.MIN_RAIL_TRAVEL_DIST = 80.0             # 동일 레일 재장전/재타격 인정 최소 이동 변위 (mm)
        self.CORNER_MIN_TRAVEL_DIST = 22.0           # 코너 인접 레일 연속 타격 시 최소 이동 변위 (mm)
        self.PER_RAIL_COOLDOWN_SEC = 0.30            # 개별 레일 자체 쿨다운 (초)

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

        # [4대 쿠션 알고리즘 고도화 상태 변수]
        self.rail_armed: Dict[Rail, bool] = {r: True for r in Rail}
        self.per_rail_last_time: Dict[Rail, float] = {r: -999.0 for r in Rail}
        self.last_cushion_hit_pos: Optional[np.ndarray] = None
        self.cue_trajectory_log: List[Tuple[float, np.ndarray, np.ndarray, float]] = []  # (t, pos, vel, min_dist_to_targets)

        # [핵심 방화벽 1] 각 공이 마지막으로 충돌/반사 이벤트를 겪은 프레임 번호 기록 (이전 충격파 누수 차단)
        self.last_impulse_frame: Dict[BallColor, int] = {c: -1 for c in BallColor}
        # [핵심 방화벽 2] 적구 간 정면 키스 시 트래커 ID 스왑 보정 매핑 테이블
        self.id_swap_map: Dict[BallColor, BallColor] = {c: c for c in BallColor}
        self._last_id_swap_check_time: float = -999.0
        self.centroid_clamp_cooldown: Dict[BallColor, int] = {c: 0 for c in BallColor}
        # [즉시 득점 발송 상태] 제2적구 충돌 즉시 점수판 반영 여부 및 해당 프레임 즉시 반환용 결과
        self.score_dispatched: bool = False
        self._pending_immediate_result: Optional[ShotResult] = None
        self.idle_anchor_pos: Dict[BallColor, np.ndarray] = {}

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

    def _find_nearest_rail(self, p: np.ndarray) -> Rail:
        """주어진 (x, y) 좌표에서 가장 가까운 당구대 4면 쿠션 레일(LEFT, RIGHT, TOP, BOTTOM)을 반환합니다."""
        dists = {
            Rail.LEFT: float(p[0]),
            Rail.RIGHT: float(TABLE_W - p[0]),
            Rail.TOP: float(p[1]),
            Rail.BOTTOM: float(TABLE_H - p[1]),
        }
        return min(dists.keys(), key=lambda r: dists[r])

    def _apply_vision_precision_signals(
        self,
        t: float,
        pos_map: Dict[BallColor, np.ndarray],
        vel_map: Dict[BallColor, np.ndarray],
        subpixel_impacts: Optional[Dict[BallColor, bool]],
        intra_frame_rail_bounce: bool,
    ) -> None:
        """
        [4대 정밀 비전 신호 최우선 반영]
        1) intra_frame_rail_bounce == True (MotionStreakVApexDetector 또는 CushionRimPixelTrigger 감지):
           프레임 사이에서 쿠션을 찍고 나온 V자 잔상/픽셀 터치를 인정하여 누락 없이 cushion_count += 1 수행
        2) subpixel_impacts[target_color] == True (SubPixelPhaseImpactDetector 0.06px 이상 위상 진동 감지):
           수구가 목적구 근방(d_min <= 95.0mm, 완화 적용)을 지날 때 좌표상 속도 변화가 0이더라도 즉시 접촉(득점) 인정
        """
        cue_color = self.active_cue_ball or self.assigned_cue_ball
        if cue_color is None or cue_color not in pos_map:
            return

        # 1. 프레임 내 모션 블러 V자 꺾임(intra_frame_rail_bounce) 기반 쿠션 카운트 즉시 반영
        if intra_frame_rail_bounce and self.state != ShotState.VALID_SCORE_PENDING:
            nearest_rail = self._find_nearest_rail(pos_map[cue_color])
            if self.rail_armed.get(nearest_rail, True) and (
                (nearest_rail != self.last_rail_hit)
                or ((t - self.per_rail_last_time.get(nearest_rail, -999.0)) >= self.PER_RAIL_COOLDOWN_SEC)
            ):
                self.cushion_count += 1
                self.rail_armed[nearest_rail] = False
                self.per_rail_last_time[nearest_rail] = t
                self.last_rail_hit = nearest_rail
                self.last_rail_time = t
                self.last_cushion_hit_pos = pos_map[cue_color].copy()
                self.last_impulse_frame[cue_color] = self.frame_counter
                self.event_log.append(
                    f"[{t:.2f}s] [V-Apex/Rim 정밀비전] 프레임 내 쿠션 반사 #{self.cushion_count} ({nearest_rail.value})"
                )

        # 2. 푸리에 위상 상관법(cv2.phaseCorrelate) 초미세 스침(subpixel_impacts) 기반 즉시 접촉 판정
        if subpixel_impacts:
            prev_snap = self.history[-2] if len(self.history) >= 2 else self.history[-1]
            for target_color, impacted in subpixel_impacts.items():
                if not impacted or target_color == cue_color or target_color not in pos_map:
                    continue
                pair_key = frozenset([cue_color, target_color])
                if (t - self.pair_last_hit_time.get(pair_key, -999.0)) < self.PAIR_COOLDOWN_SEC:
                    continue

                d_min, _, n_ab = self._subframe_ccd(
                    prev_snap.pos[cue_color],
                    pos_map[cue_color],
                    prev_snap.pos[target_color],
                    pos_map[target_color],
                )
                if d_min <= 95.0:
                    self.pair_last_hit_time[pair_key] = t
                    self.last_impulse_frame[cue_color] = self.frame_counter
                    self.last_impulse_frame[target_color] = self.frame_counter
                    self.event_log.append(
                        f"[{t:.2f}s] [SubPixel 위상감지] {target_color.name} 초미세 접촉 확정 (d_min={d_min:.1f}mm <= 95.0mm)"
                    )
                    self._register_cue_target_hit(t, target_color, n_ab, vel_map, vel_map)

    def update(
        self,
        t: float,
        coords_mm: Dict[BallColor, Optional[Tuple[float, float]]],
        subpixel_impacts: Optional[Dict[BallColor, bool]] = None,
        intra_frame_rail_bounce: bool = False,
    ) -> Optional[ShotResult]:
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
        self.history.append(
            FrameSnapshot(
                frame_idx=self.frame_counter,
                t=t,
                pos=pos_map,
                vel=vel_map,
                acc=acc_map,
                subpixel_impacts=dict(subpixel_impacts) if subpixel_impacts else {},
                intra_frame_rail_bounce=bool(intra_frame_rail_bounce),
            )
        )

        if self.state == ShotState.IDLE:
            for c in BallColor:
                spd_c = float(np.linalg.norm(vel_map[c]))
                if c not in self.idle_anchor_pos:
                    prev_s = self.history[-2] if len(self.history) >= 2 else self.history[-1]
                    self.idle_anchor_pos[c] = prev_s.pos[c].copy()
                elif spd_c < 45.0:
                    self.idle_anchor_pos[c] = 0.85 * self.idle_anchor_pos[c] + 0.15 * pos_map[c]

            fastest_color = max(vel_map.keys(), key=lambda c: float(np.linalg.norm(vel_map[c])))
            fastest_speed = float(np.linalg.norm(vel_map[fastest_color]))
            anchor_p = self.idle_anchor_pos.get(
                fastest_color,
                self.history[0].pos[fastest_color],
            )
            displacement = float(np.linalg.norm(pos_map[fastest_color] - anchor_p))
            min_disp_req = 18.0 if len(self.history) >= 4 else 6.0

            if fastest_speed >= self.STROKE_START_SPEED and displacement >= min_disp_req:
                saved_anchors = {c: self.idle_anchor_pos[c].copy() for c in BallColor if c in self.idle_anchor_pos}
                self._reset_shot()
                self.stroke_start_time = t
                self.active_cue_ball = fastest_color
                prev_snap = self.history[-2] if len(self.history) >= 2 else self.history[-1]
                self.initial_positions = (
                    saved_anchors
                    if len(saved_anchors) == 3
                    else {c: prev_snap.pos[c].copy() for c in BallColor}
                )
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

                if self.state == ShotState.IN_PLAY and (subpixel_impacts or intra_frame_rail_bounce):
                    self._apply_vision_precision_signals(
                        t, pos_map, vel_map, subpixel_impacts, intra_frame_rail_bounce
                    )
                    if self._pending_immediate_result is not None:
                        instant_res = self._pending_immediate_result
                        self._pending_immediate_result = None
                        return instant_res
            return None

        # 정밀 비전 신호(SubPixel 위상 진동 / V-Apex 쿠션 반사)가 들어온 경우 최우선 반영
        if subpixel_impacts or intra_frame_rail_bounce:
            self._apply_vision_precision_signals(
                t, pos_map, vel_map, subpixel_impacts, intra_frame_rail_bounce
            )

        if len(self.history) >= 6:
            self._evaluate_frame_with_context()

        all_stopped = all(float(np.linalg.norm(v)) < self.STOP_SPEED for v in vel_map.values())
        self.stop_counter = (self.stop_counter + 1) if all_stopped else 0

        if self.stop_counter >= self.STOP_HOLD_FRAMES:
            return self._finalize_shot(t)

        # 제2적구 충돌로 즉시 득점이 발생한 해당 프레임이면 공이 멈추기를 기다리지 않고 즉시 ShotResult 반환
        if self._pending_immediate_result is not None:
            instant_res = self._pending_immediate_result
            self._pending_immediate_result = None
            return instant_res

        return None

    def _evaluate_frame_with_context(self):
        h2, h3, h5 = self.history[2], self.history[3], self.history[5]
        t_eval = h3.t
        cue_color = self.active_cue_ball
        if cue_color is None:
            return

        v_pre, v_post1, v_post2 = self._compute_split_velocities()

        other_balls = [c for c in BallColor if c != cue_color]
        b1, b2 = other_balls[0], other_balls[1]

        # 수구 궤적 로그(cue_trajectory_log) 기록: (t, pos, vel, min_dist_to_targets)
        min_dist_to_targets = min(
            float(np.linalg.norm(h3.pos[cue_color] - h3.pos[b1])),
            float(np.linalg.norm(h3.pos[cue_color] - h3.pos[b2])),
        )
        self.cue_trajectory_log.append(
            (t_eval, h3.pos[cue_color].copy(), v_post1[cue_color].copy(), min_dist_to_targets)
        )

        # 1. 수구 쿠션 접촉 판정 (슈미트 트리거 히스테리시스 + 광역 궤적 변곡점 + 코너 핑퐁 방지 + 득점 후 동결)
        cue_near_rail = self._check_cushion_contact(
            t_eval, h2, h3, v_pre[cue_color], v_post2[cue_color], cue_color, min_dist_to_targets
        )

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

    def _line_fit_rmse_and_kink(self, ball: BallColor) -> Tuple[float, np.ndarray, np.ndarray, float]:
        """
        최근 프레임들의 궤적에 대해:
        1) 단일 등속 직선 모델 p(t) = p0 + v*t 로 피팅했을 때의 직선 잔차(line_rmse_mm)를 계산하고,
        2) 충돌 전 구간(history[-6:-3])과 충돌 후 구간(history[-3:]) 각각의 선형 회귀 속도 벡터(v_pre, v_post) 및
           굴절각(kink_deg)을 반환합니다.
        평범한 득점 실패(안 맞고 직진 통과)는 line_rmse_mm < 2.0mm 이고 kink_deg ≈ 0° 로 나타납니다.
        """
        snaps = list(self.history)[-6:]
        t_arr = np.array([s.t for s in snaps], dtype=np.float64)
        p_arr = np.array([s.pos[ball] for s in snaps], dtype=np.float64)  # (N, 2)

        t_mean = np.mean(t_arr)
        p_mean = np.mean(p_arr, axis=0)
        dt_arr = t_arr - t_mean
        denom = float(np.sum(dt_arr ** 2))
        if denom < 1e-8:
            return 0.0, np.zeros(2), np.zeros(2), 0.0

        # 전체 구간 단일 직선 최소자승(OLS) 기울기 및 잔차 RMSE 계산
        v_ols = np.sum(dt_arr[:, None] * (p_arr - p_mean), axis=0) / denom
        p_pred_line = p_mean + dt_arr[:, None] * v_ols
        residuals = np.linalg.norm(p_arr - p_pred_line, axis=1)
        line_rmse_mm = float(np.sqrt(np.mean(residuals ** 2)))

        # 전반부(충돌 전) / 후반부(충돌 후) 구간 평균 속도 벡터
        pre_snaps = snaps[:3]
        post_snaps = snaps[-3:]
        dt_pre = max(pre_snaps[-1].t - pre_snaps[0].t, 1e-4)
        dt_post = max(post_snaps[-1].t - post_snaps[0].t, 1e-4)

        v_pre_reg = (pre_snaps[-1].pos[ball] - pre_snaps[0].pos[ball]) / dt_pre
        v_post_reg = (post_snaps[-1].pos[ball] - post_snaps[0].pos[ball]) / dt_post
        kink_deg = self._angle_between_deg(v_pre_reg, v_post_reg)

        return line_rmse_mm, v_pre_reg, v_post_reg, kink_deg

    def _verify_pair_collision(
        self,
        ball_x: BallColor,
        ball_y: BallColor,
        f_pre2: Optional[FrameSnapshot] = None,
        f_prev: Optional[FrameSnapshot] = None,
        f_curr: Optional[FrameSnapshot] = None,
        f_post2: Optional[FrameSnapshot] = None,
        x_near_rail: bool = False,
        other_ball: Optional[BallColor] = None,
        subpixel_phase_hit: bool = False,
        *,
        v_pre: Optional[Dict[BallColor, np.ndarray]] = None,
        v_post1: Optional[Dict[BallColor, np.ndarray]] = None,
        v_post2: Optional[Dict[BallColor, np.ndarray]] = None,
        is_cue_second_target_check: bool = False,
        bc_kiss_this_frame: bool = False,
    ) -> bool:
        """
        [평범한 실패 오판(False Positive) 원천 차단 + 3체 인과 키스 분리 통합 버전]
        - 단순 거리 중첩(is_hard_overlap)만으로 무조건 True를 반환하던 버그를 폐지합니다.
        - 대각선 시차로 인해 d_min <= 63mm가 찍히거나 바운딩 박스 노이즈(1~2px) 및 회전 휨(2~3°)이 있더라도,
          1) 피타격구(Y)의 정지 앵커 대비 영구 순이동(>= 5.2mm) + 법선 충격 속도(>= 45mm/s)가 입증되거나,
          2) 수구(X)의 6프레임 직선 회귀가 깨지며(line_rmse >= 1.6mm) 실제 접촉 반경(d_min <= 66.5mm)에서
             법선 반대 방향으로 명확히 꺾인 경우(>= 4.0°)에만 충돌로 인정합니다.
        """
        h2 = f_prev if f_prev is not None else self.history[2]
        h3 = f_curr if f_curr is not None else self.history[3]
        t_eval = h3.t

        if other_ball is None:
            remaining = [c for c in BallColor if c not in (ball_x, ball_y)]
            other_ball = remaining[0]

        pair_key = frozenset([ball_x, ball_y])
        if (t_eval - self.pair_last_hit_time.get(pair_key, -999.0)) < self.PAIR_COOLDOWN_SEC:
            return False

        d_min, _, n_xy = self._subframe_ccd(h2.pos[ball_x], h3.pos[ball_x], h2.pos[ball_y], h3.pos[ball_y])
        if d_min > 75.0:
            return False

        # 제3의 공(other_ball)과의 최근접 거리 및 법선 벡터 계산
        d_x_other, _, n_x_other = self._subframe_ccd(h2.pos[ball_x], h3.pos[ball_x], h2.pos[other_ball], h3.pos[other_ball])
        d_y_other, _, n_other_y = self._subframe_ccd(h2.pos[other_ball], h3.pos[other_ball], h2.pos[ball_y], h3.pos[ball_y])

        # 1. 공격구(X) 및 피타격구(Y)의 6프레임 단일 직선 회귀 잔차(line_rmse_mm)와 굴절각(kink_deg) 추출
        line_rmse_x, vx_pre_reg, vx_post_reg, kink_deg_x = self._line_fit_rmse_and_kink(ball_x)
        _, vy_pre_reg, vy_post_reg, _ = self._line_fit_rmse_and_kink(ball_y)

        if v_pre is None or v_post1 is None or v_post2 is None:
            v_pre_c, v_post1_c, v_post2_c = self._compute_split_velocities()
            v_pre = v_pre or v_pre_c
            v_post1 = v_post1 or v_post1_c
            v_post2 = v_post2 or v_post2_c

        vx_pre, vx_post = v_pre[ball_x], v_post2[ball_x]
        vy_pre, vy_post = v_pre[ball_y], v_post2[ball_y]
        vy_post_immediate = v_post1[ball_y]
        vx_post_immediate = v_post1[ball_x]

        delta_vx = vx_post - vx_pre
        delta_vy = vy_post - vy_pre
        if self.centroid_clamp_cooldown.get(ball_y, 0) > 0:
            delta_vy = np.zeros(2, dtype=np.float64)
            vy_post_immediate = vy_pre.copy()

        # 2. 피타격구(Y)의 접근 전 정지 앵커(Anchor) 대비 영구 순변위(Net Displacement) 계산
        snaps = list(self.history)
        anchor_y = np.mean([s.pos[ball_y] for s in snaps[:2]], axis=0)
        post_y = np.mean([s.pos[ball_y] for s in snaps[-2:]], axis=0)
        net_disp_vec_y = post_y - anchor_y
        net_disp_mag_y = float(np.linalg.norm(net_disp_vec_y))
        net_disp_along_normal_y = float(np.dot(net_disp_vec_y, n_xy))

        # -----------------------------------------------------------------
        # [방어 A] A -> B -> C 간접 연쇄 키스(Chain Cannon) 차폐
        # -----------------------------------------------------------------
        delta_v_other = v_post2[other_ball] - v_pre[other_ball]
        if is_cue_second_target_check and d_y_other <= self.GRAY_ZONE_MAX_DIST:
            vec_xy = h3.pos[ball_y] - h3.pos[ball_x]
            vec_x_other = h3.pos[other_ball] - h3.pos[ball_x]
            dist_xy = max(float(np.linalg.norm(vec_xy)), 1e-6)
            proj_ratio = float(np.dot(vec_x_other, vec_xy / dist_xy)) / dist_xy

            other_is_between = (0.0 < proj_ratio < 1.05) and (d_x_other <= d_min + 6.0)
            if other_is_between or bc_kiss_this_frame:
                recoil_along_other = float(np.dot(delta_vx, -n_x_other))
                recoil_along_y = float(np.dot(delta_vx, -n_xy))
                if d_x_other < d_min - 1.5 and recoil_along_other >= recoil_along_y * 0.9:
                    return False

        # -----------------------------------------------------------------
        # [방어 B] 뉴턴 제3법칙 운동량 보존 기반 적구 키스 충격량 차감
        # -----------------------------------------------------------------
        delta_vy_net = delta_vy.copy()
        if d_y_other <= self.GRAY_ZONE_MAX_DIST:
            other_pushed_y = float(np.dot(delta_v_other, -n_other_y)) > 18.0 and float(np.dot(delta_vy, n_other_y)) > 18.0
            if other_pushed_y:
                transferred_from_other = float(np.dot(-delta_v_other, n_other_y)) * n_other_y
                delta_vy_net = delta_vy - transferred_from_other
                if is_cue_second_target_check:
                    self.object_ball_kiss_occurred = True
                    self.last_object_kiss_time = t_eval

        # -----------------------------------------------------------------
        # [방어 C] 즉각 충격 일관성 검증 (미래 프레임 h[4]->h[5] 충돌의 역방향 누수 차단)
        # -----------------------------------------------------------------
        delta_vy_imm = vy_post_immediate - vy_pre
        delta_vx_imm = vx_post_immediate - vx_pre

        impulse_y_normal = float(np.dot(delta_vy_net, n_xy))
        impulse_y_imm_normal = float(np.dot(delta_vy_imm, n_xy))
        delta_vy_net_mag = float(np.linalg.norm(delta_vy_net))

        recoil_x_normal = float(np.dot(delta_vx, -n_xy))
        recoil_x_imm_normal = float(np.dot(delta_vx_imm, -n_xy))
        deflection_deg_x = max(self._angle_between_deg(vx_pre, vx_post), kink_deg_x)

        # [0쿠션 초기 그레이존 엄격화]
        if (
            is_cue_second_target_check
            and self.cushion_count == 0
            and self.first_target_hit_time is not None
            and (t_eval - self.first_target_hit_time) <= 0.18
        ):
            if d_min > 64.0 and impulse_y_normal < 42.0:
                return False

        closing_speed = float(np.dot(vx_pre - vy_pre, n_xy))
        if closing_speed < -15.0:
            return False

        x_deflected_off_other = (
            (
                d_x_other < d_min - 1.5
                and float(np.dot(delta_vx, -n_x_other)) > max(recoil_x_normal * 0.85, 15.0)
            )
            or (
                d_x_other <= 72.0
                and float(np.linalg.norm(delta_v_other)) >= 35.0
                and net_disp_mag_y < 3.0
            )
            or bc_kiss_this_frame
        )
        if x_deflected_off_other and net_disp_mag_y < 5.2:
            return False

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
        # Case 1: 적구 간 키스가 있었거나, 제2적구(Y)가 이미 굴러가고 있는 상태 (speed_y_pre >= 35mm/s 등)
        if is_cue_second_target_check and (speed_y_pre >= 20.0 or recent_kiss_contaminating):
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

        # Case 2: 피타격구(Y)가 정지 상태(speed_y_pre < 35mm/s)일 때
        # [증거 1] 피타격구(Y)가 실제 물리적 타격을 받아 정지 앵커 대비 영구 이동(>= 5.2mm)하고 법선 충격 속도가 발생했는가?
        y_truly_moved = (
            (
                y_impulse_is_immediate
                and net_disp_mag_y >= 5.2
                and net_disp_along_normal_y >= 3.5
                and delta_vy_net_mag >= 45.0
                and impulse_y_normal >= 30.0
            )
            or (subpixel_phase_hit and d_min <= 68.0)
        ) and (d_min <= d_y_other + 3.0)

        # [증거 2] 목적구(Y)가 거의 안 밀린 얇은 스침(박박이)일 때, 수구(X)의 6프레임 직선이 실제로 부러졌는가?
        # 단순 스핀 휘어짐이나 노이즈(1.8°)에 속지 않도록 실제 접촉 가능 거리(d_min <= 66.5mm) 내에서만 허용하되
        # 직선 회귀 오차가 깨지고(line_rmse_x >= 1.6mm) 굴절각 >= 4.0° 이며 Y에도 최소한의 반작용 미동이 있을 때만 인정!
        x_truly_deflected = (
            d_min <= 66.5
            and (not x_near_rail)
            and (not x_deflected_off_other)
            and x_recoil_is_immediate
            and (impulse_y_normal >= 8.0 or net_disp_mag_y >= 1.0 or subpixel_phase_hit)
            and (
                (deflection_deg_x >= 4.0 and line_rmse_x >= 1.6 and recoil_x_normal >= 30.0)
                or (d_min <= 62.5 and recoil_x_normal >= 60.0)
            )
        )

        # 정면 타격 후 수구 급정지 케이스 (d_min <= 60mm 이고 수구가 정면에서 급감속 정지)
        head_on_stop_verified = (
            d_min <= 60.0
            and closing_speed >= 400.0
            and recoil_x_normal >= closing_speed * 0.75
            and float(np.linalg.norm(vx_post)) < 50.0
            and (not recent_kiss_contaminating)
        )

        # [최종 판정] 시차로 인해 d_min <= 60mm로 겹쳐 보여도,
        # 수구가 직진(not x_truly_deflected)하고 목적구도 제자리에 가만히(not y_truly_moved) 있으면 100% 노카운트(False)!
        if y_truly_moved or x_truly_deflected or head_on_stop_verified:
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
        cue_color: BallColor,
        min_dist_to_targets: float = 9999.0,
    ) -> bool:
        """
        [4대 쿠션 감지 알고리즘 고도화]
        1) 슈미트 트리거 히스테리시스(rail_armed): 92.0mm 이상 이탈 또는 마지막 타격점 대비 80.0mm 이상 이동 시에만 재장전
        2) 코너 핑퐁(Corner Ping-Pong) 방지 및 최저 속도 게이트(MIN_CUSHION_SPEED = 65.0mm/s)
        3) 광역 궤적 변곡점(Kinematic Bounce Apex): 54.0mm ~ 115.0mm 구간에서 v_norm_in < -60, v_norm_out > +45 및 목적구 거리 >= 90mm 시 인정
        4) 득점 시점 쿠션 동결(Freeze): VALID_SCORE_PENDING 진입 이후 사후 쿠션 카운트 차단
        """
        p = h3.pos[cue_color]
        rail_dists: Dict[Rail, float] = {
            Rail.LEFT: float(p[0]),
            Rail.RIGHT: float(TABLE_W - p[0]),
            Rail.TOP: float(p[1]),
            Rail.BOTTOM: float(TABLE_H - p[1]),
        }

        # ① 슈미트 트리거 히스테리시스 재장전 (92.0mm 이상 멀어지거나 마지막 쿠션 타격 위치로부터 80.0mm 이상 이동 시 재활성화)
        dist_from_last_hit = (
            float(np.linalg.norm(p - self.last_cushion_hit_pos))
            if self.last_cushion_hit_pos is not None
            else 9999.0
        )
        for r in Rail:
            if not self.rail_armed[r]:
                if rail_dists[r] >= self.RAIL_EXIT_HYSTERESIS or (
                    rail_dists[r] > self.RAIL_ENTER_MARGIN and dist_from_last_hit >= self.MIN_RAIL_TRAVEL_DIST
                ):
                    self.rail_armed[r] = True

        # 각 레일 기준 법선 속도 (레일로 접근할 때 음수 < 0, 레일에서 반사되어 나올 때 양수 > 0)
        norm_vels: Dict[Rail, Tuple[float, float]] = {
            Rail.LEFT: (float(v_in[0]), float(v_out[0])),
            Rail.RIGHT: (float(-v_in[0]), float(-v_out[0])),
            Rail.TOP: (float(v_in[1]), float(v_out[1])),
            Rail.BOTTOM: (float(-v_in[1]), float(-v_out[1])),
        }

        # ② 최저 속도 게이트: 65.0mm/s 미만 저속 정지 임박 구간에서는 바운딩 박스 노이즈 쿠션 카운트 완전 차단
        speed_max = max(float(np.linalg.norm(v_in)), float(np.linalg.norm(v_out)))
        if speed_max < self.MIN_CUSHION_SPEED:
            return False

        # ③ 근접 진입 대역(<= 54.0mm) 및 광역 궤적 변곡점(54.0mm ~ 115.0mm) 후보 추출
        current_zones: Set[Rail] = set()
        kinematic_bounce_rails: Set[Rail] = set()

        for r in Rail:
            d_r = rail_dists[r]
            vn_in, vn_out = norm_vels[r]
            # (A) 근접 진입 대역 (<= 54.0mm): 접근 후 반사 또는 마진 진입
            if d_r <= self.RAIL_ENTER_MARGIN and (vn_in < -15.0 or vn_out > 15.0):
                current_zones.add(r)
            # (B) 광역 궤적 변곡점 대역 (<= 115.0mm): 법선 속도 부호 반전(vn_in < -60, vn_out > +45) + 두 목적구 거리 >= 90.0mm
            if (
                d_r <= self.RAIL_KINEMATIC_ZONE
                and vn_in < -60.0
                and vn_out > 45.0
                and min_dist_to_targets >= 90.0
            ):
                current_zones.add(r)
                kinematic_bounce_rails.add(r)

        newly_entered = current_zones - self.in_rail_zone
        candidate_rails = newly_entered | (kinematic_bounce_rails & {r for r in Rail if self.rail_armed[r]})
        self.in_rail_zone = current_zones

        # ④ 득점 확정(VALID_SCORE_PENDING) 이후에는 쿠션 수를 동결(Freeze)하여 사후 쿠션을 더하지 않음!
        if self.state != ShotState.VALID_SCORE_PENDING:
            for rail in candidate_rails:
                if not self.rail_armed[rail]:
                    continue
                if (t - self.per_rail_last_time[rail]) < self.PER_RAIL_COOLDOWN_SEC:
                    continue

                cur_travel = (
                    float(np.linalg.norm(p - self.last_cushion_hit_pos))
                    if self.last_cushion_hit_pos is not None
                    else 9999.0
                )
                vn_in, vn_out = norm_vels[rail]
                has_true_normal_reversal = (vn_in < -45.0) and (vn_out > 35.0)

                # 직전 타격 레일과 동일한 레일은 다른 레일이나 목적구 충돌 전까지 연속 카운트 원천 차단
                if rail == self.last_rail_hit:
                    continue

                # 코너 핑퐁 방지: 서로 다른 레일 간에도 최소 이동 변위(22.0mm) 또는 실제 법선 속도 반전이 확인될 때만 카운트
                if (
                    rail != self.last_rail_hit
                    and self.last_cushion_hit_pos is not None
                    and cur_travel < self.CORNER_MIN_TRAVEL_DIST
                    and not has_true_normal_reversal
                ):
                    continue

                self.cushion_count += 1
                self.rail_armed[rail] = False
                self.per_rail_last_time[rail] = t
                self.last_rail_hit = rail
                self.last_rail_time = t
                self.last_cushion_hit_pos = p.copy()
                self.last_impulse_frame[cue_color] = h3.frame_idx
                self.event_log.append(f"[{t:.2f}s] 쿠션 접촉 #{self.cushion_count} ({rail.value})")

        normal_flipped = (v_in[0] * v_out[0] < -100.0) or (v_in[1] * v_out[1] < -100.0)
        if len(current_zones) > 0 and normal_flipped:
            self.last_impulse_frame[cue_color] = h3.frame_idx
        return ((len(current_zones) > 0) and normal_flipped) or ((t - self.last_rail_time) <= 0.10)

    def _retroactive_cushion_audit(self) -> int:
        """
        [④ 궤적 역추적 쿠션 복원 (Retroactive Bounce Audit)]
        제2적구 접촉 순간 self.cushion_count < 3 일 때 즉시 NO_SCORE_LOCKED로 죽이지 않고,
        지금까지 기록된 cue_trajectory_log에서 가장자리(135.0mm 이내) 법선 속도 반전(변곡점) 횟수를
        역추적 재검산하여 누락된 쿠션을 복원합니다.
        """
        log = self.cue_trajectory_log
        if len(log) < 3:
            return self.cushion_count

        AUDIT_ZONE = 135.0
        audited_count = 0
        audit_armed: Dict[Rail, bool] = {r: True for r in Rail}
        audit_last_t: Dict[Rail, float] = {r: -999.0 for r in Rail}
        audit_last_pos: Optional[np.ndarray] = None
        audit_last_rail: Optional[Rail] = None

        for idx in range(1, len(log) - 1):
            t_cur, p_cur, _, min_t_dist = log[idx]
            p_prev = log[idx - 1][1]
            p_next = log[idx + 1][1]
            dt1 = max(log[idx][0] - log[idx - 1][0], 1e-4)
            dt2 = max(log[idx + 1][0] - log[idx][0], 1e-4)
            v_in = (p_cur - p_prev) / dt1
            v_out = (p_next - p_cur) / dt2

            if max(float(np.linalg.norm(v_in)), float(np.linalg.norm(v_out))) < self.MIN_CUSHION_SPEED:
                continue

            r_dists = {
                Rail.LEFT: float(p_cur[0]),
                Rail.RIGHT: float(TABLE_W - p_cur[0]),
                Rail.TOP: float(p_cur[1]),
                Rail.BOTTOM: float(TABLE_H - p_cur[1]),
            }
            for r in Rail:
                if not audit_armed[r] and r_dists[r] >= self.RAIL_EXIT_HYSTERESIS:
                    audit_armed[r] = True

            if min_t_dist < 85.0:
                continue

            n_vels = {
                Rail.LEFT: (float(v_in[0]), float(v_out[0])),
                Rail.RIGHT: (float(-v_in[0]), float(-v_out[0])),
                Rail.TOP: (float(v_in[1]), float(v_out[1])),
                Rail.BOTTOM: (float(-v_in[1]), float(-v_out[1])),
            }
            for r in Rail:
                if not audit_armed[r]:
                    continue
                if r_dists[r] > AUDIT_ZONE:
                    continue
                vn_in, vn_out = n_vels[r]
                if vn_in < -45.0 and vn_out > 35.0:
                    if (t_cur - audit_last_t[r]) < self.PER_RAIL_COOLDOWN_SEC:
                        continue
                    travel = (
                        float(np.linalg.norm(p_cur - audit_last_pos))
                        if audit_last_pos is not None
                        else 9999.0
                    )
                    if r == audit_last_rail and travel < self.MIN_RAIL_TRAVEL_DIST:
                        continue
                    if r != audit_last_rail and audit_last_pos is not None and travel < self.CORNER_MIN_TRAVEL_DIST:
                        continue
                    audited_count += 1
                    audit_armed[r] = False
                    audit_last_t[r] = t_cur
                    audit_last_pos = p_cur.copy()
                    audit_last_rail = r

        return max(self.cushion_count, audited_count)

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
            # 제2적구 접촉 순간 cushion_count < 3 이라면 역추적 구제(_retroactive_cushion_audit) 실행!
            if self.cushion_count < 3:
                audited_count = self._retroactive_cushion_audit()
                if audited_count > self.cushion_count:
                    self.event_log.append(
                        f"[{t:.2f}s] [궤적 역추적 쿠션 복원] 누락 쿠션 복구 ({self.cushion_count}회 -> {audited_count}회)"
                    )
                    self.cushion_count = audited_count

            self.second_target_ball = target_color
            # 득점 즉시 쿠션 수 동결(Freeze): 제2적구 충돌 시점의 쿠션 수로 확정
            self.cushions_at_second_hit = self.cushion_count
            if self.state == ShotState.IN_PLAY:
                if self.cushion_count >= 3 and not self.double_hit_foul:
                    self.state = ShotState.VALID_SCORE_PENDING
                    self.event_log.append(
                        f"[{t:.2f}s] ★ 제2적구 정상 접촉 ({target_color.name}) -> 충돌 즉시 실시간 득점(+1) 발송!"
                    )
                    if not self.score_dispatched:
                        self.score_dispatched = True
                        instant_result = ShotResult(
                            scored=True,
                            shooter_ball=self.active_cue_ball or self.assigned_cue_ball,
                            cushions_at_second_hit=self.cushions_at_second_hit,
                            total_cushions=self.cushions_at_second_hit,
                            is_bank_shot=self.is_bank_shot,
                            object_ball_kiss_occurred=self.object_ball_kiss_occurred,
                            double_hit_foul=self.double_hit_foul,
                            centroid_shift_corrected_count=self.centroid_shift_corrected_count,
                            gpt_occlusion_restored_frames=self.gpt_occlusion_restored_frames,
                            kiss_id_swap_corrected=self.kiss_id_swap_corrected,
                            fail_reason=None,
                            event_log=list(self.event_log),
                            already_dispatched=False,
                        )
                        self._pending_immediate_result = instant_result
                        if self.on_immediate_score is not None:
                            self.on_immediate_score(instant_result)
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
        was_dispatched = bool(self.score_dispatched)
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
            event_log=list(self.event_log),
            already_dispatched=was_dispatched,
        )
        self._reset_shot()
        return result
