"""
폴리라인 꼭짓점 단순화(cv2.approxPolyDP) + 목적구 위치 이탈(Displacement) 감지 기반
초단순·고신뢰 3쿠션 판정 엔진 (CausalKissSafeFSM)
파일 경로: three_cushion_vision/causal_kiss_fsm.py

[4대 핵심 원리]
1. 원리 1 (수구 결정 및 턴 유지/교대):
   - 세 공(WHITE, YELLOW, RED)이 정지(speed < 20mm/s)해 있을 때 정지 위치(rest_pos)를 저장합니다.
   - 적구(RED)는 수구가 될 수 없으며, WHITE와 YELLOW 중 정지 위치에서 가장 먼저 이탈한 공을 수구로 결정합니다.
   - 득점 성공(scored == True) 시 수구(assigned_cue_ball)를 그대로 유지하고,
     득점 실패(scored == False) 시 모든 공이 멈추면 반대 색상(WHITE <-> YELLOW)으로 수구를 교체합니다.
2. 원리 2 (cv2.approxPolyDP 폴리라인 꼭짓점 기반 쿠션 카운트):
   - 수구 이동 궤적(cue_path)에 cv2.approxPolyDP(pts, epsilon=18.0, closed=False)를 적용하여
     바운딩 박스 떨림과 쿠션 슬라이딩 노이즈를 모두 직선으로 펴고 실제 꺾인 꼭짓점(Vertex)만 추출합니다.
   - 각 꼭짓점에서 진행 방향이 10.0도 이상 꺾였고, 당구대 4면 가장자리(125.0mm 이내)에 있으며,
     목적구 충돌 꼭짓점이 아니면 유효 쿠션 1회로 카운트합니다 (0쿠션 누락 및 5/11쿠션 폭주 원천 차단).
3. 원리 3 (목적구 충돌 판정 단순화 - 위치 이탈 OR 확실한 꼭짓점 굴절):
   - 수구와 목적구가 72.0mm 이내로 근접했을 때:
     * 현상 A (목적구 위치 이탈): 근접 3프레임 전 위치 대비 통과 2프레임 후 목적구가 5.5mm 이상 새로운 방향으로 밀려났고
       다른 목적구에 의한 간접 키스가 아니면 충돌로 확정합니다.
     * 현상 B (얇은 스침 수구 꼭짓점 확인): 목적구가 5.5mm 미만으로 움직였더라도 수구 폴리라인 꼭짓점이
       목적구 옆(d_min <= 65.5mm)에서 4.2도 이상 꺾였고(또는 서브픽셀 위상 진동 감지), 다른 목적구 충돌이나 쿠션 반사 중이 아니면 인정합니다.
     * 목적구도 안 밀렸고 수구도 직선으로 통과했다면 거리가 55~58mm로 찍혀도 100% 평범한 실패(노카운트)로 기각합니다.
4. 원리 4 (뱅크샷 및 제2적구 충돌 즉시 득점):
   - 제1적구 최초 충돌 이전 폴리라인 쿠션 꼭짓점 수(cushions_before_first_hit >= 1, 1쿠션 넣어치기 포함)로 뱅크샷을 판정합니다.
   - 제2적구 최초 충돌 확정 순간 누적 쿠션이 3회 이상이면 즉시 score_dispatched = True로 잠그고
     on_immediate_score 콜백을 호출하여 실시간 +1점을 반영한 뒤 잔여 구름 동안 모든 판정을 동결합니다.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
import math
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple

import cv2
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
    cushions_before_first_hit: int = 0
    shot_id: int = 0
    is_turn_end: bool = True
    fail_reason: Optional[str] = None
    event_log: List[str] = field(default_factory=list)
    already_dispatched: bool = False
    shot_event_tokens: List[str] = field(default_factory=list)

    @property
    def cue_ball(self) -> BallColor:
        """이번 샷을 수행한 수구 색상 반환 (shooter_ball 별칭)"""
        return self.shooter_ball

    @property
    def next_cue_ball(self) -> BallColor:
        """다음 샷에 배정될 수구 색상 반환 (득점 시 유지, 실패 시 반대 수구 전환)"""
        if self.scored:
            return self.shooter_ball
        return (
            BallColor.YELLOW
            if self.shooter_ball == BallColor.WHITE
            else BallColor.WHITE
        )


@dataclass
class CuePathPoint:
    """수구 시계열 궤적의 단일 샘플 데이터"""
    frame_idx: int
    t: float
    pos: np.ndarray
    min_dist_to_targets: float
    intra_frame_v_bounce: bool = False


class CausalKissSafeFSM:
    """
    cv2.approxPolyDP 폴리라인 꼭짓점 단순화 + 목적구 위치 이탈(Displacement) 감지 기반의
    초단순·고신뢰 3쿠션 판정 유한 상태 머신(FSM)입니다.
    """

    # 원리 1 & 기술 2: 수구 출발 및 정지 스냅샷 대조 상수
    STROKE_START_SPEED: float = 160.0
    STOP_SPEED: float = 18.0
    MIN_STROKE_DISP_MM: float = 20.0
    MIN_VALID_SHOT_TRAVEL_MM: float = 120.0

    # 원리 2 & 기술 3: 구간별 선형 회귀(approxPolyDP) + 충돌 평면 광선 교차 쿠션 상수
    POLY_EPSILON_CUSHION_MM: float = 18.0
    POLY_MIN_CUSHION_BEND_DEG: float = 10.0
    RAIL_VERTEX_ZONE_MM: float = 125.0
    RAIL_KINEMATIC_ZONE: float = 115.0
    RAIL_ENTER_MARGIN: float = 54.0
    SAME_RAIL_DEEP_EXIT_MM: float = 220.0

    # 원리 3 & 기술 4: 정지 앵커 이탈(증거 A) 및 직선 파괴(Line-Fit RMSE, 증거 B) 상수
    TARGET_APPROACH_ZONE_MM: float = 72.0
    TARGET_MOVED_DISP_TH_MM: float = 5.2
    TARGET_MIN_DELTA_V_MM_S: float = 42.0
    THIN_HIT_MAX_DIST_MM: float = 66.5
    THIN_HIT_MIN_BEND_DEG: float = 4.0
    THIN_HIT_MIN_LINE_RMSE_MM: float = 1.6
    PAIR_COOLDOWN_SEC: float = 0.08

    # 투터치 및 블롭 병합 방어 상수
    DOUBLE_HIT_INIT_DIST_MAX: float = 115.0
    DOUBLE_HIT_WINDOW_SEC: float = 0.16
    DOUBLE_HIT_REACCEL_THRESH: float = 1500.0
    DOUBLE_HIT_FOLLOW_RATIO: float = 0.75
    BLOB_PROXIMITY_ZONE_MM: float = 92.0
    MAX_CENTROID_JUMP_RESIDUAL_MM: float = 14.0
    MAX_OCCLUSION_INTERP_FRAMES: int = 10

    def __init__(
        self,
        fps: float = 60.0,
        assigned_cue_ball: BallColor = BallColor.WHITE,
        gpt_model: Optional[BilliardNanoGPT] = None,
        device: str = "cpu",
        on_immediate_score: Optional[Callable[[ShotResult], None]] = None,
    ) -> None:
        self.fps: float = fps
        self.assigned_cue_ball: BallColor = assigned_cue_ball
        self.device: str = device
        self.gpt_model: BilliardNanoGPT = (gpt_model or BilliardNanoGPT()).to(device).eval()
        self.on_immediate_score: Optional[Callable[[ShotResult], None]] = on_immediate_score

        self.STOP_HOLD_FRAMES: int = max(10, int(fps * 0.25))
        self.shot_seq_id: int = 0
        self.turn_locked_by_previous_shot: bool = False

        self.history: Deque[FrameSnapshot] = deque(maxlen=20)
        self.prev_raw_pos: Dict[BallColor, Tuple[np.ndarray, float]] = {}
        self.prev_ema_vel: Dict[BallColor, np.ndarray] = {}
        self.gpt_buffers: Dict[BallColor, Deque[List[float]]] = {
            c: deque(maxlen=6) for c in BallColor
        }
        self.missing_streak: Dict[BallColor, int] = {c: 0 for c in BallColor}
        self.frame_counter: int = 0
        self.rest_pos: Dict[BallColor, np.ndarray] = {}
        self.idle_anchor_pos: Dict[BallColor, np.ndarray] = self.rest_pos
        self.target_rest_anchor: Dict[BallColor, np.ndarray] = {}
        self.onset_candidate_time: Dict[BallColor, Optional[float]] = {
            BallColor.WHITE: None,
            BallColor.YELLOW: None,
        }
        self.onset_time: Dict[BallColor, Optional[float]] = self.onset_candidate_time
        self.first_mover_from_idle: Optional[BallColor] = None
        self.shot_event_tokens: List[str] = []
        self._reset_shot()

    def _reset_shot(self) -> None:
        """새로운 샷 대기를 위해 내부 샷 상태 변수를 초기화합니다."""
        self.state: ShotState = ShotState.IDLE
        self.active_cue_ball: Optional[BallColor] = None
        self.stroke_start_time: float = 0.0
        self.initial_positions: Dict[BallColor, np.ndarray] = {}
        self.initial_ball_distances: Dict[BallColor, float] = {}
        self.stroke_origin_pos: Optional[np.ndarray] = None
        self.initial_stroke_dir: Optional[np.ndarray] = None
        self._initial_stroke_samples: List[np.ndarray] = []
        self.min_rail_dist_before_first_hit: float = 9999.0

        self.first_target_ball: Optional[BallColor] = None
        self.second_target_ball: Optional[BallColor] = None
        self.first_target_hit_time: Optional[float] = None
        self.first_target_hit_path_idx: Optional[int] = None
        self.first_target_normal: Optional[np.ndarray] = None

        self.cushions_before_first_hit: int = 0
        self.cushions_after_first_hit: int = 0
        self.cushion_count: int = 0
        self.cushions_at_second_hit: int = 0
        self.is_bank_shot: bool = False
        self._geom_one_bank_recovered: bool = False

        self.object_ball_kiss_occurred: bool = False
        self.last_object_kiss_time: float = -999.0
        self.double_hit_foul: bool = False
        self.score_dispatched: bool = False
        self.cue_total_travel_mm: float = 0.0
        self.centroid_shift_corrected_count: int = 0
        self.gpt_occlusion_restored_frames: int = 0
        self.kiss_id_swap_corrected: bool = False
        self.fail_reason: Optional[str] = None
        self.event_log: List[str] = []

        self.stop_counter: int = 0
        self.cue_path: List[CuePathPoint] = []
        self.all_ball_paths: Dict[BallColor, List[Tuple[int, float, np.ndarray]]] = {
            c: [] for c in BallColor
        }
        self._logged_cushion_frames: Set[int] = set()
        self.pair_last_hit_time: Dict[frozenset, float] = {}
        self.id_swap_map: Dict[BallColor, BallColor] = {c: c for c in BallColor}
        self._last_id_swap_check_time: float = -999.0
        self.centroid_clamp_cooldown: Dict[BallColor, int] = {c: 0 for c in BallColor}
        self._pending_immediate_result: Optional[ShotResult] = None
        self.rest_pos.clear()
        self.target_rest_anchor.clear()
        self.onset_candidate_time = {
            BallColor.WHITE: None,
            BallColor.YELLOW: None,
        }
        self.onset_time = self.onset_candidate_time
        self.first_mover_from_idle = None
        self.shot_event_tokens = []
        self._kiss_token_count: int = 0
        self.last_hit_rail: Optional[Rail] = None

    @staticmethod
    def _angle_between_deg(v1: np.ndarray, v2: np.ndarray) -> float:
        """두 2D 벡터 사이의 각도(도, 0 ~ 180)를 계산합니다."""
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cos_t = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_t)))

    @staticmethod
    def _subframe_ccd(
        pA0: np.ndarray, pA1: np.ndarray, pB0: np.ndarray, pB1: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """프레임 간 두 공의 선형 이동 선분 사이 최근접 거리(d_min), 시각 비율(tau), 단위 법선 벡터(n_ab)를 반환합니다."""
        r0 = pB0 - pA0
        v_rel = (pB1 - pB0) - (pA1 - pA0)
        v_rel_sq = float(np.dot(v_rel, v_rel))
        if v_rel_sq < 1e-6:
            tau = 0.0
        else:
            tau = float(np.clip(-np.dot(r0, v_rel) / v_rel_sq, 0.0, 1.0))
        pos_a = pA0 + tau * (pA1 - pA0)
        pos_b = pB0 + tau * (pB1 - pB0)
        diff = pos_b - pos_a
        dist = float(np.linalg.norm(diff))
        n_ab = diff / max(dist, 1e-6)
        return dist, tau, n_ab

    def _predict_next_state_with_gpt(
        self, color: BallColor, dt: float, use_gpt: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """선수 신체에 의해 공이 가려졌을 때(None) BilliardNanoGPT 및 탄성 반사 물리 모델로 좌표를 복원합니다."""
        buf = self.gpt_buffers[color]
        p_prev, _ = self.prev_raw_pos[color]
        v_prev = self.prev_ema_vel.get(color, np.zeros(2, dtype=np.float64)).copy()

        # 물리적 등속 전진 + 레일 경계 탄성 반사 계산
        next_x = float(p_prev[0] + v_prev[0] * dt)
        next_y = float(p_prev[1] + v_prev[1] * dt)
        next_vx, next_vy = float(v_prev[0]), float(v_prev[1])

        if next_x < BALL_R:
            next_x = BALL_R + (BALL_R - next_x)
            next_vx = abs(next_vx) * 0.88
        elif next_x > TABLE_W - BALL_R:
            next_x = (TABLE_W - BALL_R) - (next_x - (TABLE_W - BALL_R))
            next_vx = -abs(next_vx) * 0.88

        if next_y < BALL_R:
            next_y = BALL_R + (BALL_R - next_y)
            next_vy = abs(next_vy) * 0.88
        elif next_y > TABLE_H - BALL_R:
            next_y = (TABLE_H - BALL_R) - (next_y - (TABLE_H - BALL_R))
            next_vy = -abs(next_vy) * 0.88

        phys_pos = np.array([next_x, next_y], dtype=np.float64)
        phys_vel = np.array([next_vx, next_vy], dtype=np.float64)

        if use_gpt and len(buf) >= 2 and float(np.linalg.norm(v_prev)) > 25.0:
            hist_tensor = torch.tensor([list(buf)], dtype=torch.float32, device=self.device)
            pred_state = (
                self.gpt_model.rollout_occlusion(hist_tensor, n_missing_frames=1)[0, 0]
                .cpu()
                .numpy()
            )
            gpt_pos = np.array([float(pred_state[0]), float(pred_state[1])], dtype=np.float64)
            # GPT 추론 좌표가 물리 반사 궤적과 크게 벗어나지 않을 때만 블렌딩
            if float(np.linalg.norm(gpt_pos - phys_pos)) <= 80.0:
                return 0.3 * gpt_pos + 0.7 * phys_pos, phys_vel

        return phys_pos, phys_vel

    def _stabilize_blob_merging(
        self,
        t: float,
        raw_coords_mm: Dict[BallColor, Optional[Tuple[float, float]]],
    ) -> Optional[Tuple[Dict[BallColor, np.ndarray], Dict[BallColor, Optional[np.ndarray]]]]:
        """
        바운딩 박스 병합(Blob Merging)에 의한 정지 공 중심점 끌림(Centroid Shift)과
        가려짐(None) 결측 프레임을 보정합니다.
        """
        mapped_coords: Dict[BallColor, Optional[Tuple[float, float]]] = {
            true_c: raw_coords_mm.get(obs_c, None)
            for true_c, obs_c in self.id_swap_map.items()
        }

        predicted_pos: Dict[BallColor, np.ndarray] = {}
        predicted_vel: Dict[BallColor, np.ndarray] = {}
        stabilized_pos: Dict[BallColor, np.ndarray] = {}
        gpt_restored_vel_override: Dict[BallColor, Optional[np.ndarray]] = {
            c: None for c in BallColor
        }

        for color in BallColor:
            obs = mapped_coords.get(color, None)
            if color in self.prev_raw_pos:
                _, t_prev = self.prev_raw_pos[color]
                dt = max(t - t_prev, 1e-4)
                p_pred, v_pred = self._predict_next_state_with_gpt(
                    color, dt, use_gpt=(obs is None)
                )
                predicted_pos[color] = p_pred
                predicted_vel[color] = v_pred
            elif obs is not None:
                p_init = np.array([float(obs[0]), float(obs[1])], dtype=np.float64)
                predicted_pos[color] = p_init
                predicted_vel[color] = np.zeros(2, dtype=np.float64)

            if obs is None:
                self.missing_streak[color] += 1
                if (
                    color in predicted_pos
                    and self.missing_streak[color] <= self.MAX_OCCLUSION_INTERP_FRAMES
                ):
                    stabilized_pos[color] = predicted_pos[color]
                    gpt_restored_vel_override[color] = predicted_vel[color]
                    self.gpt_occlusion_restored_frames += 1
                else:
                    return None
            else:
                self.missing_streak[color] = 0
                stabilized_pos[color] = np.array(
                    [float(obs[0]), float(obs[1])], dtype=np.float64
                )

        cue_ref = self.active_cue_ball or self.assigned_cue_ball
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
                        # 수구(cue_ref) 자체는 스트로크 출발 주체이므로 클램핑에서 제외
                        if target_c == cue_ref or mapped_coords.get(target_c, None) is None:
                            continue
                        p_obs = stabilized_pos[target_c]
                        p_prd = predicted_pos[target_c]
                        shift_vec = p_obs - p_prd
                        shift_mag = float(np.linalg.norm(shift_vec))

                        to_neighbor = (
                            self.prev_raw_pos[neighbor_c][0]
                            - self.prev_raw_pos[target_c][0]
                        )
                        to_neighbor_norm = to_neighbor / max(
                            float(np.linalg.norm(to_neighbor)), 1e-6
                        )
                        pull_alignment = float(
                            np.dot(shift_vec / max(shift_mag, 1e-6), to_neighbor_norm)
                        )
                        prev_speed = float(
                            np.linalg.norm(
                                self.prev_ema_vel.get(target_c, np.zeros(2))
                            )
                        )
                        neighbor_speed = float(
                            np.linalg.norm(
                                self.prev_ema_vel.get(neighbor_c, np.zeros(2))
                            )
                        )
                        if (
                            self.MAX_CENTROID_JUMP_RESIDUAL_MM < shift_mag <= 30.0
                            and pull_alignment > 0.45
                            and prev_speed < 50.0
                            and neighbor_speed > 200.0
                        ):
                            stabilized_pos[target_c] = 0.85 * p_prd + 0.15 * p_obs
                            self.centroid_shift_corrected_count += 1
                            self.centroid_clamp_cooldown[target_c] = 6

        return stabilized_pos, gpt_restored_vel_override

    def _detect_and_fix_kiss_id_swap(
        self, t_eval: float, b1: BallColor, b2: BallColor
    ) -> bool:
        """
        적구 정면 키스(자리바꿈 키스) 시 칼만 필터 트래커가 직진 관성에 속아
        정지한 공(실제 1적구)과 날아간 공(실제 2적구)의 ID를 서로 뒤바꾼 경우 즉시 역스왑합니다.
        """
        if (t_eval - self._last_id_swap_check_time) < 0.15 or len(self.history) < 5:
            return False

        f_pre2 = self.history[-5]
        f_prev = self.history[-4]
        f_curr = self.history[-3]
        f_post = self.history[-1]

        d_min, _, _ = self._subframe_ccd(
            f_prev.pos[b1], f_curr.pos[b1], f_prev.pos[b2], f_curr.pos[b2]
        )
        if d_min > 35.0:
            return False

        dt_pre = max(f_prev.t - f_pre2.t, 1e-4)
        dt_post = max(f_post.t - f_curr.t, 1e-4)
        v1_pre = (f_prev.pos[b1] - f_pre2.pos[b1]) / dt_pre
        v2_pre = (f_prev.pos[b2] - f_pre2.pos[b2]) / dt_pre
        v1_post = (f_post.pos[b1] - f_curr.pos[b1]) / dt_post
        v2_post = (f_post.pos[b2] - f_curr.pos[b2]) / dt_post

        s1_pre, s2_pre = float(np.linalg.norm(v1_pre)), float(np.linalg.norm(v2_pre))
        s1_post, s2_post = float(np.linalg.norm(v1_post)), float(np.linalg.norm(v2_post))

        mover: Optional[BallColor] = None
        stayer: Optional[BallColor] = None
        v_m_pre = v_m_post = np.zeros(2)
        s_m_pre = s_m_post = s_s_post = 0.0

        if s1_pre > 250.0 and s2_pre < 45.0:
            mover, stayer = b1, b2
            v_m_pre, v_m_post = v1_pre, v1_post
            s_m_pre, s_m_post, s_s_post = s1_pre, s1_post, s2_post
        elif s2_pre > 250.0 and s1_pre < 45.0:
            mover, stayer = b2, b1
            v_m_pre, v_m_post = v2_pre, v2_post
            s_m_pre, s_m_post, s_s_post = s2_pre, s2_post, s1_post

        if mover is None or stayer is None:
            return False

        cos_dir = float(
            np.dot(v_m_pre, v_m_post) / max(s_m_pre * s_m_post, 1e-6)
        )
        if s_m_post > s_m_pre * 0.65 and s_s_post < 45.0 and cos_dir > 0.92:
            self.id_swap_map[b1], self.id_swap_map[b2] = (
                self.id_swap_map[b2],
                self.id_swap_map[b1],
            )
            for snap in self.history:
                if snap.t >= f_curr.t:
                    snap.pos[b1], snap.pos[b2] = (
                        snap.pos[b2].copy(),
                        snap.pos[b1].copy(),
                    )
                    snap.vel[b1], snap.vel[b2] = (
                        snap.vel[b2].copy(),
                        snap.vel[b1].copy(),
                    )
            self.all_ball_paths[b1], self.all_ball_paths[b2] = (
                self.all_ball_paths[b2],
                self.all_ball_paths[b1],
            )
            self.prev_raw_pos[b1], self.prev_raw_pos[b2] = (
                self.prev_raw_pos[b2],
                self.prev_raw_pos[b1],
            )
            self.prev_ema_vel[b1], self.prev_ema_vel[b2] = (
                self.prev_ema_vel[b2].copy(),
                self.prev_ema_vel[b1].copy(),
            )
            self.kiss_id_swap_corrected = True
            self.object_ball_kiss_occurred = True
            self._last_id_swap_check_time = t_eval
            self.event_log.append(
                f"[{t_eval:.2f}s] [트래커 ID 역스왑 복원] {mover.name}<->{stayer.name} 정면 키스 자리바꿈 교정 완료"
            )
            return True
        return False

    def _extract_polyline_cushions(
        self, up_to_path_idx: Optional[int] = None
    ) -> Tuple[int, int, List[Tuple[int, float, Rail, np.ndarray]]]:
        """
        [원리 2: cv2.approxPolyDP 폴리라인 꼭짓점 기반 쿠션 카운트]
        수구 이동 궤적(self.cue_path)에 cv2.approxPolyDP(pts, epsilon=18.0, closed=False)를 적용하여
        자잘한 떨림을 모두 직선으로 펴고, 실제로 궤적이 꺾인 쿠션 꼭짓점만 추출합니다.

        반환값:
            (cushions_before_first_hit, cushions_after_first_hit, valid_cushion_vertices)
        """
        path = (
            self.cue_path
            if up_to_path_idx is None
            else self.cue_path[: up_to_path_idx + 1]
        )
        if len(path) < 3:
            c_before = 1 if self._geom_one_bank_recovered else 0
            return c_before, 0, []

        pts_arr = np.array([pt.pos for pt in path], dtype=np.float32).reshape(-1, 1, 2)
        approx = cv2.approxPolyDP(
            pts_arr, epsilon=self.POLY_EPSILON_CUSHION_MM, closed=False
        ).reshape(-1, 2)

        if len(approx) < 3:
            c_before = 1 if self._geom_one_bank_recovered else 0
            return c_before, 0, []

        # approx의 각 꼭짓점에 대응하는 원본 cue_path 인덱스(orig_indices) 매핑
        orig_indices: List[int] = []
        search_start = 0
        for v_pt in approx:
            best_i = search_start
            best_d = float("inf")
            for idx in range(search_start, len(path)):
                d = float(np.linalg.norm(path[idx].pos - v_pt))
                if d < best_d:
                    best_d = d
                    best_i = idx
                if d < 1e-3:
                    break
            orig_indices.append(best_i)
            search_start = best_i

        valid_vertices: List[Tuple[int, float, Rail, np.ndarray]] = []
        last_rail: Optional[Rail] = None
        last_vertex_path_idx: int = -1

        for k in range(1, len(approx) - 1):
            p_prev = approx[k - 1]
            p_curr = approx[k]
            p_next = approx[k + 1]
            idx_curr = orig_indices[k]
            pt_info = path[idx_curr]

            vec_in = p_curr - p_prev
            vec_out = p_next - p_curr
            len_in = float(np.linalg.norm(vec_in))
            len_out = float(np.linalg.norm(vec_out))

            min_seg_len = 12.0 if (k == 1 or k == len(approx) - 2) else 18.0
            if len_in < min_seg_len or len_out < min_seg_len:
                continue

            bend_deg = self._angle_between_deg(vec_in, vec_out)
            if bend_deg < self.POLY_MIN_CUSHION_BEND_DEG:
                continue

            # 꼭짓점 전후 1프레임 윈도우 내 최소 레일 거리 및 프레임 내 V자 반사 여부 확인
            w_lo = max(0, idx_curr - 1)
            w_hi = min(len(path) - 1, idx_curr + 1)
            has_v_bounce = any(path[i].intra_frame_v_bounce for i in range(w_lo, w_hi + 1))
            max_rail_zone = 135.0 if has_v_bounce else self.RAIL_VERTEX_ZONE_MM

            d_left = min(float(path[i].pos[0]) for i in range(w_lo, w_hi + 1))
            d_right = min(float(TABLE_W - path[i].pos[0]) for i in range(w_lo, w_hi + 1))
            d_top = min(float(path[i].pos[1]) for i in range(w_lo, w_hi + 1))
            d_bottom = min(float(TABLE_H - path[i].pos[1]) for i in range(w_lo, w_hi + 1))

            # 해당 꼭짓점에서 실제 반사가 일어난 레일 후보 탐색
            rail_candidates: List[Tuple[float, Rail]] = []
            if d_left <= max_rail_zone and vec_in[0] < -1.5 and vec_out[0] > 1.5:
                rail_candidates.append((d_left, Rail.LEFT))
            if d_right <= max_rail_zone and vec_in[0] > 1.5 and vec_out[0] < -1.5:
                rail_candidates.append((d_right, Rail.RIGHT))
            if d_top <= max_rail_zone and vec_in[1] < -1.5 and vec_out[1] > 1.5:
                rail_candidates.append((d_top, Rail.TOP))
            if d_bottom <= max_rail_zone and vec_in[1] > 1.5 and vec_out[1] < -1.5:
                rail_candidates.append((d_bottom, Rail.BOTTOM))

            if not rail_candidates:
                continue

            rail_candidates.sort(key=lambda item: item[0])
            d_rail_min, hit_rail = rail_candidates[0]

            # 목적구 충돌 꼭짓점과 쿠션 반사 꼭짓점 분리:
            # 1) 만약 꼭짓점 시점이 제1적구 최초 충돌 시점과 겹치는 경우:
            #    일반 샷(Test 2: 레일 앞 제1적구를 직접 먼저 맞혀 꺾인 경우)은 쿠션에서 배제하되,
            #    1쿠션 넣어치기(Test 6: 초기 스트로크 방향이 제1적구 중심을 비껴나 쿠션을 먼저 찍은 경우)는 선행 쿠션으로 보존!
            is_before_first_hit = (
                self.first_target_hit_path_idx is None
                or idx_curr < self.first_target_hit_path_idx
            )
            if (
                self.first_target_hit_path_idx is not None
                and abs(idx_curr - self.first_target_hit_path_idx) <= 1
            ):
                if self._geom_one_bank_recovered:
                    is_before_first_hit = True
                else:
                    continue

            # 2) 제1적구 충돌 이후 구간에서 꼭짓점 반경 72.0mm 이내에 목적구가 있고
            #    레일 밀착(<= 42.0mm) 반사가 아니라면 목적구 재키스/충돌 꼭짓점이므로 배제
            if not is_before_first_hit:
                if pt_info.min_dist_to_targets < 72.0 and d_rail_min > 42.0:
                    continue

            # 3) 동일 레일 슬라이딩 중복 카운트 차단:
            #    직전 쿠션 꼭짓점과 같은 레일이면, 중간에 제1적구 충돌이 있었거나
            #    레일로부터 220mm 이상 깊숙이 이탈했다가 다시 들어온 경우에만 인정
            if hit_rail == last_rail and last_vertex_path_idx >= 0:
                crossed_first_hit = (
                    self.first_target_hit_path_idx is not None
                    and last_vertex_path_idx < self.first_target_hit_path_idx < idx_curr
                )
                if not crossed_first_hit:
                    max_away = 0.0
                    for m_i in range(last_vertex_path_idx, idx_curr + 1):
                        mp = path[m_i].pos
                        if hit_rail == Rail.LEFT:
                            r_d = float(mp[0])
                        elif hit_rail == Rail.RIGHT:
                            r_d = float(TABLE_W - mp[0])
                        elif hit_rail == Rail.TOP:
                            r_d = float(mp[1])
                        else:
                            r_d = float(TABLE_H - mp[1])
                        if r_d > max_away:
                            max_away = r_d
                    if max_away < self.SAME_RAIL_DEEP_EXIT_MM:
                        continue

            valid_vertices.append((pt_info.frame_idx, pt_info.t, hit_rail, p_curr.copy()))
            last_rail = hit_rail
            last_vertex_path_idx = idx_curr

        c_before = 0
        c_after = 0
        for f_idx, t_v, r_v, _ in valid_vertices:
            if (
                self.first_target_hit_time is None
                or t_v < self.first_target_hit_time - 0.005
            ):
                c_before += 1
            else:
                c_after += 1

        if self._geom_one_bank_recovered and c_before == 0:
            if c_after > 0 and self.first_target_hit_time is not None:
                # 첫 쿠션 꼭짓점이 제1적구 충돌 직후 1프레임에 걸친 경우 선쿠션으로 전환
                first_v_t = valid_vertices[0][1]
                if abs(first_v_t - self.first_target_hit_time) <= 0.04:
                    c_before = 1
                    c_after = max(0, c_after - 1)
                else:
                    c_before = 1
            else:
                c_before = 1

        if valid_vertices:
            self.last_hit_rail = valid_vertices[-1][2]

        return c_before, c_after, valid_vertices

    def _sync_shot_event_tokens(self) -> None:
        """
        [기술 5: arXiv:2407.19686 이벤트 시퀀스 문법 파서]
        이닝 진행 중 확정된 이벤트를 self.shot_event_tokens 리스트에
        "C"(쿠션), "B1"(제1적구), "KISS_12"(적구 간 키스), "B2"(제2적구) 토큰으로 순서대로 동기화합니다.
        """
        tokens: List[str] = []
        for _ in range(self.cushions_before_first_hit):
            tokens.append("C")
        if self.first_target_ball is not None:
            tokens.append("B1")
        for _ in range(self._kiss_token_count):
            tokens.append("KISS_12")
        for _ in range(self.cushions_after_first_hit):
            tokens.append("C")
        if self.second_target_ball is not None:
            tokens.append("B2")
        self.shot_event_tokens = tokens

    def _parse_event_grammar_on_b2(self) -> Tuple[bool, bool, int, int]:
        """
        [기술 5: arXiv:2407.19686 문법 검증]
        "B2" 토큰이 최초로 기록된 시점에 self.shot_event_tokens를 파싱하여:
        - "B2" 앞선 토큰들 중 "B1"이 존재하고 "C"의 개수가 3개 이상이면 scored = True
        - "B1" 토큰보다 앞에 있는 "C" 토큰이 1개 이상이면 is_bank_shot = True
        반환값: (is_valid_score, is_bank_shot, total_c_before_b2, c_before_b1)
        """
        self._sync_shot_event_tokens()
        if "B2" not in self.shot_event_tokens:
            return False, False, self.cushion_count, self.cushions_before_first_hit

        b2_idx = self.shot_event_tokens.index("B2")
        prefix = self.shot_event_tokens[:b2_idx]
        has_b1 = "B1" in prefix
        total_c = prefix.count("C")
        c_before_b1 = prefix[: prefix.index("B1")].count("C") if has_b1 else total_c
        is_valid = has_b1 and (total_c >= 3) and (not self.double_hit_foul)
        is_bank = c_before_b1 >= 1
        return is_valid, is_bank, total_c, c_before_b1

    def _update_polyline_cushion_counts(
        self, up_to_path_idx: Optional[int] = None
    ) -> None:
        """폴리라인 꼭짓점 및 충돌 평면 광선 교차 분석 결과를 FSM 쿠션 카운터 및 이벤트 토큰에 동기화합니다."""
        if self.score_dispatched:
            return
        c_before, c_after, vertices = self._extract_polyline_cushions(up_to_path_idx)
        self.cushions_before_first_hit = c_before
        self.cushions_after_first_hit = c_after
        self.cushion_count = c_before + c_after
        if self.cushions_before_first_hit >= 1:
            self.is_bank_shot = True
        self._sync_shot_event_tokens()

        for f_idx, t_v, rail, pos_v in vertices:
            if f_idx not in self._logged_cushion_frames:
                self._logged_cushion_frames.add(f_idx)
                self.event_log.append(
                    f"[{t_v:.2f}s] 쿠션 꼭짓점 #{len(self._logged_cushion_frames)} ({rail.name}) "
                    f"pos=({pos_v[0]:.1f}, {pos_v[1]:.1f}) "
                    f"[선쿠션:{self.cushions_before_first_hit} + 후쿠션:{self.cushions_after_first_hit} = 총:{self.cushion_count}]"
                )

    @staticmethod
    def _compute_line_fit_rmse(pts: np.ndarray) -> float:
        """
        [기술 4: 빈 공대 TU Wien 논문]
        주어진 2D 궤적 점들(N x 2)에 단일 직선을 최소자승법(SVD/PCA)으로 피팅했을 때의
        수직 잔차 RMSE(mm)를 계산합니다. 직선 등속 주행이면 0에 수렴하고, 미세 충돌 시 >= 1.6mm로 증가합니다.
        """
        if len(pts) < 3:
            return 0.0
        pts_f = np.asarray(pts, dtype=np.float64)
        mean_pt = np.mean(pts_f, axis=0)
        centered = pts_f - mean_pt
        cov = centered.T @ centered
        eigvals, _ = np.linalg.eigh(cov)
        min_eig = max(0.0, float(eigvals[0]))
        return float(np.sqrt(min_eig / len(pts_f)))

    def _check_cue_polyline_vertex_at_target(
        self,
        target_pos: np.ndarray,
        f_pre2: FrameSnapshot,
        f_prev: FrameSnapshot,
        f_curr: FrameSnapshot,
        f_post2: FrameSnapshot,
        cue_color: BallColor,
    ) -> Tuple[bool, float, float]:
        """
        [원리 3 & 기술 4 - 증거 B 보조]:
        수구가 목적구와 최근접한 시점(f_curr) 전후 궤적에 대해
        1) cv2.approxPolyDP 꼭짓점 굴절각(kink_deg)과
        2) 단일 직선 회귀 잔차(line_rmse)를 함께 계산하여 반환합니다.
        반환값: (is_broken_line, best_bend_deg, line_rmse_mm)
        """
        p0 = f_pre2.pos[cue_color]
        p1 = f_prev.pos[cue_color]
        p2 = f_curr.pos[cue_color]
        p3 = f_post2.pos[cue_color]

        bend_at_curr = self._angle_between_deg(p2 - p1, p3 - p2)
        if bend_at_curr < 1.8:
            return False, 0.0, 0.0

        pts_4 = np.array([p0, p1, p2, p3], dtype=np.float64)
        line_rmse = self._compute_line_fit_rmse(pts_4)

        local_pts = pts_4.astype(np.float32).reshape(-1, 1, 2)
        approx = cv2.approxPolyDP(local_pts, epsilon=1.8, closed=False).reshape(-1, 2)

        max_bend = 0.0
        for k in range(1, len(approx) - 1):
            v_in = approx[k] - approx[k - 1]
            v_out = approx[k + 1] - approx[k]
            if float(np.linalg.norm(v_in)) < 5.0 or float(np.linalg.norm(v_out)) < 5.0:
                continue
            bend = self._angle_between_deg(v_in, v_out)
            dist_to_t = float(np.linalg.norm(approx[k] - target_pos))
            if dist_to_t <= 74.0 and bend > max_bend:
                max_bend = bend

        best_bend = max(max_bend, bend_at_curr)
        is_valid_kink = (best_bend >= self.THIN_HIT_MIN_BEND_DEG) and (
            line_rmse >= self.THIN_HIT_MIN_LINE_RMSE_MM or best_bend >= 4.2
        )
        return is_valid_kink, best_bend, line_rmse

    def _verify_pair_collision(
        self,
        ball_x: BallColor,
        ball_y: BallColor,
        f_pre2: FrameSnapshot,
        f_prev: FrameSnapshot,
        f_curr: FrameSnapshot,
        f_post2: FrameSnapshot,
        other_ball: BallColor,
        subpixel_phase_hit: bool = False,
        is_object_ball_kiss_check: bool = False,
    ) -> bool:
        """
        [원리 3: 목적구 충돌 판정의 단순화 (직접 위치 이동 OR 확실한 꼭짓점 굴절)]
        복잡한 6차원 미분 행렬 대신, 두 공이 72.0mm 이내로 근접했을 때:
        - 현상 A: 목적구(ball_y)가 근접 전 위치(p_before) 대비 통과 후 위치(p_after)에서
                  5.5mm 이상 새로운 방향으로 밀려났고, 다른 공(other_ball)에 의한 간접 키스가 아닌 경우 -> True!
        - 현상 B: 목적구가 5.5mm 미만으로 움직였더라도 수구(ball_x)의 approxPolyDP 꼭짓점이
                  목적구 옆(d_min <= 65.5mm)에서 4.2도 이상 꺾였고(또는 서브픽셀 위상 진동 감지),
                  다른 목적구 충돌이나 쿠션 반사 중이 아닌 경우 -> True!
        - 둘 다 아니면 거리가 55~58mm로 찍혔더라도 100% 평범한 실패(False)로 기각합니다!
        """
        t_eval = f_curr.t
        pair_key = frozenset([ball_x, ball_y])
        if (t_eval - self.pair_last_hit_time.get(pair_key, -999.0)) < self.PAIR_COOLDOWN_SEC:
            return False

        d_min, _, n_xy = self._subframe_ccd(
            f_prev.pos[ball_x], f_curr.pos[ball_x], f_prev.pos[ball_y], f_curr.pos[ball_y]
        )
        # 30fps 고속 샷에서 프레임 사이(Mid-Frame) 정지 적구를 타격하고 튕겨나간 경우 최대 135mm까지 스냅샷 이격이 발생할 수 있음
        if d_min > 135.0:
            return False

        # 1) 적구 간 키스(B <-> C) 판정인 경우:
        if is_object_ball_kiss_check:
            disp_b1 = float(np.linalg.norm(f_post2.pos[ball_x] - f_pre2.pos[ball_x]))
            disp_b2 = float(np.linalg.norm(f_post2.pos[ball_y] - f_pre2.pos[ball_y]))
            if d_min <= 66.0 and (disp_b1 >= 5.2 or disp_b2 >= 5.2):
                self.pair_last_hit_time[pair_key] = t_eval
                # [기술 2-3] 적구 간 키스 발생 즉시 두 목적구의 정지 앵커를 키스 후 위치로 갱신
                self.target_rest_anchor[ball_x] = f_post2.pos[ball_x].copy()
                self.target_rest_anchor[ball_y] = f_post2.pos[ball_y].copy()
                return True
            return False

        # 2) 수구(ball_x) <-> 목적구(ball_y) 충돌 판정:
        # 제3의 공(other_ball)과의 거리 확인 (A -> B -> C 간접 연쇄 키스 및 3구 밀집 착시 차단)
        d_x_other, _, _ = self._subframe_ccd(
            f_prev.pos[ball_x], f_curr.pos[ball_x], f_prev.pos[other_ball], f_curr.pos[other_ball]
        )
        d_y_other, _, _ = self._subframe_ccd(
            f_prev.pos[other_ball], f_curr.pos[other_ball], f_prev.pos[ball_y], f_curr.pos[ball_y]
        )
        d_y_other_post = float(np.linalg.norm(f_post2.pos[ball_y] - f_post2.pos[other_ball]))

        # 만약 제2적구(ball_y) 판정인데, 제1적구(other_ball)가 ball_y에 밀착(<= 66mm)해 있으면서
        # 수구(ball_x)가 제1적구(other_ball)에 더 먼저/가깝게 부딪혀 제1적구가 제2적구를 친 간접 연쇄 키스(Case 2)라면 기각!
        if self.first_target_ball is not None and ball_y != self.first_target_ball:
            if (
                min(d_y_other, d_y_other_post) <= 66.0
                and d_x_other <= 64.0
                and d_x_other < d_min
            ):
                return False

        # 근접 3프레임 전 위치(p_before)와 통과 2프레임 후 위치(p_after) 추출
        p_y_before = f_pre2.pos[ball_y]
        p_y_mid = f_prev.pos[ball_y]
        p_y_after = f_post2.pos[ball_y]

        dt_before = max(f_prev.t - f_pre2.t, 1e-4)
        dt_after = max(f_post2.t - f_curr.t, 1e-4)
        dt_total = max(f_post2.t - f_pre2.t, 1e-4)
        v_y_before = (p_y_mid - p_y_before) / dt_before
        v_y_after = (p_y_after - f_curr.pos[ball_y]) / dt_after
        delta_v_y = v_y_after - v_y_before
        delta_v_y_mag = float(np.linalg.norm(delta_v_y))
        speed_y_before = float(np.linalg.norm(v_y_before))

        v_x_before = (f_curr.pos[ball_x] - f_pre2.pos[ball_x]) / max(f_curr.t - f_pre2.t, 1e-4)
        v_x_after = (f_post2.pos[ball_x] - f_curr.pos[ball_x]) / dt_after
        delta_v_x = v_x_after - v_x_before

        v_z_before = (f_prev.pos[other_ball] - f_pre2.pos[other_ball]) / dt_before
        v_z_after = (f_post2.pos[other_ball] - f_curr.pos[other_ball]) / dt_after
        delta_v_z = v_z_after - v_z_before
        delta_v_z_mag = float(np.linalg.norm(delta_v_z))

        # ------------------------------------------------------------------
        # [기술 4 - 증거 A] 정지 앵커 이탈(Pre/Post Stationary Anchor Diffing) + 직접 타격
        # ------------------------------------------------------------------
        has_cue_vertex, cue_bend_deg, line_rmse = self._check_cue_polyline_vertex_at_target(
            f_curr.pos[ball_y], f_pre2, f_prev, f_curr, f_post2, ball_x
        )
        recent_ob_kiss = abs(t_eval - self.last_object_kiss_time) <= 0.08

        phenomenon_a_hit = False
        if speed_y_before < 45.0 and not recent_ob_kiss:
            # 목적구가 정지해 있었고 직전 적구 간 키스가 없었던 경우:
            # 1) 일반 근접(d_min <= 68.5mm): 5.2mm 이상 이동 & ||Delta v_Y|| >= 42.0mm/s
            # 2) 30fps 프레임 사이 고속 반발(68.5mm < d_min <= 135.0mm):
            #    관성의 법칙에 따라 정지해 있던 적구가 스스로 움직일 수 없으므로,
            #    정지 적구가 15.0mm 이상 강하게 튀어나가고(||Delta v_Y|| >= 140mm/s) 제3의 공 Z가 멀리(> 85mm) 있을 때 인정!
            net_disp_y = float(np.linalg.norm(p_y_after - p_y_before))
            step_after_hit = float(np.linalg.norm(p_y_after - f_curr.pos[ball_y]))
            if d_min <= self.TARGET_APPROACH_ZONE_MM:
                if (
                    net_disp_y >= self.TARGET_MOVED_DISP_TH_MM
                    and step_after_hit >= 1.5
                    and delta_v_y_mag >= self.TARGET_MIN_DELTA_V_MM_S
                ):
                    if d_min <= d_x_other + 6.0 or d_y_other > 68.0:
                        phenomenon_a_hit = True
            elif (
                net_disp_y >= 15.0
                and step_after_hit >= 6.0
                and delta_v_y_mag >= 140.0
                and d_y_other > 85.0
                and d_x_other > d_min
            ):
                phenomenon_a_hit = True
        elif d_min <= self.TARGET_APPROACH_ZONE_MM:
            # 목적구가 이미 적구 간 키스로 굴러가고 있거나 직전(0.08s 이내) 키스가 발생한 경우 (Case 1, Case 5, Case 7):
            # 원래 등속 직진 예상 위치(expected_after) 대비 새로운 방향으로 12.0mm 이상 꺾여 밀려났고,
            # 동시에 수구도 반작용으로 꼭짓점 굴절이 동반되었을 때만 진짜 타격으로 인정!
            expected_y_after = p_y_before + v_y_before * dt_total
            deviation_from_coast = float(np.linalg.norm(p_y_after - expected_y_after))
            y_bend_deg = self._angle_between_deg(v_y_before, v_y_after)
            if (
                deviation_from_coast >= 12.0
                and y_bend_deg >= 8.0
                and has_cue_vertex
                and cue_bend_deg >= 5.5
                and d_y_other > 68.0
                and not recent_ob_kiss
            ):
                phenomenon_a_hit = True

        # ------------------------------------------------------------------
        # [기술 4 - 증거 B] 초미세 스침에 의한 수구 직선 파괴(Line-Fit RMSE) 및 운동량 흡수 검증
        # ------------------------------------------------------------------
        phenomenon_b_hit = False
        if subpixel_phase_hit and d_min <= 95.0 and d_y_other > 72.0:
            phenomenon_b_hit = True
        elif not phenomenon_a_hit and not recent_ob_kiss:
            # 수구가 현재 레일 쿠션 반사 중인지 확인 (쿠션 옆 목적구 비접촉 반사 착시 Test 3 / Illusion 3 차단)
            px, py = float(f_curr.pos[ball_x][0]), float(f_curr.pos[ball_x][1])
            d_rail_now = min(px, TABLE_W - px, py, TABLE_H - py)
            is_bouncing_on_rail = d_rail_now <= self.RAIL_ENTER_MARGIN

            other_moved = float(np.linalg.norm(f_post2.pos[other_ball] - f_pre2.pos[other_ball])) >= 5.2

            # [기술 4-2 핵심 방어: 2적구 위치에서의 1적구 재충돌 운동량 흡수 차단]
            # 만약 제1적구(Z=other_ball)가 수구(A) 근방(<= 74.0mm)에 있으면서
            # 반작용 운동량을 흡수(||Delta v_Z|| >= 28.0 mm/s 및 ||Delta v_A + Delta v_Z|| < ||Delta v_A + Delta v_Y||)했다면,
            # 수구(A)를 꺾이게 만든 범인은 제2적구(Y)가 아니라 제1적구(Z)이므로 증거 B를 즉시 무효화합니다!
            first_ball_absorbed_momentum = False
            if self.first_target_ball is not None and ball_y != self.first_target_ball:
                res_az = float(np.linalg.norm(delta_v_x + delta_v_z))
                res_ay = float(np.linalg.norm(delta_v_x + delta_v_y))
                if (
                    d_x_other <= 74.0
                    and delta_v_z_mag >= 28.0
                    and res_az < res_ay
                ):
                    first_ball_absorbed_momentum = True

            if self.first_target_ball is None:
                # 제1적구 최초 타격 시:
                if (
                    d_min <= self.THIN_HIT_MAX_DIST_MM
                    and has_cue_vertex
                    and (not is_bouncing_on_rail or float(np.linalg.norm(p_y_after - p_y_before)) >= 3.0)
                ):
                    phenomenon_b_hit = True
            elif not first_ball_absorbed_momentum:
                # 제2적구 얇은 스침 시:
                net_disp_y = float(np.linalg.norm(p_y_after - p_y_before))
                y_bend_deg = (
                    self._angle_between_deg(v_y_before, v_y_after)
                    if speed_y_before >= 40.0
                    else 999.0
                )
                if (
                    d_min <= self.THIN_HIT_MAX_DIST_MM
                    and has_cue_vertex
                    and cue_bend_deg >= self.THIN_HIT_MIN_BEND_DEG
                    and line_rmse >= self.THIN_HIT_MIN_LINE_RMSE_MM
                    and not is_bouncing_on_rail
                    and d_x_other > 72.0
                    and not other_moved
                    and net_disp_y >= 1.5
                    and y_bend_deg >= 3.5
                ):
                    phenomenon_b_hit = True
                else:
                    # 수구가 고속(>= 400mm/s)으로 목적구 중심 거리 56.0mm 이내까지 정면 진입한 뒤
                    # 통과하지 못하고 목적구 앞에서 급감속 정지(Stop-Shot)한 경우 충돌 인정
                    v_x_pre = (f_curr.pos[ball_x] - f_prev.pos[ball_x]) / max(f_curr.t - f_prev.t, 1e-4)
                    v_x_post = (f_post2.pos[ball_x] - f_curr.pos[ball_x]) / dt_after
                    s_x_pre = float(np.linalg.norm(v_x_pre))
                    s_x_post = float(np.linalg.norm(v_x_post))
                    if (
                        d_min <= 56.0
                        and s_x_pre >= 400.0
                        and s_x_post <= s_x_pre * 0.15
                        and not is_bouncing_on_rail
                        and d_x_other > 72.0
                    ):
                        phenomenon_b_hit = True

        if phenomenon_a_hit or phenomenon_b_hit:
            self.pair_last_hit_time[pair_key] = t_eval
            self.target_rest_anchor[ball_y] = p_y_after.copy()
            return True

        return False

    def _register_cue_target_hit(
        self,
        t: float,
        target_color: BallColor,
        n_ab: np.ndarray,
        f_prev: FrameSnapshot,
        f_curr: FrameSnapshot,
        f_post2: FrameSnapshot,
    ) -> None:
        """
        [원리 4: 뱅크샷(1쿠션 넣어치기 포함) 및 제2적구 충돌 즉시 득점]
        """
        if self.first_target_ball is None:
            self.first_target_ball = target_color
            self.first_target_hit_time = t
            self.first_target_hit_path_idx = max(0, len(self.cue_path) - 3)
            self.first_target_normal = n_ab

            # 1쿠션 넣어치기(원뱅크 걸어치기) 기하학적 보장:
            # 제1적구 접촉 전 수구가 쿠션 대역(<= 130.0mm)에 진입한 적이 있고,
            # 초기 스트로크 방향 직선이 제1적구 중심을 비껴나(impact_param > 45.0mm 또는 굴절 >= 8.0도)
            # 쿠션을 먼저 맞고 들어온 궤적이면 선쿠션 1회를 보장합니다.
            cue_c = self.active_cue_ball or self.assigned_cue_ball
            if (
                cue_c is not None
                and self.stroke_origin_pos is not None
                and self.initial_stroke_dir is not None
                and self.min_rail_dist_before_first_hit <= 130.0
            ):
                t1_init = self.initial_positions.get(target_color, f_curr.pos[target_color])
                to_t1 = t1_init - self.stroke_origin_pos
                proj_s = float(np.dot(to_t1, self.initial_stroke_dir))
                perp_vec = to_t1 - proj_s * self.initial_stroke_dir
                impact_param = float(np.linalg.norm(perp_vec)) if proj_s > 0.0 else 999.0

                vec_to_hit = f_curr.pos[cue_c] - self.stroke_origin_pos
                deflect_init = self._angle_between_deg(self.initial_stroke_dir, vec_to_hit)
                if deflect_init >= 8.0 or impact_param > 45.0:
                    self._geom_one_bank_recovered = True

            self._update_polyline_cushion_counts(self.first_target_hit_path_idx)
            if self.cushions_before_first_hit >= 1:
                self.is_bank_shot = True

            self.event_log.append(
                f"[{t:.2f}s] 제1적구 접촉 ({target_color.name}, 선쿠션: {self.cushions_before_first_hit}, 총쿠션: {self.cushion_count})"
            )
            self._check_double_hit_foul(t, f_prev, f_curr, f_post2)

        elif target_color == self.first_target_ball:
            self.event_log.append(
                f"[{t:.2f}s] [수구-제1적구 재키스] {target_color.name} 재접촉 (득점 무관)"
            )

        elif self.second_target_ball is None:
            # 제2적구 최초 충돌 확정 순간 폴리라인 쿠션 꼭짓점 개수를 최종 집계
            self._update_polyline_cushion_counts()
            self.second_target_ball = target_color
            self.cushions_at_second_hit = self.cushion_count
            self._sync_shot_event_tokens()
            grammar_valid, grammar_bank, _, _ = self._parse_event_grammar_on_b2()

            if self.state == ShotState.IN_PLAY:
                if (
                    grammar_valid
                    and self.cushion_count >= 3
                    and not self.double_hit_foul
                ):
                    self.state = ShotState.VALID_SCORE_PENDING
                    self.is_bank_shot = bool(
                        grammar_bank or self.cushions_before_first_hit >= 1
                    )
                    self.event_log.append(
                        f"[{t:.2f}s] ★ 제2적구 정상 접촉 ({target_color.name}) -> 충돌 즉시 실시간 득점(+1) 발송! (Tokens: {self.shot_event_tokens})"
                    )
                    if not self.score_dispatched:
                        self.score_dispatched = True
                        instant_result = ShotResult(
                            scored=True,
                            shooter_ball=self.active_cue_ball or self.assigned_cue_ball,
                            cushions_at_second_hit=self.cushions_at_second_hit,
                            total_cushions=self.cushions_at_second_hit,
                            cushions_before_first_hit=self.cushions_before_first_hit,
                            is_bank_shot=self.is_bank_shot,
                            object_ball_kiss_occurred=self.object_ball_kiss_occurred,
                            double_hit_foul=self.double_hit_foul,
                            centroid_shift_corrected_count=self.centroid_shift_corrected_count,
                            gpt_occlusion_restored_frames=self.gpt_occlusion_restored_frames,
                            kiss_id_swap_corrected=self.kiss_id_swap_corrected,
                            fail_reason=None,
                            event_log=list(self.event_log),
                            shot_id=self.shot_seq_id,
                            is_turn_end=False,
                            already_dispatched=False,
                            shot_event_tokens=list(self.shot_event_tokens),
                        )
                        self._pending_immediate_result = instant_result
                        if self.on_immediate_score is not None:
                            self.on_immediate_score(instant_result)
                else:
                    self.state = ShotState.NO_SCORE_LOCKED
                    self.fail_reason = (
                        f"EARLY_SECOND_BALL_KISS (쿠션 {self.cushion_count}회 < 3회)"
                    )
                    self.event_log.append(
                        f"[{t:.2f}s] ✖ 제2적구 조기 키스 ({target_color.name}) -> 노카운트 잠금"
                    )

    def _check_double_hit_foul(
        self,
        t_eval: float,
        f_prev: FrameSnapshot,
        f_curr: FrameSnapshot,
        f_post2: FrameSnapshot,
    ) -> None:
        """수구와 제1적구가 근접(<= 115mm)해 있을 때 발생하는 투터치(밀어치기) 파울을 감지합니다."""
        if (
            self.double_hit_foul
            or self.active_cue_ball is None
            or self.first_target_ball is None
            or self.first_target_hit_time is None
            or self.first_target_normal is None
        ):
            return
        if (self.first_target_hit_time - self.stroke_start_time) > self.DOUBLE_HIT_WINDOW_SEC:
            return
        if not (0.0 <= (t_eval - self.first_target_hit_time) <= 0.09):
            return

        cue = self.active_cue_ball
        target = self.first_target_ball
        if cue in self.initial_positions and target in self.initial_positions:
            init_dist = float(
                np.linalg.norm(self.initial_positions[target] - self.initial_positions[cue])
            )
            if init_dist > self.DOUBLE_HIT_INIT_DIST_MAX:
                return

        dt_pre = max(f_curr.t - f_prev.t, 1e-4)
        dt_post = max(f_post2.t - f_curr.t, 1e-4)
        v_cue_pre = (f_curr.pos[cue] - f_prev.pos[cue]) / dt_pre
        v_cue_post = (f_post2.pos[cue] - f_curr.pos[cue]) / dt_post
        v_target_post = (f_post2.pos[target] - f_curr.pos[target]) / dt_post
        acc_cue = f_post2.acc[cue]
        n_ab = self.first_target_normal

        speed_cue_post = float(np.linalg.norm(v_cue_post))
        if speed_cue_post < 80.0:
            return

        forward_reaccel = float(np.dot(acc_cue, v_cue_post / speed_cue_post))
        normal_reaccel = float(np.dot(acc_cue, n_ab))
        is_reaccelerating = (
            forward_reaccel > self.DOUBLE_HIT_REACCEL_THRESH
            and normal_reaccel > self.DOUBLE_HIT_REACCEL_THRESH * 0.7
        )

        pre_speed = float(np.linalg.norm(v_cue_pre))
        head_on_cos = (
            float(np.dot(v_cue_pre / max(pre_speed, 1e-6), n_ab))
            if pre_speed > 50.0
            else 0.0
        )
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
            self.event_log.append(
                f"[{t_eval:.2f}s] ✖ 파울 감지: 투터치(Double Hit) 확정 -> NO_SCORE_LOCKED 전환"
            )

    def _evaluate_frame_with_context(
        self, subpixel_impacts: Optional[Dict[BallColor, bool]] = None
    ) -> None:
        """
        매 프레임 5프레임 컨텍스트(충돌 3프레임 전 ~ 충돌 2프레임 후)를 이용해
        1) 적구 간 키스 및 ID 스왑 복원
        2) 수구-목적구 충돌 판정 (원리 3)
        3) 폴리라인 꼭짓점 기반 쿠션 카운트 (원리 2)
        4) 투터치 파울 검사
        를 순서대로 수행합니다.
        """
        if self.score_dispatched or self.active_cue_ball is None or len(self.history) < 3:
            return

        f_pre2 = self.history[-5] if len(self.history) >= 5 else self.history[0]
        f_prev = self.history[-4] if len(self.history) >= 4 else self.history[-2]
        f_curr = self.history[-3] if len(self.history) >= 3 else self.history[-1]
        f_post2 = self.history[-1]
        t_eval = f_curr.t
        cue_color = self.active_cue_ball

        other_balls = [c for c in BallColor if c != cue_color]
        b1, b2 = other_balls[0], other_balls[1]

        # [Step 0] 적구 정면 키스 시 트래커 ID 스왑(자리바꿈 키스) 감지 및 역스왑
        self._detect_and_fix_kiss_id_swap(t_eval, b1, b2)

        # [Step 1] 적구 간 키스 감지 (B <-> C) 및 제2적구 정지 앵커(target_rest_anchor) 즉시 갱신
        if self._verify_pair_collision(
            ball_x=b1,
            ball_y=b2,
            f_pre2=f_pre2,
            f_prev=f_prev,
            f_curr=f_curr,
            f_post2=f_post2,
            other_ball=cue_color,
            is_object_ball_kiss_check=True,
        ):
            self.object_ball_kiss_occurred = True
            self.last_object_kiss_time = t_eval
            self._kiss_token_count += 1
            self.target_rest_anchor[b1] = f_post2.pos[b1].copy()
            self.target_rest_anchor[b2] = f_post2.pos[b2].copy()
            self._sync_shot_event_tokens()
            self.event_log.append(
                f"[{t_eval:.2f}s] [적구 간 키스 감지] {b1.name} <-> {b2.name} (앵커 갱신 완료)"
            )

        # [Step 2] 폴리라인 꼭짓점 쿠션 카운트 최신화
        self._update_polyline_cushion_counts()

        # [Step 3] 수구-목적구 충돌 감지 (원리 3: 목적구 위치 이탈 OR 확실한 꼭짓점 굴절)
        sorted_targets = sorted(
            other_balls,
            key=lambda c: 0 if c == self.first_target_ball else 1,
        )
        eval_subpixel = f_curr.subpixel_impacts or subpixel_impacts or {}

        for target_color in sorted_targets:
            third_ball = b2 if target_color == b1 else b1
            sp_hit = bool(eval_subpixel.get(target_color, False))
            if self._verify_pair_collision(
                ball_x=cue_color,
                ball_y=target_color,
                f_pre2=f_pre2,
                f_prev=f_prev,
                f_curr=f_curr,
                f_post2=f_post2,
                other_ball=third_ball,
                subpixel_phase_hit=sp_hit,
                is_object_ball_kiss_check=False,
            ):
                _, _, n_ab = self._subframe_ccd(
                    f_prev.pos[cue_color],
                    f_curr.pos[cue_color],
                    f_prev.pos[target_color],
                    f_curr.pos[target_color],
                )
                self._register_cue_target_hit(
                    t_eval, target_color, n_ab, f_prev, f_curr, f_post2
                )

        # [Step 4] 투터치 파울 검사
        if not self.score_dispatched:
            self._check_double_hit_foul(t_eval, f_prev, f_curr, f_post2)

    def update(
        self,
        t: float,
        coords_mm: Dict[BallColor, Optional[Tuple[float, float]]],
        subpixel_impacts: Optional[Dict[BallColor, bool]] = None,
        intra_frame_rail_bounce: bool = False,
    ) -> Optional[ShotResult]:
        """
        매 프레임 3구 좌표를 입력받아 4대 원리에 따라 3쿠션 득점 여부를 판정합니다.
        제2적구 충돌로 3쿠션 득점이 확정되는 순간 즉시 ShotResult를 반환합니다.
        """
        stabilized = self._stabilize_blob_merging(t, coords_mm)
        if stabilized is None:
            return None
        pos_map, gpt_vel_override = stabilized

        vel_map: Dict[BallColor, np.ndarray] = {}
        acc_map: Dict[BallColor, np.ndarray] = {}

        for c in BallColor:
            p = pos_map[c]
            if c in self.prev_raw_pos:
                p_prev, t_prev = self.prev_raw_pos[c]
                dt = max(t - t_prev, 1e-4)
                raw_v = (
                    gpt_vel_override[c]
                    if gpt_vel_override[c] is not None
                    else (p - p_prev) / dt
                )
                if self.centroid_clamp_cooldown[c] > 0:
                    self.centroid_clamp_cooldown[c] -= 1
                    ema_v = self.prev_ema_vel[c].copy()
                else:
                    ema_v = 0.65 * raw_v + 0.35 * self.prev_ema_vel[c]
                acc_map[c] = (ema_v - self.prev_ema_vel[c]) / dt
                vel_map[c] = ema_v
                self.prev_ema_vel[c] = ema_v
            else:
                vel_map[c] = np.zeros(2, dtype=np.float64)
                acc_map[c] = np.zeros(2, dtype=np.float64)
                self.prev_ema_vel[c] = np.zeros(2, dtype=np.float64)

            self.prev_raw_pos[c] = (p, t)
            self.gpt_buffers[c].append(
                [float(p[0]), float(p[1]), float(vel_map[c][0]), float(vel_map[c][1])]
            )

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

        # ------------------------------------------------------------------
        # [원리 1] IDLE 상태에서 정지 위치(rest_pos) 기록 및 가장 먼저 출발한 수구(WHITE/YELLOW) 결정
        # ------------------------------------------------------------------
        if self.state == ShotState.IDLE:
            for c in BallColor:
                spd_c = float(np.linalg.norm(vel_map[c]))
                if c not in self.rest_pos:
                    prev_s = self.history[-2] if len(self.history) >= 2 else self.history[-1]
                    self.rest_pos[c] = prev_s.pos[c].copy()
                elif spd_c < 20.0:
                    self.rest_pos[c] = 0.85 * self.rest_pos[c] + 0.15 * pos_map[c]

            all_colors = list(BallColor)
            min_pair_dist = min(
                float(np.linalg.norm(pos_map[all_colors[i]] - pos_map[all_colors[j]]))
                for i in range(3)
                for j in range(i + 1, 3)
            )
            if min_pair_dist >= 66.0:
                for cand_c in (BallColor.WHITE, BallColor.YELLOW):
                    cand_spd = float(np.linalg.norm(vel_map[cand_c]))
                    cand_anc = self.rest_pos.get(cand_c, self.history[0].pos[cand_c])
                    cand_disp = float(np.linalg.norm(pos_map[cand_c] - cand_anc))
                    if cand_disp >= 12.0 and cand_spd >= 65.0:
                        if self.onset_candidate_time.get(cand_c) is None:
                            self.onset_candidate_time[cand_c] = t
                        if self.first_mover_from_idle is None:
                            self.first_mover_from_idle = cand_c
                    elif cand_disp < 8.0 and cand_spd < 35.0:
                        self.onset_candidate_time[cand_c] = None
                        if self.first_mover_from_idle == cand_c:
                            other_cand = (
                                BallColor.YELLOW
                                if cand_c == BallColor.WHITE
                                else BallColor.WHITE
                            )
                            self.first_mover_from_idle = (
                                other_cand
                                if self.onset_candidate_time.get(other_cand) is not None
                                else None
                            )

            min_disp_req = self.MIN_STROKE_DISP_MM if len(self.history) >= 4 else 6.0
            cand_ready: List[BallColor] = []
            for cand_c in (BallColor.WHITE, BallColor.YELLOW):
                c_spd = float(np.linalg.norm(vel_map[cand_c]))
                c_anc = self.rest_pos.get(cand_c, self.history[0].pos[cand_c])
                c_disp = float(np.linalg.norm(pos_map[cand_c] - c_anc))
                if c_spd >= self.STROKE_START_SPEED and c_disp >= min_disp_req:
                    cand_ready.append(cand_c)

            cue = self.assigned_cue_ball
            stroke_started = False
            occluded_first_hit_target: Optional[BallColor] = None

            if cand_ready:
                stroke_started = True
                if not self.turn_locked_by_previous_shot:
                    if self.first_mover_from_idle in cand_ready:
                        cue = self.first_mover_from_idle
                    elif (
                        self.first_mover_from_idle is not None
                        and self.onset_candidate_time.get(self.first_mover_from_idle)
                        is not None
                    ):
                        cue = self.first_mover_from_idle
                    elif len(cand_ready) == 1:
                        cue = cand_ready[0]
                    else:
                        d_w = float(
                            np.linalg.norm(
                                pos_map[BallColor.WHITE] - self.rest_pos[BallColor.WHITE]
                            )
                        )
                        d_y = float(
                            np.linalg.norm(
                                pos_map[BallColor.YELLOW] - self.rest_pos[BallColor.YELLOW]
                            )
                        )
                        cue = BallColor.WHITE if d_w >= d_y else BallColor.YELLOW
                    self.assigned_cue_ball = cue
                else:
                    if self.assigned_cue_ball in cand_ready:
                        cue = self.assigned_cue_ball
                    elif (
                        self.first_mover_from_idle == self.assigned_cue_ball
                        and self.onset_candidate_time.get(self.assigned_cue_ball)
                        is not None
                    ):
                        cue = self.assigned_cue_ball
                    else:
                        other_cue = cand_ready[0]
                        assigned_anc = self.rest_pos.get(
                            self.assigned_cue_ball, pos_map[self.assigned_cue_ball]
                        )
                        assigned_disp = float(
                            np.linalg.norm(pos_map[self.assigned_cue_ball] - assigned_anc)
                        )
                        if (
                            min_pair_dist >= 66.0
                            and assigned_disp < 8.0
                            and self.first_mover_from_idle == other_cue
                        ):
                            cue = other_cue
                            self.assigned_cue_ball = other_cue
                        else:
                            cue = self.assigned_cue_ball
                            occluded_first_hit_target = other_cue
            else:
                red_spd = float(np.linalg.norm(vel_map[BallColor.RED]))
                red_anchor = self.rest_pos.get(BallColor.RED, self.history[0].pos[BallColor.RED])
                red_disp = float(np.linalg.norm(pos_map[BallColor.RED] - red_anchor))
                if red_spd >= 350.0 and red_disp >= 35.0:
                    stroke_started = True
                    cue = self.first_mover_from_idle or self.assigned_cue_ball
                    self.assigned_cue_ball = cue
                    occluded_first_hit_target = BallColor.RED

            if stroke_started:
                cue_anchor = self.rest_pos.get(cue, self.history[0].pos[cue])
                cue_disp = float(np.linalg.norm(pos_map[cue] - cue_anchor))
                saved_anchors = {
                    c: self.rest_pos[c].copy() for c in BallColor if c in self.rest_pos
                }
                self._reset_shot()
                self.shot_seq_id += 1
                self.stroke_start_time = t
                self.active_cue_ball = cue
                self.assigned_cue_ball = cue
                self.state = ShotState.IN_PLAY
                prev_snap = self.history[-2] if len(self.history) >= 2 else self.history[-1]
                self.initial_positions = (
                    saved_anchors
                    if len(saved_anchors) == 3
                    else {c: prev_snap.pos[c].copy() for c in BallColor}
                )
                for c_k, p_init in self.initial_positions.items():
                    self.rest_pos[c_k] = p_init.copy()
                    self.target_rest_anchor[c_k] = p_init.copy()

                cue_init_p = self.initial_positions[cue]
                self.stroke_origin_pos = cue_init_p.copy()
                init_vec = pos_map[cue] - self.stroke_origin_pos
                init_norm = float(np.linalg.norm(init_vec))
                if init_norm >= 5.0:
                    self.initial_stroke_dir = init_vec / init_norm
                self._initial_stroke_samples = [pos_map[cue].copy()]

                # 출발 직전 히스토리 프레임을 cue_path에 적재하여 초구 1쿠션 뱅크샷 꼭짓점이 누락되지 않도록 보존
                other_cs = [c for c in BallColor if c != cue]
                for s_h in self.history:
                    p_h = s_h.pos[cue]
                    d_r_h = min(
                        float(p_h[0]),
                        float(TABLE_W - p_h[0]),
                        float(p_h[1]),
                        float(TABLE_H - p_h[1]),
                    )
                    if d_r_h < self.min_rail_dist_before_first_hit:
                        self.min_rail_dist_before_first_hit = d_r_h
                    min_d_t = min(
                        float(np.linalg.norm(p_h - s_h.pos[oc])) for oc in other_cs
                    )
                    self.cue_path.append(
                        CuePathPoint(
                            frame_idx=s_h.frame_idx,
                            t=s_h.t,
                            pos=p_h.copy(),
                            min_dist_to_targets=min_d_t,
                            intra_frame_v_bounce=s_h.intra_frame_rail_bounce,
                        )
                    )
                    for bc in BallColor:
                        self.all_ball_paths[bc].append(
                            (s_h.frame_idx, s_h.t, s_h.pos[bc].copy())
                        )

                self.initial_ball_distances = {
                    c: float(np.linalg.norm(self.initial_positions[c] - cue_init_p))
                    for c in BallColor
                    if c != cue
                }
                self.cue_total_travel_mm = max(
                    cue_disp,
                    150.0 if occluded_first_hit_target is not None else cue_disp,
                )

                if occluded_first_hit_target is not None:
                    self.first_target_ball = occluded_first_hit_target
                    self.first_target_hit_time = t
                    self.first_target_hit_path_idx = max(0, len(self.cue_path) - 1)
                    self.event_log.append(
                        f"[{t:.2f}s] 스트로크 출발 및 가려짐 중 제1적구({occluded_first_hit_target.name}) 피격 감지 (수구 고정: {cue.name})"
                    )
                else:
                    self.event_log.append(f"[{t:.2f}s] 스트로크 출발 (수구: {cue.name})")

                if len(self.history) >= 3:
                    self._evaluate_frame_with_context(subpixel_impacts)
                if self._pending_immediate_result is not None:
                    instant_res = self._pending_immediate_result
                    self._pending_immediate_result = None
                    return instant_res
            return None

        # ------------------------------------------------------------------
        # IN_PLAY 상태: 수구 궤적(cue_path) 적재 및 폴리라인/목적구 충돌 평가
        # ------------------------------------------------------------------
        if self.active_cue_ball is not None:
            cue_c = self.active_cue_ball
            p_cue_now = pos_map[cue_c]
            other_cs = [c for c in BallColor if c != cue_c]
            min_d_targets = min(
                float(np.linalg.norm(p_cue_now - pos_map[oc])) for oc in other_cs
            )
            if len(self.cue_path) >= 1:
                step_travel = float(np.linalg.norm(p_cue_now - self.cue_path[-1].pos))
                self.cue_total_travel_mm += step_travel

            self.cue_path.append(
                CuePathPoint(
                    frame_idx=self.frame_counter,
                    t=t,
                    pos=p_cue_now.copy(),
                    min_dist_to_targets=min_d_targets,
                    intra_frame_v_bounce=bool(intra_frame_rail_bounce),
                )
            )
            for bc in BallColor:
                self.all_ball_paths[bc].append(
                    (self.frame_counter, t, pos_map[bc].copy())
                )

            if self.first_target_ball is None:
                d_rail_now = min(
                    float(p_cue_now[0]),
                    float(TABLE_W - p_cue_now[0]),
                    float(p_cue_now[1]),
                    float(TABLE_H - p_cue_now[1]),
                )
                if d_rail_now < self.min_rail_dist_before_first_hit:
                    self.min_rail_dist_before_first_hit = d_rail_now
                if (
                    len(self._initial_stroke_samples) < 3
                    and self.cushion_count == 0
                    and d_rail_now > self.RAIL_ENTER_MARGIN
                    and self.stroke_origin_pos is not None
                ):
                    vec_from_origin = p_cue_now - self.stroke_origin_pos
                    norm_from_origin = float(np.linalg.norm(vec_from_origin))
                    if norm_from_origin >= 10.0:
                        cand_dir = vec_from_origin / norm_from_origin
                        if (
                            self.initial_stroke_dir is None
                            or self._angle_between_deg(self.initial_stroke_dir, cand_dir)
                            <= 6.0
                        ):
                            self.initial_stroke_dir = cand_dir
                            self._initial_stroke_samples.append(p_cue_now.copy())

        if not self.score_dispatched and len(self.history) >= 3:
            self._evaluate_frame_with_context(subpixel_impacts)

        all_stopped = all(
            float(np.linalg.norm(v)) < self.STOP_SPEED for v in vel_map.values()
        )
        self.stop_counter = (self.stop_counter + 1) if all_stopped else 0

        if self.stop_counter >= self.STOP_HOLD_FRAMES:
            # 120mm 미만 노이즈 가짜 출발 취소 (턴 유지)
            if (
                not self.score_dispatched
                and self.state != ShotState.VALID_SCORE_PENDING
                and self.first_target_ball is None
                and not self.double_hit_foul
                and self.cue_total_travel_mm < self.MIN_VALID_SHOT_TRAVEL_MM
            ):
                self._reset_shot()
                return None
            return self._finalize_shot(t)

        if self._pending_immediate_result is not None:
            instant_res = self._pending_immediate_result
            self._pending_immediate_result = None
            return instant_res

        return None

    def _finalize_shot(self, t: float) -> ShotResult:
        """
        모든 공이 정지했을 때 최종 샷 결과를 반환하고,
        득점 시 수구 유지 / 실패 시 반대 수구(WHITE <-> YELLOW)로 전환합니다.
        """
        if not self.score_dispatched:
            self._update_polyline_cushion_counts()
        self._sync_shot_event_tokens()

        scored = (
            self.state == ShotState.VALID_SCORE_PENDING or self.score_dispatched
        ) and (not self.double_hit_foul)
        was_dispatched = bool(self.score_dispatched)
        shooter = self.active_cue_ball or self.assigned_cue_ball

        if not scored and self.fail_reason is None:
            self.fail_reason = (
                "NO_FIRST_BALL_HIT"
                if self.first_target_ball is None
                else "SECOND_BALL_MISSED"
            )

        final_cushions = (
            self.cushions_at_second_hit if scored else self.cushion_count
        )
        result = ShotResult(
            scored=scored,
            shooter_ball=shooter,
            cushions_at_second_hit=self.cushions_at_second_hit,
            total_cushions=final_cushions,
            cushions_before_first_hit=self.cushions_before_first_hit,
            is_bank_shot=self.is_bank_shot if scored else False,
            object_ball_kiss_occurred=self.object_ball_kiss_occurred,
            double_hit_foul=self.double_hit_foul,
            centroid_shift_corrected_count=self.centroid_shift_corrected_count,
            gpt_occlusion_restored_frames=self.gpt_occlusion_restored_frames,
            kiss_id_swap_corrected=self.kiss_id_swap_corrected,
            fail_reason=None if scored else self.fail_reason,
            event_log=list(self.event_log),
            shot_id=self.shot_seq_id,
            is_turn_end=(not scored),
            already_dispatched=was_dispatched,
            shot_event_tokens=list(self.shot_event_tokens),
        )

        # [원리 1-3 & 1-4]
        # - 득점 성공 시: 수구(assigned_cue_ball) 그대로 유지!
        # - 득점 실패 시: 반대 색깔(WHITE <-> YELLOW)로 수구 변경!
        if scored:
            self.assigned_cue_ball = shooter
        else:
            self.assigned_cue_ball = (
                BallColor.YELLOW
                if shooter == BallColor.WHITE
                else BallColor.WHITE
            )
        self.turn_locked_by_previous_shot = True

        self._reset_shot()
        self.history.clear()
        return result
