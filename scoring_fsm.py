"""
scoring_fsm.py:
수구(Cue Ball)의 처음 출발(Start)부터 마지막 정지(Stop)까지 전생애(Full-Lifecycle) 궤적 방향,
변곡점, 벽면 거리 극소점 돌출도(Valley Prominence), 3대 목적구 타격 형태를 추적하는
CueBallLifecycleTracker 및 CausalKissSafeFSM 통합 모듈
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Set, Tuple
import numpy as np

from three_cushion_vision.causal_kiss_fsm import (
    BALL_DIAM,
    BALL_R,
    TABLE_H,
    TABLE_W,
    BallColor,
    CausalKissSafeFSM,
    FrameSnapshot,
    Rail,
    ShotResult,
    ShotState,
)
from three_cushion_vision.scoring_fsm import ThreeCushionScoringFSM
from ball_identity_tracker import LabHungarianBallIdentifier
from cue_flow_physics_tracker import (
    CueFlowEvent,
    CueFlowPhysicsTwin,
    CueTrajectorySegment,
    FlowEventType,
)

BallID = BallColor


class LifecycleSample(NamedTuple):
    """수구 전생애 궤적(cue_lifecycle_path)의 단일 프레임 기록 (튜플 언패킹 및 속성 접근 동시 지원)"""
    t: float
    pos_cue: np.ndarray
    vel_cue: np.ndarray
    pos_b1: np.ndarray
    pos_b2: np.ndarray


@dataclass
class RailValleyState:
    """각 레일(LEFT, RIGHT, TOP, BOTTOM)의 거리 극소점(Local Minimum Valley) 추적 상태"""
    in_valley: bool = False
    entry_dist: float = 9999.0
    min_dist: float = 9999.0
    min_pos: Optional[np.ndarray] = None
    min_time: float = 0.0
    min_idx: int = -1
    locked: bool = False
    max_dist_since_lock: float = 0.0


class CueBallLifecycleTracker:
    """
    수구가 처음 움직일 때(Start)부터 마지막 멈출 때(Stop)까지의 전생애(Lifecycle) 움직임 방향,
    1적구 충돌, 쿠션 충돌(벽면 거리 극소점 돌출도 Valley Prominence), 2적구 충돌을
    하나의 연속된 수구 궤적 히스토리 위에서 추적하는 초정밀 판정 트래커입니다.
    """

    RAIL_VALLEY_ZONE_MM: float = 115.0
    RAIL_CONTACT_MARGIN_MM: float = 54.0
    SAME_RAIL_UNLOCK_DIST_MM: float = 220.0
    MIN_VALLEY_PROMINENCE_MM: float = 10.5
    MIN_NORMAL_VEL_FLIP_MM_S: float = 25.0
    MIN_AUDIT_BEND_DEG: float = 8.0

    TARGET_CLOSEST_MAX_MM: float = 74.0
    FORM1_MIN_NORMAL_DISP_MM: float = 5.2
    FORM1_MIN_DELTA_V_MM_S: float = 42.0
    FORM2_MAX_DIST_MM: float = 66.5
    FORM2_MIN_BEND_DEG: float = 4.0
    FORM2_MIN_LINE_RMSE_MM: float = 1.6
    FORM2_MIN_RECOIL_MM_S: float = 30.0
    FORM3_MAX_DIST_MM: float = 64.0
    FORM3_MIN_COS_THETA: float = 0.90
    FORM3_MIN_SPEED_DROP_MM_S: float = 120.0

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """새로운 이닝(샷)을 위해 전생애 궤적 및 레일 극소점 상태를 초기화합니다."""
        self.idle_anchor_pos: Dict[BallColor, np.ndarray] = {}
        self.onset_time: Dict[BallColor, Optional[float]] = {
            BallColor.WHITE: None,
            BallColor.YELLOW: None,
        }
        self.active_cue_ball: Optional[BallColor] = None
        self.cue_lifecycle_path: List[LifecycleSample] = []
        self.rail_valleys: Dict[Rail, RailValleyState] = {
            r: RailValleyState() for r in Rail
        }
        self.confirmed_cushions: List[Tuple[int, float, Rail, np.ndarray]] = []
        self.timeline_events: List[Tuple[float, str, str]] = []

    @staticmethod
    def rail_distance_and_normal(pos: np.ndarray, rail: Rail) -> Tuple[float, np.ndarray]:
        """수구 위치에서 지정된 레일까지의 거리 d_rail(mm)과 테이블 안쪽 방향 단위 법선 벡터를 반환합니다."""
        x, y = float(pos[0]), float(pos[1])
        if rail == Rail.LEFT:
            return x, np.array([1.0, 0.0], dtype=np.float64)
        if rail == Rail.RIGHT:
            return float(TABLE_W - x), np.array([-1.0, 0.0], dtype=np.float64)
        if rail == Rail.TOP:
            return y, np.array([0.0, 1.0], dtype=np.float64)
        return float(TABLE_H - y), np.array([0.0, -1.0], dtype=np.float64)

    @staticmethod
    def fit_regression_direction(
        pts: List[np.ndarray], times: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        3~4프레임 구간의 위치 좌표열에 선형 회귀(최소자승법)를 적용하여
        바운딩 박스 미세 떨림에 흔들리지 않는 단위 진행 방향 벡터 u를 추출합니다.
        """
        n = len(pts)
        if n < 2:
            return np.zeros(2, dtype=np.float64)
        arr = np.asarray(pts, dtype=np.float64)
        if times is not None and len(times) == n and (times[-1] - times[0]) > 1e-6:
            t_arr = np.asarray(times, dtype=np.float64)
        else:
            t_arr = np.arange(n, dtype=np.float64)
        t_centered = t_arr - np.mean(t_arr)
        denom = float(np.dot(t_centered, t_centered))
        if denom < 1e-9:
            diff = arr[-1] - arr[0]
        else:
            diff = (t_centered @ (arr - np.mean(arr, axis=0))) / denom
        norm = float(np.linalg.norm(diff))
        if norm < 1e-6:
            return np.zeros(2, dtype=np.float64)
        return diff / norm

    def extract_in_out_directions(
        self, center_idx: int, window: int = 3
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        cue_lifecycle_path의 center_idx 기준 전방 3~4프레임 회귀 방향 벡터 u_in과
        후방 3~4프레임 회귀 방향 벡터 u_out, 그리고 꺾임 각도(도)를 반환합니다.
        """
        n = len(self.cue_lifecycle_path)
        if n < 3 or center_idx <= 0 or center_idx >= n - 1:
            return np.zeros(2), np.zeros(2), 0.0
        lo = max(0, center_idx - window)
        hi = min(n - 1, center_idx + window)
        in_pts = [self.cue_lifecycle_path[k].pos_cue for k in range(lo, center_idx + 1)]
        in_ts = [self.cue_lifecycle_path[k].t for k in range(lo, center_idx + 1)]
        out_pts = [self.cue_lifecycle_path[k].pos_cue for k in range(center_idx, hi + 1)]
        out_ts = [self.cue_lifecycle_path[k].t for k in range(center_idx, hi + 1)]

        u_in = self.fit_regression_direction(in_pts, in_ts)
        u_out = self.fit_regression_direction(out_pts, out_ts)
        if float(np.linalg.norm(u_in)) < 1e-6 or float(np.linalg.norm(u_out)) < 1e-6:
            return u_in, u_out, 0.0
        cos_v = float(np.clip(np.dot(u_in, u_out), -1.0, 1.0))
        bend_deg = float(np.degrees(math.acos(cos_v)))
        return u_in, u_out, bend_deg

    def append_sample(
        self,
        t: float,
        pos_cue: np.ndarray,
        vel_cue: np.ndarray,
        pos_b1: np.ndarray,
        pos_b2: np.ndarray,
        min_dist_to_targets: float = 9999.0,
    ) -> List[Tuple[int, float, Rail, np.ndarray]]:
        """
        현재 프레임의 수구 및 두 목적구 상태를 전생애 궤적에 추가하고,
        벽면 거리 극소점 돌출도(Valley Prominence) + 법선 속도 반전으로 새로 확정된 쿠션을 반환합니다.
        """
        sample = LifecycleSample(
            t=float(t),
            pos_cue=pos_cue.copy(),
            vel_cue=vel_cue.copy(),
            pos_b1=pos_b1.copy(),
            pos_b2=pos_b2.copy(),
        )
        self.cue_lifecycle_path.append(sample)
        curr_idx = len(self.cue_lifecycle_path) - 1
        newly_confirmed: List[Tuple[int, float, Rail, np.ndarray]] = []

        for rail in Rail:
            d_now, n_inward = self.rail_distance_and_normal(pos_cue, rail)
            st = self.rail_valleys[rail]

            # 잠금 해제 조건: 해당 레일에서 220.0mm 이상 깊숙이 벗어났을 때
            if st.locked:
                if d_now > st.max_dist_since_lock:
                    st.max_dist_since_lock = d_now
                if d_now >= self.SAME_RAIL_UNLOCK_DIST_MM:
                    st.locked = False
                    st.in_valley = False
                    st.entry_dist = d_now
                    st.min_dist = 9999.0
                continue

            if d_now > st.entry_dist or not st.in_valley:
                if d_now > self.RAIL_VALLEY_ZONE_MM:
                    st.entry_dist = d_now
                elif not st.in_valley:
                    st.entry_dist = max(st.entry_dist, d_now)

            if d_now <= self.RAIL_VALLEY_ZONE_MM:
                if not st.in_valley:
                    st.in_valley = True
                    st.min_dist = d_now
                    st.min_pos = pos_cue.copy()
                    st.min_time = t
                    st.min_idx = curr_idx
                elif d_now < st.min_dist:
                    st.min_dist = d_now
                    st.min_pos = pos_cue.copy()
                    st.min_time = t
                    st.min_idx = curr_idx

            if st.in_valley and st.min_pos is not None and curr_idx > st.min_idx:
                v_normal_out = float(np.dot(vel_cue, n_inward))
                if curr_idx >= 1 and st.min_idx < curr_idx:
                    dt_s = max(t - self.cue_lifecycle_path[st.min_idx].t, 1e-4)
                    v_disp_normal = float(np.dot(pos_cue - st.min_pos, n_inward)) / dt_s
                    v_normal_out = max(v_normal_out, v_disp_normal)

                approach_prominence = st.entry_dist - st.min_dist
                rebound_prominence = d_now - st.min_dist
                _, _, local_bend = self.extract_in_out_directions(st.min_idx, window=2)

                # 미세 떨림(±5mm, 돌출도 <= 8mm)은 무시하고,
                # 진짜 극소점 반사(돌출도 >= 10.5mm & 법선 속도 반전 > +25mm/s) 또는 밀착 반사(<= 54mm)만 확정
                is_valid_prominence = (
                    approach_prominence >= self.MIN_VALLEY_PROMINENCE_MM
                    and rebound_prominence >= self.MIN_VALLEY_PROMINENCE_MM
                    and v_normal_out > self.MIN_NORMAL_VEL_FLIP_MM_S
                    and local_bend >= 6.0
                )
                is_tight_bounce = (
                    st.min_dist <= self.RAIL_CONTACT_MARGIN_MM
                    and approach_prominence >= 9.0
                    and rebound_prominence >= 8.0
                    and v_normal_out > self.MIN_NORMAL_VEL_FLIP_MM_S
                    and local_bend >= 6.0
                )
                if (is_valid_prominence or is_tight_bounce) and (
                    min_dist_to_targets >= 68.0 or st.min_dist <= 45.0
                ):
                    ev = (st.min_idx, st.min_time, rail, st.min_pos.copy())
                    self.confirmed_cushions.append(ev)
                    newly_confirmed.append(ev)
                    self.timeline_events.append((st.min_time, "CUSHION", rail.name))
                    # 다른 레일 잠금은 해제하고 현재 레일은 재카운트 잠금
                    for other_r, other_st in self.rail_valleys.items():
                        if other_r == rail:
                            other_st.locked = True
                            other_st.in_valley = False
                            other_st.max_dist_since_lock = d_now
                        else:
                            other_st.locked = False
                            other_st.in_valley = False
                            other_st.entry_dist = self.rail_distance_and_normal(pos_cue, other_r)[0]
                            other_st.min_dist = 9999.0

        return newly_confirmed

    def unlock_all_rails_on_ball_hit(self) -> None:
        """수구가 목적구를 타격했을 때 동일 레일 재카운트 잠금을 해제합니다."""
        for r, st in self.rail_valleys.items():
            st.locked = False
            st.in_valley = False
            st.min_dist = 9999.0
            if self.cue_lifecycle_path:
                st.entry_dist = self.rail_distance_and_normal(
                    self.cue_lifecycle_path[-1].pos_cue, r
                )[0]


__all__ = [
    "BALL_DIAM",
    "BALL_R",
    "TABLE_H",
    "TABLE_W",
    "BallColor",
    "BallID",
    "CausalKissSafeFSM",
    "CueBallLifecycleTracker",
    "CueFlowEvent",
    "CueFlowPhysicsTwin",
    "CueTrajectorySegment",
    "FlowEventType",
    "FrameSnapshot",
    "LabHungarianBallIdentifier",
    "LifecycleSample",
    "Rail",
    "RailValleyState",
    "ShotResult",
    "ShotState",
    "ThreeCushionScoringFSM",
]
