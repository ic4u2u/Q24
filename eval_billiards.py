"""
Immutable Evaluator for 3-Cushion Vision Engine (`eval_billiards.py`)
파일 경로: eval_billiards.py

[절대 규칙 - Karpathy Autoresearch]
이 파일은 평가 기준선(Immutable Evaluator)이므로 실험 루프 도중 절대 수정해서는 안 됩니다.

평가 구성 (총 30개 고난도 현장 시나리오):
1. [S01 ~ S06] 0.3mm 초미세 스침 (Ultra-Micro Kiss: 굴절 1.25~1.75도 또는 목적구 미세 밀림 2.2~3.3mm)
2. [S07 ~ S12] 시속 35km(9722 mm/s) 초고속 뱅크샷 및 고속 쿠션 반사 (프레임 간 162mm 이동으로 인한 레일 마진 터널링 포함)
3. [S13 ~ S18] 1.0mm ~ 2.0mm 차이 근접 비접촉 통과 (1mm Gap Near-Miss: d_min = 62.5~63.5mm 무굴절·무충격 통과 및 지터 노이즈)
4. [S19 ~ S24] 코너 더블 쿠션(Corner Double Cushion) 및 레일 쿠션 타기(Rail Hugging) 분리
5. [S25 ~ S30] 일시적 프레임 드랍(Frame Drop, dt=2/60s~3/60s), 조기 2쿠션 미세 키스 잠금, 복합 가우시안 지터 노이즈
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from three_cushion_vision.config import BallColor, VisionConfig
from three_cushion_vision.scoring_fsm import ShotResult, ThreeCushionScoringFSM
from three_cushion_vision.tracker_adapter import OCSortTrackerAdapter

Point2D = Tuple[float, float]


@dataclass(frozen=True)
class BenchmarkScenario:
    """개별 벤치마크 시나리오 정의 데이터클래스"""

    scenario_id: str
    category: str
    description: str
    expected_scored: bool
    expected_bank_shot: bool
    generator: Callable[[], List[Tuple[float, Point2D, Point2D, Point2D]]]


def _interp_segment(
    p0: Point2D, p1: Point2D, steps: int
) -> List[Point2D]:
    """두 좌표 사이를 steps 개수만큼 선형 보간합니다."""
    return [
        (
            p0[0] + (p1[0] - p0[0]) * (i / float(steps)),
            p0[1] + (p1[1] - p0[1]) * (i / float(steps)),
        )
        for i in range(1, steps + 1)
    ]


def _build_piecewise_path(
    waypoints: List[Point2D], steps_per_seg: List[int]
) -> List[Point2D]:
    """구간별 지정된 프레임 수(steps_per_seg)로 궤적 리스트를 생성합니다."""
    path: List[Point2D] = [waypoints[0]]
    for idx, steps in enumerate(steps_per_seg):
        path.extend(_interp_segment(waypoints[idx], waypoints[idx + 1], max(1, steps)))
    return path


def _pack_frames(
    cue_pts: List[Point2D],
    red_pts: List[Point2D],
    yel_pts: List[Point2D],
    dt: float = 1.0 / 60.0,
    stop_frames: int = 22,
    drop_indices: Tuple[int, ...] = (),
) -> List[Tuple[float, Point2D, Point2D, Point2D]]:
    """
    60fps 프레임 시퀀스(타임스탬프, WHITE, RED, YELLOW 좌표)를 패키징합니다.
    drop_indices에 포함된 프레임은 드랍되어 dt 간격이 벌어지는 현상을 시뮬레이션합니다.
    """
    n_active = max(len(cue_pts), len(red_pts), len(yel_pts))
    total = n_active + stop_frames
    frames: List[Tuple[float, Point2D, Point2D, Point2D]] = []
    drop_set = set(drop_indices)

    for i in range(total):
        if i in drop_set:
            continue
        t = i * dt
        c = cue_pts[min(i, len(cue_pts) - 1)]
        r = red_pts[min(i, len(red_pts) - 1)]
        y = yel_pts[min(i, len(yel_pts) - 1)]
        frames.append((t, c, r, y))
    return frames


# ==============================================================================
# 그룹 1 [S01 ~ S06]: 0.3mm 초미세 스침 (Ultra-Micro Kiss)
# 60fps 샘플링 한계로 인해 프레임 간 보간 최소거리 d_min이 64.5~74.5mm에 위치하고,
# 수구 굴절각이 1.25~1.75도이거나 목적구가 충돌 법선 방향으로 2.2~3.2mm 미세 밀리는 시나리오
# ==============================================================================
def _make_ultra_micro_kiss_scenario(
    deflection_dy_mm: float,
    target_push_mm: float,
    closest_y_gap_mm: float,
    kiss_on_first_ball: bool = False,
) -> Callable[[], List[Tuple[float, Point2D, Point2D, Point2D]]]:
    def _gen() -> List[Tuple[float, Point2D, Point2D, Point2D]]:
        if not kiss_on_first_ball:
            # 제1목적구(RED) 정상 타격 -> 3쿠션(TOP, RIGHT, BOTTOM) -> 제2목적구(YELLOW) 0.3mm 초미세 스침
            waypoints = [
                (400.0, 700.0),
                (800.0, 500.0),          # RED(800, 450) 타격
                (1500.0, 32.0),          # 1C: TOP
                (2808.0, 650.0),         # 2C: RIGHT
                (2000.0, 1388.0),        # 3C: BOTTOM
                (1800.0, 700.0),         # 직선 접근 시작
                (1350.0, 700.0),         # YELLOW(1350, 700 - closest_y_gap_mm) 초미세 스침!
                (900.0, 700.0 + deflection_dy_mm),  # 스침 직후 미세 굴절(수구가 +y로 꺾임, YELLOW는 -y에 위치)
            ]
            steps = [8, 8, 8, 8, 8, 8, 8]
            cue = _build_piecewise_path(waypoints, steps)
            red = [(800.0, 450.0)] * 9 + [(820.0, 390.0)] * (len(cue) - 9)
            y_base = 700.0 - closest_y_gap_mm
            yel: List[Point2D] = []
            # 스침 프레임은 index 48 (6구간 * 8스텝)
            for idx in range(len(cue)):
                if idx <= 48:
                    yel.append((1350.0, y_base))
                elif idx == 49:
                    yel.append((1350.0, y_base - target_push_mm * 0.45))
                elif idx == 50:
                    yel.append((1350.0, y_base - target_push_mm * 0.80))
                else:
                    yel.append((1350.0, y_base - target_push_mm))
            return _pack_frames(cue, red, yel)
        else:
            # 제1목적구(RED)를 0.3mm 초미세 스침으로 맞힌 뒤 3쿠션 후 제2목적구(YELLOW) 정면 타격
            r_base_y = 700.0 - closest_y_gap_mm
            waypoints = [
                (400.0, 700.0),
                (850.0, 700.0),          # RED(850, r_base_y) 0.3mm 초미세 스침!
                (1300.0, 700.0 + deflection_dy_mm),
                (1600.0, 32.0),          # 1C: TOP
                (2808.0, 650.0),         # 2C: RIGHT
                (2000.0, 1388.0),        # 3C: BOTTOM
                (1200.0, 850.0),         # YELLOW(1200, 800) 정면 충돌
                (1120.0, 790.0),
            ]
            steps = [8, 8, 8, 8, 8, 8, 6]
            cue = _build_piecewise_path(waypoints, steps)
            red = []
            for idx in range(len(cue)):
                if idx <= 8:
                    red.append((850.0, r_base_y))
                elif idx == 9:
                    red.append((850.0, r_base_y - target_push_mm * 0.5))
                else:
                    red.append((850.0, r_base_y - target_push_mm))
            yel = [(1200.0, 800.0)] * 49 + [(1150.0, 760.0)] * (len(cue) - 49)
            return _pack_frames(cue, red, yel)

    return _gen


# ==============================================================================
# 그룹 2 [S07 ~ S12]: 시속 35km (9722 mm/s) 초고속 뱅크샷 및 고속 쿠션 터널링
# 60fps에서 프레임당 ~160mm 이동하여 레일 마진(44.75mm) 내부에 프레임 샘플이 찍히지 않고
# 프레임 사이(Sub-Frame)에서 쿠션을 치고 튕겨 나오는 고속 터널링 시나리오 포함
# ==============================================================================
def _make_high_speed_35kmh_bank_shot(
    subframe_rail_tunneling: bool,
    cushions_before_first: int = 3,
    valid_second_hit: bool = True,
) -> Callable[[], List[Tuple[float, Point2D, Point2D, Point2D]]]:
    def _gen() -> List[Tuple[float, Point2D, Point2D, Point2D]]:
        if subframe_rail_tunneling:
            # 35km/h (프레임당 ~160mm 이동):
            # 레일 실제 반사점(30.75mm) 전후로 샘플링 프레임이 마진(44.75mm) 바깥(예: y=68mm -> y=74mm)에 찍힘!
            cue: List[Point2D] = [
                (500.0, 600.0),
                (630.0, 440.0),
                (760.0, 280.0),
                (890.0, 120.0),
                # TOP 레일(y=30.75) 서브프레임 반사! (이전 프레임 y=62.0, 다음 프레임 y=68.0 -> 둘 다 44.75mm 바깥!)
                (940.0, 62.0),
                (1060.0, 68.0),
                (1220.0, 200.0),
                (1550.0, 330.0),
                (1950.0, 450.0),
                (2380.0, 560.0),
                # RIGHT 레일(x=2809.25) 서브프레임 반사! (x=2778.0 -> x=2772.0 -> 둘 다 2795.25mm 안쪽!)
                (2778.0, 660.0),
                (2772.0, 760.0),
                (2520.0, 960.0),
                (2240.0, 1180.0),
                # BOTTOM 레일(y=1389.25) 서브프레임 반사! (y=1358.0 -> y=1352.0 -> 둘 다 1375.25mm 위쪽!)
                (2020.0, 1358.0),
                (1900.0, 1352.0),
                (1720.0, 1140.0),
                (1550.0, 940.0),
                (1400.0, 780.0),  # 제1목적구(RED: 1400, 740) 타격!
                (1180.0, 620.0),
                (960.0, 480.0) if valid_second_hit else (960.0, 250.0),  # 제2목적구(YELLOW: 960, 440)
                (880.0, 420.0) if valid_second_hit else (880.0, 200.0),
            ]
            red = [(1400.0, 740.0)] * 19 + [(1350.0, 690.0)] * (len(cue) - 19)
            yel = (
                [(960.0, 440.0)] * 21 + [(910.0, 390.0)] * (len(cue) - 21)
                if valid_second_hit
                else [(960.0, 440.0)] * len(cue)
            )
            return _pack_frames(cue, red, yel)
        else:
            # 35km/h 고속 주행 (구간당 3~4프레임 = 약 9000~10000 mm/s)
            waypoints = [
                (400.0, 700.0),
                (1100.0, 31.0),          # 1C: TOP
                (2809.0, 550.0),         # 2C: RIGHT
                (2100.0, 1389.0),        # 3C: BOTTOM
                (1400.0, 780.0),         # RED(1400, 740) 타격
                (950.0, 480.0),          # YELLOW(950, 440) 타격
                (860.0, 410.0),
            ]
            steps = [4, 5, 4, 4, 4, 4]
            cue = _build_piecewise_path(waypoints, steps)
            red = [(1400.0, 740.0)] * 18 + [(1340.0, 680.0)] * (len(cue) - 18)
            yel = [(950.0, 440.0)] * 22 + [(890.0, 380.0)] * (len(cue) - 22)
            return _pack_frames(cue, red, yel)

    return _gen


# ==============================================================================
# 그룹 3 [S13 ~ S18]: 1.0mm ~ 2.0mm 차이 근접 비접촉 통과 (1mm Gap Near-Miss)
# 표면 간격 1.0mm (중심거리 62.5mm) ~ 2.0mm (중심거리 63.5mm)로 스치지 않고 직진 통과!
# 카메라 센서 노이즈(무작위 지터 <= 0.7mm)가 섞여 있어도 절대 득점으로 오탐하면 안 됨!
# ==============================================================================
def _make_1mm_gap_near_miss_scenario(
    surface_gap_mm: float,
    camera_jitter_mm: float = 0.0,
    seed: int = 42,
) -> Callable[[], List[Tuple[float, Point2D, Point2D, Point2D]]]:
    def _gen() -> List[Tuple[float, Point2D, Point2D, Point2D]]:
        rng = np.random.default_rng(seed)
        center_dist_mm = 61.5 + surface_gap_mm  # 2R(61.5mm) + gap(1.0~2.5mm)
        waypoints = [
            (400.0, 700.0),
            (800.0, 500.0),          # 제1목적구(RED: 800, 450) 정상 접촉
            (1500.0, 32.0),          # 1C: TOP
            (2808.0, 600.0),         # 2C: RIGHT
            (2000.0, 1388.0),        # 3C: BOTTOM
            (1800.0, 700.0),         # y=700 직선 진입
            (1350.0, 700.0),         # YELLOW(1350, 700 - center_dist_mm) 옆을 간격 surface_gap_mm로 무굴절 통과!
            (900.0, 700.0),          # 굴절 없이 y=700 직선 유지 후 정지
        ]
        cue = _build_piecewise_path(waypoints, [8, 8, 8, 8, 8, 8, 8])
        red = [(800.0, 450.0)] * len(cue)
        y_rest = 700.0 - center_dist_mm
        yel: List[Point2D] = []
        for _ in range(len(cue)):
            if camera_jitter_mm > 0.0:
                jx = float(rng.uniform(-camera_jitter_mm, camera_jitter_mm))
                jy = float(rng.uniform(-camera_jitter_mm, camera_jitter_mm))
                yel.append((1350.0 + jx, y_rest + jy))
            else:
                yel.append((1350.0, y_rest))
        return _pack_frames(cue, red, yel)

    return _gen


# ==============================================================================
# 그룹 4 [S19 ~ S24]: 코너 더블 쿠션(Corner Double Cushion) 및 레일 쿠션 타기(Rail Hugging)
# ==============================================================================
def _make_corner_or_hugging_scenario(
    mode: str,
) -> Callable[[], List[Tuple[float, Point2D, Point2D, Point2D]]]:
    def _gen() -> List[Tuple[float, Point2D, Point2D, Point2D]]:
        if mode == "CORNER_DOUBLE_TOP_RIGHT":
            # RED 타격 -> 우상단 코너에서 TOP(y=32) 치고 바로 다음 프레임에 RIGHT(x=2808) 연속 반사(2쿠션) -> BOTTOM(3쿠션) -> YELLOW 타격
            waypoints = [
                (600.0, 600.0),
                (1200.0, 400.0),     # RED(1200, 350) 타격
                (2765.0, 32.0),      # 1C: TOP 코너 진입
                (2808.0, 75.0),      # 2C: RIGHT 코너 즉시 반사 (코너 더블 쿠션!)
                (2100.0, 1388.0),    # 3C: BOTTOM
                (1400.0, 800.0),     # YELLOW(1400, 750) 타격
                (1300.0, 720.0),
            ]
            cue = _build_piecewise_path(waypoints, [8, 8, 2, 10, 8, 6])
            red = [(1200.0, 350.0)] * 9 + [(1230.0, 300.0)] * (len(cue) - 9)
            yel = [(1400.0, 750.0)] * 37 + [(1350.0, 700.0)] * (len(cue) - 37)
            return _pack_frames(cue, red, yel)

        if mode == "CORNER_SUBFRAME_TOP_LEFT":
            # 좌상단 코너 깊숙이(x=34, y=34) 한 번에 진입하여 vx와 vy가 동시에 반전되는 코너 2쿠션! -> BOTTOM(3쿠션) -> 제1·2목적구 타격(뱅크샷)
            waypoints = [
                (800.0, 600.0),
                (33.0, 33.0),        # 좌상단 코너(LEFT + TOP 동시 반전 -> 2쿠션!)
                (900.0, 1388.0),     # 3C: BOTTOM
                (1400.0, 800.0),     # RED(1400, 750) 타격
                (1800.0, 500.0),     # YELLOW(1800, 450) 타격
                (1900.0, 420.0),
            ]
            cue = _build_piecewise_path(waypoints, [10, 10, 8, 8, 6])
            red = [(1400.0, 750.0)] * 29 + [(1440.0, 700.0)] * (len(cue) - 29)
            yel = [(1800.0, 450.0)] * 37 + [(1850.0, 400.0)] * (len(cue) - 37)
            return _pack_frames(cue, red, yel)

        if mode == "CORNER_DOUBLE_BOTTOM_RIGHT":
            # RED 타격 -> TOP(1C) -> 우하단 코너에서 RIGHT(2C) 후 즉시 BOTTOM(3C) -> YELLOW 타격
            waypoints = [
                (500.0, 700.0),
                (900.0, 500.0),      # RED(900, 450) 타격
                (1800.0, 32.0),      # 1C: TOP
                (2808.0, 1340.0),    # 2C: RIGHT 코너 인접
                (2760.0, 1388.0),    # 3C: BOTTOM 코너 연속 반사
                (1600.0, 800.0),     # YELLOW(1600, 750) 타격
                (1500.0, 720.0),
            ]
            cue = _build_piecewise_path(waypoints, [8, 8, 8, 2, 8, 6])
            red = [(900.0, 450.0)] * 9 + [(930.0, 400.0)] * (len(cue) - 9)
            yel = [(1600.0, 750.0)] * 35 + [(1550.0, 700.0)] * (len(cue) - 35)
            return _pack_frames(cue, red, yel)

        if mode == "RAIL_HUGGING_2C_ONLY":
            # RED 타격 -> TOP 레일(y=32 -> y=38 -> y=31)을 따라 미끄러진 뒤(1쿠션만 인정되어야 함!) -> RIGHT(2쿠션) -> YELLOW 타격 (총 2쿠션이므로 노카운트!)
            waypoints = [
                (500.0, 500.0),
                (800.0, 300.0),      # RED(800, 250) 타격
                (1100.0, 32.0),      # 1C: TOP 진입
                (1500.0, 39.0),      # TOP 마진 내 미끄러짐 (쿠션 타기 - 추가 쿠션 아님!)
                (1900.0, 31.0),      # TOP 마진 내 재접근 (쿠션 타기 - 추가 쿠션 아님!)
                (2808.0, 500.0),     # 2C: RIGHT
                (2000.0, 900.0),     # YELLOW(2000, 850) 타격 (2쿠션 상태이므로 노카운트!)
                (1900.0, 950.0),
            ]
            cue = _build_piecewise_path(waypoints, [8, 8, 6, 6, 8, 8, 6])
            red = [(800.0, 250.0)] * len(cue)
            yel = [(2000.0, 850.0)] * len(cue)
            return _pack_frames(cue, red, yel)

        if mode == "SHALLOW_BOUNCE_OUT_AND_REENTER_SAME_RAIL":
            # RED 타격 -> TOP(1C) -> RIGHT(2C) -> TOP(3C, 완전히 마진 밖 y=120으로 나왔다가 다시 TOP 진입) -> YELLOW 타격 (정당한 3쿠션 득점!)
            waypoints = [
                (500.0, 500.0),
                (900.0, 350.0),      # RED(900, 300) 타격
                (1600.0, 32.0),      # 1C: TOP
                (2808.0, 220.0),     # 2C: RIGHT
                (2100.0, 32.0),      # 3C: TOP 재진입
                (1400.0, 600.0),     # YELLOW(1400, 550) 타격
                (1300.0, 650.0),
            ]
            cue = _build_piecewise_path(waypoints, [8, 8, 8, 8, 8, 6])
            red = [(900.0, 300.0)] * len(cue)
            yel = [(1400.0, 550.0)] * len(cue)
            return _pack_frames(cue, red, yel)

        # mode == "EARLY_2C_MICRO_KISS_LOCK"
        # RED 타격 -> 2쿠션 후 YELLOW를 0.3mm 미세 스침(조기 키스! NO_SCORE_LOCKED 되어야 함) -> 3번째 쿠션 후 YELLOW 재타격 (노카운트!)
        waypoints = [
            (400.0, 700.0),
            (800.0, 500.0),          # RED(800, 450) 타격
            (1500.0, 32.0),          # 1C: TOP
            (2808.0, 600.0),         # 2C: RIGHT
            (2200.0, 900.0),         # YELLOW(2200, 830) 2쿠션 상태에서 초미세 스침(굴절 + 밀림)!
            (1800.0, 1388.0),        # 3C: BOTTOM (사후 3쿠션)
            (1500.0, 850.0),         # 이동한 YELLOW(1500, 800) 재타격
            (1400.0, 800.0),
        ]
        cue = _build_piecewise_path(waypoints, [8, 8, 8, 8, 8, 8, 6])
        red = [(800.0, 450.0)] * len(cue)
        yel = (
            [(2200.0, 830.0)] * 32
            + [(2200.0, 826.0), (2200.0, 824.0)]
            + [(1500.0, 800.0)] * (len(cue) - 34)
        )
        return _pack_frames(cue, red, yel)

    return _gen


# ==============================================================================
# 그룹 5 [S25 ~ S30]: 일시적 프레임 드랍(Frame Drop) 및 복합 노이즈 시나리오
# ==============================================================================
def _make_frame_drop_or_noisy_scenario(
    variant: str,
) -> Callable[[], List[Tuple[float, Point2D, Point2D, Point2D]]]:
    def _gen() -> List[Tuple[float, Point2D, Point2D, Point2D]]:
        if variant == "FRAME_DROP_AT_2ND_BALL_HIT":
            # 일반 3쿠션 득점 샷인데 제2목적구(YELLOW) 충돌 직전/직후 프레임(39, 40)이 카메라 드랍됨 -> CCD 보간으로 득점 잡아내야 함!
            waypoints = [
                (400.0, 700.0),
                (800.0, 500.0),      # RED(800, 450)
                (1500.0, 32.0),      # 1C
                (2808.0, 650.0),     # 2C
                (1900.0, 1388.0),    # 3C
                (1200.0, 900.0),     # YELLOW(1200, 855) 통과
                (1000.0, 780.0),
            ]
            cue = _build_piecewise_path(waypoints, [8, 8, 8, 8, 8, 8])
            red = [(800.0, 450.0)] * len(cue)
            yel = [(1200.0, 855.0)] * len(cue)
            return _pack_frames(cue, red, yel, drop_indices=(39, 40))

        if variant == "FRAME_DROP_AT_CUSHION_APEX":
            # 2번째 쿠션(RIGHT: x=2808, index 24) 프레임이 드랍되어 x=2620 -> x=2640으로 바로 넘어감 -> 서브프레임 쿠션 복원 필요!
            waypoints = [
                (400.0, 700.0),
                (800.0, 500.0),      # RED(800, 450)
                (1500.0, 32.0),      # 1C: TOP
                (2808.0, 650.0),     # 2C: RIGHT (index 24 드랍!)
                (1900.0, 1388.0),    # 3C: BOTTOM
                (1200.0, 890.0),     # YELLOW(1200, 850)
                (1100.0, 820.0),
            ]
            cue = _build_piecewise_path(waypoints, [8, 8, 8, 8, 8, 6])
            red = [(800.0, 450.0)] * len(cue)
            yel = [(1200.0, 850.0)] * len(cue)
            return _pack_frames(cue, red, yel, drop_indices=(24,))

        if variant == "JITTER_NOISE_VALID_3C":
            # 측정 좌표에 +-0.6mm 가우시안 노이즈가 낀 일반 3쿠션 득점
            rng = np.random.default_rng(101)
            waypoints = [
                (400.0, 700.0),
                (800.0, 500.0),
                (1500.0, 32.0),
                (2808.0, 650.0),
                (1900.0, 1388.0),
                (1200.0, 890.0),
                (1100.0, 820.0),
            ]
            cue_clean = _build_piecewise_path(waypoints, [8, 8, 8, 8, 8, 6])
            cue = [
                (x + float(rng.normal(0, 0.5)), y + float(rng.normal(0, 0.5)))
                for (x, y) in cue_clean
            ]
            red = [(800.0, 450.0)] * len(cue)
            yel = [(1200.0, 850.0)] * len(cue)
            return _pack_frames(cue, red, yel)

        if variant == "JITTER_NOISE_2C_MISS":
            # 2쿠션만 치고 제2적구를 맞힌 노카운트 시나리오 (가우시안 노이즈 포함)
            rng = np.random.default_rng(202)
            waypoints = [
                (400.0, 700.0),
                (800.0, 500.0),      # RED(800, 450)
                (1500.0, 32.0),      # 1C: TOP
                (2808.0, 650.0),     # 2C: RIGHT
                (1600.0, 900.0),     # YELLOW(1600, 855) -> 2쿠션 타격 (노카운트)
                (1500.0, 920.0),
            ]
            cue_clean = _build_piecewise_path(waypoints, [8, 8, 8, 8, 6])
            cue = [
                (x + float(rng.normal(0, 0.5)), y + float(rng.normal(0, 0.5)))
                for (x, y) in cue_clean
            ]
            red = [(800.0, 450.0)] * len(cue)
            yel = [(1600.0, 855.0)] * len(cue)
            return _pack_frames(cue, red, yel)

        if variant == "WRONG_SIDE_JITTER_NEAR_MISS":
            # 제2목적구(1350, 632 -> d_min=68mm 그레이존) 옆을 지나가는데,
            # 카메라 노이즈로 인해 목적구가 수구 쪽(+y 방향, 즉 물리적으로 불가능한 끌려오는 방향)으로 3.6mm 튄 경우 -> 스침이 아니므로 기각해야 함!
            waypoints = [
                (400.0, 700.0),
                (800.0, 500.0),      # RED(800, 450)
                (1500.0, 32.0),      # 1C
                (2808.0, 600.0),     # 2C
                (2000.0, 1388.0),    # 3C
                (1800.0, 700.0),
                (1350.0, 700.0),     # YELLOW(1350, 632) 옆을 굴절 0도로 직진 통과
                (900.0, 700.0),
            ]
            cue = _build_piecewise_path(waypoints, [8, 8, 8, 8, 8, 8, 8])
            red = [(800.0, 450.0)] * len(cue)
            # 물리적 충돌이라면 YELLOW(y=632)는 수구(y=700) 반대 방향인 -y(y < 632)로 밀려야 하지만,
            # 센서 글리치로 +y(y = 635.8, 수구 쪽으로 끌려옴)로 튄 가짜 임펄스!
            yel = [(1350.0, 632.0)] * 49 + [(1350.0, 635.8)] * (len(cue) - 49)
            return _pack_frames(cue, red, yel)

        # variant == "DOUBLE_MICRO_KISS_BOTH_TARGETS"
        # 제1목적구(RED)도 0.3mm 미세 스침(굴절 1.5도 + 밀림 2.6mm), 3쿠션 후 제2목적구(YELLOW)도 0.3mm 미세 스침(굴절 1.4도 + 밀림 2.5mm)
        waypoints = [
            (400.0, 700.0),
            (850.0, 700.0),          # RED(850, 631) 미세 스침!
            (1300.0, 712.0),         # +1.5도 굴절
            (1600.0, 32.0),          # 1C: TOP
            (2808.0, 600.0),         # 2C: RIGHT
            (2000.0, 1388.0),        # 3C: BOTTOM
            (1800.0, 700.0),
            (1350.0, 700.0),         # YELLOW(1350, 631) 미세 스침!
            (900.0, 711.5),          # +1.46도 굴절
        ]
        cue = _build_piecewise_path(waypoints, [8, 8, 8, 8, 8, 8, 8, 8])
        red = [(850.0, 631.0)] * 9 + [(850.0, 628.4)] * (len(cue) - 9)
        yel = [(1350.0, 631.0)] * 57 + [(1350.0, 628.5)] * (len(cue) - 57)
        return _pack_frames(cue, red, yel)

    return _gen


def build_all_30_scenarios() -> List[BenchmarkScenario]:
    """30개의 극한 쓰리쿠션 물리 벤치마크 시나리오를 생성합니다."""
    return [
        # --- Group 1: 0.3mm Ultra-Micro Kiss (S01 ~ S06) ---
        BenchmarkScenario(
            "S01", "ULTRA_MICRO_KISS",
            "0.3mm kiss on 2nd ball (d_min=68mm, deflection=1.40 deg, push=2.6mm)",
            True, False,
            _make_ultra_micro_kiss_scenario(deflection_dy_mm=11.0, target_push_mm=2.6, closest_y_gap_mm=68.0),
        ),
        BenchmarkScenario(
            "S02", "ULTRA_MICRO_KISS",
            "0.3mm feather kiss on 2nd ball (d_min=71mm, deflection=1.30 deg, push=2.3mm)",
            True, False,
            _make_ultra_micro_kiss_scenario(deflection_dy_mm=10.2, target_push_mm=2.3, closest_y_gap_mm=71.0),
        ),
        BenchmarkScenario(
            "S03", "ULTRA_MICRO_KISS",
            "0.3mm impulse-only kiss on 2nd ball (d_min=69mm, deflection=0.8 deg, push=2.5mm)",
            True, False,
            _make_ultra_micro_kiss_scenario(deflection_dy_mm=6.2, target_push_mm=2.5, closest_y_gap_mm=69.0),
        ),
        BenchmarkScenario(
            "S04", "ULTRA_MICRO_KISS",
            "0.3mm deflection-dominant kiss on 2nd ball (d_min=67mm, deflection=1.55 deg, push=1.8mm)",
            True, False,
            _make_ultra_micro_kiss_scenario(deflection_dy_mm=12.2, target_push_mm=1.8, closest_y_gap_mm=67.0),
        ),
        BenchmarkScenario(
            "S05", "ULTRA_MICRO_KISS",
            "0.3mm ultra-thin kiss on 1st ball then 3C then 2nd ball hit",
            True, False,
            _make_ultra_micro_kiss_scenario(deflection_dy_mm=11.5, target_push_mm=2.4, closest_y_gap_mm=69.0, kiss_on_first_ball=True),
        ),
        BenchmarkScenario(
            "S06", "ULTRA_MICRO_KISS",
            "0.5mm standard micro-kiss on 2nd ball (d_min=70mm, deflection=2.5 deg, push=4.5mm)",
            True, False,
            _make_ultra_micro_kiss_scenario(deflection_dy_mm=20.0, target_push_mm=4.5, closest_y_gap_mm=70.0),
        ),

        # --- Group 2: 35km/h High-Speed Bank Shots & Sub-Frame Tunneling (S07 ~ S12) ---
        BenchmarkScenario(
            "S07", "HIGH_SPEED_35KMH",
            "35km/h bank shot with sub-frame rail tunneling on all 3 cushions",
            True, True,
            _make_high_speed_35kmh_bank_shot(subframe_rail_tunneling=True, valid_second_hit=True),
        ),
        BenchmarkScenario(
            "S08", "HIGH_SPEED_35KMH",
            "35km/h bank shot with sub-frame rail tunneling but misses 2nd ball",
            False, False,
            _make_high_speed_35kmh_bank_shot(subframe_rail_tunneling=True, valid_second_hit=False),
        ),
        BenchmarkScenario(
            "S09", "HIGH_SPEED_35KMH",
            "35km/h direct rail bank shot (9700 mm/s)",
            True, True,
            _make_high_speed_35kmh_bank_shot(subframe_rail_tunneling=False, valid_second_hit=True),
        ),
        BenchmarkScenario(
            "S10", "HIGH_SPEED_35KMH",
            "35km/h fast regular 3-cushion with sub-frame TOP rail bounce (y0=58 -> y1=64)",
            True, False,
            lambda: _pack_frames(
                _build_piecewise_path(
                    [(400.0, 700.0), (800.0, 500.0), (1450.0, 58.0), (1600.0, 64.0), (2808.0, 600.0), (2000.0, 1388.0), (1200.0, 880.0), (1100.0, 820.0)],
                    [5, 4, 1, 5, 5, 5, 4],
                ),
                [(800.0, 450.0)] * 35,
                [(1200.0, 840.0)] * 35,
            ),
        ),
        BenchmarkScenario(
            "S11", "HIGH_SPEED_35KMH",
            "35km/h fast 3-cushion with sub-frame RIGHT rail bounce (x0=2782 -> x1=2776)",
            True, False,
            lambda: _pack_frames(
                _build_piecewise_path(
                    [(400.0, 700.0), (800.0, 500.0), (1500.0, 32.0), (2782.0, 580.0), (2776.0, 680.0), (1950.0, 1388.0), (1200.0, 880.0), (1100.0, 820.0)],
                    [5, 5, 4, 1, 5, 5, 4],
                ),
                [(800.0, 450.0)] * 35,
                [(1200.0, 840.0)] * 35,
            ),
        ),
        BenchmarkScenario(
            "S12", "HIGH_SPEED_35KMH",
            "35km/h fast 3-cushion with sub-frame LEFT rail bounce (x0=58 -> x1=65)",
            True, False,
            lambda: _pack_frames(
                _build_piecewise_path(
                    [(2400.0, 700.0), (2000.0, 500.0), (1300.0, 32.0), (58.0, 580.0), (65.0, 680.0), (900.0, 1388.0), (1600.0, 880.0), (1700.0, 820.0)],
                    [5, 5, 4, 1, 5, 5, 4],
                ),
                [(2000.0, 450.0)] * 35,
                [(1600.0, 840.0)] * 35,
            ),
        ),

        # --- Group 3: 1mm ~ 2mm Gap Near-Miss Without Contact (S13 ~ S18) ---
        BenchmarkScenario(
            "S13", "NEAR_MISS_1MM",
            "1.0mm surface gap near-miss (d_min=62.5mm, 0 deg deflection, 0mm push)",
            False, False,
            _make_1mm_gap_near_miss_scenario(surface_gap_mm=1.0, camera_jitter_mm=0.0),
        ),
        BenchmarkScenario(
            "S14", "NEAR_MISS_1MM",
            "1.2mm surface gap near-miss (d_min=62.7mm, 0 deg deflection, 0mm push)",
            False, False,
            _make_1mm_gap_near_miss_scenario(surface_gap_mm=1.2, camera_jitter_mm=0.0),
        ),
        BenchmarkScenario(
            "S15", "NEAR_MISS_1MM",
            "1.5mm surface gap near-miss with 0.5mm camera jitter (d_min=63.0mm)",
            False, False,
            _make_1mm_gap_near_miss_scenario(surface_gap_mm=1.5, camera_jitter_mm=0.5, seed=15),
        ),
        BenchmarkScenario(
            "S16", "NEAR_MISS_1MM",
            "1.8mm surface gap near-miss with 0.6mm camera jitter (d_min=63.3mm)",
            False, False,
            _make_1mm_gap_near_miss_scenario(surface_gap_mm=1.8, camera_jitter_mm=0.6, seed=16),
        ),
        BenchmarkScenario(
            "S17", "NEAR_MISS_1MM",
            "3.0mm surface gap near-miss in gray-zone (d_min=64.5mm, jitter=0.6mm)",
            False, False,
            _make_1mm_gap_near_miss_scenario(surface_gap_mm=3.0, camera_jitter_mm=0.6, seed=17),
        ),
        BenchmarkScenario(
            "S18", "NEAR_MISS_1MM",
            "8.5mm surface gap near-miss in gray-zone (d_min=70.0mm, jitter=0.7mm)",
            False, False,
            _make_1mm_gap_near_miss_scenario(surface_gap_mm=8.5, camera_jitter_mm=0.7, seed=18),
        ),

        # --- Group 4: Corner Double Cushion & Rail Hugging (S19 ~ S24) ---
        BenchmarkScenario(
            "S19", "CORNER_AND_RAIL",
            "Corner double cushion (TOP -> RIGHT in 2 frames) + BOTTOM = 3C valid",
            True, False,
            _make_corner_or_hugging_scenario("CORNER_DOUBLE_TOP_RIGHT"),
        ),
        BenchmarkScenario(
            "S20", "CORNER_AND_RAIL",
            "Simultaneous corner apex bounce (LEFT + TOP in single vertex) + BOTTOM bank shot",
            True, True,
            _make_corner_or_hugging_scenario("CORNER_SUBFRAME_TOP_LEFT"),
        ),
        BenchmarkScenario(
            "S21", "CORNER_AND_RAIL",
            "Corner double cushion (RIGHT -> BOTTOM in 2 frames) after TOP = 3C valid",
            True, False,
            _make_corner_or_hugging_scenario("CORNER_DOUBLE_BOTTOM_RIGHT"),
        ),
        BenchmarkScenario(
            "S22", "CORNER_AND_RAIL",
            "Rail hugging along TOP rail + RIGHT rail = only 2 cushions (must be NO SCORE)",
            False, False,
            _make_corner_or_hugging_scenario("RAIL_HUGGING_2C_ONLY"),
        ),
        BenchmarkScenario(
            "S23", "CORNER_AND_RAIL",
            "TOP -> RIGHT -> TOP re-entry after leaving rail zone = 3 cushions valid",
            True, False,
            _make_corner_or_hugging_scenario("SHALLOW_BOUNCE_OUT_AND_REENTER_SAME_RAIL"),
        ),
        BenchmarkScenario(
            "S24", "CORNER_AND_RAIL",
            "Early 2nd-ball micro-kiss at 2 cushions then 3rd cushion then re-hit (NO_SCORE_LOCKED)",
            False, False,
            _make_corner_or_hugging_scenario("EARLY_2C_MICRO_KISS_LOCK"),
        ),

        # --- Group 5: Frame Drops & Sensor Glitch Robustness (S25 ~ S30) ---
        BenchmarkScenario(
            "S25", "FRAME_DROP_NOISE",
            "2-frame drop right across 2nd ball impact (sub-frame CCD interpolation)",
            True, False,
            _make_frame_drop_or_noisy_scenario("FRAME_DROP_AT_2ND_BALL_HIT"),
        ),
        BenchmarkScenario(
            "S26", "FRAME_DROP_NOISE",
            "Frame drop right at RIGHT cushion apex (x=2620 -> x=2640 velocity flip)",
            True, False,
            _make_frame_drop_or_noisy_scenario("FRAME_DROP_AT_CUSHION_APEX"),
        ),
        BenchmarkScenario(
            "S27", "FRAME_DROP_NOISE",
            "Gaussian position jitter (sigma=0.5mm) on valid 3-cushion shot",
            True, False,
            _make_frame_drop_or_noisy_scenario("JITTER_NOISE_VALID_3C"),
        ),
        BenchmarkScenario(
            "S28", "FRAME_DROP_NOISE",
            "Gaussian position jitter (sigma=0.5mm) on 2-cushion miss",
            False, False,
            _make_frame_drop_or_noisy_scenario("JITTER_NOISE_2C_MISS"),
        ),
        BenchmarkScenario(
            "S29", "FRAME_DROP_NOISE",
            "Wrong-direction sensor glitch on stationary 2nd ball during near-miss (must reject)",
            False, False,
            _make_frame_drop_or_noisy_scenario("WRONG_SIDE_JITTER_NEAR_MISS"),
        ),
        BenchmarkScenario(
            "S30", "FRAME_DROP_NOISE",
            "Double 0.3mm ultra-micro kiss on both 1st ball (RED) and 2nd ball (YELLOW)",
            True, False,
            _make_frame_drop_or_noisy_scenario("DOUBLE_MICRO_KISS_BOTH_TARGETS"),
        ),
    ]


def run_benchmark() -> Dict[str, object]:
    """30개 시나리오 전체를 실행하고 정량 지표(Accuracy, FPR, FNR, ms/frame)를 산출합니다."""
    scenarios = build_all_30_scenarios()
    tp = tn = fp = fn = 0
    total_frames = 0
    total_kinematics_sec = 0.0
    failures: List[str] = []

    for sc in scenarios:
        frames = sc.generator()
        adapter = OCSortTrackerAdapter()
        fsm = ThreeCushionScoringFSM(initial_cue_ball=BallColor.WHITE)
        shot_result: ShotResult | None = None

        for t_sec, c_pt, r_pt, y_pt in frames:
            dets = [
                {"color": BallColor.WHITE, "x_mm": c_pt[0], "y_mm": c_pt[1], "track_id": 1},
                {"color": BallColor.RED, "x_mm": r_pt[0], "y_mm": r_pt[1], "track_id": 2},
                {"color": BallColor.YELLOW, "x_mm": y_pt[0], "y_mm": y_pt[1], "track_id": 3},
            ]
            balls = adapter.update_from_detections(dets, timestamp=t_sec, enable_ema=False)

            t_start = time.perf_counter()
            res = fsm.update(balls=balls, timestamp=t_sec)
            t_elapsed = time.perf_counter() - t_start

            total_kinematics_sec += t_elapsed
            total_frames += 1

            if res is not None:
                shot_result = res
                break

        pred_scored = bool(shot_result.scored) if shot_result is not None else False
        pred_bank = bool(shot_result.is_bank_shot) if shot_result is not None else False
        cushions_2nd = shot_result.cushions_before_second_ball if shot_result is not None else -1
        fail_reason = shot_result.fail_reason if shot_result is not None else "NO_RESULT"

        bank_ok = (pred_bank == sc.expected_bank_shot) if sc.expected_scored else True
        is_correct = (pred_scored == sc.expected_scored) and bank_ok

        if is_correct:
            if sc.expected_scored:
                tp += 1
            else:
                tn += 1
        else:
            if sc.expected_scored and not pred_scored:
                fn += 1
            elif not sc.expected_scored and pred_scored:
                fp += 1
            else:
                fn += 1  # bank_shot mismatch on positive case
            failures.append(
                f"  - [{sc.scenario_id}] ({sc.category}) {sc.description}\n"
                f"      Expected: scored={sc.expected_scored}, bank={sc.expected_bank_shot} | "
                f"Got: scored={pred_scored}, bank={pred_bank}, cushions_at_2nd={cushions_2nd}, reason={fail_reason}"
            )

    total_cases = len(scenarios)
    accuracy = ((tp + tn) / float(total_cases)) * 100.0
    fpr = (fp / float(max(1, fp + tn))) * 100.0
    fnr = (fn / float(max(1, fn + tp))) * 100.0
    ms_per_frame = (total_kinematics_sec / float(max(1, total_frames))) * 1000.0

    print("=" * 78)
    print("3-CUSHION VISION ENGINE AUTORESEARCH BENCHMARK (30 EXTREME SCENARIOS)")
    print("=" * 78)
    print(f"Total Scenarios : {total_cases} (Positives: {tp + fn}, Negatives: {tn + fp})")
    print(f"Accuracy        : {accuracy:.2f}% ({tp + tn}/{total_cases})")
    print(f"False Pos Rate  : {fpr:.2f}% (FP={fp} / Neg={tn + fp})")
    print(f"False Neg Rate  : {fnr:.2f}% (FN={fn} / Pos={tp + fn})")
    print(f"Frame Latency   : {ms_per_frame:.4f} ms/frame (Target <= 0.5000 ms, Total Frames={total_frames})")
    print("-" * 78)
    if failures:
        print(f"FAILURE CASES ({len(failures)}):")
        for line in failures:
            print(line)
    else:
        print("ALL 30 EXTREME SCENARIOS PASSED (100.00% ACCURACY)!")
    print("=" * 78)

    return {
        "accuracy": accuracy,
        "fpr": fpr,
        "fnr": fnr,
        "ms_per_frame": ms_per_frame,
        "failures": failures,
    }


if __name__ == "__main__":
    run_benchmark()
