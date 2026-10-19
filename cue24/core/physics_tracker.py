"""
대대 평면(2840x1420) 당구공 후보 검출, 물리 충돌 가드 위치 추적 및 스핀 정지 감지 엔진
파일 경로: cue24/core/physics_tracker.py

1. BallCandidateDetector: HSV 파란색 라사지 배경 차분 및 진원도(> 0.72) 필터 기반 공 3개 검출
2. PhysicsTracker: 최단 거리 고속 추적 및 법선 벡터(N) 코사인 유사도(>= 0.85) 기반 ID 스왑 강제 원복 가드
3. SpinDetector: 공 내부 도트 패턴 프레임 차분 에너지 및 선속도 기반 15프레임 완전 정지 판정
"""

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from cue24.config.settings import settings

Point2D = Tuple[float, float]


@dataclass
class BallState:
    """단일 당구공의 물리 상태(위치, 속도, 이전 위치, 접촉 이력)를 관리하는 데이터 클래스"""

    ball_id: int
    position: np.ndarray  # 현재 (x, y) 밀리미터 좌표, shape=(2,)
    prev_position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    speed: float = 0.0
    color: str = "UNKNOWN"

    def update_kinematics(self, new_pos: Point2D) -> None:
        """새로운 좌표로 위치를 갱신하고 프레임 간 선속도 벡터 및 속력을 계산합니다."""
        new_vec = np.array([float(new_pos[0]), float(new_pos[1])], dtype=np.float64)
        self.prev_position = self.position.copy()
        self.velocity = new_vec - self.prev_position
        self.speed = float(np.linalg.norm(self.velocity))
        self.position = new_vec


class BallCandidateDetector:
    """
    2840 x 1420 정규화 평면 영상에서 파란색 라사지 배경을 차분(HSV 색공간)하고,
    진원도(Circularity = 4*pi*area / perimeter^2 > 0.72) 및 면적 필터를 적용하여
    선수의 팔이나 큐대 그림자를 완벽히 배제하고 당구공 3개의 중심 좌표와 반지름을 추출합니다.
    """

    CIRCULARITY_TH: float = 0.72

    def __init__(
        self,
        ball_radius_mm: float = settings.BALL_RADIUS_MM,
        hsv_blue_lower: Tuple[int, int, int] = (90, 50, 40),
        hsv_blue_upper: Tuple[int, int, int] = (135, 255, 255),
        min_area_ratio: float = 0.40,
        max_area_ratio: float = 2.20,
    ) -> None:
        """
        BallCandidateDetector 초기화

        Args:
            ball_radius_mm: 표준 당구공 반지름 (30.75mm, 1px = 1mm 기준)
            hsv_blue_lower: 파란색 당구대 라사지 HSV 하한값
            hsv_blue_upper: 파란색 당구대 라사지 HSV 상한값
            min_area_ratio: 이론적 당구공 단면적 대비 최소 허용 면적 비율
            max_area_ratio: 이론적 당구공 단면적 대비 최대 허용 면적 비율
        """
        self.ball_radius_mm: float = ball_radius_mm
        self.hsv_blue_lower: np.ndarray = np.array(hsv_blue_lower, dtype=np.uint8)
        self.hsv_blue_upper: np.ndarray = np.array(hsv_blue_upper, dtype=np.uint8)

        theoretical_area = math.pi * (self.ball_radius_mm ** 2)  # 약 2970.57 mm^2
        self.min_area: float = theoretical_area * min_area_ratio
        self.max_area: float = theoretical_area * max_area_ratio
        self.ideal_area: float = theoretical_area

    def detect(
        self, warped_bgr_frame: np.ndarray, max_balls: int = 3
    ) -> List[Dict[str, float]]:
        """
        평면 보정된 BGR 영상에서 당구공 후보를 검출하여 최대 3개의 공 정보를 반환합니다.

        Args:
            warped_bgr_frame: (1420, 2840, 3) 크기의 정규화된 BGR 당구대 영상
            max_balls: 반환할 최대 공 개수 (기본값: 3)

        Returns:
            candidates: [{"x": float, "y": float, "radius": float, "circularity": float, "area": float}, ...]
        """
        if warped_bgr_frame is None or warped_bgr_frame.size == 0:
            raise ValueError("유효하지 않은 빈 영상이 입력되었습니다.")

        # 1. BGR -> HSV 변환 후 파란색 라사지 배경 마스크 생성 및 반전(전경 추출)
        hsv = cv2.cvtColor(warped_bgr_frame, cv2.COLOR_BGR2HSV)
        cloth_mask = cv2.inRange(hsv, self.hsv_blue_lower, self.hsv_blue_upper)
        foreground_mask = cv2.bitwise_not(cloth_mask)

        # 미세 노이즈 제거를 위한 모폴로지 연산 (열림 연산 후 닫힘 연산)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

        # 2. 외곽 윤곽선(Contours) 검출
        contours, _ = cv2.findContours(
            foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        valid_candidates: List[Dict[str, float]] = []

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.min_area or area > self.max_area:
                continue

            perimeter = float(cv2.arcLength(cnt, True))
            if perimeter <= 0.0:
                continue

            # 3. 진원도(Circularity = 4 * pi * area / perimeter^2) 계산 및 필터링
            circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
            if circularity <= self.CIRCULARITY_TH:
                # 큐대, 선수 팔, 긴 그림자 등은 진원도가 0.72 이하이므로 완벽히 배제됨
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            valid_candidates.append(
                {
                    "x": float(cx),
                    "y": float(cy),
                    "radius": float(radius),
                    "circularity": float(circularity),
                    "area": area,
                }
            )

        # 진원도가 높고 이상적인 당구공 면적에 가까운 순서로 정렬하여 상위 3개 반환
        valid_candidates.sort(
            key=lambda c: (
                -c["circularity"],
                abs(c["area"] - self.ideal_area),
            )
        )
        return valid_candidates[:max_balls]


class PhysicsTracker:
    """
    샷 주행 중 색상을 무시하고 최단 거리 매칭으로 공 3개를 초고속 위치 추적하며,
    충돌 순간 두 공의 중심 거리(<= 61.5mm)와 충돌 법선 벡터(Normal Vector N)를 계산해
    수구와 적구의 ID가 뒤바뀌는 현상(ID Swap)을 감지 및 강제 원복하는 물리 충돌 가드 추적기입니다.
    """

    COSINE_SIM_SWAP_TH: float = 0.85

    def __init__(
        self,
        ball_diameter_mm: float = settings.BALL_DIAMETER_MM,
        stop_velocity_th: float = settings.STOP_VELOCITY_TH,
        collision_margin_mm: float = 4.0,
    ) -> None:
        """
        PhysicsTracker 초기화

        Args:
            ball_diameter_mm: 당구공 직경 (기본값: 61.5mm)
            stop_velocity_th: 정지 상태 판별 선속도 임계값 (기본값: 2.5mm/frame)
            collision_margin_mm: 프레임 샘플링 간격을 고려한 근접 충돌 감지 여유 마진 (mm)
        """
        self.ball_diameter_mm: float = ball_diameter_mm
        self.collision_distance_th: float = ball_diameter_mm + collision_margin_mm
        self.stop_velocity_th: float = stop_velocity_th

        self.balls: Dict[int, BallState] = {}
        self.cue_ball_id: Optional[int] = None

        # 직전 프레임에서 충돌(근접 접촉)이 감지된 (수구ID, 목적구ID) 쌍과 그때의 충돌 법선 벡터 N 저장
        self._pending_collision_guards: Dict[int, Dict[str, object]] = {}
        self.swap_correction_count: int = 0

    def initialize_tracks(
        self,
        initial_positions: Dict[int, Point2D],
        cue_ball_id: int,
        ball_colors: Optional[Dict[int, str]] = None,
    ) -> None:
        """
        샷 시작 전 정지 상태의 공 3개 초기 좌표와 고정된 수구(Cue Ball) ID를 등록합니다.

        Args:
            initial_positions: {ball_id: (x_mm, y_mm)} 형태의 초기 위치 딕셔너리
            cue_ball_id: ColorLocker에서 확정된 현재 공격자의 수구 ID
            ball_colors: 선택적 공 색상 매핑 딕셔너리
        """
        self.balls.clear()
        self._pending_collision_guards.clear()
        self.cue_ball_id = cue_ball_id

        for bid, pos in initial_positions.items():
            vec = np.array([float(pos[0]), float(pos[1])], dtype=np.float64)
            color = ball_colors.get(bid, "UNKNOWN") if ball_colors else "UNKNOWN"
            self.balls[bid] = BallState(
                ball_id=bid,
                position=vec.copy(),
                prev_position=vec.copy(),
                velocity=np.zeros(2, dtype=np.float64),
                speed=0.0,
                color=color,
            )

    def _match_nearest_positions(
        self, detected_points: List[Point2D]
    ) -> Dict[int, Point2D]:
        """
        현재 추적 중인 공 ID들과 새로 검출된 2D 좌표 리스트 간의 전체 거리 합이 최소가 되는
        1:1 최단 거리 매칭(Exact Permutations Assignment for 3 balls)을 수행합니다.
        """
        ball_ids = list(self.balls.keys())
        if len(detected_points) != len(ball_ids):
            raise ValueError(
                f"검출된 좌표 개수({len(detected_points)})가 추적 중인 공 개수({len(ball_ids)})와 다릅니다."
            )

        best_perm: Optional[Tuple[Point2D, ...]] = None
        min_total_dist: float = float("inf")

        for perm in itertools.permutations(detected_points):
            total_dist = 0.0
            for bid, pt in zip(ball_ids, perm):
                cur_pos = self.balls[bid].position
                dist = math.hypot(pt[0] - cur_pos[0], pt[1] - cur_pos[1])
                total_dist += dist
            if total_dist < min_total_dist:
                min_total_dist = total_dist
                best_perm = perm

        assert best_perm is not None
        return {bid: pt for bid, pt in zip(ball_ids, best_perm)}

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """두 2D 벡터 간의 코사인 유사도(-1.0 ~ 1.0)를 계산합니다."""
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def _swap_ball_states(self, id_a: int, id_b: int) -> None:
        """
        ID 스왑이 감지되었을 때 두 공(수구와 적구)의 위치, 이전 위치, 속도 벡터를 강제 원복 교환합니다.
        """
        state_a = self.balls[id_a]
        state_b = self.balls[id_b]

        state_a.position, state_b.position = (
            state_b.position.copy(),
            state_a.position.copy(),
        )
        state_a.prev_position, state_b.prev_position = (
            state_b.prev_position.copy(),
            state_a.prev_position.copy(),
        )
        state_a.velocity, state_b.velocity = (
            state_b.velocity.copy(),
            state_a.velocity.copy(),
        )
        state_a.speed, state_b.speed = state_b.speed, state_a.speed
        self.swap_correction_count += 1

    def update(self, detected_points: List[Point2D]) -> Dict[int, BallState]:
        """
        매 프레임 검출된 공 3개의 좌표를 입력받아 최단 거리 매칭으로 상태를 갱신하고,
        [물리 충돌 가드]를 작동시켜 ID 스왑 발생 시 수구와 적구의 ID/위치/속도를 강제 원복합니다.

        Args:
            detected_points: 현재 프레임에서 검출된 3개 공의 [(x1, y1), (x2, y2), (x3, y3)] 좌표 리스트

        Returns:
            balls: 갱신 및 물리 가드 보정이 완료된 {ball_id: BallState} 딕셔너리
        """
        if self.cue_ball_id is None or self.cue_ball_id not in self.balls:
            raise RuntimeError("PhysicsTracker가 초기화되지 않았거나 cue_ball_id가 설정되지 않았습니다.")

        # 0. 매칭 전 각 공의 직전 정지 여부 백업 (충돌 직전 목적구가 정지해 있었는지 확인용)
        was_stationary_before_step: Dict[int, bool] = {
            bid: (b.speed <= self.stop_velocity_th) for bid, b in self.balls.items()
        }

        # 1. 색상 무시 최단 거리 위치 매칭 및 속도 업데이트
        matched_coords = self._match_nearest_positions(detected_points)
        for bid, pt in matched_coords.items():
            self.balls[bid].update_kinematics(pt)

        cue_state = self.balls[self.cue_ball_id]

        # 2. [물리 충돌 가드 - 단계 A]: 직전 프레임에서 등록된 충돌 후보에 대해 ID 스왑 여부 판정 및 원복
        # 정지해 있던 목적구(target)는 충돌 후 반드시 충돌 법선 벡터 N 방향으로 튕겨 나가야 함.
        # 만약 최단 거리 매칭 결과, 수구(cue)가 법선 벡터 N 방향(코사인 유사도 >= 0.85)으로 날아가고 있고
        # 정작 목적구(target)가 N 방향이 아니거나 멈춰 있다면 트래커가 두 공의 ID를 스왑한 것임!
        resolved_targets: List[int] = []
        for target_id, guard_info in list(self._pending_collision_guards.items()):
            normal_vec: np.ndarray = guard_info["normal_vec"]  # type: ignore[assignment]
            target_state = self.balls[target_id]

            cue_cos = self._cosine_similarity(cue_state.velocity, normal_vec)
            target_cos = self._cosine_similarity(target_state.velocity, normal_vec)

            # 수구의 속도 벡터가 법선 N과 코사인 유사도 0.85 이상이고,
            # 정지해 있던 목적구가 여전히 멈춰 있거나 법선 방향 속도가 수구보다 현저히 낮을 때 ID 스왑으로 판정
            if (
                cue_state.speed > self.stop_velocity_th
                and cue_cos >= self.COSINE_SIM_SWAP_TH
                and (
                    target_state.speed <= self.stop_velocity_th
                    or cue_cos > target_cos
                    or (cue_state.speed > target_state.speed and target_cos >= self.COSINE_SIM_SWAP_TH)
                )
            ):
                # 충돌 직전 각자의 고유 위치(pre_pos)를 기준으로 속도와 위치를 복원
                pre_cue_pos: np.ndarray = guard_info["pre_cue_pos"]  # type: ignore[assignment]
                pre_target_pos: np.ndarray = guard_info["pre_target_pos"]  # type: ignore[assignment]

                true_cue_new_pos = target_state.position.copy()
                true_target_new_pos = cue_state.position.copy()

                cue_state.prev_position = pre_cue_pos.copy()
                cue_state.position = true_cue_new_pos
                cue_state.velocity = cue_state.position - cue_state.prev_position
                cue_state.speed = float(np.linalg.norm(cue_state.velocity))

                target_state.prev_position = pre_target_pos.copy()
                target_state.position = true_target_new_pos
                target_state.velocity = target_state.position - target_state.prev_position
                target_state.speed = float(np.linalg.norm(target_state.velocity))

                self.swap_correction_count += 1

            resolved_targets.append(target_id)

        for tid in resolved_targets:
            self._pending_collision_guards.pop(tid, None)

        # 3. [물리 충돌 가드 - 단계 B]: 현재 프레임에서 수구와 목적구 간 거리가 61.5mm 이하(또는 근접 임계값)인지 검사
        for bid, other_state in self.balls.items():
            if bid == self.cue_ball_id:
                continue

            diff_vec = other_state.position - cue_state.position
            dist = float(np.linalg.norm(diff_vec))

            # 두 공의 거리가 61.5mm 이하(또는 샘플링 접촉 허용 범위 내)로 근접한 순간 충돌 법선 벡터 N 계산
            if dist <= self.collision_distance_th and dist > 1e-6:
                normal_vec = diff_vec / dist  # 수구 중심 -> 목적구 중심을 향하는 단위 법선 벡터 N

                if was_stationary_before_step.get(bid, True):
                    # 다음 프레임(충돌 직후 튕겨 나가는 프레임)에서 궤적을 검증할 수 있도록 충돌 법선 벡터 N을 가드에 등록
                    self._pending_collision_guards[bid] = {
                        "normal_vec": normal_vec,
                        "target_was_stationary": True,
                        "pre_cue_pos": cue_state.position.copy(),
                        "pre_target_pos": other_state.position.copy(),
                    }

        return self.balls


class SpinDetector:
    """
    각 당구공 내부 도트(점) 패턴 ROI의 프레임 간 픽셀 명암 차분 평균 에너지를 측정하여,
    병진 속도 < STOP_VELOCITY_TH (2.5mm/frame) 및 스핀 에너지 < STOP_SPIN_ENERGY_TH (2.0) 상태가
    연속 STOP_CONFIRM_FRAMES (15프레임, 약 0.5초) 이상 유지될 때 완전 정지(is_fully_stopped = True)로 판정합니다.
    """

    def __init__(
        self,
        stop_velocity_th: float = settings.STOP_VELOCITY_TH,
        stop_spin_energy_th: float = settings.STOP_SPIN_ENERGY_TH,
        stop_confirm_frames: int = settings.STOP_CONFIRM_FRAMES,
        roi_radius_px: int = 20,
    ) -> None:
        """
        SpinDetector 초기화

        Args:
            stop_velocity_th: 선속도 정지 임계값 (기본값: 2.5 mm/frame)
            stop_spin_energy_th: 픽셀 차분 스핀 에너지 임계값 (기본값: 2.0)
            stop_confirm_frames: 완전 정지 확정 연속 프레임 수 (기본값: 15 프레임)
            roi_radius_px: 공 중심에서 스핀 에너지를 추출할 내부 원형 ROI 반지름 (픽셀)
        """
        self.stop_velocity_th: float = stop_velocity_th
        self.stop_spin_energy_th: float = stop_spin_energy_th
        self.stop_confirm_frames: int = stop_confirm_frames
        self.roi_radius_px: int = roi_radius_px

        # 공 ID별 직전 프레임의 내부 Grayscale ROI 패치 저장소
        self._prev_rois: Dict[int, np.ndarray] = {}
        # 모든 공이 정지 조건을 만족한 연속 프레임 카운터
        self.consecutive_stop_frames: int = 0
        self.is_fully_stopped: bool = False

    def reset(self) -> None:
        """새로운 샷이 시작될 때 스핀 감지기 상태를 초기화합니다."""
        self._prev_rois.clear()
        self.consecutive_stop_frames = 0
        self.is_fully_stopped = False

    def _extract_circular_gray_roi(
        self, frame: np.ndarray, center: Point2D
    ) -> Optional[np.ndarray]:
        """
        공 중심 좌표를 기준으로 일정한 크기의 정사각형 Grayscale 패치를 잘라내고
        내부 도트 패턴 영역을 반환합니다.
        """
        r = self.roi_radius_px
        cx, cy = int(round(center[0])), int(round(center[1]))
        h, w = frame.shape[:2]

        x1, y1 = cx - r, cy - r
        x2, y2 = cx + r, cy + r

        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            padded = cv2.copyMakeBorder(
                frame, r, r, r, r, cv2.BORDER_REPLICATE
            )
            crop = padded[y1 + r : y2 + r, x1 + r : x2 + r]
        else:
            crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        return gray

    def compute_spin_energy(self, ball_id: int, current_roi: np.ndarray) -> float:
        """
        단일 공의 직전 프레임 ROI와 현재 프레임 ROI 간의 원형 내부 픽셀 명암 절대 차분 평균(Spin Energy)을 계산합니다.
        """
        if ball_id not in self._prev_rois or self._prev_rois[ball_id].shape != current_roi.shape:
            self._prev_rois[ball_id] = current_roi.copy()
            return 0.0

        prev_roi = self._prev_rois[ball_id]
        diff = cv2.absdiff(current_roi, prev_roi)
        self._prev_rois[ball_id] = current_roi.copy()

        # 원형 내부 영역만 평균 에너지 계산
        h, w = diff.shape
        y_idx, x_idx = np.ogrid[:h, :w]
        dist_sq = (x_idx - w / 2.0) ** 2 + (y_idx - h / 2.0) ** 2
        circle_mask = dist_sq <= (min(h, w) / 2.0) ** 2

        if not np.any(circle_mask):
            return float(np.mean(diff))

        energy = float(np.mean(diff[circle_mask]))
        return energy

    def update(
        self,
        ball_states: Dict[int, BallState],
        warped_frame: Optional[np.ndarray] = None,
        ball_crops_gray: Optional[Dict[int, np.ndarray]] = None,
    ) -> Dict[str, object]:
        """
        매 프레임 모든 공의 선속도와 내부 도트 스핀 에너지를 평가하여 완전 정지 여부를 판정합니다.

        Args:
            ball_states: PhysicsTracker에서 갱신된 {ball_id: BallState} 딕셔너리
            warped_frame: 전체 평면 보정 영상 (제공 시 공 중심에서 ROI 자동 추출)
            ball_crops_gray: 공 ID별 개별 크롭 패치 딕셔너리 (단위 테스트 등에서 직접 패치 주입 시 사용)

        Returns:
            status: {
                "is_fully_stopped": bool,
                "consecutive_stop_frames": int,
                "max_velocity": float,
                "max_spin_energy": float,
                "spin_energies": Dict[int, float]
            }
        """
        spin_energies: Dict[int, float] = {}
        max_velocity: float = 0.0
        max_spin_energy: float = 0.0

        for bid, state in ball_states.items():
            if state.speed > max_velocity:
                max_velocity = state.speed

            roi: Optional[np.ndarray] = None
            if ball_crops_gray is not None and bid in ball_crops_gray:
                crop = ball_crops_gray[bid]
                roi = (
                    cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    if crop.ndim == 3
                    else crop.copy()
                )
            elif warped_frame is not None:
                roi = self._extract_circular_gray_roi(
                    warped_frame, (float(state.position[0]), float(state.position[1]))
                )

            if roi is not None:
                energy = self.compute_spin_energy(bid, roi)
            else:
                energy = 0.0

            spin_energies[bid] = energy
            if energy > max_spin_energy:
                max_spin_energy = energy

        # 모든 공의 병진 속도 < STOP_VELOCITY_TH 이고 스핀 에너지 < STOP_SPIN_ENERGY_TH 인지 확인
        if (
            max_velocity < self.stop_velocity_th
            and max_spin_energy < self.stop_spin_energy_th
        ):
            self.consecutive_stop_frames += 1
        else:
            self.consecutive_stop_frames = 0

        self.is_fully_stopped = (
            self.consecutive_stop_frames >= self.stop_confirm_frames
        )

        return {
            "is_fully_stopped": self.is_fully_stopped,
            "consecutive_stop_frames": self.consecutive_stop_frames,
            "max_velocity": max_velocity,
            "max_spin_energy": max_spin_energy,
            "spin_energies": spin_energies,
        }
