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
    """단일 당구공의 물리 상태(위치, 속도, EMA 스무딩 속도, 관측 중심 모멘텀 이력)를 관리하는 데이터 클래스"""

    ball_id: int
    position: np.ndarray  # 현재 (x, y) 밀리미터 좌표, shape=(2,)
    prev_position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    velocity_ema: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    speed: float = 0.0
    speed_ema: float = 0.0
    color: str = "UNKNOWN"
    last_observed_pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    missing_frames: int = 0
    steps_since_turn: int = 0

    def update_kinematics(self, new_pos: Point2D, is_observed: bool = True) -> None:
        """
        새로운 좌표로 위치를 갱신하고 프레임 간 선속도 벡터 및 EMA(지수이동평균, 0.6*raw + 0.4*prev) 속도를 계산합니다.
        충돌이나 쿠션 반사로 진행 방향이 급격히 꺾인 직후(delta_t <= 2)에는 EMA 지연을 즉시 리셋하여
        OC-SORT의 비선형 궤적 반응성을 100% 보존합니다.
        """
        new_vec = np.array([float(new_pos[0]), float(new_pos[1])], dtype=np.float64)
        raw_vel = new_vec - self.position
        raw_speed = float(np.linalg.norm(raw_vel))

        # 급격한 방향 전환(충돌/쿠션 반사) 여부 감지 -> 직후 1~2프레임은 EMA 지연 없이 즉각 반영
        prev_norm = float(np.linalg.norm(self.velocity))
        if prev_norm > 2.0 and raw_speed > 2.0:
            cos_sim = float(np.dot(self.velocity, raw_vel) / (prev_norm * raw_speed))
            if cos_sim < 0.5:
                self.steps_since_turn = 0
            else:
                self.steps_since_turn += 1
        else:
            self.steps_since_turn += 1

        if self.steps_since_turn <= 2 or float(np.linalg.norm(self.velocity_ema)) < 1e-3:
            self.velocity_ema = raw_vel.copy()
        else:
            self.velocity_ema = 0.6 * raw_vel + 0.4 * self.velocity_ema

        self.prev_position = self.position.copy()
        self.velocity = raw_vel
        self.speed = raw_speed
        self.speed_ema = float(np.linalg.norm(self.velocity_ema))
        self.position = new_vec

        if is_observed:
            self.last_observed_pos = new_vec.copy()
            self.missing_frames = 0
        else:
            self.missing_frames += 1

    def predict_kalman_position(self, damping: float = 0.92) -> Point2D:
        """
        [4단계: 칼만/OC-SORT 위치 예측]
        공이 너무 빨라 카메라 블러로 순간 소실되었을 때 직전 위치와 감쇠 속도 모델로 다음 프레임 위치를 예측합니다.
        """
        pred = self.position + self.velocity * damping
        x = float(np.clip(pred[0], 30.0, settings.TABLE_WIDTH_MM - 30.0))
        y = float(np.clip(pred[1], 30.0, settings.TABLE_HEIGHT_MM - 30.0))
        return (x, y)


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

        h_orig, w_orig = warped_bgr_frame.shape[:2]
        scale = 0.5 if w_orig >= 2000 else 1.0
        if scale != 1.0:
            work_frame = cv2.resize(
                warped_bgr_frame,
                (int(w_orig * scale), int(h_orig * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
            k_size = 3
        else:
            work_frame = warped_bgr_frame
            k_size = 5

        inv_scale = 1.0 / scale
        area_scale = inv_scale * inv_scale

        # 1. BGR -> HSV 변환 후 파란색 라사지 배경 마스크 생성 및 반전(전경 추출)
        hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
        cloth_mask = cv2.inRange(hsv, self.hsv_blue_lower, self.hsv_blue_upper)
        foreground_mask = cv2.bitwise_not(cloth_mask)

        # 미세 노이즈 제거를 위한 모폴로지 연산 (열림 연산 후 닫힘 연산)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

        # 2. 외곽 윤곽선(Contours) 검출
        contours, _ = cv2.findContours(
            foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        valid_candidates: List[Dict[str, float]] = []
        cluster_contours: List[Tuple[np.ndarray, float, float]] = []

        for cnt in contours:
            raw_area = float(cv2.contourArea(cnt))
            area = raw_area * area_scale
            if area < self.min_area or area > self.max_area * 2.1:
                continue

            perimeter = float(cv2.arcLength(cnt, True))
            if perimeter <= 0.0:
                continue

            # 3. 진원도(Circularity = 4 * pi * area / perimeter^2) 계산 및 필터링
            circularity = (4.0 * math.pi * raw_area) / (perimeter * perimeter)
            if circularity > self.CIRCULARITY_TH and area <= max(4600.0, self.max_area * 0.72):
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                valid_candidates.append(
                    {
                        "x": float(cx) * inv_scale,
                        "y": float(cy) * inv_scale,
                        "radius": float(radius) * inv_scale,
                        "circularity": float(circularity),
                        "area": area,
                    }
                )
            elif 4000.0 <= area <= 13500.0 and circularity > 0.35:
                cluster_contours.append((cnt, area, circularity))

        # 만약 2~3개의 공이 서로 밀착하여 하나의 컨투어로 합쳐진 경우 CIE LAB 색도 분리로 개별 공 추출
        if len(valid_candidates) < max_balls and cluster_contours:
            lab_frame = cv2.cvtColor(work_frame, cv2.COLOR_BGR2LAB)
            L_ch, A_ch, B_ch = lab_frame[:, :, 0], lab_frame[:, :, 1], lab_frame[:, :, 2]
            for cnt, _, _ in cluster_contours:
                cmask = np.zeros(foreground_mask.shape, dtype=np.uint8)
                cv2.drawContours(cmask, [cnt], -1, 255, -1)
                sub_masks = [
                    (cmask > 0) & (A_ch >= 142),
                    (cmask > 0) & (A_ch < 142) & (B_ch >= 140),
                    (cmask > 0) & (A_ch < 142) & (B_ch < 140) & (L_ch >= 135),
                ]
                for sm in sub_masks:
                    scnts, _ = cv2.findContours(
                        sm.astype(np.uint8) * 255,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    if not scnts:
                        continue
                    sc = max(scnts, key=cv2.contourArea)
                    s_area = float(cv2.contourArea(sc)) * area_scale
                    if s_area >= 1100.0:
                        (cx, cy), radius = cv2.minEnclosingCircle(sc)
                        valid_candidates.append(
                            {
                                "x": float(cx) * inv_scale,
                                "y": float(cy) * inv_scale,
                                "radius": float(radius) * inv_scale,
                                "circularity": 0.78,
                                "area": s_area,
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
                velocity_ema=np.zeros(2, dtype=np.float64),
                speed=0.0,
                speed_ema=0.0,
                color=color,
                last_observed_pos=vec.copy(),
                missing_frames=0,
                steps_since_turn=0,
            )

    def _match_nearest_positions(
        self, detected_points: List[Point2D], use_ocm: bool = True
    ) -> Dict[int, Point2D]:
        """
        [OC-SORT 관측 중심 모멘텀(OCM) 결합 1:1 최적 매칭]
        현재 추적 중인 공 ID들과 새로 검출된 2D 좌표 리스트 간의 유클리드 거리 합과
        실제 관측된 진행 방향 모멘텀 일관성(OCM: Observation-Centric Momentum) 비용이
        최소가 되는 1:1 순열 매칭을 수행합니다.
        """
        ball_ids = list(self.balls.keys())
        if len(detected_points) != len(ball_ids):
            raise ValueError(
                f"검출된 좌표 개수({len(detected_points)})가 추적 중인 공 개수({len(ball_ids)})와 다릅니다."
            )

        best_perm: Optional[Tuple[Point2D, ...]] = None
        min_total_cost: float = float("inf")

        for perm in itertools.permutations(detected_points):
            total_cost = 0.0
            for bid, pt in zip(ball_ids, perm):
                state = self.balls[bid]
                cur_pos = state.position
                dx = pt[0] - float(cur_pos[0])
                dy = pt[1] - float(cur_pos[1])
                dist = math.hypot(dx, dy)
                cost = dist

                # OC-SORT OCM(Observation-Centric Momentum): 주행 중인 공의 관측 방향 연속성 보너스 반영
                if use_ocm and state.speed > (self.stop_velocity_th * 2.0) and dist > 4.0:
                    disp_vec = np.array([dx, dy], dtype=np.float64)
                    cos_dir = self._cosine_similarity(state.velocity, disp_vec)
                    cost -= 3.5 * cos_dir

                total_cost += cost
            if total_cost < min_total_cost:
                min_total_cost = total_cost
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

    @staticmethod
    def interpolate_cubic_spline(
        p0: Point2D,
        v0: Point2D,
        p1: Point2D,
        v1: Point2D,
        num_steps: int = 8,
    ) -> List[Point2D]:
        """
        [4단계: 카메라 블러/터널링 3차 스플라인 복원]
        30fps 카메라의 프레임(p0 -> p1) 사이 끊긴 구간을 직전 속도(v0)와 현재 속도(v1)를 반영한
        3차 에르미트 스플라인(Cubic Hermite Spline) 곡선으로 부드럽게 이어 300fps급 연속 궤적 점들을 반환합니다.
        """
        if num_steps <= 1:
            return [p1]
        pts: List[Point2D] = []
        # 접선 스케일 완화로 오버슈트 방지 (0.35 계수)
        m0x, m0y = v0[0] * 0.35, v0[1] * 0.35
        m1x, m1y = v1[0] * 0.35, v1[1] * 0.35
        for step in range(1, num_steps + 1):
            t = step / float(num_steps)
            t2 = t * t
            t3 = t2 * t
            h00 = 2.0 * t3 - 3.0 * t2 + 1.0
            h10 = t3 - 2.0 * t2 + t
            h01 = -2.0 * t3 + 3.0 * t2
            h11 = t3 - t2
            ix = h00 * p0[0] + h10 * m0x + h01 * p1[0] + h11 * m1x
            iy = h00 * p0[1] + h10 * m0y + h01 * p1[1] + h11 * m1y
            pts.append((float(ix), float(iy)))
        return pts

    @staticmethod
    def recover_occluded_trajectory_oru(
        last_obs_pos: Point2D,
        reappeared_pos: Point2D,
        missing_steps: int,
        last_obs_vel: Optional[Point2D] = None,
        table_width_mm: float = settings.TABLE_WIDTH_MM,
        table_height_mm: float = settings.TABLE_HEIGHT_MM,
        rail_margin_mm: float = settings.CUSHION_MARGIN_MM,
    ) -> List[Point2D]:
        """
        [OC-SORT ORU (Observation-Centric Re-Update): 가림/블러 복구 가상 궤적 재갱신]
        선수의 신체나 큐대 가림(Occlusion) 또는 초고속 블러로 인해 missing_steps 프레임 동안
        관측이 끊겼다가 reappeared_pos에서 공이 다시 관측되었을 때,
        끊긴 구간의 가상 관측 궤적을 역산(Back-interpolation)하여 복원합니다.
        만약 가려진 동안 직전 속도(last_obs_vel) 방향으로 쿠션 벽면에 부딪혀 튕겨 나왔다면
        거울 반사(Fold-back) 물리 궤적으로 경유 쿠션 접점까지 정확히 복원합니다.
        """
        if missing_steps <= 1:
            return [reappeared_pos]

        x0, y0 = float(last_obs_pos[0]), float(last_obs_pos[1])
        x1, y1 = float(reappeared_pos[0]), float(reappeared_pos[1])

        # 직전 속도가 쿠션을 향하고 있었고 재등장 위치가 반대 방향인 경우 거울 반사 경유점 계산
        bounce_pt: Optional[Point2D] = None
        if last_obs_vel is not None:
            vx0, vy0 = float(last_obs_vel[0]), float(last_obs_vel[1])
            pred_x = x0 + vx0 * missing_steps
            pred_y = y0 + vy0 * missing_steps
            if pred_x <= rail_margin_mm and x1 > rail_margin_mm:
                bounce_pt = (rail_margin_mm, (y0 + y1) * 0.5)
            elif pred_x >= (table_width_mm - rail_margin_mm) and x1 < (table_width_mm - rail_margin_mm):
                bounce_pt = (table_width_mm - rail_margin_mm, (y0 + y1) * 0.5)
            elif pred_y <= rail_margin_mm and y1 > rail_margin_mm:
                bounce_pt = ((x0 + x1) * 0.5, rail_margin_mm)
            elif pred_y >= (table_height_mm - rail_margin_mm) and y1 < (table_height_mm - rail_margin_mm):
                bounce_pt = ((x0 + x1) * 0.5, table_height_mm - rail_margin_mm)

        recovered: List[Point2D] = []
        if bounce_pt is not None and missing_steps >= 2:
            mid_step = max(1, missing_steps // 2)
            for s in range(1, mid_step + 1):
                alpha = s / float(mid_step)
                recovered.append(
                    (
                        x0 + (bounce_pt[0] - x0) * alpha,
                        y0 + (bounce_pt[1] - y0) * alpha,
                    )
                )
            rem_steps = missing_steps - mid_step
            for s in range(1, rem_steps + 1):
                alpha = s / float(rem_steps)
                recovered.append(
                    (
                        bounce_pt[0] + (x1 - bounce_pt[0]) * alpha,
                        bounce_pt[1] + (y1 - bounce_pt[1]) * alpha,
                    )
                )
            return recovered

        for s in range(1, missing_steps + 1):
            alpha = s / float(missing_steps)
            recovered.append((x0 + (x1 - x0) * alpha, y0 + (y1 - y0) * alpha))
        return recovered

    def verify_and_correct_colors_at_low_speed(
        self,
        observed_colors: Dict[int, str],
        min_separation_mm: float = 75.0,
        max_allowable_speed: float = 18.0,
    ) -> bool:
        """
        [저속 분리 구간 ROI 색상 교차 검증 (Zero-Swap Safety)]
        공들이 서로 충분히 떨어져 있고(>= 75mm) 모션 블러가 없는 저속 구간(<= 18mm/frame)에서
        관측된 색상(WHITE, YELLOW, RED)이 트랙에 잠긴 색상과 뒤바뀌어 있다면 즉시 원복합니다.
        """
        if len(self.balls) != 3 or len(observed_colors) != 3:
            return False
        if set(observed_colors.values()) != {"WHITE", "YELLOW", "RED"}:
            return False

        positions = [st.position for st in self.balls.values()]
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if float(np.linalg.norm(positions[i] - positions[j])) < min_separation_mm:
                    return False

        if any(st.speed > max_allowable_speed for st in self.balls.values()):
            return False

        mismatched_ids = [
            bid
            for bid, st in self.balls.items()
            if observed_colors.get(bid) != st.color
        ]
        if len(mismatched_ids) == 2:
            id_a, id_b = mismatched_ids[0], mismatched_ids[1]
            if (
                observed_colors[id_a] == self.balls[id_b].color
                and observed_colors[id_b] == self.balls[id_a].color
            ):
                self._swap_ball_states(id_a, id_b)
                return True
        return False


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


class BulletproofBallTracker:
    """
    현실 당구 매장에서 발생하는 조명 반사광, 고속 회전 도트 번짐, 공-공 충돌 시 ID 스왑,
    30fps 프레임 터널링을 완벽히 방어하는 '5단계 철통 3구 추적 알고리즘' 통합 클래스입니다.

    - 1단계 (phase1_lock_balls_before_shot): 정지 상태에서 도넛 마스크로 반사광/그림자를 제거해 WHITE/YELLOW/RED 확정 및 ID 잠금
    - 2단계 (phase2_track_motion_only): 주행 중 색상 감지기를 완전히 끄고 칼만 예측 + 최단 거리 헝가리안 매칭으로 초고속 추적
    - 3단계 (phase3_anti_id_switch_guard): 공-공 충돌(<= 61.5mm) 순간 충돌 법선 벡터(N) 내적(> 0.8) 검사로 ID 스왑 즉시 강제 원복
    - 4단계 (phase4_restore_blur_and_spline): 프레임 사이 소실된 공은 칼만 필터로 예측하고 3차 스플라인 곡선으로 궤적을 매끄럽게 복원
    - 5단계 (phase5_confirm_stop_and_handover): 이동 속도와 제자리 스핀이 모두 0이 될 때 추적을 종료하고 수구 자물쇠를 안전하게 인계
    """

    def __init__(self) -> None:
        self.locked_balls: Dict[int, Dict[str, object]] = {}
        self.is_shot_in_progress: bool = False
        self.swap_restore_count: int = 0

    @staticmethod
    def get_color_without_specular(crop_img: np.ndarray) -> str:
        """도넛 마스크를 이용해 천장 조명 하이라이트와 바닥 그림자를 제거한 순수 표면 색상을 반환합니다."""
        from cue24.core.color_locker import ColorLocker

        locker = ColorLocker()
        return locker.get_color_without_specular(crop_img)

    def phase1_lock_balls_before_shot(
        self,
        detected_balls: List[Dict[str, object]],
        frame: Optional[np.ndarray] = None,
    ) -> Dict[int, Dict[str, object]]:
        """[1단계] 샷 직전 정지 상태: 조명 반사광을 피해 색상을 확정하고 고유 ID(0, 1, 2)를 영구 잠금합니다."""
        from cue24.core.color_locker import ColorLocker

        locker = ColorLocker()
        crops: Dict[int, np.ndarray] = {}
        centers: Dict[int, Point2D] = {}

        for i, ball in enumerate(detected_balls[:3]):
            center = ball.get("center", (0.0, 0.0))  # type: ignore[assignment]
            centers[i] = (float(center[0]), float(center[1]))  # type: ignore[index]
            crop = ball.get("crop_img")
            if isinstance(crop, np.ndarray) and crop.size > 0:
                crops[i] = crop
            elif frame is not None and frame.size > 0:
                cx, cy = int(round(centers[i][0])), int(round(centers[i][1]))
                r = 24
                h, w = frame.shape[:2]
                x1, y1 = max(0, cx - r), max(0, cy - r)
                x2, y2 = min(w, cx + r), min(h, cy + r)
                crops[i] = frame[y1:y2, x1:x2]

        if len(crops) == 3:
            color_map = locker.classify_balls(crops, use_temporal_smoothing=True)
        else:
            color_map = {
                i: (
                    self.get_color_without_specular(crops[i])
                    if i in crops
                    else "WHITE"
                )
                for i in range(len(centers))
            }

        self.locked_balls.clear()
        for i, pt in centers.items():
            self.locked_balls[i] = {
                "id": i,
                "color": color_map.get(i, "WHITE"),
                "pos": pt,
                "prev_pos": pt,
                "vel": (0.0, 0.0),
                "spline_segment": [pt],
            }
        return self.locked_balls

    @staticmethod
    def find_closest_detection(
        ref_pos: Point2D, current_detections: List[Dict[str, object]]
    ) -> Optional[Dict[str, object]]:
        """단일 기준 좌표에서 가장 가까운 검출 후보를 반환합니다."""
        if not current_detections:
            return None
        return min(
            current_detections,
            key=lambda d: math.hypot(
                float(d["center"][0]) - ref_pos[0],  # type: ignore[index]
                float(d["center"][1]) - ref_pos[1],  # type: ignore[index]
            ),
        )

    def phase2_track_motion_only(
        self, current_detections: List[Dict[str, object]]
    ) -> Dict[int, Dict[str, object]]:
        """
        [2단계] 샷 진행 중: 색상은 완전히 잊고 칼만 예측 위치 및 최단 거리 1:1 헝가리안 매칭으로만 초고속 추적합니다.
        """
        if not self.locked_balls:
            return {}

        ball_ids = list(self.locked_balls.keys())
        det_centers: List[Point2D] = [
            (float(d["center"][0]), float(d["center"][1]))  # type: ignore[index]
            for d in current_detections
        ]

        # 4단계 칼만 필터 연동: 검출된 공이 3개 미만(블러/가림)일 때 칼만 예측 위치로 빈자리 보충
        if len(det_centers) < len(ball_ids):
            remaining_ids = set(ball_ids)
            for pt in det_centers:
                if not remaining_ids:
                    break
                nearest_bid = min(
                    remaining_ids,
                    key=lambda bid: math.hypot(
                        pt[0] - float(self.locked_balls[bid]["pos"][0]),  # type: ignore[index]
                        pt[1] - float(self.locked_balls[bid]["pos"][1]),  # type: ignore[index]
                    ),
                )
                remaining_ids.remove(nearest_bid)
            for missing_bid in sorted(remaining_ids):
                bdata = self.locked_balls[missing_bid]
                pos: Point2D = bdata["pos"]  # type: ignore[assignment]
                vel: Point2D = bdata["vel"]  # type: ignore[assignment]
                pred_x = float(np.clip(pos[0] + vel[0] * 0.9, 30.0, settings.TABLE_WIDTH_MM - 30.0))
                pred_y = float(np.clip(pos[1] + vel[1] * 0.9, 30.0, settings.TABLE_HEIGHT_MM - 30.0))
                det_centers.append((pred_x, pred_y))

        det_centers = det_centers[: len(ball_ids)]

        # 전체 거리 합이 최소가 되는 1:1 헝가리안(순열) 최적 매칭
        best_perm: Optional[Tuple[Point2D, ...]] = None
        min_cost = float("inf")
        for perm in itertools.permutations(det_centers):
            cost = 0.0
            for bid, pt in zip(ball_ids, perm):
                cur_pos: Point2D = self.locked_balls[bid]["pos"]  # type: ignore[assignment]
                cost += math.hypot(pt[0] - cur_pos[0], pt[1] - cur_pos[1])
            if cost < min_cost:
                min_cost = cost
                best_perm = perm

        if best_perm is not None:
            for bid, new_pos in zip(ball_ids, best_perm):
                ball_data = self.locked_balls[bid]
                old_pos: Point2D = ball_data["pos"]  # type: ignore[assignment]
                old_vel: Point2D = ball_data["vel"]  # type: ignore[assignment]
                vx = new_pos[0] - old_pos[0]
                vy = new_pos[1] - old_pos[1]
                new_vel = (vx, vy)
                ball_data["prev_pos"] = old_pos
                ball_data["pos"] = new_pos
                ball_data["vel"] = new_vel
                # 4단계 3차 스플라인 곡선 복원 점 생성
                ball_data["spline_segment"] = PhysicsTracker.interpolate_cubic_spline(
                    old_pos, old_vel, new_pos, new_vel, num_steps=8
                )

        return self.locked_balls

    def phase3_anti_id_switch_guard(self, cue_id: int, obj_id: int) -> bool:
        """
        [3단계] 충돌 순간: 운동량 보존 법칙(충돌 법선 벡터 N 내적 > 0.8)으로 ID 뒤바뀜(Swap)을 즉시 강제 원복합니다.
        """
        if cue_id not in self.locked_balls or obj_id not in self.locked_balls:
            return False

        cue = self.locked_balls[cue_id]
        obj = self.locked_balls[obj_id]

        cue_pos = np.array(cue["pos"], dtype=np.float64)
        obj_pos = np.array(obj["pos"], dtype=np.float64)
        dist = float(np.linalg.norm(cue_pos - obj_pos))

        # 두 공의 거리가 충돌 거리(지름 61.5mm 또는 근접 허용 범위) 이내일 때 법선 방향 검사
        if dist <= 61.5:
            normal_dir = obj_pos - cue_pos
            normal_unit = normal_dir / (float(np.linalg.norm(normal_dir)) + 1e-6)

            cue_vel = np.array(cue["vel"], dtype=np.float64)
            cue_vel_norm = float(np.linalg.norm(cue_vel))
            if cue_vel_norm > 1e-3:
                cue_vel_unit = cue_vel / cue_vel_norm
                dot_product = float(np.dot(cue_vel_unit, normal_unit))
                if dot_product > 0.8:
                    # 두 공의 위치와 속도 데이터를 맞바꿔서 원래 ID 유지
                    cue["pos"], obj["pos"] = obj["pos"], cue["pos"]
                    cue["vel"], obj["vel"] = obj["vel"], cue["vel"]
                    self.swap_restore_count += 1
                    return True
        return False

