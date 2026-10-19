"""
조명 반사광 및 라사지 그림자 제거 기반 당구공 색상 확정 및 수구 고정 모듈
파일 경로: cue24/core/color_locker.py

정지 상태의 당구공 3개 BGR 크롭 영상을 입력받아 CIE-L*a*b* 색공간으로 변환한 뒤,
상단 중심 35%(LED 반사광)와 외곽 테두리 15%(라사지 그림자)를 배제한 환형(도넛) 마스크로
공의 고유 색상("WHITE", "YELLOW", "RED")을 확정하고 공격자 수구(Cue Ball) ID를 고정합니다.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from cue24.config.settings import settings


class ColorLocker:
    """
    환형(도넛) 마스크와 CIE-L*a*b* 색공간 임계값을 이용해 당구공 3개의 색상을 분류하고,
    현재 턴 공격 플레이어의 수구(Cue Ball) ID를 고정(Lock)하는 클래스입니다.
    """

    # 환형(도넛) 마스크 비율 상수
    HIGHLIGHT_INNER_RATIO: float = 0.35  # 중심부 35% 제거 (천장 조명 하이라이트 반사광 영역)
    SHADOW_OUTER_MARGIN: float = 0.15    # 외곽 테두리 15% 제거 (바닥 라사지 번짐 및 그림자 영역)

    def __init__(
        self,
        red_a_th: int = settings.LAB_RED_A_TH,
        yellow_b_th: int = settings.LAB_YELLOW_B_TH,
        player_cue_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        ColorLocker 초기화

        Args:
            red_a_th: CIE-LAB a* 채널 적구(RED) 판별 임계값 (기본값: 146)
            yellow_b_th: CIE-LAB b* 채널 황구(YELLOW) 판별 임계값 (기본값: 142)
            player_cue_map: 플레이어별 지정 수구 색상 매핑 (기본값: {"P1": "WHITE", "P2": "YELLOW"})
        """
        self.red_a_th: int = red_a_th
        self.yellow_b_th: int = yellow_b_th
        self.player_cue_map: Dict[str, str] = player_cue_map or {
            "P1": "WHITE",
            "P2": "YELLOW",
        }

        # 현재 프레임/턴에서 확정된 공 ID별 색상 및 수구 고정 상태 저장소
        self.locked_ball_colors: Dict[int, str] = {}
        self.locked_cue_ball_id: Optional[int] = None
        self.active_player: Optional[str] = None
        # [1단계 강화] 정지 상태 다중 프레임 누적 투표(EMA) 히스토리 및 주행 중 색상 분석 OFF 자물쇠
        self._ab_ema_history: Dict[int, Tuple[float, float]] = {}
        self.is_color_locked_for_shot: bool = False

    def set_shot_lock(self, locked: bool) -> None:
        """
        [2단계 연동] 수구가 출발하면 색상 감지기를 완전히 OFF(locked=True)하고,
        모든 공이 완전히 정지하면 다시 ON(locked=False)으로 전환합니다.
        """
        self.is_color_locked_for_shot = locked
        if not locked:
            self._ab_ema_history.clear()

    def get_color_without_specular(self, bgr_crop: np.ndarray) -> str:
        """
        [1단계] 천장 조명 하이라이트(중심 35%)와 바닥 그림자(외곽 15%)를 도려낸
        도넛 마스크를 씌워 순수 표면 색상(WHITE, YELLOW, RED)을 판별합니다.
        """
        return self.classify_single_ball(bgr_crop)

    def create_annular_mask(self, height: int, width: int) -> np.ndarray:
        """
        크롭된 당구공 패치 크기에 맞춰 상단 중심 35%(조명 반사광)와
        바닥 외곽 테두리 15%(라사지 그림자)를 마스킹(제거)한 환형(도넛) 마스크를 생성합니다.

        Args:
            height: 크롭 영상의 세로 픽셀 크기
            width: 크롭 영상의 가로 픽셀 크기

        Returns:
            mask: 유효 분석 영역(중간 링: 반지름 35% ~ 85% 구간)만 255로 채워진 uint8 단일 채널 마스크
        """
        if height <= 0 or width <= 0:
            raise ValueError(f"유효하지 않은 크롭 영상 크기입니다: ({height}, {width})")

        center_x = width / 2.0
        center_y = height / 2.0
        max_radius = min(width, height) / 2.0

        # 외곽 15% 그림자 제거 -> 최대 반지름의 85% 까지만 포함
        outer_radius = max_radius * (1.0 - self.SHADOW_OUTER_MARGIN)
        # 중심 35% 조명 반사광 제거 -> 최대 반지름의 35% 안쪽은 제외
        inner_radius = max_radius * self.HIGHLIGHT_INNER_RATIO

        # 픽셀 그리드 거리 계산으로 정밀한 환형(도넛) 마스크 생성
        y_indices, x_indices = np.ogrid[:height, :width]
        dist_from_center = np.sqrt(
            (x_indices - center_x) ** 2 + (y_indices - center_y) ** 2
        )

        ring_condition = (dist_from_center >= inner_radius) & (
            dist_from_center <= outer_radius
        )
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[ring_condition] = 255

        # 만약 패치가 극단적으로 작아 링 픽셀이 비어있다면 전체 원 내부를 사용하도록 안전장치(Fallback) 적용
        if np.count_nonzero(mask) == 0:
            mask[dist_from_center <= max_radius] = 255

        return mask

    def extract_lab_ab_mean(self, bgr_crop: np.ndarray) -> Tuple[float, float]:
        """
        단일 공의 BGR 크롭 영상을 CIE-L*a*b*로 변환하고,
        환형(도넛) 마스크가 적용된 중간 링 영역의 (a*, b*) 채널 평균값을 계산합니다.

        Args:
            bgr_crop: 단일 당구공의 BGR 이미지 패치 (H, W, 3)

        Returns:
            (mean_a, mean_b): 마스크 영역 내의 a* 평균값, b* 평균값
        """
        if bgr_crop is None or bgr_crop.ndim != 3 or bgr_crop.shape[2] != 3:
            raise ValueError("입력 크롭 영상은 3채널 BGR 이미지(H, W, 3)여야 합니다.")

        h, w, _ = bgr_crop.shape
        lab_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB)
        mask = self.create_annular_mask(h, w)

        # OpenCV cv2.mean은 mask가 255인 영역의 채널별 평균 (L, a, b, 0)을 반환
        mean_l, mean_a, mean_b, _ = cv2.mean(lab_crop, mask=mask)
        return (float(mean_a), float(mean_b))

    def classify_single_ball(self, bgr_crop: np.ndarray) -> str:
        """
        단일 당구공 BGR 크롭 영상을 분석하여 "RED", "YELLOW", "WHITE" 중 하나로 분류합니다.
        """
        mean_a, mean_b = self.extract_lab_ab_mean(bgr_crop)

        # 1순위: a* 채널 값이 임계값(146) 이상이면 적구(RED)
        if mean_a >= self.red_a_th:
            return "RED"
        # 2순위: b* 채널 값이 임계값(142) 이상이면 황구(YELLOW)
        if mean_b >= self.yellow_b_th:
            return "YELLOW"
        # 3순위: 그 외는 백구(WHITE)
        return "WHITE"

    def classify_balls(
        self, ball_crops: Dict[int, np.ndarray], use_temporal_smoothing: bool = False
    ) -> Dict[int, str]:
        """
        정지 상태의 공 3개 BGR 크롭 영상 딕셔너리({ball_id: bgr_crop})를 입력받아,
        각 공의 색상을 "WHITE", "YELLOW", "RED"로 100% 확정하여 반환합니다.

        기본 임계값 판정을 수행하되, 만약 극단적인 색온도 변화로 인해 3개 공의 색상이 중복될 경우
        a*, b* 상대 크기 순위를 기반으로 3개 공을 각각 RED, YELLOW, WHITE로 상호 배타적 보정합니다.

        Args:
            ball_crops: {공 ID(int): BGR 크롭 영상(np.ndarray)} 형태의 딕셔너리 (총 3개 공)
            use_temporal_smoothing: 다중 프레임 지수 이동 평균(EMA) 누적 투표를 사용할지 여부

        Returns:
            classifications: {공 ID(int): 색상 문자열("WHITE" | "YELLOW" | "RED")}
        """
        if self.is_color_locked_for_shot and len(self.locked_ball_colors) == 3:
            return self.locked_ball_colors.copy()

        if len(ball_crops) != 3:
            raise ValueError(
                f"3쿠션 판독을 위해 정확히 3개의 공 크롭 영상이 필요합니다. (입력된 개수: {len(ball_crops)})"
            )

        ab_stats: Dict[int, Tuple[float, float]] = {}
        classifications: Dict[int, str] = {}

        for ball_id, crop in ball_crops.items():
            mean_a, mean_b = self.extract_lab_ab_mean(crop)
            if use_temporal_smoothing and ball_id in self._ab_ema_history:
                prev_a, prev_b = self._ab_ema_history[ball_id]
                mean_a = prev_a * 0.70 + mean_a * 0.30
                mean_b = prev_b * 0.70 + mean_b * 0.30
            if use_temporal_smoothing:
                self._ab_ema_history[ball_id] = (mean_a, mean_b)

            ab_stats[ball_id] = (mean_a, mean_b)
            if mean_a >= self.red_a_th:
                classifications[ball_id] = "RED"
            elif mean_b >= self.yellow_b_th:
                classifications[ball_id] = "YELLOW"
            else:
                classifications[ball_id] = "WHITE"

        # 3개 공이 모두 서로 다른 색상(RED, YELLOW, WHITE)으로 분류되었는지 검증
        expected_set = {"RED", "YELLOW", "WHITE"}
        if set(classifications.values()) != expected_set:
            # 조명 편차로 중복 발생 시: a*가 가장 높은 공을 RED, 남은 2개 중 b*가 높은 공을 YELLOW, 나머지를 WHITE로 확정
            sorted_by_a = sorted(
                ab_stats.keys(), key=lambda bid: ab_stats[bid][0], reverse=True
            )
            red_id = sorted_by_a[0]
            remaining_ids = sorted_by_a[1:]
            if ab_stats[remaining_ids[0]][1] >= ab_stats[remaining_ids[1]][1]:
                yellow_id, white_id = remaining_ids[0], remaining_ids[1]
            else:
                yellow_id, white_id = remaining_ids[1], remaining_ids[0]

            classifications = {
                red_id: "RED",
                yellow_id: "YELLOW",
                white_id: "WHITE",
            }

        self.locked_ball_colors = classifications.copy()
        return classifications

    def lock_cue_ball(
        self,
        ball_crops: Dict[int, np.ndarray],
        active_player: str = "P1",
    ) -> Dict[str, object]:
        """
        샷 시작 전 정지 상태의 공 3개 색상을 확정하고,
        현재 공격 플레이어(예: P1=WHITE, P2=YELLOW)의 공을 수구(Cue Ball)로 지정하여 ID를 고정(Lock)합니다.

        Args:
            ball_crops: {공 ID(int): BGR 크롭 영상(np.ndarray)} (3개 공)
            active_player: 현재 공격권을 가진 플레이어 식별자 ("P1" 또는 "P2" 또는 직접 색상명 "WHITE"/"YELLOW")

        Returns:
            lock_info: {
                "active_player": 현재 플레이어,
                "cue_ball_id": 고정된 수구 ID (int),
                "cue_ball_color": 고정된 수구 색상 ("WHITE" | "YELLOW"),
                "object_ball_ids": 제1·2목적구 ID 리스트 (List[int]),
                "ball_colors": 전체 공 ID별 색상 매핑 딕셔너리
            }
        """
        # 1. 공 3개 색상 확정
        ball_colors = self.classify_balls(ball_crops)

        # 2. 현재 공격 플레이어의 목표 수구 색상 결정
        if active_player in self.player_cue_map:
            target_cue_color = self.player_cue_map[active_player]
        elif active_player in ("WHITE", "YELLOW"):
            target_cue_color = active_player
        else:
            raise ValueError(
                f"지원하지 않는 플레이어 또는 수구 식별자입니다: {active_player} "
                f"(허용값: {list(self.player_cue_map.keys())} 또는 'WHITE', 'YELLOW')"
            )

        # 3. 확정된 색상 딕셔너리에서 수구(Cue Ball) ID와 목적구(Object Balls) ID 분리 및 고정(Lock)
        cue_ball_id: Optional[int] = None
        object_ball_ids: List[int] = []

        for ball_id, color_name in ball_colors.items():
            if color_name == target_cue_color:
                cue_ball_id = ball_id
            else:
                object_ball_ids.append(ball_id)

        if cue_ball_id is None:
            raise RuntimeError(
                f"수구 색상({target_cue_color})에 해당하는 공 ID를 찾을 수 없습니다: {ball_colors}"
            )

        self.locked_cue_ball_id = cue_ball_id
        self.active_player = active_player

        return {
            "active_player": active_player,
            "cue_ball_id": cue_ball_id,
            "cue_ball_color": target_cue_color,
            "object_ball_ids": sorted(object_ball_ids),
            "ball_colors": ball_colors,
        }
