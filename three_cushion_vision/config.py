"""
국제식 대대 물리 규격(2840x1420mm, 공 반지름 30.75mm) 및 60fps 비전·물리 임계값 설정 모듈
파일 경로: three_cushion_vision/config.py
"""

from dataclasses import dataclass
from enum import Enum


class BallColor(str, Enum):
    """당구 3구 고유 색상 열거형"""

    WHITE = "WHITE"
    YELLOW = "YELLOW"
    RED = "RED"


class Rail(str, Enum):
    """당구대 4면 쿠션 레일 열거형"""

    LEFT = "LEFT"      # x = 0.0 mm
    RIGHT = "RIGHT"    # x = 2840.0 mm
    TOP = "TOP"        # y = 0.0 mm
    BOTTOM = "BOTTOM"  # y = 1420.0 mm


class ShotState(str, Enum):
    """쓰리쿠션 공식 규칙 결정론적 유한 상태 머신(FSM) 4단계 상태"""

    IDLE = "IDLE"                                # 3공 모두 완전 정지 대기 상태
    IN_PLAY = "IN_PLAY"                          # 지정 수구 출발 후 샷 진행 중
    VALID_SCORE_PENDING = "VALID_SCORE_PENDING"  # 3쿠션 이상 달성 후 제2목적구 타격 성공 (정지 시 득점 확정)
    NO_SCORE_LOCKED = "NO_SCORE_LOCKED"          # 조기 제2목적구 접촉(키스 노카운트) 또는 파울로 무효 잠금된 상태


class FailReason(str, Enum):
    """노카운트(공타) 및 파울 상세 사유"""

    NONE = "NONE"
    INSUFFICIENT_CUSHIONS = "INSUFFICIENT_CUSHIONS"          # 3쿠션 미만
    EARLY_SECOND_BALL_KISS = "EARLY_SECOND_BALL_KISS"        # 3쿠션 달성 전 제2목적구 조기 접촉(키스 무효 잠금)
    MISSING_SECOND_BALL = "MISSING_SECOND_BALL"              # 제2목적구 미접촉
    MISSING_FIRST_BALL = "MISSING_FIRST_BALL"                # 제1목적구 미접촉 (완전 공타)
    WRONG_CUE_BALL_FOUL = "WRONG_CUE_BALL_FOUL"              # 상대방 수구 또는 적구 선행 출발 파울
    BALL_OUT_OF_TABLE = "BALL_OUT_OF_TABLE"                  # 당구대 장외 이탈 파울
    DOUBLE_HIT_FOUL = "DOUBLE_HIT_FOUL"                      # 큐대 투터치(밀어치기) 파울


@dataclass(frozen=True)
class VisionConfig:
    """
    국제식 대대 물리 상수 및 60fps Vision-Only 판정 임계값 설정 데이터클래스
    """

    # 1. 국제식 대대 내측 쿠션(석판 경계) 규격 및 당구공 치수 (단위: mm)
    TABLE_WIDTH_MM: float = 2840.0
    TABLE_HEIGHT_MM: float = 1420.0
    BALL_DIAMETER_MM: float = 61.5
    BALL_RADIUS_MM: float = 30.75

    # 2. 카메라 프레임 레이트 및 대각선 카메라 시차(Parallax) 보정 계수
    DEFAULT_FPS: float = 60.0
    CAMERA_TILT_RATIO: float = 0.12  # 바운딩 박스 높이 h 대비 바닥 접점 하향 보정 비율

    # 3. 적응형 EMA(Adaptive Exponential Moving Average) 필터 계수
    EMA_ALPHA_STATIONARY: float = 0.25       # 정지/저속 구간 강한 떨림 억제
    EMA_ALPHA_FAST: float = 0.75             # 고속 이동 구간 빠른 궤적 응답
    EMA_FAST_SPEED_TH_MM_S: float = 120.0    # 저속 vs 고속 전환 기준 속력 (mm/s)

    # 4. 서브프레임 CCD 및 미세 스침(Micro-Kiss) 이중 게이트 임계값
    DIRECT_CONTACT_MARGIN_MM: float = 2.0
    DIRECT_CONTACT_DIST_MM: float = 63.5     # 2R(61.5mm) + 2.0mm = 확실한 정면/측면 충돌 임계값
    GRAY_ZONE_MAX_DIST_MM: float = 78.0      # 63.5mm < d_min <= 78.0mm: 미세 스침(키스) 후보 그레이존
    MICRO_KISS_VERIFY_FRAMES: int = 3        # 그레이존 진입 후 물리 증거 관찰 대기 프레임 수 (3프레임)
    MIN_DEFLECTION_ANGLE_DEG: float = 1.8    # 증거 1: 수구 전후 3프레임 평균 진행 벡터 사잇각 >= 1.8도
    MIN_TARGET_MICRO_DISP_MM: float = 3.5    # 증거 2A: 정지 목적구가 초기 위치 대비 >= 3.5mm 일관되게 밀림
    MIN_TARGET_MICRO_SPEED_MM_S: float = 25.0  # 증거 2B: 정지 목적구에 >= 25.0 mm/s의 속도 발생
    CONTACT_COOLDOWN_SEC: float = 0.08       # 동일 목적구 연속 프레임 중복 접촉 쿨다운 (초)

    # 5. 법선 속도 반전 기반 4면 쿠션 판정 임계값
    RAIL_EXTRA_MARGIN_MM: float = 14.0
    RAIL_ZONE_MARGIN_MM: float = 44.75       # R(30.75mm) + 14.0mm = 레일 마진 진입 영역
    MIN_RAIL_APPROACH_SPEED_MM_S: float = 15.0

    # 6. 쓰리쿠션 공식 규칙 유한 상태 머신(FSM) 속도 및 시간 임계값
    IDLE_STOP_SPEED_MM_S: float = 15.0       # 정지 판정 속력 임계값 (< 15.0 mm/s)
    IDLE_CONFIRM_DURATION_SEC: float = 0.30  # 모든 공 정지 유지 확정 시간 (0.3초 = 60fps 기준 18프레임)
    STROKE_START_SPEED_MM_S: float = 180.0   # 수구 타구 출발 감지 속력 (>= 180.0 mm/s)
    OUT_OF_TABLE_TOLERANCE_MM: float = 35.0  # 당구대 이탈 허용 오차 마진 (mm)

    # 7. 투터치(Double Hit) 파울 감지 및 바운딩 박스 병합(Blob Merging) 중심점 보정 임계값
    DOUBLE_HIT_WINDOW_SEC: float = 0.15            # 스트로크 직후 0.15초(9프레임) 이내
    DOUBLE_HIT_INIT_DIST_MAX_MM: float = 115.0     # 수구-제1적구 초기 근접 거리 임계값 (mm)
    DOUBLE_HIT_REACCEL_THRESH_MM_S2: float = 1500.0  # 충돌 직후 재가속 임계값 (mm/s^2)
    DOUBLE_HIT_FOLLOW_RATIO: float = 0.75          # 정면충돌 후 수구/제1적구 법선 속도 비율 한계
    BLOB_PROXIMITY_ZONE_MM: float = 85.0           # 박스 병합 간섭이 일어나는 근접 반경 (mm)
    MAX_CENTROID_JUMP_RESIDUAL_MM: float = 14.0    # 1프레임 내 비물리적 중심점 끌림 허용 한계 (mm)
    MAX_OCCLUSION_INTERP_FRAMES: int = 6           # 결측 시 외삽 복원 최대 프레임 수


DEFAULT_CONFIG = VisionConfig()
