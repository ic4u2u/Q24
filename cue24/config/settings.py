"""
CUE24 전역 환경 설정 및 물리/비전 규격 상수 관리 모듈
파일 경로: cue24/config/settings.py

국제식 대대(Carom) 물리 규격, 당구공 치수, 쿠션 임계값,
정지 판정 및 색상 분류(CIE-LAB) 임계값을 전역으로 관리합니다.
"""

from typing import Dict, Tuple
from pydantic import BaseModel, Field


# =====================================================================
# 1. 국제식 대대(Billiard Table) 및 당구공 물리 규격 (단위: mm)
# =====================================================================
# 국제식 대대 내경(쿠션 코 끝 기준) 가로 및 세로 길이 (2:1 비율)
TABLE_WIDTH_MM: float = 2840.0
TABLE_HEIGHT_MM: float = 1420.0

# 3쿠션 공인구 직경 및 반지름
BALL_DIAMETER_MM: float = 61.5
BALL_RADIUS_MM: float = 30.75

# 쿠션 레일 접촉 판정을 위한 마진 (공 중심이 벽면에서 이 거리 이내로 접근 시 쿠션 충돌 후보로 판정)
# 공 반지름(30.75mm) + 고무 쿠션 탄성 변형 및 비전 허용 오차(1.25mm) = 32.0mm
CUSHION_MARGIN_MM: float = 32.0


# =====================================================================
# 2. 물리 엔진 정지(Stop) 감지 및 경기 진행 임계값
# =====================================================================
# 프레임당 이동 거리 임계값 (단위: mm/frame) - 이 값 이하로 이동 시 선속도 정지 상태로 간주
STOP_VELOCITY_TH: float = 2.5

# 제자리 회전(스핀) 감지용 ROI 픽셀 차분 평균 에너지 임계값
# 공이 위치는 멈췄으나 제자리에서 강하게 스핀하고 있을 때 샷 종료로 오판하는 것을 방지
STOP_SPIN_ENERGY_TH: float = 2.0

# 완전 정지 확정 프레임 수 (30fps 기준 15프레임 = 약 0.5초 동안 연속 정지 조건 만족 시 턴 종료 확정)
STOP_CONFIRM_FRAMES: int = 15

# 3쿠션 표준 샷 제한 시간 (단위: 초)
DEFAULT_SHOT_TIMER: int = 40


# =====================================================================
# 3. 컴퓨터 비전 색상 분류 임계값 (OpenCV 8-bit CIE-LAB 색공간 기준: 0 ~ 255)
# =====================================================================
# L: 명도(밝기), A: 초록(-) ~ 빨강(+), B: 파랑(-) ~ 노랑(+)
# 색상 분류 임계값
LAB_RED_A_TH: int = 146      # A 채널 값이 146 이상이면 적구(RED)로 판정
LAB_YELLOW_B_TH: int = 142   # 적구가 아닌 공 중 B 채널 값이 142 이상이면 황구(YELLOW), 미만이면 백구(WHITE)로 판정

# 실시간 스코어보드 연동 기본 녹화 영상 경로
DEFAULT_VIDEO_PATH: str = r"H:\OBS녹화\new1020.mp4"


class Cue24Settings(BaseModel):
    """
    Pydantic 기반 CUE24 전역 설정 검증 및 조회 클래스
    런타임에서 설정값의 타입 안정성을 보장하고 API나 서비스 레이어에 주입합니다.
    """

    # 프로젝트 메타 정보
    PROJECT_NAME: str = Field(default="CUE24", description="대대 3쿠션 무인 관제 및 자동 판독 SaaS 시스템")
    VERSION: str = Field(default="1.0.0", description="시스템 버전")
    DEFAULT_VIDEO_PATH: str = Field(default=DEFAULT_VIDEO_PATH, description="실시간 스코어보드 테스트용 기본 당구 녹화 영상 경로")

    # 테이블 및 당구공 물리 규격
    TABLE_WIDTH_MM: float = Field(default=TABLE_WIDTH_MM, description="국제식 대대 가로 내경 (mm)")
    TABLE_HEIGHT_MM: float = Field(default=TABLE_HEIGHT_MM, description="국제식 대대 세로 내경 (mm)")
    BALL_DIAMETER_MM: float = Field(default=BALL_DIAMETER_MM, description="당구공 직경 (mm)")
    BALL_RADIUS_MM: float = Field(default=BALL_RADIUS_MM, description="당구공 반지름 (mm)")
    CUSHION_MARGIN_MM: float = Field(default=CUSHION_MARGIN_MM, description="쿠션 레일 접촉 판정 마진 (mm)")

    # 정지 감지 및 타이머 설정
    STOP_VELOCITY_TH: float = Field(default=STOP_VELOCITY_TH, description="정지 판정 선속도 임계값 (mm/frame)")
    STOP_SPIN_ENERGY_TH: float = Field(default=STOP_SPIN_ENERGY_TH, description="정지 판정 스핀 픽셀 차분 에너지 임계값")
    STOP_CONFIRM_FRAMES: int = Field(default=STOP_CONFIRM_FRAMES, description="완전 정지 확정 연속 프레임 수")
    DEFAULT_SHOT_TIMER: int = Field(default=DEFAULT_SHOT_TIMER, description="기본 샷 제한시간 (초)")

    # 색상 분류 임계값
    LAB_RED_A_TH: int = Field(default=LAB_RED_A_TH, description="CIE-LAB 적구 판별 A채널 임계값")
    LAB_YELLOW_B_TH: int = Field(default=LAB_YELLOW_B_TH, description="CIE-LAB 황구 판별 B채널 임계값")

    @property
    def cushion_bounds_mm(self) -> Dict[str, float]:
        """
        당구대 내경(0 ~ 2840, 0 ~ 1420)에서 쿠션 접촉으로 판정하는 4면(상/하/좌/우) 경계선 좌표를 반환합니다.
        """
        return {
            "LEFT": self.CUSHION_MARGIN_MM,
            "RIGHT": self.TABLE_WIDTH_MM - self.CUSHION_MARGIN_MM,
            "TOP": self.CUSHION_MARGIN_MM,
            "BOTTOM": self.TABLE_HEIGHT_MM - self.CUSHION_MARGIN_MM,
        }

    @property
    def table_dimensions_mm(self) -> Tuple[float, float]:
        """테이블 가로, 세로 규격 튜플 (width, height) 반환"""
        return (self.TABLE_WIDTH_MM, self.TABLE_HEIGHT_MM)


# 전역에서 싱글톤처럼 불러와 사용할 수 있는 기본 설정 인스턴스
settings = Cue24Settings()


if __name__ == "__main__":
    # 설정 파일 단독 실행 시 현재 적용된 물리/비전 규격 출력 검증
    print(f"[{settings.PROJECT_NAME} v{settings.VERSION}] 전역 물리 및 비전 규격 설정 로드 완료")
    print(f"- 테이블 규격: {settings.TABLE_WIDTH_MM}mm x {settings.TABLE_HEIGHT_MM}mm")
    print(f"- 당구공 규격: 직경 {settings.BALL_DIAMETER_MM}mm (반지름 {settings.BALL_RADIUS_MM}mm)")
    print(f"- 쿠션 판정 경계(mm): {settings.cushion_bounds_mm}")
    print(f"- 정지 판정: 속도 <= {settings.STOP_VELOCITY_TH}mm/frame, 스핀에너지 <= {settings.STOP_SPIN_ENERGY_TH}, 확정 {settings.STOP_CONFIRM_FRAMES}프레임")
    print(f"- 색상 임계값(LAB): RED A >= {settings.LAB_RED_A_TH}, YELLOW B >= {settings.LAB_YELLOW_B_TH}")
