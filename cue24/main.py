"""
CUE24 통합 실행 엔트리포인트
파일 경로: cue24/main.py
"""

from cue24.config.settings import settings


def main() -> None:
    """CUE24 시스템 초기화 및 설정 로드 확인 엔트리포인트"""
    print(f"[{settings.PROJECT_NAME} v{settings.VERSION}] 시스템 초기화 완료")
    print(f"- 대대 규격: {settings.TABLE_WIDTH_MM}mm x {settings.TABLE_HEIGHT_MM}mm | 당구공 직경: {settings.BALL_DIAMETER_MM}mm")


if __name__ == "__main__":
    main()
