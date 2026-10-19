# 프로젝트 설계도 및 파일 지도 (PRD.md)

> **규칙**: 새로운 파일이나 기능이 추가될 때마다 AI는 이 문서를 최신 상태로 업데이트합니다.

---

## 1. 서비스 목적 및 핵심 기능 요약
- **프로젝트명**: Q24 (CUE24)
- **서비스 목적**: 대대 3쿠션 무인 관제 및 컴퓨터 비전 기반 자동 판독 SaaS 시스템
- **절대 깨지면 안 되는 핵심 기능**:
  - 국제식 대대(2840x1420mm) 및 당구공(61.5mm) 전역 물리/비전 규격 상수 설정 (`cue24/config/settings.py`)
  - 호모그래피 기반 2840x1420mm 표준 당구대 평면 정규화 (`cue24/core/table_calibrator.py`)
  - 도넛(환형) 마스크 기반 반사광/그림자 제거 당구공 3색 분류 및 수구 ID 고정 (`cue24/core/color_locker.py`)
  - 진원도(>0.72) 필터 당구공 검출, 충돌 법선 벡터(N) 기반 ID 스왑 원복 물리 가드, 15프레임 스핀 정지 감지 (`cue24/core/physics_tracker.py`)
  - 법선 속도 부호 반전 기반 쿠션 판정 및 일반 3C/뱅크샷/공타 룰 평가 엔진 (`cue24/core/scoring_engine.py`)
  - Micro-Hawk 서브픽셀 미세 접촉(간극 <= 0mm, 굴절각 > 0.35도) 및 쿠션 이중 터치(90ms 내 극솟점 2회) VAR 심판 모듈 (`cue24/core/var_referee.py`)
  - 3쿠션 경기 FSM(WAIT_SHOT / SHOT_IN_PROGRESS / BALL_STOPPED), 점수/이닝/에버리지/40초 타이머, PBA 뱅크샷 2점제/이닝 기록지/타임아웃 관리 (`cue24/services/match_manager.py`)
  - 실시간 라이브캠 & 3공 정지 시 직전 샷 자동 리플레이(Auto-Replay) 링버퍼 엔진 (`cue24/core/replay_buffer.py`)
  - FastAPI 실시간 WebSocket(`/ws/scoreboard`) 브로드캐스트, MJPEG 라이브/리플레이 스트리밍 및 비전 파이프라인 서버 (`cue24/api/server.py`)
  - 20인치 태블릿 CUESCO 스타일 다크 모드 디지털 점수판 및 Hawk-Eye VAR 팝업 UI (`cue24/static/index.html`)

---

## 2. 현재 폴더 및 파일 구조도 (초보자용 한 줄 설명)
```text
h:\WorkSpace\Q24\
├── RULES.md                # AI가 코드를 멋대로 고치지 못하게 강제하는 현장 안전 수칙
├── .antigravityrules       # 안티그래비티 시스템이 자동으로 읽는 규칙 파일 (RULES.md와 동일)
├── PRD.md                  # 우리 서비스의 전체 설계도 및 각 파일 역할 지도 (현재 문서)
├── CHECKLIST.md            # 코드 수정 직후 1분 점검표 & 배포 전 5분 점검표
├── CHEATSHEET.md           # 평소에 복사해서 바로 쓰는 상황별 AI 명령어 모음집
├── .gitignore              # 비밀키(.env)와 무거운 설치 폴더(node_modules) 유출을 막는 보안 검문소
└── cue24/                  # CUE24 메인 애플리케이션 패키지
    ├── requirements.txt    # 필수 파이썬 라이브러리(OpenCV, FastAPI, PyTorch 등) 목록
    ├── main.py             # Uvicorn 웹서버 가동 및 비디오 스트림 백그라운드 파이프라인 통합 엔트리포인트
    ├── config/
    │   └── settings.py     # 테이블 규격, 물리 상수, 정지/색상 판독 임계값 전역 관리
    ├── core/               # 핵심 컴퓨터 비전 및 물리 엔진 모듈 폴더
    │   ├── table_calibrator.py # 4개 모서리 픽셀을 2840x1420mm 평면으로 펴주는 호모그래피 보정기
    │   ├── color_locker.py     # 조명 반사광(35%)과 그림자(15%)를 제거해 공 색상을 확정하고 수구를 고정하는 모듈
    │   ├── physics_tracker.py  # 진원도 공 검출기, 충돌 법선 벡터 ID 스왑 방지 추적기, 스핀 정지 감지 엔진
    │   ├── scoring_engine.py   # 쿠션 법선 속도 반전 판정 및 일반 3C/뱅크샷/공타 룰 판정 엔진
    │   ├── var_referee.py      # Micro-Hawk 서브픽셀 미세 접촉 및 쿠션 이중 터치(Double Touch) VAR 판독 모듈
    │   └── replay_buffer.py    # 샷 주행 녹화 링버퍼 및 3공 정지 시 직전 샷 자동 리플레이/가상 카메라 생성기
    ├── services/           # 경기 규칙 FSM(상태 머신) 및 스코어 관리자 폴더
    │   └── match_manager.py    # 턴, 점수, 이닝 기록지, PBA 뱅크샷 2점제, 타임아웃(+40초), 수동 보정 FSM 관리자
    ├── api/                # FastAPI 웹소켓 및 REST 라우터 폴더
    │   └── server.py           # /ws/scoreboard 웹소켓 및 /api/video/live, /api/video/replay 스트리밍 서버
    ├── static/             # 20인치 태블릿 점수판 프론트엔드 HTML/JS/CSS 폴더
    │   └── index.html          # CUESCO 초대형 폰트, 빌리보드 이닝 기록지, PBA 룰 토글, Hawk-Eye VAR 팝업 UI
    └── tests/              # 모듈별 단위 테스트 폴더
        ├── test_step2.py   # 평면 보정기 및 반사광 제거 색상 확정기 단위 테스트
        ├── test_step3.py   # 진원도 필터, 물리 충돌 가드 ID 스왑 원복, 스핀 정지 감지 단위 테스트
        ├── test_step4.py   # 3쿠션 득점/공타 판정 엔진 및 Micro-Hawk VAR 판독 모듈 단위 테스트
        ├── test_step5.py   # FSM 경기 관리자 및 WebSocket 실시간 브로드캐스트 단위 테스트
        ├── test_step6.py   # 20인치 태블릿 점수판 HTML UI 및 통합 엔트리포인트 단위 테스트
        ├── test_step7.py   # 리플레이 링버퍼, PBA 뱅크샷 2점 룰, 이닝 기록지, 타임아웃 단위 테스트
        └── test_step8.py   # 큐스코 초대형 스코어 폰트 및 빌리보드 이닝 스코어시트 UI 단위 테스트
```

---

## 3. 데이터베이스(DB) 테이블 구조 요약
- **사용 예정 DB**: Supabase (RLS 행 단위 보안 필수 적용)
- **테이블 목록**:
  - (추후 DB 설계 시 테이블명, 주요 컬럼, RLS 보안 적용 여부를 이곳에 기록합니다)
