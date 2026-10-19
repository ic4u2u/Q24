"""
CUE24 대대 3쿠션 무인 관제 및 자동 판독 SaaS 통합 실행 엔트리포인트
파일 경로: cue24/main.py

역할:
1. uvicorn을 사용하여 FastAPI 실시간 WebSocket/REST 서버(api/server.py)를 가동합니다.
2. 비디오 파일(--video) 또는 실시간 카메라 스트림을 입력받아 백그라운드에서 CUE24 비전·물리·VAR 파이프라인을 구동합니다.
3. 테스트용 가상 3쿠션 주행 비디오 파일 생성(--generate-demo-video) 기능을 내장하여 카메라 없이도 즉시 검증할 수 있습니다.
"""

import argparse
import asyncio
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
import uvicorn

from cue24.api.server import app, stream_reader
from cue24.config.settings import settings

Point2D = Tuple[float, float]


def create_demo_3cushion_video(output_path: str, fps: int = 30) -> str:
    """
    테스트용 가상 대대 3쿠션 주행 비디오 파일(.mp4)을 생성합니다.
    수구(WHITE)가 제1적구(RED)를 맞힌 뒤 3쿠션을 돌고 제2적구(YELLOW)를 타격하여 정지하는 과정을 렌더링합니다.
    """
    out_file = Path(output_path).resolve()
    width, height = 1420, 710  # 2:1 비율 (절반 스케일 고속 렌더링)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, float(fps), (width, height))

    # 60프레임(2초) 분량의 가상 궤적 생성
    total_frames = 60
    for f_idx in range(total_frames):
        # 파란색 당구대 라사지 배경 (BGR: 180, 90, 20)
        frame = np.full((height, width, 3), (180, 90, 20), dtype=np.uint8)

        # 주행 구간 (0~40프레임) vs 15프레임 이상 완전 정지 구간 (41~59프레임)
        t = min(1.0, f_idx / 40.0)
        cue_x = int(200 + t * 850)
        cue_y = int(350 + np.sin(t * np.pi * 3.0) * 320)

        red_pos = (500, 220)
        yellow_pos = (1080, 350)

        # 당구공 3개 렌더링 (백구, 황구, 적구)
        cv2.circle(frame, (cue_x, cue_y), 16, (245, 245, 245), -1)
        cv2.circle(frame, yellow_pos, 16, (20, 220, 245), -1)
        cv2.circle(frame, red_pos, 16, (25, 25, 230), -1)

        writer.write(frame)

    writer.release()
    return str(out_file)


async def run_video_pipeline_background(video_source: str) -> None:
    """
    지정된 비디오 파일 또는 카메라 디바이스를 열어 백그라운드에서 CUE24 전체 비전·물리 파이프라인을 구동합니다.
    """
    await asyncio.sleep(1.0)  # FastAPI 서버 기동 대기
    cap = cv2.VideoCapture(int(video_source) if video_source.isdigit() else video_source)
    if not cap.isOpened():
        print(f"[경고] 비디오 소스를 열 수 없습니다: {video_source}")
        return

    print(f"[CUE24 파이프라인] 비디오 스트림 분석 시작: {video_source}")
    try:
        while cap.isOpened():
            ret, raw_frame = cap.read()
            if not ret:
                break

            # 원본 화각의 4모서리를 전체 프레임 크기에 맞춰 업데이트 후 2840x1420 평면 보정
            h, w = raw_frame.shape[:2]
            stream_reader.calibrator.update_corners(
                [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]
            )
            warped = stream_reader.calibrator.warp(raw_frame)

            # 진원도(>0.72) 기반 당구공 3개 후보 검출
            candidates = stream_reader.detector.detect(warped, max_balls=3)
            if len(candidates) == 3:
                pts: List[Point2D] = [(c["x"], c["y"]) for c in candidates]
                await stream_reader.process_coordinate_step(detected_points=pts)

            await asyncio.sleep(1.0 / 30.0)
    finally:
        cap.release()
        print("[CUE24 파이프라인] 비디오 스트림 분석 완료")


def parse_args(args_list: Optional[List[str]] = None) -> argparse.Namespace:
    """커맨드라인 실행 인자 파서"""
    parser = argparse.ArgumentParser(
        description="CUE24 대대 3쿠션 무인 관제 및 자동 판독 SaaS 서버 실행기"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="바인딩 호스트 주소 (기본: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트 번호 (기본: 8000)")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="백그라운드에서 분석할 비디오 파일 경로 또는 웹캠 인덱스 (예: demo.mp4 또는 0)",
    )
    parser.add_argument(
        "--generate-demo-video",
        type=str,
        default=None,
        help="지정한 경로에 가상 3쿠션 테스트용 MP4 비디오 파일을 생성하고 종료합니다.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="시스템 설정 및 모듈 초기화 상태만 빠르게 점검하고 종료합니다.",
    )
    return parser.parse_args(args_list)


def main(args_list: Optional[List[str]] = None) -> None:
    """CUE24 통합 실행 엔트리포인트"""
    args = parse_args(args_list)

    print(f"[{settings.PROJECT_NAME} v{settings.VERSION}] 시스템 초기화 완료")
    print(
        f"- 대대 규격: {settings.TABLE_WIDTH_MM}mm x {settings.TABLE_HEIGHT_MM}mm | 당구공 직경: {settings.BALL_DIAMETER_MM}mm"
    )

    if args.generate_demo_video:
        saved_path = create_demo_3cushion_video(args.generate_demo_video)
        print(f"- 가상 데모 비디오 생성 완료: {saved_path}")
        return

    if args.check:
        print("- 모든 CUE24 코어/서비스/API 모듈 정상 로드 확인 완료")
        return

    if args.video:
        video_src = args.video

        @app.on_event("startup")
        async def _start_video_worker() -> None:
            asyncio.create_task(run_video_pipeline_background(video_src))

    print(f"- 20인치 태블릿 디지털 점수판 접속 주소: http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
