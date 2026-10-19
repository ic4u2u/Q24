"""
실시간 라이브캠 프레임 생성기 및 3공 정지 시 직전 샷 자동 리플레이(Auto-Replay) 링버퍼 모듈
파일 경로: cue24/core/replay_buffer.py

1. ReplayRingBuffer:
   - 샷 주행 중 프레임을 순환 버퍼(최대 300프레임, 약 10초)에 연속 저장합니다.
   - 공 3개가 완전히 멈추는 순간(BALL_STOPPED), 방금 친 샷의 프레임 시퀀스를 확정(lock_last_shot)하여
     태블릿 중앙 상단 모니터에서 즉시 자동 리플레이(Auto-Replay)할 수 있도록 제공합니다.
2. SyntheticCamGenerator:
   - 물리 카메라가 연결되지 않은 환경에서도 국제식 대대(파란색 라사지, 다이나믹 포인트 마커) 위에서
     공 3개(백구/황구/적구)가 이동하고 3쿠션 궤적을 그리는 고선명 라이브 및 리플레이 영상을 생성합니다.
"""

from collections import deque
import math
import time
from typing import Any, Deque, Dict, List, Optional, Tuple
import cv2
import numpy as np

Point2D = Tuple[float, float]


class ReplayRingBuffer:
    """
    샷 시작부터 3공 완전 정지까지의 프레임을 기록하고,
    직전 샷 리플레이 클립을 보관 및 추출하는 링버퍼 클래스입니다.
    """

    def __init__(self, max_frames: int = 300) -> None:
        self.max_frames: int = max_frames
        self._active_buffer: Deque[bytes] = deque(maxlen=max_frames)
        self._last_locked_clip: List[bytes] = []
        self._last_shot_meta: Dict[str, Any] = {
            "clip_id": 0,
            "shot_type": "READY",
            "is_scored": False,
            "frame_count": 0,
            "timestamp": 0.0,
        }
        self._clip_counter: int = 0

    def push_frame(self, jpeg_bytes: bytes) -> None:
        """현재 진행 중인 샷 버퍼에 JPEG 인코딩된 프레임을 추가합니다."""
        if jpeg_bytes:
            self._active_buffer.append(jpeg_bytes)

    def clear_active(self) -> None:
        """새로운 샷 녹화를 위해 현재 작업 버퍼를 비웁니다."""
        self._active_buffer.clear()

    def lock_last_shot(
        self,
        shot_type: str = "REGULAR_3C",
        is_scored: bool = True,
        message: str = "",
    ) -> Dict[str, Any]:
        """
        공 3개가 완전히 정지(BALL_STOPPED)했을 때 호출되어,
        현재 버퍼에 기록된 프레임들을 '직전 샷 자동 리플레이 클립'으로 확정(Lock)합니다.
        """
        if len(self._active_buffer) > 0:
            self._last_locked_clip = list(self._active_buffer)
        self._clip_counter += 1
        self._last_shot_meta = {
            "clip_id": self._clip_counter,
            "shot_type": shot_type,
            "is_scored": is_scored,
            "message": message,
            "frame_count": len(self._last_locked_clip),
            "timestamp": round(time.time(), 2),
        }
        return dict(self._last_shot_meta)

    def get_locked_clip(self) -> List[bytes]:
        """확정된 직전 샷의 전체 JPEG 프레임 리스트를 반환합니다."""
        return list(self._last_locked_clip)

    def get_last_shot_meta(self) -> Dict[str, Any]:
        """직전 샷 리플레이 메타데이터를 반환합니다."""
        return dict(self._last_shot_meta)


class SyntheticCamGenerator:
    """
    카메라 미연결 시에도 태블릿 중앙 상단 모니터에서 실시간 라이브캠과
    3공 정지 후 직전 샷 자동 리플레이(궤적 오버레이 포함)를 완벽히 시연하기 위한
    고화질 대대 탑뷰(Top-View) 합성 영상 생성기입니다.
    """

    def __init__(self, width: int = 960, height: int = 480) -> None:
        self.width: int = width
        self.height: int = height
        self._tick: int = 0

    def _draw_table_base(self) -> np.ndarray:
        """국제식 대대 나무 프레임, 다이아몬드 포인트, 파란색 라사지 배경을 렌더링합니다."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 1. 외곽 우드 프레임 (어두운 월넛 브라운)
        frame[:, :] = (28, 36, 52)
        margin = 28

        # 2. 쿠션 레일 영역 (진한 블루)
        cv2.rectangle(
            frame,
            (margin - 8, margin - 8),
            (self.width - margin + 8, self.height - margin + 8),
            (135, 65, 15),
            thickness=-1,
        )

        # 3. 대대 내경 라사지 (프로 대회용 로얄 블루 BGR: 185, 92, 22)
        cv2.rectangle(
            frame,
            (margin, margin),
            (self.width - margin, self.height - margin),
            (185, 92, 22),
            thickness=-1,
        )

        # 4. 당구대 다이아몬드 포인트 (가로 8등분, 세로 4등분)
        inner_w = self.width - 2 * margin
        inner_h = self.height - 2 * margin
        for i in range(9):
            px = int(margin + (inner_w * i) / 8.0)
            cv2.circle(frame, (px, margin // 2), 3, (220, 220, 220), -1)
            cv2.circle(frame, (px, self.height - margin // 2), 3, (220, 220, 220), -1)
        for j in range(5):
            py = int(margin + (inner_h * j) / 4.0)
            cv2.circle(frame, (margin // 2, py), 3, (220, 220, 220), -1)
            cv2.circle(
                frame, (self.width - margin // 2, py), 3, (220, 220, 220), -1
            )

        return frame

    def render_live_frame(
        self,
        active_cue_color: str = "WHITE",
        fsm_state: str = "WAIT_SHOT",
    ) -> bytes:
        """
        실시간 라이브캠용 단일 JPEG 프레임을 생성합니다.
        WAIT_SHOT 상태에서는 정지한 공 3개와 조준 가이드라인을 표시합니다.
        """
        self._tick += 1
        frame = self._draw_table_base()

        # 기본 공 위치 (백구, 황구, 적구)
        w_pos = (260, 260)
        y_pos = (260, 140)
        r_pos = (700, 240)

        cue_pos = w_pos if active_cue_color == "WHITE" else y_pos
        pulse = int(4 * abs(math.sin(self._tick * 0.15)))

        # 현재 공격자 수구 주변에 네온 타겟 링 표시
        cv2.circle(frame, cue_pos, 18 + pulse, (255, 230, 0), 2, cv2.LINE_AA)

        # 당구공 3개 그리기 (그림자 + 구체 + 조명 하이라이트)
        self._draw_ball(frame, w_pos, (245, 245, 245))
        self._draw_ball(frame, y_pos, (25, 215, 250))
        self._draw_ball(frame, r_pos, (30, 30, 235))

        # 상단 HUD 텍스트
        cv2.putText(
            frame,
            f"CUE24 LIVE CAM | STATE: {fsm_state} | CUE: {active_cue_color}",
            (42, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 250, 255),
            2,
            cv2.LINE_AA,
        )

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        return buf.tobytes() if ok else b""

    def generate_shot_replay_clip(
        self,
        scenario: str = "REGULAR_3C",
        active_cue_color: str = "WHITE",
        num_frames: int = 36,
    ) -> List[bytes]:
        """
        시나리오(일반 3C, 뱅크샷, 2C 공타, VAR 미세접촉, VAR 더블터치)에 맞는
        수구 주행 궤적과 3공 정지 장면을 시뮬레이션하여 리플레이 JPEG 프레임 리스트를 생성합니다.
        """
        margin = 28
        left_x, right_x = margin + 14, self.width - margin - 14
        top_y, bottom_y = margin + 14, self.height - margin - 14

        start_cue = (240.0, 280.0) if active_cue_color == "WHITE" else (240.0, 160.0)
        other_cue = (240.0, 160.0) if active_cue_color == "WHITE" else (240.0, 280.0)
        red_start = (480.0, 140.0)

        # 시나리오별 수구 경유 웨이포인트 정의
        if scenario == "BANK_SHOT":
            waypoints = [
                start_cue,
                (380.0, float(top_y)),
                (float(right_x), 220.0),
                (620.0, float(bottom_y)),
                red_start,
                other_cue,
            ]
        elif scenario == "MISS_2C":
            waypoints = [
                start_cue,
                red_start,
                (650.0, float(top_y)),
                (float(right_x), 260.0),
                (520.0, 320.0),
            ]
        elif scenario == "VAR_DOUBLE_TOUCH":
            waypoints = [
                start_cue,
                red_start,
                (520.0, float(top_y)),
                (565.0, float(top_y + 8)),
                (610.0, float(top_y)),
                (400.0, 300.0),
            ]
        else:
            # REGULAR_3C 또는 VAR_MICRO_CONTACT
            waypoints = [
                start_cue,
                red_start,
                (620.0, float(top_y)),
                (float(right_x), 240.0),
                (540.0, float(bottom_y)),
                other_cue,
            ]

        # 웨이포인트 보간으로 프레임별 수구 좌표 생성 (마지막 6프레임은 완전 정지 표현)
        moving_frames = max(10, num_frames - 6)
        cue_path: List[Tuple[int, int]] = []
        segments = len(waypoints) - 1

        for idx in range(moving_frames):
            progress = (idx / float(moving_frames - 1)) * segments
            seg_i = min(segments - 1, int(progress))
            local_t = progress - seg_i
            p1 = waypoints[seg_i]
            p2 = waypoints[seg_i + 1]
            cx = int(p1[0] + (p2[0] - p1[0]) * local_t)
            cy = int(p1[1] + (p2[1] - p1[1]) * local_t)
            cue_path.append((cx, cy))

        # 마지막 6프레임 정지 상태 유지
        for _ in range(num_frames - moving_frames):
            cue_path.append(cue_path[-1])

        frames_bytes: List[bytes] = []
        cue_bgr = (245, 245, 245) if active_cue_color == "WHITE" else (25, 215, 250)
        other_bgr = (25, 215, 250) if active_cue_color == "WHITE" else (245, 245, 245)

        for f_idx, c_pt in enumerate(cue_path):
            canvas = self._draw_table_base()

            # 지금까지 이동한 수구 궤적 선 그리기
            if f_idx >= 1:
                pts_arr = np.array(cue_path[: f_idx + 1], dtype=np.int32).reshape(
                    (-1, 1, 2)
                )
                cv2.polylines(
                    canvas, [pts_arr], False, (0, 242, 254), 2, cv2.LINE_AA
                )

            # 제1적구(RED)가 충돌 이후 약간 밀려나는 효과
            hit_progress = max(0.0, (f_idx - 6) / float(num_frames))
            red_curr = (
                int(red_start[0] + hit_progress * 70),
                int(red_start[1] + hit_progress * 45),
            )
            other_curr = (int(other_cue[0]), int(other_cue[1]))

            self._draw_ball(canvas, other_curr, other_bgr)
            self._draw_ball(canvas, red_curr, (30, 30, 235))
            self._draw_ball(canvas, c_pt, cue_bgr)

            status_tag = (
                "BALLS STOPPED (AUTO-REPLAY)"
                if f_idx >= moving_frames
                else f"REPLAYING SHOT ({scenario})"
            )
            cv2.putText(
                canvas,
                f"AUTO REPLAY | {status_tag} | FRAME {f_idx + 1}/{num_frames}",
                (42, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (80, 255, 180),
                2,
                cv2.LINE_AA,
            )

            ok, buf = cv2.imencode(
                ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 82]
            )
            if ok:
                frames_bytes.append(buf.tobytes())

        return frames_bytes

    @staticmethod
    def _draw_ball(
        canvas: np.ndarray, center: Tuple[int, int], bgr_color: Tuple[int, int, int]
    ) -> None:
        """라사지 바닥 그림자와 상단 조명 하이라이트가 포함된 입체 당구공을 렌더링합니다."""
        cx, cy = center
        # 바닥 그림자
        cv2.circle(canvas, (cx + 3, cy + 3), 13, (60, 30, 10), -1, cv2.LINE_AA)
        # 당구공 본체
        cv2.circle(canvas, (cx, cy), 13, bgr_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 13, (40, 40, 40), 1, cv2.LINE_AA)
        # 천장 조명 하이라이트
        cv2.circle(canvas, (cx - 4, cy - 4), 4, (255, 255, 255), -1, cv2.LINE_AA)
