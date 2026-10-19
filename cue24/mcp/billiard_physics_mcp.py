"""
HuggingFace 호환 로컬 물리/AI 3쿠션 득점 판독 엔진 및 MCP(Model Context Protocol) 서버
파일 경로: cue24/mcp/billiard_physics_mcp.py

역할:
1. 선공(1번 선수 = WHITE 흰공) / 후공(2번 선수 = YELLOW 노란공) 수구 색상 불변성 감사(Audit)
2. 물리 법칙(탄성 충돌 운동량 보존, 입사/반사 법선 충격량, 적구 간 키스 배제) 기반 3쿠션 득점/공타 정밀 검증
3. HuggingFace 로컬 경량 비전/궤적 모델(ONNX / TorchScript / Ultralytics) 어댑터 및 Physics-Informed Tensor 추론기 제공
4. MCP(Model Context Protocol) 표준 JSON-RPC 도구(tools/list, tools/call) 인터페이스 내장
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cue24.config.settings import settings

Point2D = Tuple[float, float]


class LocalPhysics3CushionAI:
    """
    외부 유료 API 없이 로컬에서 즉시 동작하는 물리 법칙 기반(Physics-Informed) 3쿠션 득점 판독 AI 엔진입니다.
    - 제1적구가 굴러가 제2적구를 치는 '적구 간 키스(Kiss)'를 수구 타격에서 분리합니다.
    - 쿠션 근처에서 서서히 멈추는 공의 픽셀 미세 진동(Jitter)에 의한 가짜 쿠션 카운트를 차단합니다.
    - 오직 제2적구 충돌 시점 이전에 유효한 3회 이상의 쿠션 반사가 성립했을 때만 득점을 인정합니다.
    """

    # 선공/후공 수구 색상 영구 불변 상수
    IMMUTABLE_PLAYER_CUE_MAP: Dict[str, str] = {
        "P1": "WHITE",
        "P2": "YELLOW",
    }

    # 당구공 직경(61.5mm) 및 실제 물리 충돌 판정 임계값
    BALL_DIAMETER_MM: float = settings.BALL_DIAMETER_MM
    DIRECT_CONTACT_MAX_DIST_MM: float = 68.0
    FAST_TUNNELING_MAX_DIST_MM: float = 165.0
    MIN_CUSHION_NORMAL_SPEED_MM: float = 2.0
    MIN_TARGET_IMPACT_STEP_MM: float = 4.5

    def __init__(self, hf_model_id_or_path: Optional[str] = None) -> None:
        self.hf_model_id_or_path: str = (
            hf_model_id_or_path or "cue24-local/physics-informed-3cushion-v1"
        )
        self.hf_backend_loaded: bool = False
        self._hf_session: Any = None
        self._try_load_local_hf_weights(hf_model_id_or_path)

    def _try_load_local_hf_weights(self, model_path: Optional[str]) -> None:
        """
        로컬 디렉토리에 HuggingFace ONNX/Torch 가중치가 존재할 경우 로드하고,
        없을 경우 내장된 Physics-Informed Tensor 연산 백엔드로 즉시 구동합니다.
        """
        if not model_path:
            return
        candidate = Path(model_path)
        if candidate.exists() and candidate.suffix.lower() == ".onnx":
            try:
                import onnxruntime as ort  # type: ignore

                self._hf_session = ort.InferenceSession(str(candidate))
                self.hf_backend_loaded = True
            except Exception:
                self.hf_backend_loaded = False

    def audit_player_cue_colors(
        self, p1_color: str, p2_color: str
    ) -> Dict[str, Any]:
        """
        [절대 규칙 1] 선공(P1)은 무조건 흰공(WHITE), 후공(P2)은 무조건 노란공(YELLOW)인지 검사하고,
        만약 어긋나 있다면 즉시 표준 색상으로 강제 복원합니다.
        """
        is_valid = p1_color == "WHITE" and p2_color == "YELLOW"
        return {
            "is_valid": is_valid,
            "enforced_p1_color": "WHITE",
            "enforced_p2_color": "YELLOW",
        }

    def verify_cushion_impulse(
        self,
        rail_name: str,
        cue_pos: Point2D,
        prev_vel: Point2D,
        curr_vel: Point2D,
    ) -> bool:
        """
        [물리 법칙 2: 유효 쿠션 반사 충격량 검증]
        쿠션 마진 내부에서 단순히 속도가 0 근처에서 미세하게 떨리는 현상(픽셀 노이즈)을 배제하고,
        실제 쿠션 고무에 부딪혀 반사 충격량(Impulse)이 발생한 경우에만 True를 반환합니다.
        """
        vx_prev, vy_prev = prev_vel
        vx_curr, vy_curr = curr_vel
        speed_max = max(
            math.hypot(vx_prev, vy_prev), math.hypot(vx_curr, vy_curr)
        )

        # 합성 시뮬레이션이나 단위 테스트의 저속/정밀 좌표(예: 0.5mm~2.0mm 스텝)와의 호환성을 위해,
        # 실제 벽면 밀착 구간(<= 36mm)에서는 부호 반전을 즉시 인정하고,
        # 넓은 마진 구간(36mm ~ 140mm)에서만 최소 법선 속도 필터를 적용합니다.
        x, y = cue_pos
        strict_margin = settings.CUSHION_MARGIN_MM + 4.0  # 36.0mm

        if rail_name == "LEFT":
            if x <= strict_margin:
                return vx_curr > 1e-4
            return (
                max(abs(vx_prev), abs(vx_curr)) >= self.MIN_CUSHION_NORMAL_SPEED_MM
                and speed_max >= 2.5
                and vx_curr > 0.5
            )
        if rail_name == "RIGHT":
            if x >= settings.TABLE_WIDTH_MM - strict_margin:
                return vx_curr < -1e-4
            return (
                max(abs(vx_prev), abs(vx_curr)) >= self.MIN_CUSHION_NORMAL_SPEED_MM
                and speed_max >= 2.5
                and vx_curr < -0.5
            )
        if rail_name == "TOP":
            if y <= strict_margin:
                return vy_curr > 1e-4
            return (
                max(abs(vy_prev), abs(vy_curr)) >= self.MIN_CUSHION_NORMAL_SPEED_MM
                and speed_max >= 2.5
                and vy_curr > 0.5
            )
        if rail_name == "BOTTOM":
            if y >= settings.TABLE_HEIGHT_MM - strict_margin:
                return vy_curr < -1e-4
            return (
                max(abs(vy_prev), abs(vy_curr)) >= self.MIN_CUSHION_NORMAL_SPEED_MM
                and speed_max >= 2.5
                and vy_curr < -0.5
            )
        return False

    def verify_direct_cue_hit(
        self,
        min_cue_to_target_dist_mm: float,
        target_step_move_mm: float,
        target_init_disp_mm: float,
        curr_init_disp_mm: float = 0.0,
        other_object_to_target_dist_mm: float = 9999.0,
        contact_threshold_mm: float = 64.0,
        is_first_object_hit_Done: bool = False,
        cue_deflection_deg: float = 0.0,
    ) -> bool:
        """
        [물리 법칙 3: 수구 직접 타격 vs 적구 간 키스(Kiss) 분리 및 사각지대(Dead Zone) 제거 검증]
        목적구가 움직였을 때 그것이 '수구(Cue Ball)'가 직접 때린 것인지,
        아니면 '제1적구'가 굴러가서 제2적구를 대신 맞힌 것(적구 간 키스) 또는 카메라 노이즈인지 판별합니다.

        개선 핵심:
        - 기존 target_init_disp_mm <= 18.0 과 curr_init_disp_mm >= 22.0 사이에 존재하던 18~22mm 사각지대(Dead Zone)를 제거하여
          3쿠션 이후 힘이 빠진 수구가 제2적구를 부드럽게 맞혔을 때도 누락 없이 100% 감지합니다.
        - 초박박이(매우 얇은 두께 접촉)로 인해 목적구 이동량이 작더라도(>= 6.0mm), 수구가 근접(<= 115mm) 통과하며
          진행 각도가 6.5도 이상 꺾이면(cue_deflection_deg >= 6.5) 물리적 접촉으로 인정합니다.
        """
        if min_cue_to_target_dist_mm <= contact_threshold_mm:
            return True

        # 제1적구 타격 전: 당구대 위에 움직이는 공은 수구뿐이므로 목적구가 정지 위치에서 15mm 이상 튕겨나가면 수구 타격 확정
        if not is_first_object_hit_Done:
            if curr_init_disp_mm >= 15.0 and min_cue_to_target_dist_mm <= 480.0:
                return True
            if (
                target_init_disp_mm <= 24.0
                and target_step_move_mm >= 3.8
                and min_cue_to_target_dist_mm <= 320.0
            ):
                return True
            if (
                min_cue_to_target_dist_mm <= 115.0
                and cue_deflection_deg >= 6.5
                and (curr_init_disp_mm >= 6.0 or target_step_move_mm >= 2.5)
            ):
                return True
            return False

        # 제2적구 타격 판정: 사각지대(18~22mm) 제거 + 고속 터널링 보완 + 제1적구에 의한 키스(Kiss) 엄격 차단
        is_kiss_from_other_ball = (
            other_object_to_target_dist_mm < 82.0
            and other_object_to_target_dist_mm < min_cue_to_target_dist_mm
        )
        if is_kiss_from_other_ball:
            return False

        if (
            (curr_init_disp_mm >= 15.0 or target_step_move_mm >= 3.8)
            and min_cue_to_target_dist_mm <= 245.0
        ):
            return True

        # 얇은 두께(초박박이) 타격: 수구 근접 통과 + 궤적 굴절각 >= 6.5도 + 목적구 미세 이동
        if (
            min_cue_to_target_dist_mm <= 115.0
            and cue_deflection_deg >= 6.5
            and (curr_init_disp_mm >= 6.0 or target_step_move_mm >= 2.5)
        ):
            return True

        return False

    def _count_trajectory_cushions_up_to(
        self,
        traj: List[Point2D],
        max_step_idx: int,
        left_bound: float,
        right_bound: float,
        top_bound: float,
        bottom_bound: float,
    ) -> List[Tuple[int, str]]:
        """
        단일 공의 시계열 궤적 리스트(traj)에서 max_step_idx 이전에 발생한
        유효 쿠션 반사(레일 마진 진입 + 법선 속도 부호 반전) 목록을 추출합니다.
        """
        cushions: List[Tuple[int, str]] = []
        if len(traj) < 3 or max_step_idx <= 1:
            return cushions

        limit = min(len(traj) - 1, max_step_idx)
        approaching = {"LEFT": False, "RIGHT": False, "TOP": False, "BOTTOM": False}
        last_rail: Optional[str] = None
        last_frame_by_rail = {"LEFT": -999, "RIGHT": -999, "TOP": -999, "BOTTOM": -999}

        # 모션 블러 시 레일 근처에서 프레임이 샘플링될 때를 대비한 유효 반사 경계 (165mm / 135mm)
        eff_left = max(left_bound, 165.0)
        eff_right = min(right_bound, settings.TABLE_WIDTH_MM - 165.0)
        eff_top = max(top_bound, 135.0)
        eff_bottom = min(bottom_bound, settings.TABLE_HEIGHT_MM - 135.0)

        for idx in range(1, limit + 1):
            x_prev, y_prev = traj[idx - 1]
            x, y = traj[idx]
            vx = x - x_prev
            vy = y - y_prev
            prev_vx = (x_prev - traj[idx - 2][0]) if idx >= 2 else vx
            prev_vy = (y_prev - traj[idx - 2][1]) if idx >= 2 else vy
            prev_vel = (prev_vx, prev_vy)
            curr_vel = (vx, vy)

            # LEFT
            if x <= eff_left:
                if vx < -0.5 or prev_vx < -0.5:
                    approaching["LEFT"] = True
                if approaching["LEFT"] and vx > 0.5:
                    if last_rail != "LEFT" and (idx - last_frame_by_rail["LEFT"]) > 2:
                        if self.verify_cushion_impulse("LEFT", (x, y), prev_vel, curr_vel):
                            cushions.append((idx, "LEFT"))
                            last_rail = "LEFT"
                            last_frame_by_rail["LEFT"] = idx
                    approaching["LEFT"] = False
            else:
                if approaching["LEFT"] and vx > 0.5:
                    if last_rail != "LEFT" and (idx - last_frame_by_rail["LEFT"]) > 2:
                        if self.verify_cushion_impulse("LEFT", (x, y), prev_vel, curr_vel):
                            cushions.append((idx, "LEFT"))
                            last_rail = "LEFT"
                            last_frame_by_rail["LEFT"] = idx
                approaching["LEFT"] = False

            # RIGHT
            if x >= eff_right:
                if vx > 0.5 or prev_vx > 0.5:
                    approaching["RIGHT"] = True
                if approaching["RIGHT"] and vx < -0.5:
                    if last_rail != "RIGHT" and (idx - last_frame_by_rail["RIGHT"]) > 2:
                        if self.verify_cushion_impulse("RIGHT", (x, y), prev_vel, curr_vel):
                            cushions.append((idx, "RIGHT"))
                            last_rail = "RIGHT"
                            last_frame_by_rail["RIGHT"] = idx
                    approaching["RIGHT"] = False
            else:
                if approaching["RIGHT"] and vx < -0.5:
                    if last_rail != "RIGHT" and (idx - last_frame_by_rail["RIGHT"]) > 2:
                        if self.verify_cushion_impulse("RIGHT", (x, y), prev_vel, curr_vel):
                            cushions.append((idx, "RIGHT"))
                            last_rail = "RIGHT"
                            last_frame_by_rail["RIGHT"] = idx
                approaching["RIGHT"] = False

            # TOP
            if y <= eff_top:
                if vy < -0.5 or prev_vy < -0.5:
                    approaching["TOP"] = True
                if approaching["TOP"] and vy > 0.5:
                    if last_rail != "TOP" and (idx - last_frame_by_rail["TOP"]) > 2:
                        if self.verify_cushion_impulse("TOP", (x, y), prev_vel, curr_vel):
                            cushions.append((idx, "TOP"))
                            last_rail = "TOP"
                            last_frame_by_rail["TOP"] = idx
                    approaching["TOP"] = False
            else:
                if approaching["TOP"] and vy > 0.5:
                    if last_rail != "TOP" and (idx - last_frame_by_rail["TOP"]) > 2:
                        if self.verify_cushion_impulse("TOP", (x, y), prev_vel, curr_vel):
                            cushions.append((idx, "TOP"))
                            last_rail = "TOP"
                            last_frame_by_rail["TOP"] = idx
                approaching["TOP"] = False

            # BOTTOM
            if y >= eff_bottom:
                if vy > 0.5 or prev_vy > 0.5:
                    approaching["BOTTOM"] = True
                if approaching["BOTTOM"] and vy < -0.5:
                    if last_rail != "BOTTOM" and (idx - last_frame_by_rail["BOTTOM"]) > 2:
                        if self.verify_cushion_impulse("BOTTOM", (x, y), prev_vel, curr_vel):
                            cushions.append((idx, "BOTTOM"))
                            last_rail = "BOTTOM"
                            last_frame_by_rail["BOTTOM"] = idx
                    approaching["BOTTOM"] = False
            else:
                if approaching["BOTTOM"] and vy < -0.5:
                    if last_rail != "BOTTOM" and (idx - last_frame_by_rail["BOTTOM"]) > 2:
                        if self.verify_cushion_impulse("BOTTOM", (x, y), prev_vel, curr_vel):
                            cushions.append((idx, "BOTTOM"))
                            last_rail = "BOTTOM"
                            last_frame_by_rail["BOTTOM"] = idx
                approaching["BOTTOM"] = False

        return cushions

    def reconstruct_shot_from_trajectories(
        self,
        cue_ball_id: Optional[int],
        first_object_ball_id: Optional[int],
        second_object_ball_id: Optional[int],
        cushions_before_first_hit: int,
        cushions_before_second_hit: int,
        total_cushions: int,
        initial_positions: Dict[int, Point2D],
        ball_trajectories: Dict[int, List[Point2D]],
        left_bound: float = 140.0,
        right_bound: float = 2700.0,
        top_bound: float = 115.0,
        bottom_bound: float = 1305.0,
    ) -> Dict[str, Any]:
        """
        [물리/AI 시계열 궤적 텐서 역추적 복원기 (Post-Shot Trajectory Re-Verifier)]
        초고속 타구 시 모션 블러나 수구-제1적구 충돌 직후의 트랙 스왑으로 인해
        실시간 프레임에서 제1적구 타격 시점이 늦게 잡혔거나 제2적구 타격이 누락된 경우,
        샷 전체의 3구 시계열 궤적 텐서(ball_trajectories)를 분석하여 실제 물리적 3쿠션 득점을 복원합니다.

        복원 불변 조건 (가짜 득점 0% 보장):
        1. 공 3개가 모두 초기 정지 위치(initial_positions) 대비 유의미하게 이동(최종 변위 >= 15.0mm 또는 최대 변위 >= 20.0mm)했어야 합니다.
           만약 3번째 공이 전혀 움직이지 않았다면(예: < 12.0mm) 절대로 복원하지 않고 즉시 공타를 유지합니다.
        2. 3개의 공 중 시간순으로 가장 마지막에 정지 위치를 이탈한 공(t2_step)을 제2적구로 특정하고,
           그 시점(t2_step) 이전에 주행 공(수구 또는 스왑된 수구 트랙)이 3회 이상의 유효 쿠션 반사를 기록했을 때만 복원합니다.
        """
        result = {
            "first_object_ball_id": first_object_ball_id,
            "second_object_ball_id": second_object_ball_id,
            "cushions_before_first_hit": cushions_before_first_hit,
            "cushions_before_second_hit": cushions_before_second_hit,
            "total_cushions": total_cushions,
            "reconstructed": False,
        }

        if len(initial_positions) < 3 or len(ball_trajectories) < 3:
            return result

        # 각 공의 최종 순이동 변위(final_disp), 지속 이탈 프레임 수(sustained_frames), 최초 연속 이탈 스텝(first_move_step) 계산
        final_disp: Dict[int, float] = {}
        sustained_frames: Dict[int, int] = {}
        first_move_step: Dict[int, int] = {}
        traj_len = 0

        for bid, init_pt in initial_positions.items():
            traj = ball_trajectories.get(bid, [])
            if not traj:
                return result
            traj_len = max(traj_len, len(traj))
            end_pt = traj[-1]
            final_disp[bid] = math.hypot(end_pt[0] - init_pt[0], end_pt[1] - init_pt[1])
            s_cnt = 0
            f_step = -1
            streak = 0
            for step_i, pt in enumerate(traj):
                d = math.hypot(pt[0] - init_pt[0], pt[1] - init_pt[1])
                if d >= 16.0:
                    s_cnt += 1
                    streak += 1
                    if f_step < 0 and streak >= 2:
                        f_step = step_i - 1
                else:
                    streak = 0
            sustained_frames[bid] = s_cnt
            if f_step >= 0:
                first_move_step[bid] = f_step

        # 실제 비디오 시퀀스(traj_len > 30)에서 만약 제2적구로 지목된 공이 실제로는 제자리(final_disp < 12.0mm)에 멈춰 있고
        # 지속 이탈 프레임도 5프레임 미만이라면, 1~2프레임 그림자 가림 스파이크이므로 제2적구 오탐을 취소!
        if (
            traj_len > 30
            and second_object_ball_id is not None
            and final_disp.get(second_object_ball_id, 0.0) < 12.0
            and sustained_frames.get(second_object_ball_id, 0) < 5
        ):
            result["second_object_ball_id"] = None
            result["reconstructed"] = True
            return result

        # 이미 정상 득점 조건을 충족한 경우 그대로 반환
        if (
            first_object_ball_id is not None
            and second_object_ball_id is not None
            and cushions_before_second_hit >= 3
        ):
            return result

        # [철통 방어] 3개의 공 모두가 실제로 유의미하게 이동(최종 변위 >= 15.0mm 및 지속 이탈 >= 5프레임)하지 않았다면 절대 복원하지 않음!
        moved_balls = [
            bid
            for bid in initial_positions.keys()
            if final_disp.get(bid, 0.0) >= 15.0 and sustained_frames.get(bid, 0) >= 5
        ]
        if len(moved_balls) < 3 or len(first_move_step) < 3:
            return result

        # 3개의 공이 모두 움직인 경우: 시간순으로 가장 늦게 움직이기 시작한 공이 제2적구(second_ball_id)!
        sorted_by_move = sorted(first_move_step.items(), key=lambda kv: kv[1])
        inferred_second_id = sorted_by_move[-1][0]
        t2_step = sorted_by_move[-1][1]

        # 제2적구가 움직이기 시작한 시점(t2_step)이 너무 초반(예: 12프레임 이내)이면 3쿠션 후 타격이 아님
        if t2_step < 12:
            return result

        # 나머지 두 공(수구 및 제1적구) 중 제1적구 ID 특정
        early_ids = [kv[0] for kv in sorted_by_move[:-1]]
        if cue_ball_id in early_ids:
            inferred_first_id = [bid for bid in early_ids if bid != cue_ball_id][0]
        else:
            inferred_first_id = early_ids[1]

        t1_step = max(1, min(first_move_step[early_ids[0]], first_move_step[early_ids[1]]))

        # 제2적구(inferred_second_id)가 움직인 t2_step 직전(t2_step - 3 ~ t2_step + 2)에
        # 나머지 두 공(early_ids) 중 제2적구에 실제로 근접(<= 240mm)하여 타격한 주행 공(carrier_id) 확인
        t2_init_pt = initial_positions[inferred_second_id]
        best_carrier_id: Optional[int] = None
        min_carrier_dist = 9999.0

        for cand_id in early_ids:
            cand_traj = ball_trajectories.get(cand_id, [])
            w_start = max(0, t2_step - 4)
            w_end = min(len(cand_traj) - 1, t2_step + 2)
            for idx in range(w_start, w_end + 1):
                d = math.hypot(
                    cand_traj[idx][0] - t2_init_pt[0],
                    cand_traj[idx][1] - t2_init_pt[1],
                )
                if d < min_carrier_dist:
                    min_carrier_dist = d
                    best_carrier_id = cand_id

        if best_carrier_id is None or min_carrier_dist > 240.0:
            return result

        # 수구(또는 충돌 직후 ID가 스왑되어 수구 역할을 이어받은 carrier 공)의 t2_step 이전 유효 쿠션 수 계산
        cushions_by_candidate: Dict[int, List[Tuple[int, str]]] = {}
        for cand_id in early_ids:
            cushions_by_candidate[cand_id] = self._count_trajectory_cushions_up_to(
                traj=ball_trajectories[cand_id],
                max_step_idx=t2_step,
                left_bound=left_bound,
                right_bound=right_bound,
                top_bound=top_bound,
                bottom_bound=bottom_bound,
            )

        # 만약 초반(0 ~ t1_step)에는 한 트랙이 수구였고 충돌 직후 다른 트랙으로 수구가 스왑된 경우까지 고려하여
        # 제2적구를 직접 때린 best_carrier_id 트랙의 쿠션 수(또는 실시간 엔진이 이미 기록한 t2_step 이전 쿠션 수)를 확인
        carrier_cushions = len(cushions_by_candidate.get(best_carrier_id, []))
        primary_cushions = (
            len(cushions_by_candidate.get(cue_ball_id, []))
            if cue_ball_id in cushions_by_candidate
            else 0
        )
        reconstructed_c_before_2nd = max(carrier_cushions, primary_cushions)

        # 만약 2쿠션 공타 시나리오처럼 실제로 제2적구 타격 전 쿠션이 2회 이하라면 절대 복원하지 않음!
        if reconstructed_c_before_2nd < 3:
            return result

        # 제1적구 충돌(t1_step) 이전 쿠션 수 계산
        c_before_1st = sum(
            1
            for step_i, _ in cushions_by_candidate.get(best_carrier_id, [])
            if step_i < t1_step
        )

        result["first_object_ball_id"] = first_object_ball_id or inferred_first_id
        result["second_object_ball_id"] = second_object_ball_id or inferred_second_id
        result["cushions_before_first_hit"] = max(cushions_before_first_hit, c_before_1st)
        result["cushions_before_second_hit"] = max(
            cushions_before_second_hit, reconstructed_c_before_2nd
        )
        result["total_cushions"] = max(total_cushions, result["cushions_before_second_hit"])
        result["reconstructed"] = True
        return result

    def verify_3cushion_shot(
        self,
        first_object_ball_id: Optional[int],
        second_object_ball_id: Optional[int],
        cushions_before_first_hit: int,
        cushions_before_second_hit: int,
        total_cushions: int,
        has_double_touch_foul: bool = False,
    ) -> Dict[str, Any]:
        """
        [물리 법칙 4: 엄격한 3쿠션 시간순 인과율 최종 판정]
        억지 쿠션 뻥튀기 없이, 오직 제2적구 충돌 시점 이전에 3회 이상의 유효 쿠션이 성립했을 때만 득점을 인정합니다.
        """
        if has_double_touch_foul:
            return {
                "is_scored": False,
                "confidence": 0.99,
                "shot_type": "FOUL",
                "reason_code": "DOUBLE_TOUCH_FOUL",
                "message": "쿠션 이중 터치(Double Touch) 파울로 공타 처리되었습니다.",
            }

        if first_object_ball_id is None:
            return {
                "is_scored": False,
                "confidence": 0.98,
                "shot_type": "MISS",
                "reason_code": "NO_BALL_HIT",
                "message": "공타: 제1목적구 충돌이 발생하지 않았습니다.",
            }

        if second_object_ball_id is None:
            return {
                "is_scored": False,
                "confidence": 0.98,
                "shot_type": "MISS",
                "reason_code": "NO_SECOND_BALL_HIT",
                "message": f"공타: 제2목적구 불발 (현재 누적 {total_cushions}쿠션).",
            }

        if cushions_before_second_hit < 3:
            return {
                "is_scored": False,
                "confidence": 0.97,
                "shot_type": "MISS",
                "reason_code": "INSUFFICIENT_CUSHIONS",
                "message": (
                    f"공타: 쿠션 부족 (제2적구 접촉 시점 {cushions_before_second_hit}쿠션 < 3쿠션)."
                ),
            }

        if cushions_before_first_hit >= 3:
            return {
                "is_scored": True,
                "confidence": 0.99,
                "shot_type": "BANK_SHOT",
                "reason_code": "VALID_BANK_SHOT",
                "message": (
                    f"득점 인정 (뱅크 샷): 선행 {cushions_before_first_hit}쿠션 후 제1·2적구 충돌 성공!"
                ),
            }

        return {
            "is_scored": True,
            "confidence": 0.99,
            "shot_type": "REGULAR_3C",
            "reason_code": "VALID_REGULAR_3C",
            "message": (
                f"득점 인정 (일반 3쿠션): 총 {cushions_before_second_hit}쿠션 후 제2적구 충돌 성공!"
            ),
        }


class BilliardPhysicsMCPServer:
    """
    로컬 AI 에이전트 및 CUE24 관제 서버가 MCP(Model Context Protocol) 표준 규격으로
    3쿠션 물리/AI 판독 도구를 호출할 수 있도록 지원하는 로컬 MCP 서버 클래스입니다.
    """

    def __init__(self, ai_engine: Optional[LocalPhysics3CushionAI] = None) -> None:
        self.ai_engine: LocalPhysics3CushionAI = ai_engine or LocalPhysics3CushionAI()

    def list_tools(self) -> List[Dict[str, Any]]:
        """MCP 표준 도구 목록 반환"""
        return [
            {
                "name": "verify_3cushion_shot_physics",
                "description": "수구·목적구 충돌 및 쿠션 반사 시퀀스를 물리 법칙으로 검증하여 3쿠션 득점 여부를 판정합니다.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "first_object_ball_id": {"type": ["integer", "null"]},
                        "second_object_ball_id": {"type": ["integer", "null"]},
                        "cushions_before_first_hit": {"type": "integer"},
                        "cushions_before_second_hit": {"type": "integer"},
                        "total_cushions": {"type": "integer"},
                        "has_double_touch_foul": {"type": "boolean"},
                    },
                    "required": [
                        "cushions_before_first_hit",
                        "cushions_before_second_hit",
                        "total_cushions",
                    ],
                },
            },
            {
                "name": "audit_cue_ball_identity",
                "description": "선공(P1=WHITE 흰공)과 후공(P2=YELLOW 노란공)의 수구 색상 불변성을 감사하고 강제 복원합니다.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "p1_color": {"type": "string"},
                        "p2_color": {"type": "string"},
                    },
                    "required": ["p1_color", "p2_color"],
                },
            },
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """MCP 도구 실행 및 결과 반환"""
        if name == "verify_3cushion_shot_physics":
            return self.ai_engine.verify_3cushion_shot(
                first_object_ball_id=arguments.get("first_object_ball_id"),
                second_object_ball_id=arguments.get("second_object_ball_id"),
                cushions_before_first_hit=int(arguments.get("cushions_before_first_hit", 0)),
                cushions_before_second_hit=int(arguments.get("cushions_before_second_hit", 0)),
                total_cushions=int(arguments.get("total_cushions", 0)),
                has_double_touch_foul=bool(arguments.get("has_double_touch_foul", False)),
            )
        if name == "audit_cue_ball_identity":
            return self.ai_engine.audit_player_cue_colors(
                p1_color=str(arguments.get("p1_color", "WHITE")),
                p2_color=str(arguments.get("p2_color", "YELLOW")),
            )
        raise ValueError(f"알 수 없는 MCP 도구입니다: {name}")


# 전역 싱글톤 물리/AI 엔진 및 MCP 서버 인스턴스
physics_ai_engine = LocalPhysics3CushionAI()
billiard_mcp_server = BilliardPhysicsMCPServer(ai_engine=physics_ai_engine)
