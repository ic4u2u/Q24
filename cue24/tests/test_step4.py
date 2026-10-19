"""
Step 4 단위 테스트: 3쿠션 판정 엔진(ScoringEngine) 및 Micro-Hawk 초정밀 VAR 모듈(VARReferee) 검증
파일 경로: cue24/tests/test_step4.py
"""

import unittest

from cue24.core.scoring_engine import ScoringEngine
from cue24.core.var_referee import VARReferee


class TestStep4ScoringAndVAR(unittest.TestCase):
    """ScoringEngine 및 VARReferee 단위 테스트 스위트"""

    def test_regular_3cushion_scoring_success(self) -> None:
        """
        [시나리오 1: 일반 3쿠션 득점 성공]
        1적구(ID 2) 충돌 -> TOP, LEFT, BOTTOM 3개 쿠션 법선 속도 반전 접촉 -> 2적구(ID 3) 충돌 시
        득점 인정(is_scored=True, REGULAR_3C)되는지 검증
        """
        engine = ScoringEngine()
        engine.reset_shot(cue_ball_id=1, object_ball_ids=[2, 3])

        # 고정 목적구 초기 위치: ID 2=(500, 500), ID 3=(2000, 1000)
        # Frame 1: 수구(ID 1)가 제1적구(ID 2)와 접촉 (중심 거리 60.0mm <= 61.5mm)
        engine.update_frame(
            frame_idx=1,
            ball_positions={1: (440.0, 500.0), 2: (500.0, 500.0), 3: (2000.0, 1000.0)},
            cue_velocity=(20.0, -15.0),
        )

        # Frame 2~3: TOP 쿠션(y <= 32.0) 진입(vy < 0) 후 반사(vy > 0) -> 1쿠션 인정
        engine.update_frame(
            frame_idx=2,
            ball_positions={1: (400.0, 28.0), 2: (550.0, 500.0), 3: (2000.0, 1000.0)},
            cue_velocity=(-15.0, -20.0),
        )
        engine.update_frame(
            frame_idx=3,
            ball_positions={1: (380.0, 30.0), 2: (550.0, 500.0), 3: (2000.0, 1000.0)},
            cue_velocity=(-15.0, 20.0),
        )

        # Frame 4~5: LEFT 쿠션(x <= 32.0) 진입(vx < 0) 후 반사(vx > 0) -> 2쿠션 인정
        engine.update_frame(
            frame_idx=4,
            ball_positions={1: (25.0, 600.0), 2: (550.0, 500.0), 3: (2000.0, 1000.0)},
            cue_velocity=(-20.0, 18.0),
        )
        engine.update_frame(
            frame_idx=5,
            ball_positions={1: (29.0, 620.0), 2: (550.0, 500.0), 3: (2000.0, 1000.0)},
            cue_velocity=(22.0, 18.0),
        )

        # Frame 6~7: BOTTOM 쿠션(y >= 1388.0) 진입(vy > 0) 후 반사(vy < 0) -> 3쿠션 인정
        engine.update_frame(
            frame_idx=6,
            ball_positions={1: (1200.0, 1392.0), 2: (550.0, 500.0), 3: (2000.0, 1000.0)},
            cue_velocity=(22.0, 18.0),
        )
        engine.update_frame(
            frame_idx=7,
            ball_positions={1: (1230.0, 1390.0), 2: (550.0, 500.0), 3: (2000.0, 1000.0)},
            cue_velocity=(22.0, -18.0),
        )

        # Frame 8: 제2적구(ID 3, 2000, 1000)와 충돌 (거리 60.0mm)
        engine.update_frame(
            frame_idx=8,
            ball_positions={1: (1940.0, 1000.0), 2: (550.0, 500.0), 3: (2000.0, 1000.0)},
            cue_velocity=(20.0, -10.0),
        )

        result = engine.evaluate_shot()
        self.assertTrue(result["is_scored"])
        self.assertEqual(result["shot_type"], "REGULAR_3C")
        self.assertEqual(result["cushions_before_second_hit"], 3)

    def test_2cushion_miss_scenario(self) -> None:
        """
        [시나리오 2: 2쿠션 공타 시나리오]
        1적구 충돌 -> 2회 유효 쿠션 반사 후 제2적구를 충돌했을 때,
        쿠션 부족(INSUFFICIENT_CUSHIONS)으로 정확히 공타(is_scored=False) 판정하는지 검증
        """
        engine = ScoringEngine()
        engine.reset_shot(cue_ball_id=1, object_ball_ids=[2, 3])

        # 1. 제1적구(ID 2) 충돌
        engine.update_frame(
            frame_idx=1,
            ball_positions={1: (440.0, 500.0), 2: (500.0, 500.0), 3: (1800.0, 800.0)},
            cue_velocity=(15.0, -20.0),
        )

        # 2. TOP 쿠션 1회 반사 (vy: -20 -> +20)
        engine.update_frame(
            frame_idx=2,
            ball_positions={1: (600.0, 26.0), 2: (500.0, 500.0), 3: (1800.0, 800.0)},
            cue_velocity=(15.0, -20.0),
        )
        engine.update_frame(
            frame_idx=3,
            ball_positions={1: (620.0, 29.0), 2: (500.0, 500.0), 3: (1800.0, 800.0)},
            cue_velocity=(15.0, 20.0),
        )

        # 3. BOTTOM 쿠션 1회 반사 (vy: +20 -> -20) => 총 2쿠션
        engine.update_frame(
            frame_idx=4,
            ball_positions={1: (1200.0, 1393.0), 2: (500.0, 500.0), 3: (1800.0, 800.0)},
            cue_velocity=(15.0, 20.0),
        )
        engine.update_frame(
            frame_idx=5,
            ball_positions={1: (1220.0, 1389.0), 2: (500.0, 500.0), 3: (1800.0, 800.0)},
            cue_velocity=(15.0, -20.0),
        )

        # 4. 2쿠션 상태에서 바로 제2적구(ID 3) 충돌
        engine.update_frame(
            frame_idx=6,
            ball_positions={1: (1740.0, 800.0), 2: (500.0, 500.0), 3: (1800.0, 800.0)},
            cue_velocity=(15.0, -10.0),
        )

        result = engine.evaluate_shot()
        self.assertFalse(result["is_scored"])
        self.assertEqual(result["shot_type"], "MISS")
        self.assertEqual(result["reason_code"], "INSUFFICIENT_CUSHIONS")
        self.assertEqual(result["cushions_before_second_hit"], 2)

    def test_var_micro_contact_and_cushion_double_touch(self) -> None:
        """
        [시나리오 3: Micro-Hawk VAR 미세 접촉 및 쿠션 더블터치 시나리오]
        1) 서브픽셀 간극 <= 0mm 및 굴절각 > 0.35도일 때 미세 접촉 확정 검증
        2) 90ms 이내에 쿠션 압축 극솟점(Local Minima)이 2회 검출될 때 Double Touch 파울 판정 검증
        """
        var_ref = VARReferee()

        # 1) 공-공 미세 접촉 검증:
        # 목적구는 (1000.0, 561.35)에 정지해 있음.
        # 수구가 (900, 500) -> (1000, 500) [간극 = 61.35 - 61.50 = -0.15mm <= 0mm] 통과 후
        # y 방향으로 살짝 꺾여 (1100, 501.2)로 진행 (굴절각 약 0.68도 > 0.35도)
        cue_traj = [
            (900.0, 500.0),
            (1000.0, 500.0),
            (1100.0, 501.2),
        ]
        target_traj = [(1000.0, 561.35)]

        micro_res = var_ref.evaluate_micro_contact(cue_traj, target_traj)
        self.assertTrue(micro_res["var_triggered"])
        self.assertTrue(micro_res["is_contact"])
        self.assertEqual(micro_res["status"], "CONTACT_CONFIRMED")
        self.assertLessEqual(float(micro_res["gap_mm"]), 0.0)
        self.assertGreater(float(micro_res["deflection_deg"]), 0.35)

        # 2) 쿠션 이중 터치(Double Touch) 검증:
        # 0ms ~ 100ms 동안 수구와 쿠션 거리가 32.0mm 이하에서
        # t=20ms (극솟점 1: 29.8mm) -> t=45ms (살짝 뜸: 31.2mm) -> t=70ms (극솟점 2: 30.1mm) 발생
        # 두 극솟점 간격 = 70ms - 20ms = 50ms (<= 90ms) 이므로 DOUBLE_TOUCH_FOUL 판정!
        distances_mm = [35.0, 29.8, 31.2, 30.1, 36.0]
        timestamps_ms = [0.0, 20.0, 45.0, 70.0, 100.0]

        dt_res = var_ref.evaluate_cushion_double_touch(distances_mm, timestamps_ms)
        self.assertTrue(dt_res["var_triggered"])
        self.assertTrue(dt_res["is_double_touch"])
        self.assertEqual(dt_res["status"], "DOUBLE_TOUCH_FOUL")
        self.assertEqual(dt_res["minima_count"], 2)
        self.assertAlmostEqual(float(dt_res["minima_interval_ms"]), 50.0, places=1)

        # ScoringEngine과 연동하여 더블터치 파울 시 득점이 취소되는지 최종 확인
        engine = ScoringEngine()
        engine.reset_shot(cue_ball_id=1, object_ball_ids=[2, 3])
        foul_eval = engine.evaluate_shot(has_double_touch_foul=bool(dt_res["is_double_touch"]))
        self.assertFalse(foul_eval["is_scored"])
        self.assertEqual(foul_eval["shot_type"], "FOUL")
        self.assertEqual(foul_eval["reason_code"], "DOUBLE_TOUCH_FOUL")


if __name__ == "__main__":
    unittest.main(verbosity=2)
