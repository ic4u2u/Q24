"""
단일 카메라 비전 기반(Vision-Only, 60fps) 대대 3쿠션 자동 득점 판정 코어 패키지
파일 경로: three_cushion_vision/__init__.py

오디오(마이크) 센서 없이 순수 비전 기하학 및 물리 운동학(Sub-Frame CCD, 궤적 굴절각,
목적구 미세 충격 변위, 법선 반전 쿠션 판정, 결정론적 4단계 FSM)만으로
0.5mm 미세 스침(키스)과 뱅크샷(가락), 조기 키스 파울 잠금을 완벽히 판별합니다.
"""

from three_cushion_vision.billiard_nanogpt import (
    BilliardGPTConfig,
    BilliardNanoGPT,
    OcclusionTrajectoryRestorer,
    get_or_train_billiard_nanogpt,
)
from three_cushion_vision.config import (
    BallColor,
    FailReason,
    Rail,
    ShotState,
    VisionConfig,
)
from three_cushion_vision.demo_visualizer import (
    BilliardDemoVisualizer,
    UltralyticsOCSortDetector,
    run_video_demo,
)
from three_cushion_vision.homography import TableHomography
from three_cushion_vision.kinematics_detector import (
    ConfirmedContactEvent,
    CushionImpactEvent,
    KinematicsDetector,
    SubFrameCCDResult,
)
from three_cushion_vision.scoreboard_bridge import ScoreboardBridge
from three_cushion_vision.scoring_fsm import ShotResult, ThreeCushionScoringFSM
from three_cushion_vision.tracker_adapter import OCSortTrackerAdapter, TrackedBall

__all__ = [
    "BallColor",
    "FailReason",
    "Rail",
    "ShotState",
    "VisionConfig",
    "TableHomography",
    "TrackedBall",
    "OCSortTrackerAdapter",
    "SubFrameCCDResult",
    "ConfirmedContactEvent",
    "CushionImpactEvent",
    "KinematicsDetector",
    "ShotResult",
    "ThreeCushionScoringFSM",
    "ScoreboardBridge",
    "UltralyticsOCSortDetector",
    "BilliardDemoVisualizer",
    "run_video_demo",
    "BilliardGPTConfig",
    "BilliardNanoGPT",
    "OcclusionTrajectoryRestorer",
    "get_or_train_billiard_nanogpt",
]
