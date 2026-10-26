"""
단일 중앙 AI/물리 코어 엔진 기반 다중 당구대(Multi-Table) 독립 세션 관제 허브
파일 경로: cue24/services/multi_table_hub.py

[핵심 아키텍처]
1. Stateless Shared Core (단일 코어 엔진 공유):
   - 무거운 딥러닝 모델(BilliardNanoGPT 가중치), 물리 검증기(billiard_mcp_server), OpenCV 커널은
     서버 메모리에 단 1개만 로드하여 모든 당구대가 공유합니다 (메모리 중복 0%).
2. Per-Table Isolated Context (테이블별 상태 완벽 격리):
   - 각 당구대(TABLE_1, TABLE_2, TABLE_3, TABLE_4 ...)는 고유한
     MatchManager(점수·이닝·40초 타이머), ConnectionManager(테이블 전용 WebSocket 방),
     LiveStreamReader(카메라/영상 캡처·칼만 필터·CausalKissSafeFSM 궤적 상태),
     그리고 테이블 전용 비동기 프레임 락(asyncio.Lock)을 독립 보유합니다.
   - 따라서 1번 테이블과 2번 테이블이 동시에 각기 다른 카메라/영상을 재생하고 채점해도
     점수, 타이머, 공 좌표, 리플레이 버퍼가 절대 섞이거나 프레임 병목이 발생하지 않습니다.
"""

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from cue24.services.match_manager import MatchManager


@dataclass
class TableEngineContext:
    """개별 당구대(Table) 1대의 독립된 경기·비전·웹소켓 실행 컨텍스트"""

    table_id: str
    table_number: int
    table_label: str
    ws_manager: Any
    match_manager: MatchManager
    stream_reader: Any
    frame_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    timer_task: Optional[asyncio.Task] = None

    def ensure_timer_running(self) -> None:
        """현재 이벤트 루프가 실행 중이면 해당 테이블의 1초 주기 타이머 루프를 자동 기동합니다."""
        if self.timer_task is not None and not self.timer_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
            self.timer_task = loop.create_task(self.stream_reader.run_timer_loop())
        except RuntimeError:
            pass

    def to_summary_dict(self) -> Dict[str, Any]:
        """매장 전체 테이블 현황판 및 스위처 UI용 요약 정보를 반환합니다."""
        p1 = self.match_manager.players["P1"]
        p2 = self.match_manager.players["P2"]
        v_src = str(getattr(self.stream_reader, "video_path", "") or "")
        return {
            "table_id": self.table_id,
            "table_number": self.table_number,
            "table_label": self.table_label,
            "fsm_state": str(self.match_manager.fsm_state.value),
            "current_inning": self.match_manager.current_inning,
            "active_player_id": self.match_manager.active_player_id,
            "rule_mode": self.match_manager.rule_mode,
            "is_match_paused": self.match_manager.is_match_paused,
            "p1_name": p1.name,
            "p1_score": p1.score,
            "p2_name": p2.name,
            "p2_score": p2.score,
            "video_source": Path(v_src).name if v_src else "LIVE_CAM",
            "connected_tablets": len(getattr(self.ws_manager, "active_connections", [])),
        }


class MultiTableEngineHub:
    """
    하나의 중앙 비전·물리 엔진으로 여러 대의 당구대(TABLE_1 ~ TABLE_N)를
    간섭 없이 동시 제어하는 멀티 테이블 세션 레지스트리 허브입니다.
    """

    def __init__(
        self,
        ws_factory: Callable[[], Any],
        reader_factory: Callable[[MatchManager, Any], Any],
    ) -> None:
        self._ws_factory = ws_factory
        self._reader_factory = reader_factory
        self._tables: Dict[str, TableEngineContext] = {}

    @staticmethod
    def normalize_table_id(raw_table_id: Optional[str] = None) -> Tuple[str, int]:
        """
        입력된 테이블 식별자(예: None, '1', 'table_2', 'TABLE_3')를
        표준 키('TABLE_1', 1) 형태로 정규화합니다.
        """
        if not raw_table_id:
            return ("TABLE_1", 1)
        cleaned = str(raw_table_id).strip().upper()
        match = re.search(r"(\d+)", cleaned)
        if match:
            num = max(1, min(99, int(match.group(1))))
            return (f"TABLE_{num}", num)
        return ("TABLE_1", 1)

    def register_primary_table(
        self,
        ws_manager: Any,
        match_manager: MatchManager,
        stream_reader: Any,
    ) -> TableEngineContext:
        """기본 1번 당구대(TABLE_1)를 기존 싱글톤 객체와 1:1 바인딩하여 하위 호환성을 100% 유지합니다."""
        ctx = TableEngineContext(
            table_id="TABLE_1",
            table_number=1,
            table_label="1번 테이블",
            ws_manager=ws_manager,
            match_manager=match_manager,
            stream_reader=stream_reader,
        )
        self._tables["TABLE_1"] = ctx
        return ctx

    def get_table(self, table_id: Optional[str] = None) -> TableEngineContext:
        """
        요청된 table_id의 독립 컨텍스트를 반환하며, 아직 생성되지 않은 테이블이면
        공유 AI 가중치(BilliardNanoGPT 캐시)를 재사용하여 즉시 경량 컨텍스트를 생성합니다.
        """
        norm_id, num = self.normalize_table_id(table_id)
        ctx = self._tables.get(norm_id)
        if ctx is None:
            ws_mgr = self._ws_factory()
            mm = MatchManager()
            reader = self._reader_factory(mm, ws_mgr)
            ctx = TableEngineContext(
                table_id=norm_id,
                table_number=num,
                table_label=f"{num}번 테이블",
                ws_manager=ws_mgr,
                match_manager=mm,
                stream_reader=reader,
            )
            self._tables[norm_id] = ctx
        ctx.ensure_timer_running()
        return ctx

    def start_all_timers(self) -> None:
        """현재 등록된 모든 당구대의 백그라운드 타이머 루프를 기동합니다."""
        for ctx in self._tables.values():
            ctx.ensure_timer_running()

    def stop_all_timers(self) -> None:
        """서버 종료 시 모든 당구대의 타이머 루프와 비디오 캡처를 안전하게 정리합니다."""
        for ctx in self._tables.values():
            ctx.stream_reader.is_running = False
            if ctx.timer_task is not None and not ctx.timer_task.done():
                ctx.timer_task.cancel()

    def list_tables_summary(self, min_tables: int = 4) -> List[Dict[str, Any]]:
        """
        매장 내 기본 테이블(1번 ~ min_tables번) 및 추가 활성화된 테이블들의 상태 목록을 반환합니다.
        """
        result: List[Dict[str, Any]] = []
        known_nums = {ctx.table_number for ctx in self._tables.values()}
        all_nums = sorted(known_nums.union(range(1, min_tables + 1)))
        for num in all_nums:
            tid = f"TABLE_{num}"
            if tid in self._tables:
                result.append(self._tables[tid].to_summary_dict())
            else:
                result.append(
                    {
                        "table_id": tid,
                        "table_number": num,
                        "table_label": f"{num}번 테이블",
                        "fsm_state": "WAIT_SHOT",
                        "current_inning": 1,
                        "active_player_id": "P1",
                        "rule_mode": "STANDARD",
                        "is_match_paused": False,
                        "p1_name": "1번 선수",
                        "p1_score": 0,
                        "p2_name": "2번 선수",
                        "p2_score": 0,
                        "video_source": Path(
                            getattr(
                                self._tables.get("TABLE_1", None),
                                "stream_reader",
                                None,
                            ).video_path
                            if "TABLE_1" in self._tables
                            else "99613_31738_261025150244.mp4"
                        ).name,
                        "connected_tablets": 0,
                    }
                )
        return result
