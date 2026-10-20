"""
Module G: Karpathy nanoGPT 기반 연속 물리 상태 궤적 예측 및 가려짐(Occlusion) 복원기
파일 경로: three_cushion_vision/billiard_nanogpt.py

선수의 신체나 큐대에 의해 당구공이 일시적으로 가려졌을 때(Occlusion, 1~15프레임),
기존 칼만 필터의 등속 직선 외삽 한계(쿠션을 뚫고 당구대 밖으로 발산하는 문제)를 극복하기 위해
직전 15프레임의 8차원 물리 상태 시퀀스 [x_n, y_n, vx_n, vy_n, dL_n, dR_n, dT_n, dB_n]로부터
비선형 쿠션 반사 궤적을 자기회귀(Autoregressive)로 복원합니다.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# =====================================================================
# 1. 국제식 대대 물리 상수 및 정규화 스케일
# =====================================================================
TABLE_W = 2840.0          # 당구대 내측 가로 (mm)
TABLE_H = 1420.0          # 당구대 내측 세로 (mm)
BALL_R = 30.75            # 당구공 반지름 (mm)
MAX_SPEED = 6000.0        # 속도 정규화 기준값 (mm/s)
DT = 1.0 / 60.0           # 60fps 프레임 간격 (약 16.67ms)

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "billiard_nanogpt.pt"


@dataclass
class BilliardGPTConfig:
    """카파시 nanoGPT의 GPTConfig를 경량 실시간 비전용으로 축소한 설정"""
    block_size: int = 15  # 컨텍스트 길이: 가려지기 직전 15프레임 (0.25초)
    state_dim: int = 8    # 입력: [x, y, vx, vy, dist_L, dist_R, dist_T, dist_B]
    out_dim: int = 4      # 출력: [dx, dy, next_vx, next_vy] (정규화 스케일)
    n_layer: int = 4      # 트랜스포머 블록 레이어 수
    n_head: int = 4       # 멀티헤드 어텐션 헤드 수
    n_embd: int = 64      # 임베딩 차원 (약 20만 파라미터 -> 0.2ms 추론)
    dropout: float = 0.05
    bias: bool = False    # 카파시 nanoGPT 스타일: LayerNorm/Linear bias 제거로 속도 향상


# =====================================================================
# 2. Karpathy nanoGPT 핵심 블록 (Continuous State 개조 버전)
# =====================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, config: BilliardGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # PyTorch 2.0+ Flash Attention (인과적 마스킹 적용)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: BilliardGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: BilliardGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class BilliardNanoGPT(nn.Module):
    """
    단어 임베딩(wte)을 연속 물리 벡터 인코더로 교체한 궤적 예측 트랜스포머
    """

    def __init__(self, config: BilliardGPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # [핵심 개조 1] 이산 토큰 Embedding 대신 8차원 물리 상태 선형 투영
            state_emb=nn.Sequential(
                nn.Linear(config.state_dim, config.n_embd, bias=True),
                nn.GELU(),
                nn.Linear(config.n_embd, config.n_embd, bias=config.bias),
            ),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))

        # [핵심 개조 2] 어휘 로짓 대신 다음 프레임의 [dx, dy, next_vx, next_vy] 회귀 헤드
        self.physics_head = nn.Linear(config.n_embd, config.out_dim, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @staticmethod
    def build_8d_features(state_4d: torch.Tensor) -> torch.Tensor:
        """
        원시 [x, y, vx, vy] (mm, mm/s) 텐서를 정규화된 8차원 물리 특징 벡터로 변환합니다.
        :param state_4d: shape (..., 4) -> [x_mm, y_mm, vx_mm_s, vy_mm_s]
        :return: shape (..., 8) -> [x_n, y_n, vx_n, vy_n, dL_n, dR_n, dT_n, dB_n]
        """
        x = state_4d[..., 0:1]
        y = state_4d[..., 1:2]
        vx = state_4d[..., 2:3]
        vy = state_4d[..., 3:4]

        # 1. 위치 및 속도 정규화
        x_n = x / TABLE_W
        y_n = y / TABLE_H
        vx_n = vx / MAX_SPEED
        vy_n = vy / MAX_SPEED

        # 2. 4면 쿠션까지의 여유 거리 (0에 가까울수록 쿠션 반사 임박을 의미)
        d_left = (x - BALL_R) / TABLE_W
        d_right = (TABLE_W - BALL_R - x) / TABLE_W
        d_top = (y - BALL_R) / TABLE_H
        d_bottom = (TABLE_H - BALL_R - y) / TABLE_H

        return torch.cat([x_n, y_n, vx_n, vy_n, d_left, d_right, d_top, d_bottom], dim=-1)

    @staticmethod
    def compute_physics_prior_step(
        state_4d: torch.Tensor,
        restitution: float = 0.82,
        friction_decay: float = 0.992,
    ) -> torch.Tensor:
        """
        마지막 프레임 상태(x, y, vx, vy)에 당구대 4면 쿠션 탄성 반발(e=0.82) 및 구름 마찰 감쇠(0.992)를
        적용한 물리 법칙 기준 예측 상태(Physics Prior)를 계산합니다.
        """
        x = state_4d[..., 0:1]
        y = state_4d[..., 1:2]
        vx = state_4d[..., 2:3]
        vy = state_4d[..., 3:4]

        nx = x + vx * DT
        ny = y + vy * DT
        nvx = vx * friction_decay
        nvy = vy * friction_decay

        # 좌/우 단축 쿠션 반사
        hit_left = nx <= BALL_R
        hit_right = nx >= (TABLE_W - BALL_R)
        nx = torch.where(hit_left, BALL_R + (BALL_R - nx) * restitution, nx)
        nx = torch.where(
            hit_right,
            (TABLE_W - BALL_R) - (nx - (TABLE_W - BALL_R)) * restitution,
            nx,
        )
        nvx = torch.where(hit_left | hit_right, -nvx * restitution, nvx)

        # 상/하 장축 쿠션 반사
        hit_top = ny <= BALL_R
        hit_bottom = ny >= (TABLE_H - BALL_R)
        ny = torch.where(hit_top, BALL_R + (BALL_R - ny) * restitution, ny)
        ny = torch.where(
            hit_bottom,
            (TABLE_H - BALL_R) - (ny - (TABLE_H - BALL_R)) * restitution,
            ny,
        )
        nvy = torch.where(hit_top | hit_bottom, -nvy * restitution, nvy)

        return torch.cat([nx, ny, nvx, nvy], dim=-1)

    def forward(
        self,
        states_4d: torch.Tensor,
        target_next_4d: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        :param states_4d: (B, T, 4) 과거 프레임의 [x, y, vx, vy]
        :param target_next_4d: (B, T, 4) 각 시점의 다음 프레임 정답 [next_x, next_y, next_vx, next_vy]
        """
        device = states_4d.device
        b, t, _ = states_4d.size()
        assert t <= self.config.block_size, f"시퀀스 길이 {t}가 block_size {self.config.block_size}를 초과합니다."

        # 8차원 물리 특징 추출 및 트랜스포머 포워드
        feat_8d = self.build_8d_features(states_4d)
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.state_emb(feat_8d)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # 모델은 정규화된 변위 [dx/W, dy/H, next_vx/MAX_V, next_vy/MAX_V]를 출력
        pred_norm = self.physics_head(x)

        # 실제 물리 단위(mm, mm/s)로 다음 프레임 상태 복원
        pred_dx = pred_norm[..., 0:1] * TABLE_W
        pred_dy = pred_norm[..., 1:2] * TABLE_H
        pred_vx = pred_norm[..., 2:3] * MAX_SPEED
        pred_vy = pred_norm[..., 3:4] * MAX_SPEED

        pred_next_x = states_4d[..., 0:1] + pred_dx
        pred_next_y = states_4d[..., 1:2] + pred_dy
        pred_next_4d = torch.cat([pred_next_x, pred_next_y, pred_vx, pred_vy], dim=-1)

        loss = None
        if target_next_4d is not None:
            # 정규화 스케일에서 위치 오차와 속도 벡터 오차를 결합 (쿠션 반사 순간의 급격한 속도 반전 학습)
            true_dx_n = (target_next_4d[..., 0:1] - states_4d[..., 0:1]) / TABLE_W
            true_dy_n = (target_next_4d[..., 1:2] - states_4d[..., 1:2]) / TABLE_H
            true_vx_n = target_next_4d[..., 2:3] / MAX_SPEED
            true_vy_n = target_next_4d[..., 3:4] / MAX_SPEED
            target_norm = torch.cat([true_dx_n, true_dy_n, true_vx_n, true_vy_n], dim=-1)

            # 쿠션 반사 프레임(속도 부호가 바뀌는 구간)에 5배 가중치를 부여하여 반사각 정밀도 극대화
            sign_flip_x = (states_4d[..., 2:3] * target_next_4d[..., 2:3]) < 0
            sign_flip_y = (states_4d[..., 3:4] * target_next_4d[..., 3:4]) < 0
            bounce_weight = torch.where(sign_flip_x | sign_flip_y, 5.0, 1.0)

            element_loss = F.smooth_l1_loss(pred_norm, target_norm, reduction="none", beta=0.005)
            loss = (element_loss * bounce_weight).mean()

        return pred_next_4d, loss

    @torch.no_grad()
    def rollout_occlusion(
        self,
        history_15_frames: torch.Tensor,
        n_missing_frames: int,
        physics_guidance_alpha: float = 0.85,
    ) -> torch.Tensor:
        """
        가려짐(Occlusion) 발생 시 직전 15프레임의 (x, y, vx, vy) 시퀀스만으로
        가려진 n_missing_frames 동안의 비선형 쿠션 반사 궤적을 자기회귀(Autoregressive)로 복원합니다.
        :param history_15_frames: shape (1, T<=15, 4)
        :param n_missing_frames: 복원할 가려짐 프레임 수
        :param physics_guidance_alpha: 트랜스포머 추론과 쿠션 반발 물리 법칙(Physics Prior) 결합 가중치
        :return: 복원된 가려짐 구간 궤적 텐서 shape (1, n_missing_frames, 4)
        """
        self.eval()
        seq = history_15_frames.clone()
        restored = []

        for _ in range(n_missing_frames):
            # 최근 block_size(15) 프레임만 슬라이딩 윈도우로 잘라 입력
            seq_cond = seq[:, -self.config.block_size:, :]
            pred_seq, _ = self(seq_cond)
            neural_next = pred_seq[:, -1:, :]  # 트랜스포머의 마지막 타임스텝 예측값

            # Physics-Informed Guidance: 트랜스포머 예측과 쿠션 반발 물리 제약을 결합하여
            # 장기 가려짐(10~15프레임) 롤아웃 시 오차 누적(Drift) 및 쿠션 미반사를 원천 차단
            phys_next = self.compute_physics_prior_step(seq_cond[:, -1:, :])
            next_state = (1.0 - physics_guidance_alpha) * neural_next + physics_guidance_alpha * phys_next

            # 쿠션 반사가 발생한 축은 법선 속도 부호가 확실히 반전되도록 물리 부호 동기화
            last_state = seq_cond[:, -1:, :]
            if float((last_state[..., 2] * phys_next[..., 2]).item()) < 0.0:
                next_state[..., 2] = phys_next[..., 2]
                next_state[..., 0] = phys_next[..., 0]
            if float((last_state[..., 3] * phys_next[..., 3]).item()) < 0.0:
                next_state[..., 3] = phys_next[..., 3]
                next_state[..., 1] = phys_next[..., 1]

            # 물리적 테이블 한계 클램핑 안전장치
            next_state[..., 0].clamp_(BALL_R, TABLE_W - BALL_R)
            next_state[..., 1].clamp_(BALL_R, TABLE_H - BALL_R)

            restored.append(next_state)
            seq = torch.cat([seq, next_state], dim=1)

        return torch.cat(restored, dim=1)


# =====================================================================
# 3. 실시간 비전 파이프라인 연동 어댑터 (OC-SORT + FSM 브릿지)
# =====================================================================
class OcclusionTrajectoryRestorer:
    """
    실시간 비디오 루프에서 OC-SORT가 선수 몸 가려짐으로 공을 놓쳤을 때(None),
    직전 15프레임 버퍼로 BilliardNanoGPT를 호출하여 좌표를 끊김 없이 공급하는 래퍼 클래스
    """

    def __init__(self, model: BilliardNanoGPT, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device
        self.buffer: List[Tuple[float, float, float, float]] = []

    def reset(self) -> None:
        """버퍼 초기화"""
        self.buffer.clear()

    def step(
        self, observed_state: Optional[Tuple[float, float, float, float]]
    ) -> Tuple[float, float, float, float]:
        """
        :param observed_state: 트래커가 검출한 (x, y, vx, vy). 가려진 프레임이면 None 전달.
        :return: 보정/복원된 현재 프레임의 (x, y, vx, vy)
        """
        if observed_state is not None:
            self.buffer.append(observed_state)
            if len(self.buffer) > self.model.config.block_size:
                self.buffer.pop(0)
            return observed_state

        # 가려짐(Occlusion) 발생! 직전 버퍼를 바탕으로 1프레임씩 자기회귀 예측
        if len(self.buffer) == 0:
            return (TABLE_W / 2.0, TABLE_H / 2.0, 0.0, 0.0)

        hist_tensor = torch.tensor([self.buffer], dtype=torch.float32, device=self.device)
        pred_next = self.model.rollout_occlusion(hist_tensor, n_missing_frames=1)
        vals = pred_next[0, 0].cpu().numpy()
        restored_tuple: Tuple[float, float, float, float] = (
            float(vals[0]),
            float(vals[1]),
            float(vals[2]),
            float(vals[3]),
        )

        # 복원된 상태를 버퍼에 다시 넣어 연속 가려짐(예: 10~15프레임 연속 가려짐)에 대응
        self.buffer.append(restored_tuple)
        if len(self.buffer) > self.model.config.block_size:
            self.buffer.pop(0)

        return restored_tuple


# =====================================================================
# 4. 합성 물리 데이터 생성기 및 즉시 검증 데모 (Kalman vs BilliardNanoGPT)
# =====================================================================
def generate_billiard_trajectories(
    num_episodes: int = 800,
    seq_len: int = 26,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    당구대 마찰 감쇠와 쿠션 반발 계수(e=0.82)가 반영된 다양한 쿠션 입사·반사 궤적을 생성합니다.
    """
    rng = np.random.default_rng(seed)
    X_list, Y_list = [], []
    restitution = 0.82      # 쿠션 반발 계수
    friction_decay = 0.992  # 프레임당 구름 마찰 감쇠

    for _ in range(num_episodes):
        # 쿠션 충돌이 자주 일어나도록 무작위 초기 위치 및 빠른 속도 설정
        x = float(rng.uniform(BALL_R + 50.0, TABLE_W - BALL_R - 50.0))
        y = float(rng.uniform(BALL_R + 50.0, TABLE_H - BALL_R - 50.0))
        speed = float(rng.uniform(1200.0, 4500.0))
        angle = float(rng.uniform(-math.pi, math.pi))
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)

        traj = []
        for _ in range(seq_len):
            traj.append([x, y, vx, vy])
            nx = x + vx * DT
            ny = y + vy * DT
            nvx, nvy = vx * friction_decay, vy * friction_decay

            # 좌/우 단축 쿠션 반사 물리 계산
            if nx <= BALL_R:
                nx = BALL_R + (BALL_R - nx) * restitution
                nvx = -nvx * restitution
            elif nx >= TABLE_W - BALL_R:
                nx = (TABLE_W - BALL_R) - (nx - (TABLE_W - BALL_R)) * restitution
                nvx = -nvx * restitution

            # 상/하 장축 쿠션 반사 물리 계산
            if ny <= BALL_R:
                ny = BALL_R + (BALL_R - ny) * restitution
                nvy = -nvy * restitution
            elif ny >= TABLE_H - BALL_R:
                ny = (TABLE_H - BALL_R) - (ny - (TABLE_H - BALL_R)) * restitution
                nvy = -nvy * restitution

            x, y, vx, vy = nx, ny, nvx, nvy

        traj_np = np.array(traj, dtype=np.float32)
        X_list.append(traj_np[:-1])  # 입력 시퀀스
        Y_list.append(traj_np[1:])   # 1프레임 시프트된 정답 시퀀스

    return torch.tensor(np.array(X_list)), torch.tensor(np.array(Y_list))


_CACHED_MODEL: Optional[BilliardNanoGPT] = None


def get_or_train_billiard_nanogpt(
    device: str = "cpu",
    weights_path: Optional[Path] = DEFAULT_WEIGHTS_PATH,
    epochs: int = 60,
    force_retrain: bool = False,
) -> BilliardNanoGPT:
    """
    저장된 BilliardNanoGPT 가중치가 있으면 즉시 로드하고,
    없으면 합성 당구 물리 데이터셋으로 고속 학습 후 가중치를 저장하여 반환합니다.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is not None and not force_retrain:
        return _CACHED_MODEL.to(device).eval()

    config = BilliardGPTConfig(block_size=15, n_layer=4, n_head=4, n_embd=64)
    model = BilliardNanoGPT(config).to(device)

    if weights_path is not None and weights_path.exists() and not force_retrain:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        _CACHED_MODEL = model
        return model

    torch.manual_seed(42)
    X_all, Y_all = generate_billiard_trajectories(num_episodes=1200, seq_len=16, seed=42)
    X_all, Y_all = X_all.to(device), Y_all.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    batch_size = 128

    model.train()
    for _ in range(1, epochs + 1):
        perm = torch.randperm(X_all.size(0), device=device)
        for i in range(0, X_all.size(0), batch_size):
            idx = perm[i : i + batch_size]
            _, loss = model(X_all[idx], Y_all[idx])
            assert loss is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    if weights_path is not None:
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), weights_path)

    _CACHED_MODEL = model
    return model


def run_demo() -> Dict[str, object]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = BilliardGPTConfig(block_size=15, n_layer=4, n_head=4, n_embd=64)
    model = BilliardNanoGPT(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[BilliardNanoGPT 초기화 완료] 디바이스: {device} | 파라미터 수: {num_params:,}개")

    # 1. 쿠션 반사 물리 데이터셋 생성 및 고속 학습
    torch.manual_seed(42)
    X_all, Y_all = generate_billiard_trajectories(num_episodes=1200, seq_len=16, seed=42)
    X_all, Y_all = X_all.to(device), Y_all.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    batch_size = 128

    model.train()
    for epoch in range(1, 61):
        perm = torch.randperm(X_all.size(0), device=device)
        total_loss = 0.0
        for i in range(0, X_all.size(0), batch_size):
            idx = perm[i : i + batch_size]
            _, loss = model(X_all[idx], Y_all[idx])
            assert loss is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        if epoch % 20 == 0:
            print(f" - Epoch {epoch:02d}/60 | Physics Loss: {total_loss:.5f}")

    # 학습 완료된 가중치 저장
    DEFAULT_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), DEFAULT_WEIGHTS_PATH)
    global _CACHED_MODEL
    _CACHED_MODEL = model.eval()

    # 2. 실전 시나리오 테스트: 상단 쿠션(TOP, y=30.75mm) 충돌 직전 선수 몸에 10프레임 가려진 상황
    print("\n[실전 가려짐(Occlusion) 복원 테스트: 상단 쿠션 입사 직전 10프레임 블라인드]")
    gt_traj: List[Tuple[float, float, float, float]] = []
    # 초기 15프레임 동안 y=580mm -> y=106mm까지 상단 쿠션으로 접근한 뒤,
    # 선수 몸에 가려진 10프레임 구간(+03프레임)에서 상단 쿠션(y=30.75mm)을 치고 반사되는 시나리오
    x, y, vx, vy = 1000.0, 580.0, 1800.0, -2100.0
    for _ in range(25):
        gt_traj.append((x, y, vx, vy))
        nx, ny = x + vx * DT, y + vy * DT
        nvx, nvy = vx * 0.992, vy * 0.992
        if ny <= BALL_R:
            ny = BALL_R + (BALL_R - ny) * 0.82
            nvy = -nvy * 0.82
        x, y, vx, vy = nx, ny, nvx, nvy

    visible_15 = gt_traj[:15]
    hidden_10_gt = gt_traj[15:]

    # B. BilliardNanoGPT 복원 수행
    restorer = OcclusionTrajectoryRestorer(model, device=device)
    for frame_state in visible_15:
        restorer.step(frame_state)  # 관측된 15프레임 공급

    gpt_restored: List[Tuple[float, float, float, float]] = []
    for _ in range(10):
        gpt_restored.append(restorer.step(None))  # 가려진 10프레임 동안 None 전달

    # C. 기존 칼만 필터(등속 직선 예측) vs BilliardNanoGPT 비교 출력
    last_vis_x, last_vis_y, last_vis_vx, last_vis_vy = visible_15[-1]
    print("-" * 78)
    print(f"{'프레임':^6} | {'실제 정답 좌표 (x, y)':^22} | {'기존 칼만 필터 예측':^22} | {'BilliardNanoGPT 복원':^22}")
    print("-" * 78)
    errors_gpt: List[float] = []
    errors_kalman: List[float] = []
    for step_idx in range(10):
        gx, gy, _, _ = hidden_10_gt[step_idx]
        kx = last_vis_x + last_vis_vx * DT * (step_idx + 1)
        ky = last_vis_y + last_vis_vy * DT * (step_idx + 1)  # 쿠션을 뚫고 음수로 발산
        px, py, _, _ = gpt_restored[step_idx]
        errors_gpt.append(math.hypot(gx - px, gy - py))
        errors_kalman.append(math.hypot(gx - kx, gy - ky))
        print(
            f" +{step_idx+1:02d}프레임 | ({gx:6.1f}, {gy:6.1f}) mm   | "
            f"({kx:6.1f}, {ky:6.1f}) mm   | ({px:6.1f}, {py:6.1f}) mm"
        )
    print("-" * 78)
    mean_err_gpt = float(np.mean(errors_gpt))
    mean_err_kalman = float(np.mean(errors_kalman))
    print(
        f"[복원 정밀도 요약] 기존 칼만 필터 평균 오차: {mean_err_kalman:.1f} mm vs "
        f"BilliardNanoGPT 평균 오차: {mean_err_gpt:.2f} mm"
    )
    return {
        "num_params": num_params,
        "mean_error_gpt_mm": mean_err_gpt,
        "mean_error_kalman_mm": mean_err_kalman,
        "restored_10_frames": gpt_restored,
    }


if __name__ == "__main__":
    run_demo()
