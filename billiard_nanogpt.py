"""
Physics-Informed Residual NanoGPT (billiard_nanogpt.py)
카파시 nanoGPT 아키텍처에 당구 쿠션 반사 물리 Prior를 결합한 하이브리드 궤적 복원 트랜스포머
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F

TABLE_W = 2840.0
TABLE_H = 1420.0
BALL_R = 30.75
MAX_SPEED = 6000.0
DT = 1.0 / 60.0


@dataclass
class BilliardGPTConfig:
    block_size: int = 15
    state_dim: int = 8    # [x_n, y_n, vx_n, vy_n, dL_n, dR_n, dT_n, dB_n]
    out_dim: int = 4      # [dx_n, dy_n, next_vx_n, next_vy_n]
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0
    bias: bool = False
    restitution: float = 0.82
    friction_decay: float = 0.992


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
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
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
    카파시 nanoGPT 아키텍처에 당구 쿠션 반사 물리 Prior를 결합한 하이브리드 궤적 복원 트랜스포머
    """
    def __init__(self, config: Optional[BilliardGPTConfig] = None):
        super().__init__()
        self.config = config or BilliardGPTConfig()

        self.transformer = nn.ModuleDict(dict(
            state_emb=nn.Sequential(
                nn.Linear(self.config.state_dim, self.config.n_embd, bias=True),
                nn.GELU(),
                nn.Linear(self.config.n_embd, self.config.n_embd, bias=self.config.bias),
            ),
            wpe=nn.Embedding(self.config.block_size, self.config.n_embd),
            drop=nn.Dropout(self.config.dropout),
            h=nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),
            ln_f=nn.LayerNorm(self.config.n_embd, bias=self.config.bias),
        ))
        self.physics_head = nn.Linear(self.config.n_embd, self.config.out_dim, bias=True)
        # 초기 상태에서 물리 Prior를 해치지 않도록 출력 헤드를 미세 가중치로 초기화
        torch.nn.init.zeros_(self.physics_head.bias)
        torch.nn.init.normal_(self.physics_head.weight, mean=0.0, std=1e-4)

    @staticmethod
    def build_8d_features(state_4d: torch.Tensor) -> torch.Tensor:
        x, y, vx, vy = state_4d[..., 0:1], state_4d[..., 1:2], state_4d[..., 2:3], state_4d[..., 3:4]
        x_n, y_n = x / TABLE_W, y / TABLE_H
        vx_n, vy_n = vx / MAX_SPEED, vy / MAX_SPEED
        d_left = (x - BALL_R) / TABLE_W
        d_right = (TABLE_W - BALL_R - x) / TABLE_W
        d_top = (y - BALL_R) / TABLE_H
        d_bottom = (TABLE_H - BALL_R - y) / TABLE_H
        return torch.cat([x_n, y_n, vx_n, vy_n, d_left, d_right, d_top, d_bottom], dim=-1)

    def _compute_physics_prior(self, states_4d: torch.Tensor) -> torch.Tensor:
        """단일 스텝 쿠션 탄성 반사 및 마찰 감쇠 물리 Prior 계산"""
        x = states_4d[..., 0:1]
        y = states_4d[..., 1:2]
        vx = states_4d[..., 2:3]
        vy = states_4d[..., 3:4]

        e = self.config.restitution
        decay = self.config.friction_decay

        nx = x + vx * DT
        ny = y + vy * DT
        nvx = vx * decay
        nvy = vy * decay

        # 좌/우 쿠션 반사
        hit_left = nx < BALL_R
        nx = torch.where(hit_left, BALL_R + (BALL_R - nx) * e, nx)
        nvx = torch.where(hit_left, -nvx * e, nvx)

        hit_right = nx > (TABLE_W - BALL_R)
        nx = torch.where(hit_right, (TABLE_W - BALL_R) - (nx - (TABLE_W - BALL_R)) * e, nx)
        nvx = torch.where(hit_right, -nvx * e, nvx)

        # 상/하 쿠션 반사
        hit_top = ny < BALL_R
        ny = torch.where(hit_top, BALL_R + (BALL_R - ny) * e, ny)
        nvy = torch.where(hit_top, -nvy * e, nvy)

        hit_bottom = ny > (TABLE_H - BALL_R)
        ny = torch.where(hit_bottom, (TABLE_H - BALL_R) - (ny - (TABLE_H - BALL_R)) * e, ny)
        nvy = torch.where(hit_bottom, -nvy * e, nvy)

        return torch.cat([nx, ny, nvx, nvy], dim=-1)

    def forward(
        self,
        states_4d: torch.Tensor,
        target_next_4d: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = states_4d.device
        b, t, _ = states_4d.size()
        assert t <= self.config.block_size

        feat_8d = self.build_8d_features(states_4d)
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        x = self.transformer.drop(self.transformer.state_emb(feat_8d) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # 트랜스포머가 예측한 비선형 스핀/마찰 잔차(Residual)
        res_norm = self.physics_head(x)
        res_4d = torch.cat([
            res_norm[..., 0:1] * TABLE_W,
            res_norm[..., 1:2] * TABLE_H,
            res_norm[..., 2:3] * MAX_SPEED,
            res_norm[..., 3:4] * MAX_SPEED,
        ], dim=-1)

        prior_next_4d = self._compute_physics_prior(states_4d)
        pred_next_4d = prior_next_4d + res_4d

        loss = None
        if target_next_4d is not None:
            loss = F.smooth_l1_loss(pred_next_4d, target_next_4d, beta=2.0)

        return pred_next_4d, loss

    @torch.no_grad()
    def rollout_occlusion(self, history_15_frames: torch.Tensor, n_missing_frames: int = 1) -> torch.Tensor:
        """
        가려지기 직전 최대 15프레임의 (x, y, vx, vy) 텐서를 입력받아
        가려진 n_missing_frames 동안의 쿠션 반사 포함 궤적을 자기회귀로 복원합니다.
        """
        self.eval()
        seq = history_15_frames.clone()
        restored = []

        for _ in range(n_missing_frames):
            seq_cond = seq[:, -self.config.block_size:, :]
            pred_seq, _ = self(seq_cond)
            next_state = pred_seq[:, -1:, :]
            next_state[..., 0].clamp_(BALL_R, TABLE_W - BALL_R)
            next_state[..., 1].clamp_(BALL_R, TABLE_H - BALL_R)
            restored.append(next_state)
            seq = torch.cat([seq, next_state], dim=1)

        return torch.cat(restored, dim=1)
