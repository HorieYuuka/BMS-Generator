"""TokenSelectionModel and LaneAssignmentModel.

The model architecture is fixed by design. Forward signatures
match a fixed inference contract, §21.4 /
§21.5 so exported TorchScript models plug directly into placement_engine.py.
"""

from __future__ import annotations

import torch
from torch import nn


class TokenSelectionModel(nn.Module):
    """Siamese MLP scoring each pool token for placement.

    Forward inputs:
      measure_tensor  shape (4,)    — [measure_index, density_rank, phase_encoded, candidate_count]
      pool_tensor     shape (P, 14) — 14 pool feature columns in inference order
      context_tensor  shape (4, 3)  — 4 preceding eligible measures × [tkey_delta, placed_count, mean_attack_rms]
    Returns:
      scores          shape (P,)    — higher is more preferred
    """

    def __init__(self, hidden: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        in_dim = 4 + 14 + 12
        self.norm = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        measure_tensor: torch.Tensor,
        pool_tensor: torch.Tensor,
        context_tensor: torch.Tensor,
    ) -> torch.Tensor:
        P = pool_tensor.shape[0]
        measure_row = measure_tensor.reshape(1, 4).expand(P, 4)
        context_flat = context_tensor.reshape(1, 12).expand(P, 12)
        x = torch.cat([measure_row, pool_tensor, context_flat], dim=-1)
        x = self.norm(x)
        out = self.net(x)
        return out.squeeze(-1)


class TokenSelectionFlat(nn.Module):
    """Training-time wrapper that accepts pre-flattened batched inputs.

    The exported TorchScript model uses `TokenSelectionModel` with its
    per-measure forward signature; this wrapper shares the same underlying
    Sequential so training can batch multiple measures concatenated along
    the row dimension without going through the expand/broadcast path.
    """

    def __init__(self, core: TokenSelectionModel) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        measure_rows: torch.Tensor,   # (sum_P, 4)
        pool_rows: torch.Tensor,      # (sum_P, 14)
        context_rows: torch.Tensor,   # (sum_P, 12)
    ) -> torch.Tensor:
        x = torch.cat([measure_rows, pool_rows, context_rows], dim=-1)
        x = self.core.norm(x)
        out = self.core.net(x)
        return out.squeeze(-1)


class LaneAssignmentModel(nn.Module):
    """Direct 7-way MLP with masked logits.

    Forward inputs:
      event_tensor    shape (16,)   — 10 basic + 6 spectral features
      context_tensor  shape (8, 5)  — 8 preceding events × [tkey_delta, lane_index, attack_rms, idx192, is_padded]
      mask_tensor     shape (7,)    — 1.0 for available lanes, 0.0 otherwise
    Returns:
      logits          shape (7,)    — masked lanes receive -inf
    """

    def __init__(self, hidden: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        in_dim = 16 + 40
        self.norm = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 7),
        )
        self.register_buffer("_neg_inf", torch.tensor(float("-inf")))

    def forward(
        self,
        event_tensor: torch.Tensor,
        context_tensor: torch.Tensor,
        mask_tensor: torch.Tensor,
    ) -> torch.Tensor:
        event = event_tensor.reshape(1, 16) if event_tensor.dim() == 1 else event_tensor
        ctx = context_tensor.reshape(event.shape[0], 40)
        mask = mask_tensor.reshape(event.shape[0], 7)
        x = torch.cat([event, ctx], dim=-1)
        x = self.norm(x)
        logits = self.net(x)
        logits = torch.where(mask > 0, logits, torch.full_like(logits, float("-inf")))
        if event_tensor.dim() == 1:
            return logits.squeeze(0)
        return logits
