"""
Retraining scorer (Section 3.5).

Architecture:
    encoder = PoolMLPEncoder(window)
        Temporal pool of standardized state vectors over the recent N steps
        into  p_t = [AvgPool; MaxPool; most-recent]  in R^(3*C),
        followed by a single-hidden-layer MLP -> embedding in R^E.
    head = Linear(E, 1)
        Maps the embedding to the scalar retraining score s_t.

Training: Huber loss on the chosen pseudo-label g_t (Equations 21-22).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ScorerConfig:
    n_state_channels: int = 5            # paper default: 5-channel state
    hidden_dim: int = 64
    embedding_dim: int = 64
    dropout: float = 0.1
    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4


class PoolMLPEncoder(nn.Module):
    """
    Temporal pool of the state window followed by a single-hidden-layer MLP
    (Section 3.5, Equation 19).
    """

    def __init__(self, config: ScorerConfig):
        super().__init__()
        in_dim = config.n_state_channels * 3  # mean + max + most-recent
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, window_len, n_state_channels) -> (B, embedding_dim)"""
        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        last_step = x[:, -1, :]
        pooled = torch.cat([mean_pool, max_pool, last_step], dim=-1)
        return self.mlp(pooled)


class UtilityPredictor(nn.Module):
    """
    Single-window scorer: window of standardized state vectors -> scalar
    retraining score s_t (Equation 23).
    """

    def __init__(self, config: ScorerConfig):
        super().__init__()
        self.encoder = PoolMLPEncoder(config)
        self.head = nn.Linear(config.embedding_dim, 1)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """window: (B, window_len, n_state_channels) -> (B,) scalar score."""
        return self.head(self.encoder(window)).squeeze(-1)

    def predict(self, window: np.ndarray, device: torch.device) -> float:
        self.eval()
        w_t = torch.tensor(window[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            return float(self.forward(w_t).item())


def train_utility_predictor(
    windows: np.ndarray,            # (N, window_len, n_state_channels)
    gains: np.ndarray,              # (N,) float32 pseudo-labels
    config: ScorerConfig,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    val_split: float = 0.1,
    patience: int = 7,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    oversample_positive: bool = True,
    delta_huber: float = 0.05,
) -> "UtilityPredictor":
    """
    Train the scorer end-to-end with Huber loss on the supplied pseudo-labels.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UtilityPredictor(config).to(device)

    if len(windows) == 0:
        return model.eval()

    w_tensor = torch.tensor(windows, dtype=torch.float32)
    g_tensor = torch.tensor(gains, dtype=torch.float32)
    if oversample_positive:
        pos = (g_tensor > 0).numpy()
        if pos.any() and (~pos).any():
            n_neg = int((~pos).sum())
            pos_idx = np.where(pos)[0]
            if len(pos_idx) > 0 and len(pos_idx) < n_neg:
                rep = int(np.ceil(n_neg / max(1, len(pos_idx))))
                pos_extra = np.tile(pos_idx, rep)[: n_neg - len(pos_idx)]
                idx = np.concatenate([np.arange(len(w_tensor)), pos_extra])
                w_tensor = w_tensor[idx]
                g_tensor = g_tensor[idx]

    n = len(w_tensor)
    n_val = max(1, int(round(n * val_split)))
    perm = torch.randperm(n)
    tr, va = perm[:-n_val], perm[-n_val:]
    train_ds = TensorDataset(w_tensor[tr], g_tensor[tr])
    val_ds = TensorDataset(w_tensor[va], g_tensor[va])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.HuberLoss(delta=delta_huber)

    best_val, best_state, since = math.inf, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                val_loss += float(loss_fn(model(xb), yb).item()) * len(xb)
        val_loss /= max(1, len(val_ds))
        if verbose:
            print(f"  [scorer] epoch {epoch:3d}  val_loss={val_loss:.4f}")
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since = 0
        else:
            since += 1
            if since >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()
