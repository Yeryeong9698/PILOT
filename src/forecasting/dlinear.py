"""
DLinear forecasting backbone for adaptive forecasting with drift detection.

DLinear: Decomposition-Linear model.
  - moving-average decomposition of lookback
  - separate linear heads for trend and seasonal components
  - final forecast = yhat_trend + yhat_seasonal

Returns a dict with:
  yhat, yhat_trend, yhat_seasonal, x_trend, x_seasonal

Input shape:  (batch, lookback, n_features)
Output shape: (batch, horizon,  n_features)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.forecasting.base import BaseForecaster


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DLinearConfig:
    lookback: int = 96          # L_f
    horizon: int = 96           # H
    n_features: int = 1         # D (set by data at fit time)
    ma_kernel: int = 25         # moving-average kernel size for decomposition
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.0
    device: Optional[str] = None


# ---------------------------------------------------------------------------
# Moving-Average Decomposition
# ---------------------------------------------------------------------------

class MovingAverageDecomposition(nn.Module):
    """
    Decompose input into trend (moving average) and seasonal (remainder).
    Pads symmetrically so output length equals input length.
    """

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        # kernel_size must be odd for symmetric padding
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        pad = kernel_size // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        self.pad = pad

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, n_features)
        returns x_trend, x_seasonal  both (batch, seq_len, n_features)
        """
        # Pool over time: transpose to (batch, n_features, seq_len) first
        x_t = x.transpose(1, 2)                  # (B, D, L)
        # Symmetric padding on time dimension
        x_padded = torch.nn.functional.pad(x_t, (self.pad, self.pad), mode="replicate")
        x_trend_t = self.avg(x_padded)            # (B, D, L)
        x_trend = x_trend_t.transpose(1, 2)       # (B, L, D)
        x_seasonal = x - x_trend
        return x_trend, x_seasonal


# ---------------------------------------------------------------------------
# DLinear Model
# ---------------------------------------------------------------------------

class DLinearModel(nn.Module):
    """
    DLinear (Decomposition-Linear) model.

    Forward returns a dict:
    {
        "yhat":           (B, H, D),
        "yhat_trend":     (B, H, D),
        "yhat_seasonal":  (B, H, D),
        "x_trend":        (B, L, D),
        "x_seasonal":     (B, L, D),
    }
    """

    def __init__(self, config: DLinearConfig):
        super().__init__()
        self.config = config
        L = config.lookback
        H = config.horizon
        D = config.n_features

        self.decomp = MovingAverageDecomposition(kernel_size=config.ma_kernel)

        # Per-feature linear mapping: L -> H independently for each feature
        # We implement this as a Conv1d with groups=D (channel-wise), kernel=1
        # Equivalent: Linear(L, H) applied per channel
        # Actually simplest: use nn.Linear(L, H) and apply to each feature dim
        # We store as weight matrices: (D, L) -> (D, H)
        self.trend_linear = nn.Linear(L, H)      # shared across features (standard DLinear)
        self.seasonal_linear = nn.Linear(L, H)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (batch, L, D)
        """
        x_trend, x_seasonal = self.decomp(x)     # both (B, L, D)

        # Apply linear along time (L) dimension for each feature
        # Transpose to (B, D, L), apply linear over last dim, transpose back
        xt = x_trend.transpose(1, 2)             # (B, D, L)
        xs = x_seasonal.transpose(1, 2)           # (B, D, L)

        yhat_trend = self.trend_linear(xt)        # (B, D, H)
        yhat_seasonal = self.seasonal_linear(xs)  # (B, D, H)

        yhat_trend = yhat_trend.transpose(1, 2)   # (B, H, D)
        yhat_seasonal = yhat_seasonal.transpose(1, 2)  # (B, H, D)

        yhat = yhat_trend + yhat_seasonal         # (B, H, D)

        return {
            "yhat": yhat,
            "yhat_trend": yhat_trend,
            "yhat_seasonal": yhat_seasonal,
            "x_trend": x_trend,
            "x_seasonal": x_seasonal,
        }


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def _build_sliding_windows(
    data: np.ndarray,
    lookback: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    data: (T, D) float32
    Returns X: (N, lookback, D), Y: (N, horizon, D)
    """
    T, D = data.shape
    N = T - lookback - horizon + 1
    if N <= 0:
        raise ValueError(
            f"Data length {T} too short for lookback={lookback} + horizon={horizon}"
        )
    X = np.zeros((N, lookback, D), dtype=np.float32)
    Y = np.zeros((N, horizon, D), dtype=np.float32)
    for i in range(N):
        X[i] = data[i: i + lookback]
        Y[i] = data[i + lookback: i + lookback + horizon]
    return X, Y


# ---------------------------------------------------------------------------
# DLinear Forecaster Wrapper
# ---------------------------------------------------------------------------

class DLinearForecaster(BaseForecaster):
    """
    High-level wrapper around DLinearModel.

    Usage:
        cfg = DLinearConfig(lookback=96, horizon=96, n_features=D)
        fcst = DLinearForecaster(cfg)
        fcst.fit(train_array)          # train_array: (T, D) float32
        out = fcst.predict(x_window)   # x_window: (lookback, D) → dict of (H, D) arrays
        fcst.retrain(new_data)         # fine-tune / full retrain on new data window
    """

    def __init__(self, config: DLinearConfig):
        self._config = config
        self.device = torch.device(
            config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._build_model()
        self._is_fitted = False
        self.train_mean: Optional[np.ndarray] = None
        self.train_std: Optional[np.ndarray] = None

    @property
    def config(self):
        return self._config

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @is_fitted.setter
    def is_fitted(self, value: bool):
        self._is_fitted = value

    def _build_model(self):
        self.model = DLinearModel(self.config).to(self.device)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        return (data - self.train_mean) / (self.train_std + 1e-8)

    def _denormalize(self, data: np.ndarray) -> np.ndarray:
        return data * (self.train_std + 1e-8) + self.train_mean

    def fit(
        self,
        data: np.ndarray,
        verbose: bool = False,
        **_kw,  # ignore TSLib-style kwargs (val_data, time_features, …)
    ) -> "DLinearForecaster":
        """Fit on (T, D) float32 data; normalization stats are fit on data."""
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]

        # Compute and store normalization stats
        self.train_mean = data.mean(axis=0, keepdims=True)  # (1, D)
        self.train_std = data.std(axis=0, keepdims=True)    # (1, D)
        self.train_std = np.where(self.train_std < 1e-8, 1.0, self.train_std)

        # Update n_features in config and rebuild model if needed
        D = data.shape[1]
        if D != self.config.n_features:
            self.config.n_features = D
            self._build_model()

        data_norm = self._normalize(data)
        X, Y = _build_sliding_windows(data_norm, self.config.lookback, self.config.horizon)

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.config.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out["yhat"], yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)
            if verbose:
                avg = total_loss / max(1, len(dataset))
                print(f"  Epoch {epoch+1}/{self.config.epochs}  loss={avg:.6f}")

        self.model.eval()
        self.is_fitted = True
        return self

    def predict_batch(
        self,
        windows: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Predict on a batch of lookback windows.
        windows: (N, lookback, D)
        Returns dict of numpy arrays.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_batch()")
        windows = np.asarray(windows, dtype=np.float32)
        windows_norm = self._normalize(windows)  # broadcast over N
        x_t = torch.tensor(windows_norm, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(x_t)
        result = {}
        for k, v in out.items():
            arr = v.cpu().numpy()
            # Denormalize forecast outputs (not internal decompositions of input)
            if k in ("yhat", "yhat_trend", "yhat_seasonal"):
                arr = self._denormalize(arr)
            else:
                # x_trend, x_seasonal: denormalize input-space arrays
                arr = arr * (self.train_std + 1e-8) + self.train_mean  # approx; trend already centered
                # Actually for x_trend/x_seasonal we undo normalization properly:
                # x_norm = (x - mean)/std → x = x_norm * std + mean
                # x_trend_norm = MA(x_norm), x_trend = x_trend_norm * std + mean  (approx)
                pass
            result[k] = arr
        return result

    def predict_single(
        self,
        window: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Predict on a single lookback window.
        window: (lookback, D)
        Returns dict of (H, D) numpy arrays.
        """
        batch_out = self.predict_batch(window[None, ...])
        return {k: v[0] for k, v in batch_out.items()}

    def rolling_predict(
        self,
        data: np.ndarray,
        start: int = 0,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Sequentially produce one forecast per time step.
        For step t (t = start, start+1, ..., T-H):
          - lookback window: data[t-L:t]
          - forecast horizon: data[t:t+H]

        Returns list of dicts, one per forecast step.
        Each dict has: yhat, yhat_trend, yhat_seasonal, x_trend, x_seasonal (all (H, D))
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before rolling_predict()")
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]

        L = self.config.lookback
        H = self.config.horizon
        T = len(data)
        results = []

        # Batch all valid windows for efficiency
        first = max(start, L)
        valid_starts = list(range(first, T - H + 1))
        if not valid_starts:
            return results

        windows = np.stack([data[t - L: t] for t in valid_starts], axis=0)  # (N, L, D)
        batch_out = self.predict_batch(windows)

        # Attach actual y values
        for idx, t in enumerate(valid_starts):
            step_dict = {k: v[idx] for k, v in batch_out.items()}
            step_dict["t"] = t
            step_dict["y_true"] = data[t: t + H]  # (H, D)
            step_dict["residual"] = step_dict["y_true"] - step_dict["yhat"]
            results.append(step_dict)

        return results

    def retrain(
        self,
        data: np.ndarray,
        reset_stats: bool = True,
    ) -> "DLinearForecaster":
        """
        Retrain the model on new data.
        reset_stats=True  → full reinit (cold start): recompute normalization and rebuild
        reset_stats=False → fine-tune (warm start):   keep existing normalization stats
        """
        if reset_stats or not self.is_fitted:
            return self.fit(data)
        # Fine-tune: keep existing stats, just re-run training
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        data_norm = self._normalize(data)
        X, Y = _build_sliding_windows(data_norm, self.config.lookback, self.config.horizon)
        if len(X) == 0:
            return self
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        criterion = nn.MSELoss()
        self.model.train()
        for _ in range(self.config.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out["yhat"], yb)
                loss.backward()
                optimizer.step()
        self.model.eval()
        return self

    def warm_retrain(
        self,
        data: np.ndarray,
        epochs: int = 5,
        lr_multiplier: float = 0.1,
    ) -> "DLinearForecaster":
        """
        Warm-start fine-tuning: keep model weights AND normalization stats.
        Uses a reduced learning rate (lr * lr_multiplier) to avoid destabilizing
        a well-trained model. This is the primary retrain mode to try first.

        data: (T, D) — recent data. Normalization uses EXISTING stats.
        epochs: number of fine-tune epochs (typically 5–10, much less than initial).
        lr_multiplier: scale factor on initial lr (default 0.1).
        """
        if not self.is_fitted:
            return self.fit(data)
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        data_norm = self._normalize(data)   # use existing stats — no distribution shift
        X, Y = _build_sliding_windows(data_norm, self.config.lookback, self.config.horizon)
        if len(X) == 0:
            return self
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        warm_lr = self.config.lr * lr_multiplier
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=warm_lr, weight_decay=self.config.weight_decay
        )
        criterion = nn.MSELoss()
        self.model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out["yhat"], yb)
                loss.backward()
                optimizer.step()
        self.model.eval()
        return self

    def clone(self) -> "DLinearForecaster":
        """Return a deep copy of this forecaster (for acceptance gate candidate cloning)."""
        import copy
        return copy.deepcopy(self)


