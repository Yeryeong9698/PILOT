"""
TSLib-based forecaster wrappers.

Uses official model implementations from thuml/Time-Series-Library
(vendored under ./tslib/).  Supported models: TimesNet, iTransformer.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.forecasting.base import BaseForecaster


def _compute_time_features(benchmarks_dir, dataset: str, n_train: int, n_val: int):
    """Return (train_tf, val_tf) using the TSLib timeF embedding.

    Reads ``<benchmarks_dir>/<dataset>/Y_df.csv`` for the date column, builds
    the full-series time features, and slices the leading ``n_train`` rows for
    training and the following ``n_val`` rows for validation.  ``val_tf`` is
    None when ``n_val == 0``.
    """
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tslib"))
    from utils.timefeatures import time_features as _tf_fn
    df = pd.read_csv(Path(benchmarks_dir) / dataset / "Y_df.csv")
    dates = pd.to_datetime(df["date"])
    feat_cols = [c for c in df.columns if c != "date"]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = ~df[feat_cols].isna().all(axis=1)
    dates = dates[keep].reset_index(drop=True)
    all_tf = _tf_fn(pd.DatetimeIndex(dates.values), freq="h").T.astype(np.float32)
    train_tf = all_tf[:n_train]
    val_tf = all_tf[n_train: n_train + n_val] if n_val > 0 else None
    return train_tf, val_tf

# Add TSLib to path
_TSLIB_DIR = str(Path(__file__).parent.parent.parent / "tslib")
if _TSLIB_DIR not in sys.path:
    sys.path.insert(0, _TSLIB_DIR)


# ---------------------------------------------------------------------------
# TSLib configs adapter (argparse-like namespace)
# ---------------------------------------------------------------------------

class TSLibConfigs:
    """Mimics the argparse namespace that TSLib models expect."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_timesnet_configs(seq_len, pred_len, enc_in, d_model=64, d_ff=64,
                            e_layers=2, top_k=3, num_kernels=6, dropout=0.1):
    return TSLibConfigs(
        task_name='long_term_forecast',
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=enc_in,
        c_out=enc_in,
        d_model=d_model,
        d_ff=d_ff,
        e_layers=e_layers,
        top_k=top_k,
        num_kernels=num_kernels,
        dropout=dropout,
        embed='timeF',
        freq='h',
        label_len=0,
        num_class=0,
    )


# ---------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------

def _build_sliding_windows(data, lookback, horizon, time_features=None):
    """Build sliding windows. If time_features (T, D_time) provided, also
    return Mark_enc (N, lookback, D_time) and Mark_dec (N, horizon, D_time).
    """
    T, D = data.shape
    N = T - lookback - horizon + 1
    if N <= 0:
        raise ValueError(f"Data length {T} too short for lookback={lookback} + horizon={horizon}")
    X = np.zeros((N, lookback, D), dtype=np.float32)
    Y = np.zeros((N, horizon, D), dtype=np.float32)
    if time_features is not None:
        Dt = time_features.shape[1]
        Me = np.zeros((N, lookback, Dt), dtype=np.float32)
        Md = np.zeros((N, horizon, Dt), dtype=np.float32)
    for i in range(N):
        X[i] = data[i: i + lookback]
        Y[i] = data[i + lookback: i + lookback + horizon]
        if time_features is not None:
            Me[i] = time_features[i: i + lookback]
            Md[i] = time_features[i + lookback: i + lookback + horizon]
    if time_features is not None:
        return X, Y, Me, Md
    return X, Y


# ---------------------------------------------------------------------------
# Base TSLib Forecaster
# ---------------------------------------------------------------------------

class _TSLibForecasterBase(BaseForecaster):
    """Shared logic for all TSLib-based forecasters."""

    def __init__(self, model_cls, tslib_configs, lookback, horizon, n_features,
                 epochs=20, batch_size=64, lr=1e-3, weight_decay=1e-4, device=None):
        self._lookback = lookback
        self._horizon = horizon
        self._n_features = n_features
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._weight_decay = weight_decay
        self._model_cls = model_cls
        self._tslib_configs = tslib_configs

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model_cls(tslib_configs).to(self.device)
        self._is_fitted = False
        self.train_mean: Optional[np.ndarray] = None
        self.train_std: Optional[np.ndarray] = None

    @property
    def config(self):
        return self._tslib_configs

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @is_fitted.setter
    def is_fitted(self, value: bool):
        self._is_fitted = value

    def _normalize(self, data):
        return (data - self.train_mean) / (self.train_std + 1e-8)

    def _denormalize(self, data):
        return data * (self.train_std + 1e-8) + self.train_mean

    def _rebuild_model(self):
        self.model = self._model_cls(self._tslib_configs).to(self.device)

    def fit(self, data: np.ndarray,
            val_data: np.ndarray = None, patience: int = 3,
            epochs: int = None, time_features: np.ndarray = None,
            val_time_features: np.ndarray = None,
            skip_global_norm: bool = False,
            benchmarks_dir=None, dataset: str = None,
            ) -> "_TSLibForecasterBase":
        """Fit the forecaster.

        If ``time_features`` is None and ``benchmarks_dir`` + ``dataset`` are
        provided, the date column from ``<benchmarks_dir>/<dataset>/Y_df.csv``
        is read and the TSLib timeF embedding is computed automatically (with
        a per-split slice for ``val_time_features``).  Callers that already
        have the features in memory can keep passing them explicitly.

        skip_global_norm=True: feed raw data to the model and rely on its
            built-in per-window RevIN (used for TimesNet/iTransformer).
        """
        if time_features is None and benchmarks_dir is not None and dataset is not None:
            try:
                time_features, val_time_features = _compute_time_features(
                    benchmarks_dir, dataset,
                    n_train=len(data), n_val=len(val_data) if val_data is not None else 0,
                )
            except Exception as _e:
                time_features = val_time_features = None
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]

        self._skip_global_norm = bool(skip_global_norm)
        if self._skip_global_norm:
            # No global normalization; per-window RevIN inside model handles it.
            self.train_mean = np.zeros((1, data.shape[1]), dtype=np.float32)
            self.train_std = np.ones((1, data.shape[1]), dtype=np.float32)
        else:
            self.train_mean = data.mean(axis=0, keepdims=True)
            self.train_std = data.std(axis=0, keepdims=True)
            self.train_std = np.where(self.train_std < 1e-8, 1.0, self.train_std)

        D = data.shape[1]
        if D != self._n_features:
            self._n_features = D
            self._tslib_configs.enc_in = D
            self._tslib_configs.c_out = D
            self._rebuild_model()

        ep = epochs if epochs is not None else self._epochs
        kw = {}
        if val_data is not None:
            kw["val_data"] = val_data
        if patience is not None:
            kw["patience"] = patience
        if time_features is not None:
            kw["time_features"] = time_features
        if val_time_features is not None:
            kw["val_time_features"] = val_time_features
        self._train_loop(data, ep, self._lr, **kw)
        return self

    def _train_loop(self, data, epochs, lr, patience=3, val_data=None,
                    time_features=None, val_time_features=None):
        data_norm = self._normalize(data)
        if time_features is not None:
            X, Y, Me, Md = _build_sliding_windows(data_norm, self._lookback, self._horizon, time_features)
            dataset = TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(Y, dtype=torch.float32),
                torch.tensor(Me, dtype=torch.float32),
                torch.tensor(Md, dtype=torch.float32),
            )
        else:
            X, Y = _build_sliding_windows(data_norm, self._lookback, self._horizon)
            dataset = TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(Y, dtype=torch.float32),
            )
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True, drop_last=False)

        # Validation loader for early stopping
        val_loader = None
        val_has_marks = False
        if val_data is not None:
            val_norm = self._normalize(np.asarray(val_data, dtype=np.float32))
            if val_time_features is not None:
                Xv, Yv, Mev, Mdv = _build_sliding_windows(val_norm, self._lookback, self._horizon, val_time_features)
                if len(Xv) > 0:
                    val_dataset = TensorDataset(
                        torch.tensor(Xv, dtype=torch.float32),
                        torch.tensor(Yv, dtype=torch.float32),
                        torch.tensor(Mev, dtype=torch.float32),
                        torch.tensor(Mdv, dtype=torch.float32),
                    )
                    val_has_marks = True
            else:
                Xv, Yv = _build_sliding_windows(val_norm, self._lookback, self._horizon)
                if len(Xv) > 0:
                    val_dataset = TensorDataset(
                        torch.tensor(Xv, dtype=torch.float32),
                        torch.tensor(Yv, dtype=torch.float32),
                    )
            if len(Xv) > 0:
                val_loader = DataLoader(val_dataset, batch_size=self._batch_size, shuffle=False, drop_last=False)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=self._weight_decay
        )

        criterion = nn.MSELoss()

        best_loss = float('inf')
        best_state = None
        no_improve = 0

        for _ep in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in loader:
                if len(batch) == 4:
                    xb, yb, mb_e, mb_d = [b.to(self.device) for b in batch]
                    # x_dec: zeros of horizon length (signals decoder mode).
                    # iTransformer / TimesNet do not consume x_dec content directly;
                    # only the encoder marks (x_mark_enc) carry temporal information.
                    bsz = xb.shape[0]
                    x_dec = torch.zeros(bsz, self._horizon, xb.shape[-1], device=self.device)
                    out = self.model(xb, mb_e, x_dec, mb_d)
                else:
                    xb, yb = [b.to(self.device) for b in batch]
                    out = self.model(xb, None, None, None)
                optimizer.zero_grad()
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            train_avg = epoch_loss / max(n_batches, 1)

            # Compute monitor loss (val if provided, else train)
            if val_loader is not None:
                self.model.eval()
                v_loss = 0.0
                v_batches = 0
                with torch.no_grad():
                    for batch in val_loader:
                        if val_has_marks:
                            xb, yb, mb_e, mb_d = [b.to(self.device) for b in batch]
                            bsz = xb.shape[0]
                            x_dec = torch.zeros(bsz, self._horizon, xb.shape[-1], device=self.device)
                            out = self.model(xb, mb_e, x_dec, mb_d)
                        else:
                            xb, yb = [b.to(self.device) for b in batch]
                            out = self.model(xb, None, None, None)
                        v_loss += criterion(out, yb).item()
                        v_batches += 1
                monitor_loss = v_loss / max(v_batches, 1)
            else:
                monitor_loss = train_avg

            if monitor_loss < best_loss - 1e-6:
                best_loss = monitor_loss
                best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self._is_fitted = True

    def predict_batch(self, windows: np.ndarray, marks_enc: np.ndarray = None,
                      marks_dec: np.ndarray = None) -> Dict[str, np.ndarray]:
        if not self._is_fitted:
            raise RuntimeError("Call fit() first")
        windows = np.asarray(windows, dtype=np.float32)
        windows_norm = self._normalize(windows)
        self.model.eval()

        CHUNK = 32 if self._n_features > 100 else 512
        yhat_chunks = []
        has_marks = marks_enc is not None
        with torch.no_grad():
            for i in range(0, len(windows_norm), CHUNK):
                x_t = torch.tensor(windows_norm[i:i+CHUNK], dtype=torch.float32, device=self.device)
                if has_marks:
                    me_t = torch.tensor(marks_enc[i:i+CHUNK], dtype=torch.float32, device=self.device)
                    md_t = torch.tensor(marks_dec[i:i+CHUNK], dtype=torch.float32, device=self.device) \
                           if marks_dec is not None else None
                    bsz = x_t.shape[0]
                    x_dec = torch.zeros(bsz, self._horizon, x_t.shape[-1], device=self.device)
                    out = self.model(x_t, me_t, x_dec, md_t)
                else:
                    out = self.model(x_t, None, None, None)
                yhat_chunks.append(out.cpu().numpy())
                del x_t, out
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        yhat_norm = np.concatenate(yhat_chunks, axis=0)
        yhat = self._denormalize(yhat_norm)
        return {"yhat": yhat}

    def predict_single(self, window: np.ndarray) -> Dict[str, np.ndarray]:
        batch_out = self.predict_batch(window[None, ...])
        return {k: v[0] for k, v in batch_out.items()}

    def rolling_predict(self, data: np.ndarray, start: int = 0,
                        time_features: np.ndarray = None) -> List[Dict[str, np.ndarray]]:
        if not self._is_fitted:
            raise RuntimeError("Call fit() first")
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]

        L, H = self._lookback, self._horizon
        T = len(data)
        first = max(start, L)
        valid_starts = list(range(first, T - H + 1))
        if not valid_starts:
            return []

        windows = np.stack([data[t - L: t] for t in valid_starts], axis=0)
        marks_enc = None
        marks_dec = None
        if time_features is not None:
            marks_enc = np.stack([time_features[t - L: t] for t in valid_starts], axis=0).astype(np.float32)
            marks_dec = np.stack([time_features[t: t + H] for t in valid_starts], axis=0).astype(np.float32)
        batch_out = self.predict_batch(windows, marks_enc=marks_enc, marks_dec=marks_dec)
        results = []
        for idx, t in enumerate(valid_starts):
            step_dict = {k: v[idx] for k, v in batch_out.items()}
            step_dict["t"] = t
            step_dict["y_true"] = data[t: t + H]
            step_dict["residual"] = step_dict["y_true"] - step_dict["yhat"]
            results.append(step_dict)
        return results

    def retrain(self, data: np.ndarray, reset_stats: bool = True,
                time_features: np.ndarray = None) -> "_TSLibForecasterBase":
        if reset_stats or not self._is_fitted:
            return self.fit(data, time_features=time_features)
        kw = {}
        if time_features is not None:
            kw["time_features"] = time_features
        self._train_loop(data, self._epochs, self._lr, **kw)
        return self

    def warm_retrain(self, data: np.ndarray, epochs: int = 5,
                     lr_multiplier: float = 0.1,
                     time_features: np.ndarray = None) -> "_TSLibForecasterBase":
        if not self._is_fitted:
            return self.fit(data, time_features=time_features)
        kw = {}
        if time_features is not None:
            kw["time_features"] = time_features
        self._train_loop(data, epochs, self._lr * lr_multiplier, **kw)
        return self

    def clone(self) -> "_TSLibForecasterBase":
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Concrete TSLib Forecasters
# ---------------------------------------------------------------------------

class TSLibTimesNetForecaster(_TSLibForecasterBase):
    """TimesNet forecaster using official TSLib implementation."""
    _backbone_name = "timesnet"

    def __init__(self, lookback=96, horizon=96, n_features=1, epochs=20,
                 d_model=64, d_ff=64, e_layers=2, top_k=3, num_kernels=6,
                 dropout=0.1, batch_size=64, lr=1e-3, device=None, **kw):
        from models.TimesNet import Model as TimesNetModel
        configs = _make_timesnet_configs(
            seq_len=lookback, pred_len=horizon, enc_in=n_features,
            d_model=d_model, d_ff=d_ff, e_layers=e_layers,
            top_k=top_k, num_kernels=num_kernels, dropout=dropout,
        )
        super().__init__(
            model_cls=TimesNetModel, tslib_configs=configs,
            lookback=lookback, horizon=horizon, n_features=n_features,
            epochs=epochs, batch_size=batch_size, lr=lr, device=device,
        )


# ---------------------------------------------------------------------------
# iTransformer
# ---------------------------------------------------------------------------

class TSLibiTransformerForecaster(_TSLibForecasterBase):
    """iTransformer forecaster using official TSLib implementation."""
    _backbone_name = "itransformer"

    def __init__(self, lookback=96, horizon=96, n_features=1, epochs=20,
                 d_model=128, d_ff=256, n_heads=4, e_layers=2, dropout=0.1,
                 batch_size=64, lr=1e-3, device=None, **kw):
        from models.iTransformer import Model as iTransformerModel
        configs = TSLibConfigs(
            task_name='long_term_forecast',
            seq_len=lookback, pred_len=horizon,
            enc_in=n_features, c_out=n_features,
            d_model=d_model, d_ff=d_ff, n_heads=n_heads,
            e_layers=e_layers, dropout=dropout,
            factor=1, activation='gelu',
            embed='timeF', freq='h',
            num_class=0, label_len=0,
        )
        super().__init__(
            model_cls=iTransformerModel, tslib_configs=configs,
            lookback=lookback, horizon=horizon, n_features=n_features,
            epochs=epochs, batch_size=batch_size, lr=lr, device=device,
        )

    def _train_loop(self, data, epochs, lr, patience=999, val_data=None,
                    time_features=None, val_time_features=None):
        """iTransformer: no early stopping by default. Pass val_data to enable val-loss-based ES."""
        super()._train_loop(data, epochs, lr, patience=patience, val_data=val_data,
                            time_features=time_features, val_time_features=val_time_features)


