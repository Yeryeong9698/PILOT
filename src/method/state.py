"""
Forecast-error state representation (Section 3.3).

State channels at decision step t (paper notation):
  with_history (5 channels — default, used in the main results):
    [r_t, e_MAE_t, e_MSE_t, h_MAE_t, h_RMSE_t]

  no_history (3 channels — ablation `w/o History`):
    [r_t, e_MAE_t, e_MSE_t]

The standardized per-step vectors are stacked over the most recent N steps
to form the scorer input C_t (Section 3.3, equation 10).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

_EPS = 1e-9

# Channel counts (used by callers to size encoder input dimensions).
N_STATE_CHANNELS_WITH_HIST = 5                  # paper default: 5-channel state
N_STATE_CHANNELS_NO_HIST = 3


def _n_channels(state_mode: str) -> int:
    if state_mode == "with_history":
        return N_STATE_CHANNELS_WITH_HIST
    if state_mode == "no_history":
        return N_STATE_CHANNELS_NO_HIST
    raise ValueError(f"Unsupported state_mode: {state_mode!r}")


def build_state_windows_from_forecasts(
    forecast_results: List[Dict],
    state_mode: str = "with_history",
    history_window: int = 20,
) -> np.ndarray:
    """
    Build a (N, C) state array from a list of rolling-forecast results.

    state_mode='with_history': 5 channels [r, e_MAE, e_MSE, h_MAE, h_RMSE]
    state_mode='no_history':   3 channels [r, e_MAE, e_MSE]  (ablation: no history)
    """
    if state_mode not in ("with_history", "no_history"):
        raise ValueError(f"Unsupported state_mode: {state_mode!r}")

    states: List[np.ndarray] = []
    mse_history: List[float] = []
    abs_history: List[float] = []

    for r in forecast_results:
        R = np.asarray(r["residual"], dtype=np.float64)
        signed_res = float(R.mean())
        abs_err = float(np.abs(R).mean())
        if r.get("y_true") is not None and r.get("yhat") is not None:
            mse_k = float(np.mean((np.asarray(r["y_true"], dtype=np.float64)
                                   - np.asarray(r["yhat"], dtype=np.float64)) ** 2))
        else:
            mse_k = float(np.mean(R ** 2))

        if state_mode == "no_history":
            states.append(np.array([signed_res, abs_err, mse_k], dtype=np.float32))
            continue

        # with_history: append rolling-history summaries.
        mse_history.append(mse_k)
        abs_history.append(abs_err)
        recent_mse = mse_history[-history_window:]
        recent_abs = abs_history[-history_window:]
        win_mae = float(np.mean(recent_abs))
        win_rmse = float(np.sqrt(np.mean(np.array(recent_mse))))
        states.append(np.array(
            [signed_res, abs_err, mse_k, win_mae, win_rmse], dtype=np.float32
        ))

    return np.stack(states, axis=0) if states else np.zeros(
        (0, _n_channels(state_mode)), dtype=np.float32
    )


def build_state_windows(
    forecaster,
    data: np.ndarray,
    start: int,
    state_mode: str = "with_history",
) -> Tuple[np.ndarray, List[Dict]]:
    """Run rolling predict and build the state sequence."""
    results = forecaster.rolling_predict(data, start=start)
    states = build_state_windows_from_forecasts(results, state_mode=state_mode)
    return states, results


def fit_standardizer(states: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute per-channel mean / std from detector-training states."""
    if len(states) == 0:
        n_ch = states.shape[1] if states.ndim == 2 else 1
        return {"mean": np.zeros(n_ch, dtype=np.float32),
                "std":  np.ones(n_ch, dtype=np.float32)}
    return {
        "mean": states.mean(axis=0).astype(np.float32),
        "std":  np.maximum(states.std(axis=0), _EPS).astype(np.float32),
    }


def apply_standardizer(
    states: np.ndarray, stats: Dict[str, np.ndarray]
) -> np.ndarray:
    """Standardize per-channel using fit_standardizer stats."""
    if len(states) == 0:
        return states
    return ((states - stats["mean"]) / (stats["std"] + _EPS)).astype(np.float32)


def stack_state_windows(
    states: np.ndarray, window_len: int = 24
) -> np.ndarray:
    """
    Stack the most recent `window_len` standardized state vectors at each step.

    Returns windows of shape (T - window_len + 1, window_len, C).
    """
    T = len(states)
    if T < window_len:
        return np.zeros((0, window_len, states.shape[1]), dtype=np.float32)
    N = T - window_len + 1
    return np.stack(
        [states[i: i + window_len] for i in range(N)], axis=0
    ).astype(np.float32)
