"""
Pseudo-label construction (Section 3.4).

Five label types (paper variants):
  mean_deg       : mean future error - mean past error
  cum_deg        : cumulative future error - W_f * mean past error
  rel_deg        : relative degradation ratio
                   (mse_future - mse_past) / mse_past
  binary_utility : 1 if retrain gain > 0
  ordinal_utility: 0 / 1 / 2 by utility quantiles (Q50, Q80)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.method.state import (
    build_state_windows_from_forecasts,
    stack_state_windows,
    apply_standardizer,
    _n_channels,
)

_EPS = 1e-8


def _stale_mse_from_arr(
    mse_arr: np.ndarray,
    step_start: int,
    step_end: int,
    lookback: int,
) -> float:
    """Stale-model MSE for [step_start, step_end) from precomputed mse_arr.
    mse_arr[i] corresponds to global step (lookback + i)."""
    idx_start = max(0, step_start - lookback)
    idx_end = min(len(mse_arr), step_end - lookback)
    if idx_end <= idx_start:
        return float("nan")
    return float(np.mean(mse_arr[idx_start:idx_end]))


def _eval_mse_range(
    fcst,
    data: np.ndarray,
    step_start: int,
    step_end: int,
    lookback: int,
    horizon: int,
) -> float:
    """Evaluate forecaster MSE on data[step_start:step_end] (step-indexed)."""
    results = fcst.rolling_predict(data, start=step_start)
    mses = []
    for r in results:
        t = r["t"]
        if t >= step_end:
            break
        mses.append(float(np.mean((r["y_true"] - r["yhat"]) ** 2)))
    return float(np.mean(mses)) if mses else float("nan")


# ====================================================================
# Common: rolling predict + state window preparation
# ====================================================================

def _prepare_mse_and_windows(
    det_data, backbone_fcst, std_stats,
    lookback, horizon, window_len, state_mode,
    dataset=None, seed=None, verbose=True, label_name="",
):
    """Rolling predict → per-step MSE array + standardized state windows."""
    L, H = lookback, horizon
    T = len(det_data)

    if verbose:
        print(f"[label_builder/{label_name}] Rolling predict (T={T})...")
    base_results = backbone_fcst.rolling_predict(det_data, start=L)
    if not base_results:
        return None, None, None

    mse_arr = np.array(
        [float(np.mean((r["y_true"] - r["yhat"]) ** 2)) for r in base_results],
        dtype=np.float32,
    )

    raw_states = build_state_windows_from_forecasts(base_results, state_mode=state_mode)
    if std_stats is not None and std_stats["mean"].shape[0] == raw_states.shape[1]:
        std_states = apply_standardizer(raw_states, std_stats)
    else:
        std_states = raw_states

    all_windows = stack_state_windows(std_states, window_len=window_len)
    return mse_arr, all_windows, base_results


def _empty_result(window_len, state_mode):
    n_ch = _n_channels(state_mode)
    return (np.zeros((0, window_len, n_ch), dtype=np.float32),
            np.zeros(0, dtype=np.float32), pd.DataFrame())


# ====================================================================
# Mean degradation
# ====================================================================

def build_mean_deg_labels(
    det_data, backbone_fcst, std_stats,
    lookback, horizon, window_len, state_mode,
    current_window=50, future_window=50,
    stride=3, dataset=None, seed=None, verbose=True,
):
    """
    d_t = mean(e[t+1:t+Hd]) - mean(e[t-W+1:t])
    Continuous regression label.
    """
    mse_arr, all_windows, _ = _prepare_mse_and_windows(
        det_data, backbone_fcst, std_stats,
        lookback, horizon, window_len, state_mode,
        dataset, seed, verbose, "mean_deg")

    if mse_arr is None:
        return _empty_result(window_len, state_mode)

    n_steps = len(mse_arr)
    n_wins = len(all_windows)
    windows_list, labels_list, meta_rows = [], [], []

    candidates = list(range(current_window, n_steps - future_window, stride))
    _iter = candidates

    for mse_idx in _iter:
        win_idx = mse_idx - window_len + 1
        if win_idx < 0 or win_idx >= n_wins:
            continue
        mean_past = float(np.mean(mse_arr[mse_idx - current_window:mse_idx]))
        mean_future = float(np.mean(mse_arr[mse_idx:mse_idx + future_window]))
        label = mean_future - mean_past

        windows_list.append(all_windows[win_idx])
        labels_list.append(label)
        meta_rows.append({
            "step": lookback + mse_idx, "mean_past": mean_past,
            "mean_future": mean_future, "label": label,
            "label_type": "mean_deg",
            "dataset": dataset or "", "seed": seed or -1,
        })

    if not windows_list:
        return _empty_result(window_len, state_mode)
    if verbose:
        _l = np.array(labels_list)
        print(f"[mean_deg] samples={len(_l)} mean={_l.mean():.4f} std={_l.std():.4f} pos_ratio={((_l > 0).mean()):.1%}")
    return (np.stack(windows_list).astype(np.float32),
            np.array(labels_list, dtype=np.float32), pd.DataFrame(meta_rows))


# ====================================================================
# Cumulative degradation
# ====================================================================

def build_cum_deg_labels(
    det_data, backbone_fcst, std_stats,
    lookback, horizon, window_len, state_mode,
    current_window=50, future_window=50,
    stride=3, dataset=None, seed=None, verbose=True,
):
    """
    d_t = sum(e[t+1:t+Hd]) - Hd * mean(e[t-W+1:t])
    """
    mse_arr, all_windows, _ = _prepare_mse_and_windows(
        det_data, backbone_fcst, std_stats,
        lookback, horizon, window_len, state_mode,
        dataset, seed, verbose, "cum_deg")

    if mse_arr is None:
        return _empty_result(window_len, state_mode)

    n_steps = len(mse_arr)
    n_wins = len(all_windows)
    windows_list, labels_list, meta_rows = [], [], []

    candidates = list(range(current_window, n_steps - future_window, stride))
    _iter = candidates

    Hd = future_window
    for mse_idx in _iter:
        win_idx = mse_idx - window_len + 1
        if win_idx < 0 or win_idx >= n_wins:
            continue
        mean_past = float(np.mean(mse_arr[mse_idx - current_window:mse_idx]))
        cum_future = float(np.sum(mse_arr[mse_idx:mse_idx + future_window]))
        label = cum_future - Hd * mean_past

        windows_list.append(all_windows[win_idx])
        labels_list.append(label)
        meta_rows.append({
            "step": lookback + mse_idx, "mean_past": mean_past,
            "cum_future": cum_future, "label": label,
            "label_type": "cum_deg",
            "dataset": dataset or "", "seed": seed or -1,
        })

    if not windows_list:
        return _empty_result(window_len, state_mode)
    if verbose:
        _l = np.array(labels_list)
        print(f"[cum_deg] samples={len(_l)} mean={_l.mean():.4f} std={_l.std():.4f} pos_ratio={((_l > 0).mean()):.1%}")
    return (np.stack(windows_list).astype(np.float32),
            np.array(labels_list, dtype=np.float32), pd.DataFrame(meta_rows))


# ====================================================================
# Binary utility label
# ====================================================================

def build_binary_utility_labels(
    det_data, backbone_fcst, std_stats,
    lookback, horizon, window_len, state_mode,
    retrain_window=1000,
    eval_horizon=None,
    gain_margin=0.0, stride=10,
    future_window=50,
    dataset=None, seed=None, verbose=True,
    device=None,
    **_kw,
):
    """
    y_t = 1 if u_t > δ, where u_t = L_stale - L_retrain.
    Requires retrain simulation (slow).
    """
    import torch
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if eval_horizon is None:
        eval_horizon = future_window

    L, H = lookback, horizon
    T = len(det_data)

    mse_arr, all_windows, base_results = _prepare_mse_and_windows(
        det_data, backbone_fcst, std_stats,
        lookback, horizon, window_len, state_mode,
        dataset, seed, verbose, "binary_util")

    if mse_arr is None:
        return _empty_result(window_len, state_mode)

    n_wins = len(all_windows)
    windows_list, labels_list, meta_rows = [], [], []
    n_ok = n_skip = 0

    # Use coarser stride for utility (retrain sim is expensive)
    candidates = list(range(L + window_len, T - H, stride))
    _iter = candidates


    for k in _iter:
        mse_idx = k - L
        win_idx = mse_idx - window_len + 1
        if win_idx < 0 or win_idx >= n_wins:
            n_skip += 1; continue

        # Build retrain data
        recent = det_data[max(0, k - retrain_window):k]
        if len(recent) < L + H:
            n_skip += 1; continue

        # Simulate retrain via warm-start (paper Section 3.6); 5 epochs at lr * 0.1.
        candidate = backbone_fcst.clone()
        candidate.warm_retrain(recent, epochs=5, lr_multiplier=0.1)

        eval_end = min(k + eval_horizon, T - H + 1)
        if eval_end <= k:
            n_skip += 1; continue

        mse_stale = _stale_mse_from_arr(mse_arr, k, eval_end, L)
        mse_retrain = _eval_mse_range(candidate, det_data, k, eval_end, L, H)

        if np.isnan(mse_stale) or np.isnan(mse_retrain):
            n_skip += 1; continue

        utility = mse_stale - mse_retrain  # positive = retrain helps
        label = 1.0 if utility > gain_margin else 0.0

        windows_list.append(all_windows[win_idx])
        labels_list.append(label)
        meta_rows.append({
            "step": k, "utility": float(utility),
            "mse_stale": float(mse_stale), "mse_retrain": float(mse_retrain),
            "label": int(label), "gain_margin": gain_margin,
            "label_type": "binary_utility",
            "dataset": dataset or "", "seed": seed or -1,
        })
        n_ok += 1

    if not windows_list:
        return _empty_result(window_len, state_mode)
    if verbose:
        _l = np.array(labels_list)
        print(f"[binary_util] samples={n_ok} skipped={n_skip} pos_ratio={_l.mean():.1%}")
    return (np.stack(windows_list).astype(np.float32),
            np.array(labels_list, dtype=np.float32), pd.DataFrame(meta_rows))


# ====================================================================
# Ordinal utility label (3-class)
# ====================================================================

def build_ordinal_utility_labels(
    det_data, backbone_fcst, std_stats,
    lookback, horizon, window_len, state_mode,
    retrain_window=1000,
    eval_horizon=None,
    stride=10,
    future_window=50,
    q_mild=0.5, q_strong=0.8,
    dataset=None, seed=None, verbose=True,
    device=None,
    **_kw,
):
    """
    3-class ordinal label:
      0 = stable (u ≤ q50)
      1 = mild benefit (q50 < u ≤ q80)
      2 = strong benefit (u > q80)
    Requires retrain simulation (slow).
    """
    import torch
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if eval_horizon is None:
        eval_horizon = future_window

    L, H = lookback, horizon
    T = len(det_data)

    mse_arr, all_windows, base_results = _prepare_mse_and_windows(
        det_data, backbone_fcst, std_stats,
        lookback, horizon, window_len, state_mode,
        dataset, seed, verbose, "ordinal_util")

    if mse_arr is None:
        return _empty_result(window_len, state_mode)

    n_wins = len(all_windows)
    # First pass: collect all utilities
    utilities = []
    win_indices = []
    meta_list = []
    n_skip = 0

    candidates = list(range(L + window_len, T - H, stride))
    _iter = candidates


    for k in _iter:
        mse_idx = k - L
        win_idx = mse_idx - window_len + 1
        if win_idx < 0 or win_idx >= n_wins:
            n_skip += 1; continue

        recent = det_data[max(0, k - retrain_window):k]
        if len(recent) < L + H:
            n_skip += 1; continue

        candidate = backbone_fcst.clone()
        candidate.warm_retrain(recent, epochs=5, lr_multiplier=0.1)

        eval_end = min(k + eval_horizon, T - H + 1)
        if eval_end <= k:
            n_skip += 1; continue

        mse_stale = _stale_mse_from_arr(mse_arr, k, eval_end, L)
        mse_retrain = _eval_mse_range(candidate, det_data, k, eval_end, L, H)

        if np.isnan(mse_stale) or np.isnan(mse_retrain):
            n_skip += 1; continue

        utility = mse_stale - mse_retrain
        utilities.append(utility)
        win_indices.append(win_idx)
        meta_list.append({
            "step": k, "utility": float(utility),
            "mse_stale": float(mse_stale), "mse_retrain": float(mse_retrain),
            "dataset": dataset or "", "seed": seed or -1,
        })

    if not utilities:
        return _empty_result(window_len, state_mode)

    # Second pass: assign ordinal labels by quantiles
    u_arr = np.array(utilities, dtype=np.float32)
    q50 = float(np.quantile(u_arr, q_mild))
    q80 = float(np.quantile(u_arr, q_strong))

    windows_list, labels_list, meta_rows = [], [], []
    for i, (u, wi) in enumerate(zip(utilities, win_indices)):
        if u <= q50:
            label = 0
        elif u <= q80:
            label = 1
        else:
            label = 2
        windows_list.append(all_windows[wi])
        labels_list.append(float(label))
        row = meta_list[i].copy()
        row["label"] = label
        row["label_type"] = "ordinal_utility"
        row["q50_threshold"] = q50
        row["q80_threshold"] = q80
        meta_rows.append(row)

    if verbose:
        _l = np.array(labels_list)
        print(f"[ordinal_util] samples={len(_l)} skipped={n_skip} "
              f"class0={(_l == 0).mean():.1%} class1={(_l == 1).mean():.1%} class2={(_l == 2).mean():.1%} "
              f"q50={q50:.6f} q80={q80:.6f}")

    return (np.stack(windows_list).astype(np.float32),
            np.array(labels_list, dtype=np.float32), pd.DataFrame(meta_rows))


# ====================================================================
# Relative degradation label
# ====================================================================

def build_rel_deg_labels(
    det_data, backbone_fcst, std_stats,
    lookback, horizon, window_len, state_mode,
    current_window=50, future_window=50,
    stride=3, dataset=None, seed=None, verbose=True,
    degradation_margin=0.05,
):
    """
    d_t = (mean_future - mean_past) / (mean_past + eps)
    Normalized relative degradation ratio. Positive = model getting worse.
    """
    mse_arr, all_windows, _ = _prepare_mse_and_windows(
        det_data, backbone_fcst, std_stats,
        lookback, horizon, window_len, state_mode,
        dataset, seed, verbose, "rel_deg")

    if mse_arr is None:
        return _empty_result(window_len, state_mode)

    n_steps = len(mse_arr)
    n_wins = len(all_windows)
    windows_list, labels_list, meta_rows = [], [], []

    candidates = list(range(current_window, n_steps - future_window, stride))
    _iter = candidates

    for mse_idx in _iter:
        win_idx = mse_idx - window_len + 1
        if win_idx < 0 or win_idx >= n_wins:
            continue
        mean_past = float(np.mean(mse_arr[mse_idx - current_window:mse_idx]))
        mean_future = float(np.mean(mse_arr[mse_idx:mse_idx + future_window]))
        label = (mean_future - mean_past) / (mean_past + _EPS)  # normalized ratio

        windows_list.append(all_windows[win_idx])
        labels_list.append(label)
        meta_rows.append({
            "step": lookback + mse_idx, "mean_past": mean_past,
            "mean_future": mean_future, "degradation": label, "label": label,
            "label_type": "rel_deg",
            "dataset": dataset or "", "seed": seed or -1,
        })

    if not windows_list:
        return _empty_result(window_len, state_mode)
    if verbose:
        _l = np.array(labels_list)
        print(f"[rel_deg] samples={len(_l)} mean={_l.mean():.4f} std={_l.std():.4f} "
              f"pos_ratio={((_l > degradation_margin).mean()):.1%}")
    return (np.stack(windows_list).astype(np.float32),
            np.array(labels_list, dtype=np.float32), pd.DataFrame(meta_rows))


# ====================================================================
# Dispatcher
# ====================================================================

LABEL_TYPES = ("mean_deg", "cum_deg", "rel_deg", "binary_utility", "ordinal_utility")
# Which labels are classification (vs regression)
IS_CLASSIFICATION = {"binary_utility", "ordinal_utility"}


def build_scorer_labels(
    label_type: str,
    det_data, val_data, backbone_fcst, std_stats,
    lookback, horizon, window_len, state_mode,
    retrain_window=1000,
    stride=3, dataset=None, seed=None, verbose=True,
    device=None,
    rel_deg_window=None, label_current_window=None, label_future_window=None,
):
    """
    Dispatch to the appropriate label builder.
    Returns (windows_all, labels_all, meta_all) combining det_data + val_data.
    """
    # Priority: explicit current/future > rel_deg_window > defaults
    if label_current_window is not None:
        _cw = label_current_window
    elif rel_deg_window is not None:
        _cw = rel_deg_window
    else:
        _cw = lookback
    if label_future_window is not None:
        _fw = label_future_window
    elif rel_deg_window is not None:
        _fw = rel_deg_window
    else:
        _fw = horizon
    common = dict(
        backbone_fcst=backbone_fcst, std_stats=std_stats,
        lookback=lookback, horizon=horizon,
        window_len=window_len, state_mode=state_mode,
        current_window=_cw, future_window=_fw,
        dataset=dataset, seed=seed, verbose=verbose,
    )

    # Utility labels use coarser stride
    util_stride = max(stride, 10)

    builders = {
        "mean_deg": lambda data: build_mean_deg_labels(det_data=data, stride=stride, **common),
        "cum_deg": lambda data: build_cum_deg_labels(det_data=data, stride=stride, **common),
        "binary_utility": lambda data: build_binary_utility_labels(
            det_data=data,
            retrain_window=retrain_window,
            stride=util_stride, device=device, **common),
        "ordinal_utility": lambda data: build_ordinal_utility_labels(
            det_data=data,
            retrain_window=retrain_window,
            stride=util_stride, device=device, **common),
        # rel_deg = relative degradation ratio
        "rel_deg": lambda data: build_rel_deg_labels(det_data=data, stride=stride, **common),
    }

    if label_type not in builders:
        raise ValueError(f"Unknown label_type: {label_type}. Choose from {LABEL_TYPES}")

    build_fn = builders[label_type]

    if verbose:
        print(f"[scorer] Building {label_type} labels on det_data...")
    w_det, l_det, m_det = build_fn(det_data)

    if verbose:
        print(f"[scorer] Building {label_type} labels on val_data...")
    w_val, l_val, m_val = build_fn(val_data)

    # Regression labels are returned raw; clipping is applied by the caller.
                    
    # Combine
    if len(w_det) > 0 and len(w_val) > 0:
        windows = np.concatenate([w_det, w_val], axis=0)
        labels = np.concatenate([l_det, l_val], axis=0)
        meta = pd.concat([m_det, m_val], ignore_index=True)
    elif len(w_det) > 0:
        windows, labels, meta = w_det, l_det, m_det
    else:
        windows, labels, meta = w_val, l_val, m_val

    if verbose and len(labels) > 0:
        print(f"[scorer] Total: {len(labels)} samples, "
              f"label mean={labels.mean():.4f} std={labels.std():.4f}")

    return windows, labels, meta
