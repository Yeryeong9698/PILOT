"""
PILOT main entry: per-cell evaluation of the PILOT method together with
the six baselines (No Retrain, Periodic, ADWIN, KSWIN, CARA, UPF).

Usage:
  python scripts/run_method.py --backbone timesnet --datasets ETTm1 --seed 42
"""
from __future__ import annotations

import argparse
import copy
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.forecasting.dlinear import DLinearConfig, DLinearForecaster
from src.forecasting.base import BaseForecaster
from src.method.state import (
    build_state_windows, stack_state_windows,
    fit_standardizer, _n_channels,
)
from src.method.scorer import ScorerConfig, UtilityPredictor, train_utility_predictor

_LOG_FILE: Optional[str] = None

# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------

def create_forecaster(backbone: str, lookback: int, horizon: int,
                      n_features: int, epochs: int = 20,
                      ma_kernel: int = 25, **kw) -> BaseForecaster:
    """Create a forecaster instance for one of the three paper backbones."""
    bs = kw.get('batch_size', 128)
    lr = kw.get('lr', 1e-4)
    if backbone == "dlinear":
        return DLinearForecaster(DLinearConfig(
            lookback=lookback, horizon=horizon, n_features=n_features,
            ma_kernel=ma_kernel, epochs=epochs, batch_size=bs, lr=lr,
        ))
    elif backbone == "timesnet":
        from src.forecasting.tslib import TSLibTimesNetForecaster
        return TSLibTimesNetForecaster(
            lookback=lookback, horizon=horizon, n_features=n_features, epochs=epochs,
            batch_size=bs, lr=lr,
            top_k=kw.get('top_k', 3), d_model=kw.get('d_model', 64),
            d_ff=kw.get('d_ff', 64), e_layers=kw.get('e_layers', 2),
        )
    elif backbone == "itransformer":
        from src.forecasting.tslib import TSLibiTransformerForecaster
        return TSLibiTransformerForecaster(
            lookback=lookback, horizon=horizon, n_features=n_features, epochs=epochs,
            d_model=kw.get('d_model', 128), d_ff=kw.get('d_ff', 256),
            e_layers=kw.get('e_layers', 2),
            batch_size=bs, lr=lr,
        )
    else:
        raise ValueError(
            f"Unknown backbone: {backbone!r}. "
            f"Supported: dlinear, itransformer, timesnet."
        )

_BACKBONE_ARCH_CACHE = None

def load_backbone_arch(backbone: str, dataset: str) -> Dict:
    """Read the per-cell architecture from ``configs/backbone_arch.yaml``.

    The yaml ships the configuration used for the paper's main results
    (Table 1).  When the (backbone, dataset) cell is missing the call
    returns an empty dict so the caller falls back to the backbone's
    standard architecture defaults; this lets users run the script on
    additional datasets without editing the yaml.
    """
    global _BACKBONE_ARCH_CACHE
    if _BACKBONE_ARCH_CACHE is None:
        import yaml
        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "backbone_arch.yaml"
        with open(cfg_path) as f:
            _BACKBONE_ARCH_CACHE = yaml.safe_load(f)
    try:
        return dict(_BACKBONE_ARCH_CACHE[backbone][dataset])
    except KeyError:
        return {}

def _log(msg: str):
    print(msg, flush=True)
    if _LOG_FILE:
        try:
            with open(_LOG_FILE, "a") as f:
                f.write(msg + "\n")
        except Exception:
            pass

import datetime
def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")

# ---------------------------------------------------------------------------
# Dataset loading / splitting
# ---------------------------------------------------------------------------

def load_dataset(benchmarks_dir: Path, dataset: str,
                 normalize: bool = False) -> Tuple[np.ndarray, List[str]]:
    """Load raw dataset. normalize=False by default (normalization done after split)."""
    csv_path = benchmarks_dir / dataset / "Y_df.csv"
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError(f"No 'date' column in {csv_path}")
    feat_cols = [c for c in df.columns if c != "date"]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=feat_cols, how="all")
    data = df[feat_cols].ffill().bfill().values.astype(np.float32)
    if normalize:
        # Full-series z-score (only applied when normalize=True; splits are normalized separately elsewhere).
        _mean = data.mean(axis=0, keepdims=True)
        _std  = data.std(axis=0, keepdims=True)
        _std  = np.where(_std < 1e-8, 1.0, _std)
        data  = ((data - _mean) / _std).astype(np.float32)
    return data, feat_cols

def infer_step_hours(benchmarks_dir: Path, dataset: str) -> float:
    """Infer the sampling interval (in hours) from the dataset's date column."""
    csv_path = benchmarks_dir / dataset / "Y_df.csv"
    df = pd.read_csv(csv_path, usecols=["date"], nrows=100)
    dates = pd.to_datetime(df["date"])
    median_delta = dates.diff().dropna().median()
    step_hours = median_delta.total_seconds() / 3600.0
    return step_hours

def normalize_splits(
    bb_data: np.ndarray,
    *other_splits: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    """
    Normalize all splits using bb_data (training set) statistics.
    This matches the DLinear paper convention:
      x_norm = (x - bb_mean) / bb_std  (per channel)

    Returns (bb_norm, *others_norm) as float32 arrays.
    MSE computed on these arrays is in the same space as DLinear paper.
    """
    bb_mean = bb_data.mean(axis=0, keepdims=True)
    bb_std  = bb_data.std(axis=0, keepdims=True)
    bb_std  = np.where(bb_std < 1e-8, 1.0, bb_std)
    def _norm(x: np.ndarray) -> np.ndarray:
        return ((x - bb_mean) / bb_std).astype(np.float32)
    return tuple(_norm(s) for s in (bb_data, *other_splits))

def chronological_split(data: np.ndarray, dataset_name: str):
    """Paper Section 4.1 split: backbone / scorer-train / val / test.

    ETT datasets:  60 / 10 / 10 / 20
    Other datasets: 70 /  5 /  5 / 20

    The standard val split is halved so the first half trains the scorer and
    the second half tunes hyperparameters.
    """
    T = len(data)
    if dataset_name.startswith("ETT"):
        train_r, val_r = 0.6, 0.2
    else:
        train_r, val_r = 0.7, 0.1
    n_train = int(T * train_r)
    n_val = int(T * val_r)
    n_test = T - n_train - n_val
    if n_test <= 0:
        raise ValueError(f"chronological_split: not enough data (T={T}, dataset={dataset_name})")
    train_d = data[:n_train]
    val_full = data[n_train: n_train + n_val]
    test_d = data[n_train + n_val:]
    half = len(val_full) // 2
    det_d = val_full[:half]
    val_d = val_full[half:]
    return train_d, det_d, val_d, test_d

# ---------------------------------------------------------------------------
# Retrain backend helpers
# ---------------------------------------------------------------------------

def do_retrain(
    fcst: DLinearForecaster,
    backbone_data: np.ndarray,
    test_data: np.ndarray,
    trigger_step: int,
    retrain_window: int,
    replay_ratio: float,  # 0.0 = recent only, 0.2 = 20% backbone tail
    acceptance_delta: float,  # -1 = no gate, 0.0 = strict, 0.001*baseline...
    baseline_mse: float,      # used for delta scaling
    lookback: int,
    horizon: int,
    test_time_features: np.ndarray = None,
    backbone_time_features: np.ndarray = None,
) -> Tuple[DLinearForecaster, bool, float]:
    """
    Execute one retrain event.
    Returns (updated_fcst, was_accepted, candidate_mse).
    """
    L = lookback
    H = horizon

    # --- Build retrain data (and aligned time_features) ---
    rec_start = max(0, trigger_step - retrain_window)
    rec_end = trigger_step
    recent = test_data[rec_start:rec_end]
    recent_tf = (test_time_features[rec_start:rec_end]
                 if test_time_features is not None else None)
    if replay_ratio > 0:
        n_replay = int(len(recent) * replay_ratio / (1 - replay_ratio + 1e-9))
        n_replay = min(n_replay, len(backbone_data))
        replay = backbone_data[-n_replay:] if n_replay > 0 else np.zeros((0, backbone_data.shape[1]))
        retrain_data = np.concatenate([replay, recent], axis=0) if len(replay) > 0 else recent
        if recent_tf is not None and backbone_time_features is not None and n_replay > 0:
            replay_tf = backbone_time_features[-n_replay:]
            retrain_tf = np.concatenate([replay_tf, recent_tf], axis=0)
        else:
            retrain_tf = recent_tf
    else:
        retrain_data = recent
        retrain_tf = recent_tf

    if len(retrain_data) < L + H:
        return fcst, False, float("nan")

    # --- Acceptance gate: hold out tail as acceptance set ---
    if acceptance_delta >= 0 and len(retrain_data) >= 2 * (L + H):
        accept_len = min(L + H, len(retrain_data) // 4)
        accept_data = retrain_data[-accept_len:]
        train_data_for_retrain = retrain_data[:-accept_len]
        train_tf_for_retrain = (retrain_tf[:-accept_len] if retrain_tf is not None else None)
        if len(train_data_for_retrain) < L + H:
            train_data_for_retrain = retrain_data
            train_tf_for_retrain = retrain_tf
            accept_data = None
    else:
        train_data_for_retrain = retrain_data
        train_tf_for_retrain = retrain_tf
        accept_data = None

    # --- Clone candidate, retrain ---
    candidate = fcst.clone()

    _retrain_kw = {"time_features": train_tf_for_retrain} if train_tf_for_retrain is not None else {}
    candidate.warm_retrain(train_data_for_retrain, epochs=5,
                           lr_multiplier=0.1, **_retrain_kw)

    # --- Acceptance gate evaluation ---
    candidate_mse = float("nan")
    if acceptance_delta >= 0 and accept_data is not None and len(accept_data) >= L + H:
        # evaluate both models on acceptance set
        def eval_on_window(model: DLinearForecaster, data_seg: np.ndarray) -> float:
            mses = []
            for t in range(L, len(data_seg) - H + 1):
                win = data_seg[t - L: t]
                y_true = data_seg[t: t + H]
                pred = model.predict_single(win)
                mses.append(float(np.mean((y_true - pred["yhat"]) ** 2)))
            return float(np.mean(mses)) if mses else float("nan")

        current_mse = eval_on_window(fcst, accept_data)
        candidate_mse = eval_on_window(candidate, accept_data)

        delta_threshold = acceptance_delta * baseline_mse
        if np.isnan(candidate_mse) or candidate_mse > current_mse - delta_threshold:
            return fcst, False, candidate_mse   # rejected
        return candidate, True, candidate_mse
    else:
        # No gate: always accept
        return candidate, True, candidate_mse

from src.baselines.adwin import (
    ADWIN as _ADWIN,
    DEFAULT_DELTA as _ADWIN_DEFAULT_DELTA,
    DEFAULT_DELTA_CANDIDATES as _ADWIN_DELTA_CANDIDATES,
)
from src.baselines.cara import CARARetrainingDecider
from src.baselines.kswin import (
    KSWIN as _KSWIN,
    DEFAULT_ALPHA as _KSWIN_DEFAULT_ALPHA,
    DEFAULT_ALPHA_CANDIDATES as _KSWIN_ALPHA_CANDIDATES,
    DEFAULT_STAT_SIZE as _KSWIN_DEFAULT_STAT_SIZE,
    DEFAULT_WINDOW_SIZE as _KSWIN_DEFAULT_WINDOW_SIZE,
)
from src.baselines.periodic import (
    DEFAULT_PERIODIC_INTERVAL as _PERIODIC_DEFAULT_INTERVAL,
    DEFAULT_PERIODIC_INTERVALS as _PERIODIC_INTERVALS,
)

# ---------------------------------------------------------------------------
# Core online evaluation loop
# ---------------------------------------------------------------------------

def evaluate_method(
    data: np.ndarray,
    backbone_data: np.ndarray,
    fcst_config: DLinearConfig,
    detector: Optional[object],
    std_stats: Optional[Dict],
    method: str,    # "no_retrain"|"periodic"|"pilot"|"adwin"|"kswin"|"cara"|"upf"
    threshold: Optional[float],
    periodic_interval: Optional[int],
    cooldown: int,
    min_gap: int,
    retrain_window: int,
    replay_ratio: float,
    acceptance_delta: float,
    device: torch.device,
    window_len: int = 24,
    pre_fitted_fcst=None,
    state_mode: str = "with_history",
    utility_predictor=None,
    method_params: Optional[Dict] = None,   # extra per-method kwargs
    data_time_features: np.ndarray = None,    # aligned with `data` (test or val)
    backbone_time_features: np.ndarray = None,  # aligned with backbone_data
    **kwargs,  # E0/E1/E2/E4 ablation_mode passed through here
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    L = fcst_config.lookback if fcst_config else kwargs.get('_lookback', 96)
    H = fcst_config.horizon if fcst_config else kwargs.get('_horizon', 96)
    T = len(data)

    # Reuse pre-fitted backbone if provided (avoids re-training on every tuning call)
    if pre_fitted_fcst is not None:
        fcst = pre_fitted_fcst.clone()
    elif fcst_config is not None:
        fcst = DLinearForecaster(fcst_config)
        fcst.fit(backbone_data)
    else:
        # Non-DLinear backbone: use factory
        _bb = kwargs.get('backbone', 'dlinear')
        _D = backbone_data.shape[1] if backbone_data.ndim > 1 else 1
        fcst = create_forecaster(_bb, L, H, _D,
                                  epochs=kwargs.get('backbone_epochs', 20),
                                  ma_kernel=kwargs.get('ma_kernel', 25),
                                  top_k=kwargs.get('top_k', 3),
                                  d_model=kwargs.get('d_model', 64),
                                  d_ff=kwargs.get('d_ff', 64),
                                  e_layers=kwargs.get('e_layers', 2))
        fcst.fit(backbone_data)

    # Compute baseline MSE on backbone tail for acceptance gate scaling
    baseline_mse = 1.0
    if acceptance_delta >= 0:
        try:
            bb_tail = backbone_data[-(L * 4):]
            res_list = fcst.rolling_predict(bb_tail, start=L)
            if res_list:
                baseline_mse = float(np.mean([np.mean((r["y_true"] - r["yhat"])**2) for r in res_list]))
        except Exception:
            pass

    if method == "pilot" and utility_predictor is not None:
        utility_predictor.eval()

    # Running z-score state (Welford's online algorithm, Section 3.6).
    _zscore_n = [0]
    _zscore_mean = [0.0]
    _zscore_m2 = [0.0]
    _ZSCORE_WARMUP = 50  # N_wu: warm-up steps before triggering (Section 4.1).

    std_state_history: List = []      # incrementally standardized states
    predictions, retrain_rows, alarm_rows = [], [], []

    last_retrain = -999_999
    retrain_id = 0
    n_accepted = 0
    n_rejected = 0
    total_runtime = 0.0

    # ── Early Exit for Baselines ──────────────────────────
    if method == "no_retrain":
        start_time = time.perf_counter()
        _rp_kw = {}
        if data_time_features is not None:
            _rp_kw["time_features"] = data_time_features
        res_list = fcst.rolling_predict(data, start=L, **_rp_kw)
        total_runtime += (time.perf_counter() - start_time)

        if not res_list:
            summary = {
                "mse": float("nan"), "mae": float("nan"), "cumulative_mse": float("nan"),
                "retrain_count": 0, "alarm_count": 0, "accepted_count": 0, "rejected_count": 0,
                "runtime": total_runtime,
            }
            return summary, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Vectorized MSE / MAE over the whole batch
        y_true_all = np.array([r["y_true"] for r in res_list])
        yhat_all = np.array([r["yhat"] for r in res_list])
        
        # Determine axes to average over (supports both 2D (batch, features) and 3D (batch, horizon, features) shapes)
        avg_axes = tuple(range(1, y_true_all.ndim))
        mse_arr = np.mean((y_true_all - yhat_all) ** 2, axis=avg_axes)
        mae_arr = np.mean(np.abs(y_true_all - yhat_all), axis=avg_axes)
        
        pred_df = pd.DataFrame({
            "step": np.arange(L, L + len(res_list)),
            "mse": mse_arr,
            "mae": mae_arr,
            "retrain_flag": 0
        })
        summary = {
            "mse": float(mse_arr.mean()) if len(mse_arr) else float("nan"),
            "mae": float(mae_arr.mean()) if len(mae_arr) else float("nan"),
            "cumulative_mse": float(mse_arr.sum()) if len(mse_arr) else float("nan"),
            "retrain_count": 0,
            "alarm_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "runtime": total_runtime,
        }
        return summary, pred_df, pd.DataFrame(), pd.DataFrame()

    # ── Standard drift detector initialization ────────────────────────────
    _mp = method_params or {}
    _adwin_det = None
    _kswin_det = None
    if method == "adwin":
        _adwin_det = _ADWIN(delta=float(threshold if threshold is not None else _ADWIN_DEFAULT_DELTA))
    if method == "kswin":
        _kswin_det = _KSWIN(
            alpha=float(threshold if threshold is not None else _KSWIN_DEFAULT_ALPHA),
            window_size=int(_mp.get("window_size", _KSWIN_DEFAULT_WINDOW_SIZE)),
            stat_size=int(_mp.get("stat_size", _KSWIN_DEFAULT_STAT_SIZE)),
        )

    # ── UPF / CARA initialization ────────────────────────────────────────
    _upf = None
    _cara_decider = None
    if method == "upf":
        from src.baselines.upf import UPF
        if "upf_fitted_model" in _mp:
            _upf = _mp["upf_fitted_model"]
            _upf.reset()
        else:
            _upf = UPF(
                alpha=float(_mp.get("alpha", 0.1)),
                H_lookahead=int(_mp.get("H_lookahead", 100)),
                delta=float(_mp.get("delta", 0.95)),
                N_mc=int(_mp.get("N_mc", 100)),
                K_checkpoints=int(_mp.get("K_checkpoints", 8)),
                refit_interval=int(_mp.get("refit_interval", 10)),
                min_observations=20,
                cooldown=cooldown,
                min_gap=min_gap,
                verbose=False,
            )
            _det_data_upf = _mp.get("det_data", None)
            _bb_fcst_upf  = _mp.get("backbone_fcst", pre_fitted_fcst)
            if _det_data_upf is not None and _bb_fcst_upf is not None:
                _upf.fit_forecaster(
                    det_data=_det_data_upf,
                    backbone_fcst=_bb_fcst_upf,
                    lookback=L, horizon=H,
                    retrain_window=retrain_window,

                )
            _upf.reset()
    elif method == "cara":
        _cara_decider = CARARetrainingDecider(
            variant=str(_mp.get("variant", "T")),
            tau=float(threshold if threshold is not None else _mp.get("tau", 0.05)),
            baseline_window=50,
            min_observations=20,
        )

    # -----------------------------------------------------------------------
    # SEGMENT-BATCH LOOP: for each method, batch-predict the entire segment
    # between retrains on GPU, then scan for alarm in numpy.
    # GPU is now busy with large contiguous batch calls, not per-step tiny ones.
    # -----------------------------------------------------------------------

    def _batch_segment_and_scan(
        seg_start: int,
        _sm: str = state_mode,   # capture state_mode at definition time (avoid closure bugs)
    ):
        """
        GPU-batch predict data[seg_start : T-H+1], compute states + scores,
        then scan for the first alarm step.
        Returns (results, states_std, scores_arr) all as numpy arrays.
        - results:     list of rolling_predict dicts for [seg_start, T-H+1)
        - scores_arr:  (N,) float32 alarm scores (0.0 if not yet in bank)
        """
        # ── 1. Batch predict entire remaining data on GPU ────────────────
        _rp_kw = {}
        if data_time_features is not None:
            _rp_kw["time_features"] = data_time_features
        roll = fcst.rolling_predict(data, start=seg_start, **_rp_kw)
        if not roll:
            return [], np.array([]), np.array([]), np.array([])

        # ── 2. Vectorized state computation (numpy, no Python loop) ──────
        residuals   = np.stack([r["residual"]      for r in roll], axis=0)  # (N,H,D)
        yhats       = np.stack([r["yhat"]          for r in roll], axis=0)  # (N,H,D)
        ytrues      = np.stack([r["y_true"]        for r in roll], axis=0)  # (N,H,D)

        mean_r       = residuals.mean(axis=(1, 2))                       # (N,)
        mabs_r       = np.abs(residuals).mean(axis=(1, 2))                # (N,)
        mse_per_step = np.mean((ytrues - yhats) ** 2, axis=(1, 2))        # (N,)

        _n_state_ch = _n_channels(_sm)
        if _sm == "with_history":
            # 5 channels: [signed_res, abs_err, mse, h_MAE, h_RMSE]
            N = len(mean_r)
            _s1_w = 20
            abs_hist = list(getattr(_batch_segment_and_scan, '_s1_abs_hist', [])) + list(mabs_r)
            mse_hist_s1 = list(getattr(_batch_segment_and_scan, '_s1_mse_hist', [])) + list(mse_per_step)
            win_mae_arr = np.zeros(N, dtype=np.float32)
            win_rmse_arr = np.zeros(N, dtype=np.float32)
            base_off = len(getattr(_batch_segment_and_scan, '_s1_abs_hist', []))
            for _vi in range(N):
                idx = base_off + _vi
                recent_abs = abs_hist[max(0, idx+1-_s1_w):idx+1]
                recent_mse = mse_hist_s1[max(0, idx+1-_s1_w):idx+1]
                win_mae_arr[_vi] = float(np.mean(recent_abs))
                win_rmse_arr[_vi] = float(np.sqrt(np.mean(np.array(recent_mse))))
            raw_seg = np.stack(
                [mean_r, mabs_r, mse_per_step, win_mae_arr, win_rmse_arr],
                axis=1,
            ).astype(np.float32)
            _batch_segment_and_scan._s1_abs_hist = abs_hist[-_s1_w:]
            _batch_segment_and_scan._s1_mse_hist = mse_hist_s1[-_s1_w:]
        else:
            # no_history: [signed_res, abs_err, mse] — no rolling history
            raw_seg = np.stack(
                [mean_r, mabs_r, mse_per_step], axis=1
            ).astype(np.float32)

        # Standardize; skip if std_stats is None OR channel dim doesn't match (shape mismatch)
        if std_stats is not None and std_stats["mean"].shape[0] == raw_seg.shape[1]:
            std_seg = ((raw_seg - std_stats["mean"]) / (std_stats["std"] + 1e-8)).astype(np.float32)
        else:
            std_seg = raw_seg  # skip standardisation if stats are missing or shape-mismatched

        # ── 3. Build detector windows for the segment ────────────────────
        # Prepend existing std_state_history as context for the first windows (only need window_len - 1)
        _ctx_hist = std_state_history[-(window_len - 1):] if std_state_history and window_len > 1 else []
        ctx = np.stack(_ctx_hist, axis=0) if _ctx_hist else np.zeros((0, _n_state_ch), dtype=np.float32)
        full_std = np.concatenate([ctx, std_seg], axis=0)      # (ctx_len + N, _n_state_ch)
        full_wins = stack_state_windows(full_std, window_len=window_len)  # (M, W, _n_state_ch)
        # Windows aligned to seg indices: full_wins[i] corresponds to step_in_seg = i - ctx_len
        ctx_len = len(ctx)

        # ── 4. Batch GPU encode all windows at once ──────────────────────
        scores_arr = np.zeros(len(roll), dtype=np.float32)

        # ── ABLATION: wo_learning_oracle — bypass scorer, compute oracle labels ──
        _abl_var_inner = kwargs.get("ablation_variant", None)
        if _abl_var_inner == "wo_learning_oracle" and method == "pilot":
            # Retrieve oracle config
            _oracle_lbl = kwargs.get("oracle_label_type", "mean_deg")
            _oracle_cw = kwargs.get("label_current_window", 48)
            _oracle_fw = kwargs.get("label_future_window", 48)

            # Build full MSE array: past history + current segment
            _prev_mse_hist = list(getattr(_batch_segment_and_scan, '_oracle_mse_hist', []))
            _full_mse = np.concatenate([np.array(_prev_mse_hist, dtype=np.float32), mse_per_step])
            _base_off = len(_prev_mse_hist)

            for _oi in range(len(roll)):
                idx = _base_off + _oi  # position in _full_mse
                past_start = max(0, idx - _oracle_cw)
                future_end = min(len(_full_mse), idx + _oracle_fw)

                # Need enough past and future data
                if idx - past_start < 3 or future_end - idx < 3:
                    scores_arr[_oi] = 0.0
                    continue

                mean_past_o = float(np.mean(_full_mse[past_start:idx]))
                mean_future_o = float(np.mean(_full_mse[idx:future_end]))
                future_errs_o = _full_mse[idx:future_end]

                if _oracle_lbl == "mean_deg":
                    scores_arr[_oi] = mean_future_o - mean_past_o
                elif _oracle_lbl == "rel_deg":
                    scores_arr[_oi] = (mean_future_o - mean_past_o) / (mean_past_o + 1e-8)
                elif _oracle_lbl == "cum_deg":
                    fw_actual = future_end - idx
                    scores_arr[_oi] = float(np.sum(future_errs_o)) - fw_actual * mean_past_o
                elif _oracle_lbl in ("binary_utility", "ordinal_utility"):
                    # Proxy: use mean_deg as the degradation signal (no retraining at test time)
                    scores_arr[_oi] = mean_future_o - mean_past_o
                else:
                    scores_arr[_oi] = mean_future_o - mean_past_o

            # Persist MSE history for next segment (keep last cw values)
            _batch_segment_and_scan._oracle_mse_hist = list(_full_mse[-_oracle_cw:])

        elif method == "pilot" and utility_predictor is not None and len(full_wins) > 0:
            # Batch predict gain for all windows
            all_win_t = torch.tensor(full_wins, dtype=torch.float32, device=device)
            ENCODE_CHUNK = 512
            with torch.no_grad():
                chunks = [utility_predictor(all_win_t[i:i+ENCODE_CHUNK])
                          for i in range(0, len(all_win_t), ENCODE_CHUNK)]
                all_preds = torch.cat(chunks, dim=0)  # (M,) predicted gains on GPU
            for seg_i in range(len(roll)):
                win_i = ctx_len + seg_i - window_len + 1
                if win_i < 0 or win_i >= len(full_wins):
                    continue
                scores_arr[seg_i] = float(all_preds[win_i].item())

        return roll, raw_seg, std_seg, scores_arr

    # ---- main segment loop ----
    current_step = L
    _retrains_so_far = 0
    while current_step < T - H + 1:

        if method == "periodic":
            # Periodic uses the segment-based fast-path from the while loop above
            # (already handled by the segment window logic)
            # Use single-step detection for simplicity + batch prediction per segment
            if periodic_interval is not None and (current_step - max(0, last_retrain)) >= periodic_interval:
                alarm = (current_step - last_retrain) >= max(cooldown, min_gap)
            else:
                alarm = False

            if alarm:
                # Batch predict just this step's prediction
                if current_step >= L and current_step < T - H + 1:
                    x_in = data[current_step - L: current_step]
                    pred = fcst.predict_single(x_in)
                    y_true = data[current_step: current_step + H]
                    yhat = pred["yhat"]
                    mse_s = float(np.mean((y_true - yhat) ** 2))
                    mae_s = float(np.mean(np.abs(y_true - yhat)))
                    predictions.append({"step": current_step, "mse": mse_s, "mae": mae_s, "retrain_flag": 0})
                    alarm_rows.append({"step": current_step, "score": 0.0, "alarm": 1, "accepted": 0, "sc_decision": float('nan')})

                    fcst, was_accepted, cand_mse = do_retrain(
                        fcst=fcst, backbone_data=backbone_data, test_data=data,
                        trigger_step=current_step, retrain_window=retrain_window,
                        replay_ratio=replay_ratio,
                        acceptance_delta=acceptance_delta, baseline_mse=baseline_mse,
                        lookback=L, horizon=H,
                        test_time_features=data_time_features,
                        backbone_time_features=backbone_time_features,
                    )
                    retrain_id += 1
                    if was_accepted:
                        last_retrain = current_step
                        n_accepted += 1
                        retrain_rows.append({"retrain_id": retrain_id, "trigger_step": current_step,
                                             "accepted": 1, "candidate_mse": cand_mse,
                                             "retrain_window": retrain_window})
                        alarm_rows[-1]["accepted"] = 1
                    else:
                        n_rejected += 1
                        retrain_rows.append({"retrain_id": -1, "trigger_step": current_step,
                                             "accepted": 0, "candidate_mse": cand_mse,
                                             "retrain_window": retrain_window})
                current_step += 1
                continue

            # Batch predict up to next periodic trigger
            next_trig = last_retrain + periodic_interval if periodic_interval is not None else T
            seg_end = min(T - H + 1, next_trig)
            if seg_end <= current_step:
                seg_end = current_step + 1

            _rp_kw_p = {}
            if data_time_features is not None:
                _rp_kw_p["time_features"] = data_time_features
            roll = fcst.rolling_predict(data, start=current_step, **_rp_kw_p)
            # trim to segment
            roll = [r for r in roll if r["t"] < seg_end]
            for r in roll:
                t = r["t"]
                mse_s = float(np.mean((r["y_true"] - r["yhat"]) ** 2))
                mae_s = float(np.mean(np.abs(r["y_true"] - r["yhat"])))
                predictions.append({"step": t, "mse": mse_s, "mae": mae_s, "retrain_flag": 0})
                alarm_rows.append({"step": t, "score": 0.0, "alarm": 0, "accepted": 0, "sc_decision": float('nan')})
            current_step = seg_end
            continue

        # ── PILOT and drift-detector-based baselines (kswin, adwin, upf, ...) ──
        roll, raw_seg, std_seg, scores_arr = _batch_segment_and_scan(current_step)

        if not roll:
            break

        # ── Scan for first alarm in the segment ──────────────────────────
        first_alarm_idx = None
        for seg_i, r in enumerate(roll):
            step = r["t"]
            cooldown_ok = (step - last_retrain) >= cooldown
            gap_ok      = (step - last_retrain) >= min_gap
            trigger_ok  = cooldown_ok and gap_ok

            if method == "pilot":
                sc    = float(scores_arr[seg_i])

                # ── Online z-score calibration (Section 3.6, Equation 23) ──
                # Welford running mean / variance over the score stream.
                _zscore_n[0] += 1
                delta = sc - _zscore_mean[0]
                _zscore_mean[0] += delta / _zscore_n[0]
                delta2 = sc - _zscore_mean[0]
                _zscore_m2[0] += delta * delta2
                if _zscore_n[0] >= _ZSCORE_WARMUP:
                    running_std = max((_zscore_m2[0] / _zscore_n[0]) ** 0.5, 1e-8)
                    sc_decision = (sc - _zscore_mean[0]) / running_std
                else:
                    # Warm-up period: triggering is disabled (Equation 23).
                    sc_decision = 0.0

                # Single-step threshold trigger (Section 3.6, Equation 25).
                alarm = (
                    trigger_ok
                    and threshold is not None
                    and sc_decision >= threshold
                )

            # ── Standard drift detection baselines ────────────────────────
            elif method == "adwin":
                mse_step = float(np.mean((r["y_true"] - r["yhat"]) ** 2))
                _adwin_det.update(mse_step)
                sc = mse_step
                alarm = trigger_ok and _adwin_det.drift_detected
                if alarm:
                    _adwin_det = _ADWIN(delta=float(threshold if threshold is not None else _ADWIN_DEFAULT_DELTA))
                scores_arr[seg_i] = sc

            elif method == "kswin":
                mse_step = float(np.mean((r["y_true"] - r["yhat"]) ** 2))
                _kswin_det.update(mse_step)
                sc = mse_step
                alarm = trigger_ok and _kswin_det.drift_detected
                if alarm:
                    _kswin_det = _KSWIN(
                        alpha=float(threshold if threshold is not None else _KSWIN_DEFAULT_ALPHA),
                        window_size=int(_mp.get("window_size", _KSWIN_DEFAULT_WINDOW_SIZE)),
                        stat_size=int(_mp.get("stat_size", _KSWIN_DEFAULT_STAT_SIZE)),
                    )
                scores_arr[seg_i] = sc

            elif method == "upf" and _upf is not None:
                mse_step = float(np.mean((r["y_true"] - r["yhat"]) ** 2))
                _upf.update(mse_step)
                sc = mse_step
                alarm = trigger_ok and _upf.should_retrain()
                if alarm:
                    _upf.notify_retrain(_upf.decider._current_step if _upf.decider else step)
                scores_arr[seg_i] = sc

            elif method == "cara":
                mse_step = float(np.mean((r["y_true"] - r["yhat"]) ** 2))
                _cara_decider.update(mse_step)
                sc = mse_step
                alarm = trigger_ok and _cara_decider.should_retrain()
                if alarm:
                    _cara_decider.notify_retrain()
                scores_arr[seg_i] = sc

            else:
                alarm, sc = False, 0.0

            mse_s = float(np.mean((r["y_true"] - r["yhat"]) ** 2))
            mae_s = float(np.mean(np.abs(r["y_true"] - r["yhat"])))
            predictions.append({"step": step, "mse": mse_s, "mae": mae_s, "retrain_flag": 0})
            _sc_dec = float(sc_decision) if method == "pilot" and 'sc_decision' in dir() else float('nan')
            alarm_rows.append({"step": step, "score": float(scores_arr[seg_i]), "alarm": int(alarm), "accepted": 0, "sc_decision": _sc_dec})

            # For pilot: commit state history up to (and including) this step
            if method == "pilot" and seg_i < len(raw_seg):
                std_state_history.append(std_seg[seg_i])

            if alarm:
                first_alarm_idx = seg_i
                break

        if first_alarm_idx is None:
            # No alarm in entire remaining data — we're done
            break

        # ── Retrain at alarm ──────────────────────────────────────────────
        alarm_step = roll[first_alarm_idx]["t"]

        # Pre-check: skip silently if not enough data for retrain (avoid wasting retrain count)
        _recent_len = len(data[max(0, alarm_step - retrain_window): alarm_step])
        if replay_ratio > 0:
            _n_replay = min(int(_recent_len * replay_ratio / (1 - replay_ratio + 1e-9)), len(backbone_data))
            _retrain_len = _recent_len + _n_replay
        else:
            _retrain_len = _recent_len
        if _retrain_len < L + H:
            # Not enough data yet — skip alarm without counting as retrain or consuming cooldown
            current_step = alarm_step + 1
            continue

        fcst, was_accepted, cand_mse = do_retrain(
            fcst=fcst, backbone_data=backbone_data, test_data=data,
            trigger_step=alarm_step, retrain_window=retrain_window,
            replay_ratio=replay_ratio,
            acceptance_delta=acceptance_delta, baseline_mse=baseline_mse,
            lookback=L, horizon=H,
            test_time_features=data_time_features,
            backbone_time_features=backbone_time_features,
        )
        retrain_id += 1
        _retrains_so_far += 1
        if was_accepted:
            last_retrain = alarm_step
            n_accepted  += 1
            retrain_rows.append({"retrain_id": retrain_id, "trigger_step": alarm_step,
                                  "accepted": 1, "candidate_mse": cand_mse,
                                  "retrain_window": retrain_window})
            alarm_rows[-1]["accepted"] = 1
            predictions[-1]["retrain_flag"] = 1
            # Notify stateful baselines of accepted retrain.
            if method == "cara" and _cara_decider is not None:
                _cara_decider.notify_retrain()
            elif method == "upf" and _upf is not None:
                if _upf.decider is not None:
                    _upf.notify_retrain(_upf.decider._current_step)
        else:
            n_rejected += 1
            retrain_rows.append({"retrain_id": -1, "trigger_step": alarm_step,
                                  "accepted": 0, "candidate_mse": cand_mse,
                                  "retrain_window": retrain_window})

        current_step = alarm_step + 1

    pred_df = pd.DataFrame(predictions)
    retrain_df = pd.DataFrame(retrain_rows if retrain_rows else
                              [{"retrain_id": 0, "trigger_step": -1, "accepted": 0,
                                "candidate_mse": float("nan"), "retrain_window": retrain_window}])
    alarm_df = pd.DataFrame(alarm_rows)

    mse_arr = pred_df["mse"].values
    mae_arr = pred_df["mae"].values
    retrain_steps = retrain_df[retrain_df["trigger_step"] >= 0]["trigger_step"].tolist()

    summary = {
        "mse": float(mse_arr.mean()) if len(mse_arr) else float("nan"),
        "mae": float(mae_arr.mean()) if len(mae_arr) else float("nan"),
        "cumulative_mse": float(mse_arr.sum()) if len(mse_arr) else float("nan"),
        "retrain_count": len(retrain_steps),
        "alarm_count": int(alarm_df["alarm"].sum()),
        "accepted_count": n_accepted,
        "rejected_count": n_rejected,
        "runtime": total_runtime,
    }
    return summary, pred_df, retrain_df, alarm_df

# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_run(out_dir: Path, summary: Dict, pred_df: pd.DataFrame,
             retrain_df: pd.DataFrame, alarm_df: pd.DataFrame, config_dict: Dict,
             tuned_params: Dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    retrain_df.to_csv(out_dir / "retrains.csv", index=False)
    alarm_df.to_csv(out_dir / "alarms.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "config.yaml", "w") as f:
        json.dump(config_dict, f, indent=2)
    with open(out_dir / "tuned_params.json", "w") as f:
        json.dump(tuned_params, f, indent=2, default=str)

def tune_and_eval_baselines(
    *,
    det_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    bb_data: np.ndarray,
    fcst,
    fcst_config,
    args,
    std_stats,
    state_mode: str,
    device: torch.device,
    dataset: str,
    seed: int,
    out_dir_base: Path,
    save_method_results: bool = True,
) -> Dict[str, Dict]:
    """Tune and evaluate the six paper baselines on (val_full -> test).

    Each baseline's hyperparameter is selected on the full validation
    split (det_data + val_data) by minimising
    ``J = val_MSE + lambda_retrain * retrain_rate``; the test summary
    uses the chosen hyperparameter.  Grids and fixed hyperparameters
    follow paper Section 4.1.

    Returns a dict ``{method_name: {"test_summary", "tuned_params"}}`` and, if
    ``save_method_results`` is True, also writes the standard run files
    (predictions/retrains/alarms/metrics/config) under
    ``out_dir_base / <method_name> / seed_<seed>``.
    """
    val_full = np.concatenate([det_data, val_data], axis=0) if len(det_data) > 0 else val_data
    n_val_full = max(1, len(val_full) - fcst_config.horizon + 1)
    days_val = n_val_full / fcst_config.lookback

    common = dict(
        backbone_data=bb_data, fcst_config=fcst_config,
        detector=None, std_stats=std_stats,
        cooldown=args.cooldown, min_gap=20,
        retrain_window=args.retrain_window,
        replay_ratio=0.0, acceptance_delta=-1, device=device,
        pre_fitted_fcst=fcst, state_mode=state_mode,
    )

    def _save(method_name: str, summary, pred_df, retrain_df, alarm_df, tuned):
        if not save_method_results:
            return
        out_dir = out_dir_base / method_name / f"seed_{seed}"
        summary = {**summary, "dataset": dataset}
        save_run(
            out_dir, summary, pred_df, retrain_df, alarm_df,
            {"method": method_name,
             "retrain_window": args.retrain_window, "replay_ratio": 0.0,
             "acceptance_delta": -1},
            tuned,
        )

    results: Dict[str, Dict] = {}

    # ---- no_retrain ----
    s, pd_b, rt_b, al_b = evaluate_method(
        data=test_data, method="no_retrain", threshold=None, periodic_interval=None,
        **common,
    )
    _save("no_retrain", s, pd_b, rt_b, al_b,
          {"threshold": None, "method_params": {}, "periodic_interval": None})
    results["no_retrain"] = {"test_summary": s, "tuned_params": {}}

    # ---- periodic ----
    best_p_int, best_p_j = _PERIODIC_DEFAULT_INTERVAL, float("inf")
    for it in _PERIODIC_INTERVALS:
        sv, _, _, _ = evaluate_method(
            data=val_full, method="periodic", threshold=None, periodic_interval=it,
            **common,
        )
        j = sv["mse"] + args.lambda_retrain * (sv["retrain_count"] / max(1.0, days_val))
        if j < best_p_j:
            best_p_j, best_p_int = j, it
    st, pd_b, rt_b, al_b = evaluate_method(
        data=test_data, method="periodic", threshold=None, periodic_interval=best_p_int,
        **common,
    )
    _save("periodic", st, pd_b, rt_b, al_b,
          {"threshold": None, "method_params": {}, "periodic_interval": best_p_int,
           
           "interval": best_p_int, "val_j": best_p_j})
    results["periodic"] = {"test_summary": st,
                            "tuned_params": {"interval": best_p_int, "val_j": best_p_j}}

    # ---- adwin ----
    best_d, best_dj = _ADWIN_DEFAULT_DELTA, float("inf")
    for d in _ADWIN_DELTA_CANDIDATES:
        sv, _, _, _ = evaluate_method(
            data=val_full, method="adwin", threshold=d, periodic_interval=None,
            method_params={"delta": d}, **common,
        )
        j = sv["mse"] + args.lambda_retrain * (sv["retrain_count"] / max(1.0, days_val))
        if j < best_dj:
            best_dj, best_d = j, d
    st, pd_b, rt_b, al_b = evaluate_method(
        data=test_data, method="adwin", threshold=best_d, periodic_interval=None,
        method_params={"delta": best_d}, **common,
    )
    _save("adwin", st, pd_b, rt_b, al_b,
          {"threshold": best_d, "method_params": {}, "periodic_interval": None,
           
           "delta": best_d, "val_j": best_dj})
    results["adwin"] = {"test_summary": st,
                        "tuned_params": {"delta": best_d, "val_j": best_dj}}

    # ---- kswin ----
    best_kw, best_kj = (
        {"alpha": _KSWIN_ALPHA_CANDIDATES[0],
         "window_size": _KSWIN_DEFAULT_WINDOW_SIZE,
         "stat_size": _KSWIN_DEFAULT_STAT_SIZE},
        float("inf"),
    )
    for a in _KSWIN_ALPHA_CANDIDATES:
        mp = {"alpha": a, "window_size": _KSWIN_DEFAULT_WINDOW_SIZE,
              "stat_size": _KSWIN_DEFAULT_STAT_SIZE}
        sv, _, _, _ = evaluate_method(
            data=val_full, method="kswin", threshold=a, periodic_interval=None,
            method_params=mp, **common,
        )
        j = sv["mse"] + args.lambda_retrain * (sv["retrain_count"] / max(1.0, days_val))
        if j < best_kj:
            best_kj, best_kw = j, mp
    st, pd_b, rt_b, al_b = evaluate_method(
        data=test_data, method="kswin", threshold=best_kw["alpha"], periodic_interval=None,
        method_params=best_kw, **common,
    )
    _save("kswin", st, pd_b, rt_b, al_b,
          {"threshold": best_kw["alpha"], "method_params": best_kw,
           "periodic_interval": None,
           
           "val_j": best_kj})
    results["kswin"] = {"test_summary": st,
                        "tuned_params": {"method_params": best_kw, "val_j": best_kj}}

    # ---- cara ----
    best_cmp, best_cj = {"variant": "T", "tau": 0.05}, float("inf")
    for v in ["T", "C"]:
        mp = {"variant": v, "tau": 0.05}
        sv, _, _, _ = evaluate_method(
            data=val_full, method="cara", threshold=0.05, periodic_interval=None,
            method_params=mp, **common,
        )
        j = sv["mse"] + args.lambda_retrain * (sv["retrain_count"] / max(1.0, days_val))
        if j < best_cj:
            best_cj, best_cmp = j, mp
    st, pd_b, rt_b, al_b = evaluate_method(
        data=test_data, method="cara", threshold=0.05, periodic_interval=None,
        method_params=best_cmp, **common,
    )
    _save("cara", st, pd_b, rt_b, al_b,
          {"threshold": 0.05, "method_params": best_cmp, "periodic_interval": None,
           
           "val_j": best_cj})
    results["cara"] = {"test_summary": st,
                       "tuned_params": {"method_params": best_cmp, "val_j": best_cj}}

    # ---- upf ----
    from src.baselines.upf import UPF
    _alpha_candidates = [0.05, 0.5, 1.0]
    best_uf_alpha, best_uf_j, best_uf_model = _alpha_candidates[0], float("inf"), None
    for _alpha in _alpha_candidates:
        _uf_v = UPF(
            alpha=_alpha, H_lookahead=100, delta=0.95,
            N_mc=getattr(args, "upf_n_mc", 100),
            K_checkpoints=8, refit_interval=10, min_observations=20,
            cooldown=args.cooldown, min_gap=20, verbose=False,
        )
        _uf_v.fit_forecaster(
            det_data=det_data, backbone_fcst=fcst,
            lookback=fcst_config.lookback, horizon=fcst_config.horizon,
            retrain_window=args.retrain_window,
        )
        sv, _, _, _ = evaluate_method(
            data=val_full, method="upf", threshold=None, periodic_interval=None,
            method_params={"upf_fitted_model": _uf_v, "alpha": _alpha}, **common,
        )
        j = sv["mse"] + args.lambda_retrain * (sv["retrain_count"] / max(1.0, days_val))
        if j < best_uf_j:
            best_uf_j, best_uf_alpha, best_uf_model = j, _alpha, _uf_v
    best_uf_model.reset()
    best_uf_model.fit_forecaster(
        det_data=det_data, backbone_fcst=fcst,
        lookback=fcst_config.lookback, horizon=fcst_config.horizon,
        retrain_window=args.retrain_window,
    )
    st, pd_b, rt_b, al_b = evaluate_method(
        data=test_data, method="upf", threshold=None, periodic_interval=None,
        method_params={"upf_fitted_model": best_uf_model, "alpha": best_uf_alpha},
        **common,
    )
    _save("upf", st, pd_b, rt_b, al_b,
          {"threshold": None,
           "method_params": {"upf_fitted_model": repr(best_uf_model), "alpha": best_uf_alpha},
           "periodic_interval": None,
           
           "alpha": best_uf_alpha, "val_j": best_uf_j})
    results["upf"] = {"test_summary": st,
                                "tuned_params": {"alpha": best_uf_alpha, "val_j": best_uf_j}}

    return results

def _compute_state_vector(
    residual: np.ndarray, mse_step: float,
    abs_hist: list, mse_hist: list, state_mode: str,
) -> np.ndarray:
    """Per-step state vector consistent with build_state_windows_from_forecasts."""
    signed_res = float(residual.mean())
    abs_err = float(np.abs(residual).mean())
    mse_hist.append(mse_step); abs_hist.append(abs_err)
    if state_mode == "no_history":
        return np.array([signed_res, abs_err, mse_step], dtype=np.float32)
    win_mae  = float(np.mean(abs_hist[-20:]))
    win_rmse = float(np.sqrt(np.mean(mse_hist[-20:])))
    return np.array([signed_res, abs_err, mse_step, win_mae, win_rmse], dtype=np.float32)

def evaluate_voting(
    *,
    data: np.ndarray,
    bb_data: np.ndarray,
    bb_fcst,
    scorers: Dict[str, "UtilityPredictor"],
    std_stats: Dict[str, np.ndarray],
    state_mode: str,
    window_len: int,
    lookback: int,
    horizon: int,
    z_tau: float,
    vote_threshold: int,
    cooldown: int,
    retrain_window: int,
    device: torch.device,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Online voting eval over (data) using the supplied per-label scorers.

    At each step, every scorer scores the current standardized state window.
    Each scorer maintains its own z-score history; the trigger fires when at
    least ``vote_threshold`` scorers have z > ``z_tau`` simultaneously.  On a
    retrain trigger every scorer's score history is reset.
    """
    from collections import deque
    fcst = copy.deepcopy(bb_fcst)
    L, H = lookback, horizon
    T = len(data)

    state_history: List[np.ndarray] = []
    mse_history: List[float] = []
    abs_history: List[float] = []
    z_windows = {lbl: deque(maxlen=300) for lbl in scorers}

    predictions, retrains, alarms = [], [], []
    last_retrain = -cooldown
    n_retrains = 0

    current_step = L
    while current_step < T - H + 1:
        x_in = data[current_step - L: current_step]
        pred = fcst.predict_single(x_in)
        y_true = data[current_step: current_step + H]
        residual = y_true - pred["yhat"]
        mse_step = float(np.mean(residual ** 2))
        predictions.append({"step": int(current_step), "mse": mse_step,
                            "mae": float(np.mean(np.abs(residual))), "retrain_flag": 0})

        s_raw = _compute_state_vector(residual, mse_step, abs_history, mse_history, state_mode)
        s_std = ((s_raw - std_stats["mean"]) / (std_stats["std"] + 1e-8)).astype(np.float32)
        state_history.append(s_std)

        fired = False
        votes = 0
        if (current_step - L >= window_len and
                current_step - last_retrain > cooldown):
            window = np.stack(state_history[-window_len:], axis=0)
            wt = torch.from_numpy(window).float().unsqueeze(0).to(device)
            with torch.no_grad():
                for lbl, scorer in scorers.items():
                    sc = float(scorer(wt).item())
                    z_windows[lbl].append(sc)
                    if len(z_windows[lbl]) >= 5:
                        hist = np.fromiter(z_windows[lbl], dtype=np.float64)
                        mu, sigma = hist.mean(), max(hist.std(), 0.1)
                        if (sc - mu) / sigma > z_tau:
                            votes += 1
            fired = votes >= vote_threshold
            alarms.append({"step": int(current_step), "votes": int(votes),
                            "alarm": int(fired)})

        if fired:
            fcst, accepted, _ = do_retrain(
                fcst=fcst, backbone_data=bb_data, test_data=data,
                trigger_step=current_step, retrain_window=retrain_window,
                 replay_ratio=0.0,
                acceptance_delta=-1, baseline_mse=1.0,
                lookback=L, horizon=H,
            )
            if accepted:
                n_retrains += 1
                last_retrain = current_step
                retrains.append({"retrain_id": n_retrains,
                                 "trigger_step": int(current_step),
                                 "accepted": 1,
                                 "retrain_window": int(retrain_window),
})
                predictions[-1]["retrain_flag"] = 1
                for lbl in scorers:
                    z_windows[lbl].clear()
        current_step += 1

    pred_df = pd.DataFrame(predictions)
    retrain_df = pd.DataFrame(retrains if retrains else
                              [{"retrain_id": 0, "trigger_step": -1, "accepted": 0,
                                "retrain_window": int(retrain_window),
}])
    alarm_df = pd.DataFrame(alarms)
    summary = {
        "mse": float(pred_df["mse"].mean()) if len(pred_df) else float("nan"),
        "mae": float(pred_df["mae"].mean()) if len(pred_df) else float("nan"),
        "retrain_count": n_retrains,
        "alarm_count": int(alarm_df["alarm"].sum()) if len(alarm_df) else 0,
        "accepted_count": n_retrains,
        "rejected_count": 0,
        "runtime": 0.0,
    }
    return summary, pred_df, retrain_df, alarm_df

# ---------------------------------------------------------------------------
# PILOT main evaluation
# ---------------------------------------------------------------------------

def run_pilot(args, dataset: str, seed: int, device: torch.device,
                best_window: int, best_replay: float,
                best_delta: float) -> Dict:
    _log(f"[{_ts()}] === PILOT evaluation: {dataset} seed={seed} ===")
    torch.manual_seed(seed)
    np.random.seed(seed)
    data, _ = load_dataset(Path(args.benchmarks_dir), dataset)
    T, D = data.shape
    bb_data, det_data, val_data, test_data = chronological_split(data, dataset_name=dataset)
    bb_data, det_data, val_data, test_data = normalize_splits(bb_data, det_data, val_data, test_data)

    backbone = getattr(args, 'backbone', 'dlinear')
    effective_state_mode = args.state_mode

    _bb_epochs = args.backbone_epochs
    _bb_bs = args.backbone_batch_size
    _bb_lr = args.backbone_lr

    # Parameter container used downstream for L/H, ma_kernel, and epoch counts.
    fcst_config = DLinearConfig(
        lookback=args.lookback, horizon=args.horizon, n_features=D,
        ma_kernel=args.ma_kernel, epochs=_bb_epochs,
        batch_size=_bb_bs, lr=_bb_lr,
    )

    # Train backbone + detector
    bb_fcst = create_forecaster(backbone, args.lookback, args.horizon, D,
                                 epochs=_bb_epochs, ma_kernel=args.ma_kernel,
                                 top_k=args.top_k, d_model=args.d_model,
                                 d_ff=args.d_ff, e_layers=args.e_layers,
                                 batch_size=_bb_bs, lr=_bb_lr)

    _tf_bb = _tf_det = _tf_val = _tf_test = _tf_val_full = None
    _val_full_train = (
        np.concatenate([det_data, val_data], axis=0)
        if len(det_data) > 0 else val_data
    )
    bb_fcst.fit(
        bb_data, val_data=_val_full_train,
        benchmarks_dir=args.benchmarks_dir, dataset=dataset,
    )
    state_mode   = effective_state_mode
    n_state_ch   = _n_channels(state_mode)
    det_states_raw, _ = build_state_windows(bb_fcst, det_data, start=args.lookback,
                                              state_mode=state_mode)
    std_stats = fit_standardizer(det_states_raw)
    det_config = ScorerConfig(
        n_state_channels=n_state_ch,
        hidden_dim=64, embedding_dim=64, dropout=0.1,
        epochs=10, batch_size=64, lr=1e-3,
    )

    # ── Train UtilityPredictor (scorer) right after backbone ──
    scorer = None
    # Loop over the requested label types with a shared backbone (already
    # trained above) and a single baseline tuning pass (run on the first
    # label, reused for the rest).
    _label_list = [l.strip() for l in args.label_types.split(',') if l.strip()]
    _log(f"[{_ts()}]   [MULTI-LABEL] Backbone trained once. Labels: {_label_list}")
    _orig_output_dir = args.output_dir
    all_results = {}
    _voting_scorers: Dict[str, "UtilityPredictor"] = {}
    _voting_std_stats = None

    # Baselines run once on the first label, then reused across all labels.
    _baselines_tuned = False

    for _li, _lbl in enumerate(_label_list):
        _lbl_out = _orig_output_dir.rstrip('/') + f'_{_lbl}'
        # Skip if already done
        _done_path = Path(_lbl_out) / 'results' / dataset / 'pilot' / f'seed_{seed}' / 'metrics.json'
        if _done_path.exists():
            _log(f"[{_ts()}]   [MULTI-LABEL] SKIP {_lbl} (already done)")
            # Still mark baselines as tuned if this was would-be first
            if not _baselines_tuned:
                _baselines_tuned = True  # baselines results exist from previous run
            continue

        _is_first = not _baselines_tuned
        _log(f"[{_ts()}]   [MULTI-LABEL] ── label={_lbl} ({_li+1}/{len(_label_list)}) baselines={'YES' if _is_first else 'NO'} ──")

        # ── Build label & train scorer ──
        from src.method.labels import (
            build_scorer_labels, IS_CLASSIFICATION
        )
        _log(f"[{_ts()}]   [SCORER] label_type={_lbl}  state_mode={state_mode}")

        # ── ABLATION: wo_learning_oracle skips scorer training ──
        _is_oracle_ablation = getattr(args, 'ablation_variant', None) == 'wo_learning_oracle'
        if _is_oracle_ablation:
            _log(f"[{_ts()}]   [SCORER] wo_learning_oracle: Skipping scorer MLP training (oracle at test time)")
            scorer = None
        else:
            _log(f"[{_ts()}]   [SCORER] Building labels on det_data + val_data...")

            windows_sl, labels_sl, meta_sl = build_scorer_labels(
                label_type=_lbl,
                det_data=det_data, val_data=val_data,
                backbone_fcst=bb_fcst, std_stats=std_stats,
                lookback=args.lookback, horizon=args.horizon,
                window_len=args.state_window_len, state_mode=state_mode,
                retrain_window=args.retrain_window, 
                stride=3, dataset=dataset, seed=seed, verbose=True,
                device=device,
                rel_deg_window=getattr(args, "rel_deg_window", None),
                label_current_window=getattr(args, "label_current_window", None),
                label_future_window=getattr(args, "label_future_window", None),
            )

            scorer = None
            if len(windows_sl) > 0:
                if _lbl == "binary_utility" and "utility" in meta_sl.columns:
                    diffs = meta_sl['step'].diff()
                    decreases = meta_sl[diffs < 0].index.tolist()
                    if decreases:
                        reset_idx = decreases[0]
                        train_meta = meta_sl.iloc[:reset_idx]
                        train_ut = train_meta['utility'].values
                        pos_train_ut = train_ut[train_ut > 0]
                        _gmq = 0.50
                        if len(pos_train_ut) > 0 and _gmq > 0.0:
                            delta_q = float(np.quantile(pos_train_ut, _gmq))
                            new_labels = (meta_sl['utility'] > delta_q).astype(float)
                            meta_sl['label'] = new_labels
                            labels_sl = new_labels.values.astype(np.float32)
                            _log(f"[{_ts()}]   [SCORER] Applied dynamic Q{int(_gmq*100)} margin: {delta_q:.4f} (PosRate: {labels_sl.mean():.1%})")

                _log(f"[{_ts()}]   [SCORER] Total samples: {len(labels_sl)}  "
                     f"label mean={labels_sl.mean():.4f}  std={labels_sl.std():.4f}")
                sl_dir = Path(_lbl_out) / "scorer_labels_metadata" / dataset / f"seed_{seed}"
                sl_dir.mkdir(parents=True, exist_ok=True)
                meta_sl.to_csv(sl_dir / f"{_lbl}_{state_mode}_metadata.csv", index=False)

                is_cls = _lbl in IS_CLASSIFICATION
                if is_cls:
                    gains_s1a = labels_sl
                    _log(f"[{_ts()}]   [SCORER] Classification mode (n_classes={int(labels_sl.max())+1})")
                else:
                    gains_s1a = np.clip(labels_sl, -0.5, 2.0)
                    _log(f"[{_ts()}]   [SCORER] Regression mode")

                _log(f"[{_ts()}]   [SCORER] Training UtilityPredictor ...")
                scorer = train_utility_predictor(
                    windows=windows_sl, gains=gains_s1a, config=det_config,
                    epochs=30, lr=1e-3, device=device, verbose=True,
                    oversample_positive=True,
                )
                _log(f"[{_ts()}]   [SCORER] Training complete.")
                _voting_scorers[_lbl] = scorer
                _voting_std_stats = std_stats
            else:
                _log(f"[{_ts()}]   [SCORER] WARNING: 0 samples for {_lbl}. Skipping.")

        # ── Baseline tuning + eval (FIRST LABEL ONLY) ──
        if _is_first:
            _log(f"[{_ts()}]   Tuning + evaluating baselines on val_full -> test ...")
            _bl_results = tune_and_eval_baselines(
                det_data=det_data, val_data=val_data, test_data=test_data,
                bb_data=bb_data, fcst=bb_fcst, fcst_config=fcst_config,
                args=args, std_stats=None, state_mode=state_mode, device=device,
                dataset=dataset, seed=seed,
                out_dir_base=Path(_lbl_out) / "results" / dataset,
            )
            for _bm, _br in _bl_results.items():
                _bs = _br["test_summary"]
                _bs["dataset"] = dataset
                all_results[_bm] = _bs
                _log(f"[{_ts()}]   {_bm:<20}  "
                     f"MSE={_bs['mse']:.6f}  retrains={_bs['retrain_count']}")
            _baselines_tuned = True

        # ── PILOT scorer evaluation (every label) ──
        # The threshold theta is fixed to z_tau = 1.0 (Section 4.1).
        if scorer is not None or _is_oracle_ablation:
            z_tau = getattr(args, 'z_tau', 1.0)
            best_tau_pilot = z_tau
            _s, _, _, _ = evaluate_method(
                data=val_data, backbone_data=bb_data, fcst_config=fcst_config,
                detector=None, std_stats=std_stats,
                method="pilot", threshold=z_tau, periodic_interval=None,
                cooldown=args.cooldown, min_gap=args.min_gap,
                retrain_window=best_window,
                replay_ratio=best_replay, acceptance_delta=best_delta, device=device,
                pre_fitted_fcst=bb_fcst,
                state_mode=state_mode,
                utility_predictor=scorer,
                oracle_label_type=_lbl,
                label_current_window=getattr(args, 'label_current_window', 48),
                label_future_window=getattr(args, 'label_future_window', 48),
                data_time_features=_tf_val, backbone_time_features=_tf_bb,
                ablation_variant=getattr(args, 'ablation_variant', None),
                dataset=dataset,
            )
            n_val = max(1, len(val_data) - fcst_config.horizon + 1)
            _n_val_days = n_val / fcst_config.lookback
            best_j = _s["mse"] + args.lambda_retrain * (
                _s["retrain_count"] / max(1.0, _n_val_days)
            )
            _log(f"[{_ts()}]     pilot ({_lbl})  z_tau={z_tau}  J={best_j:.6f}")

            # ── Test pilot ──
            summary_p, pred_p, retrain_p, alarm_p = evaluate_method(
                data=test_data, backbone_data=bb_data,
                fcst_config=fcst_config,
                detector=None, std_stats=std_stats,
                method="pilot", threshold=best_tau_pilot,
                periodic_interval=None,
                cooldown=args.cooldown, min_gap=args.min_gap,
                retrain_window=best_window,
                replay_ratio=best_replay, acceptance_delta=best_delta, device=device,
                pre_fitted_fcst=bb_fcst,
                state_mode=state_mode,
                utility_predictor=scorer,
                data_time_features=_tf_test, backbone_time_features=_tf_bb,
                dataset=dataset,
                ablation_variant=getattr(args, 'ablation_variant', None),
                oracle_label_type=_lbl,
                label_current_window=getattr(args, 'label_current_window', 48),
                label_future_window=getattr(args, 'label_future_window', 48),
            )
            summary_p["dataset"] = dataset
            all_results[f"pilot_{_lbl}"] = summary_p
            _log(f"[{_ts()}]   {'pilot':<20}  label={_lbl}  "
                 f"MSE={summary_p['mse']:.6f}  retrains={summary_p['retrain_count']}")
            out_dir_p = (Path(_lbl_out) / "results" / dataset /
                          "pilot" / f"seed_{seed}")
            save_run(out_dir_p, summary_p, pred_p, retrain_p, alarm_p,
                     {"method": "pilot",
                      "retrain_window": best_window, "replay_ratio": best_replay,
                      "acceptance_delta": best_delta},
                     {"threshold": best_tau_pilot, "label_type": _lbl})

    _log(f"[{_ts()}]   [MULTI-LABEL] All {len(_label_list)} labels done for {dataset}.")

    # ── Voting (paper Section 3.6): q ∈ {3, 4, 5} tuned on val_full ──
    if len(_voting_scorers) >= 3 and _voting_std_stats is not None:
        val_full = (np.concatenate([det_data, val_data], axis=0)
                    if len(det_data) > 0 else val_data)
        n_val_full = max(1, len(val_full) - fcst_config.horizon + 1)
        days_val = n_val_full / fcst_config.lookback
        z_tau = getattr(args, 'z_tau', 1.0)
        best_q, best_q_j = 3, float("inf")
        for q in (3, 4, 5):
            sv, _, _, _ = evaluate_voting(
                data=val_full, bb_data=bb_data, bb_fcst=bb_fcst,
                scorers=_voting_scorers, std_stats=_voting_std_stats,
                state_mode=state_mode, window_len=args.state_window_len,
                lookback=fcst_config.lookback, horizon=fcst_config.horizon,
                z_tau=z_tau, vote_threshold=q,
                cooldown=args.cooldown,
                retrain_window=best_window,
                device=device,
            )
            j = sv["mse"] + args.lambda_retrain * (sv["retrain_count"] / max(1.0, days_val))
            if j < best_q_j:
                best_q_j, best_q = j, q
        _log(f"[{_ts()}]   voting               q={best_q}  J={best_q_j:.6f}")
        st, pred_v, retrain_v, alarm_v = evaluate_voting(
            data=test_data, bb_data=bb_data, bb_fcst=bb_fcst,
            scorers=_voting_scorers, std_stats=_voting_std_stats,
            state_mode=state_mode, window_len=args.state_window_len,
            lookback=fcst_config.lookback, horizon=fcst_config.horizon,
            z_tau=z_tau, vote_threshold=best_q,
            cooldown=args.cooldown,
            retrain_window=best_window,
            device=device,
        )
        st["dataset"] = dataset
        all_results["voting"] = st
        _log(f"[{_ts()}]   {'voting':<20}  q={best_q}  "
             f"MSE={st['mse']:.6f}  retrains={st['retrain_count']}")
        out_dir_vote = Path(_orig_output_dir.rstrip('/') + "_voting") / "results" / dataset / "voting" / f"seed_{seed}"
        save_run(out_dir_vote, st, pred_v, retrain_v, alarm_v,
                 {"method": "voting",
                  "retrain_window": best_window, "replay_ratio": 0.0,
                  "acceptance_delta": -1},
                 {"q": int(best_q), "val_j": float(best_q_j),
                  "z_tau": float(z_tau)})

    return all_results
# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmarks_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--lookback", type=int, default=96,
                        help="Input lookback length L (paper Section 4.1, default 96).")
    parser.add_argument("--horizon", type=int, default=96,
                        help="Forecast horizon H (paper Section 4.1, default 96).")
    parser.add_argument("--backbone", type=str, default="dlinear",
                        choices=["dlinear", "itransformer", "timesnet"],
                        help="Forecasting backbone (paper Section 4.1).  "
                             "Per-cell architecture is read from configs/backbone_arch.yaml.")
    parser.add_argument("--label_types", type=str, required=True,
                        help="Comma-separated list of label types (paper Section 3.4): "
                             "mean_deg, cum_deg, rel_deg, binary_utility, ordinal_utility. "
                             "All five paper variants are usually run together.")
    parser.add_argument("--label_current_hours", type=float, default=48,
                        help="W_c in physical time (hours).  Converted to steps per dataset "
                             "using the inferred sampling interval.  Paper uses 48h.")
    parser.add_argument("--label_future_hours", type=float, default=48,
                        help="W_f in physical time (hours).  Paper uses 48h.")
    parser.add_argument("--z_tau", type=float, default=1.0,
                        help="Threshold theta in z-score units (Section 4.1, default 1.0).")
    parser.add_argument("--cooldown_hours", type=float, default=100,
                        help="Cooldown horizon N_cd in physical hours (paper Section 4.1).  "
                             "Converted to steps per dataset using the sampling interval.")
    parser.add_argument("--retrain_window", type=int, default=1000)
    parser.add_argument("--lambda_retrain", type=float, default=0.05)
    parser.add_argument("--ablation_variant", type=str, default=None,
                        choices=["n1", "wo_historical", "wo_learning_oracle"],
                        help="Ablation variant (paper Table 3): "
                             "n1 (w/o Stacking: window_len=1), "
                             "wo_historical (w/o History: 3-channel state), "
                             "wo_learning_oracle (w/o Scorer: oracle pseudo-label at test).")
    # Internal: state_mode and encoder/scorer choices are fixed to paper defaults.
    args = parser.parse_args()
    args.state_mode = "with_history"
    args.state_window_len = 24
    import builtins
    builtins._GLOBAL_ARGS = args

    # ── Apply ablation_variant settings (paper Table 3 columns). ──────────
    _abl_var = getattr(args, 'ablation_variant', None)
    if _abl_var == 'n1':
        args.state_window_len = 1
        print("[ABLATION n1] Forcing state_window_len=1 (no temporal stacking)")
    elif _abl_var == 'wo_historical':
        args.state_mode = 'no_history'
        print("[ABLATION wo_historical] state_mode=no_history (3-ch state, no rolling history)")
    elif _abl_var == 'wo_learning_oracle':
        print("[ABLATION wo_learning_oracle] Bypassing the scorer (oracle pseudo-label at test)")

    global _LOG_FILE
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = str(out_root / "run.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"[{_ts()}] Device: {device}  Datasets: {args.datasets}")

    seed = args.seed

    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    reattach_rows = []

    # Default retraining settings (fixed across cells per Section 4.1).
    best_settings: Dict[str, Dict] = {}
    for ds in args.datasets:
        best_settings[ds] = {
            "window": args.retrain_window,
            "replay": 0.0,
            "delta": -1,  # no acceptance gate
        }

    for dataset in args.datasets:
        _log(f"\n[{_ts()}] ════ Dataset: {dataset} ════")
        bs = best_settings[dataset]

        # ── Per-cell backbone architecture (paper Table 1 configs) ──
        _arch = load_backbone_arch(args.backbone, dataset)
        args.ma_kernel = _arch.get("ma_kernel", 25)
        args.d_model   = _arch.get("d_model",   64)
        args.d_ff      = _arch.get("d_ff",      64)
        args.e_layers  = _arch.get("e_layers",   2)
        args.top_k     = _arch.get("top_k",      3)
        if args.backbone == "dlinear":
            args.backbone_batch_size = 64
            args.backbone_epochs     = 20
            args.backbone_lr         = 1e-3
        else:
            args.backbone_batch_size = 128
            args.backbone_epochs     = 10
            args.backbone_lr         = 1e-4
        _log(f"  [Arch] {args.backbone}/{dataset}: {_arch}")

        # ── Physical-time cooldown: hours → per-dataset steps ──
        _step_hours = infer_step_hours(Path(args.benchmarks_dir), dataset)
        args.cooldown = max(1, round(args.cooldown_hours / _step_hours))
        _log(f"  [Cooldown] {args.cooldown_hours}h / {_step_hours:.4f}h per step = {args.cooldown} steps")
        args.min_gap = 20

        # ── Physical-time label window conversion ──
        if getattr(args, 'label_current_hours', None) is not None:
            args.label_current_window = max(1, round(args.label_current_hours / _step_hours))
            _log(f"  [LabelWindow] current={args.label_current_hours}h → {args.label_current_window} steps")
        if getattr(args, 'label_future_hours', None) is not None:
            args.label_future_window = max(1, round(args.label_future_hours / _step_hours))
            _log(f"  [LabelWindow] future={args.label_future_hours}h → {args.label_future_window} steps")

        rg = run_pilot(args, dataset, seed, device,
                         best_window=bs["window"],
                         best_replay=bs["replay"], best_delta=bs["delta"])
        for m, s in rg.items():
            reattach_rows.append({**s, "method": m})

    if reattach_rows:
        pd.DataFrame(reattach_rows).to_csv(summary_dir / "detector_reattach_results.csv", index=False)

    _log(f"\n[{_ts()}] All done. Summary saved to {summary_dir}/")
    _log(f"[{_ts()}] Best settings per dataset:")
    for ds, bs in best_settings.items():
        _log(f"  {ds}: {bs}")

if __name__ == "__main__":
    main()
