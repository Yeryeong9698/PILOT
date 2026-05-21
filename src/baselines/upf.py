"""
UPF Reimplementation for Streaming Time-Series Forecasting
====================================================================
Regol et al., "Towards Retraining-Free Adaptable Networks via Online Learning
of Performance Forecasters", ICML 2025.

Adaptations from the original:
  - Performance variable: MSE (unbounded) instead of bounded 0-1 accuracy
  - Distribution model: Gaussian (original Appendix A.8 allows this for unbounded loss)
  - Forecaster: ElasticNetCV (identical to original)
  - Features: [model_age, eval_step, gap] (no extra TS-specific feature)
  - Training data: cross-product over K evenly-spaced detector-training checkpoints
  - Decision rule: forward simulation with horizon truncation (H_lookahead << T_test)
    to avoid O(2^T_test) exhaustive enumeration (original T_test ~ 8, ours ~ 1000-5000)
  - Quantile δ: 0.95 (identical to original)
  - Online forecaster update: refit every R steps on accumulated data
"""
from __future__ import annotations

import copy
import time
from typing import List, Optional, Tuple

import numpy as np

try:
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    print("[UPF] WARNING: sklearn not available. Install scikit-learn.")


# ---------------------------------------------------------------------------
# Training-data generation (offline)
# ---------------------------------------------------------------------------

def build_upf_training_data(
    det_data: np.ndarray,
    backbone_fcst,
    lookback: int,
    horizon: int,
    retrain_window: int,
    K: int = 8,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build cross-product (model_k, eval_step_j) dataset for ElasticNet training.

    Parameters
    ----------
    det_data         : detector training split data (T_det, D)
    backbone_fcst    : pre-trained backbone forecaster (clone()d internally)
    lookback, horizon: L, H
    retrain_window   : number of steps to use for each checkpoint retrain
    K                : number of evenly-spaced retrain checkpoints

    Returns
    -------
    X_train : (N_pairs, 3) float32   features = [model_age, eval_step, gap]
    y_train : (N_pairs,) float32     MSE values
    """
    if not _SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn required for UPF")

    T_det = len(det_data)
    L, H = lookback, horizon

    # When T_det < retrain_window the history buffer is truncated, mirroring
    # the behaviour of the online evaluation loop.
    # Minimum required to train a backbone is L + H.
    t_start_ideal = L + retrain_window
    t_start = min(t_start_ideal, T_det // 2)
    t_start = max(t_start, L + H)
    t_end = T_det - H
    
    if t_start >= t_end:
        raise ValueError(f"det_data too short: T_det={T_det}, need > {t_start + H}")

    checkpoints = np.linspace(t_start, t_end, K, dtype=int).tolist()
    checkpoints = sorted(set(checkpoints))  # deduplicate

    if verbose:
        print(f"[UPF] Building training data: K={len(checkpoints)} checkpoints, "
              f"T_det={T_det}")

    X_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    n_skip = 0

    for k_idx, t_k in enumerate(checkpoints):
        # Retrain backbone at checkpoint t_k
        recent = det_data[max(0, t_k - retrain_window): t_k]
        if len(recent) < L + H:
            n_skip += 1
            continue

        candidate = backbone_fcst.clone()
        # Paper Section 4.1: warm-restart with 5 epochs at lr * 0.1.
        candidate.warm_retrain(recent, epochs=5, lr_multiplier=0.1)

        # Evaluate candidate at all future eval steps j > t_k
        eval_steps = [t_j for t_j in range(t_k + 1, T_det - H + 1)]
        if not eval_steps:
            continue

        cand_results = candidate.rolling_predict(det_data[t_k:], start=L)
        for r in cand_results:
            t_j_local = r["t"]
            t_j_global = t_k + t_j_local

            if t_j_global >= T_det - H + 1:
                break

            mse_ij = float(np.mean((r["y_true"] - r["yhat"]) ** 2))

            # Feature vector: [model_age, eval_step, gap] — matches the original UPF
            model_age = float(t_k)
            eval_step = float(t_j_global)
            gap = float(t_j_global - t_k)

            X_rows.append(np.array([model_age, eval_step, gap], dtype=np.float32))
            y_rows.append(mse_ij)

    if verbose:
        print(f"[UPF] Training pairs: {len(X_rows)} (skipped {n_skip} checkpoints)")

    if len(X_rows) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)

    X_train = np.stack(X_rows, axis=0).astype(np.float32)
    y_train = np.array(y_rows, dtype=np.float32)
    return X_train, y_train


# ---------------------------------------------------------------------------
# ElasticNet performance forecaster
# ---------------------------------------------------------------------------

class UPFPerformanceForecaster:
    """
    Trains ElasticNetCV on (model_age, eval_step, gap) → MSE.
    Online updates append new (X, y) pairs and refit every R steps.

    Matches original UPF: μ_φ from ElasticNet, σ_φ = constant (residual std).
    """

    def __init__(self, refit_interval: int = 10, verbose: bool = False):
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required")
        self.refit_interval = refit_interval
        self.verbose = verbose
        self.scaler = StandardScaler()
        self.model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
            cv=3,
            max_iter=2000,
            n_jobs=1,
            random_state=42,
        )
        self.sigma: float = 0.1        # residual std; updated after fit
        self.is_fitted: bool = False
        self._X_buffer: List[np.ndarray] = []
        self._y_buffer: List[float] = []
        self._steps_since_refit: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "UPFPerformanceForecaster":
        """Initial fit on detector-training data."""
        if len(X) < 10:
            if self.verbose:
                print("[UPFForecaster] Too few samples to fit ElasticNet. Skipping.")
            return self
        X_sc = self.scaler.fit_transform(X)
        self.model.fit(X_sc, y)
        residuals = y - self.model.predict(X_sc)
        self.sigma = float(np.std(residuals)) if len(residuals) > 1 else 0.1
        self.sigma = max(self.sigma, 1e-6)
        self.is_fitted = True
        # Seed online buffer with training data
        self._X_buffer = list(X)
        self._y_buffer = list(y)
        if self.verbose:
            print(f"[UPFForecaster] Fitted ElasticNet: alpha={self.model.alpha_:.4f} "
                  f"l1_ratio={self.model.l1_ratio_:.2f} sigma={self.sigma:.4f} "
                  f"n_samples={len(X)}")
        return self

    def predict_mu(self, X: np.ndarray) -> np.ndarray:
        """Predict mean MSE for feature matrix X. Returns (N,) float array."""
        if not self.is_fitted:
            return np.full(len(X), 1.0)
        X_sc = self.scaler.transform(X)
        return self.model.predict(X_sc).astype(np.float64)

    def predict_sigma(self) -> float:
        return self.sigma

    def online_update(self, x: np.ndarray, y_obs: float):
        """Append one observation and refit if interval reached."""
        self._X_buffer.append(x.copy())
        self._y_buffer.append(y_obs)
        self._steps_since_refit += 1
        if self._steps_since_refit >= self.refit_interval and len(self._X_buffer) >= 10:
            X_all = np.stack(self._X_buffer, axis=0)
            y_all = np.array(self._y_buffer, dtype=np.float64)
            X_sc = self.scaler.transform(X_all)   # keep original scaler
            try:
                self.model.fit(X_sc, y_all)
                residuals = y_all - self.model.predict(X_sc)
                self.sigma = max(float(np.std(residuals)), 1e-6)
            except Exception:
                pass
            self._steps_since_refit = 0


# ---------------------------------------------------------------------------
# Online decision rule (forward simulation with MC)
# ---------------------------------------------------------------------------

class UPFDecider:
    """
    UPF decision rule with truncated forward simulation + Monte Carlo.

    At each step t:
      1. Identify features for current context (model_age, eval_step, gap)
      2. Simulate H_lookahead steps forward for KEEP and RETRAIN branches
      3. For each branch, draw N_mc samples from N(μ_φ, σ_φ) for each future step
      4. Compute δ-quantile of cumulative cost for each branch
      5. Retrain if quantile(retrain_branch) < quantile(keep_branch)
    """

    def __init__(
        self,
        forecaster: UPFPerformanceForecaster,
        alpha: float = 0.1,            # retraining cost penalty
        H_lookahead: int = 100,        # forward simulation window
        delta: float = 0.95,           # quantile level (original UPF: 0.95)
        N_mc: int = 100,               # Monte Carlo samples
        min_observations: int = 20,    # warm-up period
        cooldown: int = 0,
        min_gap: int = 0,
        verbose: bool = False,
    ):
        self.forecaster = forecaster
        self.alpha = alpha
        self.H_lookahead = H_lookahead
        self.delta = delta
        self.N_mc = N_mc
        self.min_observations = min_observations
        self.cooldown = cooldown
        self.min_gap = min_gap
        self.verbose = verbose

        self._current_step: int = 0
        self._last_retrain_step: int = 0
        self._mse_history: List[float] = []
        self._steps_since_retrain: int = 0

    def update(self, mse: float):
        """Call after each forecast step with observed MSE."""
        self._current_step += 1
        self._mse_history.append(mse)

        # Online update of forecaster with current observation
        if self.forecaster.is_fitted:
            x_obs = self._make_feature(
                model_age=float(self._last_retrain_step),
                eval_step=float(self._current_step),
                gap=float(self._current_step - self._last_retrain_step),
            )
            self.forecaster.online_update(x_obs, mse)

    def notify_retrain(self, step: int):
        """Call when retrain is triggered."""
        self._last_retrain_step = step
        self._steps_since_retrain = 0

    def _make_feature(self, model_age: float, eval_step: float, gap: float) -> np.ndarray:
        """Build 3-dim feature vector for forecaster (matches the original UPF)."""
        return np.array([model_age, eval_step, gap], dtype=np.float32)

    def should_retrain(self) -> bool:
        """
        Returns True if the δ-quantile of retrain cost < δ-quantile of keep cost
        over the next H_lookahead steps.
        """
        # Warm-up guard
        if len(self._mse_history) < self.min_observations:
            return False
        if not self.forecaster.is_fitted:
            return False

        # Cooldown / min_gap guards
        steps_since = self._current_step - self._last_retrain_step
        if steps_since < self.cooldown or steps_since < self.min_gap:
            return False

        t = self._current_step
        t_last = self._last_retrain_step
        sigma = self.forecaster.predict_sigma()

        # ── KEEP branch ──────────────────────────────────────────────────────
        # Future steps t+1, ..., t+H assuming current stale model
        keep_mu = np.zeros(self.H_lookahead, dtype=np.float64)
        for h in range(1, self.H_lookahead + 1):
            x = self._make_feature(
                model_age=float(t_last),
                eval_step=float(t + h),
                gap=float(t + h - t_last),
            )
            keep_mu[h - 1] = float(self.forecaster.predict_mu(x[None])[0])

        # MC samples: each future step ~ N(mu, sigma), cumulative sum
        rng = np.random.default_rng(seed=t % (2**31))
        keep_samples = rng.normal(
            loc=keep_mu,                          # (H,)
            scale=sigma,
            size=(self.N_mc, self.H_lookahead),   # (N_mc, H)
        ).clip(0)   # MSE non-negative
        keep_costs = keep_samples.sum(axis=1)     # (N_mc,) cumulative cost
        keep_quantile = float(np.quantile(keep_costs, self.delta))

        # ── RETRAIN branch ────────────────────────────────────────────────────
        # If we retrain NOW: new model at step t, evaluated for h=1,...,H
        retrain_mu = np.zeros(self.H_lookahead, dtype=np.float64)
        for h in range(1, self.H_lookahead + 1):
            x = self._make_feature(
                model_age=float(t),       # freshly retrained at t
                eval_step=float(t + h),
                gap=float(h),
            )
            retrain_mu[h - 1] = float(self.forecaster.predict_mu(x[None])[0])

        retrain_samples = rng.normal(
            loc=retrain_mu,
            scale=sigma,
            size=(self.N_mc, self.H_lookahead),
        ).clip(0)
        retrain_costs = retrain_samples.sum(axis=1) + self.alpha  # +α for one-time retrain cost
        retrain_quantile = float(np.quantile(retrain_costs, self.delta))

        decision = retrain_quantile < keep_quantile

        if self.verbose:
            print(f"  [UPF] t={t}  gap={steps_since}  "
                  f"keep_q={keep_quantile:.4f}  retrain_q={retrain_quantile:.4f}  "
                  f"→ {'RETRAIN' if decision else 'KEEP'}")

        return decision


# ---------------------------------------------------------------------------
# Full UPF class (integrates training + online decision)
# ---------------------------------------------------------------------------

class UPF:
    """
    Full UPF implementation.

    Usage
    -----
    1. upf = UPF(alpha=0.1, H_lookahead=100, ...)
    2. upf.fit_forecaster(det_data, backbone_fcst, ...)
    3. upf.reset()                                         # reset online state
    4. At each test step t:
         upf.update(observed_mse)
         if upf.should_retrain():
             do_retrain()
             upf.notify_retrain(t)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        H_lookahead: int = 100,
        delta: float = 0.95,
        N_mc: int = 100,
        K_checkpoints: int = 8,
        refit_interval: int = 10,
        min_observations: int = 20,
        cooldown: int = 0,
        min_gap: int = 0,
        verbose: bool = False,
    ):
        self.alpha = alpha
        self.H_lookahead = H_lookahead
        self.delta = delta
        self.N_mc = N_mc
        self.K_checkpoints = K_checkpoints
        self.refit_interval = refit_interval
        self.min_observations = min_observations
        self.cooldown = cooldown
        self.min_gap = min_gap
        self.verbose = verbose

        self.perf_forecaster = UPFPerformanceForecaster(
            refit_interval=refit_interval, verbose=verbose
        )
        self.decider: Optional[UPFDecider] = None
        # Snapshot of perf_forecaster captured right after fit_forecaster() so that
        # reset() can restore the post-offline-fit state without leaking online
        # observations from prior eval runs (e.g., across hyperparameter tuning).
        self._forecaster_snapshot: Optional[UPFPerformanceForecaster] = None

    def fit_forecaster(
        self,
        det_data: np.ndarray,
        backbone_fcst,
        lookback: int,
        horizon: int,
        retrain_window: int,
    ):
        """Generate training data and fit the performance forecaster."""
        t0 = time.perf_counter()
        X_train, y_train = build_upf_training_data(
            det_data=det_data,
            backbone_fcst=backbone_fcst,
            lookback=lookback,
            horizon=horizon,
            retrain_window=retrain_window,
            K=self.K_checkpoints,
            verbose=self.verbose,
        )
        if len(X_train) > 0:
            self.perf_forecaster.fit(X_train, y_train)
        else:
            if self.verbose:
                print("[UPF] WARNING: No training data generated. Forecaster unfitted.")
        # Capture clean snapshot of post-offline-fit forecaster state.  reset()
        # restores from this so each online eval pass starts from the same fit.
        self._forecaster_snapshot = copy.deepcopy(self.perf_forecaster)
        if self.verbose:
            print(f"[UPF] fit_forecaster done in {time.perf_counter()-t0:.1f}s")

    def reset(self, reset_forecaster: bool = True):
        """Reset online state for new test run.

        Args:
            reset_forecaster: If True (default), restore perf_forecaster to its
                post-fit_forecaster() snapshot so online observations from prior
                runs do not leak into this one. Set False only when intentionally
                chaining online sessions on the same instance.
        """
        if reset_forecaster and self._forecaster_snapshot is not None:
            self.perf_forecaster = copy.deepcopy(self._forecaster_snapshot)
        self.decider = UPFDecider(
            forecaster=self.perf_forecaster,
            alpha=self.alpha,
            H_lookahead=self.H_lookahead,
            delta=self.delta,
            N_mc=self.N_mc,
            min_observations=self.min_observations,
            cooldown=self.cooldown,
            min_gap=self.min_gap,
            verbose=self.verbose,
        )

    def update(self, mse: float):
        if self.decider is None:
            self.reset()
        self.decider.update(mse)

    def should_retrain(self) -> bool:
        if self.decider is None:
            return False
        return self.decider.should_retrain()

    def notify_retrain(self, step: int):
        if self.decider is not None:
            self.decider.notify_retrain(step)
