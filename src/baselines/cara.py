"""CARA — staleness-based retraining baseline.

Adapted from Mahadevan & Mathioudakis, "Cost-Aware Retraining for Machine Learning"
(KBS 2024). Tracks staleness = recent_mse - baseline_mse and triggers retrain when
staleness (variant T) or cumulative staleness (variant C) exceeds threshold tau.

The decider is consumed by ``scripts/run_method.py`` (``evaluate_method`` with
``method='cara'``); validation tuning is performed inline in
``tune_and_eval_baselines``.
"""
from __future__ import annotations

import numpy as np


class CARARetrainingDecider:
    """Staleness-based retraining decider.

    Parameters
    ----------
    variant : {"T", "C"}
        T = trigger when current staleness > tau.
        C = trigger when cumulative staleness > tau (resets on retrain).
    tau : float
        Staleness threshold.
    baseline_window : int
        Window length used to (re-)estimate the baseline MSE.
    min_observations : int
        Minimum observations before any trigger is allowed.
    """

    def __init__(self, variant: str = "T", tau: float = 0.05,
                 baseline_window: int = 50, min_observations: int = 20):
        assert variant in ("T", "C"), f"variant must be 'T' or 'C', got {variant!r}"
        self.variant = variant
        self.tau = tau
        self.baseline_window = baseline_window
        self.min_observations = min_observations
        self._mse_history: list[float] = []
        self._baseline_mse: float = 0.0
        self._cumulative_staleness: float = 0.0
        self._initialized: bool = False

    def update(self, mse: float) -> None:
        self._mse_history.append(mse)
        if not self._initialized and len(self._mse_history) >= self.baseline_window:
            self._baseline_mse = float(np.mean(self._mse_history[-self.baseline_window:]))
            self._initialized = True

    def _current_staleness(self) -> float:
        if not self._initialized or len(self._mse_history) < 5:
            return 0.0
        recent_w = min(10, len(self._mse_history))
        recent_mse = float(np.mean(self._mse_history[-recent_w:]))
        return max(0.0, recent_mse - self._baseline_mse)

    def should_retrain(self) -> bool:
        if len(self._mse_history) < self.min_observations or not self._initialized:
            return False
        staleness = self._current_staleness()
        if self.variant == "T":
            return staleness > self.tau
        if self.variant == "C":
            self._cumulative_staleness += staleness
            return self._cumulative_staleness > self.tau
        return False

    def notify_retrain(self) -> None:
        if self.variant == "C":
            self._cumulative_staleness = 0.0
        if len(self._mse_history) >= self.baseline_window:
            self._baseline_mse = float(np.mean(self._mse_history[-self.baseline_window:]))
