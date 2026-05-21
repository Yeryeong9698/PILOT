"""
Abstract base class for forecasting backbones.

All forecasting models (DLinear, iTransformer, TimesNet) implement this
interface, which the retraining loop uses to drive rolling forecasts and
warm-start retraining.

Data convention:
  - All data arrays are (T, D) float32 where T=time steps, D=features.
  - predict_single() takes a (lookback, D) window and returns a dict.
  - rolling_predict() returns a list of dicts, one per forecast step.

Return dict keys:
  Required: "yhat", "y_true", "residual"  — all (H, D)
  Optional: "yhat_trend", "yhat_seasonal"  — only for decomposition models (e.g., DLinear)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np


class BaseForecaster(ABC):
    """Abstract forecaster interface for the retraining pipeline."""

    @property
    @abstractmethod
    def config(self):
        """Return the forecaster config object."""
        ...

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the model has been trained."""
        ...

    @abstractmethod
    def fit(self, data: np.ndarray, verbose: bool = False) -> "BaseForecaster":
        """
        Fit on data: (T, D) float32.
        Returns self for chaining.
        """
        ...

    @abstractmethod
    def predict_single(self, window: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict on a single lookback window.
        window: (lookback, D)
        Returns dict with at minimum: {"yhat": (H, D)}
        """
        ...

    @abstractmethod
    def rolling_predict(
        self, data: np.ndarray, start: int = 0
    ) -> List[Dict[str, np.ndarray]]:
        """
        Sequentially produce one forecast per time step.
        Returns list of dicts, each with:
          - "yhat": (H, D)
          - "y_true": (H, D)
          - "residual": (H, D)
          - "t": int (time step index)
          Optional (decomposition models only):
          - "yhat_trend": (H, D)
          - "yhat_seasonal": (H, D)
        """
        ...

    @abstractmethod
    def retrain(self, data: np.ndarray, reset_stats: bool = True) -> "BaseForecaster":
        """
        Retrain the model on new data.
        reset_stats=True  → cold start (recompute normalization)
        reset_stats=False → warm start (keep existing stats)
        """
        ...

    @abstractmethod
    def warm_retrain(
        self,
        data: np.ndarray,
        epochs: int = 5,
        lr_multiplier: float = 0.1,
    ) -> "BaseForecaster":
        """
        Warm-start fine-tuning: keep model weights AND normalization stats.
        Uses reduced learning rate.
        """
        ...

    @abstractmethod
    def clone(self) -> "BaseForecaster":
        """Return a deep copy of this forecaster."""
        ...
