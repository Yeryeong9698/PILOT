"""KSWIN — Kolmogorov-Smirnov windowing detector (Raab et al., 2020).

We use the river implementation (``river.drift.KSWIN``) directly. Per-step MSE
is streamed; ``drift_detected`` triggers a retrain alarm.

Production use lives in ``scripts/run_method.py`` (``evaluate_method`` with
``method='kswin'``). Validation tuning of ``(alpha, window_size)`` is
performed inline in ``tune_and_eval_baselines``; the candidate grid is
:data:`DEFAULT_ALPHA_CANDIDATES` × :data:`DEFAULT_WINDOW_CANDIDATES` below.
"""
from __future__ import annotations

from river.drift import KSWIN

DEFAULT_ALPHA = 0.005
DEFAULT_WINDOW_SIZE = 100
DEFAULT_STAT_SIZE = 30
DEFAULT_ALPHA_CANDIDATES = [0.001, 0.01, 0.05]
DEFAULT_WINDOW_CANDIDATES = [100]

__all__ = [
    "KSWIN",
    "DEFAULT_ALPHA",
    "DEFAULT_WINDOW_SIZE",
    "DEFAULT_STAT_SIZE",
    "DEFAULT_ALPHA_CANDIDATES",
    "DEFAULT_WINDOW_CANDIDATES",
]
