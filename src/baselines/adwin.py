"""ADWIN — adaptive sliding-window change detector (Bifet & Gavaldà, SDM 2007).

We use the river implementation (``river.drift.ADWIN``) directly as the
retraining trigger: at each step the per-step MSE is streamed to the detector,
and ``drift_detected`` is interpreted as an alarm.

Production use lives in ``scripts/run_method.py`` (``evaluate_method`` with
``method='adwin'``). Validation tuning of the ``delta`` hyperparameter is
performed inline in ``tune_and_eval_baselines``; the candidate grid is
:data:`DEFAULT_DELTA_CANDIDATES` below.
"""
from __future__ import annotations

from river.drift import ADWIN

DEFAULT_DELTA = 0.002
DEFAULT_DELTA_CANDIDATES = [0.002, 0.01, 0.05]

__all__ = ["ADWIN", "DEFAULT_DELTA", "DEFAULT_DELTA_CANDIDATES"]
