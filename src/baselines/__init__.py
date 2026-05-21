"""Baseline retraining triggers used in the paper.

Six baselines:
  - :mod:`.no_retrain`  — never retrain
  - :mod:`.periodic`    — fixed-period retrain
  - :mod:`.adwin`       — ADWIN drift detector (river)
  - :mod:`.kswin`       — KSWIN drift detector (river)
  - :mod:`.cara`        — staleness-based decider (Mahadevan & Mathioudakis 2024)
  - :mod:`.upf`         — UPF (uncertainty-based proactive forecasting)

Production decision logic is inlined in ``scripts/run_method.py`` for
performance; this package documents the algorithms and exposes the decider
classes for inspection/reuse.
"""
from .adwin import ADWIN
from .cara import CARARetrainingDecider
from .kswin import KSWIN
from .no_retrain import NoRetrainDecider
from .periodic import PeriodicRetrainingDecider
from .upf import UPF

__all__ = [
    "ADWIN",
    "CARARetrainingDecider",
    "KSWIN",
    "NoRetrainDecider",
    "PeriodicRetrainingDecider",
    "UPF",
]
