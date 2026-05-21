"""Periodic retraining baseline.

Triggers a retrain every ``period`` steps (modulo cooldown / min-gap). The
period is tuned on validation over :data:`DEFAULT_PERIODIC_INTERVALS` by
picking the interval that minimises J = val_MSE + λ · retrain_rate.

The production decision logic is inlined in ``evaluate_method`` (``method=='periodic'``)
for speed; the class below is a reference implementation.
"""
from __future__ import annotations


DEFAULT_PERIODIC_INTERVAL = 100
DEFAULT_PERIODIC_INTERVALS = [50, 100, 200]


class PeriodicRetrainingDecider:
    """Fixed-period retraining decider.

    Parameters
    ----------
    period : int
        Number of steps between retrains (counted from the previous retrain).
    """

    def __init__(self, period: int):
        assert period > 0
        self.period = int(period)
        self._steps_since_retrain = 0

    def update(self, mse: float) -> None:  # noqa: ARG002 — period ignores observations
        self._steps_since_retrain += 1

    def should_retrain(self) -> bool:
        return self._steps_since_retrain >= self.period

    def notify_retrain(self) -> None:
        self._steps_since_retrain = 0
