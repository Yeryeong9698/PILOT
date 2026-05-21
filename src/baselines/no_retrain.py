"""No-retraining baseline.

Returns the fixed backbone's predictions for the whole test stream without ever
retraining. Used as the reference point against which other methods' MSE / RT
trade-offs are measured.

The production short-circuit lives in ``scripts/run_method.py``
(``evaluate_method`` with ``method='no_retrain'``); the class below is the
reference decider.
"""
from __future__ import annotations


class NoRetrainDecider:
    """Decider that never triggers a retrain."""

    def update(self, mse: float) -> None:  # noqa: ARG002
        pass

    def should_retrain(self) -> bool:
        return False

    def notify_retrain(self) -> None:
        pass
