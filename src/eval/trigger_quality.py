"""Trigger-quality metrics for the paper's Table 2.

Given the per-step MSE stream of the No-Retrain baseline as the reference,
and a method's retrain events, compute four quality metrics:

* Precision      — fraction of triggers where utility(t) > 0.
* Trig.Need      — mean relative utility ``utility(t) / past_mean(t)`` at trigger steps.
* Episode-hit    — fraction of positive-utility episodes that contain at least
                   one trigger.
* Detection Lag  — median number of physical hours between the start of a
                   positive episode and its first trigger (only over episodes
                   that were hit).

Utility at step ``t`` is the difference between the average MSE of the next
``W`` steps and the average MSE of the previous ``W`` steps measured on the
No-Retrain stream (``W`` corresponds to the 48-hour window from Section 4.1,
converted to steps via the dataset's sampling interval).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


WINDOW_HOURS = 48.0


def _utility_streams(no_retrain_mse: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (utility(t), past_mean(t)) arrays aligned with the input MSE stream."""
    n = len(no_retrain_mse)
    util = np.full(n, np.nan)
    past = np.full(n, np.nan)
    for i in range(window, n - window):
        recent = no_retrain_mse[i - window: i].mean()
        future = no_retrain_mse[i + 1: i + 1 + window].mean()
        util[i] = future - recent
        past[i] = recent
    return util, past


def _find_episodes(positive: np.ndarray, gap_tol: int) -> List[Tuple[int, int]]:
    """Runs of 1s in ``positive``, allowing up to ``gap_tol`` interior zeros."""
    episodes: List[Tuple[int, int]] = []
    in_ep, start, last_one, zero_run = False, None, None, 0
    for i, v in enumerate(positive):
        if v == 1:
            if not in_ep:
                in_ep, start = True, i
            zero_run, last_one = 0, i
        else:
            if in_ep:
                zero_run += 1
                if zero_run > gap_tol:
                    episodes.append((start, last_one))
                    in_ep, zero_run = False, 0
    if in_ep:
        episodes.append((start, last_one))
    return episodes


def compute_trigger_quality(
    no_retrain_predictions: pd.DataFrame,
    method_retrains: pd.DataFrame,
    step_hours: float,
    window_hours: float = WINDOW_HOURS,
) -> Dict[str, float]:
    """Compute the four trigger-quality metrics for a single (cell, method).

    Parameters
    ----------
    no_retrain_predictions
        ``predictions.csv`` of the No-Retrain run (columns ``step`` and ``mse``).
    method_retrains
        ``retrains.csv`` of the candidate method.  Each row is treated
        as one trigger.
    step_hours
        Sampling interval of the dataset, in hours per step.
    window_hours
        Window length used to define utility; the paper uses 48 hours.
    """
    nr = no_retrain_predictions.sort_values("step").reset_index(drop=True)
    mse = nr["mse"].to_numpy()
    steps = nr["step"].to_numpy()
    window = max(1, round(window_hours / step_hours))
    util, past = _utility_streams(mse, window)

    trigger_steps = method_retrains["trigger_step"].dropna().astype(int).tolist()
    trigger_steps = [t for t in trigger_steps if t >= 0]

    out = {"precision": float("nan"), "trig_need": float("nan"),
           "ep_hit": float("nan"), "det_lag_hours": float("nan"),
           "n_triggers": len(trigger_steps), "n_episodes": 0, "n_hits": 0}
    if not trigger_steps:
        return out

    step_to_idx = {int(s): i for i, s in enumerate(steps)}
    trig_indices = [step_to_idx[t] for t in trigger_steps if t in step_to_idx]
    if not trig_indices:
        return out
    trig_u = util[np.asarray(trig_indices)]
    trig_p = past[np.asarray(trig_indices)]
    valid_trig = ~np.isnan(trig_u)
    if valid_trig.any():
        out["precision"] = float((trig_u[valid_trig] > 0).mean())
        rel = np.where(trig_p[valid_trig] > 0,
                        trig_u[valid_trig] / trig_p[valid_trig], np.nan)
        out["trig_need"] = float(np.nanmean(rel)) if np.any(~np.isnan(rel)) else float("nan")

    valid_mask = ~np.isnan(util)
    positive = (util > 0).astype(int)
    positive[~valid_mask] = 0
    episodes = _find_episodes(positive[valid_mask], gap_tol=max(1, window // 4))
    valid_steps = steps[valid_mask]
    out["n_episodes"] = len(episodes)
    if not episodes:
        return out

    trig_set = set(int(s) for s in trigger_steps)
    hits, lags_steps = 0, []
    for s_idx, e_idx in episodes:
        ep_steps = valid_steps[s_idx: e_idx + 1]
        in_ep = [int(s) for s in ep_steps if int(s) in trig_set]
        if in_ep:
            hits += 1
            lags_steps.append(min(in_ep) - int(ep_steps[0]))
    out["n_hits"] = hits
    out["ep_hit"] = hits / len(episodes)
    if lags_steps:
        out["det_lag_hours"] = float(np.median(lags_steps)) * step_hours
    return out
