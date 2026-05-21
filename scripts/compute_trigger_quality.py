"""Compute the four trigger-quality metrics (paper Table 2) per cell.

Reads the existing run_method.py output tree and writes a single CSV with
``backbone, dataset, method, precision, trig_need, ep_hit, det_lag_hours``.

Usage:
    python scripts/compute_trigger_quality.py \\
        --output_root ./results_table1 \\
        --benchmarks_dir ./data \\
        --out ./trigger_quality.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import infer_step_hours
from src.eval import compute_trigger_quality


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", type=Path, required=True,
                    help="Same directory passed as OUTPUT_ROOT to reproduce_table1.sh")
    ap.add_argument("--benchmarks_dir", type=Path, default=Path("./data"))
    ap.add_argument("--out", type=Path, default=Path("./trigger_quality.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbones", nargs="+",
                    default=["dlinear", "itransformer", "timesnet"])
    ap.add_argument("--datasets", nargs="+",
                    default=["ETTh2", "ETTm1", "Traffic"])
    ap.add_argument("--methods", nargs="+",
                    default=["periodic", "adwin", "kswin", "cara", "upf",
                             "pilot_mean_deg", "pilot_cum_deg", "pilot_rel_deg",
                             "pilot_binary_utility", "pilot_ordinal_utility",
                             "voting"])
    args = ap.parse_args()

    rows = []
    for bb in args.backbones:
        for ds in args.datasets:
            # No-Retrain predictions are saved under the FIRST label's output
            # directory.  reproduce_table1.sh uses mean_deg as the first label.
            nr_path = (args.output_root / f"{bb}_{ds}_mean_deg" / "results" /
                       ds / "no_retrain" / f"seed_{args.seed}" / "predictions.csv")
            if not nr_path.exists():
                continue
            try:
                step_hours = infer_step_hours(args.benchmarks_dir, ds)
            except FileNotFoundError:
                continue
            nr_df = pd.read_csv(nr_path)

            for m in args.methods:
                if m.startswith("pilot_"):
                    label = m.removeprefix("pilot_")
                    rt_path = (args.output_root / f"{bb}_{ds}_{label}" / "results" /
                               ds / "pilot" / f"seed_{args.seed}" / "retrains.csv")
                elif m == "voting":
                    rt_path = (args.output_root / f"{bb}_{ds}_voting" / "results" /
                               ds / "voting" / f"seed_{args.seed}" / "retrains.csv")
                else:
                    rt_path = (args.output_root / f"{bb}_{ds}_mean_deg" / "results" /
                               ds / m / f"seed_{args.seed}" / "retrains.csv")
                if not rt_path.exists():
                    continue
                rt_df = pd.read_csv(rt_path)
                metrics = compute_trigger_quality(nr_df, rt_df, step_hours)
                rows.append({"backbone": bb, "dataset": ds, "method": m, **metrics})

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    if not df.empty:
        print(df.groupby("method")[["precision", "trig_need",
                                     "ep_hit", "det_lag_hours"]].mean())


if __name__ == "__main__":
    main()
