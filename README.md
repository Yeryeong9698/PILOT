# PILOT (Pseudo-label-Informed Learned Online Trigger)


<div align="center">
    <h3>Pseudo-Label-Triggered Retraining from Forecast Errors for Online Time Series Forecasting</h3>
    <!-- figure -->
    <!-- <br> -->
    <img src="https://anonymous.4open.science/r/PILOT-B641/pic/Overall_framework_of_PILOT.jpg" alt="PILOT Framework" width="600"/>

</div>

Code supplement for the submission.

This supplement provides a self-contained reproducer covering **three datasets (ETTh2, ETTm1, Traffic)** across **three forecasting backbones (DLinear, iTransformer, TimesNet)** — nine (backbone, dataset) cells in total.  
Configurations for the additional cells reported in the paper follow each backbone's original recommended training setup and are available upon request.

## 1. Setup

- Python 3.10 or newer.
- Create a virtual environment, then install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- PyTorch ≥ 2.0 (CUDA optional but recommended).

## 2. Data

Each of the three datasets is available from the public Time-Series-Library [1] release.  
Place the CSV files under `./data/<dataset>/Y_df.csv` (or pass a custom path via `--benchmarks_dir`).  
Splits follow the standard convention used in long-term forecasting work: ETT 60/20/20, others 70/10/20 (train/val/test); 
the validation slice is further halved to obtain a scorer-training split.  
See `chronological_split()` in `scripts/run_method.py`.

## 3. Forecasting backbones

We use three forecasting backbones spanning different architectural
families:

- **DLinear** [2] — linear family
- **iTransformer** [3] — Transformer family
- **TimesNet** [4] — convolutional family

iTransformer and TimesNet use the official Time-Series-Library model architectures, vendored under `tslib/`.  
DLinear is implemented in `src/forecasting/dlinear.py`.  
All training in this supplement uses the training code in `scripts/run_method.py`; backbone architecture hyperparameters per cell live in `configs/backbone_arch.yaml`.  
The backbone is trained on the first split only and shared across all retraining methods, so performance differences are attributable solely to each method's retraining policy.

## 4. Run PILOT on a single (backbone, dataset)

```bash
python scripts/run_method.py \
  --backbone timesnet --datasets ETTm1 --seed 42 \
  --label_types mean_deg \
  --benchmarks_dir ./data \
  --output_dir ./outputs/timesnet_ETTm1
```

The command runs the full PILOT pipeline together with all six baselines (No Retrain, Periodic, ADWIN, KSWIN, CARA, UPF) on the specified cell.  
Per-method results are written to
`./outputs/timesnet_ETTm1_<label>/results/<dataset>/<method>/seed_<N>/{metrics,predictions,retrains,alarms,tuned_params,config}.*`.
You can use multiple `label_types` from `mean_deg,cum_deg,rel_deg,binary_utility,ordinal_utility` separated with comma, where `mean_deg` is the default pseudo-label type for PILOT. The others are extended alternatives.

Each `metrics.json` contains the test-set `mse`, `mae`, `cumulative_mse`, `retrain_count`, and `alarm_count`.
`tuned_params.json` records the hyperparameter selected on the validation split.

## 5. Evaluating on additional datasets

The pipeline accepts any multivariate time series in the Time-Series-Library [1] CSV format 
(a `date` column followed by variate columns).  
To run PILOT on a dataset other than the three listed above:

1. Place the CSV at `<benchmarks_dir>/<dataset>/Y_df.csv`.
2. (Optional) Add an entry in `configs/backbone_arch.yaml` to pin
   backbone architecture hyperparameters for the (backbone, dataset)
   cell; otherwise, the backbone's standard defaults are used.
3. Pass `--datasets <dataset>` to `scripts/run_method.py`.

The remaining five benchmarks reported in Section 4.1 of the paper
(ETTh1, ETTm2, Electricity, Exchange, Weather) are likewise available
from the Time-Series-Library release.  Per-cell configurations used
in the paper for these datasets follow each backbone's original
recommended setup and can be provided upon request.

## 6. Reproducing the paper tables

### Quick smoke test (≈5 min on one GPU)

```bash
python scripts/run_method.py \
  --backbone dlinear --datasets ETTh2 --seed 42 \
  --label_types mean_deg --output_dir ./outputs/smoke
```
A successful run writes `./outputs/smoke_mean_deg/results/ETTh2/pilot/seed_42/metrics.json`
with an `mse` field; the expected value is approximately 0.22.

### Full table reproduction

**Table 1 — Main MSE results** (9 cells × 12 methods)
```bash
bash scripts/reproduce_table1.sh
```

**Table 2 — Trigger quality**
```bash
bash scripts/reproduce_table2.sh ./results_table1 42
```

**Table 3 — Ablation** (TimesNet × 3 cells × {PILOT, w/o Scorer, w/o History, w/o Stacking})
```bash
bash scripts/reproduce_table3.sh
```

The scripts accept `OUTPUT_ROOT` and `SEED` as positional arguments
and read the dataset root from the `DATA_ROOT` environment variable
(default `./data`).  Results in the paper are averaged over three
random seeds; the provided scripts default to a single seed as a
sanity-check reproduction.  Single-seed results may deviate from the
paper's three-seed averages by a few percent; this is expected.
Re-run with different `SEED` values and average per-cell MSE / retrain
count.

## 7. Hyperparameters

Method and baseline hyperparameters follow paper Section 4.1; defaults
are encoded in `scripts/run_method.py` and the baseline tuning grids in
`src/baselines/*.py`.

## 8. Compute requirements

Reference runtimes on a single NVIDIA RTX 3090 (24 GB):

| Component | Wall-time per cell | Wall-time for full table |
|---|---|---|
| DLinear cell (one dataset) | ≈5–10 min | — |
| iTransformer cell (one dataset) | ≈15–25 min | — |
| TimesNet cell (one dataset) | ≈25–60 min | — |
| **Table 1** (9 cells) | — | ≈3–6 h |
| **Table 3** (TimesNet × 3 cells × 4 variants) | — | ≈6–10 h |

CPU-only runs are possible but several times slower; 
in that case we recommend starting from the quick smoke test above.

## 9. Code layout

```text
code_supplement/
├── configs/
│   └── backbone_arch.yaml        # per-cell backbone architecture
├── src/
│   ├── method/                   # PILOT scorer, state representation, label construction
│   ├── baselines/                # No Retrain, Periodic, ADWIN, KSWIN, CARA, UPF
│   ├── forecasting/              # backbone wrappers (DLinear / iTransformer / TimesNet)
│   ├── data/                     # benchmark loader and chronological split
│   └── eval/                     # trigger-quality metrics (Table 2)
├── scripts/
│   ├── run_method.py             # main entry point: per-cell PILOT + baselines
│   ├── compute_trigger_quality.py
│   ├── reproduce_table1.sh
│   ├── reproduce_table2.sh
│   └── reproduce_table3.sh
├── tslib/                        # vendored Time-Series-Library subset (iTransformer + TimesNet)
├── LICENSE
├── README.md
└── requirements.txt
```

## 10. License

This supplement is released under the MIT License for review purposes.
Vendored Time-Series-Library code under `tslib/` retains its original MIT license; 
see `tslib/LICENSE`.

## References

[1] Time-Series-Library — <https://github.com/thuml/Time-Series-Library>  
[2] DLinear — Zeng et al., "Are transformers effective for time series forecasting?," AAAI 2023.  
[3] iTransformer — Liu et al., "iTransformer: Inverted transformers are effective for time series forecasting," ICLR 2024.  
[4] TimesNet — Wu et al., "Timesnet: Temporal 2d-variation modeling for general time series analysis," ICLR 2023.  
