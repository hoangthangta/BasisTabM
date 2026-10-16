# BasisTabM

**TabM ensemble with a learnable basis-function branch.**

BasisTabM extends TabM with a Kolmogorov–Arnold-style basis branch, combining TabM’s BatchEnsemble path and learnable basis-function expansions through a shared weight matrix \(W\). Each of the \(k\) members is trained independently with a per-member loss, and predictions are averaged at inference time. This repository benchmarks BasisTabM against MLP, TabM, KAN variants, and 1D CNNs on two time-series regression tasks with **matched parameter budgets** per dataset.


## Repository structure

```
.
├── basis_tabm.py      # BasisTabM: BasisFunction, BasisTabM_Layer, BasisTabM, ensemble_loss
├── mlp.py             # MLP baseline
├── efficient_kan.py   # EfficientKAN (B-spline KAN)
├── bsrbf_kan.py       # BSRBF-KAN (B-spline + RBF KAN)
├── fast_kan.py        # FastKAN (RBF KAN)
├── cnn_1d.py          # CNN1D_Bike / CNN1D_Weather
├── resnet_1d.py       # ResNet1D_Bike / ResNet1D_Weather
├── ds_cnn_1d.py       # DSCNN1D_Bike / DSCNN1D_Weather (depthwise-separable)
├── schedulers.py      # CustomOneCycleLR and get_scheduler()
├── utils.py           # set_seed, count_params, RMSNorm, activations
├── run_bike.py        # UCI Bike Sharing (hourly demand)
└── run_weather.py     # Weather (MPI, 10-minute, univariate OT)
```

## Installation

```bash
pip install torch numpy pandas tqdm prettytable tabm
```

Both datasets are downloaded automatically on first run:

| Dataset | Saved to | Source |
|---|---|---|
| Bike Sharing (`hour.csv`) | `data/hour.csv` | UCI Machine Learning Repository |
| Weather (`weather.csv`) | `data/weather.csv` | THUML Time-Series-Library (Hugging Face) |

## Tasks

| | Bike Sharing | Weather |
|---|---|---|
| Target | `cnt` (hourly rentals) | `OT` (CO₂, ppm) |
| Frequency | 1 hour | 10 minutes |
| Lookback | 24 steps (24 h) | 144 steps (24 h) |
| Horizon | 1 step | 1 step (configurable via `--horizon`) |
| Historical input | 10 variables (`yr`, `mnth`, `hr`, `weekday`, weather, past `cnt`) | Univariate (past `OT` only) |
| Covariates at target time | hour / weekday / month (sin, cos), holiday, workingday, season | time of day (sin, cos); `--calendar all` adds weekday / month |
| Chronological split | 70 / 15 / 15 | 70 / 10 / 20 |
| Parameter budget | ≈ 65K | ≈ 38.7K |

The Weather setting is **univariate one-step-ahead forecasting**. It is *not* the standard
multivariate long-horizon Weather benchmark (horizons 96–720), so results are not directly
comparable with that literature.

**Protocol (both tasks)**
- Splits are chronological; scalers are fit on the training period only.
- Windows that cross missing or non-consecutive timestamps are skipped, not imputed.
- Optimizer: AdamW (lr 1e-3, weight decay 1e-4), 30 epochs, batch size 64.
- Scheduler: OneCycle with cosine warm-up (30%) and cool-down (peak = `--lr`, start = lr/25),
  then a final 10% annihilation phase towards lr/(25·10⁴); AdamW β₁ decreases from 0.95 to
  0.85 during warm-up and stays at 0.85.
- The checkpoint with the lowest validation RMSE is evaluated on the test set.
- Metrics: RMSE, MAE and R² on the original target scale.
- Naive baselines are reported on the same test targets: persistence and seasonal
  naive at 24 h and 168 h.

## Using BasisTabM

```python
import torch
from basis_tabm import BasisTabM, ensemble_loss

model = BasisTabM(net_layers=[146, 232, 1], k=4, basis="rbf",
                  norm1_type="layer", norm2_type="", activation="silu")

x = torch.randn(64, 146)
y = torch.randn(64)

out = model(x)                                         # (B, k, 1): one output per member
loss = ensemble_loss(out.squeeze(-1), y, task="regression")   # mean of per-member losses
pred = model.predict(x, task="regression")             # (B, 1): ensemble average
```

`k` is both the ensemble size and the number of grid points of the basis.
Supported bases: `rbf`, `sech`, `laplace`, `cauchy`, `mexican_hat`, `gaussian_d2`.

## Outputs

| Script | Output folder |
|---|---|
| `run_bike.py` | `output/bike/<model>/` |
| `run_weather.py` | `output/weather/<target>/lb<lookback>_h<horizon>_cal-<calendar>/<model>/` |

Each run saves the best checkpoint (`.pt`), per-epoch history (`history__*.jsonl`),
a summary (`summary__*.txt`; Weather also writes `summary__*.json`) and test
predictions with baseline columns (`predictions__*.csv`).

Use `--note` to tag a run. An existing run with the same tag is not overwritten
unless `--overwrite` is passed. For a quick smoke test, use `--epochs 1`.

## Commands

All experiments use 30 epochs. The comment above each command gives the model's
parameter count.

### Bike Sharing (≈ 65K parameters)

```bash
# 65267
python run_bike.py --model "mlp" --lookback 24 --hidden_layers "256" --batch_size 64 --epochs 30 --norm_type "batch" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite

# 65267
python run_bike.py --model "cnn_1d" --lookback 24 --batch_size 64 --epochs 30 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite

# 65267
python run_bike.py --model "resnet_1d" --lookback 24 --batch_size 64 --epochs 30 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite

# 65267
python run_bike.py --model "ds_cnn_1d" --lookback 24 --batch_size 64 --epochs 30 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite

# 65000
python run_bike.py --model "efficient_kan" --lookback 24 --batch_size 64 --hidden_layers "26" --grid_size 5 --epochs 30 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite

# 65806
python run_bike.py --model "bsrbf_kan" --lookback 24 --batch_size 64 --hidden_layers "29" --grid_size 5 --norm_type "layer" --epochs 30 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite

# 65172
python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "234" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "layer" --norm2_type "" --epochs 30 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite

# 65270
python run_bike.py --model "tabm" --lookback 24 --hidden_layers "166" --tabm_k 32 --batch_size 64 --epochs 30 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite
```

### Weather (≈ 38.7K parameters)

Univariate `OT`, 24-hour lookback, 10-minute next-step forecast, FP32 training (`--no_amp`).
Add `--calendar all` for the weekday/month ablation.

```bash
# 38693
python run_weather.py --model "mlp" --target "OT" --lookback 144 --horizon 1 --hidden_layers "256" --batch_size 64 --epochs 30 --norm_type "layer" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite --no_amp

# 38220
python run_weather.py --model "efficient_kan" --target "OT" --lookback 144 --horizon 1 --hidden_layers "26" --grid_size 5 --batch_size 64 --epochs 30 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite --no_amp

# 38717
python run_weather.py --model "bsrbf_kan" --target "OT" --lookback 144 --horizon 1 --hidden_layers "29" --grid_size 5 --norm_type "layer" --batch_size 64 --epochs 30 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite --no_amp

# 38693
python run_weather.py --model "cnn_1d" --target "OT" --lookback 144 --horizon 1 --batch_size 64 --epochs 30 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite --no_amp

# 38693
python run_weather.py --model "resnet_1d" --target "OT" --lookback 144 --horizon 1 --batch_size 64 --epochs 30 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite --no_amp

# 38693
python run_weather.py --model "ds_cnn_1d" --target "OT" --lookback 144 --horizon 1 --batch_size 64 --epochs 30 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite --no_amp

# 38752
python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "232" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "layer" --norm2_type "" --batch_size 64 --epochs 30 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite --no_amp

# 38584
python run_weather.py --model "tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "140" --tabm_k 32 --batch_size 64 --epochs 30 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite --no_amp
```
