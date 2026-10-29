# BasisTabM

**A TabM-style ensemble with learnable basis functions for time series forecasting.**

BasisTabM is a BatchEnsemble/TabM network (k members sharing one set of weights, each with its own
random-sign adapters) extended with Kolmogorov–Arnold-style basis functions:

1. an **input feature basis (FB)**: every input feature first passes through its own learnable
   1-D function $u_i = x_i + \sum_{g} c_{ig}\,\phi_g(x_i)$ (Gaussian RBFs, coefficients initialized at 0),
   applied once before the input is expanded to the k members;
2. optionally, an **α-mixed basis branch (Mix) in every layer**, which reuses the shared weight $W$ and is
   mixed with the linear path by a learnable per-member weight $\alpha$ (initialized at 0):
   $h = (1-\alpha)\,h_{\text{lin}} + \alpha\, W \sum_g \phi_g(x)$;
3. optionally, a **normalization after the hidden layer** (`norm2`): BatchNorm with separate
   statistics per member, or LayerNorm.

Each of the k members is trained on the target with its own loss; the prediction is the mean of
the members. The repository benchmarks BasisTabM against MLP, EfficientKAN, BSRBF-KAN, TabM,
DLinear and PatchTST on two forecasting tasks, with **matched parameter budgets** within each
setting (DLinear is a lightweight reference and is not budget-matched).

## Model variants

All variants use random-sign (±1) adapter initialization, SiLU, no dropout and no `norm1`.
They differ only in the flags below; each is run with k = 2, 4, 8, 16, 32 and
`norm2` = none / BatchNorm / LayerNorm, giving 12 variants × 5 k = 60 configurations per dataset.

| Name | What it uses | Flags |
|---|---|---|
| **NoFB** | BatchEnsemble backbone, no basis at all | `--basis_mode none --no-feature_basis` |
| **NoFB+Mix** | + α-mixed basis branch, no input feature basis | `--basis_mode mix --no-feature_basis` |
| **FB** | + input feature basis | `--basis_mode none` (feature basis is on by default) |
| **FB+Mix** | input feature basis + α-mixed basis branch | `--basis_mode mix` |
| **…+Norm** | any of the above + normalization after the hidden layer | add `--norm2_type batch` or `--norm2_type layer` |

The run scripts default to `--norm1_type ""` (no `norm1`). When building `BasisTabM` directly in
Python, pass `norm1_type=""` and `norm2_type=""` explicitly, since the constructor defaults are
`"layer"`.

## Repository structure

```
.
├── models/
│   ├── basis_tabm.py        # BasisFunction, FeatureBasis, BasisTabM_Layer, BasisTabM, ensemble_loss
│   ├── mlp.py               # MLP baseline
│   ├── efficient_kan.py     # EfficientKAN (B-spline KAN)
│   ├── bsrbf_kan.py         # BSRBF-KAN (B-spline + RBF KAN)
│   ├── dlinear.py           # DLinear (moving-average decomposition + linear layers)
│   ├── patchtst_model.py    # wrapper around the official PatchTST code
│   ├── patchtst_official/   # PatchTST backbone, layers and RevIN
│   └── fast_kan.py          # FastKAN (kept for reference; not used by the run scripts)
├── schedulers.py            # CustomOneCycleLR and get_scheduler()
├── utils.py                 # set_seed, count_params, RMSNorm, activations
├── run_bike.py              # UCI Bike Sharing (hourly demand)
├── run_weather.py           # Weather (MPI, 10-minute, univariate OT)
├── run_bike_h24_cal_all.sh  # Bike, L=H=24 + calendar: all models x 3 seeds
├── run_weather_L96_H96_all.sh # Weather, L=H=96: all models x 3 seeds
└── LICENSE
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
| Target | `cnt` (hourly rentals) | `OT` |
| Frequency | 1 hour | 10 minutes |
| Lookback | 24 steps (24 h) | 96 steps (16 h) |
| Horizon | 24 steps, all predicted at once (`--horizon 24`) | 96 steps, all predicted at once (`--horizon 96 --multi_output`) |
| Input features | 5 weather + 8 calendar (`--calendar`) + past `cnt`, for each lookback hour (14 × 24 = 336) | univariate: past `OT` only (`--calendar none`, 96 features) |
| Chronological split | 70 / 15 / 15 | 70 / 10 / 20 |
| Parameter budget | ≈ 93.6K | ≈ 50.2K |
| Mixed precision | fp16 AMP on GPU | off (`--no_amp`) |

Notes:
- **Bike calendar features** (hour, weekday and month as sin/cos, `holiday`, `workingday`) describe the
  lookback hours only; no information about the forecast hours is given to the models.
- **Weather is univariate** (OT → OT, i.e. TSLib `--features S`). It is not the usual multivariate
  Weather benchmark (`--features M`), so the numbers are not directly comparable with those tables.
  One duplicated timestamp and the sensor error code (−9999) records are removed.
- **DLinear** uses a moving-average kernel of 25. On Bike it is run in three versions:
  `univariate` (`cnt` only), `mix` (all 14 channels, shared linear layers) and `mix` with
  `--dlinear_individual 1` (per-channel linear layers); both mix versions map the channel
  forecasts to `cnt` with a learned channel-mixing layer (`--dlinear_exog mix`), as for PatchTST.
- **TabM** is built with `tabm.TabM.make(...)`; `--dropout` is passed explicitly (0.0 in all
  reported runs) because the library default is 0.1.

**Protocol (both tasks)**
- Splits are chronological; scalers are fit on the training period only.
- Windows that cross missing or non-consecutive timestamps are skipped, not imputed; windows whose
  targets straddle a split boundary are discarded.
- Optimizer: AdamW (lr 1e-3, weight decay 1e-4), batch size 64 (16 for PatchTST), 50 epochs in all
  reported experiments (the script default is 30).
- Scheduler: OneCycle with cosine warm-up (30%) and cool-down (peak = `--lr`, start = lr/25),
  then a final 10% annihilation phase towards lr/(25·10⁴); the AdamW β₁ moves between 0.95 and 0.85.
- The checkpoint with the lowest validation RMSE is evaluated on the test set.
- Metrics: RMSE, MAE and R² on the original target scale, over all forecast steps (Weather also
  reports MSE/MAE on the normalized scale). Bike predictions are clipped at zero.
- Naive baselines on the same test targets: persistence and seasonal naive at 24 h and 168 h.
- Every configuration is run with seeds 0, 1 and 2; results are reported as mean ± std.

## Using BasisTabM

Main constructor options:

| Argument | Default | Meaning |
|---|---|---|
| `k` | 8 | number of ensemble members |
| `feature_basis` | `True` | input feature basis |
| `feature_grids` | 8 | number of RBF centers of the input feature basis |
| `basis_mode` | `"none"` | `"mix"` adds the α-mixed basis branch in every layer |
| `num_grids` | `None` (= k) | number of RBF centers of the per-layer basis branch |
| `alpha_mode` | `"member"` | α per member `(k, out)` or shared `(1, out)` |
| `ensemble_init` | `"random_signs"` | `"ones"` reproduces the old init, in which all members stay identical |
| `norm1_type`, `norm2_type` | `"layer"`, `"layer"` | `""`, `"layer"`, `"batch"` (per-member statistics) or `"rms"`; applied in hidden layers only |
| `grid_min`, `grid_max` | −2, 2 | initial range of the RBF centers (centers are learnable) |
| `basis` | `"rbf"` | also `sech`, `laplace`, `cauchy`, `mexican_hat`, `gaussian_d2` |

The RBF width is fixed at 1 (`use_width=False`). The input feature basis adds only
`d_in · feature_grids + feature_grids` parameters.

## Outputs

| Script | Output folder | Files |
|---|---|---|
| `run_bike.py` | `output/bike/h<H>[_cal]/<model>/` | `<model>__<note>.pt`, `history__<note>.jsonl`, `summary__<note>.txt`, `predictions__<note>.csv` |
| `run_weather.py` | `output/weather/<model>/` | `model__<note>__s<seed>.pt`, `config__…json`, `result__…txt`, `predictions__…` (`.npz` with `--multi_output`, else `.csv`), `per_step__…csv` |

`run_bike.py` does not add the seed to the file names, so the bash script puts it in the note
(e.g. `fb_k4_h240_n1none_n2none_s0`); `run_weather.py` appends `__s<seed>` itself.
An existing run with the same note is not overwritten unless `--overwrite` is passed.
For a quick smoke test, use `--epochs 1`.

## Running the experiments

```bash
bash run_bike_h24_cal_all.sh      # Bike, L=H=24 + calendar: 81 configs x 3 seeds = 243 runs
bash run_weather_L96_H96_all.sh   # Weather, L=H=96:         74 configs x 3 seeds = 222 runs
```

Each script runs, for every seed:

| Model | Configurations | Bike | Weather |
|---|---|---|---|
| MLP | norm none / layer / batch | 3 | 3 |
| EfficientKAN | grid size 5 | 1 | 1 |
| BSRBF-KAN | grid size 5; norm none / layer / batch | 3 | 3 |
| PatchTST | Bike: all channels + channel mixing; Weather: univariate | 1 | 1 |
| DLinear | Bike: univariate / mix / mix-individual; Weather: univariate | 3 | 1 |
| TabM, dropout 0 | k = 2, 4, 8, 16, 32 | 5 | 5 |
| TabM, dropout 0.1 | k = 2, 4, 8, 16, 32 (earlier runs; optional, Bike only) | 5 | – |
| BasisTabM | 12 variants × k = 2, 4, 8, 16, 32 | 60 | 60 |
| **Total** | | **81** | **74** |

The comment above every command gives the model's parameter count; hidden widths are chosen so that
every model except DLinear matches the MLP budget of its setting.

Example (Bike, BasisTabM FB+Mix+BatchNorm, k = 4, 93,808 parameters):

```bash
python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "233" --tabm_k 4 \
  --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" \
  --epochs 50 --scheduler "OneCycleLR" --seed 0 --horizon 24 --calendar \
  --note "fbmix_k4_h233_n1none_n2batch_s0" --overwrite
```

Example (Weather, BasisTabM FB+Mix, k = 2, 50,148 parameters):

```bash
python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output \
  --hidden_layers "243" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" \
  --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed 0 \
  --note "L96_H96_mo_fbmix_k2_h243_n1none_n2none" --overwrite --no_amp
```

## Results

Mean ± std over seeds 0, 1, 2. For each model and BasisTabM variant, the configuration with the
lowest **mean validation RMSE** is reported (searched over normalization, k = 2, 4, 8, 16, 32 and,
for DLinear, the input version); its test scores are shown as they are, with no test-set selection.
RMSE and MAE are on the original scale, averaged over all forecast steps; time is the mean training
time per run (optimization only, one RTX 3060 Ti). BN/LN: BatchNorm/LayerNorm after the hidden layer;
`+Norm` rows pick the better of the two by validation RMSE. Best value per column in bold.
DLinear is a lightweight reference and is not budget-matched.

### Bike Sharing (L = H = 24, with calendar; ≈ 93.6K parameters)

| Model | Config | Params | Val RMSE ↓ | Test RMSE ↓ | Test MAE ↓ | Test R² ↑ | Time (s) |
|---|---|--:|--:|--:|--:|--:|--:|
| MLP | no norm | 92,440 | 88.31 ± 0.30 | 93.71 ± 2.14 | 64.66 | 0.807 | 29.6 |
| DLinear | mix, individual | 16,815 | 98.26 ± 0.18 | 94.00 ± 0.07 | 60.52 | 0.806 | 102.0 |
| BSRBF-KAN | LN | 94,690 | 84.17 ± 1.57 | 86.42 ± 2.57 | 58.74 | 0.836 | 60.7 |
| EfficientKAN | grid 5 | 93,600 | 80.64 ± 1.20 | 80.44 ± 1.00 | 53.68 | 0.858 | 48.9 |
| PatchTST | P=8, S=4 | 93,629 | 85.39 ± 0.47 | 81.55 ± 0.46 | 53.72 | 0.854 | 388.4 |
| TabM | k=8 | 93,728 | 83.93 ± 0.41 | 85.81 ± 0.92 | 56.80 | 0.839 | 35.7 |
| BasisTabM NoFB | k=2 | 93,732 | 83.37 ± 0.74 | 88.60 ± 1.42 | 60.55 | 0.828 | 34.3 |
| BasisTabM NoFB+Norm | k=2, BN | 93,638 | 83.27 ± 0.27 | 86.80 ± 0.78 | 58.78 | 0.835 | 37.2 |
| BasisTabM NoFB+Mix | k=8 | 93,832 | 82.58 ± 0.08 | 81.71 ± 0.59 | 52.99 | 0.854 | 45.9 |
| BasisTabM NoFB+Mix+Norm | k=4, BN | 94,184 | 79.87 ± 0.58 | 81.42 ± 0.16 | 53.72 | 0.855 | 48.2 |
| BasisTabM FB | k=4 | 93,512 | 79.76 ± 0.38 | 79.28 ± 0.46 | 51.94 | 0.862 | 36.1 |
| BasisTabM FB+Norm | k=4, BN | 93,532 | 78.39 ± 0.70 | 77.99 ± 0.88 | 50.91 | 0.867 | 38.5 |
| BasisTabM FB+Mix | k=8 | 93,784 | 79.26 ± 0.13 | 77.79 ± 0.39 | 49.74 | 0.867 | 45.7 |
| BasisTabM FB+Mix+Norm | k=4, BN | 93,808 | **76.76 ± 0.17** | **77.11 ± 0.59** | 50.07 | 0.870 | 50.0 |

TabM is reported with dropout 0, like all other models (the earlier dropout-0.1 runs are worse,
best test RMSE 86.69 at k = 8).

### Weather (univariate OT, L = H = 96; ≈ 50.2K parameters)

| Model | Config | Params | Val RMSE ↓ | Test RMSE ↓ | Test MAE ↓ | Test R² ↑ | Time (s) |
|---|---|--:|--:|--:|--:|--:|--:|
| MLP | BN | 50,208 | 12.86 ± 0.07 | 11.91 ± 0.33 | 8.28 | 0.468 | 69.8 |
| DLinear | univariate | 18,624 | 13.28 ± 0.00 | 13.49 ± 0.05 | 9.30 | 0.317 | 58.8 |
| BSRBF-KAN | BN | 50,362 | 13.59 ± 0.13 | 12.16 ± 0.16 | 8.59 | 0.445 | 140.6 |
| EfficientKAN | grid 5 | 49,920 | 13.63 ± 0.06 | 12.14 ± 0.18 | 8.44 | 0.446 | 108.0 |
| PatchTST | P=16, S=8 | 50,200 | 13.23 ± 0.07 | 12.64 ± 0.07 | 8.69 | 0.400 | 898.8 |
| TabM | k=4 | 50,056 | 12.77 ± 0.05 | 11.69 ± 0.08 | 8.12 | 0.487 | 66.9 |
| BasisTabM NoFB | k=2 | 50,274 | 12.77 ± 0.01 | 11.82 ± 0.12 | 8.27 | 0.476 | 64.9 |
| BasisTabM NoFB+Norm | k=4, BN | 50,124 | **12.69 ± 0.03** | 11.90 ± 0.03 | 8.34 | 0.469 | 71.5 |
| BasisTabM NoFB+Mix | k=2 | 50,172 | 12.82 ± 0.06 | 11.60 ± 0.13 | 8.14 | 0.495 | 107.9 |
| BasisTabM NoFB+Mix+Norm | k=4, BN | 50,144 | 12.88 ± 0.02 | 11.84 ± 0.12 | 8.29 | 0.474 | 111.3 |
| BasisTabM FB | k=8 | 50,168 | 12.79 ± 0.04 | 11.75 ± 0.03 | 8.16 | 0.482 | 73.8 |
| BasisTabM FB+Norm | k=2, BN | 50,236 | 12.82 ± 0.07 | 12.13 ± 0.41 | 8.44 | 0.447 | 82.8 |
| BasisTabM FB+Mix | k=2 | 50,148 | 12.88 ± 0.00 | **11.59 ± 0.21** | 8.14 | 0.495 | 105.3 |
| BasisTabM FB+Mix+Norm | k=4, BN | 50,272 | 12.99 ± 0.07 | 11.97 ± 0.07 | 8.34 | 0.463 | 114.8 |

On Weather most differences are within the seed variation; MLP and BSRBF-KAN with LayerNorm
collapse (test RMSE ≈ 20.9) and are not selected.

### All configurations

Every configuration of every model (mean ± std over three seeds), grouped by model and sorted by k.
Best value per column in bold.

<details>
<summary><b>Bike Sharing: all 81 configurations</b></summary>

| Model | Config | Params | Val RMSE ↓ | Test RMSE ↓ | Test MAE ↓ | Test R² ↑ | Time (s) |
|---|---|--:|--:|--:|--:|--:|--:|
| MLP | width 256 | 92,440 | 88.31 ± 0.30 | 93.71 ± 2.14 | 64.66 | 0.807 | 29.6 |
| MLP+BN | width 256 | 93,624 | 89.79 ± 1.94 | 94.13 ± 1.19 | 64.59 | 0.806 | 33.0 |
| MLP+LN | width 256 | 93,624 | 99.41 ± 1.16 | 105.69 ± 1.01 | 71.73 | 0.755 | 32.2 |
| DLinear (mix) | kernel 25 | 1,215 | 122.18 ± 0.01 | 115.10 ± 0.12 | 73.79 | 0.710 | 34.1 |
| DLinear (mix, individual) | kernel 25 | 16,815 | 98.26 ± 0.18 | 94.00 ± 0.07 | 60.52 | 0.806 | 102.0 |
| DLinear (univariate) | kernel 25 | 1,200 | 124.86 ± 0.18 | 117.46 ± 0.02 | 76.17 | 0.698 | 30.9 |
| BSRBF-KAN | grid 5 | 93,960 | 88.86 ± 1.57 | 90.56 ± 2.40 | 61.76 | 0.820 | 52.4 |
| BSRBF-KAN+BN | grid 5 | 94,690 | 85.81 ± 0.60 | 87.40 ± 1.32 | 59.09 | 0.833 | 62.4 |
| BSRBF-KAN+LN | grid 5 | 94,690 | 84.17 ± 1.57 | 86.42 ± 2.57 | 58.74 | 0.836 | 60.7 |
| EfficientKAN | grid 5 | 93,600 | 80.64 ± 1.20 | 80.44 ± 1.00 | 53.68 | 0.858 | 48.9 |
| PatchTST | P=8, S=4 | 93,629 | 85.39 ± 0.47 | 81.55 ± 0.46 | 53.72 | 0.854 | 388.4 |
| TabM (dropout 0) | k=2 | 93,452 | 86.84 ± 0.15 | 92.48 ± 1.51 | 62.89 | 0.812 | 38.8 |
| TabM (dropout 0) | k=4 | 93,840 | 84.80 ± 1.13 | 88.31 ± 1.79 | 59.50 | 0.829 | 37.2 |
| TabM (dropout 0) | k=8 | 93,728 | 83.93 ± 0.41 | 85.81 ± 0.92 | 56.80 | 0.839 | 35.7 |
| TabM (dropout 0) | k=16 | 93,744 | 84.27 ± 0.18 | 82.85 ± 0.13 | 53.45 | 0.850 | 35.2 |
| TabM (dropout 0) | k=32 | 93,280 | 87.29 ± 0.09 | 84.88 ± 0.45 | 53.91 | 0.842 | 36.4 |
| TabM (dropout 0.1) | k=2 | 93,452 | 85.55 ± 0.77 | 88.82 ± 1.79 | 59.81 | 0.827 | 33.0 |
| TabM (dropout 0.1) | k=4 | 93,840 | 85.66 ± 0.70 | 87.83 ± 1.78 | 58.04 | 0.831 | 32.9 |
| TabM (dropout 0.1) | k=8 | 93,728 | 86.58 ± 0.46 | 86.69 ± 0.56 | 56.39 | 0.835 | 32.4 |
| TabM (dropout 0.1) | k=16 | 93,744 | 89.99 ± 0.17 | 88.16 ± 0.19 | 56.25 | 0.830 | 31.4 |
| TabM (dropout 0.1) | k=32 | 93,280 | 96.47 ± 0.21 | 93.43 ± 0.28 | 59.37 | 0.809 | 32.1 |
| BasisTabM NoFB | k=2 | 93,732 | 83.37 ± 0.74 | 88.60 ± 1.42 | 60.55 | 0.828 | 34.3 |
| BasisTabM NoFB | k=4 | 93,792 | 83.94 ± 0.47 | 87.05 ± 0.65 | 58.66 | 0.834 | 39.2 |
| BasisTabM NoFB | k=8 | 93,696 | 83.88 ± 0.12 | 83.03 ± 0.27 | 54.20 | 0.849 | 35.5 |
| BasisTabM NoFB | k=16 | 93,456 | 86.68 ± 0.19 | 84.21 ± 0.35 | 53.90 | 0.845 | 33.8 |
| BasisTabM NoFB | k=32 | 93,456 | 94.05 ± 0.15 | 92.36 ± 0.33 | 59.11 | 0.813 | 33.8 |
| BasisTabM NoFB+BN | k=2 | 93,638 | 83.27 ± 0.27 | 86.80 ± 0.78 | 58.78 | 0.835 | 37.2 |
| BasisTabM NoFB+BN | k=4 | 93,496 | 84.00 ± 0.02 | 85.40 ± 0.96 | 56.94 | 0.840 | 35.7 |
| BasisTabM NoFB+BN | k=8 | 93,472 | 85.10 ± 0.66 | 84.03 ± 1.13 | 54.80 | 0.845 | 34.0 |
| BasisTabM NoFB+BN | k=16 | 93,704 | 88.80 ± 0.53 | 86.23 ± 0.78 | 55.47 | 0.837 | 34.5 |
| BasisTabM NoFB+BN | k=32 | 93,408 | 97.73 ± 0.07 | 96.01 ± 0.17 | 62.09 | 0.798 | 33.3 |
| BasisTabM NoFB+LN | k=2 | 93,504 | 99.31 ± 0.77 | 109.83 ± 2.58 | 73.87 | 0.736 | 33.2 |
| BasisTabM NoFB+LN | k=4 | 93,540 | 98.29 ± 0.83 | 106.66 ± 2.25 | 70.80 | 0.751 | 33.3 |
| BasisTabM NoFB+LN | k=8 | 93,782 | 93.67 ± 0.37 | 96.93 ± 0.89 | 63.54 | 0.794 | 33.6 |
| BasisTabM NoFB+LN | k=16 | 93,474 | 92.94 ± 0.73 | 92.33 ± 0.84 | 59.70 | 0.813 | 34.0 |
| BasisTabM NoFB+LN | k=32 | 93,812 | 97.56 ± 0.26 | 95.98 ± 0.31 | 61.88 | 0.798 | 33.2 |
| BasisTabM NoFB+Mix | k=2 | 93,924 | 84.14 ± 0.76 | 89.12 ± 2.35 | 60.59 | 0.826 | 44.7 |
| BasisTabM NoFB+Mix | k=4 | 93,760 | 84.04 ± 0.85 | 86.69 ± 0.93 | 58.01 | 0.835 | 45.9 |
| BasisTabM NoFB+Mix | k=8 | 93,832 | 82.58 ± 0.08 | 81.71 ± 0.59 | 52.99 | 0.854 | 45.9 |
| BasisTabM NoFB+Mix | k=16 | 93,480 | 85.09 ± 0.50 | 83.48 ± 0.65 | 53.09 | 0.847 | 59.2 |
| BasisTabM NoFB+Mix | k=32 | 93,640 | 93.49 ± 0.20 | 91.87 ± 0.15 | 58.48 | 0.815 | 135.6 |
| BasisTabM NoFB+Mix+BN | k=2 | 93,820 | 81.26 ± 0.85 | 84.36 ± 1.30 | 56.85 | 0.844 | 48.5 |
| BasisTabM NoFB+Mix+BN | k=4 | 94,184 | 79.87 ± 0.58 | 81.42 ± 0.16 | 53.72 | 0.855 | 48.2 |
| BasisTabM NoFB+Mix+BN | k=8 | 93,856 | 80.67 ± 0.16 | 81.39 ± 1.02 | 52.75 | 0.855 | 48.0 |
| BasisTabM NoFB+Mix+BN | k=16 | 93,656 | 84.56 ± 0.06 | 84.42 ± 0.13 | 54.06 | 0.844 | 62.4 |
| BasisTabM NoFB+Mix+BN | k=32 | 94,264 | 89.06 ± 0.73 | 89.46 ± 0.42 | 57.88 | 0.825 | 130.9 |
| BasisTabM NoFB+Mix+LN | k=2 | 94,060 | 99.38 ± 0.67 | 111.12 ± 2.50 | 74.85 | 0.729 | 48.6 |
| BasisTabM NoFB+Mix+LN | k=4 | 93,872 | 93.69 ± 1.33 | 100.67 ± 1.44 | 66.80 | 0.778 | 47.2 |
| BasisTabM NoFB+Mix+LN | k=8 | 93,506 | 87.93 ± 0.82 | 90.56 ± 0.66 | 59.30 | 0.820 | 47.0 |
| BasisTabM NoFB+Mix+LN | k=16 | 93,464 | 88.76 ± 0.28 | 88.68 ± 0.24 | 57.24 | 0.828 | 61.0 |
| BasisTabM NoFB+Mix+LN | k=32 | 93,970 | 95.85 ± 0.43 | 94.45 ± 0.44 | 60.90 | 0.804 | 135.3 |
| BasisTabM FB | k=2 | 93,500 | 81.16 ± 0.93 | 85.53 ± 0.24 | 57.60 | 0.840 | 35.0 |
| BasisTabM FB | k=4 | 93,512 | 79.76 ± 0.38 | 79.28 ± 0.46 | 51.94 | 0.862 | 36.1 |
| BasisTabM FB | k=8 | 93,704 | 79.83 ± 0.27 | 78.34 ± 0.42 | 50.32 | 0.866 | 34.8 |
| BasisTabM FB | k=16 | 93,704 | 83.20 ± 0.19 | 81.64 ± 0.08 | 51.67 | 0.854 | 35.6 |
| BasisTabM FB | k=32 | 93,416 | 91.18 ± 0.37 | 89.74 ± 0.27 | 56.75 | 0.824 | 34.1 |
| BasisTabM FB+BN | k=2 | 93,744 | 78.70 ± 0.11 | 80.37 ± 0.41 | 53.54 | 0.858 | 38.6 |
| BasisTabM FB+BN | k=4 | 93,532 | 78.39 ± 0.70 | 77.99 ± 0.88 | 50.91 | 0.867 | 38.5 |
| BasisTabM FB+BN | k=8 | 93,768 | 79.15 ± 0.41 | 77.92 ± 0.69 | 49.99 | 0.867 | 38.5 |
| BasisTabM FB+BN | k=16 | 93,760 | 83.59 ± 0.38 | 82.61 ± 0.50 | 52.64 | 0.850 | 39.7 |
| BasisTabM FB+BN | k=32 | 93,504 | 93.22 ± 0.21 | 92.38 ± 0.25 | 59.55 | 0.813 | 40.6 |
| BasisTabM FB+LN | k=2 | 93,624 | 92.76 ± 1.34 | 100.19 ± 0.75 | 66.84 | 0.780 | 39.2 |
| BasisTabM FB+LN | k=4 | 93,618 | 87.06 ± 0.63 | 89.78 ± 1.11 | 58.96 | 0.823 | 39.4 |
| BasisTabM FB+LN | k=8 | 93,776 | 83.27 ± 0.39 | 83.19 ± 0.31 | 53.67 | 0.848 | 38.3 |
| BasisTabM FB+LN | k=16 | 93,710 | 85.23 ± 0.25 | 84.60 ± 0.28 | 53.92 | 0.843 | 38.8 |
| BasisTabM FB+LN | k=32 | 93,760 | 92.23 ± 0.37 | 92.12 ± 0.49 | 58.58 | 0.814 | 34.9 |
| BasisTabM FB+Mix | k=2 | 93,676 | 82.22 ± 0.63 | 84.70 ± 1.73 | 57.11 | 0.843 | 47.4 |
| BasisTabM FB+Mix | k=4 | 93,448 | 79.60 ± 0.62 | 79.15 ± 0.50 | 51.72 | 0.863 | 48.3 |
| BasisTabM FB+Mix | k=8 | 93,784 | 79.26 ± 0.13 | 77.79 ± 0.39 | 49.74 | 0.867 | 45.7 |
| BasisTabM FB+Mix | k=16 | 93,632 | 82.56 ± 0.29 | 80.94 ± 0.32 | 50.89 | 0.856 | 60.9 |
| BasisTabM FB+Mix | k=32 | 93,408 | 91.04 ± 0.23 | 89.29 ± 0.34 | 56.21 | 0.825 | 137.5 |
| BasisTabM FB+Mix+BN | k=2 | 93,540 | 78.33 ± 0.28 | 80.63 ± 0.27 | 53.71 | 0.858 | 50.3 |
| BasisTabM FB+Mix+BN | k=4 | 93,808 | **76.76 ± 0.17** | **77.11 ± 0.59** | 50.07 | 0.870 | 50.0 |
| BasisTabM FB+Mix+BN | k=8 | 93,696 | 77.80 ± 0.60 | 77.69 ± 0.73 | 49.88 | 0.868 | 48.7 |
| BasisTabM FB+Mix+BN | k=16 | 93,616 | 78.98 ± 0.36 | 79.96 ± 0.36 | 51.09 | 0.860 | 62.3 |
| BasisTabM FB+Mix+BN | k=32 | 93,648 | 77.41 ± 0.72 | 79.03 ± 0.89 | 50.95 | 0.863 | 135.0 |
| BasisTabM FB+Mix+LN | k=2 | 93,796 | 93.60 ± 1.10 | 98.68 ± 0.68 | 65.29 | 0.787 | 49.6 |
| BasisTabM FB+Mix+LN | k=4 | 93,544 | 88.08 ± 0.60 | 90.91 ± 1.16 | 59.69 | 0.819 | 49.8 |
| BasisTabM FB+Mix+LN | k=8 | 93,444 | 82.48 ± 0.20 | 82.16 ± 0.28 | 52.87 | 0.852 | 47.6 |
| BasisTabM FB+Mix+LN | k=16 | 93,604 | 84.72 ± 0.12 | 83.98 ± 0.20 | 53.45 | 0.846 | 62.2 |
| BasisTabM FB+Mix+LN | k=32 | 93,726 | 92.48 ± 0.26 | 92.46 ± 0.35 | 58.95 | 0.813 | 139.8 |

</details>

<details>
<summary><b>Weather: all 74 configurations</b></summary>

| Model | Config | Params | Val RMSE ↓ | Test RMSE ↓ | Test MAE ↓ | Test R² ↑ | Time (s) |
|---|---|--:|--:|--:|--:|--:|--:|
| MLP | width 256 | 49,504 | 12.86 ± 0.04 | 11.84 ± 0.10 | 8.29 | 0.474 | 58.3 |
| MLP+BN | width 256 | 50,208 | 12.86 ± 0.07 | 11.91 ± 0.33 | 8.28 | 0.468 | 69.8 |
| MLP+LN | width 256 | 50,208 | 15.60 ± 0.08 | 20.83 ± 0.31 | 15.32 | -0.629 | 64.5 |
| DLinear | kernel 25 | 18,624 | 13.28 ± 0.00 | 13.49 ± 0.05 | 9.30 | 0.317 | 58.8 |
| BSRBF-KAN | grid 5 | 50,112 | 13.84 ± 0.04 | 12.42 ± 0.08 | 8.76 | 0.421 | 108.2 |
| BSRBF-KAN+BN | grid 5 | 50,362 | 13.59 ± 0.13 | 12.16 ± 0.16 | 8.59 | 0.445 | 140.6 |
| BSRBF-KAN+LN | grid 5 | 50,362 | 16.17 ± 0.04 | 20.92 ± 0.29 | 15.43 | -0.644 | 138.6 |
| EfficientKAN | grid 5 | 49,920 | 13.63 ± 0.06 | 12.14 ± 0.18 | 8.44 | 0.446 | 108.0 |
| PatchTST | P=16, S=8 | 50,200 | 13.23 ± 0.07 | 12.64 ± 0.07 | 8.69 | 0.400 | 898.8 |
| TabM | k=2 | 50,316 | 12.78 ± 0.10 | 11.60 ± 0.27 | 8.09 | 0.495 | 66.9 |
| TabM | k=4 | 50,056 | 12.77 ± 0.05 | 11.69 ± 0.08 | 8.12 | 0.487 | 66.9 |
| TabM | k=8 | 49,936 | 12.78 ± 0.05 | 11.74 ± 0.19 | 8.17 | 0.483 | 64.7 |
| TabM | k=16 | 49,664 | 12.83 ± 0.02 | 11.77 ± 0.21 | 8.20 | 0.480 | 63.7 |
| TabM | k=32 | 51,392 | 12.94 ± 0.01 | 11.93 ± 0.08 | 8.34 | 0.465 | 64.6 |
| BasisTabM NoFB | k=2 | 50,274 | 12.77 ± 0.01 | 11.82 ± 0.12 | 8.27 | 0.476 | 64.9 |
| BasisTabM NoFB | k=4 | 50,112 | 12.79 ± 0.03 | 11.84 ± 0.17 | 8.27 | 0.474 | 65.7 |
| BasisTabM NoFB | k=8 | 50,256 | 12.79 ± 0.02 | 11.77 ± 0.23 | 8.24 | 0.480 | 65.3 |
| BasisTabM NoFB | k=16 | 50,208 | 12.85 ± 0.01 | 11.84 ± 0.08 | 8.30 | 0.474 | 66.0 |
| BasisTabM NoFB | k=32 | 50,112 | 13.03 ± 0.01 | 12.41 ± 0.07 | 8.68 | 0.422 | 62.7 |
| BasisTabM NoFB+BN | k=2 | 50,268 | 12.70 ± 0.05 | 11.88 ± 0.04 | 8.32 | 0.470 | 71.9 |
| BasisTabM NoFB+BN | k=4 | 50,124 | **12.69 ± 0.03** | 11.90 ± 0.03 | 8.34 | 0.469 | 71.5 |
| BasisTabM NoFB+BN | k=8 | 50,096 | 12.71 ± 0.03 | 11.88 ± 0.14 | 8.34 | 0.470 | 71.2 |
| BasisTabM NoFB+BN | k=16 | 50,304 | 12.80 ± 0.03 | 12.09 ± 0.12 | 8.50 | 0.451 | 70.6 |
| BasisTabM NoFB+BN | k=32 | 50,048 | 13.05 ± 0.00 | 12.73 ± 0.12 | 8.90 | 0.392 | 71.2 |
| BasisTabM NoFB+LN | k=2 | 50,176 | 12.96 ± 0.08 | 11.83 ± 0.34 | 8.24 | 0.475 | 69.2 |
| BasisTabM NoFB+LN | k=4 | 50,180 | 12.95 ± 0.05 | 11.83 ± 0.27 | 8.22 | 0.474 | 70.4 |
| BasisTabM NoFB+LN | k=8 | 50,264 | 12.96 ± 0.07 | 11.85 ± 0.18 | 8.23 | 0.473 | 69.7 |
| BasisTabM NoFB+LN | k=16 | 50,104 | 12.97 ± 0.06 | 11.98 ± 0.14 | 8.32 | 0.461 | 68.6 |
| BasisTabM NoFB+LN | k=32 | 50,106 | 13.03 ± 0.02 | 12.58 ± 0.19 | 8.75 | 0.406 | 66.6 |
| BasisTabM NoFB+Mix | k=2 | 50,172 | 12.82 ± 0.06 | 11.60 ± 0.13 | 8.14 | 0.495 | 107.9 |
| BasisTabM NoFB+Mix | k=4 | 50,216 | 12.83 ± 0.07 | 11.57 ± 0.15 | 8.11 | 0.498 | 105.8 |
| BasisTabM NoFB+Mix | k=8 | 50,128 | 12.83 ± 0.06 | 11.67 ± 0.04 | 8.15 | 0.489 | 107.2 |
| BasisTabM NoFB+Mix | k=16 | 50,208 | 12.87 ± 0.02 | 12.06 ± 0.14 | 8.41 | 0.454 | 103.1 |
| BasisTabM NoFB+Mix | k=32 | 50,112 | 13.02 ± 0.01 | 12.41 ± 0.27 | 8.66 | 0.422 | 153.8 |
| BasisTabM NoFB+Mix+BN | k=2 | 50,140 | 12.91 ± 0.03 | 11.99 ± 0.24 | 8.38 | 0.460 | 108.6 |
| BasisTabM NoFB+Mix+BN | k=4 | 50,144 | 12.88 ± 0.02 | 11.84 ± 0.12 | 8.29 | 0.474 | 111.3 |
| BasisTabM NoFB+Mix+BN | k=8 | 50,128 | 12.92 ± 0.04 | 11.89 ± 0.06 | 8.31 | 0.469 | 111.0 |
| BasisTabM NoFB+Mix+BN | k=16 | 50,240 | 12.91 ± 0.06 | 12.04 ± 0.21 | 8.44 | 0.456 | 114.1 |
| BasisTabM NoFB+Mix+BN | k=32 | 50,368 | 13.03 ± 0.02 | 12.34 ± 0.19 | 8.67 | 0.428 | 146.5 |
| BasisTabM NoFB+Mix+LN | k=2 | 50,262 | 13.06 ± 0.06 | 11.67 ± 0.21 | 8.14 | 0.489 | 107.4 |
| BasisTabM NoFB+Mix+LN | k=4 | 50,264 | 13.09 ± 0.06 | 11.68 ± 0.20 | 8.16 | 0.488 | 109.0 |
| BasisTabM NoFB+Mix+LN | k=8 | 50,096 | 13.04 ± 0.04 | 11.77 ± 0.23 | 8.23 | 0.480 | 109.1 |
| BasisTabM NoFB+Mix+LN | k=16 | 50,294 | 12.96 ± 0.03 | 11.99 ± 0.21 | 8.35 | 0.460 | 108.1 |
| BasisTabM NoFB+Mix+LN | k=32 | 50,348 | 13.10 ± 0.03 | 12.21 ± 0.15 | 8.56 | 0.440 | 156.2 |
| BasisTabM FB | k=2 | 50,258 | 12.87 ± 0.03 | 11.61 ± 0.21 | 8.15 | 0.493 | 75.1 |
| BasisTabM FB | k=4 | 50,276 | 12.87 ± 0.06 | **11.53 ± 0.10** | 8.09 | 0.501 | 74.7 |
| BasisTabM FB | k=8 | 50,168 | 12.79 ± 0.04 | 11.75 ± 0.03 | 8.16 | 0.482 | 73.8 |
| BasisTabM FB | k=16 | 50,264 | 12.83 ± 0.01 | 12.02 ± 0.28 | 8.33 | 0.458 | 74.0 |
| BasisTabM FB | k=32 | 50,312 | 12.99 ± 0.01 | 12.37 ± 0.29 | 8.59 | 0.426 | 73.1 |
| BasisTabM FB+BN | k=2 | 50,236 | 12.82 ± 0.07 | 12.13 ± 0.41 | 8.44 | 0.447 | 82.8 |
| BasisTabM FB+BN | k=4 | 50,264 | 12.85 ± 0.02 | 12.03 ± 0.25 | 8.37 | 0.457 | 86.0 |
| BasisTabM FB+BN | k=8 | 50,176 | 12.84 ± 0.05 | 11.94 ± 0.08 | 8.31 | 0.465 | 83.9 |
| BasisTabM FB+BN | k=16 | 50,264 | 12.88 ± 0.05 | 12.13 ± 0.23 | 8.41 | 0.448 | 81.3 |
| BasisTabM FB+BN | k=32 | 50,120 | 13.03 ± 0.02 | 12.43 ± 0.15 | 8.68 | 0.420 | 83.4 |
| BasisTabM FB+LN | k=2 | 50,152 | 13.10 ± 0.06 | 11.76 ± 0.18 | 8.22 | 0.481 | 79.9 |
| BasisTabM FB+LN | k=4 | 50,132 | 13.08 ± 0.06 | 11.77 ± 0.12 | 8.23 | 0.480 | 79.4 |
| BasisTabM FB+LN | k=8 | 50,168 | 13.07 ± 0.03 | 11.87 ± 0.12 | 8.27 | 0.471 | 78.3 |
| BasisTabM FB+LN | k=16 | 50,154 | 13.10 ± 0.02 | 12.21 ± 0.28 | 8.45 | 0.440 | 78.2 |
| BasisTabM FB+LN | k=32 | 50,302 | 13.14 ± 0.03 | 12.43 ± 0.10 | 8.63 | 0.420 | 77.7 |
| BasisTabM FB+Mix | k=2 | 50,148 | 12.88 ± 0.00 | 11.59 ± 0.21 | 8.14 | 0.495 | 105.3 |
| BasisTabM FB+Mix | k=4 | 50,160 | 12.90 ± 0.04 | 11.69 ± 0.37 | 8.20 | 0.487 | 108.1 |
| BasisTabM FB+Mix | k=8 | 50,232 | 12.93 ± 0.05 | 11.80 ± 0.20 | 8.22 | 0.477 | 108.4 |
| BasisTabM FB+Mix | k=16 | 50,216 | 12.96 ± 0.02 | 11.98 ± 0.03 | 8.30 | 0.461 | 117.4 |
| BasisTabM FB+Mix | k=32 | 50,248 | 12.98 ± 0.01 | 12.44 ± 0.26 | 8.63 | 0.419 | 219.2 |
| BasisTabM FB+Mix+BN | k=2 | 50,304 | 12.99 ± 0.07 | 12.05 ± 0.30 | 8.42 | 0.455 | 112.2 |
| BasisTabM FB+Mix+BN | k=4 | 50,272 | 12.99 ± 0.07 | 11.97 ± 0.07 | 8.34 | 0.463 | 114.8 |
| BasisTabM FB+Mix+BN | k=8 | 50,184 | 13.04 ± 0.04 | 12.06 ± 0.04 | 8.39 | 0.454 | 114.3 |
| BasisTabM FB+Mix+BN | k=16 | 50,152 | 13.06 ± 0.03 | 12.00 ± 0.12 | 8.36 | 0.460 | 118.1 |
| BasisTabM FB+Mix+BN | k=32 | 50,376 | 13.07 ± 0.07 | 12.48 ± 0.14 | 8.67 | 0.416 | 208.4 |
| BasisTabM FB+Mix+LN | k=2 | 50,230 | 13.09 ± 0.06 | 11.71 ± 0.12 | 8.18 | 0.485 | 116.1 |
| BasisTabM FB+Mix+LN | k=4 | 50,200 | 13.07 ± 0.06 | 11.76 ± 0.15 | 8.22 | 0.481 | 118.1 |
| BasisTabM FB+Mix+LN | k=8 | 50,194 | 13.08 ± 0.03 | 11.88 ± 0.16 | 8.29 | 0.470 | 115.5 |
| BasisTabM FB+Mix+LN | k=16 | 50,296 | 13.17 ± 0.02 | 12.12 ± 0.25 | 8.48 | 0.449 | 123.3 |
| BasisTabM FB+Mix+LN | k=32 | 50,158 | 13.21 ± 0.02 | 12.38 ± 0.56 | 8.65 | 0.424 | 226.1 |

</details>

## License

Academic and non-commercial use only; see [LICENSE](LICENSE).
