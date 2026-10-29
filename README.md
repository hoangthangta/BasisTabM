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

Full per-configuration results (all 81 / 74 configurations, including MAE and R² for every k and
normalization) are produced by the run scripts in `output/`.

## License

Academic and non-commercial use only; see [LICENSE](LICENSE).
