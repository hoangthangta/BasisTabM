# BasisTabM

**A TabM-style ensemble with learnable basis functions for time-series demand forecasting.**

BasisTabM is a BatchEnsemble/TabM network (k members sharing one set of weights, each with its own
random-sign adapters) extended with Kolmogorov–Arnold-style basis functions:

1. an **input feature basis**: every input feature first passes through its own learnable
   1-D function $u_i = x_i + \sum_{g} c_{ig}\,\phi_g(x_i)$ (RBFs, coefficients initialised at 0);
2. optionally, a **basis branch in every layer**, mixed with the linear path by a learnable
   per-member weight $\alpha$: $h = (1-\alpha)\,h_{\text{lin}} + \alpha\, W \sum_g \phi_g(x)$;
3. optionally, a **per-member BatchNorm** after the hidden layer.

Each of the k members is trained on the target with its own loss; the prediction is the mean of
the members. The repository benchmarks BasisTabM against MLP, TabM, EfficientKAN, BSRBF-KAN and
PatchTST on two forecasting tasks, with **matched parameter budgets** within each setting.

## Model variants

All variants use random-sign (±1) adapter initialisation, SiLU, no dropout and no `norm1`.
They differ only in the flags below.

| Name | What it adds | Flags |
|---|---|---|
| **BasisTabM-NoFB** | BatchEnsemble backbone, no basis at all | `--basis_mode none --no-feature_basis` |
| **BasisTabM-FB** | + input feature basis | `--basis_mode none` (feature basis is on by default) |
| **BasisTabM-FBMix** | + α-mixed basis branch in every layer | `--basis_mode mix` |
| **BasisTabM-FBMixNorm** | + per-member BatchNorm after the hidden layer | `--basis_mode mix --norm2_type batch` |

Always pass `--norm1_type ""`: its default is `layer`, which would add a LayerNorm before every
hidden layer.

### Architecture

![BasisTabM variants](basistabm_variants.png)

Panels (B)–(D) correspond to NoFB, FB and FBMix; the dashed box in (D) marks where FBMixNorm adds
BatchNorm. The figure source is `basistabm_variants.tex` (TikZ).

## Repository structure

```
.
├── models/
│   ├── basis_tabm.py        # BasisFunction, FeatureBasis, BasisTabM_Layer, BasisTabM, ensemble_loss
│   ├── mlp.py               # MLP baseline
│   ├── efficient_kan.py     # EfficientKAN (B-spline KAN)
│   ├── bsrbf_kan.py         # BSRBF-KAN (B-spline + RBF KAN)
│   ├── patchtst_model.py    # wrapper around the official PatchTST code
│   ├── patchtst_official/   # PatchTST backbone, layers and RevIN
│   ├── fast_kan.py          # FastKAN, cnn_1d.py, resnet_1d.py, ds_cnn_1d.py
│   └── ...                  #   (kept for reference; not used by the run scripts)
├── schedulers.py            # CustomOneCycleLR and get_scheduler()
├── utils.py                 # set_seed, count_params, RMSNorm, activations
├── run_bike.py              # UCI Bike Sharing (hourly demand)
├── run_weather.py           # Weather (MPI, 10-minute, univariate OT)
├── run_bike_h24_cal_all.sh      # Bike, H=24 + calendar: all models x 3 seeds
├── run_weather_L96_H96_seeds.sh # Weather, L=96, H=96: all models x 3 seeds
└── run_weather_all_seeds.sh     # Weather, all five settings: all models x 3 seeds
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

| | Bike Sharing (main setting) | Weather |
|---|---|---|
| Target | `cnt` (hourly rentals) | `OT` |
| Frequency | 1 hour | 10 minutes |
| Lookback | 24 steps (24 h) | 96 steps (16 h); 144 for the single-point setting |
| Horizon | 24 steps, all predicted at once (`--horizon 24`) | 96 / 192 / 336 / 720 steps with `--multi_output`; or t+1 only |
| Input features | 5 weather + 8 calendar (`--calendar`) + past `cnt`, for each lookback hour (14 × 24 = 336) | univariate: past `OT` only (`--calendar none`) |
| Chronological split | 70 / 15 / 15 | 70 / 10 / 20 |
| Parameter budget | ≈ 93.6K | ≈ 50.2K (H=96); grows with the horizon |

Notes:
- **Bike calendar features** (`hr`, weekday, month as sin/cos, `holiday`, `workingday`) describe the
  lookback hours only; no information about the forecast hours is given to the models.
- **Weather is univariate** (OT → OT, i.e. TSLib `--features S`). It is not the usual multivariate
  Weather benchmark (`--features M`), so the numbers are not directly comparable with those tables.

**Protocol (both tasks)**
- Splits are chronological; scalers are fit on the training period only.
- Windows that cross missing or non-consecutive timestamps are skipped, not imputed.
- Optimizer: AdamW (lr 1e-3, weight decay 1e-4), batch size 64 (16 for PatchTST), 50 epochs in all
  reported experiments (the script default is 30).
- Scheduler: OneCycle with cosine warm-up (30%) and cool-down (peak = `--lr`, start = lr/25),
  then a final 10% annihilation phase towards lr/(25·10⁴); the AdamW β₁ moves between 0.95 and 0.85.
- The checkpoint with the lowest validation RMSE is evaluated on the test set.
- Metrics: RMSE, MAE and R² on the original target scale (Weather also reports MSE/MAE on the
  normalised scale).
- Naive baselines on the same test targets: persistence and seasonal naive at 24 h and 168 h.
- Every configuration is run with seeds 0, 1 and 2; results are reported as mean ± std.

## Using BasisTabM

Main constructor options:

| Argument | Default | Meaning |
|---|---|---|
| `k` | 8 | number of ensemble members |
| `feature_basis` | `True` | input feature basis |
| `feature_grids` | 8 | number of RBF centres of the input feature basis |
| `basis_mode` | `"none"` | `"mix"` adds the α-mixed basis branch in every layer |
| `num_grids` | `None` (= k) | number of RBF centres of the per-layer basis branch |
| `alpha_mode` | `"member"` | α per member `(k, out)` or shared `(1, out)` |
| `ensemble_init` | `"random_signs"` | `"ones"` reproduces the old init, in which all members stay identical |
| `norm1_type`, `norm2_type` | `"layer"`, `"layer"` | `""`, `"layer"`, `"batch"` or `"rms"`; applied in hidden layers only |
| `grid_min`, `grid_max` | −2, 2 | initial range of the RBF centres (centres are learnable) |
| `basis` | `"rbf"` | also `sech`, `laplace`, `cauchy`, `mexican_hat`, `gaussian_d2` |

## Outputs

| Script | Output folder | Files |
|---|---|---|
| `run_bike.py` | `output/bike/h<H>[_cal]/<model>/` | `<model>__<note>.pt`, `history__<note>.jsonl`, `summary__<note>.txt`, `predictions__<note>.csv` |
| `run_weather.py` | `output/weather/<model>/` | `model__<note>__s<seed>.pt`, `config__…json`, `result__…txt`, `predictions__…` (`.npz` with `--multi_output`, else `.csv`), `per_step__…csv` |

`run_bike.py` does not add the seed to the file names, so the bash scripts put it in the note
(e.g. `fb_k4_h240_n1none_n2none_s0`); `run_weather.py` appends `__s<seed>` itself.
An existing run with the same note is not overwritten unless `--overwrite` is passed.
For a quick smoke test, use `--epochs 1`.

## Running the experiments

```bash
bash run_bike_h24_cal_all.sh        # Bike, H=24 + calendar: 38 configs x 3 seeds = 114 runs
bash run_weather_L96_H96_seeds.sh   # Weather, L=96, H=96:  38 configs x 3 seeds = 114 runs
bash run_weather_all_seeds.sh       # Weather, all 5 settings: 190 configs x 3 seeds = 570 runs
```

Each script runs, for every seed: MLP and BSRBF-KAN (norm none / layer / batch), EfficientKAN,
PatchTST, TabM (k = 2, 4, 8, 16, 32) and the BasisTabM variants NoFB, FB, FBMix, FBMixNorm (batch)
and FBMixNorm (layer, extra ablation), each for k = 2, 4, 8, 16, 32. The comment above every command
gives the model's parameter count; widths are chosen so that every model matches the MLP budget of
its setting.

Example (Bike, BasisTabM-FBMixNorm, k = 4, 93,808 parameters):

```bash
python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "233" --tabm_k 4 \
  --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" \
  --epochs 50 --scheduler "OneCycleLR" --seed 0 --horizon 24 --calendar \
  --note "fbmix_k4_h233_n1none_n2batch_s0" --overwrite
```

## Results: Bike Sharing, H = 24 with calendar

Test set, mean ± std over seeds 0, 1, 2; every model ≈ 93.6K parameters. Best configuration of each
model shown (selected on validation RMSE).

| Model | Test RMSE ↓ | Test R² ↑ |
|---|---|---|
| **BasisTabM-FBMixNorm (k=4)** | **77.11 ± 0.59** | **0.870** |
| BasisTabM-FBMix (k=8) | 77.79 ± 0.39 | 0.867 |
| BasisTabM-FB (k=4) | 79.28 ± 0.46 | 0.862 |
| EfficientKAN | 80.44 ± 1.00 | 0.858 |
| PatchTST | 81.55 ± 0.46 | 0.854 |
| BSRBF-KAN (layer) | 86.42 ± 2.57 | 0.836 |
| TabM (k=2) | 88.82 ± 1.79 | 0.827 |
| MLP (none) | 93.71 ± 2.14 | 0.808 |
| Seasonal naive (168 h) | 102.63 | 0.770 |

Ablation at k = 4 (test RMSE): TabM 87.83 → NoFB 87.05 → FB 79.28 → FBMix 79.15 → FBMixNorm 77.11.
The input feature basis gives the largest single gain; BatchNorm adds about 2 more points.
