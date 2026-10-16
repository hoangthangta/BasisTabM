import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse, json, time, urllib.request, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import tabm
from mlp import MLP
from efficient_kan import EfficientKAN
from bsrbf_kan import BSRBF_KAN
from cnn_1d import CNN1D_Weather
from resnet_1d import ResNet1D_Weather
from ds_cnn_1d import DSCNN1D_Weather
from basis_tabm import BasisTabM

from schedulers import get_scheduler
from utils import set_seed, count_params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SENTINEL_THRESHOLD = -9990.0  # MPI Weather uses -9999 for invalid sensor readings


class WeatherModel(nn.Module):
    def __init__(self, model, model_type, n_calendar):
        super().__init__()
        self.model, self.model_type, self.n_calendar = model, model_type, n_calendar
    def forward(self, x, calendar):
        if self.model_type in {"cnn_1d", "resnet_1d", "ds_cnn_1d"}:
            cal = calendar.unsqueeze(-1).expand(-1, -1, x.shape[1])
            x = torch.cat([x, cal.transpose(1, 2)], dim=-1).transpose(1, 2).contiguous()
        else:
            x = torch.cat([x.flatten(1), calendar], dim=1)
        out = self.model(x)
        if self.model_type == "basis_tabm":
            if out.ndim == 3 and out.shape[-1] == 1: out = out.squeeze(-1)
            if out.ndim == 1: out = out.unsqueeze(1)
            return out
        return out.reshape(x.shape[0], -1).squeeze(-1)


def load_weather_csv(path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv"
        print(f"Downloading Weather dataset to {path} ...")
        urllib.request.urlretrieve(url, path)
    return pd.read_csv(path)


def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)


def get_dataloaders(args):
    df = load_weather_csv(args.data_path)
    if "date" not in df.columns:
        raise ValueError(f"Weather CSV must contain a 'date' column; found {list(df.columns)}")
    if args.target not in df.columns:
        raise ValueError(f"Target column {args.target!r} not found; available columns: {list(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError(f"Invalid timestamps found in 'date': {int(df['timestamp'].isna().sum())} rows.")
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    # Remove duplicate timestamps only when their weather measurements agree.
    # Conflicting records are not silently averaged or discarded.
    dup_mask = df["timestamp"].duplicated(keep=False)
    if dup_mask.any():
        value_cols = [c for c in df.columns if c not in {"date", "timestamp"}]
        dup_groups = df.loc[dup_mask].groupby("timestamp", sort=False, dropna=False)
        conflicting = []
        for stamp, group in dup_groups:
            if len(group[value_cols].drop_duplicates()) > 1:
                conflicting.append(stamp)
                if len(conflicting) >= 5:
                    break
        if conflicting:
            examples = df[df["timestamp"].isin(conflicting)][["date", "timestamp"] + value_cols[:5]]
            raise ValueError(
                f"Weather CSV has {int(dup_mask.sum())} rows with duplicate timestamps and conflicting values. "
                "Inspect these records; the script will not aggregate or discard conflicting measurements. "
                f"Examples:\n{examples.head(12).to_string(index=False)}"
            )
        n_before = len(df)
        df = df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
        print(f"Removed {n_before - len(df)} redundant rows with identical measurements at duplicate timestamps.")
    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")
    if df[args.target].isna().any():
        raise ValueError(f"Target {args.target!r} has {int(df[args.target].isna().sum())} missing/non-numeric values.")
    nonfinite_mask = ~np.isfinite(df[args.target].to_numpy(dtype=np.float64))
    if nonfinite_mask.any():
        raise ValueError(f"Target {args.target!r} has {int(nonfinite_mask.sum())} non-finite values (inf/-inf); clean them before training.")
    sentinel_mask = df[args.target] <= SENTINEL_THRESHOLD
    n_sentinel = int(sentinel_mask.sum())
    if n_sentinel:
        # Drop invalid target timestamps rather than interpolating with future values.
        # The window validator below rejects any history/target window crossing these gaps.
        df = df.loc[~sentinel_mask].reset_index(drop=True)
        print(f"Removed {n_sentinel} rows with sentinel target values (<= {SENTINEL_THRESHOLD}); "
              "windows crossing these timestamps will be skipped.")

    ts = df["timestamp"]
    timestamps = ts.to_numpy()
    raw_y = df[args.target].to_numpy(dtype=np.float32)
    # Univariate history: only the selected target's past values are model inputs.
    raw_x = raw_y[:, None]
    minute_of_day = ts.dt.hour.to_numpy() * 60 + ts.dt.minute.to_numpy()
    calendar = {
        "tod_sin": np.sin(2 * np.pi * minute_of_day / 1440),
        "tod_cos": np.cos(2 * np.pi * minute_of_day / 1440),
    }
    if args.calendar == "all":
        # Warning: Weather covers one year only, so test months (Oct-Dec) are unseen in training.
        weekday = ts.dt.dayofweek.to_numpy()
        month = ts.dt.month.to_numpy()
        calendar.update({
            "weekday_sin": np.sin(2 * np.pi * weekday / 7),
            "weekday_cos": np.cos(2 * np.pi * weekday / 7),
            "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
            "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        })
    calendar_cols = list(calendar)
    raw_cal = np.column_stack([calendar[c] for c in calendar_cols]).astype(np.float32)
    n = len(df)
    # Preserve the Weather benchmark's chronological 70/10/20 split.
    train_end, val_end = int(n * 0.70), int(n * 0.80)
    if train_end <= args.lookback + args.horizon or val_end <= train_end or n <= val_end:
        raise ValueError("Dataset is too small for the selected lookback/horizon and chronological split.")

    # Fit scalers on the training period only.
    x_mean, x_std = raw_x[:train_end].mean(0), raw_x[:train_end].std(0)
    x_std[x_std < 1e-6] = 1.0
    cal_mean, cal_std = raw_cal[:train_end].mean(0), raw_cal[:train_end].std(0)
    cal_std[cal_std < 1e-6] = 1.0
    y_mean, y_std = float(raw_y[:train_end].mean()), float(raw_y[:train_end].std())
    if not np.isfinite(y_std) or y_std < 1e-6:
        y_std = 1.0
    x_scaled = ((raw_x - x_mean) / x_std).astype(np.float32)
    cal_scaled = ((raw_cal - cal_mean) / cal_std).astype(np.float32)
    y_scaled = ((raw_y - y_mean) / y_std).astype(np.float32)

    # Vectorized window construction.
    # A window [input_start, target_t] is valid only if every step inside it is exactly 10 minutes.
    target_all = np.arange(args.lookback + args.horizon - 1, n)
    start_all = target_all - args.horizon + 1 - args.lookback
    n_candidates = len(target_all)
    step_ok = np.diff(timestamps) == np.timedelta64(10, "m")
    bad_cum = np.concatenate([[0], np.cumsum(~step_ok)])
    valid = (bad_cum[target_all] - bad_cum[start_all]) == 0
    target_idx = target_all[valid].astype(np.int64)
    input_start = start_all[valid]
    if len(target_idx) == 0:
        raise ValueError("No valid Weather windows found. Check the timestamp frequency and lookback/horizon.")

    windows = np.lib.stride_tricks.sliding_window_view(x_scaled[:, 0], args.lookback)
    X = np.ascontiguousarray(windows[input_start][..., None], dtype=np.float32)   # (N, lookback, 1)
    C = np.ascontiguousarray(cal_scaled[target_idx], dtype=np.float32)           # calendar at target time
    Y = np.ascontiguousarray(y_scaled[target_idx], dtype=np.float32)

    tr = target_idx < train_end
    va = (target_idx >= train_end) & (target_idx < val_end)
    te = target_idx >= val_end
    if not tr.any() or not va.any() or not te.any():
        raise ValueError(f"Empty split after filtering windows: train={int(tr.sum())}, val={int(va.sum())}, test={int(te.sum())}.")

    def loader(mask, shuffle=False):
        ds = TensorDataset(torch.from_numpy(X[mask]), torch.from_numpy(C[mask]), torch.from_numpy(Y[mask]))
        n_samples, bs = len(ds), args.batch_size
        g = torch.Generator(); g.manual_seed(args.seed)
        common_kw = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda",
                         persistent_workers=args.num_workers > 0, worker_init_fn=seed_worker, generator=g)
        if shuffle:
            sampler = torch.utils.data.RandomSampler(ds, generator=g)
            # Avoid a singleton final batch when using BatchNorm-based MLP configurations.
            if n_samples % bs == 1 and n_samples > bs:
                sizes = [bs] * (n_samples // bs - 1) + [bs - 1, 2]
                class SafeBatchSampler(torch.utils.data.Sampler):
                    def __iter__(self):
                        indices = list(sampler)
                        start = 0
                        for size in sizes:
                            yield indices[start:start + size]
                            start += size
                    def __len__(self):
                        return len(sizes)
                batch_sampler = SafeBatchSampler()
            else:
                batch_sampler = torch.utils.data.BatchSampler(sampler, bs, drop_last=False)
            return DataLoader(ds, batch_sampler=batch_sampler, **common_kw)
        return DataLoader(ds, batch_size=bs, shuffle=False, **common_kw)

    print(f"Time range: {ts.iloc[0]} -> {ts.iloc[-1]} ({len(df):,} rows)")
    print(f"Weather | Target: {args.target} | Train: {tr.sum():,} | Val: {va.sum():,} | Test: {te.sum():,} | "
          f"Lookback: {args.lookback} x 10 min | Horizon: {args.horizon} x 10 min")
    print(f"Input: one historical variable ({args.target}); target-time calendar: {calendar_cols}")
    print(f"Chronological split rows: train [0, {train_end}), val [{train_end}, {val_end}), test [{val_end}, {n})")
    print(f"Skipped {n_candidates - len(target_idx):,} candidate windows crossing missing/non-consecutive timestamps.")
    print("Note: validation/test windows may use earlier observations as historical input; target timestamps do not overlap.")
    return (loader(tr, True), loader(va), loader(te), 1,
            args.lookback + len(calendar_cols), len(calendar_cols),
            y_mean, y_std, target_idx[te], raw_y, timestamps, calendar_cols)


def build_model(args, n_features, input_dim, n_calendar):
    hidden = [int(v.strip()) for v in args.hidden_layers.split(",") if v.strip()]
    layers = [input_dim] + hidden + [1]
    if args.model == "mlp":
        net = MLP(layers, base_activation=args.activation, norm_type=args.norm_type)
    elif args.model == "efficient_kan":
        net = EfficientKAN(layers_hidden=layers, grid_size=args.grid_size, spline_order=3)
    elif args.model == "bsrbf_kan":
        net = BSRBF_KAN(layers_hidden=layers, grid_size=args.grid_size, spline_order=3, norm_type=args.norm_type)
    elif args.model == "tabm":
        if len(set(hidden)) != 1:
            raise ValueError("TabM needs equal widths in --hidden_layers.")
        net = tabm.TabM.make(n_num_features=input_dim, cat_cardinalities=[], d_out=1,
                             d_block=hidden[0], n_blocks=len(hidden), k=args.tabm_k,
                             num_embeddings=None)
    elif args.model == "basis_tabm":
        # Note: basis is fixed to RBF with use_width=False; there is no grid-count argument here.
        net = BasisTabM(net_layers=layers, k=args.tabm_k, basis="rbf",
                        use_width=False, width=1.0, norm1_type=args.norm1_type,
                        norm2_type=args.norm2_type, activation=args.activation, dropout=args.dropout)
    elif args.model == "cnn_1d": net = CNN1D_Weather(n_features + n_calendar)
    elif args.model == "resnet_1d": net = ResNet1D_Weather(n_features + n_calendar)
    elif args.model == "ds_cnn_1d": net = DSCNN1D_Weather(n_features + n_calendar)
    else: raise ValueError(args.model)
    return WeatherModel(net, args.model, n_calendar)


def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, per_batch, y_std, use_amp):
    model.train(); loss_sum = abs_sum = sq_sum = n_total = 0
    for x, cal, y in tqdm(loader, desc="Training", leave=False):
        x, cal, y = x.to(device, non_blocking=True), cal.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred = model(x, cal)
            if pred.ndim == 2:
                # Ensemble models (TabM): each member is trained on the target independently.
                loss = criterion(pred.float(), y.unsqueeze(1).expand_as(pred))
                pred_metric = pred.mean(dim=1)
            else:
                loss = criterion(pred.float(), y); pred_metric = pred
        old_scale = scaler.get_scale(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        # Skip the scheduler step when the optimizer step was skipped because of fp16 overflow.
        if per_batch and scaler.get_scale() >= old_scale: scheduler.step()
        err = (pred_metric.detach().float() - y) * y_std
        n = y.numel(); loss_sum += loss.item() * n; abs_sum += err.abs().sum().item(); sq_sum += err.square().sum().item(); n_total += n
    return loss_sum / n_total, abs_sum / n_total, (sq_sum / n_total) ** 0.5


@torch.no_grad()
def evaluate(model, loader, criterion, y_mean, y_std):
    model.eval(); loss_sum = abs_sum = sq_sum = 0.0; n_total = 0; ys, preds = [], []
    for x, cal, y in loader:
        x, cal, y = x.to(device, non_blocking=True), cal.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # Evaluate in FP32 to avoid FP16 quantization of regression predictions.
        pred = model(x.float(), cal.float())
        if pred.ndim == 2: pred = pred.float().mean(dim=1)
        else: pred = pred.float()
        # Note: for ensembles this loss is on the ensemble mean, unlike the per-member training loss.
        loss = criterion(pred, y.float())
        y_real = y.float() * y_std + y_mean
        pred_real = pred * y_std + y_mean
        err = pred_real - y_real; n = y.numel()
        loss_sum += loss.item() * n; abs_sum += err.abs().sum().item(); sq_sum += err.square().sum().item(); n_total += n
        ys.append(y_real.cpu().numpy()); preds.append(pred_real.cpu().numpy())
    y_true, y_pred = np.concatenate(ys), np.concatenate(preds)
    ss_res = np.square(y_true - y_pred).sum(); ss_tot = np.square(y_true - y_true.mean()).sum()
    return {"loss_scaled": loss_sum / n_total, "mae": abs_sum / n_total, "rmse": (sq_sum / n_total) ** 0.5,
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "y_true": y_true, "y_pred": y_pred}


def train_model(model, train_loader, val_loader, test_loader, args, y_mean, y_std, test_idx, raw_y, timestamps, calendar_cols):
    print(f"\nModel: {args.model}\nSeed: {args.seed}")
    model = model.to(device)
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"AMP (fp16): {'on' if use_amp else 'off'}")
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    per_batch = args.scheduler in {"OneCycleLR", "CyclicLR"}
    if args.scheduler == "StepLR":
        scheduler = get_scheduler(optimizer, name="StepLR", step_size=max(1, args.epochs // 3), lr=args.lr)
    elif args.scheduler == "CosineAnnealingLR":
        scheduler = get_scheduler(optimizer, name="CosineAnnealingLR", epochs=args.epochs, lr=args.lr)
    elif args.scheduler == "OneCycleLR":
        scheduler = get_scheduler(optimizer, name="OneCycleLR", step_size=len(train_loader) * args.epochs, lr=args.lr)
    elif args.scheduler == "ExponentialLR":
        scheduler = get_scheduler(optimizer, name="ExponentialLR", lr=args.lr)
    elif args.scheduler == "CyclicLR":
        scheduler = get_scheduler(optimizer, name="CyclicLR", step_size=len(train_loader) * 2, lr=args.lr)
    else:
        raise ValueError(args.scheduler)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    save_dir = Path("output") / "weather" / args.model
    save_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.note}__s{args.seed}"
    model_path = save_dir / f"model__{run_id}.pt"
    config_path = save_dir / f"config__{run_id}.json"
    summary_path = save_dir / f"result__{run_id}.txt"
    predictions_path = save_dir / f"predictions__{run_id}.csv"
    outputs = [model_path, config_path, summary_path, predictions_path]
    existing = [str(p) for p in outputs if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output file(s) already exist; choose a new --note or pass --overwrite: "
            + ", ".join(existing)
        )
    for p in outputs:
        if p.exists():
            p.unlink()
            print(f"Removed old output: {p}")
    config_path.write_text(json.dumps({
        "dataset": "weather", "target": args.target, "model": args.model,
        "args": vars(args), "split": {"train_ratio": 0.70, "validation_ratio": 0.10, "test_ratio": 0.20},
        "target_scaling": "train-only z-score", "calendar_features": args.calendar
    }, indent=2))
    best_val, best_epoch = float("inf"), 0
    if device.type == "cuda": torch.cuda.synchronize()
    start = time.perf_counter()

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        train_loss_scaled, train_mae, train_rmse = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler, per_batch, y_std, use_amp)
        if not per_batch: scheduler.step()
        val = evaluate(model, val_loader, criterion, y_mean, y_std)
        if device.type == "cuda": torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch+1:02d}/{args.epochs:<2d} | "
            f"Train RMSE: {train_rmse:7.4f}, MAE: {train_mae:7.4f} | "
            f"Val RMSE: {val['rmse']:7.4f}, MAE: {val['mae']:7.4f}, R2: {val['r2']:7.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:10.3e} | "
            f"Time: {epoch_time:5.1f}s"
        )
        if math.isfinite(val["rmse"]) and val["rmse"] < best_val:
            best_val, best_epoch = val["rmse"], epoch + 1
            torch.save(model.state_dict(), model_path)
    if device.type == "cuda": torch.cuda.synchronize()
    train_time = time.perf_counter() - start
    params = count_params(model)
    if best_epoch == 0 or not model_path.exists():
        raise RuntimeError("No finite validation RMSE was obtained; checkpoint was not saved. Try a lower --lr or inspect the data/model.")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    test = evaluate(model, test_loader, criterion, y_mean, y_std)
    test_idx = np.asarray(test_idx, dtype=int)
    if len(test_idx) != len(test["y_true"]):
        raise RuntimeError(f"Test size mismatch: {len(test_idx)} indices vs {len(test['y_true'])} predictions.")
    # Use the original (un-scaled) targets as ground truth; the reconstructed ones carry float32 round-off.
    y_true = raw_y[test_idx].astype(np.float64)
    if not np.allclose(test["y_true"], y_true, rtol=1e-4, atol=1e-4 * y_std):
        raise RuntimeError("Test targets do not line up with test_idx; check the data pipeline.")
    y_pred = test["y_pred"].astype(np.float64)
    test_ts = pd.DatetimeIndex(pd.to_datetime(timestamps[test_idx]))
    ts_to_y = pd.Series(raw_y.astype(np.float64), index=pd.DatetimeIndex(pd.to_datetime(timestamps)))
    pred_frame = pd.DataFrame({"timestamp": test_ts, "y_true": y_true, "y_pred": y_pred})

    h = args.horizon
    baselines = [(f"persistence_lag{h}", h)]
    for name, lag in [("seasonal_naive_24h", 144), ("seasonal_naive_168h", 1008)]:
        if lag >= h:
            baselines.append((name, lag))
    base_preds = {}
    for name, lag in baselines:
        prev_ts = test_ts - pd.Timedelta(minutes=10 * lag)
        available = np.asarray(prev_ts.isin(ts_to_y.index))
        pred_base = np.full(len(test_idx), np.nan, dtype=np.float64)
        if available.any():
            pred_base[available] = ts_to_y.loc[prev_ts[available]].to_numpy(dtype=np.float64)
        base_preds[name] = pred_base
        pred_frame[f"{name}_pred"] = pred_base

    common = np.isfinite(y_pred) & np.all([np.isfinite(p) for p in base_preds.values()], axis=0)
    if not common.any():
        raise RuntimeError("No test target has a prediction from every baseline.")
    pred_frame["in_common_subset"] = common

    def metrics(a, b):
        err = b - a
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        ss_tot = float(np.sum((a - a.mean()) ** 2))
        r2 = float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan")
        return rmse, mae, r2

    model_rmse, model_mae, model_r2 = metrics(y_true[common], y_pred[common])
    baseline_lines = [f"Common subset: n={int(common.sum()):,}/{len(y_true):,}; all methods use the same targets.",
                      f"model: RMSE={model_rmse:.6f}, MAE={model_mae:.6f}, R2={model_r2:.6f}"]
    baseline_records = []
    for name, lag in baselines:
        pred_base = base_preds[name]
        brmse, bmae, br2 = metrics(y_true[common], pred_base[common])
        baseline_lines.append(f"{name}: RMSE={brmse:.6f}, MAE={bmae:.6f}, R2={br2:.6f} "
                              f"(lag {lag} steps; available for {int(np.isfinite(pred_base).sum()):,} targets)")
        baseline_records.append({"name": name, "lag": lag, "n": int(common.sum()),
                                 "rmse": brmse, "mae": bmae, "r2": br2})

    pred_frame.to_csv(predictions_path, index=False)

    # Full test metrics use every test target; common-subset metrics are used for baseline comparison.
    test_rmse, test_mae, test_r2 = metrics(y_true, y_pred)
    sep = "=" * 72
    print(f"\n{sep}")
    print(f" {args.model.upper()} | Target: {args.target} | Horizon: {h}")
    print(sep)
    print(f" {'Parameters':<20}: {params:>15,}")
    print(f" {'Best Val RMSE':<20}: {best_val:>15.6f}  (Epoch {best_epoch})")
    print(f" {'Train time':<20}: {train_time:>14.1f} s")
    print("-" * 72)
    print(f" {'Test Set':<12} {'N':>10} {'RMSE':>12} {'MAE':>12} {'R2':>10}")
    print("-" * 72)
    print(f" {'Full':<12} {len(y_true):>10,} {test_rmse:>12.6f} {test_mae:>12.6f} {test_r2:>10.6f}")
    print(f" {'Common':<12} {int(common.sum()):>10,} {model_rmse:>12.6f} {model_mae:>12.6f} {model_r2:>10.6f}")
    print(sep)
    print("\nBaselines:")
    for line in baseline_lines:
        print(f"  {line}")

    # The separate history/summary JSON outputs are not created.
    summary_path.write_text(
        f"Dataset: Weather (Max Planck Institute; 10-minute frequency)\nTarget: {args.target}\n"
        f"Model: {args.model}\nNote: {args.note}\nSeed: {args.seed}\n"
        f"Lookback: {args.lookback} steps ({args.lookback * 10} minutes)\n"
        f"Forecast horizon: {h} step(s) ({h * 10} minutes ahead)\n"
        f"Chronological split: 70%/10%/20% by target timestamp\n"
        f"Historical input: univariate target series only; calendar covariates at target time: {calendar_cols}\n"
        f"Target scaling: train-only z-score (mean={y_mean:.8f}, std={y_std:.8f})\n"
        f"Epochs: {args.epochs}\nBatch size: {args.batch_size}\nScheduler: {args.scheduler}\n"
        f"LR: {args.lr}\nWeight decay: {args.weight_decay}\nAMP: {'fp16' if use_amp else 'off'}\n"
        f"Parameters: {params}\nTrain time: {train_time:.2f}s\nBest epoch: {best_epoch}\n"
        f"Best Val RMSE: {best_val:.8f}\n"
        f"Test full: n={len(y_true)}, RMSE={test_rmse:.8f}, MAE={test_mae:.8f}, R2={test_r2:.8f}\n"
        f"Test common subset: n={int(common.sum())}, RMSE={model_rmse:.8f}, MAE={model_mae:.8f}, R2={model_r2:.8f}\n"
        f"Predictions CSV: {predictions_path}\n\nBaselines:\n" + "\n".join(baseline_lines) + "\n")


def get_args():
    p = argparse.ArgumentParser(description="Univariate Weather forecasting with chronological train/validation/test splits.")
    p.add_argument("--model", default="mlp",
        choices=["mlp", "efficient_kan", "cnn_1d", "resnet_1d", "ds_cnn_1d", "basis_tabm", "tabm", "bsrbf_kan"])
    p.add_argument("--data_path", default="data/weather.csv")
    p.add_argument("--target", default="OT", help="Single Weather column to forecast (default: OT).")
    p.add_argument("--note", default="run_0", help="Unique output tag; must not be empty")
    p.add_argument("--overwrite", action="store_true", help="Delete existing outputs with the same note/seed before running")
    p.add_argument("--lookback", type=int, default=144, help="Historical steps; 144 steps = 24 hours at 10-minute frequency.")
    p.add_argument("--horizon", type=int, default=1, help="Forecast horizon in 10-minute steps; 1 = next 10 minutes.")
    p.add_argument("--calendar", default="tod", choices=["tod", "all"],
                   help="tod: time-of-day only (default). all: also weekday/month "
                        "(month values in the test period are unseen in training for this one-year dataset).")
    p.add_argument("--hidden_layers", default="256", help="Ignored by cnn_1d/resnet_1d/ds_cnn_1d.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--scheduler", default="OneCycleLR", choices=["StepLR", "CosineAnnealingLR", "OneCycleLR", "ExponentialLR", "CyclicLR"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--norm_type", default="layer")
    p.add_argument("--activation", default="silu")
    p.add_argument("--norm1_type", default="layer")
    p.add_argument("--norm2_type", default="")
    p.add_argument("--tabm_k", type=int, default=8, help="TabM ensemble size")
    p.add_argument("--grid_size", type=int, default=5, help="Grid size for efficient_kan/bsrbf_kan")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--no_amp", action="store_true", help="Disable fp16 mixed precision (useful for KAN stability checks)")
    return p.parse_args()


def main():
    global device
    args = get_args()
    if not args.note.strip():
        raise ValueError("--note must be non-empty to prevent accidental output overwrites.")
    if min(args.lookback, args.horizon, args.epochs, args.batch_size) < 1:
        raise ValueError("lookback/horizon/epochs/batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be >= 0.")
    if args.tabm_k < 1 or args.grid_size < 1:
        raise ValueError("--tabm_k and --grid_size must be >= 1.")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    set_seed(args.seed, device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    (train_loader, val_loader, test_loader, n_features, input_dim, n_calendar,
     y_mean, y_std, test_idx, raw_y, timestamps, calendar_cols) = get_dataloaders(args)
    model = build_model(args, n_features, input_dim, n_calendar)
    train_model(model, train_loader, val_loader, test_loader, args, y_mean, y_std,
                test_idx, raw_y, timestamps, calendar_cols)


if __name__ == "__main__":
    main()


# Example runs (Weather: univariate OT, 24-hour lookback, 10-minute next-step forecast)
# Add --calendar all for the weekday/month ablation; add --no_amp to train in FP32.

# 38693
# python run_weather.py --model "mlp" --target "OT" --lookback 144 --horizon 1 --hidden_layers "256" --batch_size 64 --epochs 30 --norm_type "layer" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite --no_amp

# 38220
# python run_weather.py --model "efficient_kan" --target "OT" --lookback 144 --horizon 1 --hidden_layers "26" --grid_size 5 --batch_size 64 --epochs 30 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite --no_amp

# 38717
# python run_weather.py --model "bsrbf_kan" --target "OT" --lookback 144 --horizon 1 --hidden_layers "29" --grid_size 5 --norm_type "layer" --batch_size 64 --epochs 1 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite --no_amp

# 38693
# python run_weather.py --model "cnn_1d" --target "OT" --lookback 144 --horizon 1 --batch_size 64 --epochs 1 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite --no_amp

# 38693
# python run_weather.py --model "resnet_1d" --target "OT" --lookback 144 --horizon 1 --batch_size 64 --epochs 1 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite --no_amp

# 38693
# python run_weather.py --model "ds_cnn_1d" --target "OT" --lookback 144 --horizon 1 --batch_size 64 --epochs 1 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite --no_amp

# 38752
# python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "232" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "layer" --norm2_type "" --batch_size 64 --epochs 1 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite --no_amp

# 38584
# python run_weather.py --model "tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "140" --tabm_k 32 --batch_size 64 --epochs 1 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite --no_amp
 