import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse, json, time, zipfile, urllib.request, math
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
from cnn_1d import CNN1D_Bike
from resnet_1d import ResNet1D_Bike
from ds_cnn_1d import DSCNN1D_Bike
from basis_tabm import BasisTabM

from schedulers import get_scheduler
from utils import set_seed, count_params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BikeModel(nn.Module):
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


def load_hour_csv(path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
        archive = path.parent / "bike-sharing-dataset.zip"
        print(f"Downloading Bike Sharing dataset to {archive} ...")
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as zf:
            member = next((n for n in zf.namelist() if n.endswith("hour.csv")), None)
            if member is None: raise FileNotFoundError("hour.csv not found in downloaded archive")
            path.write_bytes(zf.read(member))
    return pd.read_csv(path)


def get_dataloaders(args):
    df = load_hour_csv(args.data_path)
    required = {"dteday", "hr", "weekday", "mnth", "cnt", "holiday", "workingday", "season"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {required}; found {list(df.columns)}")
    df["timestamp"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df["hr_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
    df["hr_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["mnth_sin"] = np.sin(2 * np.pi * df["mnth"] / 12)
    df["mnth_cos"] = np.cos(2 * np.pi * df["mnth"] / 12)
    calendar_cols = ["hr_sin", "hr_cos", "weekday_sin", "weekday_cos", "mnth_sin", "mnth_cos", "holiday", "workingday", "season"]
    excluded = {"instant", "dteday", "timestamp", "casual", "registered"}
    features = [c for c in df.columns if c not in excluded | {"cnt"} | set(calendar_cols)]
    features.append("cnt")  # historical count only
    raw_x = df[features].to_numpy(dtype=np.float32)
    raw_cal = df[calendar_cols].to_numpy(dtype=np.float32)
    raw_y = df["cnt"].to_numpy(dtype=np.float32)
    timestamps = df["timestamp"].to_numpy()
    n = len(df)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    if train_end <= args.lookback or val_end <= train_end or n <= val_end:
        raise ValueError("Dataset is too small for the selected lookback and chronological split.")
    # Fit both scalers using training-period rows only. Scale y for stable optimization.
    x_mean, x_std = raw_x[:train_end].mean(0), raw_x[:train_end].std(0)
    x_std[x_std < 1e-6] = 1.0
    cal_mean, cal_std = raw_cal[:train_end].mean(0), raw_cal[:train_end].std(0)
    cal_std[cal_std < 1e-6] = 1.0
    y_mean, y_std = float(raw_y[:train_end].mean()), float(raw_y[:train_end].std())
    if not np.isfinite(y_std) or y_std < 1e-6: y_std = 1.0
    x_scaled, cal_scaled = (raw_x - x_mean) / x_std, (raw_cal - cal_mean) / cal_std
    y_scaled = (raw_y - y_mean) / y_std
    X, C, Y, target_idx = [], [], [], []
    for t in range(args.lookback, n):
        if not np.all(np.diff(timestamps[t - args.lookback:t + 1]) == np.timedelta64(1, "h")):
            continue
        X.append(x_scaled[t - args.lookback:t])
        C.append(cal_scaled[t])  # known calendar covariates at forecast time t
        Y.append(y_scaled[t]); target_idx.append(t)
    if not X: raise ValueError("No valid hourly windows found. Check hour.csv timestamps.")
    X, C, Y, target_idx = np.asarray(X, np.float32), np.asarray(C, np.float32), np.asarray(Y, np.float32), np.asarray(target_idx)
    tr, va, te = target_idx < train_end, (target_idx >= train_end) & (target_idx < val_end), target_idx >= val_end
    if not tr.any() or not va.any() or not te.any():
        raise ValueError(f"Empty split after filtering hourly windows: train={int(tr.sum())}, val={int(va.sum())}, test={int(te.sum())}.")
    
    
    def loader(mask, shuffle=False):
        ds = TensorDataset(torch.from_numpy(X[mask]), torch.from_numpy(C[mask]), torch.from_numpy(Y[mask]))
        n, bs = len(ds), args.batch_size

        if shuffle:
            sampler = torch.utils.data.RandomSampler(ds)
            if n % bs == 1 and n > bs:
                sizes = [bs] * (n // bs - 1) + [bs - 1, 2]

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

            return DataLoader(ds, batch_sampler=batch_sampler,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda",
                              persistent_workers=args.num_workers > 0)

        return DataLoader(ds, batch_size=bs, shuffle=False,
                          num_workers=args.num_workers, pin_memory=device.type == "cuda",
                          persistent_workers=args.num_workers > 0)

    
    print(f"Bike Sharing | Train: {tr.sum():,} | Val: {va.sum():,} | Test: {te.sum():,} | Lookback: {args.lookback}h | Historical features: {len(features)} | Target calendar features: {len(calendar_cols)}")
    print(f"Skipped {n - args.lookback - len(X):,} candidate windows crossing missing/non-consecutive hours; missing rows are not assumed to mean zero demand.")
    return loader(tr, True), loader(va), loader(te), len(features), len(features) * args.lookback + len(calendar_cols), len(calendar_cols), y_mean, y_std, target_idx[te], raw_y, timestamps

def build_model(args, n_features, input_dim, n_calendar):
    hidden = [int(v.strip()) for v in args.hidden_layers.split(",") if v.strip()]
    layers = [input_dim] + hidden + [1]
    if args.model == "mlp":
        net = MLP(layers, base_activation=args.activation, norm_type=args.norm_type)
    elif args.model == "efficient_kan":
        net = EfficientKAN(layers_hidden=layers, grid_size=args.grid_size, spline_order=3)
    elif args.model == "bsrbf_kan":
        net = BSRBF_KAN(layers_hidden=layers, grid_size=args.grid_size, spline_order=3, norm_type = args.norm_type)  
    elif args.model == "tabm":
        if len(set(hidden)) != 1:
            raise ValueError("TabM needs equal widths in --hidden_layers.")
        net = tabm.TabM.make(n_num_features=input_dim, cat_cardinalities=[], d_out=1,
                             d_block=hidden[0], n_blocks=len(hidden), k=args.tabm_k,
                             num_embeddings=None)
        
    elif args.model == "basis_tabm":
        net = BasisTabM(net_layers=layers, k=args.tabm_k, basis="rbf",
                    use_width=False, width=1.0, norm1_type=args.norm1_type,
                    norm2_type=args.norm2_type, activation=args.activation, dropout = args.dropout)
    elif args.model == "cnn_1d": net = CNN1D_Bike(n_features + n_calendar)
    elif args.model == "resnet_1d": net = ResNet1D_Bike(n_features + n_calendar)
    elif args.model == "ds_cnn_1d": net = DSCNN1D_Bike(n_features + n_calendar)
    else: raise ValueError(args.model)
    return BikeModel(net, args.model, n_calendar)


def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, per_batch, y_std):
    model.train(); loss_sum = abs_sum = sq_sum = n_total = 0
    for x, cal, y in tqdm(loader, desc="Training", leave=False):
        x, cal, y = x.to(device, non_blocking=True), cal.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            pred = model(x, cal)
            if pred.ndim == 2:
                loss = criterion(pred, y.unsqueeze(1).expand_as(pred))
                pred_metric = pred.mean(dim=1)
            else:
                loss = criterion(pred, y); pred_metric = pred
        old_scale = scaler.get_scale(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        if per_batch and scaler.get_scale() >= old_scale: scheduler.step()
        err = (pred_metric.detach().float() - y) * y_std
        n = y.numel(); loss_sum += loss.item() * n; abs_sum += err.abs().sum().item(); sq_sum += err.square().sum().item(); n_total += n
    return loss_sum / n_total, abs_sum / n_total, (sq_sum / n_total) ** 0.5


@torch.no_grad()
def evaluate(model, loader, criterion, y_mean, y_std):
    model.eval(); loss_sum = abs_sum = sq_sum = 0.0; n_total = 0; ys, preds = [], []
    for x, cal, y in loader:
        x, cal, y = x.to(device, non_blocking=True), cal.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # Evaluate in FP32 to avoid FP16 quantization of count predictions.
        pred = model(x.float(), cal.float())
        if pred.ndim == 2: pred = pred.float().mean(dim=1)
        else: pred = pred.float()
        loss = criterion(pred, y.float())
        y_real = y.float() * y_std + y_mean
        pred_real = (pred * y_std + y_mean).clamp_min(0.0)
        err = pred_real - y_real; n = y.numel()
        loss_sum += loss.item() * n; abs_sum += err.abs().sum().item(); sq_sum += err.square().sum().item(); n_total += n
        ys.append(y_real.cpu().numpy()); preds.append(pred_real.cpu().numpy())
    y_true, y_pred = np.concatenate(ys), np.concatenate(preds)
    ss_res = np.square(y_true - y_pred).sum(); ss_tot = np.square(y_true - y_true.mean()).sum()
    return {"loss_scaled": loss_sum / n_total, "mae": abs_sum / n_total, "rmse": (sq_sum / n_total) ** 0.5,
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "y_true": y_true, "y_pred": y_pred}

def train_model(model, train_loader, val_loader, test_loader, args, y_mean, y_std, test_idx, raw_y, timestamps):
    print(f"\nModel: {args.model}\nSeed: {args.seed}")
    model = model.to(device)
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

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    save_dir = Path("output") / "bike" / args.model
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"{args.model}__{args.note}.pt"
    history_path = save_dir / f"history__{args.note}.jsonl"
    summary_path = save_dir / f"summary__{args.note}.txt"
    predictions_path = save_dir / f"predictions__{args.note}.csv"
    outputs = [model_path, history_path, summary_path, predictions_path]
    existing = [str(p) for p in outputs if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output file(s) already exist; choose a new --note or pass --overwrite: "
            + ", ".join(existing)
        )
    history_path.write_text("")
    best_val, best_epoch = float("inf"), 0
    if device.type == "cuda": torch.cuda.synchronize()
    start = time.perf_counter()

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        train_loss_scaled, train_mae, train_rmse = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler, per_batch, y_std)
        if not per_batch: scheduler.step()
        val = evaluate(model, val_loader, criterion, y_mean, y_std)
        if device.type == "cuda": torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start
        print(f"Epoch {epoch+1:02d}/{args.epochs} | Train RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f} | "
              f"Val RMSE: {val['rmse']:.2f}, MAE: {val['mae']:.2f}, R2: {val['r2']:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6g} | Time: {epoch_time:.1f}s")
        if math.isfinite(val["rmse"]) and val["rmse"] < best_val:
            best_val, best_epoch = val["rmse"], epoch + 1
            torch.save(model.state_dict(), model_path)
        with history_path.open("a") as f:
            f.write(json.dumps({"epoch": epoch + 1, "train_loss_scaled": train_loss_scaled,
                "train_mae": train_mae, "train_rmse": train_rmse,
                "val": {k: v for k, v in val.items() if k not in {"y_true", "y_pred"}},
                "lr": optimizer.param_groups[0]["lr"], "time": epoch_time,
                "best_epoch": best_epoch}) + "\n")

    if device.type == "cuda": torch.cuda.synchronize()
    train_time = time.perf_counter() - start
    params = count_params(model)
    if best_epoch == 0 or not model_path.exists():
        raise RuntimeError("No finite validation RMSE was obtained; checkpoint was not saved. Try a lower --lr or inspect the data/model.")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    test = evaluate(model, test_loader, criterion, y_mean, y_std)
    test_idx = np.asarray(test_idx, dtype=int)
    # Use exact raw targets for final test metrics and prediction exports.
    test["y_true"] = np.asarray(raw_y[test_idx], dtype=np.float64)
    test_err = test["y_pred"] - test["y_true"]
    test["mae"] = float(np.mean(np.abs(test_err)))
    test["rmse"] = float(np.sqrt(np.mean(np.square(test_err))))
    test_ss_tot = float(np.sum(np.square(test["y_true"] - test["y_true"].mean())))
    test["r2"] = float(1 - np.sum(np.square(test_err)) / test_ss_tot) if test_ss_tot > 0 else float("nan")
    test_ts = pd.DatetimeIndex(pd.to_datetime(timestamps[test_idx]))
    ts_to_y = pd.Series(raw_y.astype(np.float64), index=pd.DatetimeIndex(pd.to_datetime(timestamps)))
    pred_frame = pd.DataFrame({"timestamp": test_ts, "y_true": raw_y[test_idx], "y_pred": test["y_pred"]})
    baseline_lines = []
    for name, lag in [("persistence", 1), ("seasonal_naive_24h", 24), ("seasonal_naive_168h", 168)]:
        prev_ts = test_ts - pd.Timedelta(hours=lag)
        mask = prev_ts.isin(ts_to_y.index)
        pred_base = np.full(len(test_idx), np.nan, dtype=np.float64)
        if mask.any():
            pred_base[mask] = ts_to_y.loc[prev_ts[mask]].to_numpy(dtype=np.float64)
            actual = np.asarray(raw_y[test_idx])[mask]
            model_pred = np.asarray(test["y_pred"])[mask]
            base_pred = pred_base[mask]
            def metrics(y_true, y_pred):
                rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                mae = float(np.mean(np.abs(y_true - y_pred)))
                ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
                r2 = float(1 - np.sum((y_true - y_pred) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
                return rmse, mae, r2
            b_rmse, b_mae, b_r2 = metrics(actual, base_pred)
            m_rmse, m_mae, m_r2 = metrics(actual, model_pred)
            baseline_lines.append(
                f"{name}: n={int(mask.sum())}, RMSE={b_rmse:.6f}, MAE={b_mae:.6f}, R2={b_r2:.6f}; "
                f"model_same_subset: RMSE={m_rmse:.6f}, MAE={m_mae:.6f}, R2={m_r2:.6f}"
            )
        else:
            baseline_lines.append(f"{name}: unavailable (no test targets with a timestamp exactly {lag}h earlier)")
        pred_frame[f"{name}_pred"] = pred_base
    pred_frame.to_csv(predictions_path, index=False)
    print(f"\n{'-'*58}\nModel: {args.model}\nParameters: {params:,}\n"
          f"Best Val RMSE: {best_val:.3f} (epoch {best_epoch})\n"
          f"Test RMSE: {test['rmse']:.3f}\nTest MAE: {test['mae']:.3f}\n"
          f"Test R2: {test['r2']:.4f}\nTrain time: {train_time:.1f}s\n{'-'*58}")
    print("\nBaselines (see summary file):\n" + "\n".join(baseline_lines))
    summary_path.write_text(
        f"Dataset: UCI Bike Sharing (hour.csv)\nModel: {args.model}\nNote: {args.note}\n"
        f"Lookback: {args.lookback}\nEpochs: {args.epochs}\nBatch size: {args.batch_size}\n"
        f"Scheduler: {args.scheduler}\nLR: {args.lr}\nWeight decay: {args.weight_decay}\n"
        f"Target scaling: z-score using train mean={y_mean:.6f}, std={y_std:.6f}\n"
        f"Prediction post-processing: clamp predictions to non-negative values before evaluation\n"
        f"Target-time calendar covariates: hour/weekday/month cyclic encodings, holiday, workingday, season\n"
        f"BasisTabM k: {args.tabm_k}\nEfficientKAN grid_size: {args.grid_size}\n"
        f"Parameters: {params}\nTrain time: {train_time:.2f}s\nPost-training/evaluation time excluded from train time.\n"
        f"Predictions CSV: {predictions_path}\nBest epoch: {best_epoch}\n"
        f"Best Val RMSE: {best_val:.6f}\nTest RMSE: {test['rmse']:.6f}\n"
        f"Test MAE: {test['mae']:.6f}\nTest R2: {test['r2']:.6f}\n\nBaselines (raw-count scale):\n" + "\n".join(baseline_lines) + "\n")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="basis_tabm",
        choices=["mlp", "efficient_kan", "cnn_1d", "resnet_1d", "ds_cnn_1d", "basis_tabm", "tabm", "bsrbf_kan"])
    p.add_argument("--data_path", default="data/hour.csv")
    p.add_argument("--note", default="run_0", help="Unique output tag; must not be empty")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting outputs with the same note")
    p.add_argument("--lookback", type=int, default=24)
    p.add_argument("--hidden_layers", default="256")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--scheduler", default="OneCycleLR", choices=["StepLR", "CosineAnnealingLR", "OneCycleLR", "ExponentialLR", "CyclicLR"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--norm_type", default="layer")
    p.add_argument("--activation", default="silu")
    p.add_argument("--norm1_type", default="layer")
    p.add_argument("--norm2_type", default="")
    p.add_argument("--norm_mode", default="all", choices=["none", "first", "except_first", "all"])
    p.add_argument("--num_grids", type=int, default=8, help="Basis grid count for KAN/basis functions")
    p.add_argument("--tabm_k", type=int, default=8, help="ensemble size = number of basis grid points")
    p.add_argument("--grid_size", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dropout", type=float, default=0.0)
    
    return p.parse_args()


def main():
    global device
    args = get_args()
    if not args.note.strip():
        raise ValueError("--note must be non-empty to prevent accidental output overwrites.")
    if args.lookback < 1:
        raise ValueError("--lookback must be >= 1.")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch_size must be >= 1.")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be >= 0.")
    if args.num_grids < 1 or args.tabm_k < 1 or args.grid_size < 1:
        raise ValueError("--num_grids, --tabm_k, and --grid_size must be >= 1.")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    set_seed(args.seed, device)
    print(f"Device: {device}")
    if device.type == "cuda": print(f"GPU: {torch.cuda.get_device_name(0)}")
    train_loader, val_loader, test_loader, n_features, input_dim, n_calendar, y_mean, y_std, test_idx, raw_y, timestamps = get_dataloaders(args)
    model = build_model(args, n_features, input_dim, n_calendar)
    train_model(model, train_loader, val_loader, test_loader, args, y_mean, y_std, test_idx, raw_y, timestamps)


if __name__ == "__main__":
    main()

# 65267
# python run_bike.py --model "mlp" --lookback 24 --hidden_layers "256" --batch_size 64 --epochs 1 --norm_type "batch" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite

# 65267
# python run_bike.py --model "cnn_1d" --lookback 24 --batch_size 64 --epochs 1 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite

# 65267
# python run_bike.py --model "resnet_1d" --lookback 24 --batch_size 64 --epochs 30 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite

# 65267
# python run_bike.py --model "ds_cnn_1d" --lookback 24 --batch_size 64 --epochs 30 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite

# 65000
# python run_bike.py --model "efficient_kan" --lookback 24 --batch_size 64 --hidden_layers "26" --grid_size 5 --epochs 30 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite

# 65806
# python run_bike.py --model "bsrbf_kan" --lookback 24 --batch_size 64 --hidden_layers "29" --grid_size 5 --norm_type "layer" --epochs 30 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite

# 65172
# python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "234" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "layer" --norm2_type "" --epochs 30 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite

# 65270
# python run_bike.py --model "tabm" --lookback 24 --hidden_layers "166" --tabm_k 32 --batch_size 64 --epochs 30 --scheduler "OneCycleLR" --seed 0 --note "run_0" --overwrite