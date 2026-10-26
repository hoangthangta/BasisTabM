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

from models.mlp import MLP
from models.efficient_kan import EfficientKAN
from models.bsrbf_kan import BSRBF_KAN
from models.basis_tabm import BasisTabM
from models.patchtst_model import PatchTSTModel

from schedulers import get_scheduler
from utils import set_seed, count_params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Explicit input features. "cnt" must stay last (PatchTST reads the last channel).
WEATHER_FEATURES = ["weathersit", "temp", "atemp", "hum", "windspeed"]
# Calendar features of each lookback hour (enabled with --calendar); cyclic sin/cos encoding.
# Weekday uses the dataset's own 'weekday' column (0 = Sunday).
CALENDAR_FEATURES = ["hr_sin", "hr_cos", "wd_sin", "wd_cos", "mnth_sin", "mnth_cos",
                     "holiday", "workingday"]
TARGET = "cnt"

def add_calendar_columns(df):
    df["hr_sin"], df["hr_cos"] = np.sin(2 * np.pi * df["hr"] / 24), np.cos(2 * np.pi * df["hr"] / 24)
    df["wd_sin"], df["wd_cos"] = np.sin(2 * np.pi * df["weekday"] / 7), np.cos(2 * np.pi * df["weekday"] / 7)
    m = df["mnth"] - 1
    df["mnth_sin"], df["mnth_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    return df

ENSEMBLE_MODELS = {"tabm", "basis_tabm"}


class BikeModel(nn.Module):
    """Output shapes: [B, H] for single models; [B, k, H] for ensembles.

    PatchTST (--patchtst_exog):
      univariate : use only cnt; ignore exogenous features.
      mix        : use all channels; learn a linear mapping to cnt, initialized
                   to select cnt only (equivalent to univariate at initialization).
    """

    def __init__(self, model, model_type, n_features, horizon, k=1, patchtst_exog="univariate"):
        super().__init__()
        self.model, self.model_type = model, model_type
        self.n_features, self.horizon, self.k = n_features, horizon, k
        self.is_ensemble = model_type in ENSEMBLE_MODELS
        self.patchtst_exog = patchtst_exog
        self.mix = None
        if model_type == "patchtst" and patchtst_exog == "mix":
            self.mix = nn.Linear(n_features, 1)
            with torch.no_grad():
                self.mix.weight.zero_()
                self.mix.weight[0, -1] = 1.0  # cnt is the last channel
                self.mix.bias.zero_()

    def forward(self, x):
        B = x.shape[0]
        if self.model_type == "patchtst":
            if self.patchtst_exog == "univariate":
                x = x[:, :, -1:]  # cnt channel only
            C = x.shape[2]
            out = self.model(x)
            # PatchTSTModel squeezes the channel dim when C == 1 -> restore [B, horizon, 1].
            if C == 1 and out.ndim == 2:
                out = out.unsqueeze(-1)
            # Expected PatchTST output: [B, horizon, C] (channel-independent forecast).
            if out.ndim != 3 or out.shape[1] != self.horizon or out.shape[2] != C:
                raise RuntimeError(f"PatchTST output shape {tuple(out.shape)} != "
                                   f"[B, {self.horizon}, {C}]")
            if self.mix is None:
                return out[:, :, -1]
            return self.mix(out).squeeze(-1)

        x = x.flatten(1)
        out = self.model(x)
        if self.is_ensemble:
            # TabM convention: [B, k, d_out] with d_out = horizon.
            return out.reshape(B, self.k, self.horizon)
        return out.reshape(B, self.horizon)


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
    df["timestamp"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    ts = df["timestamp"].to_numpy(dtype="datetime64[ns]")

    feature_cols = WEATHER_FEATURES.copy()
    if args.calendar:
        df = add_calendar_columns(df)
        feature_cols += CALENDAR_FEATURES
    feature_cols += [TARGET]
    assert feature_cols[-1] == TARGET, "cnt must be the last input channel"
    args.feature_names = feature_cols.copy()
    args.target_names = [TARGET]
    raw_x = df[feature_cols].to_numpy(dtype=np.float32)
    raw_y = df[TARGET].to_numpy(dtype=np.float32)

    n, L, H = len(df), args.lookback, args.horizon
    if n <= L + H - 1:
        raise ValueError(f"Not enough rows ({n}) for lookback={L}, horizon={H}.")

    X_raw, Y_raw, origins = [], [], []
    for t0 in range(L, n-H+1):
        if np.all(np.diff(ts[t0-L:t0+H]) == np.timedelta64(1, "h")):
            X_raw.append(raw_x[t0-L:t0])
            Y_raw.append(raw_y[t0:t0+H])
            origins.append(t0)
    if not origins:
        raise ValueError("No valid consecutive hourly windows found.")

    X_raw, Y_raw = np.asarray(X_raw, np.float32), np.asarray(Y_raw, np.float32)
    origins = np.asarray(origins, np.int64)
    last_idx, n_valid = origins + H - 1, len(origins)
    train_cut, val_cut = int(n_valid*.70), int(n_valid*.85)
    if train_cut == 0 or val_cut <= train_cut or val_cut >= n_valid:
        raise ValueError(f"Too few valid windows: {n_valid}.")

    train_end, val_end = int(origins[train_cut]), int(origins[val_cut])
    tr = last_idx < train_end
    va = (origins >= train_end) & (last_idx < val_end)
    te = origins >= val_end
    n_boundary = n_valid - int(tr.sum() + va.sum() + te.sum())
    total = int(tr.sum() + va.sum() + te.sum())
    if not tr.any() or not va.any() or not te.any():
        raise ValueError("At least one split is empty; check lookback/horizon.")

    print(f"Valid windows: {n_valid:,} | Boundary windows dropped: {n_boundary:,}")
    print(f"Train: {tr.sum():,} ({tr.sum()/total:.1%}) | Val: {va.sum():,} ({va.sum()/total:.1%}) | Test: {te.sum():,} ({te.sum()/total:.1%})")

    # Fit scalers on training-period rows only.
    x_mean, x_std = raw_x[:train_end].mean(0), raw_x[:train_end].std(0)
    x_std[x_std < 1e-8] = 1.0
    y_mean, y_std = raw_y[:train_end].mean(), raw_y[:train_end].std()
    if y_std < 1e-8:
        y_std = 1.0
    # PatchTST forecasts cnt in the X-scaling of the cnt channel, while loss/de-normalisation
    # use (y_mean, y_std). They must be identical.
    assert np.isclose(x_mean[-1], y_mean) and np.isclose(x_std[-1], y_std), (
        f"cnt input scaling ({x_mean[-1]}, {x_std[-1]}) != target scaling ({y_mean}, {y_std})")
    X = ((X_raw - x_mean) / x_std).astype(np.float32)
    Y = ((Y_raw - y_mean) / y_std).astype(np.float32)

    def make_loader(mask, shuffle=False):
        ds = TensorDataset(torch.from_numpy(X[mask]), torch.from_numpy(Y[mask]))
        bs = args.batch_size
        return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                          drop_last=shuffle and len(ds) % bs == 1,
                          num_workers=args.num_workers,
                          pin_memory=torch.cuda.is_available())

    train_loader, val_loader, test_loader = make_loader(tr, True), make_loader(va), make_loader(te)
    xb, yb = next(iter(train_loader))
    print(f"Lookback: {L}h | Horizon: {H}h | Input: {tuple(xb.shape)} | Output: {tuple(yb.shape)}\n"
          f"Input Features ({len(feature_cols)}): {feature_cols}\n"
          f"Output Features (1): {[TARGET]}")

    return (train_loader, val_loader, test_loader, len(feature_cols),
            len(feature_cols)*L, y_mean, y_std, origins[te], raw_y, ts)


def build_model(args, n_features, input_dim):
    H = args.horizon
    hidden = [int(v.strip()) for v in args.hidden_layers.split(",") if v.strip()]
    layers = [input_dim] + hidden + [H]
    if args.model == "mlp":
        net = MLP(layers, base_activation=args.activation, norm_type=args.norm_type)
    elif args.model == "efficient_kan":
        net = EfficientKAN(layers_hidden=layers, grid_size=args.grid_size, spline_order=3)
    elif args.model == "bsrbf_kan":
        net = BSRBF_KAN(layers_hidden=layers, grid_size=args.grid_size, spline_order=3, norm_type=args.norm_type)
    elif args.model == "tabm":
        if len(set(hidden)) != 1:
            raise ValueError("TabM needs equal widths in --hidden_layers.")
        net = tabm.TabM.make(n_num_features=input_dim, cat_cardinalities=[], d_out=H,
                             d_block=hidden[0], n_blocks=len(hidden), k=args.tabm_k,
                             num_embeddings=None)
    elif args.model == "basis_tabm":
        net = BasisTabM(net_layers=layers, k=args.tabm_k, basis=args.basis,
                        grid_min=-args.grid_range, grid_max=args.grid_range,
                        use_width=False, width=1.0, norm1_type=args.norm1_type,
                        norm2_type=args.norm2_type, activation=args.activation, dropout=args.dropout,
                        num_grids=args.basis_grids or None, ensemble_init=args.ensemble_init,
                        alpha_mode=args.alpha_mode, basis_agg=args.basis_agg, basis_mode=args.basis_mode,
                        feature_basis=args.feature_basis, feature_grids=args.feature_grids)
    elif args.model == "patchtst":
        net = PatchTSTModel(
            input_dim=1 if args.patchtst_exog == "univariate" else n_features,
            seq_len=args.lookback,
            horizon=H,
            patch_len=args.patch_len,
            stride=args.stride,
            d_model=args.d_model,
            n_heads=args.n_heads,
            e_layers=args.e_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            fc_dropout=args.fc_dropout,
            head_dropout=args.head_dropout,
            revin=bool(args.revin),
            affine=bool(args.affine),
            padding_patch=args.padding_patch,
        )
    else: raise ValueError(args.model)
    return BikeModel(net, args.model, n_features, H, k=args.tabm_k, patchtst_exog=args.patchtst_exog)


def ensemble_loss_and_pred(model, pred, y, criterion):
    """TabM-style training: each of the k members is fit to the target independently
    (loss averaged over members); the ensemble prediction is the member mean."""
    if model.is_ensemble:
        loss = criterion(pred, y.unsqueeze(1).expand_as(pred))
        return loss, pred.mean(dim=1)
    return criterion(pred, y), pred


def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, per_batch, y_std):
    model.train(); loss_sum = abs_sum = sq_sum = n_total = 0
    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            pred = model(x)
            loss, pred_metric = ensemble_loss_and_pred(model, pred, y, criterion)

        old_scale = scaler.get_scale(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        if per_batch and scaler.get_scale() >= old_scale: scheduler.step()
        err = (pred_metric.detach().float() - y) * y_std
        n = y.numel(); loss_sum += loss.item() * n; abs_sum += err.abs().sum().item(); sq_sum += err.square().sum().item(); n_total += n
    return loss_sum / n_total, abs_sum / n_total, (sq_sum / n_total) ** 0.5


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, np.float64).ravel(), np.asarray(y_pred, np.float64).ravel()
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - np.sum((y_true - y_pred) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    return rmse, mae, r2


@torch.no_grad()
def evaluate(model, loader, criterion, y_mean, y_std):
    model.eval(); loss_sum = 0.0; n_total = 0; ys, preds = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # Evaluate in FP32 to avoid FP16 quantization of count predictions.
        pred = model(x.float()).float()
        loss, pred = ensemble_loss_and_pred(model, pred, y.float(), criterion)
        y_real = y.float() * y_std + y_mean
        pred_real = (pred * y_std + y_mean).clamp_min(0.0)
        loss_sum += loss.item() * y.numel(); n_total += y.numel()
        ys.append(y_real.cpu().numpy()); preds.append(pred_real.cpu().numpy())
    y_true, y_pred = np.concatenate(ys), np.concatenate(preds)  # [N, H]
    rmse, mae, r2 = metrics(y_true, y_pred)
    return {"loss_scaled": loss_sum / n_total, "mae": mae, "rmse": rmse, "r2": r2,
            "y_true": y_true, "y_pred": y_pred}


def patchtst_input_note(args):
    if args.model != "patchtst":
        return "n/a"
    if args.patchtst_exog == "univariate":
        return "univariate (cnt channel only; exogenous features not used)"
    return (f"mix (all {len(args.feature_names)} channels encoded channel-independently, "
            f"then a learned linear channel-mixing layer -> cnt)")


def train_model(model, train_loader, val_loader, test_loader, args, y_mean, y_std, test_idx, raw_y, timestamps):
    H = args.horizon
    print(f"\nModel: {args.model}\nSeed: {args.seed}\nHorizon: {H}h")
    if args.model == "patchtst":
        print(f"PatchTST input: {patchtst_input_note(args)}")
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
    save_dir = Path("output") / "bike" / (f"h{H}_cal" if args.calendar else f"h{H}") / args.model
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
    train_time = val_time = 0.0  # optimisation time vs. per-epoch validation time

    for epoch in range(args.epochs):
        if device.type == "cuda": torch.cuda.synchronize()
        t_start = time.perf_counter()
        train_loss_scaled, train_mae, train_rmse = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler, per_batch, y_std)
        if not per_batch: scheduler.step()
        if device.type == "cuda": torch.cuda.synchronize()
        t_train = time.perf_counter()
        val = evaluate(model, val_loader, criterion, y_mean, y_std)
        if device.type == "cuda": torch.cuda.synchronize()
        t_val = time.perf_counter()

        epoch_train_time, epoch_val_time = t_train - t_start, t_val - t_train
        train_time += epoch_train_time; val_time += epoch_val_time
        print(f"Epoch {epoch+1:02d}/{args.epochs} | Train RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f} | "
              f"Val RMSE: {val['rmse']:.2f}, MAE: {val['mae']:.2f}, R2: {val['r2']:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6g} | "
              f"Time: {epoch_train_time:.1f}s train + {epoch_val_time:.1f}s val")
        if math.isfinite(val["rmse"]) and val["rmse"] < best_val:
            best_val, best_epoch = val["rmse"], epoch + 1
            torch.save(model.state_dict(), model_path)
        with history_path.open("a") as f:
            f.write(json.dumps({"epoch": epoch + 1, "train_loss_scaled": train_loss_scaled,
                "train_mae": train_mae, "train_rmse": train_rmse,
                "val": {k: v for k, v in val.items() if k not in {"y_true", "y_pred"}},
                "lr": optimizer.param_groups[0]["lr"],
                "train_time": epoch_train_time, "val_time": epoch_val_time,
                "best_epoch": best_epoch}) + "\n")

    params = count_params(model)
    if best_epoch == 0 or not model_path.exists():
        raise RuntimeError("No finite validation RMSE was obtained; checkpoint was not saved. Try a lower --lr or inspect the data/model.")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    test = evaluate(model, test_loader, criterion, y_mean, y_std)

    # Exact raw targets for final test metrics: tgt_idx[i, j] = row of step j+1 for origin i.
    test_idx = np.asarray(test_idx, dtype=int)
    N, steps = len(test_idx), np.arange(H)
    tgt_idx = test_idx[:, None] + steps[None, :]
    y_true = raw_y[tgt_idx].astype(np.float64)                 # [N, H]
    y_pred = np.asarray(test["y_pred"], dtype=np.float64)      # [N, H]
    test["rmse"], test["mae"], test["r2"] = metrics(y_true, y_pred)
    step_rmse = [metrics(y_true[:, j], y_pred[:, j])[0] for j in range(H)]

    # Baselines for the full H-hour forecast, using only values observed before t0.
    ts_to_y = pd.Series(raw_y.astype(np.float64), index=pd.DatetimeIndex(pd.to_datetime(timestamps)))
    tgt_ts = pd.DatetimeIndex(pd.to_datetime(timestamps[tgt_idx.ravel()]))
    baselines = {"persistence": np.repeat(raw_y[test_idx - 1].astype(np.float64)[:, None], H, axis=1)}
    for lag in (24, 168):
        if lag < H:  # would need values after the forecast origin
            continue
        prev_ts = tgt_ts - pd.Timedelta(hours=lag)
        mask = prev_ts.isin(ts_to_y.index)
        arr = np.full(N * H, np.nan, dtype=np.float64)
        arr[mask] = ts_to_y.loc[prev_ts[mask]].to_numpy(dtype=np.float64)
        baselines[f"seasonal_naive_{lag}h"] = arr.reshape(N, H)

    baseline_lines = []
    for name, base in baselines.items():
        valid = np.isfinite(base)
        if not valid.any():
            baseline_lines.append(f"{name}: unavailable")
            continue
        b_rmse, b_mae, b_r2 = metrics(y_true[valid], base[valid])
        m_rmse, m_mae, m_r2 = metrics(y_true[valid], y_pred[valid])
        baseline_lines.append(
            f"{name}: n_points={int(valid.sum())}, RMSE={b_rmse:.6f}, MAE={b_mae:.6f}, R2={b_r2:.6f}; "
            f"model_same_subset: RMSE={m_rmse:.6f}, MAE={m_mae:.6f}, R2={m_r2:.6f}")

    # Long-format predictions: one row per (origin, step).
    pred_frame = pd.DataFrame({
        "origin_timestamp": np.repeat(timestamps[test_idx], H),
        "step": np.tile(steps + 1, N),
        "timestamp": timestamps[tgt_idx.ravel()],
        "y_true": y_true.ravel(),
        "y_pred": y_pred.ravel(),
    })
    for name, base in baselines.items():
        pred_frame[f"{name}_pred"] = base.ravel()
    pred_frame.to_csv(predictions_path, index=False)

    step_line = ", ".join(f"{j+1}:{r:.3f}" for j, r in enumerate(step_rmse))
    print(f"\n{'-'*58}\nModel: {args.model}\nHorizon: {H}h\nParameters: {params:,}\n"
          f"Best Val RMSE: {best_val:.3f} (epoch {best_epoch})\n"
          f"Test RMSE: {test['rmse']:.3f}\nTest MAE: {test['mae']:.3f}\n"
          f"Test R2: {test['r2']:.4f}\n"
          f"Train time: {train_time:.1f}s (+ {val_time:.1f}s validation)\n{'-'*58}")
    print("\nBaselines (see summary file):\n" + "\n".join(baseline_lines))
    summary_path.write_text(
        f"Dataset: UCI Bike Sharing (hour.csv)\nModel: {args.model}\nNote: {args.note}\n"
        f"Lookback: {args.lookback}\nHorizon: {H} (cnt for the {H} hours after the last input hour)\n"
        f"Input features ({len(args.feature_names)}): {args.feature_names}\n"
        f"Calendar features: {'lookback hours only (' + ', '.join(CALENDAR_FEATURES) + ')' if args.calendar else 'none'}\n"
        f"PatchTST input: {patchtst_input_note(args)}\n"
        f"Epochs: {args.epochs}\nBatch size: {args.batch_size}\n"
        f"Scheduler: {args.scheduler}\nLR: {args.lr}\nWeight decay: {args.weight_decay}\n"
        f"Target scaling: z-score using train mean={y_mean:.6f}, std={y_std:.6f}\n"
        f"Prediction post-processing: clamp predictions to non-negative values before evaluation\n"
        f"Metrics: computed over all (origin, step) points\n"
        f"BasisTabM k: {args.tabm_k} | ensemble_init={args.ensemble_init}, basis_mode={args.basis_mode}, "
        f"alpha_mode={args.alpha_mode}, basis_agg={args.basis_agg}, feature_basis={args.feature_basis} "
        f"(grids={args.feature_grids}, range=+-{args.grid_range}, basis={args.basis}), "
        f"norm1={args.norm1_type or 'none'}, norm2={args.norm2_type or 'none'}\nEfficientKAN grid_size: {args.grid_size}\n"
        f"Parameters: {params}\n"
        f"Train time: {train_time:.2f}s (optimisation only; excludes per-epoch validation and final test evaluation)\n"
        f"Validation time (all epochs): {val_time:.2f}s\n"
        f"Predictions CSV: {predictions_path}\nBest epoch: {best_epoch}\n"
        f"Best Val RMSE: {best_val:.6f}\nTest RMSE: {test['rmse']:.6f}\n"
        f"Test MAE: {test['mae']:.6f}\nTest R2: {test['r2']:.6f}\n"
        f"Test RMSE per step: {step_line}\n\nBaselines (raw-count scale):\n" + "\n".join(baseline_lines) + "\n")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="basis_tabm",
        choices=["mlp", "efficient_kan", "basis_tabm", "tabm", "bsrbf_kan", "patchtst"])
    p.add_argument("--data_path", default="data/hour.csv")
    p.add_argument("--note", default="run_0", help="Unique output tag; must not be empty")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting outputs with the same note")
    p.add_argument("--lookback", type=int, default=24)
    p.add_argument("--horizon", type=int, default=1,
                   help="Forecast all of the next H hours; 1 = next hour only")
    p.add_argument("--calendar", action="store_true",
                   help="Add calendar features (hour/weekday/month sin-cos, holiday, workingday) of each lookback hour")
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
    p.add_argument("--norm1_type", default="")
    p.add_argument("--norm2_type", default="")
    
    # BasisTabM 
    p.add_argument("--basis", default="rbf", choices=["rbf", "sech", "laplace", "cauchy", "mexican_hat", "gaussian_d2"])
    p.add_argument("--grid_range", type=float, default=2.0, help="Basis centres span [-r, r]")
    p.add_argument("--ensemble_init", default="random_signs", choices=["random_signs", "ones"],
                   help="'ones' = old behaviour (all members identical); 'random_signs' = TabM-style")
    p.add_argument("--basis_mode", default="none", choices=["none", "mix"],
                   help="'mix' = old alpha-mixed basis branch in every layer; 'none' = no per-layer basis branch")
    p.add_argument("--alpha_mode", default="member", choices=["member", "shared"])
    p.add_argument("--basis_agg", default="sum", choices=["sum", "proj"])
    p.add_argument("--basis_grids", type=int, default=0, help="Centres of the per-layer basis; 0 = tabm_k")
    p.add_argument("--feature_basis", action=argparse.BooleanOptionalAction, default=True,
                   help="Add a learnable per-feature basis transform u = x + sum_g c[i,g] phi_g(x_i) at the input")
    p.add_argument("--feature_grids", type=int, default=8, help="Centres of the input feature basis")
    p.add_argument("--norm_mode", default="all", choices=["none", "first", "except_first", "all"])
    p.add_argument("--num_grids", type=int, default=8, help="Basis grid count for KAN/basis functions")
    p.add_argument("--tabm_k", type=int, default=8, help="ensemble size = number of basis grid points")
    p.add_argument("--grid_size", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dropout", type=float, default=0.0)

    # PatchTST
    p.add_argument("--patchtst_exog", default="univariate", choices=["univariate", "mix"],
                   help="univariate: feed only cnt (standard CI PatchTST); "
                        "mix: feed all channels + learned linear channel mixing to cnt")
    p.add_argument("--patch_len", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--e_layers", type=int, default=2)
    p.add_argument("--d_ff", type=int, default=64)
    p.add_argument("--fc_dropout", type=float, default=0.0)
    p.add_argument("--head_dropout", type=float, default=0.0)
    p.add_argument("--revin", type=int, default=1)
    p.add_argument("--affine", type=int, default=0)
    p.add_argument("--padding_patch", type=str, default="end")

    return p.parse_args()


def main():
    global device
    args = get_args()
    if not args.note.strip():
        raise ValueError("--note must be non-empty to prevent accidental output overwrites.")
    if args.lookback < 1:
        raise ValueError("--lookback must be >= 1.")
    if args.horizon < 1:
        raise ValueError("--horizon must be >= 1.")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch_size must be >= 1.")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be >= 0.")
    if args.num_grids < 1 or args.tabm_k < 1 or args.grid_size < 1:
        raise ValueError("--num_grids, --tabm_k, and --grid_size must be >= 1.")
    if args.model == "patchtst" and args.calendar and args.patchtst_exog == "univariate":
        raise ValueError("Univariate PatchTST ignores calendar features, so a --calendar run would be "
                         "identical to the no-calendar one. Use --patchtst_exog mix, or drop --calendar.")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    set_seed(args.seed, device)
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    if device.type == "cuda": print(f"GPU: {torch.cuda.get_device_name(0)}")
    train_loader, val_loader, test_loader, n_features, input_dim, y_mean, y_std, test_idx, raw_y, timestamps = get_dataloaders(args)
    model = build_model(args, n_features, input_dim)
    train_model(model, train_loader, val_loader, test_loader, args, y_mean, y_std, test_idx, raw_y, timestamps)


if __name__ == "__main__":
    main()


