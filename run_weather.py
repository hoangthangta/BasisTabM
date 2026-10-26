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

from models.mlp import MLP
from models.efficient_kan import EfficientKAN
from models.bsrbf_kan import BSRBF_KAN
from models.basis_tabm import BasisTabM
from models.patchtst_model import PatchTSTModel

from schedulers import get_scheduler
from utils import set_seed, count_params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SENTINEL_THRESHOLD = -9990.0  # MPI Weather uses -9999 for invalid sensor readings
ENSEMBLE_MODELS = {"tabm", "basis_tabm"}

# Forecasting protocols:
#   Default         : predict the single value at t + horizon (out_len=1).
#   --multi_output  : predict t+1 ... t+horizon (out_len=horizon), averaging errors
#                     across all steps, as in Autoformer/TimesNet/PatchTST tables.
#   Outputs         : (B, out_len), or (B, k, out_len) for TabM/BasisTabM ensembles.
#
# Calendar features (--calendar):
#   none : no calendar features (default); all models use only univariate history.
#   tod  : target-time time-of-day sin/cos (ablation; PatchTST ignores these features).
#   all  : adds weekday/month sin/cos (ablation only).


class WeatherModel(nn.Module):
    def __init__(self, model, model_type, n_calendar, out_len):
        super().__init__()
        self.model, self.model_type, self.n_calendar, self.out_len = model, model_type, n_calendar, out_len
        self.is_ensemble = model_type in ENSEMBLE_MODELS

    def _as_matrix(self, out, batch):
        if out.numel() != batch * self.out_len:
            raise ValueError(f"Expected {self.out_len} forecast value(s) per sample; got output shape {tuple(out.shape)}")
        return out.reshape(batch, self.out_len)

    def forward(self, x, calendar):
        batch, H = x.shape[0], self.out_len
        if self.model_type == "patchtst":
            return self._as_matrix(self.model(x), batch)
        
        inp = torch.cat([x.flatten(1), calendar], dim=1)
        out = self.model(inp)
        if self.is_ensemble:
            if out.ndim == 3 and out.shape[-1] == H:
                return out                                   # (B, k, H)
            if out.ndim == 2 and H == 1:
                return out.unsqueeze(-1)                     # (B, k) -> (B, k, 1)
            if out.ndim == 2 and out.shape[1] == H:
                return out.unsqueeze(1)                      # single member -> (B, 1, H)
            if out.ndim == 1 and H == 1:
                return out.view(batch, 1, 1)
            raise ValueError(f"Unexpected ensemble output shape {tuple(out.shape)} for out_len={H}")
        return self._as_matrix(out, batch)


@torch.no_grad()
def _probe(net, in_channels, seq_len, hook_linears=False):
    """Run a dummy batch through `net` in evaluation mode.
    Returns the output and the names of `nn.Linear` modules in execution order."""
    order, hooks = [], []
    if hook_linears:
        for mod_name, mod in net.named_modules():
            if isinstance(mod, nn.Linear):
                hooks.append(mod.register_forward_hook(lambda m, i, o, n=mod_name: order.append(n)))
    was_training = net.training
    net.eval()                       # avoid BatchNorm statistics updates during the probe
    try:
        out = net(torch.zeros(2, in_channels, seq_len))
    finally:
        for h in hooks:
            h.remove()
        net.train(was_training)
    return out, order


def adapt_output_dim(net, out_dim, name, in_channels, seq_len):
    """Widen the last-executed `nn.Linear` head to `out_dim` and verify with a second dummy pass."""
    if out_dim == 1:
        return net
    out, order = _probe(net, in_channels, seq_len, hook_linears=True)
    if not order:
        raise ValueError(f"{name}: no nn.Linear is executed in forward(); "
                         "add an output-size argument to this model for --multi_output.")
    head_name = order[-1]
    head = net.get_submodule(head_name)
    if head.out_features != 1 or out.numel() != out.shape[0]:
        raise ValueError(f"{name}: last executed nn.Linear '{head_name}' has out_features={head.out_features} "
                         f"and the model output shape is {tuple(out.shape)}; expected a single-output head. "
                         "Add an output-size argument to this model for --multi_output.")
    parent = net.get_submodule(head_name.rsplit(".", 1)[0]) if "." in head_name else net
    setattr(parent, head_name.rsplit(".", 1)[-1],
            nn.Linear(head.in_features, out_dim, bias=head.bias is not None))
    new_out, _ = _probe(net, in_channels, seq_len)
    if new_out.numel() != new_out.shape[0] * out_dim:
        raise RuntimeError(f"{name}: after widening '{head_name}', output shape is {tuple(new_out.shape)}; "
                           f"expected {out_dim} values per sample.")
    print(f"{name}: output head '{head_name}' (last executed nn.Linear) widened from 1 to {out_dim} outputs.")
    return net


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
    H_out = args.out_len
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

    # Chronological 70/10/20 split boundaries, fixed on the ORIGINAL rows (before sentinel removal),
    # so removing invalid rows never shifts the split periods.
    n0 = len(df)
    train_end_ts = df["timestamp"].iloc[int(n0 * 0.70)]
    val_end_ts = df["timestamp"].iloc[int(n0 * 0.80)]

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
    calendar = {}
    if args.calendar in {"tod", "all"}:
        minute_of_day = ts.dt.hour.to_numpy() * 60 + ts.dt.minute.to_numpy()
        calendar.update({
            "tod_sin": np.sin(2 * np.pi * minute_of_day / 1440),
            "tod_cos": np.cos(2 * np.pi * minute_of_day / 1440),
        })
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
    n = len(df)
    if calendar_cols:
        raw_cal = np.column_stack([calendar[c] for c in calendar_cols]).astype(np.float32)
    else:
        raw_cal = np.zeros((n, 0), dtype=np.float32)        # --calendar none: zero calendar columns
    # Row indices of the split boundaries in the cleaned data (first row at/after each boundary timestamp).
    train_end = int(np.searchsorted(timestamps, np.datetime64(train_end_ts)))
    val_end = int(np.searchsorted(timestamps, np.datetime64(val_end_ts)))
    if train_end <= args.lookback + args.horizon or val_end <= train_end or n <= val_end:
        raise ValueError("Dataset is too small for the selected lookback/horizon and chronological split.")

    # Fit scalers on the training period only.
    x_mean, x_std = raw_x[:train_end].mean(0), raw_x[:train_end].std(0)
    x_std[x_std < 1e-6] = 1.0
    if calendar_cols:
        cal_mean, cal_std = raw_cal[:train_end].mean(0), raw_cal[:train_end].std(0)
        cal_std[cal_std < 1e-6] = 1.0
        cal_scaled = ((raw_cal - cal_mean) / cal_std).astype(np.float32)
    else:
        cal_scaled = raw_cal
    y_mean, y_std = float(raw_y[:train_end].mean()), float(raw_y[:train_end].std())
    if not np.isfinite(y_std) or y_std < 1e-6:
        y_std = 1.0
    x_scaled = ((raw_x - x_mean) / x_std).astype(np.float32)
    y_scaled = ((raw_y - y_mean) / y_std).astype(np.float32)

    # Vectorized window construction.
    # Every window spans rows [input_start, input_start + lookback + horizon - 1]:
    #   input  = [input_start, input_start + lookback - 1]
    #   target = last row only (default) or the last `horizon` rows (--multi_output).
    # A window is valid only if every step inside it is exactly 10 minutes.
    target_all = np.arange(args.lookback + args.horizon - 1, n)
    start_all = target_all - args.horizon + 1 - args.lookback
    n_candidates = len(target_all)
    step_ok = np.diff(timestamps) == np.timedelta64(10, "m")
    bad_cum = np.concatenate([[0], np.cumsum(~step_ok)])
    valid = (bad_cum[target_all] - bad_cum[start_all]) == 0
    target_idx = target_all[valid].astype(np.int64)     # last target row
    first_idx = target_idx - (H_out - 1)                  # first target row (== target_idx when out_len = 1)
    input_start = start_all[valid]
    if len(target_idx) == 0:
        raise ValueError("No valid Weather windows found. Check the timestamp frequency and lookback/horizon.")

    windows = np.lib.stride_tricks.sliding_window_view(x_scaled[:, 0], args.lookback)
    X = np.ascontiguousarray(windows[input_start][..., None], dtype=np.float32)   # (N, lookback, 1)
    C = np.ascontiguousarray(cal_scaled[first_idx], dtype=np.float32)            # (N, n_calendar); n_calendar may be 0
    Y = np.ascontiguousarray(
        np.lib.stride_tricks.sliding_window_view(y_scaled, H_out)[first_idx], dtype=np.float32)  # (N, out_len)

    # A window belongs to a split only if all of its target steps lie inside that split.
    tr = target_idx < train_end
    va = (first_idx >= train_end) & (target_idx < val_end)
    te = first_idx >= val_end
    if not tr.any() or not va.any() or not te.any():
        raise ValueError(f"Empty split after filtering windows: train={int(tr.sum())}, val={int(va.sum())}, test={int(te.sum())}.")
    n_straddle = len(target_idx) - int(tr.sum() + va.sum() + te.sum())

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

    mode = (f"multi-output: predict t+1 ... t+{args.horizon} ({H_out} values per window)"
            if args.multi_output else f"single-point: predict t+{args.horizon} only")
    print(f"Time range: {ts.iloc[0]} -> {ts.iloc[-1]} ({len(df):,} rows)")
    print(f"Weather | Target: {args.target} | Train: {tr.sum():,} | Val: {va.sum():,} | Test: {te.sum():,} | "
          f"Lookback: {args.lookback} x 10 min | Horizon: {args.horizon} x 10 min")
    print(f"Forecast mode: {mode}")
    print(f"Input: one historical variable ({args.target}); calendar at first target step: "
          f"{calendar_cols if calendar_cols else 'none'}")
    if args.model == "patchtst" and calendar_cols:
        print("Note: PatchTST does not use calendar inputs; use --calendar none for a like-for-like comparison.")
    print(f"Chronological split (fixed on original rows): train < {train_end_ts}, "
          f"val [{train_end_ts}, {val_end_ts}), test >= {val_end_ts}")
    print(f"Skipped {n_candidates - len(target_idx):,} candidate windows crossing missing/non-consecutive timestamps.")
    if n_straddle:
        print(f"Skipped {n_straddle:,} windows whose target steps straddle a split boundary.")
    print("Note: validation/test windows may use earlier observations as historical input; target timestamps do not overlap.")
    return (loader(tr, True), loader(va), loader(te), 1,
            args.lookback + len(calendar_cols), len(calendar_cols),
            y_mean, y_std, first_idx[te], raw_y, timestamps, calendar_cols)


def build_model(args, n_features, input_dim, n_calendar):
    H_out = args.out_len
    hidden = [int(v.strip()) for v in args.hidden_layers.split(",") if v.strip()]
    layers = [input_dim] + hidden + [H_out]
    if args.model == "mlp":
        net = MLP(layers, base_activation=args.activation, norm_type=args.norm_type)
    elif args.model == "efficient_kan":
        net = EfficientKAN(layers_hidden=layers, grid_size=args.grid_size, spline_order=3)
    elif args.model == "bsrbf_kan":
        net = BSRBF_KAN(layers_hidden=layers, grid_size=args.grid_size, spline_order=3, norm_type=args.norm_type)
    elif args.model == "tabm":
        if len(set(hidden)) != 1:
            raise ValueError("TabM needs equal widths in --hidden_layers.")
        # dropout passed explicitly so TabM does not fall back to the library default.
        net = tabm.TabM.make(n_num_features=input_dim, cat_cardinalities=[], d_out=H_out,
                             d_block=hidden[0], n_blocks=len(hidden), k=args.tabm_k,
                             dropout=args.dropout, num_embeddings=None)
    elif args.model == "patchtst":
        net = PatchTSTModel(
            input_dim=n_features,
            seq_len=args.lookback,
            horizon=H_out,          # 1 in single-point mode (direct t+h forecast), horizon in multi-output mode
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
    elif args.model == "basis_tabm":
        net = BasisTabM(net_layers=layers, k=args.tabm_k, basis=args.basis,
                        grid_min=-args.grid_range, grid_max=args.grid_range,
                        use_width=False, width=1.0, norm1_type=args.norm1_type,
                        norm2_type=args.norm2_type, activation=args.activation, dropout=args.dropout,
                        num_grids=args.basis_grids or None, ensemble_init=args.ensemble_init,
                        alpha_mode=args.alpha_mode, basis_agg=args.basis_agg, basis_mode=args.basis_mode,
                        feature_basis=args.feature_basis, feature_grids=args.feature_grids)
    else: raise ValueError(args.model)
    return WeatherModel(net, args.model, n_calendar, H_out)


def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, per_batch, y_std, use_amp):
    model.train(); loss_sum = abs_sum = sq_sum = n_total = 0
    for x, cal, y in tqdm(loader, desc="Training", leave=False):
        x, cal, y = x.to(device, non_blocking=True), cal.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred = model(x, cal)                                   # (B, H) or (B, k, H)
            if model.is_ensemble:
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
        pred = model(x.float(), cal.float()).float()
        if model.is_ensemble:
            pred = pred.mean(dim=1)
        # Note: for ensembles this loss is on the ensemble mean, unlike the per-member training loss.
        loss = criterion(pred, y.float())
        y_real = y.float() * y_std + y_mean
        pred_real = pred * y_std + y_mean
        err = pred_real - y_real; n = y.numel()
        loss_sum += loss.item() * n; abs_sum += err.abs().sum().item(); sq_sum += err.square().sum().item(); n_total += n
        ys.append(y_real.cpu().numpy()); preds.append(pred_real.cpu().numpy())
    y_true, y_pred = np.concatenate(ys), np.concatenate(preds)          # (N, out_len)
    ss_res = np.square(y_true - y_pred).sum(); ss_tot = np.square(y_true - y_true.mean()).sum()
    return {"loss_scaled": loss_sum / n_total, "mae": abs_sum / n_total, "rmse": (sq_sum / n_total) ** 0.5,
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "y_true": y_true, "y_pred": y_pred}


def train_model(model, train_loader, val_loader, test_loader, args, y_mean, y_std, test_first_idx, raw_y, timestamps, calendar_cols):
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
    if args.multi_output:
        predictions_path = save_dir / f"predictions__{run_id}.npz"
        per_step_path = save_dir / f"per_step__{run_id}.csv"
        outputs = [model_path, config_path, summary_path, predictions_path, per_step_path]
    else:
        predictions_path = save_dir / f"predictions__{run_id}.csv"
        per_step_path = None
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
        "forecast_mode": "multi_output" if args.multi_output else "single_point",
        "args": vars(args), "split": {"train_ratio": 0.70, "validation_ratio": 0.10, "test_ratio": 0.20,
                                      "boundaries": "fixed on original rows before sentinel removal"},
        "target_scaling": "train-only z-score", "calendar_features": args.calendar,
        "calendar_columns": calendar_cols
    }, indent=2, default=str))
    best_val, best_epoch = float("inf"), 0
    train_time = val_time = 0.0  # optimisation time vs. per-epoch validation time

    for epoch in range(args.epochs):
        if device.type == "cuda": torch.cuda.synchronize()
        t_start = time.perf_counter()
        train_loss_scaled, train_mae, train_rmse = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler, per_batch, y_std, use_amp)
        if not per_batch: scheduler.step()
        if device.type == "cuda": torch.cuda.synchronize()
        t_train = time.perf_counter()
        val = evaluate(model, val_loader, criterion, y_mean, y_std)
        if device.type == "cuda": torch.cuda.synchronize()
        t_val = time.perf_counter()
        epoch_train_time, epoch_val_time = t_train - t_start, t_val - t_train
        train_time += epoch_train_time; val_time += epoch_val_time
        print(
            f"Epoch {epoch+1:02d}/{args.epochs:<2d} | "
            f"Train RMSE: {train_rmse:7.4f}, MAE: {train_mae:7.4f} | "
            f"Val RMSE: {val['rmse']:7.4f}, MAE: {val['mae']:7.4f}, R2: {val['r2']:7.4f}, "
            f"MSE(norm): {val['loss_scaled']:7.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:10.3e} | "
            f"Time: {epoch_train_time:5.1f}s train + {epoch_val_time:4.1f}s val"
        )
        if math.isfinite(val["rmse"]) and val["rmse"] < best_val:
            best_val, best_epoch = val["rmse"], epoch + 1
            torch.save(model.state_dict(), model_path)
    params = count_params(model)
    if best_epoch == 0 or not model_path.exists():
        raise RuntimeError("No finite validation RMSE was obtained; checkpoint was not saved. Try a lower --lr or inspect the data/model.")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    test = evaluate(model, test_loader, criterion, y_mean, y_std)

    # ---- Align test predictions with original targets: shape (N, H) ----
    H = args.out_len
    first = np.asarray(test_first_idx, dtype=np.int64)
    N = len(first)
    if test["y_pred"].shape != (N, H):
        raise RuntimeError(f"Test size mismatch: expected {(N, H)}, got {test['y_pred'].shape}.")
    target_pos = first[:, None] + np.arange(H)[None, :]
    # Use the original (un-scaled) targets as ground truth; the reconstructed ones carry float32 round-off.
    y_true = raw_y[target_pos].astype(np.float64)
    if not np.allclose(test["y_true"], y_true, rtol=1e-4, atol=1e-4 * y_std):
        raise RuntimeError("Test targets do not line up with test indices; check the data pipeline.")
    y_pred = test["y_pred"].astype(np.float64)
    target_ts = timestamps[target_pos]                                   # (N, H) datetime64

    # Steps ahead of the last observed input for each output column.
    offsets = np.arange(1, H + 1) if args.multi_output else np.array([args.horizon])
    origin_ts = pd.DatetimeIndex(target_ts[:, 0] - np.timedelta64(10, "m") * int(offsets[0]))
    if not origin_ts.is_monotonic_increasing or origin_ts.has_duplicates:
        raise RuntimeError("Test forecast origins must be sorted and unique.")

    # Naive baselines
    # persistence         : repeat the last observed value.
    # seasonal_naive_24h  : use the same time from the previous day.
    # seasonal_naive_168h : use the same time from the previous week.
    ts_index = pd.DatetimeIndex(pd.to_datetime(timestamps))
    baselines = []
    for name, period in [("persistence", None), ("seasonal_naive_24h", 144), ("seasonal_naive_168h", 1008)]:
        lags = offsets.copy() if period is None else period * np.ceil(offsets / period).astype(np.int64)
        src_ts = target_ts - (lags * 10).astype("timedelta64[m]")[None, :]
        pos = ts_index.get_indexer(pd.DatetimeIndex(src_ts.ravel()))
        pred_base = np.where(pos >= 0, raw_y[np.clip(pos, 0, None)], np.nan).astype(np.float64).reshape(N, H)
        baselines.append((name, lags, pred_base))

    common = np.isfinite(y_pred) & np.all([np.isfinite(p) for _, _, p in baselines], axis=0)
    if not common.any():
        raise RuntimeError("No test target has a prediction from every baseline.")

    def metrics(a, b):
        a = np.asarray(a, dtype=np.float64).ravel(); b = np.asarray(b, dtype=np.float64).ravel()
        err = b - a
        mse, mae = float(np.mean(err ** 2)), float(np.mean(np.abs(err)))
        ss_tot = float(np.sum((a - a.mean()) ** 2))
        r2 = float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan")
        # *_norm: errors on the train-only z-scored scale, comparable with published MSE/MAE tables.
        return {"rmse": mse ** 0.5, "mae": mae, "r2": r2, "mse_norm": mse / y_std ** 2, "mae_norm": mae / y_std}

    def fmt(m):
        return (f"RMSE={m['rmse']:.6f}, MAE={m['mae']:.6f}, R2={m['r2']:.6f}, "
                f"MSE_norm={m['mse_norm']:.6f}, MAE_norm={m['mae_norm']:.6f}")

    m_full = metrics(y_true, y_pred)
    m_common = metrics(y_true[common], y_pred[common])
    unit = "window-steps" if args.multi_output else "targets"
    baseline_lines = [f"Common subset: n={int(common.sum()):,}/{common.size:,} {unit}; all methods use the same targets.",
                      f"model: {fmt(m_common)}"]
    for name, lags, pred_base in baselines:
        bm = metrics(y_true[common], pred_base[common])
        lag_desc = f"lag {int(lags[0])} steps" if len(lags) == 1 else f"lags {int(lags.min())}-{int(lags.max())} steps"
        baseline_lines.append(f"{name}: {fmt(bm)} ({lag_desc}; available for "
                              f"{int(np.isfinite(pred_base).sum()):,} {unit})")

    if args.multi_output:
        np.savez_compressed(
            predictions_path,
            origin_timestamp=origin_ts.to_numpy(), steps_ahead=offsets,
            y_true=y_true.astype(np.float32), y_pred=y_pred.astype(np.float32),
            in_common_subset=common,
            **{f"{name}_pred": p.astype(np.float32) for name, _, p in baselines})
        rows = []
        for j in range(H):
            # Full-test model metrics for this step (all N windows).
            m = metrics(y_true[:, j], y_pred[:, j])
            row = {"step": int(offsets[j]), "minutes_ahead": int(offsets[j]) * 10, "n": N, **m}
            # Model and baselines on the SAME subset (common[:, j]) for a like-for-like comparison.
            mask_j = common[:, j]
            row["n_common"] = int(mask_j.sum())
            if mask_j.any():
                row["model_rmse_common"] = metrics(y_true[mask_j, j], y_pred[mask_j, j])["rmse"]
                row["model_mae_common"] = metrics(y_true[mask_j, j], y_pred[mask_j, j])["mae"]
                for name, _, p in baselines:
                    bm_j = metrics(y_true[mask_j, j], p[mask_j, j])
                    row[f"{name}_rmse_common"] = bm_j["rmse"]
                    row[f"{name}_mae_common"] = bm_j["mae"]
            rows.append(row)
        pd.DataFrame(rows).to_csv(per_step_path, index=False)
    else:
        pred_frame = pd.DataFrame({"timestamp": pd.DatetimeIndex(target_ts[:, 0]),
                                   "y_true": y_true[:, 0], "y_pred": y_pred[:, 0]})
        for name, _, p in baselines:
            pred_frame[f"{name}_pred"] = p[:, 0]
        pred_frame["in_common_subset"] = common[:, 0]
        pred_frame.to_csv(predictions_path, index=False)

    h = args.horizon
    mode_desc = (f"multi-output, t+1..t+{h} (errors averaged over {H} steps)" if args.multi_output
                 else f"single-point, t+{h}")
    cal_desc = calendar_cols if calendar_cols else "none"
    sep = "=" * 92
    print(f"\n{sep}")
    print(f" {args.model.upper()} | Target: {args.target} | Horizon: {h} | Mode: {mode_desc} | Calendar: {args.calendar}")
    print(sep)
    print(f" {'Parameters':<20}: {params:>15,}")
    print(f" {'Best Val RMSE':<20}: {best_val:>15.6f}  (Epoch {best_epoch})")
    print(f" {'Train time':<20}: {train_time:>14.1f} s  (+ {val_time:.1f} s validation)")
    print("-" * 92)
    print(f" {'Test Set':<10} {'N':>12} {'RMSE':>11} {'MAE':>11} {'R2':>9} {'MSE_norm':>11} {'MAE_norm':>11}")
    print("-" * 92)
    for label, n_, m in [("Full", common.size, m_full), ("Common", int(common.sum()), m_common)]:
        print(f" {label:<10} {n_:>12,} {m['rmse']:>11.6f} {m['mae']:>11.6f} {m['r2']:>9.4f} "
              f"{m['mse_norm']:>11.6f} {m['mae_norm']:>11.6f}")
    print(sep)
    print("\nBaselines:")
    for line in baseline_lines:
        print(f"  {line}")
    if args.multi_output:
        print(f"\nPer-step metrics saved to {per_step_path}")

    summary_path.write_text(
        f"Dataset: Weather (Max Planck Institute; 10-minute frequency)\nTarget: {args.target}\n"
        f"Model: {args.model}\nNote: {args.note}\nSeed: {args.seed}\n"
        + (f"BasisTabM: k={args.tabm_k}, ensemble_init={args.ensemble_init}, basis_mode={args.basis_mode}, "
           f"feature_basis={args.feature_basis} (grids={args.feature_grids}, range=+-{args.grid_range}, "
           f"basis={args.basis}), norm1={args.norm1_type or 'none'}, norm2={args.norm2_type or 'none'}\n"
           if args.model == "basis_tabm" else "") +
        f"Forecast mode: {mode_desc}\n"
        f"Lookback: {args.lookback} steps ({args.lookback * 10} minutes)\n"
        f"Forecast horizon: {h} step(s) ({h * 10} minutes ahead)\n"
        f"Chronological split: 70%/10%/20% (boundaries fixed on original rows before sentinel removal); "
        f"a window belongs to a split only if all its target steps do\n"
        f"Historical input: univariate target series only; calendar covariates at first target step: {cal_desc}"
        + (" (not used by PatchTST)" if args.model == "patchtst" and calendar_cols else "") + "\n"
        f"Target scaling: train-only z-score (mean={y_mean:.8f}, std={y_std:.8f})\n"
        f"Epochs: {args.epochs}\nBatch size: {args.batch_size}\nScheduler: {args.scheduler}\n"
        f"LR: {args.lr}\nWeight decay: {args.weight_decay}\nDropout: {args.dropout}\n"
        f"AMP: {'fp16' if use_amp else 'off'}\n"
        f"Parameters: {params}\n"
        f"Train time: {train_time:.2f}s (optimisation only; excludes per-epoch validation and final test evaluation)\n"
        f"Validation time (all epochs): {val_time:.2f}s\n"
        f"Best epoch: {best_epoch}\nBest Val RMSE: {best_val:.8f}\n"
        f"Test full: n={common.size} {unit}, {fmt(m_full)}\n"
        f"Test common subset: n={int(common.sum())} {unit}, {fmt(m_common)}\n"
        f"Predictions: {predictions_path}\n"
        + (f"Per-step metrics: {per_step_path} (compare *_common columns for model vs. baselines)\n"
           if per_step_path else "")
        + "\nBaselines:\n" + "\n".join(baseline_lines) + "\n")


def get_args():
    p = argparse.ArgumentParser(description="Univariate Weather forecasting with chronological train/validation/test splits.")
    p.add_argument("--model", default="mlp",
        choices=["mlp", "efficient_kan", "basis_tabm", "tabm", "bsrbf_kan", "patchtst"])
    p.add_argument("--data_path", default="data/weather.csv")
    p.add_argument("--target", default="OT", help="Single Weather column to forecast (default: OT).")
    p.add_argument("--note", default="run_0", help="Unique output tag; must not be empty")
    p.add_argument("--overwrite", action="store_true", help="Delete existing outputs with the same note/seed before running")
    p.add_argument("--lookback", type=int, default=144, help="Historical steps; 144 steps = 24 hours at 10-minute frequency.")
    p.add_argument("--horizon", type=int, default=1,
                   help="Forecast horizon in 10-minute steps. Default mode: predict only t+horizon. "
                        "With --multi_output: predict all steps t+1..t+horizon.")
    p.add_argument("--multi_output", action="store_true",
                   help="Benchmark protocol (Autoformer/TimesNet/PatchTST): output the whole sequence "
                        "t+1..t+horizon and average errors over all steps.")
    p.add_argument("--calendar", default="none", choices=["none", "tod", "all"],
                   help="none (default): no calendar input, same information for every model incl. PatchTST. "
                        "tod: time-of-day only (ablation). all: also weekday/month "
                        "(month values in the test period are unseen in training for this one-year dataset).")
    p.add_argument("--hidden_layers", default="256", help="")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--scheduler", default="OneCycleLR", choices=["StepLR", "CosineAnnealingLR", "OneCycleLR", "ExponentialLR", "CyclicLR"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--norm_type", default="layer")
    p.add_argument("--activation", default="silu")
    p.add_argument("--norm1_type", default="")
    p.add_argument("--norm2_type", default="")
    p.add_argument("--tabm_k", type=int, default=8, help="TabM ensemble size")
    
    # BasisTabM
    p.add_argument("--basis", default="rbf", choices=["rbf", "sech", "laplace", "cauchy", "mexican_hat", "gaussian_d2"])
    p.add_argument("--grid_range", type=float, default=2.0, help="Basis centres span [-r, r]")
    p.add_argument("--ensemble_init", default="random_signs", choices=["random_signs", "ones"],
                   help="'ones' = old behaviour (all members identical); 'random_signs' = TabM-style")
    p.add_argument("--basis_mode", default="none", choices=["none", "mix"],
                   help="'mix' = alpha-mixed basis branch in every layer; 'none' = no per-layer basis branch")
    p.add_argument("--alpha_mode", default="member", choices=["member", "shared"])
    p.add_argument("--basis_agg", default="sum", choices=["sum", "proj"])
    p.add_argument("--basis_grids", type=int, default=0, help="Centres of the per-layer basis; 0 = tabm_k")
    p.add_argument("--feature_basis", action=argparse.BooleanOptionalAction, default=True,
                   help="Learnable per-feature basis transform u = x + sum_g c[i,g] phi_g(x_i) at the input")
    p.add_argument("--feature_grids", type=int, default=8, help="Centres of the input feature basis")
    p.add_argument("--grid_size", type=int, default=5, help="Grid size for efficient_kan/bsrbf_kan")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--no_amp", action="store_true", help="Disable fp16 mixed precision (useful for KAN stability checks)")

    # PatchTST
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
    if min(args.lookback, args.horizon, args.epochs, args.batch_size) < 1:
        raise ValueError("lookback/horizon/epochs/batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be >= 0.")
    if args.tabm_k < 1 or args.grid_size < 1:
        raise ValueError("--tabm_k and --grid_size must be >= 1.")
    if args.model == "patchtst":
        if min(args.d_model, args.n_heads, args.e_layers, args.d_ff) < 1:
            raise ValueError("d_model, n_heads, e_layers, and d_ff must be positive.")
        if args.d_model % args.n_heads != 0:
            raise ValueError("--d_model must be divisible by --n_heads.")
        if args.patch_len < 1 or args.patch_len > args.lookback:
            raise ValueError("--patch_len must be in [1, --lookback].")
        if args.stride < 1:
            raise ValueError("--stride must be positive.")
    # Number of values each model outputs per window.
    args.out_len = args.horizon if args.multi_output else 1

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    set_seed(args.seed, device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    (train_loader, val_loader, test_loader, n_features, input_dim, n_calendar,
     y_mean, y_std, test_first_idx, raw_y, timestamps, calendar_cols) = get_dataloaders(args)
    model = build_model(args, n_features, input_dim, n_calendar)
    train_model(model, train_loader, val_loader, test_loader, args, y_mean, y_std,
                test_first_idx, raw_y, timestamps, calendar_cols)


if __name__ == "__main__":
    main()


