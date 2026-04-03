"""
gen_fl_core.py
Data loading, balancing, windowing, and training utilities for the
synthetic federated dataset distillation pipeline.

Key differences from fl_core.py:
  - CSV delimiter is semicolon (;)
  - Dataset balancing: shorter dataset length used for both clients
  - Synthetic series → windows helper
  - Teacher gradient computation (frozen model)
  - Discriminator update (local, client-side)
  - FedAvg aggregation (reused logic)
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models import LSTMRegressor

# ── Column names ──────────────────────────────────────────────────────────────
TIMESTAMP_COL = "DATE AND TIME GMT"
TARGET_COL = "%RH"
TIME_FEATURES = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
FEATURE_COLUMNS = [TARGET_COL] + TIME_FEATURES
MODEL_INPUT_DIM = len(FEATURE_COLUMNS)   # 5

GEN_SEQ_OUT = 512          # timesteps produced by one Generator call
LATENT_DIM = 128


# ── Dataset ───────────────────────────────────────────────────────────────────

class SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_csv(csv_path: str) -> pd.DataFrame:
    """Load semicolon-delimited climate CSV, keep timestamp + %RH only."""
    df = pd.read_csv(csv_path, sep=";")
    df.columns = [c.strip() for c in df.columns]

    # Flexible column matching
    ts_candidates = ["DATE AND TIME", "DATE AND TIME GMT", "datetime", "Datetime", "timestamp"]
    rh_candidates = ["%RH", "RH", "rh", "RelativeHumidity"]

    ts_col = next((c for c in ts_candidates if c in df.columns), None)
    rh_col = next((c for c in rh_candidates if c in df.columns), None)
    if ts_col is None:
        raise ValueError(f"No timestamp column found. Columns: {df.columns.tolist()}")
    if rh_col is None:
        raise ValueError(f"No %%RH column found. Columns: {df.columns.tolist()}")

    df = df[[ts_col, rh_col]].copy()
    df.columns = [TIMESTAMP_COL, TARGET_COL]
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], dayfirst=True, errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna().sort_values(TIMESTAMP_COL).reset_index(drop=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    hour = ts.dt.hour.astype(float)
    month = ts.dt.month.astype(float)
    df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    df["month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
    return df


# ── Balancing ─────────────────────────────────────────────────────────────────

def balance_to_shortest(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Trim all DataFrames to the length of the shortest one (keep first N rows)."""
    min_len = min(len(df) for df in dfs)
    return [df.iloc[:min_len].reset_index(drop=True) for df in dfs]


# ── Splits + windowing ────────────────────────────────────────────────────────

def split_df(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio_within_trainval: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sequential train / val / holdout split."""
    n = len(df)
    trainval_end = max(min(int(n * train_ratio), n - 20), 60)
    trainval = df.iloc[:trainval_end].reset_index(drop=True)
    holdout = df.iloc[trainval_end:].reset_index(drop=True)

    train_end = max(min(int(len(trainval) * (1.0 - val_ratio_within_trainval)),
                        len(trainval) - 10), 40)
    train = trainval.iloc[:train_end].reset_index(drop=True)
    val = trainval.iloc[train_end:].reset_index(drop=True)
    return train, val, holdout


def _normalize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (values - mean) / max(std, 1e-6)


def make_windows(
    df: pd.DataFrame,
    seq_len: int,
    target_mean: float,
    target_std: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slide windows over df. Returns x (N, seq_len, 5), y (N,) normalized."""
    feats = df[FEATURE_COLUMNS].copy()
    feats[TARGET_COL] = _normalize(feats[TARGET_COL].values.astype(np.float32),
                                   target_mean, target_std)
    feat_vals = feats.values.astype(np.float32)
    target_vals = _normalize(df[TARGET_COL].values.astype(np.float32),
                             target_mean, target_std)
    xs, ys = [], []
    for i in range(seq_len, len(df)):
        xs.append(feat_vals[i - seq_len:i])
        ys.append(float(target_vals[i]))
    if not xs:
        return (np.empty((0, seq_len, MODEL_INPUT_DIM), dtype=np.float32),
                np.empty((0,), dtype=np.float32))
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.float32)


def make_windows_with_context(
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
    seq_len: int,
    target_mean: float,
    target_std: float,
) -> Tuple[np.ndarray, np.ndarray]:
    ctx = pd.concat([history_df.tail(seq_len), target_df], ignore_index=True)
    x_all, y_all = make_windows(ctx, seq_len, target_mean, target_std)
    needed = len(target_df)
    if needed <= 0 or len(x_all) == 0:
        return (np.empty((0, seq_len, MODEL_INPUT_DIM), dtype=np.float32),
                np.empty((0,), dtype=np.float32))
    return x_all[-needed:], y_all[-needed:]


def build_client_dataloaders(
    csv_path: str,
    client_id: int,
    balanced_len: int,
    seq_len: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Load CSV, trim to balanced_len, split, window, return loaders + meta.
    Returns: fl_loader, val_loader, holdout_loader, meta
    """
    df_raw = load_csv(csv_path)
    df_raw = df_raw.iloc[:balanced_len].reset_index(drop=True)
    df = add_time_features(df_raw)

    train_df, val_df, holdout_df = split_df(df)
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    target_mean = float(train_df[TARGET_COL].mean())
    target_std = float(max(train_df[TARGET_COL].std(ddof=0), 1e-6))

    x_train, y_train = make_windows(train_df, seq_len, target_mean, target_std)
    x_val, y_val = make_windows_with_context(train_df, val_df, seq_len,
                                              target_mean, target_std)
    x_hold, y_hold = make_windows_with_context(trainval_df, holdout_df, seq_len,
                                                target_mean, target_std)

    fl_loader = DataLoader(SeqDataset(x_train, y_train),
                           batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SeqDataset(x_val, y_val),
                            batch_size=batch_size, shuffle=False)
    holdout_loader = DataLoader(SeqDataset(x_hold, y_hold),
                                batch_size=batch_size, shuffle=False)

    meta = {
        "client_id": client_id,
        "balanced_len": balanced_len,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "holdout_rows": len(holdout_df),
        "train_samples": len(x_train),
        "val_samples": len(x_val),
        "holdout_samples": len(x_hold),
        "target_mean": target_mean,
        "target_std": target_std,
        "target_min": float(train_df[TARGET_COL].min()),
        "target_max": float(train_df[TARGET_COL].max()),
        "train_window_x": x_train,
        "train_window_y": y_train,
        # Raw dataframes kept so the worker can rebuild holdout windows
        # with a different normalization (e.g. global stats for student eval).
        "holdout_df": holdout_df,
        "trainval_df": trainval_df,
    }
    return fl_loader, val_loader, holdout_loader, meta


# ── Windows from synthetic series ─────────────────────────────────────────────

def windows_from_synthetic_series(
    series: np.ndarray,
    seq_len: int,
    target_mean: float,
    target_std: float,
    start_timestamp: str = "2009-01-01 00:00:00",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a raw synthetic %RH series (N,) to sliding windows with time features.

    Args:
        series     : 1D array of %RH values, shape (N,)
        seq_len    : window length
        target_mean: normalization mean (from real training data)
        target_std : normalization std  (from real training data)
        start_timestamp: synthetic start time (hourly spacing)
    Returns:
        x: (N - seq_len, seq_len, 5)
        y: (N - seq_len,)   normalized
    """
    n = len(series)
    timestamps = pd.date_range(start=start_timestamp, periods=n, freq="h")
    df = pd.DataFrame({TIMESTAMP_COL: timestamps, TARGET_COL: series.astype(np.float32)})
    df = add_time_features(df)
    return make_windows(df, seq_len, target_mean, target_std)


# ── FL training ───────────────────────────────────────────────────────────────

def train_local_model(
    model: LSTMRegressor,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    algorithm: str = "fedavg",
    global_state: Optional[Dict] = None,
    prox_mu: float = 0.0,
    patience: int = 5,
    epoch_callback=None,
) -> Tuple[LSTMRegressor, float, float]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    global_params = None
    if algorithm == "fedprox" and global_state is not None:
        global_params = {k: v.detach().clone().to(device)
                         for k, v in global_state.items()
                         if torch.is_tensor(v) and v.is_floating_point()}

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    wait = 0

    for epoch in range(epochs):
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            if algorithm == "fedprox" and global_params is not None:
                prox = sum(torch.sum((p - global_params[name]) ** 2)
                           for name, p in model.named_parameters())
                loss = loss + 0.5 * prox_mu * prox
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = running / max(n, 1)

        val_loss = train_loss
        if val_loader is not None:
            model.eval()
            vr, vn = 0.0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    vr += criterion(model(xb), yb).item() * xb.size(0)
                    vn += xb.size(0)
            val_loss = vr / max(vn, 1)

        if epoch_callback:
            epoch_callback(epoch + 1, epochs, float(train_loss), float(val_loss))

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = float(val_loss)
            best_train_loss = float(train_loss)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, float(best_train_loss), float(best_val_loss)


# ── FedAvg ────────────────────────────────────────────────────────────────────

def fedavg(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[int],
) -> Dict[str, torch.Tensor]:
    total = float(sum(weights))
    out = {}
    for k in state_dicts[0].keys():
        out[k] = sum(sd[k] * (w / total) for sd, w in zip(state_dicts, weights))
    return out


# ── Teacher gradient computation (frozen) ────────────────────────────────────

def compute_teacher_gradients(
    teacher: LSTMRegressor,
    real_batch_x: torch.Tensor,
    real_batch_y: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Compute gradients of MSE loss w.r.t. teacher parameters on one real batch.
    Teacher weights are NOT updated.
    Returns grad dict {param_name: grad_tensor}.
    """
    teacher = teacher.to(device)
    teacher.train()   # enable grad computation through BN/dropout
    # Zero any existing grads
    for p in teacher.parameters():
        if p.grad is not None:
            p.grad.zero_()

    xb = real_batch_x.to(device)
    yb = real_batch_y.to(device)
    loss = nn.MSELoss()(teacher(xb), yb)
    loss.backward()

    grads = {}
    for name, param in teacher.named_parameters():
        grads[name] = (param.grad.detach().clone()
                       if param.grad is not None
                       else torch.zeros_like(param))
    # Restore zero grads so teacher is not accidentally updated
    for p in teacher.parameters():
        if p.grad is not None:
            p.grad.zero_()
    return grads


def aggregate_teacher_gradients(
    grad_list: List[Dict[str, torch.Tensor]],
    weights: List[int],
) -> Dict[str, torch.Tensor]:
    """Weighted average of per-client teacher gradient dicts."""
    total = float(sum(weights))
    out = {}
    for k in grad_list[0].keys():
        out[k] = sum(g[k] * (w / total) for g, w in zip(grad_list, weights))
    return out


# ── Discriminator update (local, client-side) ─────────────────────────────────

def update_discriminator(
    discriminator: nn.Module,
    real_series: torch.Tensor,
    fake_series: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    One discriminator update step.
    real_series: (batch, 512, 1)  from client's real %RH
    fake_series: (batch, 512, 1)  from Generator (detached)
    Returns D_loss scalar.
    """
    discriminator = discriminator.to(device)
    discriminator.train()
    real_series = real_series.to(device)
    fake_series = fake_series.detach().to(device)

    real_labels = torch.ones(real_series.size(0), 1, device=device)
    fake_labels = torch.zeros(fake_series.size(0), 1, device=device)

    optimizer.zero_grad()
    d_real = discriminator(real_series)
    d_fake = discriminator(fake_series)
    loss = (nn.BCELoss()(d_real, real_labels) +
            nn.BCELoss()(d_fake, fake_labels)) * 0.5
    loss.backward()
    nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    model: LSTMRegressor,
    loader: DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> Dict[str, float]:
    model = model.to(device)
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).detach().cpu().numpy())
            ys.append(yb.numpy())
    if not preds:
        return {"mse": float("nan"), "rmse": float("nan"),
                "mae": float("nan"), "r2": float("nan"), "num_samples": 0}
    y_true = np.concatenate(ys) * target_std + target_mean
    y_pred = np.concatenate(preds) * target_std + target_mean
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = (float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)
          if denom > 0 else float("nan"))
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2,
            "num_samples": int(len(y_true))}


def weighted_average_metric(eval_list: List[Dict], key: str) -> float:
    valid = [x for x in eval_list
             if int(x.get("num_samples", 0)) > 0
             and not math.isnan(float(x[key]))]
    if not valid:
        return float("nan")
    total = sum(int(x["num_samples"]) for x in valid)
    return float(sum(int(x["num_samples"]) * float(x[key]) for x in valid) / total)


# ── Synthetic series → CSV ────────────────────────────────────────────────────

def save_synthetic_timeseries_csv(
    output_path: str,
    rh_values: np.ndarray,
    start_timestamp: str = "2009-01-01 00:00:00",
) -> None:
    """Save generated %RH series as a semicolon-delimited CSV matching real format."""
    n = len(rh_values)
    timestamps = pd.date_range(start=start_timestamp, periods=n, freq="h")
    df = pd.DataFrame({
        TIMESTAMP_COL: timestamps.strftime("%d-%m-%Y %H:%M"),
        TARGET_COL: rh_values.astype(np.float32),
    })
    df.to_csv(output_path, sep=";", index=False)
    print(f"[Gen] Synthetic timeseries saved → {output_path}  ({n} rows)")
