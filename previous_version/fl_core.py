import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from models import LSTMRegressor

TIMESTAMP_COL = "DATE AND TIME GMT"
TARGET_COL = "%RH"
TIME_FEATURES = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
FEATURE_COLUMNS = [TARGET_COL] + TIME_FEATURES
MODEL_INPUT_DIM = len(FEATURE_COLUMNS)


@dataclass
class ClientDataBundle:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    trainval_df: pd.DataFrame
    holdout_df: pd.DataFrame
    full_df: pd.DataFrame
    time_delta_minutes: int
    train_rows: int
    val_rows: int
    trainval_rows: int
    holdout_rows: int
    full_rows: int
    data_start_timestamp: str
    data_end_timestamp: str
    target_mean: float
    target_std: float


class SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def _find_col(candidates, all_cols):
    for cand in candidates:
        if cand in all_cols:
            return cand
    lower_map = {str(c).lower(): c for c in all_cols}
    for cand in candidates:
        key = str(cand).lower()
        if key in lower_map:
            return lower_map[key]
    return None


def infer_time_delta_minutes(df: pd.DataFrame) -> int:
    ts = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce").dropna().sort_values()
    if len(ts) < 2:
        return 60
    deltas = ts.diff().dropna()
    deltas = deltas[deltas > pd.Timedelta(0)]
    if len(deltas) == 0:
        return 60
    return max(int(round(deltas.median().total_seconds() / 60.0)), 1)


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


def load_and_prepare_local_dataframe(
    csv_path: str,
    train_ratio: float = 0.7,
    val_ratio_within_train: float = 0.2,
) -> ClientDataBundle:
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    timestamp_candidates = [
        "DATE AND TIME GMT",
        "YYYY-MM-DD HH:MM",
        "YYYY-MM-DD HH:MM:SS",
        "created_at",
        "timestamp",
        "Timestamp",
        "datetime",
        "Datetime",
        "DATE",
        "Date",
    ]
    rh_candidates = ["%RH", "RH", "rh", "RelativeHumidity", "relative_humidity"]

    ts_col = _find_col(timestamp_candidates, df.columns.tolist())
    rh_col = _find_col(rh_candidates, df.columns.tolist())
    if ts_col is None:
        raise ValueError(f"Could not find timestamp column. Available columns: {df.columns.tolist()}")
    if rh_col is None:
        raise ValueError(f"Could not find %RH column. Available columns: {df.columns.tolist()}")

    df = df[[ts_col, rh_col]].copy()
    df.columns = [TIMESTAMP_COL, TARGET_COL]
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce", dayfirst=True)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna().sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df = add_time_features(df)

    if len(df) < 120:
        raise ValueError("Dataset is too small after cleaning.")

    trainval_end = int(len(df) * train_ratio)
    trainval_end = min(max(trainval_end, 60), len(df) - 20)
    trainval_df = df.iloc[:trainval_end].copy().reset_index(drop=True)
    holdout_df = df.iloc[trainval_end:].copy().reset_index(drop=True)

    train_end = int(len(trainval_df) * (1.0 - val_ratio_within_train))
    train_end = min(max(train_end, 40), len(trainval_df) - 10)
    train_df = trainval_df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = trainval_df.iloc[train_end:].copy().reset_index(drop=True)

    if len(train_df) < 40 or len(val_df) < 10 or len(holdout_df) < 10:
        raise ValueError("Train/validation/holdout split is too small.")

    delta = infer_time_delta_minutes(df)
    target_mean = float(train_df[TARGET_COL].mean())
    target_std = float(max(train_df[TARGET_COL].std(ddof=0), 1e-6))

    return ClientDataBundle(
        train_df=train_df,
        val_df=val_df,
        trainval_df=trainval_df,
        holdout_df=holdout_df,
        full_df=df,
        time_delta_minutes=delta,
        train_rows=len(train_df),
        val_rows=len(val_df),
        trainval_rows=len(trainval_df),
        holdout_rows=len(holdout_df),
        full_rows=len(df),
        data_start_timestamp=str(df[TIMESTAMP_COL].iloc[0]),
        data_end_timestamp=str(df[TIMESTAMP_COL].iloc[-1]),
        target_mean=target_mean,
        target_std=target_std,
    )


def _normalize_target(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (values - mean) / max(std, 1e-6)


def make_windows(df: pd.DataFrame, seq_len: int, target_mean: float, target_std: float) -> Tuple[np.ndarray, np.ndarray]:
    feats = df[FEATURE_COLUMNS].copy()
    feats[TARGET_COL] = _normalize_target(feats[TARGET_COL].values.astype(np.float32), target_mean, target_std)
    feat_values = feats.values.astype(np.float32)
    target_values = _normalize_target(df[TARGET_COL].values.astype(np.float32), target_mean, target_std)

    xs, ys = [], []
    for i in range(seq_len, len(df)):
        xs.append(feat_values[i - seq_len:i])
        ys.append(float(target_values[i]))
    if not xs:
        return np.empty((0, seq_len, MODEL_INPUT_DIM), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.float32)


def make_windows_with_context(
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
    seq_len: int,
    target_mean: float,
    target_std: float,
) -> Tuple[np.ndarray, np.ndarray]:
    context = pd.concat([history_df.tail(seq_len), target_df], ignore_index=True)
    x_all, y_all = make_windows(context, seq_len, target_mean, target_std)
    needed = len(target_df)
    if needed <= 0 or len(x_all) == 0:
        return np.empty((0, seq_len, MODEL_INPUT_DIM), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return x_all[-needed:], y_all[-needed:]


def _compute_target_profiles(df: pd.DataFrame) -> Dict:
    tmp = df[[TIMESTAMP_COL, TARGET_COL]].copy()
    tmp["month"] = tmp[TIMESTAMP_COL].dt.month
    tmp["hour"] = tmp[TIMESTAMP_COL].dt.hour
    month_mean = tmp.groupby("month")[TARGET_COL].mean().to_dict()
    month_std = tmp.groupby("month")[TARGET_COL].std().fillna(0.0).to_dict()
    hour_mean = tmp.groupby("hour")[TARGET_COL].mean().to_dict()
    hour_std = tmp.groupby("hour")[TARGET_COL].std().fillna(0.0).to_dict()
    return {
        "month_target_mean": {int(k): float(v) for k, v in month_mean.items()},
        "month_target_std": {int(k): float(v) for k, v in month_std.items()},
        "hour_target_mean": {int(k): float(v) for k, v in hour_mean.items()},
        "hour_target_std": {int(k): float(v) for k, v in hour_std.items()},
    }


def build_client_dataloaders(
    csv_path: str,
    client_id: int,
    num_clients: int,
    seq_len: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, Dict]:
    bundle = load_and_prepare_local_dataframe(csv_path)

    x_train, y_train = make_windows(bundle.train_df, seq_len, bundle.target_mean, bundle.target_std)
    x_val, y_val = make_windows_with_context(bundle.train_df, bundle.val_df, seq_len, bundle.target_mean, bundle.target_std)
    x_hold, y_hold = make_windows_with_context(bundle.trainval_df, bundle.holdout_df, seq_len, bundle.target_mean, bundle.target_std)

    train_loader = DataLoader(SeqDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    fl_loader = DataLoader(SeqDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SeqDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    holdout_loader = DataLoader(SeqDataset(x_hold, y_hold), batch_size=batch_size, shuffle=False)

    values = bundle.full_df[[TARGET_COL]].values.astype(np.float32)
    meta = {
        "client_id": client_id,
        "train_rows": bundle.train_rows,
        "val_rows": bundle.val_rows,
        "trainval_rows": bundle.trainval_rows,
        "holdout_rows": bundle.holdout_rows,
        "full_rows": bundle.full_rows,
        "train_samples": len(x_train),
        "val_samples": len(x_val),
        "holdout_samples": len(x_hold),
        "time_delta_minutes": bundle.time_delta_minutes,
        "data_start_timestamp": bundle.data_start_timestamp,
        "data_end_timestamp": bundle.data_end_timestamp,
        "target_mean": bundle.target_mean,
        "target_std": bundle.target_std,
        "target_min": float(values.min()),
        "target_max": float(values.max()),
        "temporal_profiles": _compute_target_profiles(bundle.trainval_df),
        "train_window_x": x_train,
        "train_window_y": y_train,
        "train_window_timestamps": bundle.train_df[TIMESTAMP_COL].iloc[seq_len:].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
    }
    return train_loader, fl_loader, val_loader, holdout_loader, meta


def train_local_model(
    model: LSTMRegressor,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    algorithm: str = "fedavg",
    global_state: Optional[Dict[str, torch.Tensor]] = None,
    prox_mu: float = 0.0,
    patience: int = 5,
    epoch_callback=None,
) -> Tuple[LSTMRegressor, float, float]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    global_params = None
    if algorithm == "fedprox":
        if global_state is None:
            raise ValueError("global_state must be provided for FedProx.")
        global_params = {
            k: v.detach().clone().to(device)
            for k, v in global_state.items()
            if torch.is_tensor(v) and v.is_floating_point()
        }

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    wait = 0

    for epoch_idx in range(epochs):
        model.train()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            if algorithm == "fedprox":
                prox_term = torch.zeros(1, device=device)
                for name, param in model.named_parameters():
                    prox_term = prox_term + torch.sum((param - global_params[name]) ** 2)
                loss = loss + 0.5 * prox_mu * prox_term
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = running / max(n, 1)

        val_loss = train_loss
        if val_loader is not None:
            model.eval()
            v_running = 0.0
            v_n = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    v_running += loss.item() * xb.size(0)
                    v_n += xb.size(0)
            val_loss = v_running / max(v_n, 1)

        if epoch_callback is not None:
            epoch_callback(epoch_idx + 1, epochs, float(train_loss), float(val_loss))

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


def compute_fedsgd_gradients(model: LSTMRegressor, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, torch.Tensor], float]:
    model = model.to(device)
    model.train()
    model.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = torch.sum((pred - yb) ** 2)
        loss.backward()
        total_loss += float(loss.item())
        total_n += xb.size(0)
    grad_dict = {}
    scale = 1.0 / max(total_n, 1)
    for name, param in model.named_parameters():
        if param.grad is None:
            grad_dict[name] = torch.zeros_like(param)
        else:
            grad_dict[name] = param.grad.detach().clone() * scale
    return grad_dict, float(total_loss / max(total_n, 1))


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
            xb = xb.to(device)
            pred = model(xb).detach().cpu().numpy()
            preds.append(pred)
            ys.append(yb.numpy())
    if not preds:
        return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "num_samples": 0}
    y_true = np.concatenate(ys) * target_std + target_mean
    y_pred = np.concatenate(preds) * target_std + target_mean
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 0 else float("nan")
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "num_samples": int(len(y_true))}


def fedavg(state_dicts: List[Dict[str, torch.Tensor]], weights: List[int]) -> Dict[str, torch.Tensor]:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Total weight must be positive.")
    out = {}
    for k in state_dicts[0].keys():
        out[k] = sum(sd[k] * (w / total) for sd, w in zip(state_dicts, weights))
    return out


def fedsgd(grad_dicts: List[Dict[str, torch.Tensor]], weights: List[int]) -> Dict[str, torch.Tensor]:
    return fedavg(grad_dicts, weights)


def apply_gradient_update(model: LSTMRegressor, agg_grad: Dict[str, torch.Tensor], lr: float) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            param -= lr * agg_grad[name].to(param.device)


def extract_client_stats_for_aggregator(meta: Dict) -> Dict:
    profiles = meta["temporal_profiles"]
    return {
        "num_samples": int(meta["train_samples"]),
        "target_mean": float(meta["target_mean"]),
        "target_std": float(meta["target_std"]),
        "target_min": float(meta["target_min"]),
        "target_max": float(meta["target_max"]),
        "time_delta_minutes": int(meta["time_delta_minutes"]),
        "data_start_timestamp": meta["data_start_timestamp"],
        "data_end_timestamp": meta["data_end_timestamp"],
        "full_rows": int(meta["full_rows"]),
        "trainval_rows": int(meta["trainval_rows"]),
        "month_target_mean": profiles["month_target_mean"],
        "month_target_std": profiles["month_target_std"],
        "hour_target_mean": profiles["hour_target_mean"],
        "hour_target_std": profiles["hour_target_std"],
    }


def _merge_profile_dicts(stats_list: List[Dict], key: str) -> Dict[int, float]:
    merged = {}
    for k in sorted({int(x) for s in stats_list for x in s.get(key, {}).keys()}):
        vals = []
        weights = []
        for s in stats_list:
            d = s.get(key, {})
            if k in d or str(k) in d:
                vals.append(float(d.get(k, d.get(str(k)))))
                weights.append(max(int(s["num_samples"]), 1))
        if vals:
            merged[k] = float(np.average(vals, weights=weights))
    return merged


def merge_client_stats(stats_list: List[Dict]) -> Dict:
    if not stats_list:
        raise ValueError("No client stats provided.")
    raw_weights = np.array([max(int(s["num_samples"]), 1) for s in stats_list], dtype=np.float64)
    weights = raw_weights / raw_weights.sum()
    mean = float(sum(w * s["target_mean"] for w, s in zip(weights, stats_list)))
    std = float(sum(w * s["target_std"] for w, s in zip(weights, stats_list)))
    vmin = float(min(s["target_min"] for s in stats_list))
    vmax = float(max(s["target_max"] for s in stats_list))
    delta = int(round(sum(w * s["time_delta_minutes"] for w, s in zip(weights, stats_list))))
    return {
        "mean": np.array([mean], dtype=np.float32),
        "std": np.array([max(std, 1e-6)], dtype=np.float32),
        "min": np.array([vmin], dtype=np.float32),
        "max": np.array([vmax], dtype=np.float32),
        "time_delta_minutes": max(delta, 1),
        "data_start_timestamp": min(pd.Timestamp(s["data_start_timestamp"]) for s in stats_list),
        "data_end_timestamp": max(pd.Timestamp(s["data_end_timestamp"]) for s in stats_list),
        "total_full_rows": int(sum(int(s.get("full_rows", 0)) for s in stats_list)),
        "total_trainval_rows": int(sum(int(s.get("trainval_rows", 0)) for s in stats_list)),
        "temporal_profiles": {
            "month_target_mean": _merge_profile_dicts(stats_list, "month_target_mean"),
            "month_target_std": _merge_profile_dicts(stats_list, "month_target_std"),
            "hour_target_mean": _merge_profile_dicts(stats_list, "hour_target_mean"),
            "hour_target_std": _merge_profile_dicts(stats_list, "hour_target_std"),
        },
    }


def weighted_average_metric(eval_list: List[Dict], key: str) -> float:
    valid = [x for x in eval_list if int(x.get("num_samples", 0)) > 0 and not math.isnan(float(x[key]))]
    if not valid:
        return float("nan")
    total = sum(int(x["num_samples"]) for x in valid)
    return float(sum(int(x["num_samples"]) * float(x[key]) for x in valid) / total)


def build_teacher_labeled_distillation_payload(
    train_window_x: np.ndarray,
    train_window_timestamps: List[str],
    teacher_model: LSTMRegressor,
    device: torch.device,
    max_windows: int,
    target_mean: Optional[float] = None,
    target_std: Optional[float] = None,
    seed: int = 0,
) -> Dict:
    if len(train_window_x) == 0:
        return {
            "x": np.empty((0, 0, MODEL_INPUT_DIM), dtype=np.float32),
            "y": np.empty((0,), dtype=np.float32),
            "y_denorm": np.empty((0,), dtype=np.float32),
            "timestamps": [],
            "num_samples": 0,
        }
    if len(train_window_timestamps) != len(train_window_x):
        raise ValueError("train_window_timestamps must match the number of training windows.")
    rng = np.random.default_rng(seed)
    take = min(max_windows, len(train_window_x))
    indices = np.sort(rng.choice(len(train_window_x), size=take, replace=False))
    selected_x = train_window_x[indices].astype(np.float32)
    selected_timestamps = [str(train_window_timestamps[i]) for i in indices]

    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(selected_x), 256):
            xb = torch.tensor(selected_x[start:start + 256], dtype=torch.float32, device=device)
            pred = teacher_model(xb).detach().cpu().numpy()
            preds.append(pred)
    selected_y = np.concatenate(preds).astype(np.float32) if preds else np.empty((0,), dtype=np.float32)

    if target_mean is None or target_std is None:
        selected_y_denorm = selected_y.astype(np.float32)
    else:
        selected_y_denorm = (selected_y * float(target_std) + float(target_mean)).astype(np.float32)

    return {
        "x": selected_x,
        "y": selected_y,
        "y_denorm": selected_y_denorm,
        "timestamps": selected_timestamps,
        "num_samples": int(len(selected_x)),
    }


def merge_distillation_payloads(
    payloads: List[Dict],
    shuffle_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    xs = [p["x"] for p in payloads if int(p.get("num_samples", 0)) > 0]
    ys = [p["y"] for p in payloads if int(p.get("num_samples", 0)) > 0]
    timestamps = [str(ts) for p in payloads if int(p.get("num_samples", 0)) > 0 for ts in p.get("timestamps", [])]
    y_denorm_list = [np.asarray(p.get("y_denorm", []), dtype=np.float32) for p in payloads if int(p.get("num_samples", 0)) > 0]
    if not xs:
        return (
            np.empty((0, 0, MODEL_INPUT_DIM), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            [],
            np.empty((0,), dtype=np.float32),
        )
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    y_denorm = np.concatenate(y_denorm_list, axis=0).astype(np.float32) if y_denorm_list else np.empty((0,), dtype=np.float32)

    rng = np.random.default_rng(shuffle_seed)
    idx = np.arange(len(x))
    rng.shuffle(idx)

    shuffled_timestamps = [timestamps[i] for i in idx] if len(timestamps) == len(x) else timestamps
    shuffled_y_denorm = y_denorm[idx] if len(y_denorm) == len(x) else y_denorm
    return x[idx], y[idx], shuffled_timestamps, shuffled_y_denorm


def save_distilled_dataset(output_path: str, x: np.ndarray, y: np.ndarray) -> None:
    np.savez_compressed(output_path, x=x.astype(np.float32), y=y.astype(np.float32))


def save_distilled_timeseries_csv(output_path: str, timestamps: List[str], values: np.ndarray) -> None:
    if len(timestamps) != len(values):
        raise ValueError("timestamps and values must have the same length for CSV export.")
    df = pd.DataFrame({
        TIMESTAMP_COL: pd.to_datetime(pd.Series(timestamps), errors="coerce"),
        TARGET_COL: np.asarray(values, dtype=np.float32),
    })
    df = df.dropna(subset=[TIMESTAMP_COL]).sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df[TIMESTAMP_COL] = df[TIMESTAMP_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(output_path, index=False)


def train_student_on_distilled(
    distilled_path: str,
    batch_size: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    device: torch.device,
    epochs: int,
    lr: float,
    val_fraction: float = 0.1,
) -> Tuple[LSTMRegressor, Dict]:
    blob = np.load(distilled_path)
    x = blob["x"].astype(np.float32)
    y = blob["y"].astype(np.float32)
    if len(x) == 0:
        raise ValueError("Distilled dataset is empty.")

    val_size = max(1, int(len(x) * val_fraction)) if len(x) >= 10 else 1
    val_size = min(val_size, len(x) - 1) if len(x) > 1 else 0
    if val_size > 0:
        x_train, y_train = x[:-val_size], y[:-val_size]
        x_val, y_val = x[-val_size:], y[-val_size:]
        val_loader = DataLoader(SeqDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    else:
        x_train, y_train = x, y
        val_loader = None

    train_loader = DataLoader(SeqDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    student = LSTMRegressor(
        input_dim=MODEL_INPUT_DIM,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    student, train_loss, val_loss = train_local_model(
        model=student,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        algorithm="fedavg",
        patience=5,
    )
    meta = {
        "num_rows": int(len(x)),
        "num_samples": int(len(x)),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
    }
    return student, meta
