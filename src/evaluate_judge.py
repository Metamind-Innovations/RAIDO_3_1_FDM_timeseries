"""
evaluate_judge.py
Independent evaluation using a fresh CNN1DRegressor "judge" that was never
involved in teacher training or distillation.

Two comparisons on REAL holdout data:
  Phase A — Real (per-client):  For each client, train a fresh judge on that
            client's train split, evaluate on that client's holdout.
  Phase B — Synthetic:          Train a fresh judge on the synthetic dataset
            (global norm), evaluate on each client's holdout (global norm).

All data is loaded directly from CSVs — no sockets, no running workers needed.

Usage:
    python evaluate_judge.py \
        --dataset climate \
        --client_csvs <client1_csv> <client2_csv> \
        --synthetic_csv best_synthetic_timeseries.csv \
        --stats_json best_gen_stats.json \
        --balanced_len 36842 \
        --seq_len 24 --batch_size 32 \
        --hidden_channels 64 --dropout 0.2 \
        --judge_epochs 30 --judge_lr 5e-4 --patience 5 \
        --device cuda
"""

import argparse
import json
import os
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader

import gen_fl_core
from gen_fl_core import (
    MODEL_INPUT_DIM,
    TARGET_COL,
    SeqDataset,
    add_time_features,
    evaluate_model,
    load_csv,
    make_windows,
    make_windows_with_context,
    split_df,
    train_local_model,
    weighted_average_metric,
)
from models import CNN1DRegressor


# ── Data loading ───────────────────────────────────────────────────────────────

def load_client(
    csv_path: str,
    client_id: int,
    balanced_len: int,
    seq_len: int,
    batch_size: int,
) -> Dict:
    """
    Load one client CSV, split into train/val/holdout, build DataLoaders.
    Keeps raw DataFrames so holdout windows can be rebuilt with different stats.
    """
    df_raw = load_csv(csv_path)
    df_raw = df_raw.iloc[:balanced_len].reset_index(drop=True)
    df = add_time_features(df_raw)

    train_df, val_df, holdout_df = split_df(df)
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    local_mean = float(train_df[TARGET_COL].mean())
    local_std = float(max(train_df[TARGET_COL].std(ddof=0), 1e-6))

    x_train, y_train = make_windows(train_df, seq_len, local_mean, local_std)
    x_val, y_val = make_windows_with_context(
        train_df, val_df, seq_len, local_mean, local_std
    )
    x_hold, y_hold = make_windows_with_context(
        trainval_df, holdout_df, seq_len, local_mean, local_std
    )

    train_loader = DataLoader(
        SeqDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        SeqDataset(x_val, y_val), batch_size=batch_size, shuffle=False
    )
    holdout_loader = DataLoader(
        SeqDataset(x_hold, y_hold), batch_size=batch_size, shuffle=False
    )

    print(
        f"[Client {client_id}] {os.path.basename(csv_path)} | "
        f"train={len(x_train)} val={len(x_val)} holdout={len(x_hold)} windows | "
        f"local mean={local_mean:.2f} std={local_std:.2f}"
    )

    return {
        "client_id": client_id,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "holdout_loader": holdout_loader,
        "holdout_samples": len(x_hold),
        "local_mean": local_mean,
        "local_std": local_std,
        "trainval_df": trainval_df,
        "holdout_df": holdout_df,
    }


def holdout_loader_with_stats(
    client: Dict,
    target_mean: float,
    target_std: float,
    seq_len: int,
    batch_size: int,
) -> DataLoader:
    """Rebuild holdout windows using different normalization stats."""
    x_h, y_h = make_windows_with_context(
        client["trainval_df"], client["holdout_df"],
        seq_len, target_mean, target_std,
    )
    return DataLoader(SeqDataset(x_h, y_h), batch_size=batch_size, shuffle=False)


# ── Judge factory + training ───────────────────────────────────────────────────

def fresh_judge(hidden_channels: int, dropout: float, seq_len: int) -> CNN1DRegressor:
    """New, untrained CNN judge."""
    return CNN1DRegressor(
        input_dim=MODEL_INPUT_DIM,
        seq_len=seq_len,
        hidden_channels=hidden_channels,
        dropout=dropout,
    )


def train_judge(
    judge: CNN1DRegressor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    patience: int,
    device: torch.device,
    label: str,
) -> CNN1DRegressor:
    """Train a judge with early stopping, logging every 5th epoch."""
    judge, tr_loss, va_loss = train_local_model(
        model=judge,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        algorithm="fedavg",
        patience=patience,
        epoch_callback=lambda e, total, tr, va: print(
            f"  [{label}] epoch {e:3d}/{total} | train={tr:.6f} | val={va:.6f}"
        ) if e % 5 == 0 or e == 1 else None,
    )
    print(f"  [{label}] done | best_train={tr_loss:.6f} | best_val={va_loss:.6f}")
    return judge


# ── Phase A: per-client real training ─────────────────────────────────────────

def evaluate_per_client_real(
    clients: List[Dict],
    hidden_channels: int,
    dropout: float,
    epochs: int,
    lr: float,
    patience: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> List[Dict]:
    """
    Each client trains its own fresh judge on its local real data.
    Evaluated on that client's holdout with local normalization.
    """
    results = []
    for client in clients:
        cid = client["client_id"]
        print(f"\n[Judge-Real] Client {cid} — training fresh CNN judge on real data...")
        judge = fresh_judge(hidden_channels, dropout, seq_len)
        judge = train_judge(
            judge,
            client["train_loader"],
            client["val_loader"],
            epochs, lr, patience, device,
            label=f"Judge-Real-C{cid}",
        )
        metrics = evaluate_model(
            judge,
            client["holdout_loader"],
            device,
            client["local_mean"],
            client["local_std"],
        )
        metrics["num_samples"] = client["holdout_samples"]
        print(
            f"[Judge-Real C{cid}] RMSE={metrics['rmse']:.6f} | "
            f"MAE={metrics['mae']:.6f} | R2={metrics['r2']:.6f}"
        )
        results.append({"client_id": cid, "metrics": metrics})
    return results


# ── Phase B: synthetic training ────────────────────────────────────────────────

def evaluate_synthetic(
    synthetic_csv: str,
    clients: List[Dict],
    target_mean: float,
    target_std: float,
    hidden_channels: int,
    dropout: float,
    epochs: int,
    lr: float,
    patience: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    val_fraction: float = 0.1,
) -> List[Dict]:
    """
    Train one fresh judge on synthetic data (global norm).
    Evaluated on each client's holdout rebuilt with the same global norm.
    """
    df_syn = load_csv(synthetic_csv)
    df_syn = add_time_features(df_syn)

    x_all, y_all = make_windows(df_syn, seq_len, target_mean, target_std)
    if len(x_all) == 0:
        raise ValueError(f"Synthetic CSV produced no windows: {synthetic_csv}")

    n_val = max(1, int(len(x_all) * val_fraction))
    x_train, y_train = x_all[:-n_val], y_all[:-n_val]
    x_val, y_val = x_all[-n_val:], y_all[-n_val:]

    train_loader = DataLoader(
        SeqDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        SeqDataset(x_val, y_val), batch_size=batch_size, shuffle=False
    )

    print(
        f"\n[Judge-Syn] Training fresh CNN judge on synthetic data | "
        f"total={len(x_all)} train={len(x_train)} val={len(x_val)} windows"
    )
    judge = fresh_judge(hidden_channels, dropout, seq_len)
    judge = train_judge(
        judge, train_loader, val_loader,
        epochs, lr, patience, device,
        label="Judge-Syn",
    )

    results = []
    for client in clients:
        cid = client["client_id"]
        eval_loader = holdout_loader_with_stats(
            client, target_mean, target_std, seq_len, batch_size
        )
        n_holdout = len(eval_loader.dataset)
        metrics = evaluate_model(judge, eval_loader, device, target_mean, target_std)
        metrics["num_samples"] = n_holdout
        print(
            f"[Judge-Syn C{cid}] RMSE={metrics['rmse']:.6f} | "
            f"MAE={metrics['mae']:.6f} | R2={metrics['r2']:.6f}"
        )
        results.append({"client_id": cid, "metrics": metrics})
    return results


# ── Results table ──────────────────────────────────────────────────────────────

def print_results(real_results: List[Dict], syn_results: List[Dict]) -> None:
    """Print per-client and weighted-average Real vs Synthetic judge metrics and the gap."""
    real_map = {int(r["client_id"]): r["metrics"] for r in real_results}
    syn_map = {int(r["client_id"]): r["metrics"] for r in syn_results}
    all_cids = sorted(set(real_map) | set(syn_map))

    print(f"\n{'='*82}")
    print("  Judge Evaluation (CNN1DRegressor) on REAL holdout data")
    print(f"{'='*82}")
    print(
        f"  {'Client':<8} {'Trained on':<18} {'RMSE':>10} {'MAE':>10} {'R2':>10}"
    )
    print(f"  {'-'*8} {'-'*18} {'-'*10} {'-'*10} {'-'*10}")
    for cid in all_cids:
        for label, mmap in [("Real (local)", real_map), ("Synthetic", syn_map)]:
            m = mmap.get(cid)
            if m:
                print(
                    f"  {cid:<8} {label:<18} "
                    f"{m['rmse']:>10.6f} {m['mae']:>10.6f} {m['r2']:>10.6f}"
                )
        print(f"  {'-'*8} {'-'*18} {'-'*10} {'-'*10} {'-'*10}")

    real_ml = [{"num_samples": r["metrics"]["num_samples"], **r["metrics"]}
               for r in real_results]
    syn_ml = [{"num_samples": r["metrics"]["num_samples"], **r["metrics"]}
              for r in syn_results]

    print(f"\n  Weighted average across all clients:")
    for label, ml in [("Real (local)", real_ml), ("Synthetic", syn_ml)]:
        print(
            f"  {label:<18} | "
            f"RMSE={weighted_average_metric(ml, 'rmse'):.6f} | "
            f"MAE={weighted_average_metric(ml, 'mae'):.6f} | "
            f"R2={weighted_average_metric(ml, 'r2'):.6f}"
        )

    r_rmse = weighted_average_metric(real_ml, "rmse")
    s_rmse = weighted_average_metric(syn_ml, "rmse")
    r_r2 = weighted_average_metric(real_ml, "r2")
    s_r2 = weighted_average_metric(syn_ml, "r2")
    print(f"\n  RMSE gap (Synthetic - Real): {s_rmse - r_rmse:+.6f}")
    print(f"  R2   gap (Synthetic - Real): {s_r2 - r_r2:+.6f}")
    print(f"  (smaller gap = better distillation quality)")
    print(f"{'='*82}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Judge evaluation using CNN1DRegressor (independent of distillation pipeline)"
    )
    parser.add_argument(
        "--client_csvs", nargs="+", required=True,
        help="Client CSV paths in client-id order"
    )
    parser.add_argument("--synthetic_csv", type=str, default="best_synthetic_timeseries.csv")
    parser.add_argument(
        "--stats_json", type=str, default="best_gen_stats.json",
        help="JSON file with target_mean and target_std from distillation run"
    )
    parser.add_argument("--balanced_len", type=int, default=36842)
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--judge_epochs", type=int, default=30)
    parser.add_argument("--judge_lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--target_mean", type=float, default=None,
                        help="Override normalization mean (skips stats_json)")
    parser.add_argument("--target_std", type=float, default=None,
                        help="Override normalization std (skips stats_json)")
    parser.add_argument("--dataset", type=str, default="climate", choices=["climate", "pilot"],
                        help="Data setup: climate (Church/Museum %%RH) or pilot (pilot sensors).")
    args = parser.parse_args()
    gen_fl_core.set_dataset(args.dataset)

    device = torch.device(args.device)

    # ── Normalization stats ────────────────────────────────────────────────────
    if args.target_mean is not None and args.target_std is not None:
        target_mean, target_std = args.target_mean, args.target_std
        print(f"[Judge] Stats (provided): mean={target_mean:.4f} std={target_std:.4f}")
    elif os.path.exists(args.stats_json):
        with open(args.stats_json) as f:
            stats = json.load(f)
        target_mean = float(stats["target_mean"])
        target_std = float(stats["target_std"])
        print(
            f"[Judge] Stats from {args.stats_json}: "
            f"mean={target_mean:.4f} std={target_std:.4f}"
        )
    else:
        raise FileNotFoundError(
            f"{args.stats_json} not found. "
            "Pass --target_mean and --target_std, or --stats_json <path>."
        )

    # ── Load client data ───────────────────────────────────────────────────────
    clients = [
        load_client(csv_path, client_id=idx, balanced_len=args.balanced_len,
                    seq_len=args.seq_len, batch_size=args.batch_size)
        for idx, csv_path in enumerate(args.client_csvs, start=1)
    ]

    # ── Phase A ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Phase A: Judge trained on each client's REAL data")
    print("=" * 60)
    real_results = evaluate_per_client_real(
        clients=clients,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        epochs=args.judge_epochs,
        lr=args.judge_lr,
        patience=args.patience,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
    )

    # ── Phase B ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Phase B: Judge trained on SYNTHETIC data")
    print("=" * 60)
    syn_results = evaluate_synthetic(
        synthetic_csv=args.synthetic_csv,
        clients=clients,
        target_mean=target_mean,
        target_std=target_std,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        epochs=args.judge_epochs,
        lr=args.judge_lr,
        patience=args.patience,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
    )

    # ── Results ────────────────────────────────────────────────────────────────
    print_results(real_results, syn_results)


if __name__ == "__main__":
    main()
