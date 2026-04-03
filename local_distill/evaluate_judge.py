"""
local_distill/evaluate_judge.py
Three-way independent evaluation using a fresh CNN1DRegressor judge.

Phase A — Per-client real:
    For each client chunk, train a fresh judge on that chunk's train split,
    evaluate on that chunk's holdout. Report per-client + weighted average.

Phase B — Synthetic:
    Train one fresh judge on synthetic data (global norm),
    evaluate on each client's holdout (rebuilt with global norm).
    Report per-client + weighted average.

Phase C — Centralized upper bound:
    Train one fresh judge on ALL chunks' train data concatenated (centralized norm),
    evaluate on ALL chunks' holdouts concatenated.
    This is what you'd get with no privacy constraints at all.

All data is loaded directly — no sockets needed.

Usage:
    python local_distill/evaluate_judge.py \
        --data_path oerholm-indoor-climate-data_2009-2021.csv \
        --num_chunks 3 \
        --synthetic_csv local_distill/synthetic_timeseries.csv \
        --stats_json local_distill/gen_stats.json \
        --judge_epochs 30 --judge_lr 5e-4 --patience 5 \
        --device cuda
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def load_chunk(
    df_full: pd.DataFrame,
    chunk_id: int,
    num_chunks: int,
    seq_len: int,
    batch_size: int,
) -> Dict:
    """Slice the full DataFrame into one temporal chunk and build DataLoaders."""
    n = len(df_full)
    start = (chunk_id - 1) * n // num_chunks
    end = chunk_id * n // num_chunks
    df_chunk = df_full.iloc[start:end].reset_index(drop=True)

    train_df, val_df, holdout_df = split_df(df_chunk)
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

    print(
        f"  Chunk {chunk_id}: rows {start}–{end-1} | "
        f"train={len(x_train)} val={len(x_val)} holdout={len(x_hold)} windows | "
        f"mean={local_mean:.2f} std={local_std:.2f}"
    )

    return {
        "client_id": chunk_id,
        "train_loader": DataLoader(SeqDataset(x_train, y_train),
                                   batch_size=batch_size, shuffle=True),
        "val_loader": DataLoader(SeqDataset(x_val, y_val),
                                 batch_size=batch_size, shuffle=False),
        "holdout_loader": DataLoader(SeqDataset(x_hold, y_hold),
                                     batch_size=batch_size, shuffle=False),
        "holdout_samples": len(x_hold),
        "local_mean": local_mean,
        "local_std": local_std,
        "train_dataset": SeqDataset(x_train, y_train),
        "val_dataset": SeqDataset(x_val, y_val),
        "holdout_dataset": SeqDataset(x_hold, y_hold),
        "x_train": x_train, "y_train": y_train,
        "trainval_df": trainval_df,
        "holdout_df": holdout_df,
    }


def holdout_loader_with_stats(
    chunk: Dict,
    target_mean: float,
    target_std: float,
    seq_len: int,
    batch_size: int,
) -> DataLoader:
    x_h, y_h = make_windows_with_context(
        chunk["trainval_df"], chunk["holdout_df"],
        seq_len, target_mean, target_std,
    )
    return DataLoader(SeqDataset(x_h, y_h), batch_size=batch_size, shuffle=False)


# ── Judge factory + training ───────────────────────────────────────────────────

def fresh_judge(hidden_channels: int, dropout: float, seq_len: int) -> CNN1DRegressor:
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
    judge, tr, va = train_local_model(
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
    print(f"  [{label}] done | best_train={tr:.6f} | best_val={va:.6f}")
    return judge


# ── Phase A: per-client real ───────────────────────────────────────────────────

def phase_a(
    chunks: List[Dict],
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
    Train a fresh judge on each client's real data using that client's own
    local normalization stats. No information from the distillation pipeline
    is used here — this is the clean independent baseline.
    """
    results = []
    for chunk in chunks:
        cid = chunk["client_id"]
        print(f"\n[Judge-Real] Chunk {cid} — training fresh CNN judge on real data...")

        judge = fresh_judge(hidden_channels, dropout, seq_len)
        judge = train_judge(
            judge, chunk["train_loader"], chunk["val_loader"],
            epochs, lr, patience, device, label=f"Real-C{cid}",
        )
        metrics = evaluate_model(
            judge, chunk["holdout_loader"], device,
            chunk["local_mean"], chunk["local_std"],
        )
        metrics["num_samples"] = chunk["holdout_samples"]
        print(f"[Judge-Real C{cid}] RMSE={metrics['rmse']:.6f} | "
              f"MAE={metrics['mae']:.6f} | R2={metrics['r2']:.6f}")
        results.append({"client_id": cid, "metrics": metrics})
    return results


# ── Phase B: synthetic ─────────────────────────────────────────────────────────

def phase_b(
    synthetic_csv: str,
    chunks: List[Dict],
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
    df_syn = load_csv(synthetic_csv)
    df_syn = add_time_features(df_syn)
    x_all, y_all = make_windows(df_syn, seq_len, target_mean, target_std)
    if len(x_all) == 0:
        raise ValueError(f"Synthetic CSV produced no windows: {synthetic_csv}")

    n_val = max(1, int(len(x_all) * val_fraction))
    x_tr, y_tr = x_all[:-n_val], y_all[:-n_val]
    x_va, y_va = x_all[-n_val:], y_all[-n_val:]

    train_loader = DataLoader(SeqDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SeqDataset(x_va, y_va), batch_size=batch_size, shuffle=False)

    print(f"\n[Judge-Syn] Training fresh CNN judge on synthetic data | "
          f"total={len(x_all)} train={len(x_tr)} val={len(x_va)} windows")
    judge = fresh_judge(hidden_channels, dropout, seq_len)
    judge = train_judge(judge, train_loader, val_loader,
                        epochs, lr, patience, device, label="Syn")

    results = []
    for chunk in chunks:
        cid = chunk["client_id"]
        eval_loader = holdout_loader_with_stats(
            chunk, target_mean, target_std, seq_len, batch_size
        )
        metrics = evaluate_model(judge, eval_loader, device, target_mean, target_std)
        metrics["num_samples"] = len(eval_loader.dataset)
        print(f"[Judge-Syn C{cid}] RMSE={metrics['rmse']:.6f} | "
              f"MAE={metrics['mae']:.6f} | R2={metrics['r2']:.6f}")
        results.append({"client_id": cid, "metrics": metrics})
    return results


# ── Phase C: centralized upper bound ──────────────────────────────────────────

def phase_c(
    chunks: List[Dict],
    hidden_channels: int,
    dropout: float,
    epochs: int,
    lr: float,
    patience: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    val_fraction: float = 0.1,
) -> Dict:
    """
    Train one judge on ALL chunks' train data concatenated (centralized norm),
    evaluate on ALL chunks' holdouts concatenated.
    """
    # Compute centralized normalization from all training data
    all_x = np.concatenate([c["x_train"] for c in chunks], axis=0)
    all_y = np.concatenate([c["y_train"] for c in chunks], axis=0)

    # Use the first chunk's local mean/std as reference to recover raw values,
    # then recompute global stats across all training targets
    # (simpler: collect all raw %RH from train splits via local de-normalization)
    raw_y_values = []
    for chunk in chunks:
        raw = chunk["y_train"] * chunk["local_std"] + chunk["local_mean"]
        raw_y_values.append(raw)
    all_raw = np.concatenate(raw_y_values)
    cent_mean = float(all_raw.mean())
    cent_std = float(max(all_raw.std(ddof=0), 1e-6))
    print(f"\n[Judge-Central] Centralized norm: mean={cent_mean:.4f} std={cent_std:.4f}")

    # Rebuild windows for all chunks with centralized stats
    all_train_x, all_train_y = [], []
    all_hold_x, all_hold_y = [], []
    for chunk in chunks:
        x_tr, y_tr = make_windows(
            pd.concat([
                chunk["trainval_df"].iloc[:len(chunk["trainval_df"]) - len(chunk["holdout_df"])]
            ], ignore_index=True),
            seq_len, cent_mean, cent_std,
        )
        # Rebuild train split only (approximate — use stored x_train re-normed)
        # More accurate: rebuild from raw DataFrames
        from gen_fl_core import split_df as _split_df
        # We have trainval_df and holdout_df — re-split trainval for train only
        n_tv = len(chunk["trainval_df"])
        n_train_approx = int(n_tv * 0.8)
        train_part = chunk["trainval_df"].iloc[:n_train_approx]
        val_part = chunk["trainval_df"].iloc[n_train_approx:]

        xt, yt = make_windows(train_part, seq_len, cent_mean, cent_std)
        xv, yv = make_windows_with_context(train_part, val_part,
                                           seq_len, cent_mean, cent_std)
        all_train_x.append(xt)
        all_train_y.append(yt)

        xh, yh = make_windows_with_context(
            chunk["trainval_df"], chunk["holdout_df"],
            seq_len, cent_mean, cent_std,
        )
        all_hold_x.append(xh)
        all_hold_y.append(yh)

    x_train_cat = np.concatenate(all_train_x, axis=0)
    y_train_cat = np.concatenate(all_train_y, axis=0)
    x_hold_cat = np.concatenate(all_hold_x, axis=0)
    y_hold_cat = np.concatenate(all_hold_y, axis=0)

    n_val = max(1, int(len(x_train_cat) * val_fraction))
    x_tr2, y_tr2 = x_train_cat[:-n_val], y_train_cat[:-n_val]
    x_va2, y_va2 = x_train_cat[-n_val:], y_train_cat[-n_val:]

    train_loader = DataLoader(SeqDataset(x_tr2, y_tr2),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SeqDataset(x_va2, y_va2),
                            batch_size=batch_size, shuffle=False)
    hold_loader = DataLoader(SeqDataset(x_hold_cat, y_hold_cat),
                             batch_size=batch_size, shuffle=False)

    print(f"[Judge-Central] Training on concatenated data | "
          f"train={len(x_tr2)} val={len(x_va2)} holdout={len(x_hold_cat)} windows")
    judge = fresh_judge(hidden_channels, dropout, seq_len)
    judge = train_judge(judge, train_loader, val_loader,
                        epochs, lr, patience, device, label="Central")

    metrics = evaluate_model(judge, hold_loader, device, cent_mean, cent_std)
    metrics["num_samples"] = len(x_hold_cat)
    print(f"[Judge-Central] RMSE={metrics['rmse']:.6f} | "
          f"MAE={metrics['mae']:.6f} | R2={metrics['r2']:.6f}")
    return metrics


# ── Results table ──────────────────────────────────────────────────────────────

def print_results(
    real_results: List[Dict],
    syn_results: List[Dict],
    central_metrics: Dict,
) -> None:
    real_map = {int(r["client_id"]): r["metrics"] for r in real_results}
    syn_map = {int(r["client_id"]): r["metrics"] for r in syn_results}
    all_cids = sorted(set(real_map) | set(syn_map))

    print(f"\n{'='*88}")
    print("  Judge Evaluation (CNN1DRegressor) — 3-way comparison on REAL holdout data")
    print(f"{'='*88}")
    print(f"  {'Client':<8} {'Trained on':<20} {'RMSE':>10} {'MAE':>10} {'R2':>10}")
    print(f"  {'-'*8} {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

    for cid in all_cids:
        for label, mmap in [("Real (local)", real_map), ("Synthetic", syn_map)]:
            m = mmap.get(cid)
            if m:
                print(f"  {cid:<8} {label:<20} "
                      f"{m['rmse']:>10.6f} {m['mae']:>10.6f} {m['r2']:>10.6f}")
        print(f"  {'-'*8} {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

    real_ml = [{"num_samples": r["metrics"]["num_samples"], **r["metrics"]}
               for r in real_results]
    syn_ml = [{"num_samples": r["metrics"]["num_samples"], **r["metrics"]}
              for r in syn_results]

    print(f"\n  Weighted average across all clients:")
    for label, ml in [("Real (local)", real_ml), ("Synthetic", syn_ml)]:
        print(f"  {label:<20} | "
              f"RMSE={weighted_average_metric(ml, 'rmse'):.6f} | "
              f"MAE={weighted_average_metric(ml, 'mae'):.6f} | "
              f"R2={weighted_average_metric(ml, 'r2'):.6f}")

    print(f"\n  Centralized (upper bound — all data pooled):")
    print(f"  {'Centralized':<20} | "
          f"RMSE={central_metrics['rmse']:.6f} | "
          f"MAE={central_metrics['mae']:.6f} | "
          f"R2={central_metrics['r2']:.6f}")

    r_rmse = weighted_average_metric(real_ml, "rmse")
    s_rmse = weighted_average_metric(syn_ml, "rmse")
    c_rmse = central_metrics["rmse"]
    r_r2 = weighted_average_metric(real_ml, "r2")
    s_r2 = weighted_average_metric(syn_ml, "r2")
    c_r2 = central_metrics["r2"]

    print(f"\n  RMSE: Real(local)={r_rmse:.6f} | "
          f"Synthetic={s_rmse:.6f} | Centralized={c_rmse:.6f}")
    print(f"  R2:   Real(local)={r_r2:.6f} | "
          f"Synthetic={s_r2:.6f} | Centralized={c_r2:.6f}")
    print(f"\n  Gap (Synthetic - Real):        RMSE={s_rmse - r_rmse:+.6f} | R2={s_r2 - r_r2:+.6f}")
    print(f"  Gap (Synthetic - Centralized): RMSE={s_rmse - c_rmse:+.6f} | R2={s_r2 - c_r2:+.6f}")
    print(f"{'='*88}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Museum CSV (full file, will be sliced into num_chunks)")
    parser.add_argument("--num_chunks", type=int, default=3)
    parser.add_argument("--synthetic_csv", type=str,
                        default="local_distill/synthetic_timeseries.csv")
    parser.add_argument("--stats_json", type=str,
                        default="local_distill/gen_stats.json",
                        help="gen_stats.json written by local_distill/aggregator.py")
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--judge_epochs", type=int, default=30)
    parser.add_argument("--judge_lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--target_mean", type=float, default=None)
    parser.add_argument("--target_std", type=float, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Normalization stats ───────────────────────────────────────────────────
    if args.target_mean is not None and args.target_std is not None:
        target_mean, target_std = args.target_mean, args.target_std
        print(f"[Judge] Stats (provided): mean={target_mean:.4f} std={target_std:.4f}")
    elif os.path.exists(args.stats_json):
        with open(args.stats_json) as f:
            stats = json.load(f)
        target_mean = float(stats["target_mean"])
        target_std = float(stats["target_std"])
        print(f"[Judge] Stats from {args.stats_json}: "
              f"mean={target_mean:.4f} std={target_std:.4f}")
    else:
        raise FileNotFoundError(
            f"{args.stats_json} not found. "
            "Pass --target_mean and --target_std, or --stats_json <path>."
        )

    # ── Load and slice data ───────────────────────────────────────────────────
    print(f"\n[Judge] Loading {args.data_path} → {args.num_chunks} chunks")
    df_full = load_csv(args.data_path)
    df_full = add_time_features(df_full)

    chunks = [
        load_chunk(df_full, chunk_id=i, num_chunks=args.num_chunks,
                   seq_len=args.seq_len, batch_size=args.batch_size)
        for i in range(1, args.num_chunks + 1)
    ]

    judge_kwargs = dict(
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        epochs=args.judge_epochs,
        lr=args.judge_lr,
        patience=args.patience,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
    )

    # ── Phase A ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Phase A: Judge trained on each client's REAL data (local norm)")
    print("=" * 60)
    real_results = phase_a(chunks, **judge_kwargs)

    # ── Phase B ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Phase B: Judge trained on SYNTHETIC data")
    print("=" * 60)
    syn_results = phase_b(
        args.synthetic_csv, chunks,
        target_mean, target_std,
        **judge_kwargs,
    )

    # ── Phase C ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Phase C: Judge trained on ALL data centralized (upper bound)")
    print("=" * 60)
    central_metrics = phase_c(chunks, **judge_kwargs)

    # ── Results ───────────────────────────────────────────────────────────────
    print_results(real_results, syn_results, central_metrics)


if __name__ == "__main__":
    main()
