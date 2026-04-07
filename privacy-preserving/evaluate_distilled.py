"""
evaluate_distilled.py
Phase 4: Train a Student on the synthetic dataset, then evaluate it on
real client holdout sets alongside the Baseline and Federated Teacher.

Run AFTER gen_aggregator.py has finished and clients are still waiting,
OR as a standalone script that reconnects to fresh worker processes.

Usage (standalone — start fresh workers first):
    # Terminal 1 (Church):
    python gen_worker.py --data_path <church_csv> --client_id 1 \
        --balanced_len 36842 --port 5101 ...

    # Terminal 2 (Museum):
    python gen_worker.py --data_path <museum_csv> --client_id 2 \
        --balanced_len 36842 --port 5101 ...

    # Terminal 3:
    python evaluate_distilled.py \
        --synthetic_csv synthetic_timeseries.csv \
        --teacher_pt gen_teacher.pt \
        --host 127.0.0.1 --port 5101 \
        --num_clients 2 \
        --balanced_len 36842 \
        --seq_len 24 --batch_size 32 \
        --hidden_dim 64 --num_layers 2 --dropout 0.2 \
        --student_epochs 30 --student_lr 5e-4 \
        --device cpu
"""

import argparse
import socket
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from comm_utils import bytes_to_state_dict, recv_msg, send_msg, state_dict_to_bytes
from gen_fl_core import (
    MODEL_INPUT_DIM,
    SeqDataset,
    load_csv,
    add_time_features,
    split_df,
    make_windows,
    make_windows_with_context,
    train_local_model,
    evaluate_model,
    weighted_average_metric,
    windows_from_synthetic_series,
)
from models import LSTMRegressor


# ── Socket helpers ─────────────────────────────────────────────────────────────

def accept_clients(server_sock: socket.socket, num_clients: int) -> List:
    clients = []
    while len(clients) < num_clients:
        conn, addr = server_sock.accept()
        msg = recv_msg(conn)
        if msg.get("type") != "register":
            raise ValueError("Expected register.")
        reg = dict(msg)
        reg["addr"] = addr
        clients.append((conn, reg))
        print(f"[Eval] Client {reg['client_id']} connected from {addr}")
    return clients


def request_evaluation(
    clients,
    model: LSTMRegressor,
    label: str,
    target_mean: float = None,
    target_std: float = None,
) -> List[Dict]:
    blob = state_dict_to_bytes(model.state_dict())
    msg = {"type": "evaluate", "state_blob": blob, "eval_label": label}
    if target_mean is not None and target_std is not None:
        msg["target_mean"] = float(target_mean)
        msg["target_std"] = float(target_std)
    for conn, _ in clients:
        send_msg(conn, msg)
    results = []
    for conn, _ in clients:
        r = recv_msg(conn)
        results.append(r)
    return results


# ── Student training on synthetic data ────────────────────────────────────────

def train_student_on_synthetic(
    synthetic_csv: str,
    seq_len: int,
    batch_size: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    lr: float,
    patience: int,
    device: torch.device,
    target_mean: float,
    target_std: float,
    val_fraction: float = 0.1,
) -> LSTMRegressor:
    """
    Load synthetic_timeseries.csv, build windows, train Student LSTM from scratch.
    Normalisation uses target_mean/target_std from real training data
    so the student's output space matches the teacher and the holdout evaluator.
    """
    df_syn = load_csv(synthetic_csv)
    df_syn = add_time_features(df_syn)

    from gen_fl_core import TARGET_COL, FEATURE_COLUMNS, _normalize
    import pandas as pd

    # Build windows directly (synthetic has no natural split boundary)
    from gen_fl_core import make_windows as _make_windows
    x_all, y_all = _make_windows(df_syn, seq_len, target_mean, target_std)

    if len(x_all) == 0:
        raise ValueError("Synthetic CSV produced no windows — check the file.")

    n_val = max(1, int(len(x_all) * val_fraction))
    x_train, y_train = x_all[:-n_val], y_all[:-n_val]
    x_val, y_val = x_all[-n_val:], y_all[-n_val:]

    train_loader = DataLoader(SeqDataset(x_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SeqDataset(x_val, y_val),
                            batch_size=batch_size, shuffle=False)

    student = LSTMRegressor(
        input_dim=MODEL_INPUT_DIM,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    print(f"[Eval] Training Student on synthetic data | "
          f"windows={len(x_all)} | train={len(x_train)} | val={len(x_val)}")

    student, train_loss, val_loss = train_local_model(
        model=student,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        algorithm="fedavg",
        patience=patience,
        epoch_callback=lambda e, total, tr, va: print(
            f"  Student epoch {e:3d}/{total} | train={tr:.6f} | val={va:.6f}"
        ) if e % 5 == 0 or e == 1 else None,
    )
    print(f"[Eval] Student training done | "
          f"best_train={train_loss:.6f} | best_val={val_loss:.6f}")
    return student


# ── Pretty comparison table ────────────────────────────────────────────────────

def print_comparison(
    baseline_results: List[Dict],
    teacher_results: List[Dict],
    student_results: List[Dict],
) -> None:
    baseline_map = {int(x["client_id"]): x["metrics"] for x in baseline_results}
    teacher_map = {int(x["client_id"]): x["metrics"] for x in teacher_results}
    student_map = {int(x["client_id"]): x["metrics"] for x in student_results}

    all_cids = sorted(set(baseline_map) | set(teacher_map) | set(student_map))

    header = (
        f"\n{'='*90}\n"
        f"  Per-client comparison on REAL holdout data\n"
        f"{'='*90}\n"
        f"  {'Client':<8} {'Model':<12} {'RMSE':>10} {'MAE':>10} {'R2':>10}\n"
        f"  {'-'*8} {'-'*12} {'-'*10} {'-'*10} {'-'*10}"
    )
    print(header)
    for cid in all_cids:
        for label, mmap in [("Baseline", baseline_map),
                             ("Federated", teacher_map),
                             ("Student", student_map)]:
            m = mmap.get(cid)
            if m:
                print(f"  {cid:<8} {label:<12} "
                      f"{m['rmse']:>10.6f} {m['mae']:>10.6f} {m['r2']:>10.6f}")
        print(f"  {'-'*8} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

    # Weighted aggregates
    def _metrics_with_samples(results):
        return [{"num_samples": x["metrics"]["num_samples"], **x["metrics"]}
                for x in results]

    bm = _metrics_with_samples(baseline_results)
    tm = _metrics_with_samples(teacher_results)
    sm = _metrics_with_samples(student_results)

    print(f"\n  Weighted aggregate across all clients:")
    for label, ml in [("Baseline", bm), ("Federated", tm), ("Student", sm)]:
        print(
            f"  {label:<12} | "
            f"RMSE={weighted_average_metric(ml, 'rmse'):.6f} | "
            f"MAE={weighted_average_metric(ml, 'mae'):.6f} | "
            f"R2={weighted_average_metric(ml, 'r2'):.6f}"
        )
    print(f"{'='*90}")
    print("  Key: Student vs Federated gap = distillation quality.")
    print(f"{'='*90}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_csv", type=str, default="synthetic_timeseries.csv")
    parser.add_argument("--teacher_pt", type=str, default="gen_teacher.pt",
                        help="Path to saved teacher state dict (from gen_aggregator.py)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5101)
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--balanced_len", type=int, default=36842)
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--student_epochs", type=int, default=30)
    parser.add_argument("--student_lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    # Normalisation stats for windowing the synthetic CSV
    # These should match the real training data stats used during distillation.
    # If unknown, the script estimates them from the synthetic CSV itself
    # (less accurate but functional).
    parser.add_argument("--target_mean", type=float, default=None)
    parser.add_argument("--target_std", type=float, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Connect to worker clients ─────────────────────────────────────────────
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(args.num_clients)
    print(f"[Eval] Listening on {args.host}:{args.port} — waiting for {args.num_clients} clients")

    clients = accept_clients(server_sock, args.num_clients)
    client_weights = [int(reg["holdout_samples"]) for _, reg in clients]

    # Baseline metrics already computed by workers in Stage-1
    baseline_results = []
    for _, reg in sorted(clients, key=lambda x: int(x[1]["client_id"])):
        baseline_results.append({
            "client_id": reg["client_id"],
            "metrics": {**reg["baseline_metrics"],
                        "num_samples": reg["holdout_samples"]},
        })

    # ── Load Teacher ──────────────────────────────────────────────────────────
    teacher = LSTMRegressor(
        input_dim=MODEL_INPUT_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    teacher.load_state_dict(torch.load(args.teacher_pt, map_location="cpu"))
    teacher = teacher.to(device)
    print(f"[Eval] Teacher loaded from {args.teacher_pt}")

    teacher_results = request_evaluation(clients, teacher, label="federated_teacher")
    for r in teacher_results:
        r["metrics"]["num_samples"] = r["metrics"].get("num_samples",
                                                        int(client_weights[0]))

    # ── Determine normalisation stats ─────────────────────────────────────────
    import json, os
    stats_path = os.path.join(os.path.dirname(args.teacher_pt), "gen_stats.json")
    if args.target_mean is not None and args.target_std is not None:
        target_mean = args.target_mean
        target_std = args.target_std
        print(f"[Eval] Using provided normalisation: mean={target_mean:.4f} std={target_std:.4f}")
    elif os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        target_mean = float(stats["target_mean"])
        target_std = float(stats["target_std"])
        print(f"[Eval] Normalisation loaded from gen_stats.json: "
              f"mean={target_mean:.4f} std={target_std:.4f}")
    else:
        # Fallback: estimate from synthetic CSV (inaccurate — prefer gen_stats.json)
        df_syn = load_csv(args.synthetic_csv)
        from gen_fl_core import TARGET_COL
        target_mean = float(df_syn[TARGET_COL].mean())
        target_std = float(max(df_syn[TARGET_COL].std(ddof=0), 1e-6))
        print(f"[Eval] WARNING: gen_stats.json not found. "
              f"Normalisation estimated from synthetic CSV: "
              f"mean={target_mean:.4f} std={target_std:.4f}")

    # ── Train Student on synthetic data ───────────────────────────────────────
    student = train_student_on_synthetic(
        synthetic_csv=args.synthetic_csv,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.student_epochs,
        lr=args.student_lr,
        patience=args.patience,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )

    student_results = request_evaluation(clients, student, label="student_distilled",
                                          target_mean=target_mean, target_std=target_std)
    for r in student_results:
        r["metrics"]["num_samples"] = r["metrics"].get("num_samples",
                                                        int(client_weights[0]))

    # ── Print comparison ──────────────────────────────────────────────────────
    print_comparison(baseline_results, teacher_results, student_results)

    # ── Shutdown ──────────────────────────────────────────────────────────────
    for conn, _ in clients:
        send_msg(conn, {"type": "shutdown"})
        conn.close()
    server_sock.close()
    print("[Eval] Done.")


if __name__ == "__main__":
    main()
