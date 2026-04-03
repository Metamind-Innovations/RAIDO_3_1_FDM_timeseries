"""
local_distill/worker.py
Privacy-preserving client for gradient-only federated distillation.

Key differences from ../gen_worker.py:
  - No Phase 1 (no FedAvg): client trains a local model at startup and
    NEVER sends its weights to the aggregator.
  - distill_round: gradients are computed from the local private model,
    not a shared teacher received from the server.
  - Data slicing: a single museum CSV is split into num_clients sequential
    temporal chunks; --client_id (1-indexed) selects which chunk this
    worker owns.

Only gradients (not weights, not raw data) leave the client.

Start the aggregator first, then launch one worker per client.

Launch (one process per client):
    python local_distill/worker.py \
        --data_path oerholm-indoor-climate-data_2009-2021.csv \
        --client_id 1 --num_clients 3 \
        --host 127.0.0.1 --port 5200 \
        --device cuda
"""

import argparse
import os
import sys
import time
import socket

import numpy as np
import torch

# Allow imports from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comm_utils import bytes_to_state_dict, recv_msg, send_msg, state_dict_to_bytes
from gen_fl_core import (
    MODEL_INPUT_DIM,
    SeqDataset,
    build_client_dataloaders,
    compute_teacher_gradients,
    evaluate_model,
    load_csv,
    make_windows_with_context,
    train_local_model,
    update_discriminator,
)
from gen_models import Discriminator
from models import LSTMRegressor


def slice_df(df, client_id: int, num_clients: int):
    """Return the sequential temporal chunk for this client (1-indexed)."""
    n = len(df)
    start = (client_id - 1) * n // num_clients
    end = client_id * n // num_clients
    return df.iloc[start:end].reset_index(drop=True), start, end - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to museum CSV (full file, sliced by client_id)")
    parser.add_argument("--num_clients", type=int, default=3,
                        help="Total number of clients (= how many temporal chunks)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5200)
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--local_epochs", type=int, default=20,
                        help="Epochs to train the private local model at startup")
    parser.add_argument("--local_lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--disc_lr", type=float, default=2e-4)
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Slice CSV into this client's temporal chunk ───────────────────────────
    import pandas as pd
    from gen_fl_core import add_time_features, split_df, make_windows, TARGET_COL

    df_full = load_csv(args.data_path)
    df_chunk, row_offset, chunk_len = slice_df(df_full, args.client_id, args.num_clients)
    df_chunk = add_time_features(df_chunk)
    print(
        f"[Client {args.client_id}] Chunk {args.client_id}/{args.num_clients} | "
        f"rows {row_offset}–{row_offset + chunk_len - 1} ({chunk_len} rows)"
    )

    train_df, val_df, holdout_df = split_df(df_chunk)
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    local_mean = float(train_df[TARGET_COL].mean())
    local_std = float(max(train_df[TARGET_COL].std(ddof=0), 1e-6))

    x_train, y_train = make_windows(train_df, args.seq_len, local_mean, local_std)
    x_val, y_val = make_windows_with_context(
        train_df, val_df, args.seq_len, local_mean, local_std
    )
    x_hold, y_hold = make_windows_with_context(
        trainval_df, holdout_df, args.seq_len, local_mean, local_std
    )

    from torch.utils.data import DataLoader
    fl_loader = DataLoader(SeqDataset(x_train, y_train),
                           batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SeqDataset(x_val, y_val),
                            batch_size=args.batch_size, shuffle=False)
    holdout_loader = DataLoader(SeqDataset(x_hold, y_hold),
                                batch_size=args.batch_size, shuffle=False)

    meta = {
        "train_samples": len(x_train),
        "val_samples": len(x_val),
        "holdout_samples": len(x_hold),
        "holdout_df": holdout_df,
        "trainval_df": trainval_df,
    }

    print(
        f"[Client {args.client_id}] train={meta['train_samples']} "
        f"val={meta['val_samples']} holdout={meta['holdout_samples']} windows | "
        f"local mean={local_mean:.2f} std={local_std:.2f}"
    )

    # ── Stage 1: Train local model privately (weights never leave) ────────────
    print(f"[Client {args.client_id}] Training private local model...")
    local_model = LSTMRegressor(
        input_dim=MODEL_INPUT_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    local_model, _, _ = train_local_model(
        model=local_model,
        train_loader=fl_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.local_epochs,
        lr=args.local_lr,
        algorithm="fedavg",
        patience=args.patience,
        epoch_callback=lambda e, total, tr, va: print(
            f"[Client {args.client_id}] Local epoch {e}/{total} | "
            f"train={tr:.6f} | val={va:.6f}"
        ),
    )

    baseline_metrics = evaluate_model(local_model, holdout_loader, device,
                                      local_mean, local_std)
    print(
        f"[Client {args.client_id}] Local model done (private) | "
        f"RMSE={baseline_metrics['rmse']:.6f} | R2={baseline_metrics['r2']:.6f}"
    )

    # ── Local Discriminator ───────────────────────────────────────────────────
    discriminator = Discriminator().to(device)
    disc_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=args.disc_lr, betas=(0.5, 0.999)
    )

    # Real series chunks for discriminator
    raw_train_rh = x_train[:, :, 0].flatten() * local_std + local_mean
    n_chunks = max(1, len(raw_train_rh) // 512)
    real_chunks = [raw_train_rh[i * 512: (i + 1) * 512]
                   for i in range(n_chunks) if len(raw_train_rh[i*512:(i+1)*512]) == 512]
    if real_chunks:
        real_chunks_tensor = torch.tensor(
            np.stack(real_chunks), dtype=torch.float32
        ).unsqueeze(-1)
    else:
        tiled = np.tile(raw_train_rh, (512 // len(raw_train_rh) + 1))[:512]
        real_chunks_tensor = torch.tensor(tiled, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    real_chunk_idx = 0

    # ── Connect to aggregator ─────────────────────────────────────────────────
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    max_retries, retry_delay = 30, 5
    for attempt in range(1, max_retries + 1):
        try:
            sock.connect((args.host, args.port))
            print(f"[Client {args.client_id}] Connected to aggregator.")
            break
        except ConnectionRefusedError:
            if attempt == max_retries:
                raise
            print(f"[Client {args.client_id}] Aggregator not ready, "
                  f"retrying in {retry_delay}s ({attempt}/{max_retries})...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            time.sleep(retry_delay)

    # Register — send stats only, NO model weights
    send_msg(sock, {
        "type": "register",
        "client_id": args.client_id,
        "train_samples": meta["train_samples"],
        "holdout_samples": meta["holdout_samples"],
        "baseline_metrics": baseline_metrics,
        "target_mean": local_mean,
        "target_std": local_std,
    })

    # ── Message loop ──────────────────────────────────────────────────────────
    while True:
        msg = recv_msg(sock)
        msg_type = msg.get("type")

        # ── Distillation round — gradients from LOCAL model (private) ─────────
        if msg_type == "distill_round":
            syn_np = np.array(msg["synthetic_series"], dtype=np.float32)
            synthetic_series = torch.tensor(syn_np, dtype=torch.float32)

            # Gradients from local private model on a real batch
            xb_real, yb_real = next(iter(fl_loader))
            local_grads = compute_teacher_gradients(
                local_model, xb_real, yb_real, device
            )
            grads_serialized = {k: v.cpu().tolist() for k, v in local_grads.items()}

            # Discriminator input gradients for adversarial signal
            discriminator.eval()
            syn_leaf = torch.tensor(syn_np, dtype=torch.float32,
                                    device=device, requires_grad=True)
            d_fake_score = discriminator(syn_leaf)
            d_fake_loss_t = -torch.log(d_fake_score + 1e-8).mean()
            d_fake_loss_t.backward()
            d_input_grads = syn_leaf.grad.detach().cpu().tolist()
            d_fake_loss = float(d_fake_loss_t.item())

            # Local discriminator update
            real_batch = real_chunks_tensor[real_chunk_idx % len(real_chunks_tensor)]
            real_batch = real_batch.unsqueeze(0)
            fake_batch = synthetic_series[:1]
            d_loss = update_discriminator(
                discriminator, real_batch, fake_batch, disc_optimizer, device
            )
            real_chunk_idx += 1

            send_msg(sock, {
                "type": "distill_result",
                "client_id": args.client_id,
                "num_samples": meta["train_samples"],
                "teacher_grads": grads_serialized,   # gradients from local model
                "d_fake_loss": d_fake_loss,
                "d_input_grads": d_input_grads,
                "d_loss": d_loss,
            })

        # ── Evaluate any model on holdout ─────────────────────────────────────
        elif msg_type == "evaluate":
            state_dict = bytes_to_state_dict(msg["state_blob"])
            eval_label = str(msg.get("eval_label", "model"))
            eval_model = LSTMRegressor(
                input_dim=MODEL_INPUT_DIM,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
            eval_model.load_state_dict(state_dict)

            if "target_mean" in msg and "target_std" in msg:
                eval_mean = float(msg["target_mean"])
                eval_std = float(msg["target_std"])
                x_h, y_h = make_windows_with_context(
                    meta["trainval_df"], meta["holdout_df"],
                    args.seq_len, eval_mean, eval_std,
                )
                from torch.utils.data import DataLoader as _DL
                eval_loader = _DL(SeqDataset(x_h, y_h),
                                  batch_size=args.batch_size, shuffle=False)
                metrics = evaluate_model(eval_model, eval_loader, device,
                                         eval_mean, eval_std)
            else:
                metrics = evaluate_model(eval_model, holdout_loader, device,
                                         local_mean, local_std)

            send_msg(sock, {
                "type": "eval_result",
                "client_id": args.client_id,
                "metrics": metrics,
                "eval_label": eval_label,
            })

        # ── Shutdown ──────────────────────────────────────────────────────────
        elif msg_type == "shutdown":
            print(f"[Client {args.client_id}] Shutdown received.")
            break
        else:
            raise ValueError(f"[Client {args.client_id}] Unknown message: {msg_type}")

    sock.close()


if __name__ == "__main__":
    main()
