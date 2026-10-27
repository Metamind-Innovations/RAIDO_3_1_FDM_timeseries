"""
gen_worker.py
Client process for synthetic federated dataset distillation.

Phases handled:
  Phase 1 : Federated teacher training (FedAvg / FedProx)
  Phase 2 : Distillation rounds
              - compute teacher gradients on real batch
              - run local Discriminator on synthetic series → D_fake_loss
              - update Discriminator locally
  Evaluate: evaluate any model on holdout and return metrics

Launch (one process per client):
    python gen_worker.py \
        --data_path <path_to_csv> \
        --host 127.0.0.1 --port 5100 \
        --client_id 1 \
        --balanced_len 36842 \
        --seq_len 24 --batch_size 32 \
        --hidden_dim 64 --num_layers 2 --dropout 0.2 \
        --device cpu \
        --baseline_epochs 20 --baseline_lr 1e-3 --patience 5
"""

import argparse
import random

import numpy as np
import torch

from comm_utils import bytes_to_state_dict, recv_msg, send_msg, state_dict_to_bytes
import gen_fl_core
from gen_fl_core import (
    MODEL_INPUT_DIM,
    SeqDataset,
    build_client_dataloaders,
    compute_teacher_gradients,
    evaluate_model,
    make_windows_with_context,
    train_local_model,
    update_discriminator,
)
from gen_models import Discriminator
from models import LSTMRegressor

import socket


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5100)
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--balanced_len", type=int, required=True,
                        help="Number of rows to use (balanced across clients)")
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--baseline_epochs", type=int, default=20)
    parser.add_argument("--baseline_lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--disc_lr", type=float, default=2e-4)
    parser.add_argument("--baseline_pt", type=str, default=None,
                        help="Path to save/load baseline model weights. "
                             "If the file exists, training is skipped and weights are loaded.")
    parser.add_argument("--dataset", type=str, default="climate", choices=["climate", "pilot"],
                        help="Data setup: climate (Church/Museum %%RH) or pilot (pilot sensors).")
    args = parser.parse_args()
    gen_fl_core.set_dataset(args.dataset)

    device = torch.device(args.device)

    # ── Load data ─────────────────────────────────────────────────────────────
    fl_loader, val_loader, holdout_loader, meta = build_client_dataloaders(
        csv_path=args.data_path,
        client_id=args.client_id,
        balanced_len=args.balanced_len,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )
    target_mean = meta["target_mean"]
    target_std = meta["target_std"]

    print(
        f"[Client {args.client_id}] Data loaded | balanced_len={args.balanced_len} | "
        f"train_samples={meta['train_samples']} | val_samples={meta['val_samples']} | "
        f"holdout_samples={meta['holdout_samples']}"
    )

    # ── Stage 1: Local baseline ───────────────────────────────────────────────
    baseline_model = LSTMRegressor(
        input_dim=MODEL_INPUT_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    import os
    if args.baseline_pt and os.path.exists(args.baseline_pt):
        baseline_model.load_state_dict(torch.load(args.baseline_pt, map_location="cpu"))
        baseline_model = baseline_model.to(device)
        print(f"[Client {args.client_id}] Baseline loaded from {args.baseline_pt}")
    else:
        print(f"[Client {args.client_id}] Stage-1 baseline training started")
        baseline_model, bl_train_loss, bl_val_loss = train_local_model(
            model=baseline_model,
            train_loader=fl_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.baseline_epochs,
            lr=args.baseline_lr,
            algorithm="fedavg",
            patience=args.patience,
            epoch_callback=lambda e, total, tr, va: print(
                f"[Client {args.client_id}] Baseline epoch {e}/{total} | "
                f"train={tr:.6f} | val={va:.6f}", flush=True
            ),
        )
        if args.baseline_pt:
            torch.save(baseline_model.state_dict(), args.baseline_pt)
            print(f"[Client {args.client_id}] Baseline saved to {args.baseline_pt}")
    baseline_metrics = evaluate_model(baseline_model, holdout_loader, device,
                                       target_mean, target_std)
    print(
        f"[Client {args.client_id}] Baseline done | "
        f"RMSE={baseline_metrics['rmse']:.6f} | MAE={baseline_metrics['mae']:.6f} | "
        f"R2={baseline_metrics['r2']:.6f}"
    )

    # ── Connect to aggregator (retry until aggregator is ready) ──────────────
    import time
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    max_retries = 30
    retry_delay = 5  # seconds
    for attempt in range(1, max_retries + 1):
        try:
            sock.connect((args.host, args.port))
            print(f"[Client {args.client_id}] Connected to aggregator.")
            break
        except ConnectionRefusedError:
            if attempt == max_retries:
                raise
            print(f"[Client {args.client_id}] Aggregator not ready, "
                  f"retrying in {retry_delay}s (attempt {attempt}/{max_retries})...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            time.sleep(retry_delay)
    send_msg(sock, {
        "type": "register",
        "client_id": args.client_id,
        "train_samples": meta["train_samples"],
        "holdout_samples": meta["holdout_samples"],
        "baseline_metrics": baseline_metrics,
        "target_mean": meta["target_mean"],
        "target_std": meta["target_std"],
    })

    # ── Local teacher model (shared weights, updated each FL round) ───────────
    teacher_model = LSTMRegressor(
        input_dim=MODEL_INPUT_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    # ── Local discriminator ───────────────────────────────────────────────────
    discriminator = Discriminator().to(device)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(),
                                      lr=args.disc_lr, betas=(0.5, 0.999))

    # ── Prebuilt real series batches for Discriminator ────────────────────────
    # We keep the raw %RH values from training split as 512-step chunks
    real_rh_series = torch.tensor(
        meta["train_window_x"][:, :, 0],   # (N_windows, seq_len) normalized %RH
        dtype=torch.float32
    )  # shape (N_windows, seq_len) — will be used in 512-step chunks below

    # Raw (denormalized) training %RH for discriminator real batches
    # Build 512-step chunks from the training data
    raw_train_rh = (meta["train_window_y"] * target_std + target_mean)  # denorm scalar y values
    # Pad or tile to get full 512-step chunks
    n_chunks = max(1, len(raw_train_rh) // 512)
    real_chunks = []
    for i in range(n_chunks):
        chunk = raw_train_rh[i * 512: (i + 1) * 512]
        if len(chunk) == 512:
            real_chunks.append(chunk)
    if real_chunks:
        real_chunks_tensor = torch.tensor(
            np.stack(real_chunks), dtype=torch.float32
        ).unsqueeze(-1)   # (n_chunks, 512, 1)
    else:
        # Fallback: tile y values
        tiled = np.tile(raw_train_rh, (512 // len(raw_train_rh) + 1))[:512]
        real_chunks_tensor = torch.tensor(tiled, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    real_chunk_idx = 0  # cycle through real chunks

    # ── Message loop ──────────────────────────────────────────────────────────
    while True:
        msg = recv_msg(sock)
        msg_type = msg.get("type")

        # ── FL training round ────────────────────────────────────────────────
        if msg_type == "train":
            round_idx = int(msg["round_idx"])
            fl_algo = str(msg["fl_algo"]).lower()
            local_epochs = int(msg["local_epochs"])
            lr = float(msg["lr"])
            prox_mu = float(msg.get("prox_mu", 0.0))
            state_dict = bytes_to_state_dict(msg["state_blob"])

            teacher_model.load_state_dict(state_dict)
            model_copy = LSTMRegressor(
                input_dim=MODEL_INPUT_DIM,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
            model_copy.load_state_dict(state_dict)

            model_copy, train_loss, val_loss = train_local_model(
                model=model_copy,
                train_loader=fl_loader,
                val_loader=val_loader,
                device=device,
                epochs=local_epochs,
                lr=lr,
                algorithm=fl_algo,
                global_state=state_dict if fl_algo == "fedprox" else None,
                prox_mu=prox_mu,
                patience=args.patience,
            )
            send_msg(sock, {
                "type": "train_result",
                "client_id": args.client_id,
                "round_idx": round_idx,
                "num_samples": meta["train_samples"],
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "state_blob": state_dict_to_bytes(model_copy.state_dict()),
            })
            print(
                f"[Client {args.client_id}] FL round {round_idx} | "
                f"train={train_loss:.6f} | val={val_loss:.6f}"
            )

        # ── Receive final teacher weights (frozen for Phase 2) ────────────────
        elif msg_type == "set_teacher":
            state_dict = bytes_to_state_dict(msg["state_blob"])
            teacher_model.load_state_dict(state_dict)
            teacher_model.to(device)
            # Freeze teacher — we only use it for gradient computation
            for p in teacher_model.parameters():
                p.requires_grad_(True)   # needs grad for compute_teacher_gradients
            print(f"[Client {args.client_id}] Teacher weights received and frozen.")
            send_msg(sock, {"type": "set_teacher_ack", "client_id": args.client_id})

        # ── Distillation round ────────────────────────────────────────────────
        elif msg_type == "distill_round":
            # synthetic_series: list → numpy (batch, 512, 1)
            syn_np = np.array(msg["synthetic_series"], dtype=np.float32)
            synthetic_series = torch.tensor(syn_np, dtype=torch.float32)  # (batch,512,1)

            # 5a. Teacher gradients on a real batch
            xb_real, yb_real = next(iter(fl_loader))
            teacher_grads = compute_teacher_gradients(
                teacher_model, xb_real, yb_real, device
            )
            # Serialize grads: {name: list}
            grads_serialized = {k: v.cpu().tolist() for k, v in teacher_grads.items()}

            # 5b. Discriminator input gradients w.r.t. synthetic series.
            # Treat synthetic_series as a leaf with requires_grad=True so we can
            # compute d(D_fake_loss)/d(synthetic_series) and send it back.
            # The aggregator injects these as an external gradient signal on the
            # Generator output via register_hook, making the adversarial loss
            # effectively differentiable through the Generator.
            discriminator.eval()
            syn_leaf = torch.tensor(syn_np, dtype=torch.float32,
                                    device=device, requires_grad=True)
            d_fake_score = discriminator(syn_leaf)
            d_fake_loss_t = -torch.log(d_fake_score + 1e-8).mean()
            d_fake_loss_t.backward()
            d_input_grads = syn_leaf.grad.detach().cpu().tolist()   # (batch,512,1)
            d_fake_loss = float(d_fake_loss_t.item())

            # 5c. Local Discriminator update
            real_batch = real_chunks_tensor[real_chunk_idx % len(real_chunks_tensor)]
            real_batch = real_batch.unsqueeze(0)  # (1, 512, 1)
            # Use first element of synthetic batch as fake sample
            fake_batch = synthetic_series[:1]     # (1, 512, 1)
            d_loss = update_discriminator(
                discriminator, real_batch, fake_batch, disc_optimizer, device
            )
            real_chunk_idx_val = real_chunk_idx + 1

            send_msg(sock, {
                "type": "distill_result",
                "client_id": args.client_id,
                "num_samples": meta["train_samples"],
                "teacher_grads": grads_serialized,
                "d_fake_loss": d_fake_loss,
                "d_input_grads": d_input_grads,
                "d_loss": d_loss,
            })
            real_chunk_idx = real_chunk_idx_val

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

            # If the caller provides normalization stats (e.g. global stats used
            # to train the student), rebuild holdout windows with those stats so
            # that input scaling matches what the model was trained on.
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
                                         target_mean, target_std)

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
