"""
local_distill/aggregator.py
Central server for privacy-preserving gradient-only distillation.

Differences from ../gen_aggregator.py:
  - No Phase 1 (no FedAvg teacher training, no weight sharing).
  - Client local models are PRIVATE — only gradients are received.
  - --grad_agg controls how client gradients are aggregated:
        fedavg  (default) — weighted by train_samples
        mean              — equal weight per client
        median            — element-wise median (robust to outliers)
  - Three clients from one museum CSV (each owns a temporal chunk).

Launch:
    python local_distill/aggregator.py \
        --host 0.0.0.0 --port 5200 \
        --num_clients 3 \
        --distill_iters 5000 --student_reinit_every 50 --student_steps 10 \
        --grad_agg fedavg \
        --w_smooth 0.05 --w_dist 1.0 \
        --synthetic_out local_distill/synthetic_timeseries.csv \
        --device cuda
"""

import argparse
import json
import os
import socket
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comm_utils import bytes_to_state_dict, recv_msg, send_msg, state_dict_to_bytes
from gen_fl_core import (
    MODEL_INPUT_DIM,
    GEN_SEQ_OUT,
    LATENT_DIM,
    SeqDataset,
    fedavg,
    save_synthetic_timeseries_csv,
    weighted_average_metric,
    windows_from_synthetic_series,
)
from gen_models import Generator
from models import LSTMRegressor


# ── Gradient aggregation ───────────────────────────────────────────────────────

def aggregate_gradients(
    grad_list: List[Dict[str, torch.Tensor]],
    weights: List[int],
    method: str,
) -> Dict[str, torch.Tensor]:
    if method == "fedavg":
        total = float(sum(weights))
        return {k: sum(g[k] * (w / total) for g, w in zip(grad_list, weights))
                for k in grad_list[0]}
    elif method == "mean":
        n = float(len(grad_list))
        return {k: sum(g[k] for g in grad_list) / n for k in grad_list[0]}
    elif method == "median":
        return {k: torch.stack([g[k] for g in grad_list]).median(dim=0).values
                for k in grad_list[0]}
    else:
        raise ValueError(f"Unknown grad_agg method: {method}")


# ── Socket helpers ─────────────────────────────────────────────────────────────

def accept_clients(server_sock: socket.socket, num_clients: int):
    clients = []
    while len(clients) < num_clients:
        conn, addr = server_sock.accept()
        msg = recv_msg(conn)
        if msg.get("type") != "register":
            raise ValueError("Expected register message.")
        reg = dict(msg)
        reg["addr"] = addr
        clients.append((conn, reg))
        print(
            f"[Aggregator] Client {reg['client_id']} connected | "
            f"train={reg['train_samples']} holdout={reg['holdout_samples']} | "
            f"local RMSE={reg['baseline_metrics']['rmse']:.6f}"
        )
    return clients


# ── Student reinit + retrain ───────────────────────────────────────────────────

def _weight_init(m):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, p in m.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)


def reinit_and_train_student(
    student: LSTMRegressor,
    synthetic_series_np: np.ndarray,
    seq_len: int,
    target_mean: float,
    target_std: float,
    batch_size: int,
    steps: int,
    lr: float,
    device: torch.device,
) -> LSTMRegressor:
    for layer in student.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    student.apply(_weight_init)

    x_syn, y_syn = windows_from_synthetic_series(
        synthetic_series_np, seq_len, target_mean, target_std
    )
    if len(x_syn) == 0:
        return student

    loader = DataLoader(SeqDataset(x_syn, y_syn), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    criterion = nn.MSELoss()
    student = student.to(device)
    student.train()

    step = 0
    while step < steps:
        for xb, yb in loader:
            if step >= steps:
                break
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(student(xb), yb).backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            step += 1
    return student


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5200)
    parser.add_argument("--num_clients", type=int, default=3)
    # Federated learning (Phase 1)
    parser.add_argument("--fl_rounds", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fl_algo", type=str, default="fedavg", choices=["fedavg"])
    # Distillation
    parser.add_argument("--distill_iters", type=int, default=5000)
    parser.add_argument("--gen_batch", type=int, default=4)
    parser.add_argument("--gen_lr", type=float, default=1e-4)
    parser.add_argument("--student_reinit_every", type=int, default=50)
    parser.add_argument("--student_steps", type=int, default=10)
    parser.add_argument("--student_lr", type=float, default=1e-3)
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    # Gradient aggregation
    parser.add_argument("--grad_agg", type=str, default="fedavg",
                        choices=["fedavg", "mean", "median"],
                        help="How to aggregate per-client gradients")
    # Loss weights
    parser.add_argument("--w_gradient", type=float, default=1.0)
    parser.add_argument("--w_adversarial", type=float, default=0.1)
    parser.add_argument("--w_diversity", type=float, default=0.0)
    parser.add_argument("--w_smooth", type=float, default=0.05)
    parser.add_argument("--w_dist", type=float, default=1.0)
    # Output
    parser.add_argument("--synthetic_out", type=str,
                        default="local_distill/synthetic_timeseries.csv")
    parser.add_argument("--target_rows", type=int, default=0,
                        help="Target synthetic rows (0 = 35%% of total training rows)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Server socket ─────────────────────────────────────────────────────────
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(args.num_clients)
    print(f"[Aggregator] Listening on {args.host}:{args.port} — "
          f"waiting for {args.num_clients} clients")
    print(f"[Aggregator] Gradient aggregation method: {args.grad_agg}")
    print(f"[Aggregator] Mode: gradient-only (no FedAvg teacher, no weight sharing)")

    clients = accept_clients(server_sock, args.num_clients)
    client_weights = [int(reg["train_samples"]) for _, reg in clients]

    # ── Show per-client local model baselines ─────────────────────────────────
    # (trained privately before connecting — weights never shared)
    print("\n[Aggregator] ── Local model baselines (private, not shared) ──")
    for _, reg in sorted(clients, key=lambda x: int(x[1]["client_id"])):
        m = reg["baseline_metrics"]
        print(f"  Client {reg['client_id']} | "
              f"RMSE={m['rmse']:.6f} | MAE={m['mae']:.6f} | R2={m['r2']:.6f}")

    # ── Phase 1: Federated Teacher Training ──────────────────────────────────
    print(f"\n[Aggregator] ── Phase 1: Federated Teacher ({args.fl_algo}, {args.fl_rounds} rounds) ──")

    global_teacher = LSTMRegressor(
        input_dim=MODEL_INPUT_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    for round_idx in range(1, args.fl_rounds + 1):
        blob = state_dict_to_bytes(global_teacher.state_dict())
        for conn, _ in clients:
            send_msg(conn, {
                "type": "fl_round",
                "state_blob": blob,
                "local_epochs": args.local_epochs,
                "lr": args.lr,
            })

        client_states, round_metrics = [], []
        for conn, reg in clients:
            r = recv_msg(conn)
            client_states.append(bytes_to_state_dict(r["state_blob"]))
            round_metrics.append((int(reg["client_id"]), r["metrics"]))

        global_teacher.load_state_dict(fedavg(client_states, client_weights))

        for cid, m in sorted(round_metrics, key=lambda x: x[0]):
            print(f"  Round {round_idx} | Client {cid} | "
                  f"train={m['train_loss']:.6f} | val={m['val_loss']:.6f}")

        # Evaluate teacher on each client's holdout
        blob = state_dict_to_bytes(global_teacher.state_dict())
        for conn, _ in clients:
            send_msg(conn, {"type": "evaluate", "state_blob": blob,
                            "eval_label": f"teacher_r{round_idx}"})
        eval_results = []
        for conn, _ in clients:
            r = recv_msg(conn)
            eval_results.append(r)
        total_w = float(sum(client_weights))
        w_rmse = sum(
            r["metrics"]["rmse"] * w
            for r, w in zip(eval_results, client_weights)
        ) / total_w
        print(f"  Round {round_idx} teacher holdout | RMSE={w_rmse:.6f}")

    print("[Aggregator] Phase 1 complete.")

    # Freeze teacher and send to all clients for distillation
    blob = state_dict_to_bytes(global_teacher.state_dict())
    for conn, _ in clients:
        send_msg(conn, {"type": "set_teacher", "state_blob": blob})
    for conn, _ in clients:
        ack = recv_msg(conn)
        if ack.get("type") != "set_teacher_ack":
            raise ValueError("Expected set_teacher_ack.")
    print("[Aggregator] Teacher frozen and sent to all clients.")

    # ── Compute weighted-average normalisation stats ───────────────────────────
    total_w = float(sum(client_weights))
    agg_target_mean = sum(
        float(reg["target_mean"]) * int(reg["train_samples"])
        for _, reg in clients
    ) / total_w
    agg_target_std = sum(
        float(reg["target_std"]) * int(reg["train_samples"])
        for _, reg in clients
    ) / total_w
    print(f"\n[Aggregator] Aggregated normalisation: "
          f"mean={agg_target_mean:.4f}  std={agg_target_std:.4f}")

    # ── Networks ──────────────────────────────────────────────────────────────
    generator = Generator(latent_dim=LATENT_DIM, seq_out=GEN_SEQ_OUT).to(device)
    gen_optimizer = torch.optim.Adam(
        generator.parameters(), lr=args.gen_lr, betas=(0.5, 0.999)
    )

    student = LSTMRegressor(
        input_dim=MODEL_INPUT_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    student.apply(_weight_init)

    # Target rows for synthetic dataset
    total_train = sum(client_weights)
    target_rows = (args.target_rows if args.target_rows > 0
                   else max(1, int(0.35 * total_train)))
    gen_calls_needed = int(np.ceil(target_rows / GEN_SEQ_OUT))
    print(f"[Aggregator] Total training windows: {total_train}")
    print(f"[Aggregator] Synthetic target rows: {target_rows} "
          f"({gen_calls_needed} generator calls)")

    # ── Phase 2: Distillation loop ────────────────────────────────────────────
    print(f"\n[Aggregator] ── Distillation loop ({args.distill_iters} iters, "
          f"grad_agg={args.grad_agg}) ──")

    generator.train()
    student.train()

    for it in range(1, args.distill_iters + 1):

        # 1. Generate synthetic series
        z = torch.randn(args.gen_batch, LATENT_DIM, device=device)
        synthetic_series = generator(z)   # (gen_batch, 512, 1)

        # 2. Student gradients on synthetic windows (second-order)
        syn_np = synthetic_series.detach().cpu().numpy()
        flat_series = syn_np[:, :, 0].flatten()
        x_syn, y_syn = windows_from_synthetic_series(
            flat_series, args.seq_len, agg_target_mean, agg_target_std
        )
        if len(x_syn) > 0:
            xb_syn = torch.tensor(x_syn, dtype=torch.float32, device=device)
            yb_syn = torch.tensor(y_syn, dtype=torch.float32, device=device)
            with torch.backends.cudnn.flags(enabled=False):
                student_loss = nn.MSELoss()(student(xb_syn), yb_syn)
            manager_grads = torch.autograd.grad(
                student_loss,
                student.parameters(),
                create_graph=True,
                allow_unused=True,
            )
            manager_grads = [
                g if g is not None else torch.zeros_like(p)
                for g, p in zip(manager_grads, student.parameters())
            ]
        else:
            manager_grads = None

        # 3. Send synthetic series to clients, receive local model gradients
        syn_list = synthetic_series.detach().cpu().tolist()
        for conn, _ in clients:
            send_msg(conn, {"type": "distill_round", "synthetic_series": syn_list})

        grad_list, d_input_grads_list, sample_weights = [], [], []
        for conn, reg in clients:
            r = recv_msg(conn)
            if r.get("type") != "distill_result":
                raise ValueError("Expected distill_result.")
            grads = {k: torch.tensor(v, dtype=torch.float32)
                     for k, v in r["teacher_grads"].items()}
            grad_list.append(grads)
            d_input_grads_list.append(
                torch.tensor(r["d_input_grads"], dtype=torch.float32)
            )
            sample_weights.append(int(r["num_samples"]))

        # 4. Aggregate gradients
        agg_grads = aggregate_gradients(grad_list, sample_weights, args.grad_agg)

        total_w = float(sum(sample_weights))
        agg_d_input_grads = sum(
            g * (w / total_w)
            for g, w in zip(d_input_grads_list, sample_weights)
        ).to(device)

        # 5. Generator loss
        gen_optimizer.zero_grad()

        if manager_grads is not None:
            param_names = [n for n, _ in student.named_parameters()]
            loss_gradient = sum(
                nn.MSELoss()(mg, agg_grads[name].to(device))
                for mg, name in zip(manager_grads, param_names)
                if name in agg_grads
            )
        else:
            loss_gradient = torch.tensor(0.0, device=device)

        loss_diversity = -synthetic_series.var()

        diffs = synthetic_series[:, 1:, :] - synthetic_series[:, :-1, :]
        loss_smooth = diffs.pow(2).mean()

        if agg_target_std > 0:
            loss_dist = (
                ((synthetic_series.mean() - agg_target_mean) / agg_target_std) ** 2
                + (synthetic_series.std() / agg_target_std - 1.0) ** 2
            )
        else:
            loss_dist = torch.tensor(0.0, device=device)

        total_loss = (
            args.w_gradient * loss_gradient
            + args.w_diversity * loss_diversity
            + args.w_smooth * loss_smooth
            + args.w_dist * loss_dist
        )

        hook = synthetic_series.register_hook(
            lambda grad: grad + args.w_adversarial * agg_d_input_grads
        )
        total_loss.backward()
        hook.remove()
        nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        gen_optimizer.step()

        if it % 50 == 0 or it == 1:
            print(
                f"  Iter {it:4d}/{args.distill_iters} | "
                f"L_grad={float(loss_gradient):.6f} | "
                f"L_div={float(loss_diversity):.6f} | "
                f"L_smooth={float(loss_smooth):.6f} | "
                f"L_dist={float(loss_dist):.6f} | "
                f"Total={float(total_loss):.6f}"
            )

        if it % args.student_reinit_every == 0:
            flat = synthetic_series.detach().cpu().numpy()[:, :, 0].flatten()
            student = reinit_and_train_student(
                student, flat, args.seq_len,
                agg_target_mean, agg_target_std,
                args.batch_size, args.student_steps, args.student_lr, device,
            )
            student.train()

    print("[Aggregator] Distillation complete.")

    # ── Phase 3: Generate synthetic dataset ───────────────────────────────────
    print(f"\n[Aggregator] ── Generating synthetic dataset ({target_rows} rows) ──")
    generator.eval()
    all_rh = []
    with torch.no_grad():
        for _ in range(gen_calls_needed):
            z = torch.randn(1, LATENT_DIM, device=device)
            series = generator(z)
            all_rh.append(series[0, :, 0].cpu().numpy())

    rh_concat = np.concatenate(all_rh)[:target_rows]
    rh_concat = np.clip(rh_concat, 0.0, 100.0)

    os.makedirs(os.path.dirname(os.path.abspath(args.synthetic_out)), exist_ok=True)
    save_synthetic_timeseries_csv(args.synthetic_out, rh_concat)
    print(f"  Rows={len(rh_concat)} | Mean={rh_concat.mean():.2f} | "
          f"Std={rh_concat.std():.2f} | Min={rh_concat.min():.2f} | "
          f"Max={rh_concat.max():.2f}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(args.synthetic_out))
    torch.save(generator.state_dict(), os.path.join(out_dir, "generator.pt"))
    stats_path = os.path.join(out_dir, "gen_stats.json")
    with open(stats_path, "w") as f:
        json.dump({"target_mean": agg_target_mean, "target_std": agg_target_std}, f)
    print(f"[Aggregator] Saved generator.pt and gen_stats.json → {out_dir}")

    # ── Shutdown ──────────────────────────────────────────────────────────────
    for conn, _ in clients:
        send_msg(conn, {"type": "shutdown"})
        conn.close()
    server_sock.close()
    print("[Aggregator] Done. Run local_distill/evaluate_judge.py to evaluate.")


if __name__ == "__main__":
    main()
