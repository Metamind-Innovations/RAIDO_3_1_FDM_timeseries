"""
gen_aggregator.py
Central server for synthetic federated dataset distillation.

Phase 1 : Federated teacher training (FedAvg / FedProx)
Phase 2 : Generator distillation loop
            - generate synthetic series
            - gradient matching (student grads vs aggregated teacher grads)
            - federated discriminator adversarial loss
            - diversity, temporal smoothness and distribution matching losses
            - student reinit + retrain every K iterations
Phase 3 : Generate and save the synthetic CSV (35% of balanced data by default)
          plus gen_teacher.pt, gen_generator.pt and gen_stats.json

Launch:
    python gen_aggregator.py \
        --dataset climate \
        --host 0.0.0.0 --port 5100 \
        --num_clients 2 \
        --balanced_len 36842 \
        --fl_rounds 5 --local_epochs 5 --lr 1e-3 \
        --distill_iters 500 --student_reinit_every 50 --student_steps 10 \
        --gen_lr 1e-4 \
        --synthetic_out synthetic_timeseries.csv \
        --device cpu
"""

import argparse
import json
import socket
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from comm_utils import bytes_to_state_dict, recv_msg, send_msg, state_dict_to_bytes
import gen_fl_core
from gen_fl_core import (
    MODEL_INPUT_DIM,
    GEN_SEQ_OUT,
    LATENT_DIM,
    SeqDataset,
    aggregate_teacher_gradients,
    fedavg,
    save_synthetic_timeseries_csv,
    weighted_average_metric,
    windows_from_synthetic_series,
)
from gen_models import Generator
from models import LSTMRegressor


# ── Socket helpers ─────────────────────────────────────────────────────────────

def accept_clients(server_sock: socket.socket, num_clients: int):
    """Wait for num_clients workers to register; returns [(conn, register_msg)]."""
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
            f"train_samples={reg['train_samples']} | "
            f"holdout_samples={reg['holdout_samples']} | "
            f"baseline RMSE={reg['baseline_metrics']['rmse']:.6f}"
        )
    return clients


def request_evaluation(clients, model: LSTMRegressor, label: str) -> List[Dict]:
    """Send model to every client for evaluation on its real holdout; collect results."""
    blob = state_dict_to_bytes(model.state_dict())
    for conn, _ in clients:
        send_msg(conn, {"type": "evaluate", "state_blob": blob, "eval_label": label})
    results = []
    for conn, _ in clients:
        r = recv_msg(conn)
        if r.get("type") != "eval_result":
            raise ValueError("Expected eval_result.")
        results.append(r)
    return results


def print_eval_block(title: str, results: List[Dict]) -> None:
    print(title)
    for item in sorted(results, key=lambda x: int(x["client_id"])):
        m = item["metrics"]
        print(
            f"  Client {item['client_id']} | "
            f"RMSE={m['rmse']:.6f} | MAE={m['mae']:.6f} | R2={m['r2']:.6f}"
        )


# ── Student reinit + retrain ──────────────────────────────────────────────────

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
    """Reinitialise Student weights, then train for `steps` batches on current synthetic windows."""
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


def _weight_init(m):
    """Xavier for Linear/Conv layers, orthogonal for LSTM weights, zero biases."""
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5100)
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--balanced_len", type=int, default=36842,
                        help="Rows per client after balancing (= length of shorter dataset)")
    # FL
    parser.add_argument("--fl_rounds", type=int, default=5)
    parser.add_argument("--fl_algo", type=str, default="fedavg",
                        choices=["fedavg", "fedprox"])
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--prox_mu", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    # Distillation
    parser.add_argument("--distill_iters", type=int, default=500)
    parser.add_argument("--gen_batch", type=int, default=4,
                        help="Number of 512-step series to generate per iteration")
    parser.add_argument("--gen_lr", type=float, default=1e-4)
    parser.add_argument("--student_reinit_every", type=int, default=50)
    parser.add_argument("--student_steps", type=int, default=10)
    parser.add_argument("--student_lr", type=float, default=1e-3)
    parser.add_argument("--w_gradient", type=float, default=1.0)
    parser.add_argument("--w_adversarial", type=float, default=0.1)
    parser.add_argument("--w_diversity", type=float, default=0.0)
    parser.add_argument("--w_smooth", type=float, default=0.05,
                        help="Weight for temporal smoothness loss: penalises large "
                             "step-to-step jumps in the generated series.")
    parser.add_argument("--w_dist", type=float, default=0.5,
                        help="Weight for distribution matching loss: pushes synthetic "
                             "mean and std toward real training data statistics.")
    # Output
    parser.add_argument("--synthetic_out", type=str, default="synthetic_timeseries.csv")
    parser.add_argument("--target_rows", type=int, default=0,
                        help="Target synthetic rows (0 = 35%% of balanced_len * num_clients)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dataset", type=str, default="climate", choices=["climate", "pilot"],
                        help="Data setup: climate (Church/Museum %%RH) or pilot (pilot sensors).")
    args = parser.parse_args()
    gen_fl_core.set_dataset(args.dataset)

    device = torch.device(args.device)

    # Synthetic dataset size: 35% of total balanced rows
    total_balanced = args.balanced_len * args.num_clients
    target_rows = (args.target_rows if args.target_rows > 0
                   else max(1, int(0.35 * total_balanced)))
    # Number of Generator calls needed (each produces GEN_SEQ_OUT=512 rows)
    gen_calls_needed = int(np.ceil(target_rows / GEN_SEQ_OUT))
    print(f"[Aggregator] Total balanced rows: {total_balanced}")
    print(f"[Aggregator] Synthetic target rows: {target_rows}")
    print(f"[Aggregator] Generator calls to reach target: {gen_calls_needed}")

    # ── Networks ──────────────────────────────────────────────────────────────
    global_teacher = LSTMRegressor(
        input_dim=MODEL_INPUT_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    generator = Generator(
        latent_dim=LATENT_DIM,
        seq_out=GEN_SEQ_OUT,
        rh_min=gen_fl_core.DATASET["value_range"][0],
        rh_max=gen_fl_core.DATASET["value_range"][1],
    ).to(device)
    gen_optimizer = torch.optim.Adam(generator.parameters(),
                                     lr=args.gen_lr, betas=(0.5, 0.999))

    student = LSTMRegressor(
        input_dim=MODEL_INPUT_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # ── Server socket ─────────────────────────────────────────────────────────
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(args.num_clients)
    print(f"[Aggregator] Listening on {args.host}:{args.port}")

    clients = accept_clients(server_sock, args.num_clients)
    client_weights = [int(reg["train_samples"]) for _, reg in clients]

    # Print baseline summary
    print("\n[Aggregator] ── Stage-0: Baseline (local models) ──")
    baseline_results = []
    for _, reg in sorted(clients, key=lambda x: int(x[1]["client_id"])):
        m = reg["baseline_metrics"]
        baseline_results.append({
            "client_id": reg["client_id"],
            "metrics": m,
            "num_samples": reg["holdout_samples"],
        })
        print(f"  Client {reg['client_id']} | RMSE={m['rmse']:.6f} | "
              f"MAE={m['mae']:.6f} | R2={m['r2']:.6f}")

    # ── Phase 1: Federated teacher training ───────────────────────────────────
    print(f"\n[Aggregator] ── Phase 1: Federated Teacher ({args.fl_algo}, "
          f"{args.fl_rounds} rounds) ──")

    for rnd in range(1, args.fl_rounds + 1):
        blob = state_dict_to_bytes(global_teacher.state_dict())
        for conn, _ in clients:
            send_msg(conn, {
                "type": "train",
                "round_idx": rnd,
                "fl_algo": args.fl_algo,
                "local_epochs": args.local_epochs,
                "lr": args.lr,
                "prox_mu": args.prox_mu,
                "state_blob": blob,
            })

        payloads, weights = [], []
        for conn, _ in clients:
            r = recv_msg(conn)
            if r.get("type") != "train_result":
                raise ValueError("Expected train_result.")
            payloads.append(bytes_to_state_dict(r["state_blob"]))
            weights.append(int(r["num_samples"]))
            print(
                f"  Round {rnd} | Client {r['client_id']} | "
                f"train={r['train_loss']:.6f} | val={r['val_loss']:.6f}"
            )

        global_teacher.load_state_dict(fedavg(payloads, weights))

        eval_results = request_evaluation(clients, global_teacher,
                                          label=f"teacher_round_{rnd}")
        metrics_list = [x["metrics"] for x in eval_results]
        print(
            f"  Round {rnd} teacher holdout | "
            f"RMSE={weighted_average_metric(metrics_list, 'rmse'):.6f} | "
            f"MAE={weighted_average_metric(metrics_list, 'mae'):.6f} | "
            f"R2={weighted_average_metric(metrics_list, 'r2'):.6f}"
        )

    teacher_eval = request_evaluation(clients, global_teacher, label="teacher_final")
    teacher_metrics_list = [x["metrics"] for x in teacher_eval]
    print("\n[Aggregator] Phase 1 complete — Final teacher on real holdout:")
    print_eval_block("", teacher_eval)
    print(
        f"  Weighted avg | "
        f"RMSE={weighted_average_metric(teacher_metrics_list, 'rmse'):.6f} | "
        f"MAE={weighted_average_metric(teacher_metrics_list, 'mae'):.6f} | "
        f"R2={weighted_average_metric(teacher_metrics_list, 'r2'):.6f}"
    )

    # ── Send frozen teacher to all clients (once) ────────────────────────────
    teacher_blob = state_dict_to_bytes(global_teacher.state_dict())
    for conn, _ in clients:
        send_msg(conn, {"type": "set_teacher", "state_blob": teacher_blob})
    for conn, _ in clients:
        ack = recv_msg(conn)
        if ack.get("type") != "set_teacher_ack":
            raise ValueError("Expected set_teacher_ack.")
    print("[Aggregator] Teacher weights sent to all clients (frozen for Phase 2).")

    # ── Compute weighted-average normalisation stats from registrations ───────
    # Clients sent target_mean/target_std in their register message.
    total_w = float(sum(client_weights))
    agg_target_mean = float(sum(
        float(reg["target_mean"]) * int(reg["train_samples"])
        for _, reg in clients
    ) / total_w)
    agg_target_std = float(sum(
        float(reg["target_std"]) * int(reg["train_samples"])
        for _, reg in clients
    ) / total_w)
    print(f"[Aggregator] Aggregated normalisation stats: "
          f"mean={agg_target_mean:.4f}  std={agg_target_std:.4f}")

    # ── Phase 2: Distillation loop ────────────────────────────────────────────
    print(f"\n[Aggregator] ── Phase 2: Distillation loop ({args.distill_iters} iters) ──")

    generator.train()
    student.train()
    student.apply(_weight_init)

    for it in range(1, args.distill_iters + 1):

        # ── 1-3. Generate synthetic series ──────────────────────────────────
        z = torch.randn(args.gen_batch, LATENT_DIM, device=device)
        synthetic_series = generator(z)   # (gen_batch, 512, 1)

        # ── 4. Student gradients on synthetic windows ────────────────────────
        syn_np = synthetic_series.detach().cpu().numpy()
        flat_series = syn_np[:, :, 0].flatten()  # (gen_batch*512,)
        x_syn, y_syn = windows_from_synthetic_series(
            flat_series, args.seq_len, agg_target_mean, agg_target_std
        )
        if len(x_syn) > 0:
            xb_syn = torch.tensor(x_syn, dtype=torch.float32, device=device)
            yb_syn = torch.tensor(y_syn, dtype=torch.float32, device=device)
            # CuDNN does not support double backwards for LSTMs.
            # Disable it only for this forward pass so create_graph=True works.
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

        # ── 5a-b. Send synthetic series to clients, receive teacher grads + D_fake_loss
        syn_list = synthetic_series.detach().cpu().tolist()
        for conn, _ in clients:
            send_msg(conn, {
                "type": "distill_round",
                "synthetic_series": syn_list,
            })

        teacher_grads_list = []
        d_input_grads_list = []
        sample_weights = []
        for conn, reg in clients:
            r = recv_msg(conn)
            if r.get("type") != "distill_result":
                raise ValueError("Expected distill_result.")
            grads = {k: torch.tensor(v, dtype=torch.float32)
                     for k, v in r["teacher_grads"].items()}
            teacher_grads_list.append(grads)
            d_input_grads_list.append(
                torch.tensor(r["d_input_grads"], dtype=torch.float32)
            )
            sample_weights.append(int(r["num_samples"]))

        # ── 6. Aggregate teacher gradients + discriminator input gradients ─────
        agg_teacher_grads = aggregate_teacher_gradients(teacher_grads_list, sample_weights)

        total_w = float(sum(sample_weights))
        agg_d_input_grads = sum(
            g * (w / total_w)
            for g, w in zip(d_input_grads_list, sample_weights)
        ).to(device)   # (batch, 512, 1)

        # ── 7. Compute Generator loss ─────────────────────────────────────────
        gen_optimizer.zero_grad()

        # Gradient matching loss.
        # NOTE: the synthetic windows in step 4 are built from a detached numpy copy
        # of synthetic_series, so this term currently sends no gradient to the
        # Generator (w_gradient has no effect on training).
        if manager_grads is not None:
            param_names = [n for n, _ in student.named_parameters()]
            loss_gradient = sum(
                nn.MSELoss()(mg, agg_teacher_grads[name].to(device))
                for mg, name in zip(manager_grads, param_names)
                if name in agg_teacher_grads
            )
        else:
            loss_gradient = torch.tensor(0.0, device=device)

        # Diversity loss: negative variance of generated series (keep weight low or 0;
        # high weight causes unrealistic hour-to-hour jumps).
        loss_diversity = -synthetic_series.var()

        # Temporal smoothness: penalise large consecutive step differences,
        # steering the Generator toward physically plausible, slowly varying output.
        diffs = synthetic_series[:, 1:, :] - synthetic_series[:, :-1, :]
        loss_smooth = diffs.pow(2).mean()

        # Distribution matching: push synthetic mean and std toward real data stats.
        # Normalised so the loss is scale-free (~0 when matched, ~1 per unit of std off).
        # This prevents the Generator from collapsing to the wrong mean.
        if agg_target_std and agg_target_std > 0:
            loss_dist = (
                ((synthetic_series.mean() - agg_target_mean) / agg_target_std) ** 2
                + (synthetic_series.std() / agg_target_std - 1.0) ** 2
            )
        else:
            loss_dist = torch.tensor(0.0, device=device)

        total_loss = (args.w_gradient * loss_gradient
                      + args.w_diversity * loss_diversity
                      + args.w_smooth * loss_smooth
                      + args.w_dist * loss_dist)

        # ── 8. Backprop through Generator ─────────────────────────────────────
        # Inject adversarial gradient from client discriminators via hook on
        # synthetic_series: when backward reaches synthetic_series, we add the
        # aggregated d(D_fake_loss)/d(synthetic_series) scaled by w_adversarial.
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

        # ── 9. Student reinit + retrain every K iterations ───────────────────
        if it % args.student_reinit_every == 0:
            syn_np_full = synthetic_series.detach().cpu().numpy()
            flat = syn_np_full[:, :, 0].flatten()
            student = reinit_and_train_student(
                student=student,
                synthetic_series_np=flat,
                seq_len=args.seq_len,
                target_mean=agg_target_mean,
                target_std=agg_target_std,
                batch_size=args.batch_size,
                steps=args.student_steps,
                lr=args.student_lr,
                device=device,
            )
            student.train()
            if it % 100 == 0:
                print(f"  [Student reinit] at iter {it}")

    print("[Aggregator] Phase 2 complete.")

    # ── Phase 3: Generate synthetic CSV ───────────────────────────────────────
    print(f"\n[Aggregator] ── Phase 3: Generating synthetic dataset ──")
    print(f"  Target rows : {target_rows}")
    print(f"  Generator calls: {gen_calls_needed} × {GEN_SEQ_OUT} steps")

    generator.eval()
    all_rh = []
    with torch.no_grad():
        for _ in range(gen_calls_needed):
            z = torch.randn(1, LATENT_DIM, device=device)
            series = generator(z)          # (1, 512, 1)
            all_rh.append(series[0, :, 0].cpu().numpy())

    rh_concat = np.concatenate(all_rh)[:target_rows]   # trim to exact target
    rh_concat = np.clip(rh_concat, *gen_fl_core.DATASET["value_range"])

    save_synthetic_timeseries_csv(args.synthetic_out, rh_concat)
    print(f"[Aggregator] Synthetic dataset saved → {args.synthetic_out}")
    print(f"  Rows: {len(rh_concat)} | "
          f"Mean: {rh_concat.mean():.2f} | "
          f"Std: {rh_concat.std():.2f} | "
          f"Min: {rh_concat.min():.2f} | Max: {rh_concat.max():.2f}")

    # ── Save teacher + generator + normalisation stats ───────────────────────
    torch.save(global_teacher.state_dict(), "gen_teacher.pt")
    torch.save(generator.state_dict(), "gen_generator.pt")
    with open("gen_stats.json", "w") as f:
        json.dump({"target_mean": agg_target_mean, "target_std": agg_target_std}, f)
    print("[Aggregator] Saved gen_teacher.pt, gen_generator.pt, gen_stats.json")

    # ── Shutdown clients ──────────────────────────────────────────────────────
    for conn, _ in clients:
        send_msg(conn, {"type": "shutdown"})
        conn.close()
    server_sock.close()
    print("[Aggregator] Done. Run evaluate_distilled.py to get the 3-way comparison.")


if __name__ == "__main__":
    main()
