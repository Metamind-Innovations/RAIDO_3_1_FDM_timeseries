import argparse
import socket
from typing import Dict, List, Tuple

import torch

from comm_utils import bytes_to_state_dict, recv_msg, send_msg, state_dict_to_bytes
from fl_core import (
    MODEL_INPUT_DIM,
    apply_gradient_update,
    fedavg,
    fedsgd,
    merge_client_stats,
    merge_distillation_payloads,
    save_distilled_dataset,
    save_distilled_timeseries_csv,
    train_student_on_distilled,
    weighted_average_metric,
)
from models import LSTMRegressor


def accept_clients(server_sock: socket.socket, num_clients: int) -> List[Tuple[socket.socket, Dict]]:
    clients = []
    while len(clients) < num_clients:
        conn, addr = server_sock.accept()
        msg = recv_msg(conn)
        if msg.get("type") != "register":
            raise ValueError("Expected register message from client.")
        reg = dict(msg)
        reg["addr"] = addr
        clients.append((conn, reg))
        print(
            f"[Aggregator] Client {reg['client_id']} connected from {addr} | "
            f"train_samples={reg['train_samples']} | holdout_samples={reg['holdout_samples']}"
        )
    return clients


def request_client_evaluation(clients, model: LSTMRegressor, eval_label: str) -> List[Dict]:
    state_blob = state_dict_to_bytes(model.state_dict())
    for conn, _ in clients:
        send_msg(conn, {"type": "evaluate", "state_blob": state_blob, "eval_label": eval_label})
    eval_results = []
    for conn, _ in clients:
        reply = recv_msg(conn)
        if reply.get("type") != "eval_result":
            raise ValueError("Expected eval_result from client.")
        eval_results.append(reply)
    return eval_results


def print_client_eval_block(title: str, eval_results: List[Dict]) -> None:
    print(title)
    for item in sorted(eval_results, key=lambda x: int(x["client_id"])):
        cid = int(item["client_id"])
        m = item["metrics"]
        print(f"[Aggregator] Client {cid} | MSE={m['mse']:.6f} | RMSE={m['rmse']:.6f} | MAE={m['mae']:.6f} | R2={m['r2']:.6f}")


def request_distillation_payloads(clients, teacher_model: LSTMRegressor, max_windows_per_client: int, seed: int = 0) -> List[Dict]:
    blob = state_dict_to_bytes(teacher_model.state_dict())
    for conn, _ in clients:
        send_msg(conn, {"type": "distill_request", "state_blob": blob, "max_windows": max_windows_per_client, "seed": seed})
    payloads = []
    for conn, _ in clients:
        reply = recv_msg(conn)
        if reply.get("type") != "distill_result":
            raise ValueError("Expected distill_result from client.")
        payloads.append(reply)
        print(f"[Aggregator] Distillation payload from client {reply['client_id']} | windows={reply['num_samples']}")
    return payloads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--fl_algo", type=str, default="fedavg", choices=["fedavg", "fedsgd", "fedprox"])
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--prox_mu", type=float, default=1e-2)
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--distilled_out", type=str, default="distilled_windows.npz")
    parser.add_argument("--distilled_csv_out", type=str, default="distilled_timeseries.csv")
    parser.add_argument("--distilled_rows", type=int, default=0)
    parser.add_argument("--distilled_batch_size", type=int, default=32)
    parser.add_argument("--distilled_epochs", type=int, default=20)
    parser.add_argument("--distilled_lr", type=float, default=5e-4)
    args = parser.parse_args()

    device = torch.device(args.device)
    global_model = LSTMRegressor(
        input_dim=MODEL_INPUT_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(args.num_clients)

    print(f"[Aggregator] Listening on {args.host}:{args.port}")
    print(f"[Aggregator] FL algorithm: {args.fl_algo}")
    print("[Aggregator] Task: past [%RH + time features] -> next %RH")

    clients = accept_clients(server_sock, args.num_clients)

    print("[Aggregator] Stage-1 local baseline performance on real hold-out")
    baseline_eval_list = []
    for _, reg in sorted(clients, key=lambda x: int(x[1]["client_id"])):
        cid = int(reg["client_id"])
        m = reg["baseline_metrics"]
        baseline_eval_list.append({"client_id": cid, "metrics": m})
        print(f"[Aggregator] Client {cid} | baseline_model={reg['baseline_path']} | RMSE={m['rmse']:.6f} | MAE={m['mae']:.6f} | R2={m['r2']:.6f}")

    stats_list = [reg["stats"] for _, reg in clients]
    merged_stats = merge_client_stats(stats_list)
    target_distilled_rows = args.distilled_rows if args.distilled_rows > 0 else max(1, int(0.5 * merged_stats["total_trainval_rows"]))
    max_windows_per_client = max(1, target_distilled_rows // max(len(clients), 1))
    print(
        f"[Aggregator] Merged stats ready | mean_rh={float(merged_stats['mean'][0]):.4f} | std_rh={float(merged_stats['std'][0]):.4f} | total_trainval_rows={merged_stats['total_trainval_rows']}"
    )
    print(f"[Aggregator] Target distilled windows: {target_distilled_rows} | per-client request: {max_windows_per_client}")

    for rnd in range(1, args.rounds + 1):
        print(f"\n[Aggregator] ===== Stage-2 Round {rnd}/{args.rounds} =====")
        state_blob = state_dict_to_bytes(global_model.state_dict())
        for conn, _ in clients:
            send_msg(
                conn,
                {
                    "type": "train",
                    "round_idx": rnd,
                    "fl_algo": args.fl_algo,
                    "local_epochs": args.local_epochs,
                    "lr": args.lr,
                    "prox_mu": args.prox_mu,
                    "state_blob": state_blob,
                },
            )

        client_payloads = []
        client_weights = []
        payload_kind_seen = None
        for conn, _ in clients:
            reply = recv_msg(conn)
            if reply.get("type") != "train_result":
                raise ValueError("Expected train_result from client.")
            payload_kind = str(reply["payload_kind"])
            if payload_kind_seen is None:
                payload_kind_seen = payload_kind
            elif payload_kind_seen != payload_kind:
                raise ValueError("Mixed payload kinds received from clients.")
            client_payloads.append(bytes_to_state_dict(reply["state_blob"]))
            client_weights.append(int(reply["num_samples"]))
            print(
                f"[Aggregator] Received update from client {reply['client_id']} | samples={reply['num_samples']} | train_loss={float(reply['train_loss']):.6f} | val_loss={float(reply['val_loss']):.6f} | payload={payload_kind}"
            )

        if args.fl_algo in ("fedavg", "fedprox"):
            global_model.load_state_dict(fedavg(client_payloads, client_weights))
        elif args.fl_algo == "fedsgd":
            apply_gradient_update(global_model, fedsgd(client_payloads, client_weights), lr=args.lr)
        else:
            raise ValueError(f"Unsupported FL algorithm: {args.fl_algo}")

        eval_list = request_client_evaluation(clients, global_model, eval_label=f"teacher_round_{rnd}")
        round_metrics = [x["metrics"] for x in eval_list]
        print(
            f"[Aggregator] Round {rnd} teacher on real hold-out | MSE={weighted_average_metric(round_metrics, 'mse'):.6f} | RMSE={weighted_average_metric(round_metrics, 'rmse'):.6f} | MAE={weighted_average_metric(round_metrics, 'mae'):.6f} | R2={weighted_average_metric(round_metrics, 'r2'):.6f}"
        )

    teacher_eval_list = request_client_evaluation(clients, global_model, eval_label="teacher_final")
    teacher_metrics = [x["metrics"] for x in teacher_eval_list]
    print("\n[Aggregator] Stage-2 teacher training complete.")
    print(
        f"[Aggregator] Teacher on real hold-out | MSE={weighted_average_metric(teacher_metrics, 'mse'):.6f} | RMSE={weighted_average_metric(teacher_metrics, 'rmse'):.6f} | MAE={weighted_average_metric(teacher_metrics, 'mae'):.6f} | R2={weighted_average_metric(teacher_metrics, 'r2'):.6f}"
    )
    print_client_eval_block("[Aggregator] Teacher model performance per client", teacher_eval_list)

    payloads = request_distillation_payloads(clients, global_model, max_windows_per_client=max_windows_per_client, seed=42)
    dist_x, dist_y, dist_timestamps, dist_y_denorm = merge_distillation_payloads(payloads, shuffle_seed=42)
    if len(dist_x) > target_distilled_rows:
        dist_x = dist_x[:target_distilled_rows]
        dist_y = dist_y[:target_distilled_rows]
        dist_timestamps = dist_timestamps[:target_distilled_rows]
        dist_y_denorm = dist_y_denorm[:target_distilled_rows]
    save_distilled_dataset(args.distilled_out, dist_x, dist_y)
    save_distilled_timeseries_csv(args.distilled_csv_out, dist_timestamps, dist_y_denorm)
    print(
        f"[Aggregator] Stage-3 distilled datasets saved to: {args.distilled_out} and {args.distilled_csv_out} | windows={len(dist_x)}"
    )

    student_model, distilled_meta = train_student_on_distilled(
        distilled_path=args.distilled_out,
        batch_size=args.distilled_batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=device,
        epochs=args.distilled_epochs,
        lr=args.distilled_lr,
    )
    print(
        f"[Aggregator] Stage-4 student trained on distilled data | windows={distilled_meta['num_rows']} | train_loss={distilled_meta['train_loss']:.6f} | val_loss={distilled_meta['val_loss']:.6f}"
    )
    student_eval_list = request_client_evaluation(clients, student_model, eval_label="student_from_distilled")
    student_metrics = [x["metrics"] for x in student_eval_list]
    print(
        f"[Aggregator] Student on real hold-out | MSE={weighted_average_metric(student_metrics, 'mse'):.6f} | RMSE={weighted_average_metric(student_metrics, 'rmse'):.6f} | MAE={weighted_average_metric(student_metrics, 'mae'):.6f} | R2={weighted_average_metric(student_metrics, 'r2'):.6f}"
    )
    print_client_eval_block("[Aggregator] Distilled/Student model performance per client", student_eval_list)

    print("[Aggregator] Per-client comparison on real hold-out")
    baseline_map = {int(x['client_id']): x['metrics'] for x in baseline_eval_list}
    teacher_map = {int(x['client_id']): x['metrics'] for x in teacher_eval_list}
    student_map = {int(x['client_id']): x['metrics'] for x in student_eval_list}
    for cid in sorted(set(baseline_map) | set(teacher_map) | set(student_map)):
        b = baseline_map.get(cid)
        t = teacher_map.get(cid)
        s = student_map.get(cid)
        print(
            f"[Aggregator] Client {cid} | Baseline RMSE={b['rmse']:.6f}, MAE={b['mae']:.6f}, R2={b['r2']:.6f} || Teacher RMSE={t['rmse']:.6f}, MAE={t['mae']:.6f}, R2={t['r2']:.6f} || Student RMSE={s['rmse']:.6f}, MAE={s['mae']:.6f}, R2={s['r2']:.6f}"
        )

    for conn, _ in clients:
        send_msg(conn, {"type": "shutdown"})
        conn.close()
    server_sock.close()
    print("[Aggregator] Shutdown complete.")


if __name__ == "__main__":
    main()
