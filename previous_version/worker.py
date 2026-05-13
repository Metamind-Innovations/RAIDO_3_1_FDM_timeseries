import argparse
import os
import sys
import socket

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from comm_utils import bytes_to_state_dict, recv_msg, send_msg, state_dict_to_bytes
from fl_core import (
    MODEL_INPUT_DIM,
    build_client_dataloaders,
    build_teacher_labeled_distillation_payload,
    compute_fedsgd_gradients,
    evaluate_model,
    extract_client_stats_for_aggregator,
    train_local_model,
)
from models import LSTMRegressor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--baseline_epochs", type=int, default=20)
    parser.add_argument("--baseline_lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)

    train_loader, fl_loader, val_loader, holdout_loader, meta = build_client_dataloaders(
        csv_path=args.data_path,
        client_id=args.client_id,
        num_clients=args.num_clients,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )

    baseline_model = LSTMRegressor(
        input_dim=MODEL_INPUT_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    print(
        f"[Client {args.client_id}] Stage-1 baseline training started | "
        f"train_samples={meta['train_samples']} | val_samples={meta['val_samples']} | "
        f"holdout_samples={meta['holdout_samples']}"
    )
    baseline_model, baseline_train_loss, baseline_val_loss = train_local_model(
        model=baseline_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.baseline_epochs,
        lr=args.baseline_lr,
        algorithm="fedavg",
        patience=args.patience,
        epoch_callback=lambda e, total, tr, va: print(
            f"[Client {args.client_id}] Stage-1 epoch {e}/{total} | train_loss={tr:.6f} | val_loss={va:.6f}"
        ),
    )
    baseline_metrics = evaluate_model(
        baseline_model,
        holdout_loader,
        device,
        target_mean=meta["target_mean"],
        target_std=meta["target_std"],
    )
    baseline_path = os.path.abspath(f"M{args.client_id}.pt")
    torch.save(baseline_model.state_dict(), baseline_path)

    print(
        f"[Client {args.client_id}] Stage-1 baseline saved to {baseline_path} | "
        f"train_loss={baseline_train_loss:.6f} | val_loss={baseline_val_loss:.6f} | "
        f"RMSE={baseline_metrics['rmse']:.6f} | MAE={baseline_metrics['mae']:.6f} | "
        f"R2={baseline_metrics['r2']:.6f}"
    )

    local_stats = extract_client_stats_for_aggregator(meta)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    send_msg(
        sock,
        {
            "type": "register",
            "client_id": args.client_id,
            "train_samples": meta["train_samples"],
            "val_samples": meta["val_samples"],
            "holdout_samples": meta["holdout_samples"],
            "baseline_metrics": baseline_metrics,
            "baseline_path": baseline_path,
            "stats": local_stats,
        },
    )

    while True:
        msg = recv_msg(sock)
        msg_type = msg.get("type")

        if msg_type == "train":
            round_idx = int(msg["round_idx"])
            fl_algo = str(msg["fl_algo"]).lower()
            local_epochs = int(msg["local_epochs"])
            lr = float(msg["lr"])
            prox_mu = float(msg.get("prox_mu", 0.0))
            state_dict = bytes_to_state_dict(msg["state_blob"])

            model = LSTMRegressor(
                input_dim=MODEL_INPUT_DIM,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
            model.load_state_dict(state_dict)

            if fl_algo == "fedsgd":
                grad_dict, train_loss = compute_fedsgd_gradients(model=model, loader=fl_loader, device=device)
                payload_blob = state_dict_to_bytes(grad_dict)
                payload_kind = "gradients"
                val_loss = float("nan")
            elif fl_algo in ("fedavg", "fedprox"):
                model, train_loss, val_loss = train_local_model(
                    model=model,
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
                payload_blob = state_dict_to_bytes(model.state_dict())
                payload_kind = "weights"
            else:
                raise ValueError(f"Unsupported FL algorithm on worker: {fl_algo}")

            send_msg(
                sock,
                {
                    "type": "train_result",
                    "client_id": args.client_id,
                    "round_idx": round_idx,
                    "num_samples": meta["train_samples"],
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "payload_kind": payload_kind,
                    "state_blob": payload_blob,
                },
            )
            print(
                f"[Client {args.client_id}] Finished round {round_idx} | algo={fl_algo} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | payload={payload_kind}"
            )

        elif msg_type == "evaluate":
            state_dict = bytes_to_state_dict(msg["state_blob"])
            eval_label = str(msg.get("eval_label", "model"))
            model = LSTMRegressor(
                input_dim=MODEL_INPUT_DIM,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
            model.load_state_dict(state_dict)
            metrics = evaluate_model(
                model,
                holdout_loader,
                device,
                target_mean=meta["target_mean"],
                target_std=meta["target_std"],
            )
            send_msg(sock, {"type": "eval_result", "client_id": args.client_id, "metrics": metrics, "eval_label": eval_label})

        elif msg_type == "distill_request":
            state_dict = bytes_to_state_dict(msg["state_blob"])
            max_windows = int(msg["max_windows"])
            seed = int(msg.get("seed", 0)) + int(args.client_id)
            model = LSTMRegressor(
                input_dim=MODEL_INPUT_DIM,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
            model.load_state_dict(state_dict)
            payload = build_teacher_labeled_distillation_payload(
                train_window_x=meta["train_window_x"],
                train_window_timestamps=meta["train_window_timestamps"],
                teacher_model=model,
                device=device,
                max_windows=max_windows,
                target_mean=meta["target_mean"],
                target_std=meta["target_std"],
                seed=seed,
            )
            send_msg(
                sock,
                {
                    "type": "distill_result",
                    "client_id": args.client_id,
                    "num_samples": int(payload["num_samples"]),
                    "x": payload["x"],
                    "y": payload["y"],
                    "y_denorm": payload["y_denorm"],
                    "timestamps": payload["timestamps"],
                },
            )

        elif msg_type == "shutdown":
            break
        else:
            raise ValueError(f"[Client {args.client_id}] Unknown message type: {msg_type}")

    sock.close()


if __name__ == "__main__":
    main()
