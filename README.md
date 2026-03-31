# Federated Time-Series Distillation for Univariate `%RH` Forecasting

## Overview

This repository implements a federated learning and distillation pipeline for **univariate time-series regression** on **relative humidity (`%RH`)**.

The system is designed so that:

- each client keeps its raw CSV data locally,
- the aggregator never receives raw records,
- the global **teacher** model is trained through federated learning,
- a distilled dataset is generated from teacher-labeled sequence windows,
- a **student** model is trained on the distilled dataset,
- final performance is evaluated on each client's real unseen hold-out set.

The current task is:

> **Use past `%RH` values and timestamp-derived time features to predict the next-step `%RH` value.**

---

## Repository Files

- `aggregator.py`  
  Central server. Coordinates clients, performs federated aggregation, requests distillation payloads, saves distilled datasets, and trains/evaluates the student model.

- `worker.py`  
  Client-side program. Loads local CSV data, performs local baseline training, participates in federated rounds, evaluates models on the local hold-out set, and sends teacher-labeled distilled windows to the aggregator.

- `fl_core.py`  
  Core data processing and learning utilities:
  - dataset loading and cleaning
  - time-feature generation
  - train/validation/hold-out splitting
  - window creation
  - local training and evaluation
  - federated averaging / FedSGD support
  - distillation payload generation
  - distilled dataset saving
  - student training

- `models.py`  
  Defines the `LSTMRegressor` model.

- `comm_utils.py`  
  Socket communication helpers for sending and receiving Python objects and model state dictionaries.

---

## Dataset

### Expected input format

Each client provides its own local CSV file.

The current pipeline expects at least:

- `DATE AND TIME GMT`
- `%RH`

An extra column such as `TEMP` may be present in the CSV, but it is **not used** in the current univariate pipeline.

### Example dataset structure

A sample uploaded dataset has the following columns:

- `DATE AND TIME GMT`
- `TEMP`
- `%RH`

Example rows:

| DATE AND TIME GMT      | TEMP | %RH |
|------------------------|------|-----|
| 31/12/2008 23:00:00    | 6.6  | 52  |
| 01/01/2009 00:00:00    | 6.5  | 52  |
| 01/01/2009 01:00:00    | 6.4  | 52  |

### Granularity

The implementation preserves the original **1-hour granularity** of the data.

- original client data: 1 hour
- hold-out test data: 1 hour
- final distilled CSV: 1 hour

This is important because changing the time granularity changes what the model can learn.

### What the code actually uses

Even if the original CSV contains multiple columns, the current code extracts only:

- `DATE AND TIME GMT`
- `%RH`

Then it derives the following time features internally from the timestamp:

- `hour_sin`
- `hour_cos`
- `month_sin`
- `month_cos`

So the model input at each timestep is:

- past normalized `%RH`
- hour_sin
- hour_cos
- month_sin
- month_cos

---

## Problem Formulation

This is a **regression** task.

- Input: a past sequence of length `seq_len`
- Output: the next `%RH` value

Because the output is a continuous numeric value, the task is regression rather than classification.

---

## Data Division

Each client's dataset is first:

1. loaded from CSV,
2. cleaned,
3. sorted in time order,
4. split **sequentially**, not randomly.

### Split logic

The code uses:

- **Train+Validation split**: first 70% of the time-ordered dataset
- **Hold-out/Test split**: last 30%

Inside the first 70%:

- **Train split**: first 80%
- **Validation split**: last 20%

So overall the data is split in time order as:

- train: earlier data
- validation: later data within the training range
- hold-out: final unseen future data

This avoids leakage from future data into training.

---

## Sequence Window Construction

The model does not predict from a single row.

Instead, for a sequence length `seq_len` (default `24`), the pipeline constructs sliding windows:

- input window = past `24` timesteps
- target = next-step `%RH`

For validation and hold-out evaluation, the code uses the tail of earlier data as temporal context so that the first hold-out prediction still has a full input history.

---

## Model

The model used in baseline, teacher, and student stages is the same family:

### `LSTMRegressor`

Architecture:
- multi-layer LSTM encoder
- fully connected regression head
- single scalar output

Default configuration:
- `input_dim = 5`
- `hidden_dim = 64`
- `num_layers = 2`
- `dropout = 0.2`

---

## Training Pipeline

### Stage 1: Local baseline training

Each client:

1. loads and preprocesses its local dataset,
2. creates train / validation / hold-out windows,
3. trains a local `LSTMRegressor` on its own training split,
4. uses validation loss for model selection / early stopping,
5. evaluates the local baseline model on its own hold-out set,
6. saves the local baseline model as `M<client_id>.pt`.

This gives a purely local baseline before any federated learning.

---

### Stage 2: Federated teacher training

The aggregator:

1. initializes a global `LSTMRegressor`,
2. sends current model weights to all clients,
3. receives client updates after local training,
4. aggregates them using the chosen FL algorithm,
5. repeats this for multiple communication rounds.

Supported FL algorithms:
- `fedavg`
- `fedsgd`
- `fedprox`

During local federated training, each client uses:
- train split for optimization
- validation split for monitoring / model selection

After the final FL round, the resulting global model is the **teacher model**.

The teacher is then evaluated on each client's real unseen hold-out set.

---

### Stage 3: Distillation dataset generation

After the teacher is formed:

1. the aggregator requests distilled payloads from all clients,
2. each client samples training windows from its local train split,
3. the teacher predicts targets for those windows,
4. the client sends back:
   - window inputs `x`
   - teacher targets `y`
   - denormalized targets
   - timestamps

The aggregator merges all client payloads and creates two distilled outputs:

#### 1. `distilled_windows.npz`
This is the **main distilled training artifact** used for student training.

It stores:
- `x`: teacher-labeled sequence windows
- `y`: teacher targets

#### 2. `distilled_timeseries.csv`
This is a **human-readable distilled dataset** with exactly two columns:

- `DATE AND TIME GMT`
- `%RH`

This CSV preserves the original style of the dataset and 1-hour granularity.

### Distilled dataset size

By default, the target number of distilled windows is:

- **50% of total train+validation rows across clients**

This can be overridden with the `--distilled_rows` argument.

---

### Stage 4: Student training on distilled data

The aggregator trains a **student** model on `distilled_windows.npz`.

The student:
- uses the same `LSTMRegressor` architecture family,
- is trained only on the distilled dataset,
- uses an internal validation split inside the distilled windows.

After training, the student model is sent to each client and evaluated on the client's real hold-out set.

---

## Evaluation

### Per-client evaluation

Evaluation is performed **separately on each client's hold-out test set**.

For each client, the following metrics are computed:

- `MSE`
- `RMSE`
- `MAE`
- `R²`

### Overall evaluation

The aggregator computes the overall performance by taking a **sample-count weighted average** of the client-wise metrics, using the number of hold-out samples of each client as weights.

### Important note

The final model comparison is always done on **real unseen hold-out data**, not on the distilled CSV itself.

This comparison is reported for:

- local baseline model
- federated teacher model
- distilled student model

---

## Distilled Dataset Format Clarification

Two distilled outputs are intentionally kept:

### `distilled_windows.npz`
- used for actual student training
- stores sequence windows directly
- compact and efficient for model training

### `distilled_timeseries.csv`
- human-readable final distilled dataset
- contains only:
  - `DATE AND TIME GMT`
  - `%RH`
- useful for inspection, reporting, and sharing in raw-data style

Both are generated in the current pipeline.

---

## How to Run

### 1. Start the aggregator

Example:

```bash
python3 aggregator.py --host 0.0.0.0 --port 5000 --num_clients 2 --rounds 5 --fl_algo fedavg --local_epochs 10 --lr 0.001 --device cpu --distilled_out distilled_windows.npz --distilled_csv_out distilled_timeseries.csv
```

### 2. Start client 1

```bash
python3 worker.py --data_path /path/to/client1.csv --host <aggregator_ip> --port 5000 --client_id 1 --num_clients 2 --seq_len 24 --batch_size 32 --hidden_dim 64 --num_layers 2 --dropout 0.2 --device cpu --baseline_epochs 20 --baseline_lr 0.001 --patience 5
```

### 3. Start client 2

```bash
python3 worker.py --data_path /path/to/client2.csv --host <aggregator_ip> --port 5000 --client_id 2 --num_clients 2 --seq_len 24 --batch_size 32 --hidden_dim 64 --num_layers 2 --dropout 0.2 --device cpu --baseline_epochs 20 --baseline_lr 0.001 --patience 5
```

If all processes run on the same machine, use:

- aggregator host: `0.0.0.0`
- worker host target: `127.0.0.1`

---

## Outputs

Typical outputs produced by the pipeline:

- `M1.pt`, `M2.pt`, ...  
  local baseline models saved by clients

- `distilled_windows.npz`  
  distilled teacher-labeled sequence windows

- `distilled_timeseries.csv`  
  final distilled CSV with:
  - `DATE AND TIME GMT`
  - `%RH`

---

## Notes

- The current pipeline is **univariate** with `%RH` as both the historical input signal and prediction target.
- `TEMP` is present in the sample raw CSV, but it is not used in the current implementation.
- Distillation is teacher-guided and privacy-preserving, since raw client records are never sent to the aggregator.
- The final student is evaluated on the same real hold-out data used for baseline and teacher comparison.

---

## Summary

This implementation provides a full end-to-end pipeline for:

1. local baseline training,
2. federated teacher training,
3. teacher-guided distilled dataset generation,
4. student training on distilled data,
5. hold-out evaluation on real client data.

It preserves:
- client data privacy,
- time ordering,
- 1-hour granularity,
- human-readable distilled CSV output,
- and an efficient `.npz` training artifact for student learning.