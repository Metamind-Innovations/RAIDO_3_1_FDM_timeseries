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

---

## Privacy-Preserving Federated Dataset Distillation

This directory (`privacy-preserving/`) contains an enhanced pipeline that generates **synthetic time series data** using federated learning and adversarial distillation. The goal is to produce a synthetic dataset that can train a student model with performance close to the federated teacher — without sharing any raw client data.

### Architecture Overview

The pipeline uses a **federated teacher + adversarial generator** setup:

```
┌─────────────┐         ┌─────────────┐
│   Client 1  │         │   Client 2  │
│  (Worker)   │         │  (Worker)   │
└──────┬──────┘         └──────┬──────┘
       │                        │
       │  Socket: model weights │
       │  teacher gradients      │
       │  discriminator losses   │
       ▼                        ▼
┌─────────────────────────────────────────┐
│            Aggregator Server            │
│                                         │
│  ┌─────────────┐  ┌──────────────────┐ │
│  │   Teacher   │  │    Generator     │ │
│  │  (LSTM)     │  │  (1D CNN decoder)│ │
│  │  global FL  │  │  z → 512-step    │ │
│  └─────────────┘  └──────────────────┘ │
│                                         │
│  ┌─────────────────────────────────────┐│
│  │  Student (LSTM, reinit + retrain)  ││
│  └─────────────────────────────────────┘│
└─────────────────────────────────────────┘
                    │
                    ▼
         synthetic_timeseries.csv
```

### Models

**Teacher (LSTMRegressor)**
- Identical architecture to the main pipeline: 2-layer LSTM + FC head
- Input: `[%RH + 4 time features] × seq_len`
- Output: next-step `%RH` prediction
- Trained via FedAvg across clients, then frozen

**Generator (1D CNN decoder)**
- Input: random latent vector `z ∈ ℝ¹²⁸`
- Output: synthetic `%RH` series `[batch, 512, 1]`
- Upsamples via transposed convolutions (8 → 512 timesteps)
- Values scaled to `[0, 100]` via Sigmoid

**Discriminator (1D CNN)**
- Input: 512-step `%RH` series `[batch, 512, 1]`
- Output: real/fake score `[batch, 1]`
- Runs **locally on each client** (never shares raw data)
- Standard GAN discriminator trained against Generator output

**Student (LSTMRegressor)**
- Same architecture as Teacher
- Reinitialized and retrained on synthetic data every K distillation iterations
- Uses aggregated normalization stats from all clients

### Data Handling

- Each client loads its own CSV (semicolon-delimited, columns: `DATE AND TIME GMT`, `%RH`)
- Time features derived: `hour_sin`, `hour_cos`, `month_sin`, `month_cos`
- Model input dimension: 5 (`%RH` + 4 time features)
- Sequence length: 24 (default), window = past 24 timesteps → predict next `%RH`
- **Balancing**: all clients trimmed to the shortest dataset length
- Split: 70% train+val, 30% holdout (sequential, no shuffling)

### Distillation Process

The pipeline runs in three phases:

#### Phase 1: Federated Teacher Training

1. Aggregator initializes global Teacher weights
2. Broadcasts to all clients
3. Each client trains locally for `local_epochs` rounds
4. Client updates aggregated via FedAvg
5. Repeat for `fl_rounds` iterations
6. Final Teacher frozen and distributed to all clients

#### Phase 2: Generator Distillation Loop

For each iteration (default 5000):

1. **Generator** produces synthetic 512-step `%RH` series from random `z`
2. **Student** computes gradients on synthetic windows (second-order via `create_graph=True`)
3. Synthetic series sent to all clients
4. Each client:
   - Computes **teacher gradients** on real batch (frozen teacher)
   - Computes **discriminator gradients** w.r.t. synthetic input (adversarial loss)
   - Updates local Discriminator on real vs fake series
5. Aggregator collects and averages:
   - Teacher gradients (weighted by client sample count)
   - Discriminator input gradients (adversarial signal)
6. Generator updates via backprop with loss:
   ```
   L_total = w_gradient × L_gradient
           + w_adversarial × L_adversarial
           + w_smooth × L_smooth
           + w_dist × L_distribution
   ```
   - **L_gradient**: MSE between student grads and aggregated teacher grads (knowledge transfer)
   - **L_adversarial**: Generator tries to fool local discriminators (realistic patterns)
   - **L_smooth**: Penalizes large hour-to-hour `%RH` jumps (temporal plausibility)
   - **L_distribution**: Matches synthetic mean/std to real data statistics
7. Student reinitialized and retrained on current synthetic batch every K iterations

#### Phase 3: Generate Synthetic Dataset

- Generator produces 35% of total balanced rows
- Output: `synthetic_timeseries.csv` (semicolon-delimited, matches real data format)
- Saved alongside: `gen_teacher.pt`, `gen_generator.pt`, `gen_stats.json`

### Evaluation Scripts

**evaluate_distilled.py** — Trains Student on synthetic data, compares:
- Local baseline (client's own model)
- Federated Teacher (real holdout)
- Student distilled (synthetic → real holdout)

**evaluate_judge.py** — Independent evaluation using CNN1DRegressor:
- Phase A: Judge trained on each client's **real** data
- Phase B: Judge trained on **synthetic** data
- Measures RMSE/R2 gap between real vs synthetic training

### Final Results

Results from `results/2_results_5c92a7b939e5e6e8f6b95ffd3a1e37dcb3a56a1e/`:

```
Judge Evaluation (CNN1DRegressor) on REAL holdout data

Weighted average across all clients:
  Real (local)       | RMSE=0.741436 | MAE=0.477088 | R2=0.977241
  Synthetic          | RMSE=0.818900 | MAE=0.502866 | R2=0.980337

  RMSE gap (Synthetic - Real): +0.077464
  R2   gap (Synthetic - Real): +0.003096
```

The small gap indicates high distillation quality — the synthetic data generalizes well to real client holdout sets.

### How to Run

#### Full Pipeline

```bash
# Terminal 1: Client 1 (Church)
python gen_worker.py \
    --data_path /path/to/church.csv \
    --host 127.0.0.1 --port 5100 \
    --client_id 1 --balanced_len 36842 \
    --seq_len 24 --batch_size 32 \
    --hidden_dim 64 --num_layers 2 --dropout 0.2 \
    --device cpu --baseline_epochs 20 --baseline_lr 1e-3

# Terminal 2: Client 2 (Museum)
python gen_worker.py \
    --data_path /path/to/museum.csv \
    --host 127.0.0.1 --port 5100 \
    --client_id 2 --balanced_len 36842 \
    --seq_len 24 --batch_size 32 \
    --hidden_dim 64 --num_layers 2 --dropout 0.2 \
    --device cpu --baseline_epochs 20 --baseline_lr 1e-3

# Terminal 3: Aggregator
python gen_aggregator.py \
    --host 0.0.0.0 --port 5100 \
    --num_clients 2 --balanced_len 36842 \
    --fl_rounds 10 --local_epochs 5 --lr 1e-3 \
    --distill_iters 5000 \
    --student_reinit_every 50 --student_steps 10 \
    --w_gradient 1.0 --w_adversarial 0.1 --w_smooth 0.05 --w_dist 1.0 \
    --synthetic_out synthetic_timeseries.csv \
    --device cuda
```

#### Evaluate Distillation Quality

```bash
# Reconnect to workers for 3-way comparison (baseline, teacher, student)
python evaluate_distilled.py \
    --synthetic_csv synthetic_timeseries.csv \
    --teacher_pt gen_teacher.pt \
    --host 0.0.0.0 --port 5101 \
    --num_clients 2 --balanced_len 36842 \
    --seq_len 24 --batch_size 32 \
    --hidden_dim 64 --num_layers 2 --dropout 0.2 \
    --student_epochs 30 --student_lr 5e-4 \
    --device cpu

# Independent judge evaluation (no workers needed)
python evaluate_judge.py \
    --client_csvs /path/to/church.csv /path/to/museum.csv \
    --synthetic_csv synthetic_timeseries.csv \
    --stats_json gen_stats.json \
    --balanced_len 36842 \
    --seq_len 24 --batch_size 32 \
    --hidden_channels 64 --dropout 0.2 \
    --judge_epochs 30 --judge_lr 5e-4 \
    --device cuda
```

### Output Files

After running `gen_aggregator.py`:
- `synthetic_timeseries.csv` — Synthetic `%RH` time series (35% of balanced data)
- `gen_teacher.pt` — Federated teacher model weights
- `gen_generator.pt` — Generator model weights
- `gen_stats.json` — Aggregated normalization stats (`target_mean`, `target_std`)

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