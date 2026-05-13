# Federated Time-Series Distillation for Univariate `%RH` Forecasting

## Overview

This repository implements a **privacy-preserving federated distillation pipeline** for univariate time-series regression on **relative humidity (`%RH`)**.

The system is designed so that:

- each client keeps its raw CSV data locally,
- the aggregator never receives raw records,
- the global **teacher** model is trained through federated learning,
- a **generator** produces synthetic `%RH` time series guided by teacher gradients and adversarial feedback from client-side discriminators,
- a **student** model is trained on the synthetic dataset,
- final performance is evaluated on each client's real unseen hold-out set.

The current task is:

> **Use past `%RH` values and timestamp-derived time features to predict the next-step `%RH` value.**

---

## Repository Structure

```
src/                        # Active pipeline (privacy-preserving)
    gen_aggregator.py       # Central server: FL, generator training, synthetic data generation
    gen_worker.py           # Client: local training, discriminator, gradient feedback
    gen_fl_core.py          # Data processing, FL utilities, distillation logic
    gen_models.py           # Generator and Discriminator architectures
    models.py               # LSTMRegressor and CNN1DRegressor
    comm_utils.py           # Socket communication helpers
    evaluate_judge.py       # Independent evaluation using CNN1DRegressor judge

previous_version/           # Original non-generative pipeline (reference only)
    aggregator.py
    worker.py
    fl_core.py
    models.py
```

---

## Dataset

### Expected input format

Each client provides its own local CSV file (semicolon-delimited).

Required columns:

- `DATE AND TIME GMT`
- `%RH`

An extra column such as `TEMP` may be present but is not used.

### Example rows

| DATE AND TIME GMT      | TEMP | %RH |
|------------------------|------|-----|
| 31/12/2008 23:00:00    | 6.6  | 52  |
| 01/01/2009 00:00:00    | 6.5  | 52  |
| 01/01/2009 01:00:00    | 6.4  | 52  |

### Granularity

The implementation preserves the original **1-hour granularity** of the data throughout.

### Features

From the timestamp, the pipeline derives:

- `hour_sin`, `hour_cos`
- `month_sin`, `month_cos`

Model input at each timestep: normalized `%RH` + 4 time features → `input_dim = 5`.

---

## Architecture Overview

```
┌─────────────┐         ┌─────────────┐
│   Client 1  │         │   Client 2  │
│  (Worker)   │         │  (Worker)   │
└──────┬──────┘         └──────┬──────┘
       │                        │
       │  model weights          │
       │  teacher gradients      │
       │  discriminator losses   │
       ▼                        ▼
┌─────────────────────────────────────────┐
│            Aggregator Server            │
│                                         │
│  ┌─────────────┐  ┌──────────────────┐  │
│  │   Teacher   │  │    Generator     │  │
│  │  (LSTM)     │  │  (1D CNN decoder)│  │
│  │  global FL  │  │  z → 512-step    │  │
│  └─────────────┘  └──────────────────┘  │
│                                         │
│  ┌─────────────────────────────────────┐ │
│  │  Student (LSTM, reinit + retrain)  │ │
│  └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
                    │
                    ▼
         synthetic_timeseries.csv
```

---

## Models

**Teacher — `LSTMRegressor`**
- 2-layer LSTM + FC head
- Input: `[%RH + 4 time features] × seq_len`
- Output: next-step `%RH` prediction
- Trained via FedAvg across clients, then frozen

**Generator — 1D CNN decoder**
- Input: random latent vector `z ∈ ℝ¹²⁸`
- Output: synthetic `%RH` series `[batch, 512, 1]`
- Upsamples via transposed convolutions (8 → 512 timesteps)
- Values scaled to `[0, 100]` via Sigmoid

**Discriminator — 1D CNN**
- Input: 512-step `%RH` series `[batch, 512, 1]`
- Output: real/fake score
- Runs locally on each client — raw data never leaves the client

**Student — `LSTMRegressor`**
- Same architecture as Teacher
- Reinitialized and retrained on the current synthetic batch every K distillation iterations

**Judge — `CNN1DRegressor`**
- Used only in `evaluate_judge.py`
- Independent evaluator: trained once on real data, once on synthetic data
- Measures the RMSE/R² gap between real and synthetic training

---

## Data Handling

- Split: 70% train+val / 30% holdout (sequential, no shuffling)
- Inside train+val: 80% train / 20% validation
- **Balancing**: all clients trimmed to the shortest dataset length (`--balanced_len`)
- Sequence length: 24 (default) — sliding windows of past 24 timesteps → predict next `%RH`

---

## Training Pipeline

### Phase 1: Federated Teacher Training

1. Aggregator initializes global Teacher weights
2. Broadcasts to all clients
3. Each client trains locally for `local_epochs` rounds
4. Client updates aggregated via FedAvg (or FedSGD / FedProx)
5. Repeat for `fl_rounds` iterations
6. Final Teacher frozen and distributed to all clients

### Phase 2: Generator Distillation Loop

For each iteration (default 500):

1. Generator produces synthetic 512-step `%RH` series from random `z`
2. Student computes gradients on synthetic windows (`create_graph=True`)
3. Synthetic series sent to all clients
4. Each client:
   - Computes teacher gradients on a real batch (frozen teacher)
   - Computes discriminator gradients w.r.t. synthetic input (adversarial signal)
   - Updates local Discriminator on real vs. fake series
5. Aggregator collects and averages teacher gradients and discriminator input gradients
6. Generator loss:
   ```
   L_total = w_gradient   × L_gradient      # student grads match teacher grads
           + w_adversarial × L_adversarial   # fool local discriminators
           + w_smooth      × L_smooth        # penalise large hour-to-hour jumps
           + w_dist        × L_distribution  # match synthetic mean/std to real stats
   ```
7. Student reinitialized and retrained on current synthetic batch every `student_reinit_every` steps

### Phase 3: Generate Synthetic Dataset

- Generator produces synthetic `%RH` rows (default: `target_rows = 0` → auto-sized)
- Output: `synthetic_timeseries.csv` (semicolon-delimited, matches real data format)
- Saved alongside: `gen_teacher.pt`, `gen_generator.pt`, `gen_stats.json`

---

## Evaluation

### `evaluate_judge.py` — Independent judge evaluation (no workers needed)

Uses `CNN1DRegressor` as an independent judge:

- **Phase A**: Judge trained on each client's **real** data → evaluated on real holdout
- **Phase B**: Judge trained on **synthetic** data → evaluated on real holdout
- Reports RMSE / MAE / R² for both phases and the gap between them

A small gap indicates high distillation quality.

### Metrics reported

- `MSE`, `RMSE`, `MAE`, `R²`
- Per-client and weighted average across all clients

---

## Latest Results

```
Judge Evaluation (CNN1DRegressor) on REAL holdout data

Weighted average across all clients:
  Real (local)       | RMSE=0.741436 | MAE=0.477088 | R2=0.977241
  Synthetic          | RMSE=0.818900 | MAE=0.502866 | R2=0.980337

  RMSE gap (Synthetic - Real): +0.077464
  R2   gap (Synthetic - Real): +0.003096
```

---

## How to Run

All scripts are in `src/`. Run from inside that directory or prefix paths accordingly.

### 1. Start the workers (one per client)

```bash
# Client 1
python src/gen_worker.py \
    --data_path /path/to/church.csv \
    --host 127.0.0.1 --port 5100 \
    --client_id 1 --balanced_len 36842 \
    --seq_len 24 --batch_size 32 \
    --hidden_dim 64 --num_layers 2 --dropout 0.2 \
    --device cpu --baseline_epochs 20 --baseline_lr 1e-3

# Client 2
python src/gen_worker.py \
    --data_path /path/to/museum.csv \
    --host 127.0.0.1 --port 5100 \
    --client_id 2 --balanced_len 36842 \
    --seq_len 24 --batch_size 32 \
    --hidden_dim 64 --num_layers 2 --dropout 0.2 \
    --device cpu --baseline_epochs 20 --baseline_lr 1e-3
```

### 2. Start the aggregator

```bash
python src/gen_aggregator.py \
    --host 0.0.0.0 --port 5100 \
    --num_clients 2 --balanced_len 36842 \
    --fl_rounds 10 --local_epochs 5 --lr 1e-3 \
    --distill_iters 500 \
    --student_reinit_every 50 --student_steps 10 \
    --w_gradient 1.0 --w_adversarial 0.1 --w_smooth 0.05 --w_dist 0.5 \
    --synthetic_out synthetic_timeseries.csv \
    --device cpu
```

### 3. Evaluate with the judge (no workers needed)

```bash
python src/evaluate_judge.py \
    --client_csvs /path/to/church.csv /path/to/museum.csv \
    --synthetic_csv synthetic_timeseries.csv \
    --stats_json gen_stats.json \
    --balanced_len 36842 \
    --seq_len 24 --batch_size 32 \
    --hidden_channels 64 --dropout 0.2 \
    --judge_epochs 30 --judge_lr 5e-4 \
    --device cpu
```

---

## Output Files

After running `gen_aggregator.py`:

- `synthetic_timeseries.csv` — synthetic `%RH` time series
- `gen_teacher.pt` — federated teacher model weights
- `gen_generator.pt` — generator model weights
- `gen_stats.json` — aggregated normalization stats (`target_mean`, `target_std`)
