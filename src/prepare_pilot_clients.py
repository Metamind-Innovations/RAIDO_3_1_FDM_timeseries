"""
Prepare two client CSVs from the pilot 2025 data for the FDM pipeline.

Usage:
    python src/prepare_pilot_clients.py --input_csv pilot_data/pilot_1_data/Datos_2025_2026/datos_2025.csv

Output:
    pilot_data/pilot_1_data/data_for_demo/client_4728.csv
    pilot_data/pilot_1_data/data_for_demo/client_4730.csv
"""

import argparse
import os
import pandas as pd

CLIENTS = [4728, 4730]
KEEP_ROWS = 150_000
OUT_DIR = "pilot_data/pilot_1_data/data_for_demo"


def prepare_client(df: pd.DataFrame, nup: int) -> pd.DataFrame:
    client = (
        df[df["n_up"] == nup][["f_time", "n_value"]]
        .sort_values("f_time")
        .set_index("f_time")
    )

    # drop duplicate timestamps (keep last)
    client = client[~client.index.duplicated(keep="last")]

    # fill gaps with a full 1-min grid + linear interpolation
    full_idx = pd.date_range(client.index.min(), client.index.max(), freq="1min", name="f_time")
    client = client.reindex(full_idx)
    client["n_value"] = client["n_value"].interpolate(method="linear")

    # keep the most recent 150k points
    client = client.iloc[-KEEP_ROWS:]

    return client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="pilot_data/pilot_1_data/Datos_2025_2026/datos_2025.csv",
        help="Path to datos_2025.csv",
    )
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {args.input_csv} ...")
    df = pd.read_csv(args.input_csv, parse_dates=["f_time"])
    print(f"Loaded {len(df):,} rows, {df['n_up'].nunique()} unique n_up values.")

    for nup in CLIENTS:
        print(f"\nProcessing n_up={nup} ...")
        client_df = prepare_client(df, nup)
        out_path = os.path.join(OUT_DIR, f"client_{nup}.csv")
        client_df.to_csv(out_path, sep=",")
        print(f"  Saved {len(client_df):,} rows → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
