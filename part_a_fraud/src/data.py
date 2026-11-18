"""Load creditcard.csv, drop exact duplicates, and make the time-based train/val/test split.

Usage (from the repo root):
    python -m part_a_fraud.src.data          # downloads the dataset once, then prints a summary
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CSV_PATH = DATA_DIR / "creditcard.csv"

# ULB "Credit Card Fraud Detection" dataset, mirrored on OpenML.
# Same data as Kaggle's mlg-ulb/creditcardfraud; verified by the checks in `download()`.
OPENML_ID = 1597
EXPECTED_ROWS = 284_807
EXPECTED_FRAUDS = 492

TARGET = "Class"
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def verify(df: pd.DataFrame) -> pd.DataFrame:
    """Fail loudly if this isn't the dataset we think it is (works for the Kaggle or OpenML copy)."""
    assert list(df.columns) == FEATURES + [TARGET], f"unexpected columns: {list(df.columns)}"
    assert len(df) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} rows, got {len(df)}"
    assert df[TARGET].sum() == EXPECTED_FRAUDS, f"expected {EXPECTED_FRAUDS} frauds, got {df[TARGET].sum()}"
    return df


def download(force: bool = False) -> Path:
    """Download the dataset from OpenML to part_a_fraud/data/creditcard.csv (skipped if present).

    Alternative: download archive.zip from Kaggle (mlg-ulb/creditcardfraud) and extract
    creditcard.csv into part_a_fraud/data/.
    """
    if CSV_PATH.exists() and not force:
        return CSV_PATH

    from sklearn.datasets import fetch_openml

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading OpenML dataset {OPENML_ID} ...")
    bunch = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto", data_home=DATA_DIR / ".openml_cache")
    df = bunch.frame
    df[TARGET] = df[TARGET].astype(int)
    df = verify(df[FEATURES + [TARGET]])
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {CSV_PATH} ({CSV_PATH.stat().st_size / 1e6:.0f} MB)")
    return CSV_PATH


def load(drop_duplicates: bool = True) -> pd.DataFrame:
    """Load the dataset sorted by Time. Exact duplicate rows are dropped by default.

    Duplicates must go *before* splitting: an identical row in train and test lets the
    model be scored on data it has effectively memorised.
    """
    df = verify(pd.read_csv(download()))
    if drop_duplicates:
        df = df.drop_duplicates()
    # Stable sort keeps the original order for transactions with the same second.
    return df.sort_values("Time", kind="stable").reset_index(drop=True)


def time_split(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2):
    """Chronological split: first 60% train, next 20% validation, last 20% test. Never shuffled.

    Shuffling would let the model train on transactions that happen after the ones it is
    tested on, which overstates performance compared with real deployment.
    """
    n = len(df)
    i_train, i_val = int(n * train_frac), int(n * (train_frac + val_frac))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


def xy(df: pd.DataFrame):
    """Split a frame into features X and target y."""
    return df[FEATURES], df[TARGET]


if __name__ == "__main__":
    raw = pd.read_csv(download())
    df = load()
    print(f"Raw rows: {len(raw):,} | frauds: {raw[TARGET].sum()} ({raw[TARGET].mean():.3%})")
    print(f"Exact duplicates dropped: {len(raw) - len(df):,} "
          f"(of which fraud: {raw[TARGET].sum() - df[TARGET].sum()})")
    for name, part in zip(["train", "val", "test"], time_split(df)):
        print(f"{name:>5}: {len(part):>7,} rows | {part[TARGET].sum():>3} frauds "
              f"({part[TARGET].mean():.3%}) | Time {part['Time'].min():>9,.0f}-{part['Time'].max():>9,.0f}s")
