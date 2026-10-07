"""
PROJECT ECHO Signal Processing Pipeline
Loads raw CSI CSV files, extracts environment-independent features using
the shared feature_engine, and creates an ML-ready dataset.

Usage:
    python signal_process.py
"""

import os
import numpy as np
import pandas as pd
import pickle

from feature_engine import extract_features, get_feature_names, FEATURE_COUNT

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "features.pkl")

WINDOW_SIZE = 50
WINDOW_STEP = 10
NUM_SUBCARRIERS = 64


def load_csv(filepath, label):
    try:
        df = pd.read_csv(filepath)
        df["label"] = label
        print(f"  ✓ {label}: {len(df)} samples loaded from {os.path.basename(filepath)}")
        return df
    except Exception as e:
        print(f"  ✗ Error loading {filepath}: {e}")
        return None


def process_dataset(df, label_name):
    amp_cols = [f"amp_{i}" for i in range(NUM_SUBCARRIERS)]
    amplitudes = df[amp_cols].astype(float).values  # Convert to numpy array

    windows = []
    labels = []

    for start in range(0, len(amplitudes) - WINDOW_SIZE, WINDOW_STEP):
        window = amplitudes[start:start + WINDOW_SIZE]
        features = extract_features(window)
        windows.append(features)
        labels.append(label_name)

    return windows, labels


def main():
    print("=" * 60)
    print("PROJECT ECHO — Signal Processing Pipeline v2")
    print("  (Environment-Independent Movement Features)")
    print("=" * 60)

    files = {
        "empty_room": os.path.join(DATA_DIR, "empty_room.csv"),
        "idle": os.path.join(DATA_DIR, "idle.csv"),
        "gesture_halt": os.path.join(DATA_DIR, "gesture_halt.csv"),
        "gesture_advance": os.path.join(DATA_DIR, "gesture_advance.csv"),
    }

    print(f"\n[INFO] Feature engine: {FEATURE_COUNT} movement-only features")
    print(f"[INFO] Window size: {WINDOW_SIZE}, Step: {WINDOW_STEP}")

    print("\n[1/4] Loading CSI data...")
    all_features = []
    all_labels = []

    for label, filepath in files.items():
        if not os.path.exists(filepath):
            print(f"  ⚠ {filepath} not found — skipping. Record this gesture first!")
            continue

        df = load_csv(filepath, label)
        if df is None or len(df) < WINDOW_SIZE:
            print(f"  ⚠ {label}: Not enough samples (need at least {WINDOW_SIZE}). Record more data!")
            continue

        print(f"  [2/4] Extracting movement features for {label}...")
        features, labels = process_dataset(df, label)
        all_features.extend(features)
        all_labels.extend(labels)
        print(f"  ✓ {label}: {len(features)} windows extracted")

    if not all_features:
        print("\n[ERROR] No data to process! Record CSI data first using field_calibrate.py")
        return

    X = np.array(all_features)
    y = np.array(all_labels)

    print(f"\n[3/4] Feature matrix shape: {X.shape}")
    print(f"       Labels: {dict(zip(*np.unique(y, return_counts=True)))}")

    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"  ⚠ Found {nan_count} NaN and {inf_count} Inf values — replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    output = {"X": X, "y": y, "feature_names": get_feature_names()}
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(output, f)

    print(f"\n[4/4] ✓ Saved feature matrix to: {OUTPUT_FILE}")
    print(f"       Total windows: {len(y)}")
    print(f"       Features per window: {X.shape[1]}")
    print("=" * 60)
    print("DONE — Run train_model.py next!")
    print("=" * 60)


if __name__ == "__main__":
    main()
