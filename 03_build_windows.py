"""
Step 3: Build Sliding Windows & Train/Test Split
===================================================
Goal:
1. Load the SOC dataset created in step 2 (B0005_soc_dataset.csv)
2. Normalize voltage, current, temperature to a similar scale (0-1)
3. Build sliding windows PER CYCLE (never crossing cycle boundaries)
4. Split into train/test sets, ready for LSTM training

How to use:
1. Make sure this file is in the same folder as B0005_soc_dataset.csv
2. Run: python 03_build_windows.py
"""

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
INPUT_CSV = "B0005_soc_dataset.csv"
WINDOW_SIZE = 20          # number of past timesteps the model looks at
TRAIN_TEST_SPLIT = 0.7    # 70% of cycles for training, 30% for testing
OUTPUT_PREFIX = "soc_dataset"
# ------------------------------------------------------------------


def normalize_features(df):
    """
    Scale voltage, current, temperature to a 0-1 range.
    We compute min/max across the WHOLE dataset (not per cycle) so that
    the same scaling applies consistently everywhere.
    Note: SOC is already 0-100, we'll just divide by 100 separately.
    """
    feature_cols = ["voltage", "current", "temperature"]
    mins = df[feature_cols].min()
    maxs = df[feature_cols].max()

    df_norm = df.copy()
    for col in feature_cols:
        df_norm[col] = (df[col] - mins[col]) / (maxs[col] - mins[col])

    df_norm["soc"] = df["soc"] / 100.0  # scale SOC to 0-1 too

    print("Normalization ranges used:")
    for col in feature_cols:
        print(f"  {col}: min={mins[col]:.4f}, max={maxs[col]:.4f}")

    return df_norm, mins, maxs


def build_windows_for_cycle(cycle_df, window_size):
    """
    Build sliding windows for a single cycle's data only.
    Returns X (shape: [n_windows, window_size, 3]) and y (shape: [n_windows]).
    """
    features = cycle_df[["voltage", "current", "temperature"]].values
    targets = cycle_df["soc"].values

    X, y = [], []
    n_rows = len(cycle_df)
    if n_rows <= window_size:
        # Cycle too short to build even one window - skip it
        return np.empty((0, window_size, 3)), np.empty((0,))

    for start in range(0, n_rows - window_size):
        end = start + window_size
        X.append(features[start:end])
        y.append(targets[end - 1])  # predict SOC at the last row of the window

    return np.array(X), np.array(y)


def build_all_windows(df_norm, window_size):
    """Loop through each cycle separately and stack all windows together."""
    all_X, all_y = [], []
    cycle_ids = df_norm["cycle"].unique()

    for cid in cycle_ids:
        cycle_df = df_norm[df_norm["cycle"] == cid]
        X, y = build_windows_for_cycle(cycle_df, window_size)
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    return X_all, y_all, cycle_ids


def split_train_test(df_norm, cycle_ids, split_ratio):
    """
    Split by CYCLE, not by row - this avoids leaking information between
    train and test sets (windows from the same cycle should not appear
    in both train and test).
    """
    n_train_cycles = int(len(cycle_ids) * split_ratio)
    train_cycles = cycle_ids[:n_train_cycles]
    test_cycles = cycle_ids[n_train_cycles:]

    train_df = df_norm[df_norm["cycle"].isin(train_cycles)]
    test_df = df_norm[df_norm["cycle"].isin(test_cycles)]

    print(f"\nTrain cycles: {len(train_cycles)} (cycle {train_cycles[0]} to {train_cycles[-1]})")
    print(f"Test cycles: {len(test_cycles)} (cycle {test_cycles[0]} to {test_cycles[-1]})")

    return train_df, test_df


if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    df_norm, mins, maxs = normalize_features(df)

    cycle_ids = sorted(df_norm["cycle"].unique())
    train_df, test_df = split_train_test(df_norm, cycle_ids, TRAIN_TEST_SPLIT)

    print("\nBuilding sliding windows for training set...")
    X_train, y_train, _ = build_all_windows(train_df, WINDOW_SIZE)
    print(f"Training windows: {X_train.shape}, targets: {y_train.shape}")

    print("\nBuilding sliding windows for test set...")
    X_test, y_test, _ = build_all_windows(test_df, WINDOW_SIZE)
    print(f"Test windows: {X_test.shape}, targets: {y_test.shape}")

    np.save(f"{OUTPUT_PREFIX}_X_train.npy", X_train)
    np.save(f"{OUTPUT_PREFIX}_y_train.npy", y_train)
    np.save(f"{OUTPUT_PREFIX}_X_test.npy", X_test)
    np.save(f"{OUTPUT_PREFIX}_y_test.npy", y_test)

    print(f"\nSaved 4 files: {OUTPUT_PREFIX}_X_train.npy, {OUTPUT_PREFIX}_y_train.npy, "
          f"{OUTPUT_PREFIX}_X_test.npy, {OUTPUT_PREFIX}_y_test.npy")
    print("Ready for LSTM training in the next step.")
