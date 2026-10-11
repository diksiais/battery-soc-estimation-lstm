"""
Step 5: Baseline Comparison - Coulomb Counting (with sensor noise) vs LSTM
=============================================================================
Goal:
Demonstrate WHY the LSTM approach adds value, by comparing it against a
classical Coulomb-counting-only baseline under realistic conditions.

Why add noise to the baseline?
The ground-truth SOC itself was computed using Coulomb counting on CLEAN
current readings - so comparing Coulomb counting to itself would trivially
be "perfect" and useless as a baseline. In the real world, current sensors
are never perfectly accurate. We simulate this by adding small random noise
to the current signal, then recomputing SOC via Coulomb counting on this
noisy signal - this represents what a simple, non-ML Battery Management
System (BMS) would actually experience in practice.

How to use:
1. Make sure this file is in the same folder as B0005_soc_dataset.csv
   and the soc_dataset_*.npy files from earlier steps
2. Run: python 05_baseline_comparison.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ------------------------------------------------------------------
INPUT_CSV = "B0005_soc_dataset.csv"
TRAIN_TEST_SPLIT = 0.7
NOISE_STD = 0.02          # simulated current sensor noise, in Amps (std dev)
RANDOM_SEED = 42
# ------------------------------------------------------------------


def compute_coulomb_counting_soc(time_s, current, total_capacity):
    """Same Coulomb counting logic as step 2, but reusable for noisy current."""
    time_hours = time_s / 3600.0
    charge_removed_ah = np.concatenate((
        [0.0],
        np.cumsum((current[1:] + current[:-1]) / 2 * np.diff(time_hours))
    ))
    charge_removed_ah = np.abs(charge_removed_ah)
    remaining_capacity = total_capacity - charge_removed_ah
    soc = np.clip((remaining_capacity / total_capacity) * 100.0, 0, 100)
    return soc


def build_noisy_baseline(df, test_cycles, noise_std, seed):
    """
    For each test cycle, add Gaussian noise to the current readings,
    then recompute SOC using Coulomb counting on the noisy signal.
    Returns arrays of true SOC and noisy-baseline predicted SOC,
    concatenated across all test cycles.
    """
    rng = np.random.default_rng(seed)
    all_true, all_baseline = [], []

    for cid in test_cycles:
        cycle_df = df[df["cycle"] == cid].reset_index(drop=True)
        true_soc = cycle_df["soc"].values
        current_clean = cycle_df["current"].values
        time_s = cycle_df["time_s"].values

        # Total capacity for this cycle = capacity implied by true SOC at t=0
        # (true_soc[0] should be ~100, so we back out total_capacity from
        # the original clean Coulomb-counting computation stored in the CSV)
        # Simpler: recompute total_capacity the same way step 2 did, by using
        # the fact that soc = 100 at row 0 means remaining_capacity = total_capacity there.
        # We approximate total_capacity using the clean current's full integration.
        time_hours = time_s / 3600.0
        clean_charge_removed = np.concatenate((
            [0.0],
            np.cumsum((current_clean[1:] + current_clean[:-1]) / 2 * np.diff(time_hours))
        ))
        clean_charge_removed = np.abs(clean_charge_removed)
        # total_capacity such that soc[0]=100 and soc[-1] matches stored true_soc[-1]
        # total_capacity = charge removed by end / (1 - true_soc[-1]/100)
        denom = (1 - true_soc[-1] / 100.0)
        total_capacity = clean_charge_removed[-1] / denom if denom > 1e-6 else clean_charge_removed[-1]

        # Add realistic sensor noise to current
        noisy_current = current_clean + rng.normal(0, noise_std, size=current_clean.shape)

        baseline_soc = compute_coulomb_counting_soc(time_s, noisy_current, total_capacity)

        all_true.append(true_soc)
        all_baseline.append(baseline_soc)

    return np.concatenate(all_true), np.concatenate(all_baseline)


if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    cycle_ids = sorted(df["cycle"].unique())
    n_train_cycles = int(len(cycle_ids) * TRAIN_TEST_SPLIT)
    test_cycles = cycle_ids[n_train_cycles:]

    print(f"Evaluating baseline on {len(test_cycles)} test cycles "
          f"(same cycles used for LSTM test set)")
    print(f"Simulated current sensor noise: std = {NOISE_STD} A\n")

    y_true, y_baseline = build_noisy_baseline(df, test_cycles, NOISE_STD, RANDOM_SEED)

    baseline_rmse = np.sqrt(np.mean((y_baseline - y_true) ** 2))
    baseline_mae = np.mean(np.abs(y_baseline - y_true))

    print("=== Pure Coulomb Counting (with sensor noise) ===")
    print(f"Baseline RMSE: {baseline_rmse:.3f}%")
    print(f"Baseline MAE:  {baseline_mae:.3f}%")

    print("\n=== LSTM (from step 4, for reference) ===")
    print("LSTM RMSE: 6.222%")
    print("LSTM MAE:  4.701%")

    print("\n=== Comparison ===")
    print(f"RMSE improvement: {baseline_rmse - 6.222:.3f} percentage points "
          f"({'LSTM better' if baseline_rmse > 6.222 else 'Baseline better'})")
    print(f"MAE improvement:  {baseline_mae - 4.701:.3f} percentage points "
          f"({'LSTM better' if baseline_mae > 4.701 else 'Baseline better'})")
