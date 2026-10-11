"""
Step 5b: Fairer Baseline - Coulomb Counting with Realistic Imperfections
============================================================================
The first baseline (05_baseline_comparison.py) was not a fair comparison:
- the SOC labels are themselves Coulomb-counted from the same current signal
- the baseline was given each cycle's TRUE capacity (unknown in a real BMS)
- zero-mean noise barely affects an integrated signal

A real Coulomb-counting BMS only knows:
- a FIXED nominal capacity (here: the capacity measured in the first cycle),
  even though the battery's real capacity fades with aging
- a current sensor with a small constant offset (bias) plus random noise

This script simulates those conditions and evaluates on the same test cycles,
and only on rows from index 19 onward in each cycle (the same rows the LSTM
windows predict), so the comparison with the LSTM is like-for-like.

How to use:
1. Same folder as B0005_soc_dataset.csv
2. Run: python 05b_baseline_comparison_v2.py
"""

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
INPUT_CSV = "B0005_soc_dataset.csv"
TRAIN_TEST_SPLIT = 0.7
WINDOW_SIZE = 20
NOISE_STD = 0.02                       # A, random noise
BIAS_SCENARIOS = [0.0, 0.02, 0.05]     # A, constant current-sensor offset
RANDOM_SEED = 42
LSTM_RMSE, LSTM_MAE = 6.222, 4.701     # from step 4
# ------------------------------------------------------------------


def charge_removed_ah(time_s, current):
    """Cumulative charge removed (Ah) by trapezoidal integration of |current|."""
    t_h = time_s / 3600.0
    q = np.concatenate(([0.0], np.cumsum((current[1:] + current[:-1]) / 2 * np.diff(t_h))))
    return np.abs(q)


def nominal_capacity(df):
    """Capacity of the FIRST cycle = what a BMS would assume as 'nominal'."""
    first = df[df["cycle"] == df["cycle"].min()]
    return charge_removed_ah(first["time_s"].values, first["current"].values)[-1]


def run_baseline(df, test_cycles, c_nominal, bias, noise_std, seed):
    rng = np.random.default_rng(seed)
    all_true, all_base = [], []
    for cid in test_cycles:
        c = df[df["cycle"] == cid].reset_index(drop=True)
        time_s = c["time_s"].values
        current = c["current"].values

        # Sensor model: sign convention of this dataset has discharge current negative,
        # so adding a positive bias to |I| is done by shifting the signed signal.
        measured = current - bias + rng.normal(0, noise_std, size=current.shape)

        q = charge_removed_ah(time_s, measured)
        base_soc = np.clip((c_nominal - q) / c_nominal * 100.0, 0, 100)

        # Only compare rows the LSTM also predicts (index 19 onward)
        all_true.append(c["soc"].values[WINDOW_SIZE - 1:])
        all_base.append(base_soc[WINDOW_SIZE - 1:])
    y_true = np.concatenate(all_true)
    y_base = np.concatenate(all_base)
    rmse = np.sqrt(np.mean((y_base - y_true) ** 2))
    mae = np.mean(np.abs(y_base - y_true))
    return rmse, mae


if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    cycle_ids = sorted(df["cycle"].unique())
    test_cycles = cycle_ids[int(len(cycle_ids) * TRAIN_TEST_SPLIT):]

    c_nom = nominal_capacity(df)
    print(f"Nominal capacity assumed by baseline (first cycle): {c_nom:.3f} Ah")
    last = df[df["cycle"] == cycle_ids[-1]]
    c_last = charge_removed_ah(last["time_s"].values, last["current"].values)[-1]
    print(f"Capacity of the last cycle (after aging):           {c_last:.3f} Ah")
    print(f"Capacity fade by last cycle: {(1 - c_last / c_nom) * 100:.1f}%\n")

    print(f"Test cycles: {len(test_cycles)} | noise std = {NOISE_STD} A\n")
    print(f"{'Bias (A)':<10}{'Baseline RMSE':<18}{'Baseline MAE':<16}")
    for bias in BIAS_SCENARIOS:
        rmse, mae = run_baseline(df, test_cycles, c_nom, bias, NOISE_STD, RANDOM_SEED)
        print(f"{bias:<10}{rmse:<18.3f}{mae:<16.3f}")

    print(f"\nLSTM (step 4):  RMSE {LSTM_RMSE:.3f}%  MAE {LSTM_MAE:.3f}%")
    print("\nNote: the LSTM needs no capacity value and no knowledge of where the")
    print("cycle started; the baseline needs both. Report the result as it is.")
