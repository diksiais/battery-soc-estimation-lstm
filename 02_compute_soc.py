"""
Step 2: Preprocessing - Compute SOC Ground-Truth & Build Training Dataset
============================================================================
Goal:
1. Loop through all discharge cycles in the NASA battery dataset
2. For each cycle, compute SOC(t) using Coulomb counting
3. Combine everything into one clean dataset: [Voltage, Current, Temperature] -> SOC
4. Save the result as a CSV file, ready for LSTM training (next step)

How to use:
1. Make sure this file is in the same folder as B0005.mat
2. Run: python 02_compute_soc.py
"""

import scipy.io
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
MAT_FILE_PATH = "B0005.mat"
BATTERY_NAME = "B0005"
OUTPUT_CSV = "B0005_soc_dataset.csv"
# ------------------------------------------------------------------


def load_battery_mat(path, name):
    raw = scipy.io.loadmat(path, simplify_cells=True)
    return raw[name]["cycle"]


def compute_soc_for_cycle(cycle_data, cycle_number):
    """
    Compute SOC(t) for a single discharge cycle using Coulomb counting.

    Logic:
    - At t=0 (start of discharge), SOC = 100%
    - As current flows out over time, charge removed = integral of current over time
    - Remaining capacity at time t = total cycle Capacity - charge removed so far
    - SOC(t) = remaining capacity / total cycle Capacity * 100
    """
    time = np.array(cycle_data["Time"]).flatten()
    current = np.array(cycle_data["Current_measured"]).flatten()
    voltage = np.array(cycle_data["Voltage_measured"]).flatten()
    temperature = np.array(cycle_data["Temperature_measured"]).flatten()
    total_capacity = cycle_data["Capacity"]  # Ah, measured at end of this cycle

    # Coulomb counting: cumulative charge removed (Ah) using trapezoidal integration
    # Current is discharge current (positive value flowing out), time is in seconds
    # so convert seconds -> hours for Ah units
    time_hours = time / 3600.0
    charge_removed_ah = np.concatenate((
        [0.0],
        np.cumsum(
            (current[1:] + current[:-1]) / 2 * np.diff(time_hours)
        )
    ))
    # Current in this dataset is stored as negative during discharge in some
    # versions of the NASA set; take absolute value to be safe
    charge_removed_ah = np.abs(charge_removed_ah)

    remaining_capacity = total_capacity - charge_removed_ah
    soc = (remaining_capacity / total_capacity) * 100.0
    soc = np.clip(soc, 0, 100)  # keep within valid range

    df = pd.DataFrame({
        "cycle": cycle_number,
        "time_s": time,
        "voltage": voltage,
        "current": current,
        "temperature": temperature,
        "soc": soc,
    })
    return df


def build_full_dataset(cycles):
    """Loop through all discharge cycles and stack results into one DataFrame."""
    discharge_idxs = [i for i, c in enumerate(cycles) if c["type"] == "discharge"]
    print(f"Processing {len(discharge_idxs)} discharge cycles...")

    all_dfs = []
    skipped = 0
    for cycle_number, idx in enumerate(discharge_idxs):
        try:
            cycle_data = cycles[idx]["data"]
            df = compute_soc_for_cycle(cycle_data, cycle_number)
            all_dfs.append(df)
        except Exception as e:
            # Some cycles occasionally have malformed/missing data - skip them
            skipped += 1
            continue

    print(f"Done. {len(all_dfs)} cycles processed successfully, {skipped} skipped.")
    full_df = pd.concat(all_dfs, ignore_index=True)
    return full_df


if __name__ == "__main__":
    cycles = load_battery_mat(MAT_FILE_PATH, BATTERY_NAME)
    full_dataset = build_full_dataset(cycles)

    print("\nPreview of the final dataset:")
    print(full_dataset.head())
    print(f"\nTotal rows: {len(full_dataset)}")
    print(f"SOC range: {full_dataset['soc'].min():.2f}% to {full_dataset['soc'].max():.2f}%")

    full_dataset.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved dataset to {OUTPUT_CSV}")
