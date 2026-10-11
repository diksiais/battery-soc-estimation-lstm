"""
Step 1: Initial Exploration of the NASA Battery Dataset
==========================================================
Goal: load one battery .mat file (e.g. B0005.mat), inspect its structure,
and plot voltage/current/temperature for a single discharge cycle.

How to use:
1. Download & unzip the dataset from:
   https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip
2. Find a battery folder (e.g. "B0005") -> inside it there's a B0005.mat file
3. Update MAT_FILE_PATH below to match the file location on your machine
4. Run: python 01_explore_battery_data.py
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# EDIT THIS to match the .mat file location on your machine
MAT_FILE_PATH = "B0005.mat"
BATTERY_NAME = "B0005"
# ------------------------------------------------------------------


def load_battery_mat(path, name):
    """Load a NASA battery dataset .mat file."""
    raw = scipy.io.loadmat(path, simplify_cells=True)
    # NASA dataset structure: raw[name]['cycle'] is a list of cycles
    battery = raw[name]
    cycles = battery["cycle"]
    return cycles


def inspect_structure(cycles):
    """Print a summary of the data structure so you understand its contents."""
    print(f"Total number of cycles (charge+discharge+impedance): {len(cycles)}")
    print("\nFirst 5 cycles, type and ambient temperature:")
    for i, c in enumerate(cycles[:5]):
        print(f"  [{i}] type={c['type']:<12} ambient_temp={c.get('ambient_temperature')}")

    # Find indices of all "discharge" cycles
    discharge_idxs = [i for i, c in enumerate(cycles) if c["type"] == "discharge"]
    print(f"\nNumber of discharge cycles: {len(discharge_idxs)}")
    print(f"Index of first discharge cycle: {discharge_idxs[0]}")

    # Check which fields are available inside discharge data
    first_discharge = cycles[discharge_idxs[0]]["data"]
    print(f"\nFields available in discharge data: {list(first_discharge.keys())}")

    return discharge_idxs


def plot_one_discharge_cycle(cycles, cycle_index):
    """Plot voltage, current, and temperature vs. time for one discharge cycle."""
    data = cycles[cycle_index]["data"]

    time = np.array(data["Time"]).flatten()
    voltage = np.array(data["Voltage_measured"]).flatten()
    current = np.array(data["Current_measured"]).flatten()
    temperature = np.array(data["Temperature_measured"]).flatten()

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(time, voltage, color="tab:blue")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].set_title(f"Discharge cycle #{cycle_index}")

    axes[1].plot(time, current, color="tab:orange")
    axes[1].set_ylabel("Current (A)")

    axes[2].plot(time, temperature, color="tab:red")
    axes[2].set_ylabel("Temperature (°C)")
    axes[2].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig("discharge_cycle_plot.png", dpi=150)
    print("\nPlot saved as discharge_cycle_plot.png")
    plt.show()


if __name__ == "__main__":
    cycles = load_battery_mat(MAT_FILE_PATH, BATTERY_NAME)
    discharge_idxs = inspect_structure(cycles)

    # Plot the first discharge cycle as an example
    plot_one_discharge_cycle(cycles, discharge_idxs[0])
