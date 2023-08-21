from pathlib import Path

import numpy as np
import pandas as pd
import proplot as pplt
import xarray as xr


def plot_river(source: Path, target: Path):
    river_output = xr.open_dataarray(source)
    fig, ax = pplt.subplots()
    ax.plot(river_output, marker=".")
    ax.format(
        xlabel="t",
        ylabel="$h(t)$",
        title="River water height over time",
    )
    fig.savefig(target)


def plot_groundwater(source: Path, target: Path):
    groundwater_output = xr.open_dataarray(source)
    fig, ax = pplt.subplots()
    ax.plot(groundwater_output.T, marker=".", cycle="tab20b")
    ax.format(
        xlabel="z",
        ylabel=r"$\psi(z)$",
        title=r"Groundwater potential $\psi$ at different $t$",
    )
    fig.savefig(target)


def compute_convergence_rate(convergence_log: Path):
    df = pd.read_csv(convergence_log, delim_whitespace=True)
    time_windows = df["TimeWindow"].drop_duplicates().to_numpy()
    cvg_rates = np.zeros(len(time_windows))
    for index, tw in enumerate(time_windows):
        resabs_values = df.loc[df["TimeWindow"] == tw]["ResAbs(Height)"].to_numpy()
        if len(resabs_values) == 1:
            cvg_rates[index] = np.nan
            continue
        cvg_rates[index] = np.mean(resabs_values[1:] / resabs_values[:-1])

    cvg_rates = np.ma.masked_invalid(cvg_rates)
    return np.mean(cvg_rates)


if __name__ == "__main__":
    try:
        plot_groundwater("groundwater.nc", "groundwater.png")
    except FileNotFoundError as error:
        print(error)
    try:
        plot_river("river.nc", "river.png")
    except FileNotFoundError as error:
        print(error)
