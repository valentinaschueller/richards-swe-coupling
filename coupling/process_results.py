import warnings
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
    with warnings.catch_warnings(action="ignore"):
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
    with warnings.catch_warnings(action="ignore"):
        fig.savefig(target)


def compute_convergence_rate(convergence_log: Path):
    df = pd.read_csv(convergence_log, sep="\s+")
    time_windows = df["TimeWindow"].drop_duplicates().to_numpy()
    cvg_rates = np.zeros(len(time_windows))
    for index, tw in enumerate(time_windows):
        resabs_height = df.loc[df["TimeWindow"] == tw]["ResAbs(Height)"].to_numpy()
        if len(resabs_height) < 3:
            cvg_rates[index] = np.nan
            continue
        resabs_flux = df.loc[df["TimeWindow"] == tw]["ResAbs(Flux)"].to_numpy()
        resabs_total = np.sqrt(resabs_height**2 + resabs_flux**2)
        mean_cvg_rate_total = np.mean(resabs_total[2:] / resabs_total[1:-1])
        cvg_rates[index] = mean_cvg_rate_total

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
