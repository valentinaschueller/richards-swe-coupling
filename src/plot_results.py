from pathlib import Path

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


if __name__ == "__main__":
    try:
        plot_groundwater("groundwater.nc", "groundwater.png")
    except FileNotFoundError as error:
        print(error)
    try:
        plot_river("river.nc", "river.png")
    except FileNotFoundError as error:
        print(error)
