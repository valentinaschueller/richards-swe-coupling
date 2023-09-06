from pathlib import Path

import proplot as pplt
import xarray as xr

plotting_dir = Path("plots")
plotting_dir.mkdir(exist_ok=True)

ds = xr.load_dataset(plotting_dir / "cvg_no_relaxation.nc")


fig, axs = pplt.subplots(nrows=1, ncols=2, width="50em", height="25em")
ax = axs[0]
ax.plot(
    ds[r"$\Delta t$"],
    ds[r"CR, $\Delta z=1/20$"],
    color="k",
    marker="x",
    ls="none",
    label="CR",
)
ax.plot(
    ds[r"$\Delta t$"],
    ds[r"|S|, $\Delta z=1/20$"],
    marker="o",
    mfc="none",
    ls="none",
    color="k",
    label="|S|",
)
ax.format(title=r"$\Delta z=1/20$")
ax = axs[1]
ax.plot(
    ds[r"$\Delta t$"],
    ds[r"CR, $\Delta z=1/500$"],
    color="k",
    marker="+",
    ls="none",
    label="CR",
)
ax.plot(
    ds[r"$\Delta t$"],
    ds[r"|S|, $\Delta z=1/500$"],
    marker="D",
    mfc="none",
    ls="none",
    color="k",
    label="|S|",
)
ax.format(title=r"$\Delta z=1/500$")

for ax in axs:
    ax.format(
        xlabel=r"$\Delta t$",
        ylabel="Convergence Rate",
        yscale="log",
        xscale="log",
        xformatter="sci",
        yformatter="sci",
        abc="a)",
    )
    ax.legend(ncols=1)
fig.savefig(plotting_dir / "convergence_no_relaxation.pdf")
