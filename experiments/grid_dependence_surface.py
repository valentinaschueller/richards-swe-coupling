import itertools
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from coupling.analysis import compute_coupling_behavior

plt.style.use("stylesheet.mpl")
L = 10

dt_values = np.logspace(-5, 1, 20)
dz_values = np.logspace(-5, 1, 20)

c = 1
K = 1e-5

if __name__ == "__main__":
    plotting_dir = Path("plots")
    plotting_dir.mkdir(exist_ok=True)

    omega_opt = np.zeros((len(dt_values), len(dz_values)))

    dt_grid, dz_grid = np.meshgrid(dt_values, dz_values)

    np.vectorize(compute_coupling_behavior)

    fig, ax = plt.subplots()
    M_grid = np.round(L / dz_grid)
    _, S, omega = compute_coupling_behavior(K, c, L, dt_grid, M_grid)
    assert np.all(S < 0)

    m = ax.contourf(
        dt_grid,
        dz_grid,
        omega,
        vmin=0,
        vmax=1,
    )
    ax.set(
        xlabel=r"$\Delta t$",
        ylabel=r"$\Delta z$",
        xscale="log",
        # xformatter="sci",
        yscale="log",
        # yformatter="sci",
        title=f"{c=}, {K=}",
    )

    fig.suptitle(r"Optimal relaxation parameter for varying $\Delta t, \Delta z$")
    fig.colorbar(m, label=r"$\omega_\mathrm{opt}$")

    with warnings.catch_warnings(action="ignore"):
        fig.savefig(plotting_dir / "grid_dependence_surface_2.pdf")
