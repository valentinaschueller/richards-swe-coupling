from pathlib import Path

import numpy as np
import proplot as pplt

from coupling.analysis import compute_coupling_behavior
from coupling.setup_simulation import load_params

if __name__ == "__main__":
    plotting_dir = Path("plots")
    plotting_dir.mkdir(exist_ok=True)

    params = load_params("experiments/varying_physics.yaml")

    assert params.M == 10
    assert params.dz == 1 / 10
    assert params.dt == 1 / 10
    assert params.L == 1
    assert params.N == 1
    assert params.c == 1

    # 1. keep dz constant, decrease dt
    # 2. keep dt/dz constant, decrease dt
    # 3. keep dt/dz^2 constant, decrease dt

    dt_values = 10.0 ** np.arange(-7, 1)

    grid_sizes = np.array(
        [
            len(dt_values) * [params.dz],
            dt_values,
            np.sqrt(dt_values),
        ]
    )

    omega_opt = np.zeros((len(grid_sizes), len(dt_values)))

    for dt_index, dt in enumerate(dt_values):
        params.dt = dt
        for grid_index, grid_size in enumerate(grid_sizes[:, dt_index]):
            params.dz = grid_size
            params.M = round(params.L / grid_size)

            omega_opt[grid_index, dt_index] = compute_coupling_behavior(
                params.K, params.c, params.L, params.dt, params.M
            )[2]

    fig, ax = pplt.subplots(width="30em")
    markers = ["+", "x", "o"]
    labels = [
        r"$\Delta z=1/10$",
        r"$\Delta z=\Delta t$",
        r"$\Delta z = \sqrt{\Delta t}$",
        r"$\Delta z = \sqrt[3]{\Delta t}$",
    ]
    for grid_index, grid_size in enumerate(grid_sizes):
        ax.plot(
            dt_values,
            omega_opt[grid_index],
            marker=markers[grid_index],
            color="k",
            ls="none",
            label=labels[grid_index],
            mfc="none",
        )
    ax.legend(ncols=1)
    ax.format(
        xlabel=r"$\Delta t$",
        ylabel=r"$\omega_\mathrm{opt}$",
        xscale="log",
        xformatter="sci",
        title="Optimal Relaxation Parameter for Varying Grid Setups",
    )
    fig.savefig(plotting_dir / "varying_grids.pdf")
