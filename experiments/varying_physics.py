from pathlib import Path

import numpy as np
import proplot as pplt

from coupling.analysis import compute_coupling_behavior
from coupling.process_results import compute_convergence_rate
from coupling.setup_simulation import load_params
from coupling.simulation import run_coupled_simulation

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

    c_values = 10.0 ** np.arange(-5, 6)
    grid_sizes = np.array([1 / 20, 1 / 500])
    omega_opt = np.zeros((len(grid_sizes), len(c_values)))
    S_abs = np.zeros((len(grid_sizes), len(c_values)))
    CR = np.zeros((len(grid_sizes), len(c_values)))

    for grid_index, grid_size in enumerate(grid_sizes):
        params.dz = grid_size
        params.M = round(params.L / grid_size)
        print(params.M)

        for c_index, c in enumerate(c_values):
            params.c = c
            _, S, omega = compute_coupling_behavior(
                params.K, params.c, params.L, params.dt, params.M
            )
            assert S < 0
            omega_opt[grid_index, c_index] = omega
            S_abs[grid_index, c_index] = abs(S)

            run_coupled_simulation(params)
            CR[grid_index, c_index] = compute_convergence_rate(
                "precice-RiverSolver-convergence.log"
            )

    omega_exp = 1 / (1 + CR)

    fig, axs = pplt.subplots(nrows=1, ncols=2, width="50em", height="25em")
    markers_analytical = ["o", "D"]
    markers_experiment = ["x", "+"]
    grid_size_str = ["1/20", "1/500"]
    for grid_index, grid_size in enumerate(grid_sizes):
        ax = axs[grid_index]
        ax.plot(
            c_values,
            omega_opt[grid_index],
            marker=markers_analytical[grid_index],
            color="k",
            ls="none",
            label="theory",
            mfc="none",
        )
        ax.plot(
            c_values,
            omega_exp[grid_index],
            marker=markers_experiment[grid_index],
            color="k",
            ls="none",
            label="experiment",
            mfc="none",
        )
        ax.legend(ncols=1)
        ax.format(
            xlabel="c / K",
            ylabel=r"$\omega_\mathrm{opt}$",
            ylim=[0, 1],
            xscale="log",
            xformatter="sci",
            abc="a)",
            title=rf"$\Delta z$={grid_size_str[grid_index]}",
        )
    fig.savefig(plotting_dir / "varying_physics.pdf")
