import itertools
import warnings
from pathlib import Path

import numpy as np
import proplot as pplt

from coupling.analysis import compute_coupling_behavior
from coupling.process_results import compute_convergence_rate
from coupling.setup_simulation import load_params
from coupling.simulation import run_coupled_simulation

L = 1
N = 1
t_end = 0.1
t_start = 0.0
dt = (t_end - t_start) / N

c_values = 10.0 ** np.linspace(-3, 3, 50)
K_values = 10.0 ** np.linspace(-3, 3, 30)
grid_sizes = np.array([1 / 20, 1 / 500])

if __name__ == "__main__":
    plotting_dir = Path("plots")
    plotting_dir.mkdir(exist_ok=True)

    omega_opt = np.zeros((len(grid_sizes), len(c_values), len(K_values)))
    S_abs = omega_opt.copy()
    CR = omega_opt.copy()

    c_grid, K_grid = np.meshgrid(c_values, K_values)

    np.vectorize(compute_coupling_behavior)

    fig, axs = pplt.subplots(nrows=1, ncols=2, width="70em", height="40em")
    grid_size_str = ["1/20", "1/500"]
    for g_index, grid_size in enumerate(grid_sizes):
        dz = grid_size
        M = round(L / grid_size)
        _, S, omega = compute_coupling_behavior(K_grid, c_grid, L, dt, M)

        assert np.all(S < 0)

        ax = axs[g_index]
        m = ax.contourf(
            c_grid,
            K_grid,
            omega,
        )
        ax.format(
            xlabel=r"$c$",
            ylabel=r"$K$",
            xscale="log",
            xformatter="sci",
            yscale="log",
            yformatter="sci",
            abc="a)",
            title=rf"$\Delta z$={grid_size_str[g_index]}",
            xlim=[1e-3, 1e3],
            ylim=[1e-3, 1e3],
        )

    fig.suptitle(r"Optimal relaxation parameter for varying $c, K$")
    fig.colorbar(m, label=r"$\omega_\mathrm{opt}$", loc="b", length=0.7)

    with warnings.catch_warnings(action="ignore"):
        fig.savefig(plotting_dir / "physics_dependence_surface.pdf")
