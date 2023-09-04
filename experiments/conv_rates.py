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
    omegas = np.array(
        [1e-3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.73, 0.75, 0.76, 0.78, 0.8, 0.9, 1.0]
    )
    omegas = np.linspace(0.001, 1, 21)
    convergence_rates = np.zeros(omegas.shape)
    sigma_analysis = np.zeros(omegas.shape)

    params = load_params("experiments/conv_rates.yaml")
    _, S, _ = compute_coupling_behavior(
        params.K, params.c, params.L, params.dt, params.M
    )
    sigma_analysis = omegas * S + (1 - omegas)

    for index, omega in enumerate(omegas):
        params.omega = omega
        run_coupled_simulation(params)
        convergence_rates[index] = compute_convergence_rate(
            "precice-RiverSolver-convergence.log"
        )

    fig, ax = pplt.subplots(width="35em", height="30em")
    ax.plot(
        omegas,
        convergence_rates,
        color="k",
        marker="x",
        ls="none",
        label=r"measured CR",
    )
    ax.plot(
        omegas,
        abs(sigma_analysis),
        marker="o",
        mfc="none",
        ls="none",
        color="k",
        label=r"$|\Sigma|$",
    )
    ax.format(
        xlabel=r"$\omega$",
        ylabel=r"$|\Sigma|$",
        xlim=[0, 1],
        ylim=[0, 1],
    )
    ax.legend(ncols=1)
    fig.savefig(plotting_dir / "convergence_rates.pdf")
