import numpy as np
import proplot as pplt

from coupling.analysis import compute_coupling_behavior
from coupling.process_results import compute_convergence_rate
from coupling.setup_simulation import load_params
from coupling.simulation import run_coupled_simulation

if __name__ == "__main__":
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

    fig, ax = pplt.subplots()
    ax.plot(omegas, convergence_rates, color="k", marker=".", label=r"experimental CR")
    ax.plot(omegas, abs(sigma_analysis), marker="1", label=r"$|\Sigma|$")
    ax.format(
        xlabel=r"$\omega$",
        ylabel=r"$|\Sigma|$",
        title="Experimental Convergence Rate",
        xlim=[0, 1],
        ylim=[0, 1],
    )
    fig.legend(ncols=1)
    fig.savefig("convergence_rates.png")
