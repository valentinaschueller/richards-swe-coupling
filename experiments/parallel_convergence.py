import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import proplot as pplt

from coupling.analysis import compute_coupling_behavior
from coupling.setup_simulation import load_params
from coupling.simulation import run_coupled_simulation

# from coupling.process_results import compute_convergence_rate


def compute_convergence_rate(convergence_log: Path, omega: float, fig_save_path: Path):
    df = pd.read_csv(convergence_log, delim_whitespace=True)
    time_window = df["TimeWindow"].drop_duplicates().to_numpy()[0]

    resabs_height = df.loc[df["TimeWindow"] == time_window]["ResAbs(Height)"].to_numpy()
    resabs_flux = df.loc[df["TimeWindow"] == time_window]["ResAbs(Flux)"].to_numpy()
    if len(resabs_height) < 3:
        raise ValueError("Too few iterates to compute convergence rate")

    resabs_total = np.sqrt(resabs_height**2 + resabs_flux**2)

    fig, ax = pplt.subplots(width="35em", height="30em")
    ax.semilogy(np.arange(1, len(resabs_total) + 1), resabs_total, color="k")
    ax.format(
        xtickminor=False,
        xlabel="Iteration",
        ylabel="$\|res\|_2$",
        title=rf"Residual Evolution for $\omega=${omega}",
        xlim=[0, len(resabs_total) + 1],
    )
    with warnings.catch_warnings(action="ignore"):
        fig.savefig(fig_save_path / f"residual_parallel_{round(omega, 2)}.pdf")

    cvg_rate_height = np.sqrt(np.mean(resabs_height[3::2] / resabs_height[1:-2:2]))
    cvg_rate_flux = np.sqrt(np.mean(resabs_flux[2::2] / resabs_flux[:-2:2]))

    mean_cvg_rate = np.mean(resabs_total[2:] / resabs_total[1:-1])
    std_cvg_rate = np.std(resabs_total[2:] / resabs_total[1:-1])

    return (mean_cvg_rate, std_cvg_rate)


if __name__ == "__main__":
    plotting_dir = Path("plots")
    plotting_dir.mkdir(exist_ok=True)
    residual_plot_dir = plotting_dir / "residuals"
    residual_plot_dir.mkdir(exist_ok=True)
    omegas = np.array(
        [1e-3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.73, 0.75, 0.76, 0.78, 0.8, 0.9, 1.0]
    )
    convergence_rates = np.zeros(omegas.shape)
    sigma_analysis = np.zeros(omegas.shape)

    params = load_params("experiments/conv_rates.yaml")
    params.max_iterations = 30
    params.tolerance = 1e-5
    params.coupling_scheme = "parallel-implicit"
    _, S, _ = compute_coupling_behavior(
        params.K, params.c, params.L, params.dt, params.M
    )
    sigma_analysis = omegas * S + (1 - omegas)
    sigma_parallel_analysis = np.sqrt(1 - 2 * omegas + omegas**2 * (1 - S))

    convergence_rates_par = np.zeros(omegas.shape)
    convergence_rates_par_std = np.zeros(omegas.shape)
    for index, omega in enumerate(omegas):
        expected_rate = sigma_parallel_analysis[index]
        params.omega = omega
        run_coupled_simulation(params)
        (
            convergence_rates_par[index],
            convergence_rates_par_std[index],
        ) = compute_convergence_rate(
            "precice-RiverSolver-convergence.log", omega, residual_plot_dir
        )
    fig, ax = pplt.subplots(width="29em", height="23em")
    ax.scatter(
        omegas,
        convergence_rates_par,
        color="k",
        marker=".",
        label=r"measured CR - par",
    )
    ax.scatter(
        omegas,
        abs(sigma_parallel_analysis),
        marker="d",
        edgecolors="k",
        facecolors="none",
        label=r"$\rho(\Sigma_\mathrm{par})$",
    )
    ax.scatter(
        omegas,
        abs(sigma_analysis),
        marker="o",
        edgecolors="k",
        facecolors="none",
        label=r"$\rho(\Sigma_\mathrm{seq})$",
    )
    ax.format(
        xlabel=r"$\omega$",
        ylabel=r"Convergence Rate",
        xlim=[0, 1],
        ylim=[0, 1],
        title="Convergence Rate - Parallel",
    )
    ax.legend(ncols=1)

    fig_with_std, ax = pplt.subplots(width="35em", height="30em")
    ax.scatter(
        omegas,
        convergence_rates_par,
        color="k",
        marker=".",
        label=r"measured CR - par",
    )
    ax.errorbar(
        omegas,
        convergence_rates_par,
        yerr=convergence_rates_par_std,
        color="k",
        fmt="none",
    )
    ax.scatter(
        omegas,
        abs(sigma_parallel_analysis),
        marker="d",
        edgecolors="k",
        facecolors="none",
        label=r"$|\Sigma_\mathrm{par}|$",
    )
    ax.format(
        xlabel=r"$\omega$",
        ylabel=r"$|\Sigma|$",
        xlim=[0, 1],
    )
    ax.legend(ncols=1)

    with warnings.catch_warnings(action="ignore"):
        fig.savefig(plotting_dir / "parallel_convergence_rates.pdf")
        fig_with_std.savefig(plotting_dir / "parallel_convergence_rates_errorbars.pdf")
