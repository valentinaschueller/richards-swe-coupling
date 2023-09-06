from pathlib import Path

import numpy as np
import xarray as xr

from coupling.analysis import compute_coupling_behavior
from coupling.process_results import compute_convergence_rate
from coupling.setup_simulation import load_params
from coupling.simulation import run_coupled_simulation

if __name__ == "__main__":
    plotting_dir = Path("plots")
    plotting_dir.mkdir(exist_ok=True)

    params = load_params("experiments/convergence_no_relaxation.yaml")
    assert params.M == 20
    assert params.dz == 1 / 20
    assert params.L == 1

    t_values = np.power(10.0, np.arange(-8, 1))
    S_analysis = np.zeros(t_values.shape)
    convergence_rates = np.zeros(t_values.shape)

    for index, t in enumerate(t_values):
        assert params.N == 1
        params.dt = t
        params.t_end = t
        S_analysis[index] = compute_coupling_behavior(
            params.K, params.c, params.L, params.dt, params.M
        )[1]

        run_coupled_simulation(params)
        convergence_rates[index] = compute_convergence_rate(
            "precice-RiverSolver-convergence.log"
        )

    ds = xr.Dataset(
        {
            r"|S|, $\Delta z=1/20$": abs(S_analysis),
            r"CR, $\Delta z=1/20$": convergence_rates,
        },
        coords={r"$\Delta t$": t_values},
    )

    S_analysis = np.zeros(t_values.shape)
    convergence_rates = np.zeros(t_values.shape)
    params.M = 500
    params.dz = params.L / 500

    for index, t in enumerate(t_values):
        assert params.N == 1
        params.dt = t
        params.t_end = t
        S_analysis[index] = compute_coupling_behavior(
            params.K, params.c, params.L, params.dt, params.M
        )[1]

        run_coupled_simulation(params)
        convergence_rates[index] = compute_convergence_rate(
            "precice-RiverSolver-convergence.log"
        )

    ds[r"|S|, $\Delta z=1/500$"] = abs(S_analysis)
    ds[r"CR, $\Delta z=1/500$"] = convergence_rates

    ds.to_netcdf(plotting_dir / "cvg_no_relaxation.nc")
