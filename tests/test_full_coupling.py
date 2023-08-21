from coupling.analysis import compute_coupling_behavior
from coupling.process_results import compute_convergence_rate
from coupling.setup_simulation import load_params
from coupling.simulation import run_coupled_simulation

convergence_rate = None
omega_opt = None


def test_full_coupling():
    yaml_file = "tests/test_params.yaml"
    params = load_params(yaml_file)

    run_coupled_simulation(params)

    cvg_logfile = "precice-RiverSolver-convergence.log"
    convergence_rate = compute_convergence_rate(cvg_logfile)

    _, S, omega_opt = compute_coupling_behavior(
        params.K, params.c, params.L, params.dt, params.M
    )
    assert abs(abs(S) - convergence_rate) < 1e-8

    params.omega = omega_opt

    run_coupled_simulation(params)
    optimal_cvg_rate = compute_convergence_rate(cvg_logfile)
    assert optimal_cvg_rate < 1e-8
