from coupling.analysis import compute_coupling_behavior
from coupling.process_results import compute_convergence_rate
from coupling.setup_simulation import BoundaryCondition, InitialCondition, load_params
from coupling.simulation import run_coupled_simulation

convergence_rate = None
omega_opt = None


def test_full_coupling():
    yaml_file = "tests/test_params.yaml"
    params = load_params(yaml_file)
    params.omega = 1

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


def test_bc_sensitivity():
    yaml_file = "tests/test_params.yaml"
    params = load_params(yaml_file)
    cvg_logfile = "precice-RiverSolver-convergence.log"

    S = compute_coupling_behavior(params.K, params.c, params.L, params.dt, params.M)[1]
    analytical_cvg_rate = abs(S)

    params.bc_type = BoundaryCondition.dirichlet
    params.dirichlet_value = 0.0
    run_coupled_simulation(params)
    cvg_rate_hom_dbc = compute_convergence_rate(cvg_logfile)

    params.dirichlet_value = 10.0
    run_coupled_simulation(params)
    cvg_rate_nhom_dbc = compute_convergence_rate(cvg_logfile)

    params.bc_type = BoundaryCondition.no_flux
    run_coupled_simulation(params)
    cvg_rate_no_flux = compute_convergence_rate(cvg_logfile)

    params.bc_type = BoundaryCondition.free_drainage
    run_coupled_simulation(params)
    cvg_rate_free_drainage = compute_convergence_rate(cvg_logfile)

    assert abs(cvg_rate_hom_dbc - analytical_cvg_rate) < 1e-8
    assert abs(cvg_rate_nhom_dbc - analytical_cvg_rate) < 1e-8
    assert abs(cvg_rate_free_drainage - analytical_cvg_rate) < 1e-1
    assert abs(cvg_rate_no_flux - analytical_cvg_rate) < 1e-1


def test_bc_sensitivity_grid_size():
    yaml_file = "tests/test_params.yaml"
    params = load_params(yaml_file)
    cvg_logfile = "precice-RiverSolver-convergence.log"

    S = compute_coupling_behavior(params.K, params.c, params.L, params.dt, params.M)[1]
    analytical_cvg_rate = abs(S)

    params.bc_type = BoundaryCondition.no_flux
    run_coupled_simulation(params)
    cvg_rate_no_flux = compute_convergence_rate(cvg_logfile)

    params.bc_type = BoundaryCondition.free_drainage
    run_coupled_simulation(params)
    cvg_rate_free_drainage = compute_convergence_rate(cvg_logfile)

    error_coarse_fd = abs(cvg_rate_free_drainage - analytical_cvg_rate)
    error_coarse_nf = abs(cvg_rate_no_flux - analytical_cvg_rate)

    params.M = 5 * params.M
    params.dz = 0.2 * params.dz

    S = compute_coupling_behavior(params.K, params.c, params.L, params.dt, params.M)[1]
    analytical_cvg_rate = abs(S)

    params.bc_type = BoundaryCondition.no_flux
    run_coupled_simulation(params)
    cvg_rate_no_flux = compute_convergence_rate(cvg_logfile)

    params.bc_type = BoundaryCondition.free_drainage
    run_coupled_simulation(params)
    cvg_rate_free_drainage = compute_convergence_rate(cvg_logfile)

    error_fine_fd = abs(cvg_rate_free_drainage - analytical_cvg_rate)
    error_fine_nf = abs(cvg_rate_no_flux - analytical_cvg_rate)

    assert (error_fine_fd - error_coarse_fd) < 1e-4
    assert (error_fine_nf - error_coarse_nf) < 1e-4


def test_ic_sensitivity():
    yaml_file = "tests/test_params.yaml"
    params = load_params(yaml_file)
    cvg_logfile = "precice-RiverSolver-convergence.log"

    S = compute_coupling_behavior(params.K, params.c, params.L, params.dt, params.M)[1]
    analytical_cvg_rate = abs(S)

    params.ic_type = InitialCondition.linear
    run_coupled_simulation(params)
    cvg_rate_linear = compute_convergence_rate(cvg_logfile)

    params.ic_type = InitialCondition.sin
    run_coupled_simulation(params)
    cvg_rate_sin = compute_convergence_rate(cvg_logfile)

    assert abs(cvg_rate_linear - analytical_cvg_rate) < 1e-8
    assert abs(cvg_rate_sin - analytical_cvg_rate) < 1e-8
