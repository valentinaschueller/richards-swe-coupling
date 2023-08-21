from dataclasses import asdict

from ruamel.yaml import YAML

from coupling.analysis import compute_coupling_behavior
from coupling.process_results import compute_convergence_rate
from coupling.setup_simulation import load_params
from coupling.simulation import run_coupled_simulation

convergence_rate = None
omega_opt = None


def test_full_coupling(tmp_path):
    yaml_file = "tests/test_params.yaml"
    cvg_logfile = "precice-RiverSolver-convergence.log"
    run_coupled_simulation(yaml_file)
    convergence_rate = compute_convergence_rate(cvg_logfile)

    params = load_params(yaml_file)
    _, S, omega_opt = compute_coupling_behavior(
        params.K, params.c, params.L, params.dt, params.M
    )
    assert abs(abs(S) - convergence_rate) < 1e-8

    params.omega = omega_opt
    params.bc_type = params.bc_type.value
    params.precice_config = str(params.precice_config)
    params.precice_config_template = str(params.precice_config_template)
    yaml = YAML(typ="safe")
    yaml.default_flow_style = False
    yaml.dump(asdict(params), tmp_path / "optimized.yaml")

    run_coupled_simulation(tmp_path / "optimized.yaml")
    optimal_cvg_rate = compute_convergence_rate(cvg_logfile)
    assert optimal_cvg_rate < 1e-8
