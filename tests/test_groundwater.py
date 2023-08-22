import ufl
from dune.fem.operator import linear

from coupling.analysis import get_a, get_b
from coupling.groundwater import Groundwater
from coupling.setup_simulation import load_params


def _grad_based_flux(groundwater: Groundwater):
    grad_psi_sfc = groundwater.space.interpolate(ufl.grad(groundwater.psi_h)).as_numpy[
        -1
    ]
    dune_flux = -groundwater.K.value * (grad_psi_sfc + 1)
    return dune_flux


def test_flux_computation():
    params = load_params("tests/test_params.yaml")
    groundwater = Groundwater(params)
    gw_flux = groundwater.flux
    grad_based_flux = _grad_based_flux(groundwater)
    assert gw_flux * grad_based_flux > 0
    assert abs(gw_flux - grad_based_flux) < 0.1


def test_matrix():
    params = load_params("tests/test_params.yaml")
    groundwater = Groundwater(params)
    system_matrix = linear(groundwater.scheme).as_numpy
    a = get_a(params.K, params.c, params.dt, params.dz)
    b = get_b(params.K, params.c, params.dt, params.dz)
    assert abs(system_matrix[1, 1] - a) < 1e-8
    assert abs(system_matrix[1, 2] - b) < 1e-8
    assert abs(system_matrix[2, 1] - b) < 1e-8
    assert abs(system_matrix[-2, -2] - a) < 1e-8
    assert abs(system_matrix[-3, -2] - b) < 1e-8
    assert abs(system_matrix[-2, -3] - b) < 1e-8
    assert abs(system_matrix[-1, -1] - 1) < 1e-8
