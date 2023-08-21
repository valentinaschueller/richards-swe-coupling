import ufl

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
