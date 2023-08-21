from coupling.river import River
from coupling.setup_simulation import load_params


def test_flux_computation():
    params = load_params("tests/test_params.yaml")
    river = River(params)
    height_before = river.height
    river.solve(dt=1, flux=0)
    assert river.height == height_before
