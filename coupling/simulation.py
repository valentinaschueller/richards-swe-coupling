from multiprocessing import Process

from coupling.groundwater import simulate_groundwater
from coupling.process_results import (
    compute_convergence_rate,
    plot_groundwater,
    plot_river,
)
from coupling.river import simulate_river
from coupling.setup_simulation import load_params, render


def run_coupled_simulation():
    params = load_params("params.yaml")
    render(params)
    groundwater_proc = Process(target=simulate_groundwater, args=[params])
    river_proc = Process(target=simulate_river, args=[params])
    groundwater_proc.start()
    river_proc.start()
    groundwater_proc.join()
    river_proc.join()


if __name__ == "__main__":
    run_coupled_simulation()
    plot_groundwater("groundwater.nc", "groundwater.png")
    plot_river("river.nc", "river.png")
    cvg_rate = compute_convergence_rate("precice-RiverSolver-convergence.log")
    # print in boldface
    print("\033[1m" + f"Measured Convergence Rate: {cvg_rate}" + "\033[0m")
