from multiprocessing import Process

from groundwater import simulate_groundwater
from process_results import compute_convergence_rate, plot_groundwater, plot_river
from river import simulate_river
from setup_simulation import precice_config, precice_config_template, render


def run_coupled_simulation():
    render(precice_config, precice_config_template)
    groundwater_proc = Process(target=simulate_groundwater)
    river_proc = Process(target=simulate_river)
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
