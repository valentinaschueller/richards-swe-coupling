from multiprocessing import Process

from groundwater import simulate_groundwater
from process_results import compute_convergence_rate, plot_groundwater, plot_river
from river import simulate_river
from setup_simulation import load_params, render


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
    plot_groundwater("src/groundwater.nc", "src/groundwater.png")
    plot_river("src/river.nc", "src/river.png")
    cvg_rate = compute_convergence_rate("src/precice-RiverSolver-convergence.log")
    # print in boldface
    print("\033[1m" + f"Measured Convergence Rate: {cvg_rate}" + "\033[0m")
