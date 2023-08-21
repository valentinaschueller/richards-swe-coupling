from pathlib import Path

import numpy as np
import precice
import xarray as xr

from coupling.setup_simulation import Params


class River:
    def __init__(self, params: Params) -> None:
        self.height = params.h_0
        self.time = params.t_0
        self.result = [params.h_0]
        self.t_axis = [params.t_0]

    def solve(self, dt: float, flux: float) -> float:
        self.height += dt * flux
        self.time += dt

    def end_time_step(self) -> None:
        self.result.append(self.height)
        self.t_axis.append(self.time)

    def save_state(self) -> None:
        self._time_checkpoint = self.time
        self._height_checkpoint = self.height

    def load_state(self) -> None:
        self.time = self._time_checkpoint
        self.height = self._height_checkpoint

    def save_output(self, target: Path) -> None:
        output = xr.DataArray(self.result, {"t": self.t_axis}, name="h")
        output.to_netcdf(target)


def simulate_river(params: Params):
    participant_name = "RiverSolver"
    solver_process_index = 0
    solver_process_size = 1
    interface = precice.Interface(
        participant_name,
        str(params.precice_config),
        solver_process_index,
        solver_process_size,
    )

    mesh_name = "RiverMesh"
    mesh_id = interface.get_mesh_id(mesh_name)
    dimensions = interface.get_dimensions()
    vertex = np.zeros(dimensions)
    vertex_id = interface.set_mesh_vertex(mesh_id, vertex)

    height_id = interface.get_data_id("Height", mesh_id)
    flux_id = interface.get_data_id("Flux", mesh_id)

    river = River(params)

    precice_dt = interface.initialize()

    if interface.is_action_required(precice.action_write_initial_data()):
        interface.write_scalar_data(height_id, vertex_id, params.h_0)
        interface.mark_action_fulfilled(precice.action_write_initial_data())

    interface.initialize_data()

    while interface.is_coupling_ongoing():
        if interface.is_action_required(precice.action_write_iteration_checkpoint()):
            river.save_state()
            interface.mark_action_fulfilled(precice.action_write_iteration_checkpoint())
        dt = min(params.dt, precice_dt)

        flux = interface.read_scalar_data(flux_id, vertex_id)
        river.solve(dt, flux)
        interface.write_scalar_data(height_id, vertex_id, river.height)

        precice_dt = interface.advance(dt)

        if interface.is_action_required(precice.action_read_iteration_checkpoint()):
            river.load_state()
            interface.mark_action_fulfilled(precice.action_read_iteration_checkpoint())
        else:
            river.end_time_step()

    interface.finalize()
    river.save_output("river.nc")
