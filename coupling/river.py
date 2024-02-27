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
    participant = precice.Participant(
        participant_name,
        str(params.precice_config),
        solver_process_index,
        solver_process_size,
    )

    mesh_name = "RiverMesh"
    dimensions = participant.get_mesh_dimensions(mesh_name)
    vertex = np.zeros(dimensions)

    vertex_ids = [participant.set_mesh_vertex(mesh_name, vertex)]

    read_data_name = "Flux"
    write_data_name = "Height"

    river = River(params)

    if participant.requires_initial_data():
        participant.write_data(mesh_name, write_data_name, vertex_ids, [params.h_0])

    participant.initialize()

    while participant.is_coupling_ongoing():
        if participant.requires_writing_checkpoint():
            river.save_state()
        precice_dt = participant.get_max_time_step_size()
        dt = min(params.dt, precice_dt)

        flux = participant.read_data(mesh_name, read_data_name)
        river.solve(dt, flux)
        participant.write_data(mesh_name, write_data_name, vertex_ids, [river.height])

        participant.advance(dt)

        if participant.requires_reading_checkpoint():
            river.load_state()
        else:
            river.end_time_step()

    participant.finalize()
    river.save_output("river.nc")
