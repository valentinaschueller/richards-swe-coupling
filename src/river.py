from pathlib import Path

import numpy as np
import precice
import proplot as pplt

import setup_simulation as settings


class River:
    def __init__(self, h_0: float, t_0: float) -> None:
        self.height = h_0
        self.time = t_0
        self.result = [h_0]
        self.t_axis = [t_0]

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


participant_name = "RiverSolver"
config_file = Path("precice-config.xml")
solver_process_index = 0
solver_process_size = 1
interface = precice.Interface(
    participant_name, str(config_file), solver_process_index, solver_process_size
)

mesh_name = "RiverMesh"
mesh_id = interface.get_mesh_id(mesh_name)
dimensions = interface.get_dimensions()
vertex = np.zeros(dimensions)
vertex_id = interface.set_mesh_vertex(mesh_id, vertex)

height_id = interface.get_data_id("Height", mesh_id)
flux_id = interface.get_data_id("Flux", mesh_id)

h_0 = 1

river = River(h_0, settings.t_0)

precice_dt = interface.initialize()

if interface.is_action_required(precice.action_write_initial_data()):
    interface.write_scalar_data(height_id, vertex_id, h_0)
    interface.mark_action_fulfilled(precice.action_write_initial_data())

interface.initialize_data()

while interface.is_coupling_ongoing():
    if interface.is_action_required(precice.action_write_iteration_checkpoint()):
        river.save_state()
        interface.mark_action_fulfilled(precice.action_write_iteration_checkpoint())
    dt = min(settings.dt, precice_dt)

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

_, ax = pplt.subplots()
ax.plot(river.t_axis, river.result, marker=".")
pplt.show()
