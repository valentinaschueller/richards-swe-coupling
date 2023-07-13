from pathlib import Path

import numpy as np
import precice
import proplot as pplt


class River:
    def __init__(self, h_0: float, t_0: float) -> None:
        self.h = h_0
        self.t = t_0
        self.result = [h_0]
        self.time = [t_0]

    def step(self, dt: float, flux: float) -> float:
        self.h += dt * flux
        self.t += dt

    def end_time_step(self) -> None:
        self.result.append(self.h)
        self.time.append(self.t)


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

data_id_h = interface.get_data_id("Height", mesh_id)
data_id_v = interface.get_data_id("Flux", mesh_id)

t = 0
t_end = 1
N = 10
solver_dt = (t_end - t) / N
h_0 = 1

river = River(h_0, t)

precice_dt = interface.initialize()

if interface.is_action_required(precice.action_write_initial_data()):
    interface.write_scalar_data(data_id_v, vertex_id, 0)
    interface.write_scalar_data(data_id_h, vertex_id, h_0)
    interface.mark_action_fulfilled(precice.action_write_initial_data())

interface.initialize_data()

while interface.is_coupling_ongoing():
    # solver_dt = river.begin_time_step()
    dt = min(solver_dt, precice_dt)
    read_data = interface.read_scalar_data(data_id_v, vertex_id)
    flux = read_data
    river.step(dt, flux)
    write_data = river.h
    interface.write_scalar_data(data_id_h, vertex_id, write_data)
    precice_dt = interface.advance(dt)
    river.end_time_step()
interface.finalize()

_, ax = pplt.subplots()
ax.plot(river.time, river.result)
pplt.show()
