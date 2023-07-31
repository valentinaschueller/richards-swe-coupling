from pathlib import Path

import numpy as np
import precice
import proplot as pplt
import ufl
from dune.fem.function import uflFunction
from dune.fem.scheme import galerkin
from dune.fem.space import lagrange
from dune.grid import structuredGrid
from dune.ufl import Constant, DirichletBC


class Groundwater:
    def __init__(
        self,
        c: float = 1.0,
        K: float = 1.0,
        dt: float = 1e-2,
        depth: float = 1.0,
        h: float = 1.0,
        t_0: float = 0.0,
    ) -> None:
        gridView = structuredGrid([-depth], [0], [20])

        self.space = lagrange(gridView, order=1)
        self.psi_h = self.space.interpolate(0, name="psi_h")
        self.psi_h_n = self.psi_h.copy(name="previous")

        self.c = Constant(c, name="c")
        self.K = Constant(K, name="K")
        self.dt = Constant(dt, name="dt")
        self.height = Constant(h, name="height")

        x = ufl.SpatialCoordinate(self.space)
        psi = ufl.TrialFunction(self.space)
        v = ufl.TestFunction(self.space)

        # get x-values for plotting/visualization
        x_func = uflFunction(gridView, name="x", order=1, ufl=x)
        self.x_axis = self.space.interpolate(x_func).as_numpy

        initial = depth / ufl.pi * ufl.sin(ufl.pi / depth * x[0]) + self.height
        self.psi_h.interpolate(initial)

        a = (
            ufl.dot(self.c * (psi - self.psi_h_n) / self.dt, v)
            + ufl.inner(self.K * ufl.grad(psi), ufl.grad(v))
        ) * ufl.dx

        fbnd = (-1 * v * ufl.conditional(x[0] <= (-depth + 1e-8), -1, 0)) * ufl.ds
        # dbc_bottom = DirichletBC(space, 0, x[0] <= (-depth + 1e-8))
        dbc_top = DirichletBC(self.space, self.height, x[0] >= (-1e-8))

        self.scheme = galerkin([a == fbnd, dbc_top], solver="cg")

        self.scheme.model.dt = dt
        self.scheme.model.time = t_0

        self.t_axis = [t_0]
        self.result = self.psi_h.as_numpy[np.newaxis]  # add a dimension

        self._time_checkpoint = t_0
        self._psi_checkpoint = self.psi_h.copy(name="checkpoint")

        self.flux = self._compute_flux()

    def _compute_flux(self) -> None:
        grad_psi_sfc = self.space.interpolate(ufl.grad(self.psi_h)).as_numpy[-1]
        self.flux = self.K.value * (grad_psi_sfc + 1)

    def solve(self, dt: float) -> None:
        self.psi_h_n.assign(self.psi_h)
        self.scheme.model.dt = dt
        self.dt.value = dt
        self.scheme.solve(target=self.psi_h)
        self.scheme.model.time += self.scheme.model.dt
        self._compute_flux()

    def end_time_step(self) -> None:
        self.result = np.append(self.result, self.psi_h.as_numpy[np.newaxis], axis=0)
        self.t_axis.append(self.scheme.model.time)

    def save_state(self) -> None:
        self._time_checkpoint = self.scheme.model.time
        self._psi_checkpoint.assign(self.psi_h)

    def load_state(self) -> None:
        self.scheme.model.time = self._time_checkpoint
        self.psi_h.assign(self._psi_checkpoint)


participant_name = "GroundwaterSolver"
config_file = Path("precice-config.xml")
solver_process_index = 0
solver_process_size = 1
interface = precice.Interface(
    participant_name, str(config_file), solver_process_index, solver_process_size
)

mesh_name = "GroundwaterMesh"
mesh_id = interface.get_mesh_id(mesh_name)
dimensions = interface.get_dimensions()
vertex = np.zeros(dimensions)
vertex_id = interface.set_mesh_vertex(mesh_id, vertex)

height_id = interface.get_data_id("Height", mesh_id)
flux_id = interface.get_data_id("Flux", mesh_id)

t_0 = 0
t_end = 1
N = 10
solver_dt = (t_end - t_0) / N
groundwater = Groundwater(dt=solver_dt)

precice_dt = interface.initialize()
interface.initialize_data()

while interface.is_coupling_ongoing():
    if interface.is_action_required(precice.action_write_iteration_checkpoint()):
        groundwater.save_state()
        interface.mark_action_fulfilled(precice.action_write_iteration_checkpoint())
    dt = min(solver_dt, precice_dt)

    groundwater.height.value = interface.read_scalar_data(height_id, vertex_id)
    groundwater.solve(dt)
    interface.write_scalar_data(flux_id, vertex_id, groundwater.flux)

    precice_dt = interface.advance(dt)

    if interface.is_action_required(precice.action_read_iteration_checkpoint()):
        groundwater.load_state()
        interface.mark_action_fulfilled(precice.action_read_iteration_checkpoint())
    else:
        groundwater.end_time_step()

interface.finalize()

_, ax = pplt.subplots()
ax.plot(groundwater.x_axis, groundwater.result.T, marker=".")
pplt.show()
