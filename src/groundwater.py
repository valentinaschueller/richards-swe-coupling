from pathlib import Path

import numpy as np
import precice
import ufl
import xarray as xr
from dune.fem.function import uflFunction
from dune.fem.scheme import galerkin
from dune.fem.space import lagrange
from dune.grid import structuredGrid
from dune.ufl import Constant, DirichletBC

from setup_simulation import BoundaryCondition, Params


class Groundwater:
    def __init__(self, params: Params) -> None:
        gridView = structuredGrid([-params.L], [0], [params.M])

        self.space = lagrange(gridView, order=1)
        self.psi_h = self.space.interpolate(0, name="psi_h")
        self.dz = params.L / params.M

        self.c = Constant(params.c, name="c")
        self.K = Constant(params.K, name="K")
        self.dt = Constant(params.dt, name="dt")
        self.height = Constant(params.h_0, name="height")

        x = ufl.SpatialCoordinate(self.space)
        psi = ufl.TrialFunction(self.space)
        v = ufl.TestFunction(self.space)

        # get x-values for plotting/visualization
        x_func = uflFunction(gridView, name="x", order=1, ufl=x)
        self.x_axis = self.space.interpolate(x_func).as_numpy

        initial = params.L / ufl.pi * ufl.sin(ufl.pi / params.L * x[0]) + self.height
        self.psi_h.interpolate(initial)
        self.psi_h_n = self.psi_h.copy(name="previous")

        a = (
            ufl.dot(self.c * (psi - self.psi_h_n) / self.dt, v)
            + ufl.inner(self.K * ufl.grad(psi), ufl.grad(v))
        ) * ufl.dx

        # Set boundary conditions
        rhs = 0
        dbc_top = DirichletBC(self.space, self.height, x[0] >= (-1e-8))
        dirichlet = [dbc_top]
        if params.bc_type == BoundaryCondition.dirichlet:
            dbc_bottom_value = Constant(params.dirichlet_value, name="dbc_bottom_value")
            dbc_bottom = DirichletBC(
                self.space, dbc_bottom_value, x[0] <= (-params.L + 1e-8)
            )
            dirichlet.append(dbc_bottom)
        if params.bc_type == BoundaryCondition.no_flux:
            rhs = (-1 * v * ufl.conditional(x[0] <= (-params.L + 1e-8), -1, 0)) * ufl.ds

        self.scheme = galerkin([a == rhs, *dirichlet], solver="cg")

        self.scheme.model.dt = params.dt
        self.scheme.model.time = params.t_0

        self.t_axis = [params.t_0]
        self.result = self.psi_h.as_numpy[np.newaxis].copy()  # add a dimension

        self._time_checkpoint = params.t_0
        self._psi_checkpoint = self.psi_h.copy(name="checkpoint")

        self._compute_flux()

    def _compute_flux(self) -> None:
        M = (self.c.value / 6) * np.array([1, 2])
        A = self.K.value * self.dt.value / self.dz**2 * np.array([-1, 1])
        psi_now = self.psi_h.as_numpy[-2:]
        psi_n = self.psi_h_n.as_numpy[-2:]
        v_s = -(np.dot(M, (psi_now - psi_n)) + np.dot(A, psi_now) + self.K.value)
        self.flux = v_s

    def solve(self, dt: float) -> None:
        self.psi_h_n.assign(self.psi_h)
        self.scheme.model.dt = dt
        self.dt.value = dt
        self.scheme.solve(target=self.psi_h)
        self.scheme.model.time += self.scheme.model.dt
        self._compute_flux()

    def end_time_step(self) -> None:
        self.result = np.append(
            self.result, self.psi_h.as_numpy[np.newaxis].copy(), axis=0
        )
        self.t_axis.append(self.scheme.model.time)

    def save_state(self) -> None:
        self._time_checkpoint = self.scheme.model.time
        self._psi_checkpoint.assign(self.psi_h)

    def load_state(self) -> None:
        self.scheme.model.time = self._time_checkpoint
        self.psi_h.assign(self._psi_checkpoint)

    def save_output(self, target: Path) -> None:
        output = xr.DataArray(
            self.result,
            {
                "t": self.t_axis,
                "z": self.x_axis,
            },
            name="psi",
        )
        output.to_netcdf(target)


def simulate_groundwater(params: Params):
    participant_name = "GroundwaterSolver"
    solver_process_index = 0
    solver_process_size = 1
    interface = precice.Interface(
        participant_name,
        str(params.precice_config),
        solver_process_index,
        solver_process_size,
    )

    mesh_name = "GroundwaterMesh"
    mesh_id = interface.get_mesh_id(mesh_name)
    dimensions = interface.get_dimensions()
    vertex = np.zeros(dimensions)
    vertex_id = interface.set_mesh_vertex(mesh_id, vertex)

    height_id = interface.get_data_id("Height", mesh_id)
    flux_id = interface.get_data_id("Flux", mesh_id)

    groundwater = Groundwater(params)

    precice_dt = interface.initialize()
    interface.initialize_data()

    while interface.is_coupling_ongoing():
        if interface.is_action_required(precice.action_write_iteration_checkpoint()):
            groundwater.save_state()
            interface.mark_action_fulfilled(precice.action_write_iteration_checkpoint())
        dt = min(params.dt, precice_dt)

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
    groundwater.save_output("groundwater.nc")
