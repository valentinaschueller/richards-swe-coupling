from pathlib import Path

import numpy as np
import precice
import ufl
import xarray as xr
from dune.fem.function import gridFunction
from dune.fem.operator import galerkin as galerkin_operator
from dune.fem.scheme import galerkin as galerkin_scheme
from dune.fem.space import lagrange
from dune.fem.utility import pointSample
from dune.grid import structuredGrid
from dune.ufl import Constant, DirichletBC

from coupling.setup_simulation import BoundaryCondition, InitialCondition, Params


class Groundwater:
    def __init__(self, params: Params) -> None:
        gridView = structuredGrid([-params.L], [0], [params.M])

        self.space = lagrange(gridView, order=1)
        self.psi_h = self.space.interpolate(0, name="psi_h")
        self.dz = params.L / params.M

        c = Constant(params.c, name="c")
        self.K = Constant(params.K, name="K")
        self.dt = Constant(params.dt, name="dt")
        self.height = Constant(params.h_0, name="height")

        x = ufl.SpatialCoordinate(self.space)
        cutoff_point = 0.0
        if cutoff_point == 0.0:
            self.c = c
        else:
            self.c = ufl.conditional(x[0] > cutoff_point, x[0] * (c / cutoff_point), c)
        psi = ufl.TrialFunction(self.space)
        phi = ufl.TestFunction(self.space)

        # get x-values for plotting/visualization
        x_func = gridFunction(x, gridView, name="x", order=1)
        self.x_axis = self.space.interpolate(x_func).as_numpy

        if params.ic_type == InitialCondition.linear:
            initial = (self.height - params.dirichlet_value) / params.L * x[
                0
            ] + self.height
        elif params.ic_type == InitialCondition.sin:
            initial = (
                params.L / ufl.pi * ufl.sin(ufl.pi / params.L * x[0]) + self.height
            )
        self.psi_h.interpolate(initial)
        self.psi_h_n = self.psi_h.copy(name="previous")

        mass_term = ufl.dot(self.c * (psi - self.psi_h_n), phi)
        stiffness_term = self.dt * self.K * ufl.inner(ufl.grad(psi), ufl.grad(phi))
        bilinear_form = (mass_term + stiffness_term) * ufl.dx

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
            rhs = (
                -1 * phi * ufl.conditional(x[0] <= (-params.L + 1e-8), -1, 0)
            ) * ufl.ds

        self.scheme = galerkin_scheme(
            [bilinear_form == self.dt * rhs, *dirichlet], solver="cg"
        )

        # Weak form of the flux
        weak_form_flux = (
            -mass_term / self.dt - stiffness_term / self.dt - self.K * ufl.grad(phi)[0]
        ) * ufl.dx
        self.flux_operator = galerkin_operator(weak_form_flux)
        self.vertical_flux = self.space.interpolate(initial, name="vertical_flux")

        self.scheme.model.dt = params.dt
        self.scheme.model.time = params.t_0

        self.t_axis = [params.t_0]
        self.result = self.psi_h.as_numpy[np.newaxis].copy()  # add a dimension

        self._time_checkpoint = params.t_0
        self._psi_checkpoint = self.psi_h.copy(name="checkpoint")

        self._compute_flux()

    def _compute_flux(self) -> None:
        self.flux_operator(self.psi_h, self.vertical_flux)
        self.interface_flux = pointSample(self.vertical_flux, [0.0])

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
    participant = precice.Participant(
        participant_name,
        str(params.precice_config),
        solver_process_index,
        solver_process_size,
    )

    mesh_name = "GroundwaterMesh"
    dimensions = participant.get_mesh_dimensions(mesh_name)
    vertex = np.zeros(dimensions)
    vertex_ids = [participant.set_mesh_vertex(mesh_name, vertex)]

    write_data_name = "Flux"
    read_data_name = "Height"

    groundwater = Groundwater(params)

    participant.initialize()

    while participant.is_coupling_ongoing():
        if participant.requires_writing_checkpoint():
            groundwater.save_state()

        precice_dt = participant.get_max_time_step_size()
        dt = min(params.dt, precice_dt)
        read_data = participant.read_data(mesh_name, read_data_name, vertex_ids, dt)
        groundwater.height.value = read_data[0]

        groundwater.solve(dt)

        write_data = [groundwater.interface_flux]
        participant.write_data(mesh_name, write_data_name, vertex_ids, write_data)

        precice_dt = participant.advance(dt)

        if participant.requires_reading_checkpoint():
            groundwater.load_state()
        else:
            groundwater.end_time_step()

    participant.finalize()
    groundwater.save_output("groundwater.nc")
