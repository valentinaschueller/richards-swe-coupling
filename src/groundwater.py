import ufl
from dune.fem.scheme import galerkin
from dune.fem.space import lagrange
from dune.grid import structuredGrid
from dune.ufl import Constant, DirichletBC
from mpi4py import MPI  # necessary to avoid ParallelError from DUNE

depth = 1
dt = 1e-2
N = 5
t_end = N * dt
h = 3
K = 1
c = 1

class Groundwater:
    def __init__(self) -> None:
        gridView = structuredGrid([-depth], [0], [20])

        space = lagrange(gridView, order=1)
        self.psi_h = space.interpolate(0, name="psi_h")
        self.psi_h_n = self.psi_h.copy(name="previous")

        x = ufl.SpatialCoordinate(space)
        psi = ufl.TrialFunction(space)
        v = ufl.TestFunction(space)

        initial = depth / ufl.pi * ufl.sin(ufl.pi / depth * x[0]) + h
        self.psi_h.interpolate(initial)

        a = (
            ufl.dot(c * (psi - self.psi_h_n) / dt, v) + ufl.inner(K * ufl.grad(psi), ufl.grad(v))
        ) * ufl.dx

        fbnd = (-1 * v * ufl.conditional(x[0] <= (-depth + 1e-8), 1, 0)) * ufl.ds
        # dbc_bottom = DirichletBC(space, 0, x[0] <= (-depth + 1e-8))
        dbc_top = DirichletBC(space, h, x[0] >= (-1e-8))

        self.scheme = galerkin([a == fbnd, dbc_top], solver="cg")
        self.scheme.model.dt = dt
        self.scheme.model.time = 0

    def step(self) -> None:
        self.psi_h_n.assign(self.psi_h)
        self.scheme.solve(target=self.psi_h)
        self.scheme.model.time += self.scheme.model.dt

groundwater = Groundwater()

while groundwater.scheme.model.time < (t_end - 1e-6):
    groundwater.step()

groundwater.psi_h.plot()
