import ufl
from dune.fem.scheme import galerkin
from dune.fem.space import lagrange
from dune.grid import structuredGrid
from dune.ufl import Constant, DirichletBC
from mpi4py import MPI  # necessary to avoid ParallelError from DUNE


class Groundwater:
    def __init__(
        self,
        c: float = 1.0,
        K: float = 1.0,
        dt: float = 1e-2,
        depth: float = 1.0,
        h: float = 1.0,
    ) -> None:
        gridView = structuredGrid([-depth], [0], [20])

        space = lagrange(gridView, order=1)
        self.psi_h = space.interpolate(0, name="psi_h")
        self.psi_h_n = self.psi_h.copy(name="previous")

        self.c = Constant(c, name="c")
        self.K = Constant(K, name="K")
        self.dt = Constant(dt, name="dt")
        self.h = Constant(h, name="h")

        x = ufl.SpatialCoordinate(space)
        psi = ufl.TrialFunction(space)
        v = ufl.TestFunction(space)

        initial = depth / ufl.pi * ufl.sin(ufl.pi / depth * x[0]) + self.h
        self.psi_h.interpolate(initial)

        a = (
            ufl.dot(self.c * (psi - self.psi_h_n) / self.dt, v)
            + ufl.inner(self.K * ufl.grad(psi), ufl.grad(v))
        ) * ufl.dx

        fbnd = (-1 * v * ufl.conditional(x[0] <= (-depth + 1e-8), 1, 0)) * ufl.ds
        # dbc_bottom = DirichletBC(space, 0, x[0] <= (-depth + 1e-8))
        dbc_top = DirichletBC(space, self.h, x[0] >= (-1e-8))

        self.scheme = galerkin([a == fbnd, dbc_top], solver="cg")
        self.scheme.model.dt = dt
        self.scheme.model.time = 0

    def step(self) -> None:
        self.psi_h_n.assign(self.psi_h)
        self.scheme.solve(target=self.psi_h)
        self.scheme.model.time += self.scheme.model.dt


dt = 1e-2
N = 5
t_end = N * dt
groundwater = Groundwater(dt=dt)

while groundwater.scheme.model.time < (t_end - 1e-6):
    groundwater.step()

groundwater.psi_h.plot()
