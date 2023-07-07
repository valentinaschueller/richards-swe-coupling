import ufl
from dune.fem.scheme import galerkin
from dune.fem.space import lagrange
from dune.grid import structuredGrid
from dune.ufl import DirichletBC
from mpi4py import MPI  # necessary to avoid ParallelError from DUNE

depth = 1
dt = 1e-2
N = 5
t_end = N * dt
h = 3

gridView = structuredGrid([-depth], [0], [20])

space = lagrange(gridView, order=1)
psi_h = space.interpolate(0, name="psi_h")
psi_h_n = psi_h.copy(name="previous")

x = ufl.SpatialCoordinate(space)
psi = ufl.TrialFunction(space)
v = ufl.TestFunction(space)

initial = depth / ufl.pi * ufl.sin(ufl.pi / depth * x[0]) + h
psi_h.interpolate(initial)

a = (ufl.dot((psi - psi_h_n) / dt, v) + ufl.inner(ufl.grad(psi), ufl.grad(v))) * ufl.dx

fbnd = (-1 * v * ufl.conditional(x[0] <= (-depth + 1e-8), 1, 0)) * ufl.ds
# dbc_bottom = DirichletBC(space, 0, x[0] <= (-depth + 1e-8))
dbc_top = DirichletBC(space, h, x[0] >= (-1e-8))

scheme = galerkin([a == fbnd, dbc_top], solver="cg")
scheme.model.dt = dt
scheme.model.time = 0

while scheme.model.time < (t_end - 1e-6):
    psi_h_n.assign(psi_h)
    scheme.solve(target=psi_h)
    scheme.model.time += scheme.model.dt

psi_h.plot()
