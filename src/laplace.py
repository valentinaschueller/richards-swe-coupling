import matplotlib.pyplot as plt
from dune.fem import assemble
from dune.fem.space import lagrange
from dune.grid import structuredGrid
from dune.ufl import DirichletBC
from mpi4py import MPI  # necessary to avoid ParallelError from DUNE
from scipy.sparse.linalg import spsolve as solver
from ufl import (
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    conditional,
    ds,
    dx,
    grad,
    inner,
)

L = 1

gridView = structuredGrid([-L], [0], [20])

space = lagrange(gridView, order=1)
psi_h = space.interpolate(0, name="psi_h")

x = SpatialCoordinate(space)
psi = TrialFunction(space)
v = TestFunction(space)

a = (inner(grad(psi), grad(v))) * dx

fbnd = (-1 * v * conditional(x[0] <= (-L + 1e-8), 1, 0)) * ds
# dbc_bottom = DirichletBC(space, 0, z[0] <= (-L + 1e-8))
dbc_top = DirichletBC(space, 5, x[0] >= (-1e-8))

mat, rhs = assemble([a == fbnd, dbc_top])

A = mat.as_numpy
b = rhs.as_numpy
y = psi_h.as_numpy
y[:] = solver(A, b)

psi_h.plot()

plt.imshow(A.A)
plt.show()
