from enum import Enum


class BoundaryConditions(Enum):
    free_drainage = "free drainage"
    no_flux = "no flux"
    dirichlet = "dirichlet"
