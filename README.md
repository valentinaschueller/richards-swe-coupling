# 1D Richards-SWE Coupling

Simulate 1D groundwater flow coupled to an ODE for shallow water height, using DUNE and preCICE.

## Dependencies

- [dune-fem](https://pypi.org/project/dune-fem/), which requires, among other things
  - MPI (e.g., OpenMPI)
  - [PETSc](https://petsc.org/release/)
- [preCICE](https://github.com/precice/precice) + [pyprecice](https://pypi.org/project/pyprecice/)
- [ProPlot](https://proplot.readthedocs.io/en/latest/index.html)

## Development Guidelines

### Code formatting

We use [black](https://black.readthedocs.io/en/stable/), [isort](https://pycqa.github.io/isort/), and [flake8](https://flake8.pycqa.org/en/latest/) for code formatting, in this order:
1. Run `isort` to sort imports.
2. Run `black` to format code.
3. Run `flake8`. If `flake8 --ignore E501 src/*.py` returns an error that you don't plan to fix, there has to be a good reason -- please add a comment explaining it!
