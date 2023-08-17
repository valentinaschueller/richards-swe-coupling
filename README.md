# 1D Richards-SWE Coupling

Simulate 1D groundwater flow coupled to an ODE for shallow water height, using DUNE and preCICE.

## Dependencies

- [dune-fem](https://pypi.org/project/dune-fem/), which requires, among other things
  - MPI (e.g., [OpenMPI](https://www.open-mpi.org/)) and [mpi4py](https://mpi4py.readthedocs.io/en/stable/)
  - [PETSc](https://petsc.org/release/)
- [preCICE](https://github.com/precice/precice) + [pyprecice](https://pypi.org/project/pyprecice/)
- [ruamel.yaml](https://pypi.org/project/ruamel.yaml/), [Jinja2](https://pypi.org/project/Jinja2/), and [dacite](https://github.com/konradhalas/dacite) for parsing simulation parameters
- [xarray](https://xarray.dev/), [pandas](https://pandas.pydata.org/) and [ProPlot](https://proplot.readthedocs.io/en/latest/index.html) for simulation output/plotting

## Usage

You can adjust the simulation parameters in `params.yaml`.

### Run the coupled code

To run a simulation, open a terminal and call:

```bash
> python3 src/coupled.py
```

### Return results from theoretical analysis

```bash
> python3 src/analysis.py
```

## Development Guidelines

### Code formatting

We use [black](https://black.readthedocs.io/en/stable/), [isort](https://pycqa.github.io/isort/), and [flake8](https://flake8.pycqa.org/en/latest/) for code formatting and linting, in this order:
1. Run `isort src/*.py` to sort imports.
2. Run `black src/*.py` to format code.
3. Run `flake8 src/*.py`. This should execute without errors or warnings.
