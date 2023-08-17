from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import jinja2
from dacite import from_dict
from ruamel.yaml import YAML


class BoundaryCondition(Enum):
    free_drainage = "free drainage"
    no_flux = "no flux"
    dirichlet = "dirichlet"


@dataclass
class Params:
    tolerance: float
    max_iterations: int
    coupling_scheme: str
    omega: float
    t_0: float
    t_end: float
    N: int
    M: int
    K: float
    c: float
    L: float
    h_0: float
    precice_config: str | Path
    precice_config_template: str | Path
    bc_type: str
    dirichlet_value: float = 0.0
    dt: float = field(init=False)
    dz: float = field(init=False)

    def __post_init__(self):
        self.dt = (self.t_end - self.t_0) / self.N
        self.dz = self.L / self.M
        self.precice_config = Path(self.precice_config)
        self.precice_config_template = Path(self.precice_config_template)
        self.bc_type = BoundaryCondition(self.bc_type)
        assert self.precice_config.exists()
        assert self.precice_config_template.exists()


def get_template(template_path: Path) -> jinja2.Template:
    """get Jinja2 template file"""
    loader = jinja2.FileSystemLoader(template_path.parent)
    environment = jinja2.Environment(loader=loader, undefined=jinja2.StrictUndefined)
    return environment.get_template(template_path.name)


def render(params: Params) -> None:
    jinja_template = get_template(params.precice_config_template)
    with open(params.precice_config, "w") as precice_config:
        precice_config.write(
            jinja_template.render(
                coupling_scheme=params.coupling_scheme,
                N=params.N,
                dt=params.dt,
                tolerance=params.tolerance,
                max_iterations=params.max_iterations,
                omega=params.omega,
            )
        )


def load_params(yaml_file: Path | str) -> Params:
    yaml = YAML(typ="safe")
    if isinstance(yaml_file, str):
        yaml_file = Path(yaml_file)
    params = yaml.load(yaml_file)
    params = from_dict(data_class=Params, data=params)
    return params
