from pathlib import Path

import jinja2

from enums import BoundaryConditions

coupling_scheme = "serial-implicit"
t_0 = 0
t_end = 1
N = 10
dt = (t_end - t_0) / N
tolerance = 1e-3
max_iterations = 15
omega = 1

bc_type = BoundaryConditions.no_flux
bc_value = 1.0  # only used for Dirichlet BC

precice_config_template = Path("precice-config.xml.j2")
precice_config = Path("precice-config.xml")


def get_template(template_path: Path) -> jinja2.Template:
    """get Jinja2 template file"""
    loader = jinja2.FileSystemLoader(template_path.parent)
    environment = jinja2.Environment(loader=loader, undefined=jinja2.StrictUndefined)
    return environment.get_template(template_path.name)


def render(destination: Path, template: Path) -> None:
    jinja_template = get_template(template)
    with open(destination, "w") as precice_config:
        precice_config.write(
            jinja_template.render(
                coupling_scheme=coupling_scheme,
                N=N,
                dt=dt,
                tolerance=tolerance,
                max_iterations=max_iterations,
                omega=omega,
            )
        )


if __name__ == "__main__":
    assert precice_config_template.exists()
    render(precice_config, precice_config_template)
    assert precice_config.exists()
