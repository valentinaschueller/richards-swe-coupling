from pathlib import Path

import jinja2

coupling_scheme = "serial-implicit"
t_0 = 0
t_end = 1
N = 10
dt = (t_end - t_0) / N
tolerance = 1e-3
max_iterations = 15
omega = 1


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
    src_directory = Path("src")
    precice_config_template = src_directory / "precice-config.xml.j2"
    precice_config = src_directory / "precice-config.xml"
    assert precice_config_template.exists()

    render(precice_config, precice_config_template)
