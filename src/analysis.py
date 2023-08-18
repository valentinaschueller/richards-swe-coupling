import numpy as np

from src.setup_simulation import load_params


def get_a(K: float, c: float, dt: float, dz: float) -> float:
    return 2 / 3 * c * dz + 2 * K * dt / dz


def get_b(K: float, c: float, dt: float, dz: float) -> float:
    return c / 6 * dz - K * dt / dz


def get_alpha(a: float, b: float, L: float, M: int, dz: float) -> float:
    j = np.arange(1, M)
    sum = np.sum(
        np.sin(j * np.pi * dz / L) ** 2 / (0.5 * a - b * np.cos(j * np.pi * dz / L))
    )
    alpha = dz / L * sum
    return alpha


def get_S(a: float, b: float, alpha: float, dz: float) -> float:
    S = b**2 * alpha - 0.5 * a
    return S


def get_omega_opt(S: float) -> float:
    return 1 / (1 - S)


def compute_coupling_behavior(
    K: float, c: float, L: float, dt: float, M: int
) -> tuple[float, float, float]:
    """
    compute theoretical convergence behaviour of simulation based on problem parameters.

    :param K: empirical unsaturated hydraulic conductivity
    :type K: float
    :param c: specific moisture capacity
    :type c: float
    :param L: vertical extent of groundwater domain
    :type L: float
    :param dt: time step size
    :type dt: float
    :param M: number of intervals in vertical discretization
    :type M: int
    :return: alpha, iteration matrix S, optimal relaxation parameter omega
    :rtype: tuple[float, float, float]
    """
    dz = L / M
    a = get_a(K, c, dt, dz)
    b = get_b(K, c, dt, dz)
    alpha = get_alpha(a, b, L, M, dz)
    S = get_S(a, b, alpha, dz)
    omega_opt = get_omega_opt(S)
    return alpha, S, omega_opt


if __name__ == "__main__":
    params = load_params("params.yaml")
    alpha, S, omega_opt = compute_coupling_behavior(
        params.K, params.c, params.L, params.dt, params.M
    )
    print(f"{alpha=}")
    print(f"{S=}")
    print(f"{omega_opt=}")
