from pathlib import Path

import numpy as np
import proplot as pplt

from coupling.analysis import compute_coupling_behavior, get_sigma

plot_dir = Path("plots")
plot_dir.mkdir(exist_ok=True)

K = 1.0
c = 1.0
L = 1
dt = 0.1
M = 10

_, S, omega = compute_coupling_behavior(K, c, L, dt, M)

fig, axs = pplt.subplots(nrows=1, ncols=2, width="50em", height="25em", sharex=False)

ax = axs[0]
omegas = np.linspace(0, 1, 300)

sc_steps = np.array([1, 2, 4, 8])
markers = ["x", ".", "+", "1"]
linestyles = ["-", "--", ":", "-."]

for index, m in enumerate(sc_steps):
    ax.plot(
        omegas,
        abs(get_sigma(omegas, (m + 1) / (2 * m) * S)),
        # marker=markers[index],
        ls=linestyles[index],
        label=f"{m=}",
    )
ax.legend()
ax.format(
    xlabel="$\omega$",
    ylabel="Convergence Rate",
    title="Convergence rate for varying $\omega$",
)


m = np.arange(1, 17)
S_sc = (m + 1) / (2 * m) * S
assert S_sc[0] == S

cvg_rates = abs(get_sigma(omega, S_sc))

ax = axs[1]
ax.scatter(m, cvg_rates, color="k", marker=".")
ax.format(
    xlabel="m",
    ylabel="Convergence Rate",
    title="Convergence rate for $\omega=\omega_\mathrm{opt}$",
    xminorlocator=1,
    xlocator=2,
)

axs.format(abc="a)")
fig.savefig(plot_dir / "subcycling_rates.pdf")
