from pathlib import Path

import numpy as np
import proplot as pplt

from coupling.analysis import compute_coupling_behavior

plot_dir = Path("plots")
plot_dir.mkdir(exist_ok=True)

K = 1.0
c = 1.0
L = 1

fig, ax = pplt.subplots()
fig.suptitle(r"Optimal relaxation parameter $\omega_\mathrm{opt}$")

dts = 10.0 ** np.arange(-7, 1, 1)
Ms = 10.0 ** np.arange(7, -1, -1)
dzs = L / Ms

omega_opt = np.array(
    [[compute_coupling_behavior(K, c, L, dt, M)[2] for dt in dts] for M in Ms]
)
for i in range(len(dts)):
    ax.semilogx(
        dzs, omega_opt[i, :], label=rf"$\Delta t=$ {dts[i]}", cycle="tab20b", marker="."
    )
ax.format(
    title=f"K = {K}, c = {c}, L = {L}",
    xlabel=r"Grid size $\Delta z$",
    ylabel=r"$\omega_\mathrm{opt}$",
    xformatter="sci",
)
fig.legend(ncols=1)
fig.savefig(plot_dir / "omega_opt_dz_dt.pdf")

fig, ax = pplt.subplots()
fig.suptitle(r"Optimal relaxation parameter $\omega_\mathrm{opt}$")

for i in range(len(dzs)):
    ax.semilogx(
        dts, omega_opt[:, i], label=rf"$\Delta z=$ {dzs[i]}", cycle="tab20b", marker="."
    )
ax.format(
    title=f"K = {K}, c = {c}, L = {L}",
    xlabel=r"Time step $\Delta t$",
    ylabel=r"$\omega_\mathrm{opt}$",
    xformatter="sci",
)
fig.legend(ncols=1)
fig.savefig(plot_dir / "omega_opt_dt_dz.pdf")
